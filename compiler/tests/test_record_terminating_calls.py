"""Original fatal diagnostics compose with owned-record storage, never execution."""
from copy import copy

import pytest

from compiler.frontend.descriptor_record_calls import original_record_call
from compiler.ir import CompilationError
from compiler.tests.test_descriptor_record_calls import fixture, prove
from compiler.tests.test_storage_record_regions import SOURCE

DIAGNOSTICS = """module diagnostics
use iso_fortran_env,only:error_unit
real(8),allocatable::hidden(:)
interface fail
module procedure fail_message,fail_number
end interface
contains
subroutine fail_message(message)
character(*),intent(in)::message
write(error_unit,'(a)')message
stop
end subroutine
subroutine fail_number(number)
integer,intent(in)::number
write(error_unit,'(i0)')number
stop
end subroutine
end module
"""


def source(*, diagnostics=DIAGNOSTICS, calls=1, actual='"bad input"', base=SOURCE):
    body = "\n".join("if(p%index<0)call fail("+actual+")" for _ in range(calls))
    return diagnostics+base.replace("module regions\n", "module regions\nuse diagnostics,only:fail\n").replace(
        "p=>points(i)", "p=>points(i)\n"+body)


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("terminal", ["stop", "error stop 3"])
def test_original_generic_character_diagnostic_keeps_full_record_closure_and_events(tmp_path, sealed, terminal):
    data = fixture(tmp_path, source(diagnostics=DIAGNOSTICS.replace("\nstop\n", "\n"+terminal+"\n")), sealed=sealed)
    window = prove(data).public()
    assert window["schema_version"] == 27
    owned, = window["owned_record_call_requirements"]
    storage = owned["storage_mutation"]
    assert storage["version"] == 12
    assert {row["procedure"] for row in storage["closure"]} == {"regions::touch", "diagnostics::fail_message"}
    fatal, = storage["original_terminating_calls"]
    assert fatal["callee"] == "diagnostics::fail_message"
    assert fatal["procedure"] == "regions::touch"
    assert [row["kind"] for row in fatal["original_observable_events"]] == [
        "original_external_diagnostic_write", "original_termination"]
    assert {row["kind"] for row in storage["observables"]} >= {
        "original_external_diagnostic_write", "original_termination"}
    assert fatal["requires_original_call_execution"]
    for name in ("continuation_authorized", "execution_authorized", "motion_authorized", "payload_access_authorized",
                 "capture_authorized", "entry_definition_authority", "gpu_legality_established"):
        assert fatal[name] is False
    assert not storage["frontiers"]
    assert not storage["mutations"]
    assert not storage["escapes"]


@pytest.mark.parametrize(("before", "after"), [
    ("\nstop\n", "\nreturn\n"),
    ("write(error_unit,'(a)')message", "deallocate(hidden)\nwrite(error_unit,'(a)')message"),
    ("write(error_unit,'(a)')message", "call unknown()\nwrite(error_unit,'(a)')message"),
    ("subroutine fail_message(message)", "subroutine fail_message(message) bind(c)"),
    ("character(*),intent(in)::message", "character(kind=4,len=*),intent(in)::message"),
])
def test_return_hidden_storage_unknown_foreign_and_nondefault_character_are_not_waived(tmp_path, before, after):
    data = fixture(tmp_path, source(diagnostics=DIAGNOSTICS.replace(before, after)), sealed=True)
    with pytest.raises(CompilationError):
        prove(data)
    assert not getattr(data[0], "_descriptor_window_proofs", {})


def test_many_distinct_call_sites_to_one_callee_are_not_a_global_procedure_count(tmp_path):
    data = fixture(tmp_path, source(calls=70), sealed=True)
    data[0].procedure_limit = 8
    owned, = prove(data).public()["owned_record_call_requirements"]
    assert len(owned["storage_mutation"]["original_terminating_calls"]) == 70
    assert {row["procedure"] for row in owned["storage_mutation"]["closure"]} == {
        "regions::touch", "diagnostics::fail_message"}


def test_real_cumulative_unique_terminating_callees_count_the_outer_caller(tmp_path):
    wrappers = "".join("subroutine fail_"+str(i)+"(message)\ncharacter(*),intent(in)::message\n"
                       "write(error_unit,'(a)')message\nstop\nend subroutine\n" for i in range(7))
    diagnostics = DIAGNOSTICS.replace("contains\n", "contains\n"+wrappers, 1)
    body = "\n".join('if(p%index<0)call fail_'+str(i)+'("bad input")' for i in range(7))
    text = source(diagnostics=diagnostics, calls=0).replace(
        "use diagnostics,only:fail", "use diagnostics,only:"+",".join("fail_"+str(i) for i in range(7))).replace(
        "p=>points(i)", "p=>points(i)\n"+body)
    data = fixture(tmp_path, text, sealed=True)
    data[0].procedure_limit = 8
    with pytest.raises(CompilationError, match="composed original call closure budget"):
        prove(data)


def test_real_cumulative_terminating_depth_counts_the_outer_caller(tmp_path):
    text = source(calls=0).replace("a(i)=p%value", "call relay()\na(i)=p%value")
    text = text.replace("end module\n", "end module\n", 1)
    position = text.rindex("end module")
    text = text[:position]+('subroutine relay()\ncall relay2()\nend subroutine\n'
                           'subroutine relay2()\ncall fail("bad input")\nend subroutine\n')+text[position:]
    data = fixture(tmp_path, text, sealed=True)
    data[0].depth_limit = 4
    with pytest.raises(CompilationError, match="composed original call depth budget"):
        prove(data)


@pytest.mark.parametrize("fault", ["copy", "registry", "source", "limits"])
def test_retained_terminating_authority_cannot_be_copied_detached_or_stale(tmp_path, fault):
    data = fixture(tmp_path, source(), sealed=True)
    analysis, caller, _selected, _roots, _modules = data
    call, = caller.execution.content
    proof = original_record_call(analysis, caller, call, (analysis.modules["regions"].bindings["field"],))
    terminal, = proof._storage._terminating_calls
    if fault == "copy":
        analysis._descriptor_terminating_calls[terminal._procedure, id(terminal._call)] = copy(terminal)
    elif fault == "registry":
        analysis._descriptor_terminating_calls.clear()
    elif fault == "source":
        path = caller.scope.path
        path.write_text(path.read_text()+"\n! source changed\n")
    else:
        analysis.procedure_limit += 1
    with pytest.raises(CompilationError):
        proof.validate()


@pytest.mark.parametrize("sealed", [False, True])
def test_late_terminating_registry_copy_is_rejected_after_same_validation(tmp_path, monkeypatch, sealed):
    from compiler.frontend.storage_mutation_effects import StorageMutationEffects

    analysis, caller, _selected, _roots, _modules = fixture(tmp_path, source(), sealed=sealed)
    call, = caller.execution.content
    proof = original_record_call(analysis, caller, call, (analysis.modules["regions"].bindings["field"],))
    token, = proof._storage._terminating_calls
    if sealed:
        assert proof.validate() is proof
        analysis._descriptor_terminating_calls[token._procedure, id(token._call)] = copy(token)
        with pytest.raises(CompilationError):
            proof.validate()
        return
    original = StorageMutationEffects.validate
    reached = []

    def changed(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if self is proof._storage and not reached:
            reached.append(token)
            analysis._descriptor_terminating_calls[token._procedure, id(token._call)] = copy(token)
        return result

    monkeypatch.setattr(StorageMutationEffects, "validate", changed)
    with pytest.raises(CompilationError):
        proof.validate()
    assert reached


@pytest.mark.parametrize("sealed", [False, True])
def test_identical_call_text_keeps_both_exact_positions_and_the_first_handle(tmp_path, monkeypatch, sealed):
    from compiler.frontend.storage_mutation_effects import StorageMutationEffects

    text = source(calls=2)
    marker = 'if(p%index<0)call fail("bad input")'
    second = text.index(marker, text.index(marker)+1)
    text = text[:second]+text[second:].replace("if(p%index<0)", "if(p%index==-1)", 1)
    analysis, caller, _selected, _roots, _modules = fixture(tmp_path, text, sealed=sealed)
    call, = caller.execution.content
    proof = original_record_call(analysis, caller, call, (analysis.modules["regions"].bindings["field"],))
    first, second = proof._storage._terminating_calls
    assert first is not second
    assert first._call is not second._call
    assert first.public()["original_call"] == second.public()["original_call"]
    assert first.public()["original_call_ordinal"] < second.public()["original_call_ordinal"]
    assert len(proof._storage.public()["original_terminating_calls"]) == 2
    if sealed:
        assert proof.validate() is proof
        analysis._descriptor_terminating_calls[first._procedure, id(first._call)] = copy(first)
        assert analysis._descriptor_terminating_calls[second._procedure, id(second._call)] is second
        with pytest.raises(CompilationError):
            proof.validate()
        return
    original = StorageMutationEffects.validate
    reached = []

    def changed(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if self is proof._storage and not reached:
            reached.append(first)
            analysis._descriptor_terminating_calls[first._procedure, id(first._call)] = copy(first)
        return result

    monkeypatch.setattr(StorageMutationEffects, "validate", changed)
    with pytest.raises(CompilationError):
        proof.validate()
    assert reached


def test_fatal_branch_does_not_trim_later_caller_lifetime_effects(tmp_path):
    text = source().replace("a(i)=p%value", "a(i)=p%value\ndeallocate(field)")
    with pytest.raises(CompilationError):
        prove(fixture(tmp_path, text, sealed=True))


@pytest.mark.parametrize("route", ["retained", "retirement"])
def test_genuine_emission_retains_whole_body_fixed_result_termination_and_none_facts(tmp_path, monkeypatch, route):
    import json

    from compiler.tests.test_descriptor_record_calls import (
        decline_complete_record_call,
        emitted_record_retention,
        emitted_record_retirement,
        emitted_source,
    )
    from compiler.tests.test_graph_storage_epochs import builder_at
    from compiler.tests.test_owned_record_whole_body import fixed_result_source

    before = "subroutine touch"+SOURCE.split("subroutine touch", 1)[1].rsplit("end module", 1)[0]
    after = "subroutine touch"+fixed_result_source().split("subroutine touch", 1)[1].rsplit("end module", 1)[0]
    after = after.replace("p=>points(i)", 'p=>points(i)\nif(p%index<0)call fail("bad input")')
    text = DIAGNOSTICS+emitted_source().replace("module epochs\n", "module epochs\nuse diagnostics,only:fail\n")
    assert text.count(before) == 1
    text = text.replace(before, after)
    if route == "retirement":
        decline_complete_record_call(monkeypatch)
    path, builder = builder_at(tmp_path, text)
    outputs, report = builder.run()
    inspect = emitted_record_retention if route == "retained" else emitted_record_retirement
    storage, completion = inspect(path, builder, outputs, report)
    assert {row["procedure"] for row in storage["closure"]} == {
        "epochs::touch", "epochs::update", "epochs::sample", "diagnostics::fail_message"}
    assert storage["fixed_array_function_results"][0]["procedure"] == "epochs::sample"
    fatal, = storage["original_terminating_calls"]
    assert fatal["requires_original_call_execution"]
    assert not fatal["returns_to_caller"]
    assert not fatal["continuation_authorized"]
    event, = completion["terminal_calls" if route == "retained" else "original_observable_terminal_instances"]
    assert event["termination"] == fatal
    assert not event["returns_to_caller"]
    assert event["original_order_and_branch_required"]
    assert [row["kind"] for row in fatal["original_observable_events"]] == [
        "original_external_diagnostic_write", "original_termination"]
    assert not storage["execution_authorized"]
    assert all(row["initialized"] == "none" for row in builder.facts["captures"].values())
    assert path.read_text() == text
    for name, content in outputs.items():
        target = tmp_path/"generated"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    for name, value in [("report.json", report), ("facts.json", builder.facts),
                        (route+"-records.json", {"storage": storage, "completion": completion})]:
        (tmp_path/name).write_text(json.dumps(value, indent=2)+"\n")


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("endpoint", ["roles", "source", "original"])
@pytest.mark.parametrize("target", ["formal", "caller"])
def test_standalone_issuer_retains_original_kinds_before_first_verification(tmp_path, monkeypatch, sealed, endpoint, target):
    from compiler.frontend import descriptor_termination as termination
    from compiler.frontend.source_effects import SourceEffects

    formal_kind = "" if target == "formal" else "(8)"
    caller_declaration = "" if target == "formal" else "integer,intent(in)::number\n"
    caller_argument = "" if target == "formal" else "number"
    actual = "1_8" if target == "formal" else "number"
    text = ("module terminals\nuse iso_fortran_env,only:error_unit\ncontains\n"
            "subroutine fatal(number)\ninteger"+formal_kind+",intent(in)::number\n"
            "write(error_unit,'(i0)')number\nstop\nend subroutine\n"
            "subroutine work("+caller_argument+")\n"+caller_declaration+"call fatal("+actual+")\n"
            "end subroutine\nend module\n")
    path = tmp_path/"original.f90"
    path.write_text(text)
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    caller = analysis.routines["terminals::work"]
    call, = caller.execution.content
    owner = analysis.routines["terminals::fatal"] if target == "formal" else caller
    binding = owner.scope.bindings["number"]
    assert binding.kind == 4
    # The unchanged original signature rejects before any delegated fault.
    with pytest.raises(CompilationError):
        termination.prove_descriptor_termination(analysis, caller.qualified, call)
    fired = []
    host, name = ((analysis, "verify_source") if endpoint == "source" else
                  (termination, "_scope_roles" if endpoint == "roles" else "_original"))
    original = getattr(host, name)

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if not fired:
            fired.append(binding)
            object.__setattr__(binding, "kind", 8)
        return result

    monkeypatch.setattr(host, name, changed)
    with pytest.raises(CompilationError):
        termination.prove_descriptor_termination(analysis, caller.qualified, call)
    assert fired
    assert not analysis._descriptor_terminating_calls


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("fault", ["imported_binding", "parameter_initializer"])
def test_imported_signature_declarations_and_parameter_syntax_are_retained_at_entry(tmp_path, monkeypatch, sealed, fault):
    from compiler.frontend import descriptor_termination as termination
    from compiler.frontend.source_effects import SourceEffects

    formal_kind = "(8)" if fault == "imported_binding" else ""
    text = ("module data\ninteger,parameter::k4=4,k8=8,selected_kind=k4\n"
            "integer(selected_kind),parameter::number=1\nend module\n"
            "module terminals\nuse data,only:number\nuse iso_fortran_env,only:error_unit\ncontains\n"
            "subroutine fatal(number)\ninteger"+formal_kind+",intent(in)::number\n"
            "write(error_unit,'(i0)')number\nstop\nend subroutine\n"
            "subroutine work()\ncall fatal(number)\nend subroutine\nend module\n")
    path = tmp_path/"original.f90"
    path.write_text(text)
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    caller = analysis.routines["terminals::work"]
    call, = caller.execution.content
    imported = analysis.modules["data"]
    binding = imported.bindings["number"]
    assert binding.kind == 4
    if fault == "imported_binding":
        with pytest.raises(CompilationError):
            termination.prove_descriptor_termination(analysis, caller.qualified, call)
        changed_object, field, replacement = binding, "kind", 8
    else:
        baseline = termination.prove_descriptor_termination(analysis, caller.qualified, call)
        assert baseline.public()["associations"][0]["formal_descriptor"]["kind"] == 4
        # The mutation must be rejected by a fresh issuance as well.
        analysis._descriptor_terminating_calls.clear()
        if sealed:
            analysis.operation_limit += 1  # A new context exercises a fresh derivation.
        changed_object = imported.kinds.values["selected_kind"]
        assert type(changed_object).__name__ == "Name"
        assert changed_object.string == "k4"
        field, replacement = "string", "k8"
    original = termination._scope_roles
    fired = []

    def changed(current):
        result = original(current)
        if not fired:
            fired.append(changed_object)
            object.__setattr__(changed_object, field, replacement)
        return result

    monkeypatch.setattr(termination, "_scope_roles", changed)
    with pytest.raises(CompilationError):
        termination.prove_descriptor_termination(analysis, caller.qualified, call)
    assert fired
    assert not analysis._descriptor_terminating_calls


@pytest.mark.parametrize("sealed", [False, True])
def test_imported_kind_parameter_dependency_is_retained_before_roles(tmp_path, monkeypatch, sealed):
    from compiler.frontend import descriptor_termination as termination
    from compiler.frontend.source_effects import SourceEffects

    text = ("module kinds\ninteger,parameter::k4=4,k8=8,selected_kind=k4\nend module\n"
            "module terminals\nuse kinds,only:k4,k8,selected_kind\n"
            "use iso_fortran_env,only:error_unit\ncontains\n"
            "subroutine fatal(number)\ninteger,intent(in)::number\n"
            "write(error_unit,'(i0)')number\nstop\nend subroutine\n"
            "subroutine work()\ncall fatal(1_selected_kind)\nend subroutine\nend module\n")
    path = tmp_path/"original.f90"
    path.write_text(text)
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    caller = analysis.routines["terminals::work"]
    call, = caller.execution.content
    baseline = termination.prove_descriptor_termination(analysis, caller.qualified, call)
    assert baseline.public()["associations"][0]["formal_descriptor"]["kind"] == 4
    analysis._descriptor_terminating_calls.clear()
    if sealed:
        analysis.operation_limit += 1  # A new context exercises a fresh derivation.
    initializer = analysis.modules["kinds"].kinds.values["selected_kind"]
    assert type(initializer).__name__ == "Name"
    assert initializer.string == "k4"
    original = termination._scope_roles
    fired = []

    def changed(current):
        result = original(current)
        if not fired:
            fired.append(initializer)
            object.__setattr__(initializer, "string", "k8")
        return result

    monkeypatch.setattr(termination, "_scope_roles", changed)
    with pytest.raises(CompilationError):
        termination.prove_descriptor_termination(analysis, caller.qualified, call)
    assert fired
    assert not analysis._descriptor_terminating_calls
