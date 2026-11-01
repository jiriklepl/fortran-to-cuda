"""Terminating native calls retain I/O and cannot lend continuation authority."""
from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.call_bindings import resolve_source_call
from compiler.frontend.descriptor_termination import prove_descriptor_termination
from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError


def fixture(tmp_path, *, terminal="stop", body="write(error_unit,'(a)') message", spec="",
            dtype="character(*)", actual='"bad input"', formal="message", caller_extra="",
            import_line="use iso_fortran_env,only:error_unit"):
    path = tmp_path / "terminal.f90"
    path.write_text(f"""module terminals
{import_line}
interface fail
module procedure fail_message,fail_number
end interface
contains
subroutine fail_message({formal})
{dtype},intent(in)::{formal}
{spec}
{body}
{terminal}
end subroutine
subroutine fail_number(number)
integer,intent(in)::number
write(error_unit,'(i0)')number
stop
end subroutine
subroutine work(a)
real(8),intent(inout)::a(:)
{caller_extra}
if(a(1)<0)call fail({actual})
a(1)=a(1)+1
end subroutine
subroutine advance(a)
real(8),allocatable,intent(inout)::a(:)
call work(a)
deallocate(a)
end subroutine
end module
""")
    analysis = SourceEffects([path])
    routine = analysis.routines["terminals::work"]
    call = next(node for node in walk(routine.execution) if type(node).__name__ == "Call_Stmt")
    entry = analysis.routines["terminals::advance"]
    selected = (analysis.structure(entry.qualified).node_id(entry.execution.content[0]),)
    return path, analysis, routine, call, selected


def issue(data):
    _path, analysis, routine, call, _selected = data
    return prove_descriptor_termination(analysis, routine.qualified, call)


@pytest.mark.parametrize("terminal", ["stop", "stop 3", "error stop", "error stop 2"])
def test_exact_original_generic_character_call_preserves_diagnostics_and_termination(tmp_path, terminal):
    data = fixture(tmp_path, terminal=terminal)
    path, analysis, routine, call, _selected = data
    before = path.read_bytes()
    with pytest.raises(CompilationError):
        resolve_source_call(analysis, routine.scope, call)  # Numeric resolver remains unchanged.
    proof = issue(data)
    record = proof.public()
    assert record["callee"] == "terminals::fail_message"
    assert record["original_call"] == str(call)
    assert [item["kind"] for item in record["original_observable_events"]] == [
        "original_external_diagnostic_write", "original_termination"]
    assert record["original_observable_events"][0]["unit"] == "$intrinsic::iso_fortran_env::error_unit"
    for key in ("continuation_authorized", "execution_authorized", "motion_authorized", "payload_access_authorized",
                "capture_authorized", "entry_definition_authority", "gpu_legality_established", "internal_cuts_authorized"):
        assert record[key] is False
    assert record["returns_to_caller"] is False
    assert record["original_order_and_branch_required"] is True
    assert proof.validate(analysis, routine.qualified, call) is proof
    assert path.read_bytes() == before
    record["original_observable_events"].clear()
    assert len(proof.public()["original_observable_events"]) == 2


def test_numeric_generic_case_remains_original_readonly_storage_association(tmp_path):
    data = fixture(tmp_path, actual="value", caller_extra="integer::value")
    record = issue(data).public()
    assert record["callee"] == "terminals::fail_number"
    assert record["associations"][0]["formal_descriptor"]["intent"] == "in"
    assert record["associations"][0]["original_actual"] == "value"


def test_original_keyword_rename_and_intrinsic_unit_alias_are_retained(tmp_path):
    data = fixture(tmp_path, actual='message="bad input"', import_line="use,intrinsic::iso_fortran_env,only:diagnostic=>error_unit",
                   body="write(unit=diagnostic,fmt='(a)')message")
    record = issue(data).public()
    assert record["original_arguments"][0]["keyword"].lower() == "message"
    assert record["original_observable_events"][0]["unit"] == "$intrinsic::iso_fortran_env::error_unit"


def test_descriptor_consumer_requires_token_then_retains_fatal_branch_without_completing_native_summary(tmp_path):
    from compiler.frontend.descriptor_windows import _derive

    data = fixture(tmp_path)
    _path, analysis, _routine, _call, selected = data
    with pytest.raises(CompilationError, match="unknown selected storage"):
        _derive(analysis, "terminals::advance", selected, ("argument::a",))
    before = analysis.summarize("terminals::work")
    proof = prove_descriptor_window(analysis, "terminals::advance", selected, ("argument::a",))
    record = proof.public()
    terminal, = record["native_terminating_descriptor_requirements"]
    assert terminal["requires_original_call_execution"] is True
    assert terminal["returns_to_caller"] is False
    assert record["entry_definition_authority"] is False
    assert record["execution_authorized"] is False
    assert analysis.summarize("terminals::work") == before
    assert before["complete"] is False


@pytest.mark.parametrize("body", ["continue", "if(.true.)stop", "call unknown()", "message='changed'",
    "allocate(scratch(1))", "deallocate(scratch)", "nullify(alias)", "alias=>target", "return"])
def test_no_control_escape_or_lifetime_action_is_erased_by_later_stop(tmp_path, body):
    data = fixture(tmp_path, body=body, spec="real,allocatable::scratch(:)\nreal,pointer::alias\nreal,target::target")
    with pytest.raises(CompilationError):
        issue(data)


def test_original_openmp_callee_is_not_flattened_to_one_diagnostic_event(tmp_path):
    data = fixture(tmp_path, body="!$omp parallel\nwrite(error_unit,'(a)')message\n!$omp end parallel")
    with pytest.raises(CompilationError, match="OpenMP"):
        issue(data)


@pytest.mark.parametrize("tail", ["deallocate(scratch)", "call unknown()", "write(error_unit,'(a)')message"])
def test_terminal_prefix_does_not_trim_unsafe_original_tail(tmp_path, tail):
    data = fixture(tmp_path, body="stop\n" + tail, spec="real,allocatable::scratch(:)")
    with pytest.raises(CompilationError):
        issue(data)


@pytest.mark.parametrize("terminal", ["continue", "return", "if(.true.)stop", "stop number"])
def test_termination_must_be_unconditional_exact_final_literal_stop(tmp_path, terminal):
    with pytest.raises(CompilationError):
        issue(fixture(tmp_path, terminal=terminal))


@pytest.mark.parametrize("body", ["write(message,'(a)')message", "write(6,'(a)')message",
    "write(error_unit,format)message", "write(error_unit,'(a)',iostat=status)message",
    "write(error_unit,'(a)')unknown()", "write(error_unit,'(a)')table(1)"])
def test_io_must_be_original_external_diagnostic_with_exact_scalar_dependencies(tmp_path, body):
    with pytest.raises(CompilationError):
        issue(fixture(tmp_path, body=body))


@pytest.mark.parametrize("spec", ["save", "common /shared/ message", "equivalence(message,other)",
    "namelist /shared/ message", "real,allocatable::scratch(:)", "real,pointer::alias", "integer::local=1"])
def test_complete_original_declarations_cannot_hide_persistent_or_associated_local_state(tmp_path, spec):
    with pytest.raises(CompilationError):
        issue(fixture(tmp_path, spec=spec))


@pytest.mark.parametrize("dtype", ["character(20)", "character(kind=4,len=*)", "character(:),allocatable",
                                   "character(*),target", "character(*),optional"])
def test_character_contract_is_literal_default_assumed_length_readonly_scalar_only(tmp_path, dtype):
    with pytest.raises(CompilationError):
        issue(fixture(tmp_path, dtype=dtype))


@pytest.mark.parametrize("actual", ["text", "unknown()", '"bad"//"input"', "4_'bad'"])
def test_character_expression_or_storage_is_not_a_literal_only_type_proof(tmp_path, actual):
    with pytest.raises(CompilationError):
        issue(fixture(tmp_path, actual=actual, caller_extra="character(20)::text"))


@pytest.mark.parametrize("fault", ["binding", "source", "copy", "call", "limits", "registry"])
def test_retained_token_cannot_survive_source_namespace_policy_or_identity_substitution(tmp_path, fault):
    data = fixture(tmp_path)
    proof = issue(data)
    path, analysis, routine, call, _selection = data
    if fault == "binding":
        child = analysis.routines["terminals::fail_message"]
        child.scope.bindings["message"] = copy(child.scope.bindings["message"])
    elif fault == "source":
        path.write_text(path.read_text() + "\n! changed source\n")
    elif fault == "copy":
        proof = copy(proof)
    elif fault == "call":
        call = copy(call)
    elif fault == "limits":
        analysis.operation_limit += 1
    else:
        analysis._descriptor_terminating_calls.clear()
    with pytest.raises(CompilationError):
        proof.validate(analysis, routine.qualified, call)


def test_late_dependency_mutation_cannot_publish_prior_body_evidence(tmp_path, monkeypatch):
    from compiler.frontend.descriptor_termination import _Reader

    data = fixture(tmp_path)
    original = _Reader.body

    def mutate(self, routine):
        events = original(self, routine)
        routine.scope.bindings["message"].attributes |= {"pointer"}
        return events

    monkeypatch.setattr(_Reader, "body", mutate)
    with pytest.raises(CompilationError):
        issue(data)


def test_ambiguous_original_import_result_cannot_supply_intrinsic_io_unit(tmp_path, monkeypatch):
    data = fixture(tmp_path)
    _path, analysis, _routine, _call, _selection = data
    original = analysis.use_sources

    def ambiguous(scope, name):
        return None if str(name).lower() == "error_unit" else original(scope, name)

    monkeypatch.setattr(analysis, "use_sources", ambiguous)
    with pytest.raises(CompilationError, match="unit"):
        issue(data)


def test_shared_consumer_budget_failure_is_not_an_overload_mismatch(tmp_path):
    from compiler.frontend.call_bindings import OriginalCallBudgetExceeded

    data = fixture(tmp_path)
    proof = issue(data)
    _path, analysis, routine, call, _selection = data

    def exhausted():
        raise CompilationError("shared original-source traversal budget exhausted")

    with pytest.raises(OriginalCallBudgetExceeded, match="shared original-source traversal budget"):
        proof.validate(analysis, routine.qualified, call, tick=exhausted)
