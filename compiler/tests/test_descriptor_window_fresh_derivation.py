"""Fresh diagnostic scans and immutable production fact consumption.

Unsealed diagnostics retain callback fences. Sealed production reuses an issued
source fact; supported context, registry, rollback and live-file publication
changes are covered by test_descriptor_storage_source_facts.
"""
from copy import copy
from inspect import currentframe

import pytest

from compiler.frontend import descriptor_windows as service
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

ENTRY = "fresh::run"
ROOT = "argument::a"


def fixture(tmp_path, *, sealed=False, statements="a(1)=a(1)+1.0_8", procedures=64, extra=""):
    path = tmp_path / "fresh.f90"
    path.write_text("module fresh\ncontains\n" + extra + "subroutine run(a)\n"
                    "real(8),allocatable,intent(inout)::a(:)\ninteger(8)::ticks\n"
                    + statements + "\nend subroutine\nend module\n")
    analysis = SourceEffects([path], procedures=procedures)
    if sealed:
        analysis.seal_session()
    routine = analysis.routines[ENTRY]
    graph, selected = analysis._selected_source(ENTRY, tuple(routine.execution.content))
    return path, analysis, graph, selected


def issue(data, selection=None):
    return service.prove_descriptor_window(data[1], ENTRY, data[3] if selection is None else selection, (ROOT,))


@pytest.mark.parametrize("sealed", [False, True])
def test_new_cached_and_standalone_queries_each_derive_once(tmp_path, monkeypatch, sealed):
    data = fixture(tmp_path, sealed=sealed)
    original, calls = service._derive, []

    def derive(*args, **kwargs):
        calls.append((args[1], args[2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "_derive", derive)
    proof = issue(data)
    assert len(calls) == 1
    public = proof.public()
    calls.clear()
    assert issue(data, tuple(list(data[3]))) is proof
    assert len(calls) == (0 if sealed else 1)
    assert proof.public() == public
    calls.clear()
    assert proof.validate(data[1], ENTRY, data[3]) is proof
    assert len(calls) == (0 if sealed else 1)
    assert proof.public() == public


@pytest.mark.parametrize("sealed", [False, True])
def test_more_than_64_independent_windows_keep_per_query_procedure_bound(tmp_path, sealed):
    count = 70
    data = fixture(tmp_path, sealed=sealed, procedures=1,
                   statements="\n".join(f"a(1)=a(1)+{index}.0_8" for index in range(count)))
    proofs = [issue(data, (identity,)) for identity in data[3]]
    assert len(proofs) == len({proof.identity for proof in proofs}) == count
    assert len(data[1]._descriptor_window_proofs) == count
    assert data[1].procedure_limit == 1
    assert all(proof.public()["resolution_validation"]["procedure_limit"] == 1 for proof in proofs)
    assert issue(data, (data[3][-1],)) is proofs[-1]


def test_real_per_query_source_call_limit_remains_enforced(tmp_path):
    extra = "subroutine work(a)\nreal(8),intent(inout)::a(:)\na(1)=2.0_8\nend subroutine\n"
    data = fixture(tmp_path, procedures=1, extra=extra, statements="call work(a)")
    with pytest.raises(CompilationError, match="direct call closure budget"):
        issue(data)
    assert not data[1]._descriptor_window_proofs


@pytest.mark.parametrize("sealed", [False])
@pytest.mark.parametrize("stage", ["payload", "role", "after_publisher", "final_file"])
@pytest.mark.parametrize("fault", ["binding", "namespace", "syntax", "file", "token", "record", "registry", "entry"])
def test_late_callback_cannot_reseed_cached_token_or_source(tmp_path, monkeypatch, sealed, stage, fault):
    data = fixture(tmp_path, sealed=sealed)
    path, analysis, _graph, _selected = data
    proof = issue(data)
    binding = analysis.routines[ENTRY].scope.bindings["a"]
    fired = False

    def mutate():
        nonlocal fired
        if fired:
            return
        fired = True
        if fault == "binding":
            object.__setattr__(binding, "kind", 4)
        elif fault == "namespace":
            scope = analysis.routines[ENTRY].scope
            object.__setattr__(scope, "bindings", dict(scope.bindings))
        elif fault == "syntax":
            statement = analysis.routines[ENTRY].execution.content[0]
            object.__setattr__(statement, "items", tuple(list(statement.items)))
        elif fault == "file":
            path.write_text(path.read_text().replace("+1.0_8", "+2.0_8"))
        elif fault == "token":
            object.__setattr__(proof, "_selected", tuple(list(proof._selected)))
        elif fault == "record":
            object.__setattr__(proof, "_record", service._freeze(proof.public() | {"extra": False}))
        elif fault == "registry":
            analysis._descriptor_window_proofs = dict(analysis._descriptor_window_proofs)
        else:
            analysis._descriptor_window_proofs[proof.identity] = copy(proof)

    if stage == "payload":
        original = type(analysis.inputs).identity

        def identity(inputs):
            value = original(inputs)
            frame = currentframe().f_back
            if inputs is analysis.inputs and frame.f_code.co_name == "_derive" and "check" in frame.f_locals:
                mutate()
            return value

        monkeypatch.setattr(type(analysis.inputs), "identity", identity)
    elif stage == "role":
        original = service._validate_scope_roles

        def roles(*args, **kwargs):
            value = original(*args, **kwargs)
            if args[0] is analysis and currentframe().f_back.f_code.co_name == "publish":
                mutate()
            return value

        monkeypatch.setattr(service, "_validate_scope_roles", roles)
    elif stage == "after_publisher":
        original = service._Check.publish

        def publish(check, *args):
            value = original(check, *args)
            if check.analysis is analysis:
                mutate()
            return value

        monkeypatch.setattr(service._Check, "publish", publish)
    else:
        original = type(analysis.inputs).verify_live

        def verify(inputs):
            final = inputs is analysis.inputs and currentframe().f_back.f_code.co_name == "_finish_derivation"
            # File I/O must remain inside its trusted verifier. Retained
            # source/token mutations after verification face the pure fence.
            if final and fault == "file":
                mutate()
            value = original(inputs)
            if final and fault != "file":
                mutate()
            return value

        monkeypatch.setattr(type(analysis.inputs), "verify_live", verify)
    with pytest.raises(CompilationError):
        issue(data)
    assert fired


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("fault", ["record", "registry", "owner"])
def test_retained_clock_receipt_survives_the_last_live_file_callback(tmp_path, monkeypatch, sealed, fault):
    data = fixture(tmp_path, sealed=sealed, statements="call system_clock(count=ticks)")
    original = type(data[1].inputs).verify_live
    fired = False

    def verify(inputs):
        nonlocal fired
        value = original(inputs)
        frame = currentframe().f_back
        if inputs is data[1].inputs and frame.f_code.co_name == "_finish_derivation" and not fired:
            fired = True
            check = frame.f_locals["attachments"].check
            _routine, _call, proof = next(iter(check.scalar_clock_tokens.values()))
            if fault == "record":
                object.__setattr__(proof, "_record", "{}")
            elif fault == "registry":
                proof._analysis._scalar_clock_proofs[proof.identity] = copy(proof)
            else:
                proof._analysis._scalar_clock_proofs_owner = copy(proof._analysis)
        return value

    monkeypatch.setattr(type(data[1].inputs), "verify_live", verify)
    with pytest.raises(CompilationError, match="consuming source attachment"):
        issue(data)
    assert fired


def test_admission_retry_cannot_replace_an_existing_registered_window(tmp_path, monkeypatch):
    data = fixture(tmp_path)
    proof = issue(data)
    original = service._derive
    fired = False

    def derive(*args, **kwargs):
        nonlocal fired
        if not fired:
            fired = True
            object.__setattr__(proof, "_record", service._freeze(proof.public()))
            raise service._DescriptorFunctionRequired(ENTRY, object())
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "_derive", derive)
    with pytest.raises(CompilationError, match="registered interval changed"):
        issue(data)
    assert fired


@pytest.mark.parametrize("slot", ["source_roles", "namespace", "registry", "record", "clock_registry"])
def test_final_pure_fence_never_calls_foreign_container_hooks(tmp_path, monkeypatch, slot):
    data = fixture(tmp_path, sealed=False, statements="call system_clock(count=ticks)")
    proof = issue(data)
    original = type(data[1].inputs).verify_live
    fired = False

    class Foreign(dict):
        def __len__(self):
            pytest.fail("foreign length hook ran after the final verifier")

        def get(self, *args):
            pytest.fail("foreign lookup hook ran after the final verifier")

        def items(self):
            pytest.fail("foreign iteration hook ran after the final verifier")

    def verify(inputs):
        nonlocal fired
        value = original(inputs)
        frame = currentframe().f_back
        if inputs is data[1].inputs and frame.f_code.co_name == "_finish_derivation" and not fired:
            fired = True
            analysis = data[1]
            if slot == "source_roles":
                analysis._source_roles = Foreign(analysis._source_roles)
            elif slot == "namespace":
                scope = analysis.routines[ENTRY].scope
                object.__setattr__(scope, "bindings", Foreign(scope.bindings))
            elif slot == "registry":
                analysis._descriptor_window_proofs = Foreign(analysis._descriptor_window_proofs)
            elif slot == "record":
                object.__setattr__(proof, "_record", Foreign())
            else:
                check = frame.f_locals["attachments"].check
                _routine, _call, clock = next(iter(check.scalar_clock_tokens.values()))
                clock._analysis._scalar_clock_proofs = Foreign(clock._analysis._scalar_clock_proofs)
        return value

    monkeypatch.setattr(type(data[1].inputs), "verify_live", verify)
    with pytest.raises(CompilationError):
        issue(data)
    assert fired


def test_foreign_factory_registry_is_rejected_without_using_its_hooks(tmp_path):
    data = fixture(tmp_path)

    class Foreign(dict):
        def items(self):
            pytest.fail("foreign factory registry was consumed")

    data[1]._descriptor_window_proofs = Foreign()
    with pytest.raises(CompilationError, match="compiler-owned proof registry"):
        issue(data)


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_query_baseline_precedes_original_binding_payload_callback(tmp_path, monkeypatch, sealed, cached):
    from compiler.frontend.source_effects import Binding

    data = fixture(tmp_path, sealed=sealed)
    if cached:
        issue(data)
    binding = data[1].routines[ENTRY].scope.bindings["a"]
    original = Binding.public
    fired = False

    def public(current):
        nonlocal fired
        record = original(current)
        frame = currentframe().f_back
        if (current is binding and frame.f_code.co_name == "_derive"
                and frame.f_globals.get("__name__") == service.__name__ and not fired):
            fired = True
            object.__setattr__(binding, "shape_nodes", tuple(list(binding.shape_nodes)))
        return record

    monkeypatch.setattr(Binding, "public", public)
    if sealed and cached:
        issue(data)
        assert not fired
        return
    with pytest.raises(CompilationError):
        issue(data)
    assert fired


@pytest.mark.parametrize("slot", ["binding", "syntax", "token", "property"])
def test_final_raw_fence_rejects_getter_replacement_without_invoking_it(tmp_path, monkeypatch, slot):
    data = fixture(tmp_path, sealed=False)
    proof = issue(data)
    original = type(data[1].inputs).verify_live
    fired = False

    def getter(_value, _name):
        pytest.fail("foreign instance getter ran after the final verifier")

    def property_getter(_value):
        pytest.fail("foreign field descriptor ran after the final verifier")

    def verify(inputs):
        nonlocal fired
        value = original(inputs)
        if inputs is data[1].inputs and currentframe().f_back.f_code.co_name == "_finish_derivation" and not fired:
            fired = True
            if slot == "binding":
                target = data[1].routines[ENTRY].scope.bindings["a"]
            elif slot == "syntax":
                target = data[1].routines[ENTRY].execution.content[0]
            else:
                target = proof
            if slot == "property":
                monkeypatch.setattr(type(target), "_record", property(property_getter), raising=False)
            else:
                monkeypatch.setattr(type(target), "__getattribute__", getter)
        return value

    monkeypatch.setattr(type(data[1].inputs), "verify_live", verify)
    with pytest.raises(CompilationError):
        issue(data)
    assert fired


@pytest.mark.parametrize("slot", ["map", "list"])
def test_final_container_type_change_cannot_run_a_foreign_length_hook(tmp_path, monkeypatch, slot):
    data = fixture(tmp_path, sealed=True)
    original = type(data[1].inputs).verify_live
    target = data[1]._source_roles if slot == "map" else data[1].routines[ENTRY].scope.wildcards
    fired = False

    class Foreign(type(target)):
        __slots__ = ()

        def __len__(self):
            pytest.fail("foreign container length ran after the final verifier")

    def verify(inputs):
        nonlocal fired
        value = original(inputs)
        if inputs is data[1].inputs and currentframe().f_back.f_code.co_name == "_finish_derivation" and not fired:
            fired = True
            object.__setattr__(target, "__class__", Foreign)
        return value

    monkeypatch.setattr(type(data[1].inputs), "verify_live", verify)
    with pytest.raises(CompilationError):
        issue(data)
    assert fired


@pytest.mark.parametrize("sealed", [False, True])
def test_factory_query_keeps_module_membership_without_walking_unused_sibling_bodies(tmp_path, sealed):
    extra = ("subroutine unused(a)\nreal(8),intent(inout)::a(:)\n"
             + "\n".join(f"a(1)=a(1)+{i}.0_8" for i in range(200))
             + "\nend subroutine\n")
    data = fixture(tmp_path, sealed=sealed, extra=extra)
    analysis = data[1]
    receipt = service._FactoryAttachments(analysis, ENTRY).query
    retained = {id(state[0]) for state in receipt.fields}
    own, unused = analysis.routines[ENTRY], analysis.routines["fresh::unused"]
    assert id(own.execution.content[0]) in retained
    assert id(unused.scope.node) not in retained
    assert not any(id(statement) in retained for statement in unused.execution.content)
    module = analysis.modules["fresh"].node
    members = next(node for node in module.content if type(node).__name__ == "Module_Subprogram_Part")
    assert any(value is members.content and any(item is unused.scope.node for item in items)
               for value, _runtime_type, kind, items in receipt.containers if kind == "sequence")
    assert issue(data).public()["source_call_closure"] == {}


@pytest.mark.parametrize("sealed", [False, True])
def test_visited_callee_keeps_its_complete_source_in_the_publication_receipt(tmp_path, monkeypatch, sealed):
    extra = ("subroutine work(a)\nreal(8),intent(inout)::a(:)\na(1)=2.0_8\na(2)=3.0_8\nend subroutine\n"
             "subroutine unused(a)\nreal(8),intent(inout)::a(:)\na(1)=4.0_8\nend subroutine\n")
    data = fixture(tmp_path, sealed=sealed, extra=extra, statements="call work(a)")
    original = service._finish_derivation
    seen = False

    def finish(analysis, receipt):
        nonlocal seen
        retained = {id(state[0]) for state in receipt.fields}
        work = analysis.routines["fresh::work"]
        assert all(id(statement) in retained for statement in work.execution.content)
        assert not any(id(statement) in retained for statement in analysis.routines["fresh::unused"].execution.content)
        seen = True
        return original(analysis, receipt)

    monkeypatch.setattr(service, "_finish_derivation", finish)
    assert list(issue(data).public()["source_call_closure"]) == ["fresh::work"]
    assert seen


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("target", ["callee_tail", "module_declaration", "module_membership"])
def test_narrow_receipt_still_rejects_late_reached_source_or_namespace_mutation(tmp_path, monkeypatch, sealed, target):
    extra = "subroutine work(a)\nreal(8),intent(inout)::a(:)\na(1)=2.0_8\na(2)=3.0_8\nend subroutine\n"
    data = fixture(tmp_path, sealed=sealed, extra=extra, statements="call work(a)")
    original = type(data[1].inputs).verify_live
    fired = False

    def verify(inputs):
        nonlocal fired
        value = original(inputs)
        if inputs is data[1].inputs and currentframe().f_back.f_code.co_name == "_finish_derivation" and not fired:
            fired = True
            analysis = data[1]
            if target == "callee_tail":
                node = analysis.routines["fresh::work"].execution.content[-1]
                object.__setattr__(node, "items", tuple(list(node.items)))
            elif target == "module_declaration":
                scope = analysis.modules["fresh"]
                object.__setattr__(scope, "imports", dict(scope.imports))
            else:
                part = next(node for node in analysis.modules["fresh"].node.content
                            if type(node).__name__ == "Module_Subprogram_Part")
                object.__setattr__(part, "content", tuple(list(part.content)))
        return value

    monkeypatch.setattr(type(data[1].inputs), "verify_live", verify)
    with pytest.raises(CompilationError):
        issue(data)
    assert fired


@pytest.mark.parametrize("sealed", [False])
@pytest.mark.parametrize("mode", ["native", "storage"])
@pytest.mark.parametrize("fault", ["callee_tail", "token_record", "registry"])
def test_expression_proof_and_complete_function_source_remain_owned_at_final_publication(
        tmp_path, monkeypatch, sealed, mode, fault):
    extra = "pure real(8) function twice(x)\nreal(8),intent(in)::x\ntwice=2.0_8*x\nend function\n"
    statements = "a(1)=twice(1.0_8)" if mode == "native" else "a(1)=twice(a(1))"
    data = fixture(tmp_path, sealed=sealed, extra=extra, statements=statements)
    analysis = data[1]
    if mode == "native":
        from compiler.frontend.source_effects import _Closure
        from compiler.frontend.source_functions import prove_native_source_function

        expression = analysis.routines[ENTRY].execution.content[0].items[2]
        native, _effects = prove_native_source_function(analysis, ENTRY, expression, (), _Closure())
        assert analysis._native_functions[native.identity] is native
    issue(data)
    original = type(analysis.inputs).verify_live
    fired = False

    def verify(inputs):
        nonlocal fired
        value = original(inputs)
        if inputs is analysis.inputs and currentframe().f_back.f_code.co_name == "_finish_derivation" and not fired:
            fired = True
            name = "_native_functions" if mode == "native" else "_storage_mutation_effects"
            registry = getattr(analysis, name)
            proof = next(iter(registry.values()))
            if fault == "callee_tail":
                node = analysis.numerical_helpers["fresh::twice"].execution.content[-1]
                object.__setattr__(node, "items", tuple(list(node.items)))
            elif fault == "token_record":
                object.__setattr__(proof, "_record", service._freeze(service._thaw(proof._record)))
            else:
                registry[proof.identity] = copy(proof)
        return value

    monkeypatch.setattr(type(analysis.inputs), "verify_live", verify)
    with pytest.raises(CompilationError):
        issue(data)
    assert fired


@pytest.mark.parametrize("sealed", [False])
@pytest.mark.parametrize("fault", ["token", "record", "registry"])
def test_implicit22_retained_team_tokens_stay_exact_through_final_publication(
        tmp_path, monkeypatch, sealed, fault):
    from compiler.tests.test_descriptor_team_storage import ENTRY as team_entry
    from compiler.tests.test_descriptor_team_storage import fixture as team_fixture

    _path, analysis, _routine, _group, selected = team_fixture(tmp_path, inner_end="")
    if sealed:
        analysis.seal_session()
    proof = service.prove_descriptor_window(analysis, team_entry, selected, (ROOT,))
    assert proof.public()["schema_version"] == 22
    original = type(analysis.inputs).verify_live
    fired = False

    def verify(inputs):
        nonlocal fired
        value = original(inputs)
        caller = currentframe().f_back
        if inputs is analysis.inputs and caller.f_code.co_name == "_finish_derivation" and not fired:
            check = caller.f_locals["attachments"].check
            assert check.storage_teams
            token = check.storage_teams[0]
            assert token.public()["implicit_worksharing_endings"]
            assert token._analysis._descriptor_team_proofs[token.identity] is token
            fired = True
            if fault == "token":
                check.storage_teams[0] = copy(token)
            elif fault == "record":
                object.__setattr__(token, "_record", (token._record + " ").rstrip())
            else:
                token._analysis._descriptor_team_proofs[token.identity] = copy(token)
        return value

    monkeypatch.setattr(type(analysis.inputs), "verify_live", verify)
    with pytest.raises(CompilationError):
        service.prove_descriptor_window(analysis, team_entry, selected, (ROOT,))
    assert fired


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("key_kind", ["object", "str_subclass"])
@pytest.mark.parametrize("position", ["factory_entry", "factory_final", "consumed_capture", "consumed_final"])
def test_colliding_foreign_registry_keys_reject_without_dispatching_equality(
        tmp_path, monkeypatch, sealed, key_kind, position):
    data = fixture(tmp_path, sealed=sealed)
    analysis = data[1]
    proof = issue(data)
    registry = analysis._descriptor_window_proofs
    armed, hooks = False, []

    def foreign_hash(self):
        return hash(proof.identity)

    def foreign_equal(self, other):
        if armed:
            hooks.append(other)
            raise AssertionError("foreign registry equality must never run inside a raw fence")
        return False

    base = str if key_kind == "str_subclass" else object
    key_type = type("ForeignDigest", (base,), {"__hash__": foreign_hash, "__eq__": foreign_equal})
    foreign = key_type("foreign-digest") if key_kind == "str_subclass" else key_type()

    def install():
        nonlocal armed
        held = tuple(dict.items(registry))
        dict.clear(registry)
        # The colliding foreign key must precede the real digest in the hash
        # probe chain. Setup is disarmed; only receipt lookup is observed.
        registry[foreign] = object()
        for key, value in held:
            registry[key] = value
        hooks.clear()
        armed = True

    if position == "factory_final":
        original = type(analysis.inputs).verify_live
        fired = False

        def verify(inputs):
            nonlocal fired
            value = original(inputs)
            if inputs is analysis.inputs and currentframe().f_back.f_code.co_name == "_finish_derivation":
                fired = True
                install()
            return value

        monkeypatch.setattr(type(analysis.inputs), "verify_live", verify)
        if sealed:
            assert issue(data) is proof
            assert not fired
            assert not armed
            return
        with pytest.raises(CompilationError):
            issue(data)
        assert fired
    elif position == "consumed_final":
        receipt = service._PublicationAttachments(analysis, query=proof)
        assert any(owner is analysis and token is proof
                   for owner, _name, _registry, _identity, token, _marker, _terminating in receipt.registrations)
        install()
        with pytest.raises(CompilationError):
            receipt.require()
    else:
        install()

        def invoke():
            if position == "factory_entry":
                return issue(data)
            return service._PublicationAttachments(analysis, query=proof)

        with pytest.raises(CompilationError):
            invoke()
    assert armed
    assert hooks == []
