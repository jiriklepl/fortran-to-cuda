"""Only source-preserved invocation facts seed a local graph's first epoch."""
# ruff: noqa: F811 - pytest discovers the imported module-scoped runtime fixture.
import ctypes as c
import json
from copy import copy

import pytest

from compiler.ir import CompilationError
from compiler.scopes.call_groups import OwnerLocalCallGroup, _candidates
from compiler.scopes.graph_epochs import initial_entry_snapshot
from compiler.tests.test_owner_local_call_groups import SOURCE, fixture, publish
from compiler.tests.test_scoped_runtime import Scope, access, runtime  # noqa: F401


def prepared(tmp_path, *, lifetime="invocation", initialized="whole", prefix="a(:)=a(:)+1.0_8"):
    path, builder = fixture(tmp_path, SOURCE.replace("call unknown_prefix()", prefix), lifetime=lifetime)
    for fact in builder.facts["captures"].values():
        fact["initialized"] = initialized
    return path, builder


def test_typed_original_prefix_preservation_emits_nested_v3_and_first_snapshot_only(tmp_path):
    path, builder = prepared(tmp_path)
    outputs, report = builder.run()
    publish(tmp_path, outputs, report)
    assert report["scope_count"] == 1, report["boundaries"]
    scope, = report["scopes"]
    plan = scope["checked_execution_plan"]
    epoch = plan["storage_epoch"]["selected_position_epoch"]
    assert plan["schema_version"] == 4
    assert epoch["version"] == 3
    assert epoch == scope["owner_local_call_group"]
    proof = epoch["entry_preservation"]
    assert proof["version"] == 2
    assert proof["entry_capture_facts_identity"] == plan["capture_facts_identity"]
    assert proof["sequence_exit_identity"] == epoch["sequence_exit"]["identity"]
    assert set(proof["preserved_resources"]) == {"argument::a", "argument::b"}
    replacement = outputs[report["sources"][str(path)]["replacement"]]
    root = replacement.lower().split("subroutine run(a,b,n,flag)", 1)[1]
    epoch_name = next(line.split("=")[0].strip() for line in root.splitlines()
                      if "fort_entry_epoch_" in line and "= .true." in line)
    assert root.count(epoch_name + " = .true.") == 1
    assert root.count(epoch_name + " = .false.") >= 1
    first = next(member for member in scope["module_coordinators"]
                 if member["procedure"] == "local_groups::first")
    diagnostic, = first["body_intent_diagnostics"]
    assert diagnostic["declared_intent"] == "out"
    assert diagnostic["graph_body_intent"] == "body_effects"
    assert diagnostic["entry_definitions_inferred"] is False
    assert diagnostic["strict_fortran_undefinedness_extension"] is True
    assert diagnostic["incoming_definition_authority"] == (
        "existing caller runtime definitions only; absent coverage remains absent")
    (tmp_path/"entry-preservation.json").write_text(json.dumps(proof, indent=2)+"\n")


@pytest.mark.parametrize(("lifetime", "initialized", "prefix"), [
    ("selected_window", "whole", "a(:)=a(:)+1.0_8"),
    ("invocation", "none", "a(:)=a(:)+1.0_8"),
    ("invocation", "whole", "call unknown_prefix()"),
])
def test_no_independent_preserved_entry_snapshot_keeps_exact_v2_none_route(tmp_path, lifetime, initialized, prefix):
    path, builder = prepared(tmp_path, lifetime=lifetime, initialized=initialized, prefix=prefix)
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report["boundaries"]
    epoch = report["scopes"][0]["checked_execution_plan"]["storage_epoch"]["selected_position_epoch"]
    assert epoch["version"] == 2
    assert "entry_preservation" not in epoch
    root = outputs[report["sources"][str(path)]["replacement"]].lower().split("subroutine run(a,b,n,flag)", 1)[1]
    assert not any("fort_entry_epoch_" in line and "= .true." in line for line in root.splitlines())


def test_preserved_capture_requires_original_invocation_fact_and_exact_registered_token(tmp_path):
    _path, builder = prepared(tmp_path)
    owner = OwnerLocalCallGroup(builder, next(_candidates(builder)))
    epoch = owner.position_epoch
    a = builder.entry.scope.bindings["a"]
    assert initial_entry_snapshot(owner)
    assert epoch.capture(builder, a, builder.capture_fact(a))["initialized"] == "whole"
    assert epoch.capture(builder, a, {**builder.capture_fact(a), "lifetime": "selected_window"})["initialized"] == "none"
    assert epoch.capture(builder, a, {**builder.capture_fact(a), "initialized": "none"})["initialized"] == "none"
    object.__setattr__(epoch, "_entry_preservation", copy(epoch._entry_preservation))
    with pytest.raises(CompilationError):
        initial_entry_snapshot(owner)


def test_close_rearm_and_ordinary_out_policy_do_not_infer_a_new_entry_snapshot(tmp_path):
    _path, builder = prepared(tmp_path)
    owner = OwnerLocalCallGroup(builder, next(_candidates(builder)))
    assert owner.run(owner.closed_selection)
    close = "\n".join(owner.close(prework=True))
    assert owner.epoch_pristine + " = .false." in close
    first = owner.members["local_groups::first"]
    # The existing graph policy preserves prior coverage for ordinary numeric
    # OUT arrays. It neither defines absent points nor rearms the entry flag.
    assert first.body_input_intents == frozenset({"argument::b"})
    assert first.routine.scope.bindings["b"].intent == "out"
    diagnostic, = first.body_intent_diagnostics
    assert diagnostic["entry_definitions_inferred"] is False
    assert diagnostic["strict_fortran_undefinedness_extension"] is True
    assert owner.epoch_pristine not in "\n".join(owner.rearm_epoch())
    assert first.epoch_pristine not in "\n".join(first.rearm_epoch())


@pytest.mark.parametrize("used", [False, True])
def test_used_fixed_module_capture_gets_same_descriptor_proof_while_unused_fact_stays_outside(tmp_path, used):
    source = SOURCE.replace("module local_groups\ncontains", "module local_groups\nreal(8)::metadata(3,6)\ncontains")
    source = source.replace("call unknown_prefix()", "a(:)=a(:)+1.0_8")
    if used:
        source = source.replace("b(i)=real(i,8)", "b(i)=real(i,8)+metadata(1,1)")
    path, builder = fixture(tmp_path, source, lifetime="invocation")
    for fact in builder.facts["captures"].values():
        fact["initialized"] = "whole"
    root = "local_groups::metadata"
    builder.facts["captures"][root] = dict(builder.facts["captures"]["argument::a"])
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report["boundaries"]
    epoch = report["scopes"][0]["owner_local_call_group"]
    if used:
        assert root in epoch["resources"]
        assert root in epoch["entry_preservation"]["preserved_resources"]
        proof = builder.analysis._descriptor_window_proofs[epoch["descriptor_identity"]]
        state = proof.resource_state(root)
        assert state["resource_kind"] == "original_module_fixed_array"
        assert state["entry_definition_authority"] is False
        assert state["payload_access_authorized"] is False
    else:
        assert epoch["version"] == 3
        assert root not in epoch["resources"]
        assert root not in epoch["descriptor_guard_resources"]
        assert root not in epoch["entry_preservation"]["preserved_resources"]
        assert root in builder.facts["captures"]
        replacement = outputs[report["sources"][str(path)]["replacement"]].lower()
        assert "c_loc(metadata)" not in replacement


@pytest.mark.parametrize("declaration", ["real(8),target::metadata(3,6)",
                                         "integer::width=3\nreal(8)::metadata(width,6)"])
def test_used_fixed_module_capture_cannot_bypass_issuer_alias_or_static_shape_rules(tmp_path, declaration):
    source = SOURCE.replace("module local_groups\ncontains", "module local_groups\n"+declaration+"\ncontains")
    source = source.replace("call unknown_prefix()", "a(:)=a(:)+1.0_8")
    source = source.replace("b(i)=real(i,8)", "b(i)=real(i,8)+metadata(1,1)")
    _path, builder = fixture(tmp_path, source, lifetime="invocation")
    for fact in builder.facts["captures"].values():
        fact["initialized"] = "whole"
    builder.facts["captures"]["local_groups::metadata"] = dict(builder.facts["captures"]["argument::a"])
    _outputs, report = builder.run()
    assert report["scope_count"] == 0
    assert not report["source_edits"]


def test_used_fixed_module_with_no_independent_entry_definitions_stays_none(tmp_path):
    source = SOURCE.replace("module local_groups\ncontains", "module local_groups\nreal(8)::metadata(3,6)\ncontains")
    source = source.replace("call unknown_prefix()", "a(:)=a(:)+1.0_8")
    source = source.replace("b(i)=real(i,8)", "metadata(1,1)=real(i,8)\nb(i)=metadata(1,1)")
    _path, builder = fixture(tmp_path, source, lifetime="invocation")
    for fact in builder.facts["captures"].values():
        fact["initialized"] = "whole"
    binding = builder.analysis.modules["local_groups"].bindings["metadata"]
    builder.facts["captures"][binding.root] = {**builder.facts["captures"]["argument::a"], "initialized": "none"}
    owner = OwnerLocalCallGroup(builder, next(_candidates(builder)))
    proof = owner.position_epoch._descriptor
    assert proof.resource_state(binding.root)["resource_kind"] == "original_module_fixed_array"
    assert owner.position_epoch.capture(builder, binding, builder.capture_fact(binding))["initialized"] == "none"


@pytest.mark.native
@pytest.mark.parametrize("partial", [False, True])
def test_preserved_initializer_cannot_restore_a_consumed_definition_kill(tmp_path, runtime, partial):
    """A retained entry proof cannot undo an actual runtime undefinition."""
    _path, builder = prepared(tmp_path)
    owner = OwnerLocalCallGroup(builder, next(_candidates(builder)))
    a = builder.entry.scope.bindings["a"]
    fact = owner.position_epoch.capture(builder, a, builder.capture_fact(a))
    assert fact["initialized"] == "whole"
    scope = Scope(runtime)
    host = (c.c_double * 8)(*range(8))
    status, buffer = scope.register(host, [8], initialized=fact["initialized"] == "whole")
    scope.check(status)
    device = scope.gpu(buffer, access(flags=6))
    for index in range(8):
        device[index] = 1000 + index
    scope.gpu_end(buffer)
    if partial:
        from compiler.tests.test_scoped_runtime import SIZE, TOKEN, Section

        forgotten = access(read=[([2], [6])])
        runtime.fort_scope_forget_sections_v1.argtypes = [TOKEN, TOKEN, c.POINTER(Section), SIZE]
        scope.check(runtime.fort_scope_forget_sections_v1(
            scope.handle, buffer, forgotten.reads, forgotten.read_count))
    else:
        scope.check(runtime.fort_scope_forget_definition(scope.handle, buffer))
    # Even a repeated WHOLE initializer on this same canonical allocation is
    # ignored; its old entry fact cannot restore the killed points.
    status, duplicate = scope.register(host, [8], initialized=fact["initialized"] == "whole")
    scope.check(status)
    assert duplicate.value == buffer.value
    forbidden = access(read=[([3], [4])])
    assert runtime.fort_scope_host_begin(scope.handle, buffer, c.byref(forbidden)) == 7
    assert scope.stats().download_bytes == 0
    scope.close()
    assert list(host) == [1000 + index for index in range(8)]
