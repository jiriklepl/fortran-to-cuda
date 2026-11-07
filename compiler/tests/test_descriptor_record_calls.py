"""Whole original record CALLs use independent team and storage consumers."""
import json
from copy import copy

import pytest

from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.source_functions import _hash
from compiler.ir import CompilationError
from compiler.tests.test_storage_record_regions import SOURCE


def fixture(tmp_path, source=SOURCE, *, sealed=False, dummy=False, contained=False):
    if dummy:
        source = source.replace("subroutine advance()", "subroutine advance(a)\nreal(8),allocatable,intent(inout)::a(:)")
        source = source.replace("call touch(records,field)", "call touch(records,a)")
    if contained:
        source = source.replace("end subroutine\nsubroutine touch", "contains\nsubroutine touch", 1)
        source = source.replace("end subroutine\nend module", "end subroutine\nend subroutine\nend module")
    path = tmp_path / "regions.f90"
    path.write_text(source)
    analysis = SourceEffects([path], operations=2048)
    if sealed:
        analysis.seal_session()
    routine = analysis.routines["regions::advance"]
    originals = tuple(routine.execution.content)
    graph, selected = analysis._selected_source(routine.qualified, originals)
    roots, modules = (("argument::a",), ()) if dummy else ((), ("regions::field",))
    return analysis, routine, selected, roots, modules


def prove(data):
    analysis, routine, selected, roots, modules = data
    return prove_descriptor_window(analysis, routine.qualified, selected, roots, module_roots=modules)


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("dummy", [False, True])
@pytest.mark.parametrize("contained", [False, True])
def test_complete_original_call_consumes_join_and_full_storage_body(tmp_path, sealed, dummy, contained):
    data = fixture(tmp_path, sealed=sealed, dummy=dummy, contained=contained)
    proof = prove(data)
    public = proof.public()
    assert public["schema_version"] == 27
    call, = public["owned_record_call_requirements"]
    assert call["complete_original_body_and_hidden_effects_scanned"]
    assert call["team_storage"]["version"] == 5
    assert call["storage_mutation"]["version"] == 10
    assert call["storage_mutation"]["complete"]
    assert not call["storage_mutation"]["frontiers"]
    assert not call["storage_mutation"]["mutations"]
    assert not call["storage_mutation"]["escapes"]
    assert call["team_storage"]["requires_complete_descriptor_body_and_call_rescan"]
    for name in ("payload_access_authorized", "execution_authorized", "entry_definition_authority",
                 "numerical_completion_authorized", "alias_authority"):
        assert call[name] is False
    assert proof.validate(data[0], data[1].qualified, data[2]) is proof


@pytest.mark.parametrize("effect", ["call unknown(a)", "deallocate(field)", "nullify(p)", "p=>points(i)"])
def test_whole_original_body_cannot_omit_hidden_escape_or_lifetime_tail(tmp_path, effect):
    data = fixture(tmp_path, SOURCE.replace("a(i)=p%value", "a(i)=p%value\n"+effect), sealed=True)
    with pytest.raises(CompilationError):
        prove(data)
    assert not getattr(data[0], "_descriptor_window_proofs", {})


def test_hidden_module_payload_is_scanned_but_does_not_authorize_execution(tmp_path):
    data = fixture(tmp_path, SOURCE.replace("a(i)=p%value", "a(i)=p%value\nhidden=p%value"), sealed=True)
    public = prove(data).public()
    call, = public["owned_record_call_requirements"]
    assert "regions::hidden" in call["storage_mutation"]["resources"]
    assert not call["execution_authorized"]


def test_ordinary_dummy_cannot_be_promoted_to_owned_allocation(tmp_path):
    source = SOURCE.replace("subroutine advance()", "subroutine advance(a)\nreal(8),intent(inout)::a(:)")
    source = source.replace("call touch(records,field)", "call touch(records,a)")
    data = fixture(tmp_path, source)
    data = (*data[:3], ("argument::a",), ())
    with pytest.raises(CompilationError, match="allocatable ownership"):
        prove(data)


SCALAR = """subroutine scalar_step(x)
real(8),intent(inout)::x
x=x+1.0_8
end subroutine
"""


@pytest.mark.parametrize("contained", [False, True])
def test_scalar_component_actual_uses_complete_primitive_source_effects(tmp_path, contained):
    source = SOURCE.replace("a(i)=p%value", "call scalar_step(p%value)\na(i)=p%value")
    source = source.replace("end module", SCALAR+"end module")
    data = fixture(tmp_path, source, contained=contained, sealed=True)
    call, = prove(data).public()["owned_record_call_requirements"]
    expected = "regions::advance::scalar_step" if contained else "regions::scalar_step"
    closure = call["storage_mutation"]["closure"]
    assert {row["procedure"] for row in closure} == {call["owned_record_region"]["callee"], expected}
    member, = [row for row in closure if row["procedure"] == expected]
    assert member["body_sha256"] == _hash(str(data[0].routines[member["procedure"]].scope.node))
    assert not call["execution_authorized"]


@pytest.mark.parametrize("effect", ["call unknown(x)", "deallocate(field)", "call export(x)"])
def test_scalar_component_call_keeps_hidden_callee_effects_and_unknown_suffix(tmp_path, effect):
    source = SOURCE.replace("a(i)=p%value", "call scalar_step(p%value)\na(i)=p%value")
    source = source.replace("end module", SCALAR.replace("x=x+1.0_8", effect)+"end module")
    with pytest.raises(CompilationError):
        prove(fixture(tmp_path, source, sealed=True))



def emitted_source():
    from compiler.tests.test_graph_storage_epochs import SOURCE as EPOCH
    specification = SOURCE.split("contains", 1)[0].replace("module regions", "module epochs")
    touch = SOURCE.split("subroutine touch", 1)[1].rsplit("end module", 1)[0]
    wrapper = """subroutine record_wrapper(a)
real(8),intent(inout)::a(:)
call touch(records,a)
end subroutine
subroutine touch"""+touch
    source = EPOCH.replace("module epochs\n", specification+"\n", 1)
    source = source.replace("subroutine advance", wrapper+"\nsubroutine advance", 1)
    source = source.replace("call producer(right,n)", "call producer(right,n)\ncall record_wrapper(right)")
    return source


def emit_record_fixture(tmp_path):
    from compiler.tests.test_graph_storage_epochs import builder_at
    path, builder = builder_at(tmp_path, emitted_source())
    outputs, report = builder.run()
    return path, builder, outputs, report


def emitted_record_retirement(path, builder, outputs, report):
    """Follow the adopted owner receipt, not a speculative descriptor registry."""
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    from compiler.scopes.lexical import LexicalBoundaryPatch

    assert report["scope_count"] == 1, report.get("graph_owner_attempt", report.get("boundaries"))
    owner = builder._conditional_return_draft.validate().owner
    member, = [member for member in owner.members.values()
               if member.routine.qualified == "epochs::record_wrapper"]
    patch, = [patch for _first, _last, patch in member.patches
              if isinstance(patch, LexicalBoundaryPatch)]
    assert patch.validate(member, require_source=True) is patch
    assert tuple(map(str, patch.originals)) == ("CALL touch(records, a)",)
    storage = patch.lifecycle.public()
    completion = patch.completion.public()
    assert storage["complete"]
    assert not storage["frontiers"]
    assert not storage["mutations"]
    assert not storage["escapes"]
    assert storage["source_inputs"]["sources"] == builder.facts["sources"]
    assert storage["native_lifetime_context"]["whole_original_call_and_all_callees_required"]
    assert storage["native_lifetime_context"]["retirement_required"]
    assert patch.lifecycle._native_graph is patch.completion._graph
    originals = {**builder.analysis.routines,
                 **{template.procedure: template._routine for template in patch.completion._graph.templates}}
    for row in storage["closure"]:
        assert row["body_sha256"] == _hash(str(originals[row["procedure"]].scope.node))
    for flag in ("payload_access_authorized", "entry_definition_authority", "capture_authorized",
                 "execution_authorized", "motion_authorized", "gpu_legality_established", "alias_authority"):
        assert storage[flag] is False
    assert completion["kind"] == "whole_original_native_call_retirement"
    assert completion["storage_identity"] == storage["identity"]
    assert completion["requires_completed_device_work"]
    assert completion["requires_domain_retirement"]
    assert completion["completion_scope"] == "normal return only"
    assert not completion["termination_paths_return"]
    assert not completion["stable_storage_authorized"]
    group, = completion["original_groups"]
    assert group["procedure"] == "epochs::touch"
    team, = storage["native_callee_structure_authorities"]
    assert team["selected_node_ids"] == group["selected_node_ids"]
    assert team["requires_complete_descriptor_body_and_call_rescan"]
    scope, = report["scopes"]
    assert {name.split("#")[0] for name in scope["gpu_leaves"]} == {"epochs::producer", "epochs::consumer"}
    contract = scope["checked_execution_plan"]["storage_epoch"]["return_contract"]
    returned, = [row for row in contract["members"] if row["procedure"] == member.routine.qualified]
    route, = returned["routes"]
    assert route == {"kind": "reached_retirement", "lifetime": "proved_lifecycle", "rearm_allowed": True,
                     "selected_node_ids": storage["original_selection"]["selected_node_ids"]}
    assert not returned["entry_exit"]["definition_authority"]
    assert not contract["cross_cut_replica_retention_authorized"]
    assert all(row["initialized"] == "none" for row in builder.facts["captures"].values())
    generated = outputs[report["sources"][str(path)]["replacement"]]
    wrapper = generated[generated.index("subroutine record_wrapper("):generated.index("subroutine touch(")]
    assert wrapper.index("fort_scope_resident_close_v1") < wrapper.index("call touch(records,a)")
    original_callees = "subroutine touch" + path.read_text().split("subroutine touch", 1)[1].split("subroutine advance", 1)[0]
    assert original_callees in generated
    assert builder.finish()
    validate_checked_execution_plans(report)
    return storage, completion


def decline_complete_record_call(monkeypatch):
    """Keep the retirement fallback covered after complete CALL admission."""
    from compiler.scopes import segments

    def decline(*_args, **_kwargs):
        raise CompilationError("diagnostic complete-CALL admission decline")

    monkeypatch.setattr(segments, "native_call_fragment", decline)


def emitted_record_retention(path, builder, outputs, report):
    """A stable complete CALL synchronizes payload without retiring storage."""
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    from compiler.scopes.lexical import LexicalBoundaryPatch

    assert report["scope_count"] == 1
    owner = builder._conditional_return_draft.validate().owner
    member, = [member for member in owner.members.values()
               if member.routine.qualified == "epochs::record_wrapper"]
    assert not any(isinstance(patch, LexicalBoundaryPatch) for _first, _last, patch in member.patches)
    operation, = [op for unit in member.units for op in unit.scope.native if op.call_effects is not None]
    assert operation.kind == "complete original native CALL"
    assert tuple(map(str, operation.nodes)) == ("CALL touch(records, a)",)
    proof = operation.call_effects
    assert proof.validate(operation.proof_analysis, member.routine.qualified, operation.nodes) is proof
    assert proof.storage._native_graph is proof.reachability.native_graph
    storage, completion = proof.storage.public(), proof.reachability.native_graph.public()
    assert storage["complete"]
    assert not storage["frontiers"]
    assert not storage["mutations"]
    assert not storage["escapes"]
    assert storage["source_inputs"]["sources"] == builder.facts["sources"]
    assert storage["native_lifetime_context"]["whole_original_call_and_all_callees_required"]
    originals = {**builder.analysis.routines,
                 **{template.procedure: template._routine for template in proof.reachability.native_graph.templates}}
    for row in storage["closure"]:
        assert row["body_sha256"] == _hash(str(originals[row["procedure"]].scope.node))
    for flag in ("payload_access_authorized", "entry_definition_authority", "capture_authorized",
                 "execution_authorized", "motion_authorized", "gpu_legality_established", "alias_authority"):
        assert storage[flag] is False
    for flag in ("alias_authorized", "capture_authorized", "definitions_authorized", "motion_authorized",
                 "internal_cuts_authorized"):
        assert proof.public()[flag] is False
    assert operation.coherence_effects == {"argument::a": {"read", "write"}}
    assert set(operation.host_metadata) == {"epochs::records"}
    assert not operation.overwrites
    assert operation.preserve_original
    assert operation.summary["native_completion"]["available"]
    assert completion["effect_and_association_barriers_absent"]
    assert not completion["retirement_lifetime_obligations"]
    scope, = report["scopes"]
    assert {name.split("#")[0] for name in scope["gpu_leaves"]} == {"epochs::producer", "epochs::consumer"}
    contract = scope["checked_execution_plan"]["storage_epoch"]["return_contract"]
    returned, = [row for row in contract["members"] if row["procedure"] == member.routine.qualified]
    route, = returned["routes"]
    assert route == {"kind": "checked_unit", "ordinal": 0,
                     "selected_node_ids": storage["original_selection"]["selected_node_ids"]}
    assert not returned["entry_exit"]["definition_authority"]
    assert not contract["cross_cut_replica_retention_authorized"]
    assert all(row["initialized"] == "none" for row in builder.facts["captures"].values())
    generated = outputs[report["sources"][str(path)]["replacement"]]
    wrapper = generated[generated.index("subroutine record_wrapper("):generated.index("subroutine touch(")]
    assert wrapper.count("call touch(records,a)") == 1
    assert wrapper.index("fort_scope_original_host_begin_v1") < wrapper.index("call touch(records,a)")
    assert wrapper.index("call touch(records,a)") < wrapper.index("fort_scope_host_end")
    # Rejected preflight/error paths may still close the domain. The normal
    # CALL uses one checked unit, with no lexical lifetime cut (checked above).
    original_callees = "subroutine touch" + path.read_text().split("subroutine touch", 1)[1].split("subroutine advance", 1)[0]
    assert original_callees in generated
    assert builder.finish()
    validate_checked_execution_plans(report, capture_facts=builder.facts)
    return storage, completion


@pytest.mark.parametrize("route", ["retained", "retirement"])
def test_authentic_graph_emission_retains_record_storage_receipt_and_none_facts(tmp_path, monkeypatch, route):
    if route == "retirement":
        decline_complete_record_call(monkeypatch)
    path, builder, outputs, report = emit_record_fixture(tmp_path)
    inspect = emitted_record_retention if route == "retained" else emitted_record_retirement
    storage, completion = inspect(path, builder, outputs, report)
    assert {row["procedure"] for row in storage["closure"]} == {"epochs::touch"}
    (tmp_path/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    (tmp_path/"facts.json").write_text(json.dumps(builder.facts, indent=2)+"\n")
    for name, content in outputs.items():
        destination = tmp_path/"generated"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)
    (tmp_path/(route+"-records.json")).write_text(json.dumps(
        {"storage": storage, "completion": completion}, indent=2)+"\n")
    assert str(path) in report["sources"]
    assert outputs


@pytest.mark.parametrize("fault", ["region_token", "team_token", "storage_token", "numeric_binding", "namespace"])
@pytest.mark.parametrize("sealed", [False, True])
def test_composed_late_publication_retains_source_and_every_independent_issuer(tmp_path, monkeypatch, fault, sealed):
    from compiler.frontend.descriptor_record_calls import original_record_call
    data = fixture(tmp_path, sealed=sealed)
    analysis, routine, _selected, _roots, _modules = data
    call, = routine.execution.content
    numeric = (analysis.modules["regions"].bindings["field"],)
    proof = original_record_call(analysis, routine, call, numeric)
    original = analysis.verify_publication
    fired = False

    def changed():
        nonlocal fired
        original()
        if fired:
            return
        fired = True
        if fault == "region_token":
            analysis._owned_record_regions[proof._region.identity] = copy(proof._region)
        elif fault == "team_token":
            analysis._descriptor_team_proofs[proof._team.identity] = copy(proof._team)
        elif fault == "storage_token":
            analysis._storage_mutation_effects[proof._storage.identity] = copy(proof._storage)
        elif fault == "numeric_binding":
            object.__setattr__(numeric[0], "kind", 4)
        else:
            object.__setattr__(analysis.modules["regions"], "component_types", dict(analysis.modules["regions"].component_types))

    monkeypatch.setattr(analysis, "verify_publication", changed)
    with pytest.raises(CompilationError):
        proof.validate()
    assert fired


def test_team_cannot_consume_copied_association_or_another_original_group(tmp_path):
    from compiler.frontend.descriptor_teams import prove_descriptor_team
    from compiler.frontend.storage_record_regions import prove_owned_record_region
    data = fixture(tmp_path, sealed=True)
    analysis, caller, _selected, _roots, _modules = data
    call, = caller.execution.content
    region = prove_owned_record_region(analysis, caller.qualified, call,
        (analysis.modules["regions"].bindings["field"],))
    with pytest.raises(CompilationError, match="exact registered"):
        prove_descriptor_team(analysis, region._roles[1].qualified, region._roles[6], record_region=copy(region))
    with pytest.raises(CompilationError, match="complete original joined region|membership"):
        prove_descriptor_team(analysis, caller.qualified, region._roles[6], record_region=region)


@pytest.mark.parametrize(("before", "after"), [
    ("type cell", "type cell\nsequence"),
    ("subroutine advance()", "subroutine advance() bind(c)"),
    ("type(cell),target,intent(inout)::points(:)", "class(cell),target,intent(inout)::points(:)"),
])
def test_foreign_or_reinterpretable_storage_never_supplies_allocation_separation(tmp_path, before, after):
    if before.startswith("subroutine"):
        # Keep a legal source header with the allocatable descriptor supplied
        # through BIND(C); the first slice does not claim CFI ownership.
        source = SOURCE.replace("subroutine advance()", "subroutine advance(a) bind(c)\nreal(8),allocatable,intent(inout)::a(:)")
        source = source.replace("call touch(records,field)", "call touch(records,a)")
        data = fixture(tmp_path, source)
        data = (*data[:3], ("argument::a",), ())
    else:
        data = fixture(tmp_path, SOURCE.replace(before, after))
    with pytest.raises(CompilationError):
        prove(data)



def test_full_scalar_closure_is_retained_in_window_source_inventory(tmp_path):
    source = SOURCE.replace("a(i)=p%value", "call scalar_step(p%value)\na(i)=p%value")
    source = source.replace("end module", SCALAR+"end module")
    data = fixture(tmp_path, source, sealed=True)
    public = prove(data).public()
    assert {"regions::touch", "regions::scalar_step"} <= set(public["source_call_closure"])


@pytest.mark.parametrize("declaration", ["real(8),target,intent(inout)::x", "real(8),pointer,intent(inout)::x"])
def test_scalar_field_calls_never_receive_pointer_or_target_callee_waiver(tmp_path, declaration):
    source = SOURCE.replace("a(i)=p%value", "call scalar_step(p%value)\na(i)=p%value")
    source = source.replace("end module", SCALAR.replace("real(8),intent(inout)::x", declaration)+"end module")
    with pytest.raises(CompilationError, match="unknown mutation, escape or incomplete original closure"):
        prove(fixture(tmp_path, source, sealed=True))


@pytest.mark.parametrize("sealed", [False, True])
def test_late_scalar_callee_hidden_global_binding_change_cannot_reseed_baseline(tmp_path, monkeypatch, sealed):
    from compiler.frontend.descriptor_record_calls import original_record_call
    source = SOURCE.replace("a(i)=p%value", "call scalar_step(p%value)\na(i)=p%value")
    source = source.replace("end module", SCALAR.replace("x=x+1.0_8", "hidden=x\nx=x+1.0_8")+"end module")
    analysis, caller, _selected, _roots, _modules = fixture(tmp_path, source, sealed=sealed)
    call, = caller.execution.content
    proof = original_record_call(analysis, caller, call, (analysis.modules["regions"].bindings["field"],))
    original = analysis.verify_publication
    fired = False

    def changed():
        nonlocal fired
        original()
        if not fired:
            fired = True
            object.__setattr__(analysis.modules["regions"].bindings["hidden"], "kind", 4)

    monkeypatch.setattr(analysis, "verify_publication", changed)
    with pytest.raises(CompilationError):
        proof.validate()
    assert fired


@pytest.mark.parametrize("cached", [False, True])
def test_region_encoding_finishes_before_original_live_source_publication(tmp_path, monkeypatch, cached):
    import compiler.frontend.storage_record_regions as regions
    source = SOURCE
    data = fixture(tmp_path, source, sealed=True)
    analysis, caller, _selected, _roots, _modules = data
    call, = caller.execution.content
    numeric = (analysis.modules["regions"].bindings["field"],)
    if cached:
        regions.prove_owned_record_region(analysis, caller.qualified, call, numeric)
        # Encoding belongs to a fresh immutable fact, not a cache hit. Select
        # another legal context while preserving this exact injection check.
        analysis.operation_limit += 1
    original = regions._encoded
    fired = False

    def changed(value):
        nonlocal fired
        result = original(value)
        if isinstance(value, dict) and value.get("kind") == "original_owned_fixed_record_element_region":
            fired = True
            path = tmp_path/"regions.f90"
            path.write_text(path.read_text()+"! encoder changed source\n")
        return result

    monkeypatch.setattr(regions, "_encoded", changed)
    with pytest.raises(CompilationError, match="source changed"):
        regions.prove_owned_record_region(analysis, caller.qualified, call, numeric)
    assert fired


@pytest.mark.parametrize("foreign", [False, True])
def test_cached_standalone_region_issuer_cannot_return_copied_or_foreign_token(tmp_path, foreign):
    from compiler.frontend.storage_record_regions import prove_owned_record_region
    data = fixture(tmp_path, sealed=True)
    analysis, caller, _selected, _roots, _modules = data
    call, = caller.execution.content
    numeric = (analysis.modules["regions"].bindings["field"],)
    proof = prove_owned_record_region(analysis, caller.qualified, call, numeric)
    if foreign:
        other, other_caller, _selected, _roots, _modules = fixture(tmp_path, sealed=True)
        other_call, = other_caller.execution.content
        replacement = prove_owned_record_region(other, other_caller.qualified, other_call,
            (other.modules["regions"].bindings["field"],))
        assert replacement.public() == proof.public()
    else:
        replacement = copy(proof)
    analysis._owned_record_regions[proof.identity] = replacement
    with pytest.raises(CompilationError, match="cached issuer"):
        prove_owned_record_region(analysis, caller.qualified, call, numeric)


def scalar_closure_source(count):
    names = ["scalar_step_"+str(index) for index in range(count)]
    source = SOURCE.replace("a(i)=p%value", "\n".join("call "+name+"(p%value)" for name in names)+"\na(i)=p%value")
    helpers = "".join(SCALAR.replace("scalar_step", name) for name in names)
    return source.replace("end module", helpers+"end module")


@pytest.mark.parametrize("limit", ["procedures", "depth"])
def test_composed_scalar_closure_counts_original_caller_in_source_limits(tmp_path, limit):
    from compiler.frontend.storage_record_regions import prove_owned_record_region
    source = scalar_closure_source(7 if limit == "procedures" else 1)
    data = fixture(tmp_path, source)
    if limit == "procedures":
        data[0].procedure_limit = 8
        analysis, caller, _selected, _roots, _modules = data
        call, = caller.execution.content
        proof = prove_owned_record_region(analysis, caller.qualified, call,
            (analysis.modules["regions"].bindings["field"],))
        assert tuple(proof._roles[8]) == tuple(analysis.routines["regions::touch"].execution.content)
        assert not getattr(analysis, "_original_call_associations", {})
    else:
        data[0].depth_limit = 2
    with pytest.raises(CompilationError, match="composed original call.*budget"):
        prove(data)
    assert not getattr(data[0], "_descriptor_window_proofs", {})


def test_compose_counts_previously_checked_sibling_source_procedures(tmp_path):
    from compiler.frontend.descriptor_windows import _Check
    source = scalar_closure_source(6)
    sibling = """subroutine sibling(a)
real(8),intent(inout)::a(:)
a(1)=a(1)+1.0_8
end subroutine
"""
    source = source.replace("end module", sibling+"end module")
    data = fixture(tmp_path, source)
    analysis, caller, _selected, _roots, _modules = data
    analysis.procedure_limit = 8
    assert prove(data).public()["schema_version"] == 27
    check = _Check(analysis)
    check.procedure("regions::sibling")
    assert set(check.checked) == {"regions::sibling"}
    check.active = [caller.qualified]
    check.record_numeric_allocations = (analysis.modules["regions"].bindings["field"],)
    call, = caller.execution.content
    with pytest.raises(CompilationError, match="composed original call closure budget"):
        check.owned_record_call(caller, call)


@pytest.mark.parametrize("cached", [False, True])
def test_new_team_record_encoding_precedes_final_live_source_check(tmp_path, monkeypatch, cached):
    import compiler.frontend.descriptor_teams as teams
    from compiler.frontend.storage_record_regions import prove_owned_record_region
    analysis, caller, _selected, _roots, _modules = fixture(tmp_path, sealed=True)
    call, = caller.execution.content
    region = prove_owned_record_region(analysis, caller.qualified, call,
        (analysis.modules["regions"].bindings["field"],))
    callee = region._roles[1]
    if cached:
        teams.prove_descriptor_team(analysis, callee.qualified, region._roles[6], record_region=region)
    original = teams._encoded
    fired = False

    def changed(value):
        nonlocal fired
        result = original(value)
        if value.get("version") == 5:
            fired = True
            path = tmp_path/"regions.f90"
            path.write_text(path.read_text()+"! encoder changed source\n")
        return result

    monkeypatch.setattr(teams, "_encoded", changed)
    if cached:
        teams.prove_descriptor_team(analysis, callee.qualified, region._roles[6], record_region=region)
        assert not fired
        return
    with pytest.raises(CompilationError, match="source changed"):
        teams.prove_descriptor_team(analysis, callee.qualified, region._roles[6], record_region=region)
    assert fired


@pytest.mark.parametrize("sealed", [False, True])
def test_final_composition_retains_exact_direct_scalar_call_registry_entry(tmp_path, monkeypatch, sealed):
    import inspect

    from compiler.frontend.descriptor_record_calls import DescriptorRecordCall, original_record_call
    source = scalar_closure_source(1)
    analysis, caller, _selected, _roots, _modules = fixture(tmp_path, source, sealed=sealed)
    call, = caller.execution.content
    proof = original_record_call(analysis, caller, call, (analysis.modules["regions"].bindings["field"],))
    original = analysis.verify_publication
    fired = False

    def changed():
        nonlocal fired
        original()
        if not fired and inspect.currentframe().f_back.f_code is DescriptorRecordCall.validate.__code__:
            fired = True
            token, = proof._storage._calls
            analysis._original_call_associations[(id(analysis), id(token._original))] = copy(token)

    if sealed:
        token, = proof._storage._calls
        analysis._original_call_associations[(id(analysis), id(token._original))] = copy(token)
        with pytest.raises(CompilationError, match="original call authority changed"):
            proof.validate()
        return
    monkeypatch.setattr(analysis, "verify_publication", changed)
    with pytest.raises(CompilationError, match="consumed source, namespace or issuer changed"):
        proof.validate()
    assert fired



def test_record_root_inventory_reuses_original_module_binding_validation(tmp_path, monkeypatch):
    import compiler.frontend.descriptor_windows as windows
    data = fixture(tmp_path)
    original = windows._module_binding
    visits = []

    def counted(analysis, routine, root, check):
        visits.append(root)
        return original(analysis, routine, root, check)

    monkeypatch.setattr(windows, "_module_binding", counted)
    proof = prove(data)
    assert visits == ["regions::field"]
    visits.clear()
    assert prove(data) is proof
    assert visits == ["regions::field"]
