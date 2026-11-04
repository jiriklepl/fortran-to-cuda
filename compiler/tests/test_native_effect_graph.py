"""One source template per routine, distinct effects for every original call."""
from collections import Counter
from copy import copy
from dataclasses import FrozenInstanceError

import pytest

from compiler.frontend import resource_reachability as reachability
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

SOURCE = """module work
implicit none
real(8)::a(4),b(4),source(4),observed
contains
subroutine owner()
call middle(a,source(1)+source(2))
call middle(b,source(3)+source(4))
end subroutine
subroutine middle(x,value)
real(8),intent(inout)::x(:)
real(8),intent(in)::value
call leaf(x,value)
call leaf(x,value)
end subroutine
subroutine leaf(x,value)
real(8),intent(inout)::x(:)
real(8),intent(in)::value
if(value>0.0_8)then
x(1)=x(1)+value
else
x(2)=value
endif
observed=value
end subroutine
end module
"""


def setup(tmp_path, source=SOURCE, *, seal=True, **options):
    path = tmp_path / "graph.f90"
    path.write_text(source)
    analysis = SourceEffects([path], **options)
    if seal:
        analysis.seal_session()
    routine = analysis.routines["work::owner"]
    selection = tuple(routine.execution.content)
    demands = tuple(analysis.modules["work"].bindings[name] for name in ("a", "b", "source", "observed"))
    return path, analysis, selection, demands


def prove(analysis, selection, demands):
    return reachability.prove_resource_reachability(analysis, "work::owner", selection, demands)


def test_shared_nested_templates_keep_every_distinct_actual_and_call(tmp_path, monkeypatch):
    _path, analysis, selection, demands = setup(tmp_path)
    scans = Counter()
    original = reachability._Scan.routine
    def scan(self, name, *args, **kwargs):
        scans[name] += 1
        return original(self, name, *args, **kwargs)
    monkeypatch.setattr(reachability._Scan, "routine", scan)
    proof = prove(analysis, selection, demands)
    assert proof.public()["selection_covered"], proof.public()["barriers"]
    graph = proof.native_graph
    assert scans == {"work::owner": 1, "work::middle": 1, "work::leaf": 1}
    assert len(graph.templates) == 3
    assert len(graph.instances) == 7
    assert len(proof.calls) == 6
    leaf = next(item for item in graph.templates if item.procedure == "work::leaf")
    instances = [item for item in graph.instances if item.template is leaf]
    assert [item.parent for item in instances] == [1, 1, 4, 4]
    formal = analysis.routines["work::leaf"].scope.bindings["x"]
    assert [next(path.binding for binding, path in item.associations if binding is formal)
            for item in instances] == [demands[0], demands[0], demands[1], demands[1]]
    writes = [access.storage.binding for access in proof.accesses
              if access.procedure == "work::leaf" and access.kind == "write"]
    assert writes.count(demands[0]) == 4
    assert writes.count(demands[1]) == 4
    assert writes.count(demands[3]) == 4  # Unmapped module storage survives every instance.
    reads = [item for item in proof.accesses if item.storage.binding is demands[2]]
    assert len(reads) == 4
    assert all(item.procedure == "work::owner" for item in reads)
    assert not any(item.storage.binding.name == "value" for item in proof.accesses)
    # A different demand reuses syntax but still reauthenticates and instantiates.
    other = prove(analysis, selection, (demands[0],))
    assert other.native_graph is graph
    assert scans == {"work::owner": 1, "work::middle": 1, "work::leaf": 1}
    record = graph.public()
    assert record["source_work"]["templates"] == 3
    assert record["instantiation_work"]["call_instances"] == 6
    assert record["source_identity"] == analysis.inputs.identity()
    assert not any(record[name] for name in (
        "execution_authority", "alias_authority", "cost_authority", "workload_key_authority"))
    assert "If_Construct" in str(record["templates"])


def test_template_substitution_appends_derived_paths(tmp_path):
    source = """module work
implicit none
type profile
real(8),allocatable::flux(:)
end type
type container
type(profile)::current
end type
type(container)::left,right
real(8)::a(4),b(4),source(4),observed
contains
subroutine owner()
call middle(left%current)
call middle(right%current)
end subroutine
subroutine middle(p)
type(profile),intent(inout)::p
call leaf(p%flux)
end subroutine
subroutine leaf(x)
real(8),intent(inout)::x(:)
x(1)=source(1)
end subroutine
end module
"""
    _path, analysis, selected, demands = setup(tmp_path, source)
    proof = prove(analysis, selected, demands)
    assert proof.public()["selection_covered"], proof.public()["barriers"]
    writes = [item.storage.resource for item in proof.accesses if item.kind == "write"]
    assert writes == ["work::left%current%flux", "work::right%current%flux"]
    assert len(proof.native_graph.templates) == 3
    assert len(proof.native_graph.instances) == 5


def test_nested_host_association_substitutes_enclosing_formal(tmp_path):
    source = SOURCE.replace("call leaf(x,value)\ncall leaf(x,value)", "call nested()\ncontains\n"
                           "subroutine nested()\nx(1)=value\nend subroutine")
    _path, analysis, selected, demands = setup(tmp_path, source)
    proof = prove(analysis, selected, demands)
    assert proof.public()["selection_covered"], proof.public()["barriers"]
    writes = [item.storage.binding for item in proof.accesses if item.kind == "write"]
    assert writes == [demands[0], demands[1]]


@pytest.mark.parametrize("replacement", ["call missing(x)", "allocate(x(4))", "stop"])
def test_shared_template_retains_barriers_and_every_occurrence(tmp_path, replacement):
    source = SOURCE.replace("observed=value", replacement)
    _path, analysis, selected, demands = setup(tmp_path, source)
    proof = prove(analysis, selected, demands)
    assert not proof.public()["selection_covered"]
    assert any(replacement.split("(")[0] in item.source.lower() for item in proof.barriers)
    assert len(proof.native_graph.instances) == 7
    assert len(proof.calls) == 6  # Unknown calls remain barriers; no invented resolved edge.


def test_recursion_is_rejected_without_issued_graph(tmp_path):
    source = SOURCE.replace("observed=value", "call middle(x,value)")
    _path, analysis, selected, demands = setup(tmp_path, source)
    with pytest.raises(reachability.ResourceReachabilityBudgetExceeded, match="recursive"):
        prove(analysis, selected, demands)
    assert not getattr(analysis, "_native_effect_graphs", {})


def branching_source():
    return """module work
implicit none
real(8)::a(4),b(4),source(4),observed
contains
subroutine owner()
call branch(a)
call branch(b)
end subroutine
subroutine branch(x)
real(8),intent(inout)::x(:)
""" + "call leaf(x)\n" * 4 + """end subroutine
subroutine leaf(x)
real(8),intent(inout)::x(:)
x(1)=1.0_8
end subroutine
end module
"""


def test_call_instances_do_not_inherit_per_template_statement_limit(tmp_path):
    source = branching_source()
    _path, analysis, selected, demands = setup(tmp_path, source, operations=9)
    proof = prove(analysis, selected, demands)
    assert dict(proof.native_graph.source_work)["statements"] == 7
    assert dict(proof.native_graph.instantiation_work)["call_instances"] == 10
    assert len(proof.native_graph.host_instances) == 11
    assert proof.native_graph.retired_host_completion_view().public()["available"]


def expanded_source():
    # 272 leaf invocations, but only three small original templates. The last
    # branch has a different actual: a truncated prefix loses all writes to b.
    return branching_source().replace("call branch(a)\n", "call branch(a)\n" * 16).replace(
        "call leaf(x)\n" * 4, "call leaf(x)\n" * 16)


@pytest.mark.parametrize("payload", [True, False])
def test_shared_call_graph_exceeds_default_window_and_retains_tail(tmp_path, payload):
    _path, analysis, selected, demands = setup(tmp_path, expanded_source())
    graph = reachability.prove_native_effect_graph(analysis, "work::owner", selected, project_payload=payload)
    assert dict(graph.instantiation_work)["call_instances"] == (289 if payload else 0)
    assert len(graph.instances) == (290 if payload else 1)
    assert len(graph.host_instances) == 290 > analysis.operation_limit
    assert len(graph.templates) == 3
    assert graph.retired_host_completion_view().public()["available"]
    if payload:
        record = graph.public()
        assert "call_instances" not in record["limits"]
        assert record["limits"]["substitution_and_obligation_steps"] == reachability.NATIVE_GROUP_SCAN_LIMIT
        proof = prove(analysis, selected, demands)
        assert not proof.barriers
        writes = Counter(item.storage.binding.name for item in proof.accesses if item.kind == "write")
        assert writes == {"a": 256, "b": 16}
    else:
        with pytest.raises(CompilationError, match="control-only graph role"):
            graph.public()


def test_shared_call_graph_retains_late_unknown_effect(tmp_path):
    source = expanded_source().replace("call branch(b)", "call branch(b)\ncall unknown(b)")
    _path, analysis, selected, demands = setup(tmp_path, source)
    proof = prove(analysis, selected, demands)
    assert dict(proof.native_graph.instantiation_work)["call_instances"] == 289
    assert any("unknown(b)" in barrier.source.lower() for barrier in proof.barriers)
    assert not proof.public()["selection_covered"]
    assert not proof.native_graph.retired_host_completion_view().public()["available"]


def test_repeated_accesses_share_alias_work_without_losing_effects(tmp_path, monkeypatch):
    _path, analysis, selected, demands = setup(tmp_path, expanded_source())
    graph = reachability.prove_native_effect_graph(analysis, "work::owner", selected)
    # Two original storage objects compared with four demanded objects. A
    # repeated-call cross product exhausts this bound despite identical work.
    limit = dict(graph.instantiation_work)["substitution_steps"] + 8
    monkeypatch.setattr(reachability, "NATIVE_GROUP_SCAN_LIMIT", limit)
    bounded_dir = tmp_path / "bounded"
    bounded_dir.mkdir()
    _path, analysis, selected, demands = setup(bounded_dir, expanded_source())
    proof = prove(analysis, selected, demands)
    assert not proof.barriers
    writes = Counter(item.storage.binding.name for item in proof.accesses if item.kind == "write")
    assert writes == {"a": 256, "b": 16}
    assert {(item.native.resource, item.demanded.root) for item in proof.obligations} == {
        ("work::a", "work::b"), ("work::a", "work::source"), ("work::a", "work::observed"),
        ("work::b", "work::a"), ("work::b", "work::source"), ("work::b", "work::observed")}
    assert all("write" in item.actions for item in proof.obligations)
    assert proof.public()["scan_work"]["substitution_and_obligation_steps"] == limit


@pytest.mark.parametrize("phase", ["payload", "host"])
def test_shared_call_graph_work_budget_rejects_without_issuing_prefix(tmp_path, monkeypatch, phase):
    _path, analysis, selected, demands = setup(tmp_path, expanded_source())
    proof = prove(analysis, selected, demands)
    graph = proof.native_graph
    steps = dict(graph.instantiation_work)["substitution_steps"]
    # Host discovery charges one step per instance after payload expansion.
    limit = steps - (len(graph.host_instances) if phase == "payload" else 0) - 1
    assert max(dict(t.source_work)["syntax_nodes"] for t in graph.templates) < limit
    monkeypatch.setattr(reachability, "NATIVE_GROUP_SCAN_LIMIT", limit)
    other_dir = tmp_path / "bounded"
    other_dir.mkdir()
    _path, bounded, selected, demands = setup(other_dir, expanded_source())
    with pytest.raises(reachability.ResourceReachabilityBudgetExceeded, match="instantiation budget"):
        prove(bounded, selected, demands)
    assert not getattr(bounded, "_native_effect_graphs", {})
    assert not getattr(bounded, "_resource_reachability_proofs", {})


def test_cached_template_still_charges_source_budget_and_authenticates(tmp_path):
    _path, analysis, selected, demands = setup(tmp_path, branching_source())
    first = prove(analysis, (selected[0],), demands)
    assert dict(first.native_graph.source_work)["statements"] == 6
    second = prove(analysis, selected, demands)
    assert dict(second.native_graph.source_work)["statements"] == 7
    assert first.native_graph.templates[1] is second.native_graph.templates[1]
    leaf = next(item for item in second.native_graph.templates if item.procedure == "work::leaf")
    with pytest.raises(FrozenInstanceError):
        leaf.nodes = ()
    with pytest.raises(CompilationError):
        copy(leaf).validate(analysis)
    with pytest.raises(CompilationError):
        copy(second.native_graph).validate(analysis, "work::owner", selected)
    other = SourceEffects([_path])
    with pytest.raises(CompilationError):
        second.native_graph.validate(other, "work::owner", selected)
    analysis.operation_limit -= 1
    with pytest.raises(CompilationError):
        second.native_graph.validate(analysis, "work::owner", selected)


def test_stale_cached_callee_cannot_issue_new_selection(tmp_path):
    _path, analysis, selected, demands = setup(tmp_path, seal=False)
    first = prove(analysis, (selected[0],), demands)
    leaf = analysis.routines["work::leaf"]
    leaf.scope.bindings["x"].attributes |= {"pointer"}
    with pytest.raises(CompilationError):
        first.native_graph.validate(analysis, "work::owner", (selected[0],))
    with pytest.raises(CompilationError):
        prove(analysis, (selected[1],), demands)


def test_whole_nonallocatable_formal_write_does_not_reallocate_actual(tmp_path):
    source = SOURCE.replace("real(8)::a(4),b(4),source(4),observed",
                            "real(8),allocatable::a(:),b(:)\nreal(8)::source(4),observed").replace(
        "x(1)=x(1)+value", "x=value")
    _path, analysis, selected, demands = setup(tmp_path, source)
    proof = prove(analysis, selected, demands)
    assert proof.public()["selection_covered"], proof.public()["barriers"]
    assert {item.storage.resource for item in proof.accesses if item.source == "x" and item.kind == "write"} == {
        "work::a", "work::b"}


def test_joined_completion_and_pointer_barriers_survive_template_sharing(tmp_path):
    from compiler.tests.test_resource_reachability import SOURCE as NATIVE_SOURCE
    from compiler.tests.test_resource_reachability import fixture
    source = NATIVE_SOURCE.replace("call native(star,n,enabled)",
                                   "call native(star,n,enabled)\ncall native(star,n,enabled)").replace(
        "real(8),allocatable::flux(:)", "real(8),pointer::flux(:)")
    _path, analysis, selected, demands = fixture(tmp_path, source)
    analysis.seal_session()
    proof = prove(analysis, selected, demands)
    assert any(item.kind == "association" for item in proof.barriers)
    add = next(template for template in proof.native_graph.templates if template.procedure == "work::add")
    assert len(add.completion) == 1
    assert len([item for item in proof.native_graph.instances if item.template is add]) == 2
    assert add.public()["completion_identities"] == [add.completion[0][2].identity]
    proof.validate(analysis, "work::owner", selected, demands)


def test_independent_windows_share_templates_under_their_own_procedure_bound(tmp_path):
    _path, analysis, selected, demands = setup(tmp_path, procedures=3)
    first = prove(analysis, (selected[0],), demands)
    registry = analysis._native_effect_templates
    assert len(registry) == 3
    second = prove(analysis, (selected[1],), demands)
    assert analysis._native_effect_templates is registry
    assert len(registry) == 4 > analysis.procedure_limit
    for proof in (first, second):
        assert proof.public()["selection_covered"], proof.public()["barriers"]
        assert dict(proof.native_graph.source_work)["procedures"] == analysis.procedure_limit
        assert len(proof.native_graph.templates) == 3
    original_templates = {item.procedure: item for item in first.native_graph.templates}
    later_templates = {item.procedure: item for item in second.native_graph.templates}
    assert original_templates["work::owner"] is not later_templates["work::owner"]
    assert all(original_templates[name] is later_templates[name] for name in ("work::middle", "work::leaf"))
    assert first.validate(analysis, "work::owner", (selected[0],), demands) is first
    assert second.validate(analysis, "work::owner", (selected[1],), demands) is second


def test_each_original_window_still_enforces_its_complete_call_closure_bound(tmp_path):
    _path, analysis, selected, demands = setup(tmp_path, procedures=2)
    with pytest.raises(reachability.ResourceReachabilityBudgetExceeded, match="procedure budget exhausted"):
        prove(analysis, (selected[0],), demands)
    assert not getattr(analysis, "_native_effect_graphs", {})
    assert not getattr(analysis, "_resource_reachability_proofs", {})


@pytest.mark.parametrize("sealed", [False, True])
def test_source_sized_template_registry_has_a_real_last_slot_and_exhaustion(tmp_path, sealed):
    from compiler.frontend.storage_mutation_effects import source_registry_budget

    _path, analysis, selected, demands = setup(tmp_path, procedures=3, seal=sealed)
    first = prove(analysis, (selected[0],), demands)
    registry = analysis._native_effect_templates
    budget = source_registry_budget(analysis)
    assert budget > len(registry) >= analysis.procedure_limit
    # Test-only saturation: these keys are not new issued source templates.
    # They retain one genuine object while exercising the actual source-sized
    # allocation guard without tens of thousands of source rescans.
    retained = first.native_graph.templates[0]
    registry.update((("test-only-saturation", index), retained)
                    for index in range(budget - 1 - len(registry)))
    assert len(registry) == budget - 1
    second = prove(analysis, (selected[1],), demands)
    assert second.public()["selection_covered"], second.public()["barriers"]
    assert len(registry) == budget
    unseen_key = reachability._template_key(analysis, "work::owner", selected)
    assert unseen_key not in registry
    graph_registry = analysis._native_effect_graphs
    proof_registry = analysis._resource_reachability_proofs
    graph_keys, proof_keys = tuple(graph_registry), tuple(proof_registry)
    with pytest.raises(reachability.ResourceReachabilityBudgetExceeded, match="template registry budget exhausted"):
        prove(analysis, selected, demands)
    assert analysis._native_effect_templates is registry
    assert len(registry) == budget
    assert unseen_key not in registry
    assert analysis._native_effect_graphs is graph_registry
    assert analysis._resource_reachability_proofs is proof_registry
    assert tuple(graph_registry) == graph_keys
    assert tuple(proof_registry) == proof_keys
    assert first.validate(analysis, "work::owner", (selected[0],), demands) is first
    assert second.validate(analysis, "work::owner", (selected[1],), demands) is second


def test_instantiation_budget_includes_repeated_effect_work(tmp_path, monkeypatch):
    source = branching_source().replace("call branch(a)\ncall branch(b)", "call branch(a)\n" * 8).replace(
        "call leaf(x)\n" * 4, "call leaf(x)\n" * 8)
    _path, analysis, selected, demands = setup(tmp_path, source)
    proof = prove(analysis, selected, demands)
    counts = dict(proof._work)
    assert counts["substitution_and_obligation_steps"] > counts["substitution_steps"]
    assert counts["syntax_nodes"] < counts["substitution_steps"]
    other_dir = tmp_path / "rejected"
    other_dir.mkdir()
    _path, analysis, selected, demands = setup(other_dir, source)
    monkeypatch.setattr(reachability, "NATIVE_GROUP_SCAN_LIMIT", counts["substitution_steps"] - 1)
    with pytest.raises(reachability.ResourceReachabilityBudgetExceeded, match="instantiation"):
        prove(analysis, selected, demands)
    assert not getattr(analysis, "_resource_reachability_proofs", {})


@pytest.mark.parametrize("sealed", [False, True])
def test_prefix_typed_result_retains_original_declaring_scope(tmp_path, sealed):
    source = """module work
implicit none
real(8)::a(4),b(4),source(4),observed
contains
subroutine owner()
a(1)=value(source(2))
end subroutine
real(8) function value(x)
real(8),intent(in)::x
value=x
end function
end module
"""
    _path, analysis, selected, demands = setup(tmp_path, source, seal=sealed)
    scope = analysis.numerical_helpers["work::value"].scope
    result = scope.bindings["value"]
    assert result.declaring_scope is scope
    proof = prove(analysis, selected, demands)
    public = proof.public()
    assert public["selection_covered"], public["barriers"]
    assert [item.callee for item in proof.calls] == ["work::value"]
    writes = [item for item in proof.accesses if item.storage.binding is result and item.kind == "write"]
    assert len(writes) == 1
    assert not writes[0].external
    assert proof.validate(analysis, "work::owner", selected, demands) is proof
