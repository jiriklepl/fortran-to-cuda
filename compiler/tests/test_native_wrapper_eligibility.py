"""Source-owned native-only child routes decline before mutable generation."""

from dataclasses import replace

import pytest
from fparser.two.utils import walk

from compiler.frontend.immutable_graph import original_type
from compiler.ir import CompilationError
from compiler.scopes.conditional_return import SameFamilyNativeCallLifecycle
from compiler.scopes.hierarchical_regions import HierarchicalRegionOwner
from compiler.scopes.lexical import LexicalBoundaryPatch
from compiler.scopes.native_guard import native_only_wrapper
from compiler.tests.test_hierarchical_child_regions import fixture
from compiler.tests.test_same_family_native_call import WORKER, source_fixture

CALL = "call native_wrapper(left,right,n)"
WRAPPER = """subroutine native_wrapper(a,b,n)
integer,intent(in)::n
real(8),intent(inout)::a(:),b(:)
call apply(a,b,n,scale)
contains
subroutine apply(a,b,n,fun)
integer,intent(in)::n
real(8),intent(inout)::a(:),b(:)
procedure(scale)::fun
integer::i
if(allocated(records))then
do i=1,size(records)
associate(xi=>records(i)%xi)
a(xi)=a(xi)+records(i)%flux*fun(xi)*b(xi)
end associate
enddo
endif
end subroutine
pure real(8) function scale(i)
integer,intent(in)::i
scale=1.0_8
end function
end subroutine
"""


def source(wrapper=WRAPPER):
    return (
        source_fixture()
        .replace(WORKER, wrapper)
        .replace("call native_worker(left,right,n,scale)", CALL)
        .replace("call middle(n,change)", CALL + "\ncall consumer(left,right,n)")
    )


def setup(tmp_path, text=None):
    path, builder = fixture(tmp_path, source() if text is None else text)
    calls = [node for node in walk(builder.entry.execution) if type(node).__name__ == "Call_Stmt"]
    assert [str(node.items[0]).lower() for node in calls] == ["producer", "native_wrapper", "consumer", "relocate"]
    graph, selected = builder.analysis._selected_source(builder.entry.qualified, (calls[-1],))
    owner = HierarchicalRegionOwner(builder, tuple(calls[:-1]), selected[0])
    assert owner.graph is graph
    routine = builder.analysis.routines["hierarchical::native_wrapper"]
    return path, builder, owner, calls[1], routine


@pytest.mark.parametrize("policy", ["sections", "auto"])
def test_original_wrapper_with_exact_callback_is_source_native_only_without_any_generation(tmp_path, policy):
    _path, builder, owner, call, routine = setup(tmp_path)
    builder.config = replace(builder.config, policy=policy)
    original_members, original_resources = dict(owner.members), dict(owner.resources)
    assert native_only_wrapper(owner, call, routine)
    assert owner.members == original_members
    assert owner.resources == original_resources
    assert not owner.patches
    assert not builder.outputs
    assert not builder.resolved_calls
    assert builder.variants.public()["generated_count"] == 0
    graph = next(
        graph
        for graph in builder.analysis._native_effect_graphs.values()
        if not graph._payload_requested and graph._selection == (call,)
    )
    assert {instance.procedure for instance in graph.host_instances} == {
        "hierarchical::advance",
        "hierarchical::native_wrapper",
        "hierarchical::native_wrapper::apply",
        "hierarchical::native_wrapper::scale",
    }
    assert graph.same_family_call_view().public()["available"]


@pytest.mark.parametrize("policy", ["sections", "auto"])
def test_full_call_boundary_never_constructs_child_or_old_payload_retirement(tmp_path, monkeypatch, policy):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    from compiler.scopes import conditional_return, module_owner

    text = source().replace(
        CALL + "\ncall consumer(left,right,n)\ncall relocate(left,right)\nenddo",
        "call middle(n,change)\ncall relocate(left,right)\nenddo",
    )
    path, builder = fixture(tmp_path, text)
    builder.config = replace(builder.config, policy=policy)
    if policy == "auto":
        from compiler.tests.test_scoped_planning_entries import calibration
        # Exercise AUTO discovery; absent costs intentionally bypass all owners.
        builder.config = replace(builder.config, profile=calibration())
    original = module_owner.ModuleOwner.__init__

    def constructor(self, parent, routine, *args, **kwargs):
        assert routine.qualified not in {"hierarchical::native_wrapper", "hierarchical::native_wrapper::apply"}
        return original(self, parent, routine, *args, **kwargs)

    storage = conditional_return.prove_storage_mutation_effects

    def no_payload(analysis, procedure, **kwargs):
        assert not any(CALL in str(node).lower() for node in kwargs.get("selection", ()))
        return storage(analysis, procedure, **kwargs)

    monkeypatch.setattr(module_owner.ModuleOwner, "__init__", constructor)
    monkeypatch.setattr(conditional_return, "prove_storage_mutation_effects", no_payload)
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    owner = builder._conditional_return_draft.owner
    assert "hierarchical::native_wrapper" not in owner.members
    assert "hierarchical::native_wrapper::apply" not in owner.members
    patches = [
        patch
        for member in owner.members.values()
        for _first, _last, patch in member.patches
        if isinstance(patch, LexicalBoundaryPatch) and CALL in patch.original_source.lower()
    ]
    assert len(patches) == 1
    (patch,) = patches
    assert type(patch.lifecycle) is SameFamilyNativeCallLifecycle
    assert patch.original_source.lower().count(CALL) == 1
    assert "native CALL requires complete synchronous original effects" in patch.reason
    assert "procedure-dummy association requires exact callback authority" in patch.reason
    generated = outputs[report["sources"][str(path)]["replacement"]]
    assert WRAPPER in generated
    assert generated.lower().count(CALL) == text.lower().count(CALL)
    validate_checked_execution_plans(report, capture_facts=builder.facts)
    assert builder.finish()


@pytest.mark.parametrize(
    "body",
    [
        "if(n>0)call apply(a,b,n,scale)",
        "if(n>0)then\ncall apply(a,b,n,scale)\nelse\nreturn\nendif",
        "if(n>0)then\ncall apply(a,b,n,scale)\nelseif(n<0)then\ncontinue\nendif",
    ],
)
def test_all_original_harmless_control_arms_are_classified_without_evaluation(tmp_path, body):
    text = source(WRAPPER.replace("call apply(a,b,n,scale)", body))
    _path, _builder, owner, call, routine = setup(tmp_path, text)
    assert native_only_wrapper(owner, call, routine)


@pytest.mark.parametrize(
    "body",
    [
        "integer::i\ndo i=1,n\na(i)=b(i)+1\nenddo\ncall apply(a,b,n,scale)",
        "a=b\ncall apply(a,b,n,scale)",
        "!$omp parallel\ncall apply(a,b,n,scale)\n!$omp end parallel",
        "call ordinary(a,b,n)\ncall apply(a,b,n,scale)",
    ],
)
def test_local_work_or_other_checked_child_never_becomes_zero_work(tmp_path, body):
    text = source(WRAPPER.replace("call apply(a,b,n,scale)", body))
    if "call ordinary" in body:
        text = text.replace(
            "end module",
            "subroutine ordinary(a,b,n)\ninteger,intent(in)::n\nreal(8),intent(inout)::a(:),b(:)\ninteger::i\ndo i=1,n\na(i)=b(i)+1\nenddo\nend subroutine\nend module",
        )
    _path, _builder, owner, call, routine = setup(tmp_path, text)
    assert not native_only_wrapper(owner, call, routine)


def test_no_formal_internal_checked_child_keeps_ordinary_child_preference(tmp_path):
    wrapper = WRAPPER.replace("call apply(a,b,n,scale)", "call child").replace(
        "contains\nsubroutine apply",
        "contains\nsubroutine child\ninteger::i\ndo i=1,n\na(i)=b(i)+1\nenddo\nend subroutine\nsubroutine apply",
    )
    _path, _builder, owner, call, routine = setup(tmp_path, source(wrapper))
    assert not native_only_wrapper(owner, call, routine)


def test_call_site_external_provider_precedence_is_not_bypassed(tmp_path, monkeypatch):
    from compiler.frontend import resource_reachability

    _path, builder, owner, call, routine = setup(tmp_path)
    internal_call = next(node for node in walk(routine.execution) if type(node).__name__ == "Call_Stmt")
    builder.analysis.contracts = {
        routine.qualified: {"device_tasks": [{"source": {"span": list(internal_call.item.span)}}]}
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("provider precedence must be handled before the native projection")

    monkeypatch.setattr(resource_reachability, "prove_native_effect_graph", forbidden)
    assert not native_only_wrapper(owner, call, routine)


@pytest.mark.parametrize("attribute", ["_reached_window_binding", "_mapped_reached_child"])
def test_registered_mapped_routes_are_not_replaced(tmp_path, attribute):
    _path, builder, owner, call, routine = setup(tmp_path)
    setattr(builder, attribute, object())
    assert not native_only_wrapper(owner, call, routine)


@pytest.mark.parametrize("mode", ["interval", "provider", "version"])
def test_other_policy_or_owner_roles_are_conservatively_unknown(tmp_path, monkeypatch, mode):
    from compiler.scopes import native_guard

    _path, builder, owner, call, routine = setup(tmp_path)
    if mode == "interval":
        owner.interval_draft = object()
    elif mode == "provider":
        builder.provider_only_lifetime = object()
    else:
        monkeypatch.setattr(native_guard, "GUARDED_NATIVE_ROUTING_VERSION", 3)
    assert not native_only_wrapper(owner, call, routine)


@pytest.mark.parametrize(
    "body",
    [
        "!$omp task\na(1)=1\n!$omp end task",
        "call unknown(a)",
        "allocate(scratch(n))",
    ],
)
@pytest.mark.parametrize("policy", ["sections", "auto"])
def test_complete_descendant_graph_still_rejects_unjoined_unknown_or_import_work(tmp_path, body, policy):
    wrapper = WRAPPER.replace(
        "integer::i\nif(allocated(records))",
        "integer::i\nreal(8),allocatable::scratch(:)\n" + body + "\nif(allocated(records))",
    )
    _path, builder, owner, call, routine = setup(tmp_path, source(wrapper))
    builder.config = replace(builder.config, policy=policy)
    assert not native_only_wrapper(owner, call, routine)


def test_corrupted_executable_graph_remains_an_integrity_error(tmp_path):
    _path, builder, owner, call, routine = setup(tmp_path)
    assert native_only_wrapper(owner, call, routine)
    graph = next(
        graph
        for graph in builder.analysis._native_effect_graphs.values()
        if not graph._payload_requested and graph._selection == (call,)
    )
    object.__setattr__(graph.host_instances[1], "procedure", "hierarchical::other")
    with pytest.raises(CompilationError):
        native_only_wrapper(owner, call, routine)


def test_copied_or_foreign_routine_never_acquires_a_native_only_route(tmp_path):
    from copy import copy

    _path, builder, owner, call, routine = setup(tmp_path)
    with pytest.raises(CompilationError, match="exact original routine"):
        native_only_wrapper(owner, call, copy(routine))
    assert not native_only_wrapper(owner, call, builder.analysis.routines["hierarchical::consumer"])


def test_ordinary_module_numerical_child_is_still_retained_and_preferred(tmp_path):
    from compiler.scopes import module_owner

    ordinary = "subroutine ordinary(a,b,n)\ninteger,intent(in)::n\nreal(8),intent(inout)::a(:),b(:)\ninteger::i\ndo i=1,n\na(i)=b(i)+1\nenddo\nend subroutine\n"
    text = source(WRAPPER.replace("call apply(a,b,n,scale)", "call ordinary(a,b,n)"))
    text = text.replace("end module", ordinary + "end module")
    _path, builder = fixture(tmp_path, text)
    _outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    owner = builder._conditional_return_draft.owner
    assert original_type(owner.members["hierarchical::native_wrapper"]) is module_owner.ModuleOwner
    child = owner.members["hierarchical::ordinary"]
    assert any(unit.leaves for unit in child.units)
    assert not any(
        "native-guard routing policy" in patch.reason
        for member in owner.members.values()
        for _first, _last, patch in member.patches
        if isinstance(patch, LexicalBoundaryPatch)
    )
