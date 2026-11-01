"""Retired same-family native calls preserve existing admission, never payload credit."""

from copy import deepcopy
from dataclasses import replace

import pytest

from compiler.frontend.resource_reachability import prove_native_effect_graph
from compiler.ir import CompilationError
from compiler.tests.test_retired_host_discovery import SOURCE, issue


def one(tmp_path, source=SOURCE):
    path, analysis, selection, _graph = issue(tmp_path, source)
    selection = (selection[0],)
    graph = prove_native_effect_graph(analysis, "work::owner", selection, project_payload=False)
    return path, analysis, selection, graph, graph.same_family_call_view()


def test_scalar_inherited_fields_callbacks_and_lexical_aliases_preserve_same_family(tmp_path):
    _path, analysis, selection, graph, view = one(tmp_path)
    assert view.validate(analysis, "work::owner", selection) is view
    value = view.public()
    assert value["available"], value["blockers"]
    assert value["no_allocation_imports"]
    assert value["no_descriptor_changes"]
    assert value["no_persistent_publication"]
    assert value["entry_rely"] == "same conditional allocation-family access exclusion as first checked activation"
    for field in [
        "initial_quiescence_authorized",
        "stable_storage_authorized",
        "boundary_admission_authorized",
        "payload_access_authorized",
        "alias_authority",
        "entry_definition_authority",
        "motion_authority",
        "cross_cut_retention_authorized",
        "termination_guaranteed",
    ]:
        assert value[field] is False
    with pytest.raises(CompilationError, match="control-only graph role"):
        graph.validate(analysis, "work::owner", selection)


@pytest.mark.parametrize(
    "operation",
    [
        "allocate(scratch(4))",
        "deallocate(scratch)",
        "scratch=right(:,1,1)",
        "call move_alloc(scratch,imported)",
        "saved=>right(:,1,1)",
        "nullify(saved)",
        "signal=1",
        "flags(1)=1",
        "call signal_flag(flags)",
        "src(1)%flux=2",
        "xi=3",
        "call unknown(src)",
        "!$omp task\nsignal=1\n!$omp end task",
    ],
)
def test_descriptor_import_publication_metadata_write_or_unjoined_work_is_never_preservation(tmp_path, operation):
    # All arms are scanned, including unreachable text. The whole old allocation
    # boundary API may separately prove supported mutations; this view may not.
    declaration = "real(8),allocatable::scratch(:),imported(:)\nreal(8),pointer::saved(:)\ninteger::signal,flags(4)\n"
    source = SOURCE.replace("real(8)::left", declaration + "real(8)::left")
    source = source.replace(
        "end module",
        "subroutine signal_flag(flag)\ninteger,intent(out)::flag(:)\nflag(1)=1\nend subroutine\nend module",
    )
    source = source.replace("x2(xi,yj,zk)=", operation + "\nx2(xi,yj,zk)=")
    _path, _analysis, _selection, _graph, view = one(tmp_path, source)
    assert not view.public()["available"]


def test_primitive_allocatable_out_entry_cannot_preserve_allocation_identity(tmp_path):
    source = SOURCE.replace("intent(inout)::x2(:,:,:)", "allocatable,intent(out)::x2(:,:,:)")
    _path, _analysis, _selection, _graph, view = one(tmp_path, source)
    assert not view.public()["available"]


def test_projection_requires_exact_full_call_and_structural_receipt_integrity(tmp_path):
    _path, analysis, selection, graph, view = one(tmp_path)
    with pytest.raises(CompilationError):
        view.validate(analysis, "work::owner", ())
    template = next(t for t in graph.templates if t.procedure == "work::apply")
    object.__setattr__(template, "_family_identity", "0" * 64)
    with pytest.raises(CompilationError, match="structural preservation"):
        view.validate(analysis, "work::owner", selection)


WORKER = """subroutine native_worker(a,b,n,fun)
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
"""


def source_fixture():
    from compiler.tests.test_hierarchical_child_regions import SOURCE as hierarchy

    return (
        hierarchy.replace(
            "real(8)::observed",
            """real(8)::observed
type base
integer::xi
end type
type,extends(base)::record
real(8)::flux
end type
type(record),allocatable::records(:)""",
        )
        .replace("if(change)call relocate(left,right)", "call native_worker(left,right,n,scale)")
        .replace("end module", WORKER + "end module")
    )


def emit(tmp_path, monkeypatch=None):
    from compiler.scopes.conditional_return import SameFamilyNativeCallLifecycle
    from compiler.scopes.lexical import LexicalBoundaryPatch
    from compiler.tests.test_hierarchical_child_regions import fixture

    if monkeypatch is not None:
        from compiler.scopes import conditional_return

        original = conditional_return.prove_storage_mutation_effects

        def checked(analysis, procedure, **kwargs):
            selected = kwargs.get("selection") or ()
            assert not any("CALL native_worker".lower() in str(node).lower() for node in selected)
            return original(analysis, procedure, **kwargs)

        monkeypatch.setattr(conditional_return, "prove_storage_mutation_effects", checked)
    path, builder = fixture(tmp_path, source_fixture())
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt", report.get("boundaries"))
    owner = builder._conditional_return_draft.owner
    found = [
        (member, patch)
        for member in owner.members.values()
        for _first, _last, patch in member.patches
        if isinstance(patch, LexicalBoundaryPatch) and type(patch.lifecycle) is SameFamilyNativeCallLifecycle
    ]
    assert found
    return path, builder, found, outputs, report


def test_full_original_call_retires_once_and_uses_fresh_existing_admission(tmp_path, monkeypatch):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    path, builder, found, outputs, report = emit(tmp_path, monkeypatch)
    member, patch = next(
        (member, patch) for member, patch in found if "native_worker" in str(patch.originals[0]).lower()
    )
    assert patch.completion is None
    assert patch.validate(member, require_source=True) is patch
    value = patch.lifecycle.public()
    assert value["capture_facts_identity"] == report["scopes"][0]["checked_execution_plan"]["capture_facts_identity"]
    generated = outputs[report["sources"][str(path)]["replacement"]]
    assert generated.count("call native_worker(left,right,n,scale)") == 1
    assert WORKER in generated
    rendered = patch.render(member)
    assert rendered.index("fort_scope_resident_close_v1") < rendered.index("call native_worker")
    assert rendered.index(" = 0") < rendered.index("call native_worker")
    assert patch.original_source == value["original_source"]
    returned = report["scopes"][0]["checked_execution_plan"]["storage_epoch"]["return_contract"]
    assert returned["version"] == 3
    assert returned["native_call_preservation_policy_version"] == 2
    assert not returned["entry_definition_authority"]
    assert not returned["cross_cut_replica_retention_authorized"]
    for _root, (binding, _slot) in member.owner.resources.items():
        assert builder.capture(binding)["initialized"] == "none"
    assert "fort_scope_buffer_valid_v1" in generated
    validate_checked_execution_plans(report)
    assert builder.finish()


@pytest.mark.parametrize("mutation", ["facts", "entry", "foreign", "role", "source_selection"])
def test_consumer_cannot_change_capture_window_entry_source_or_completion_role(tmp_path, mutation):
    _path, builder, found, _outputs, _report = emit(tmp_path)
    member, patch = found[0]
    if mutation == "facts":
        next(iter(builder.facts["captures"].values()))["initialized"] = "none"
    elif mutation == "entry":
        object.__setattr__(patch.lifecycle, "_entry", "{}")
    elif mutation == "foreign":
        other = tmp_path / "other"
        other.mkdir()
        foreign = emit(other)[2][0][1].lifecycle
        patch = replace(patch, lifecycle=foreign)
    elif mutation == "role":
        patch = replace(patch, lifecycle=patch.lifecycle._view)
    else:
        patch = replace(patch, originals=())
    with pytest.raises(CompilationError):
        patch.validate(member, require_source=True)


@pytest.mark.parametrize(
    "mutation",
    [
        "facts",
        "entry",
        "role",
        "edit",
        "call",
        "version",
        "family",
        "false_guard",
        "inverted_guard",
        "no_fatal",
        "call_spelling",
    ],
)
def test_reader_binds_new_distinct_route_to_admission_and_once_only_emission(tmp_path, mutation):
    from elmm_pipeline.checked_execution_plan import _digest, validate_checked_execution_plans

    _path, _builder, _found, _outputs, report = emit(tmp_path)
    bad = deepcopy(report)
    scope = bad["scopes"][0]
    plan = scope["checked_execution_plan"]
    returned = plan["storage_epoch"]["return_contract"]
    route = next(row for m in returned["members"] for row in m["routes"] if "call_preservation" in row)
    value = route["call_preservation"]
    if mutation == "facts":
        value["capture_facts_identity"] = "0" * 64
    elif mutation == "family":
        value["writable_actual_families"] = ["unmanaged::signal"]
    elif mutation == "entry":
        value["entry_exit_identity"] = "0" * 64
    elif mutation == "role":
        value["payload_access_authorized"] = True
    elif mutation == "version":
        returned["version"] = 2
    elif mutation == "call_spelling":
        value["original_call"] = "CALL different(left,right,n,scale)"
    else:
        edit = next(e for e in bad["source_edits"] if [e["first_line"], e["last_line"]] == value["source_span"])
        if mutation == "edit":
            edit["replacement"] = edit["replacement"].replace(
                "fort_close_status = fort_scope_resident_close_v1(", "fort_close_status = fort_scope_close(", 1
            )
        elif mutation == "call":
            edit["replacement"] += value["original_source"]
        elif mutation == "false_guard":
            edit["replacement"] = (
                "if (.false.) then\n"
                + edit["replacement"][: -len(value["original_source"])]
                + "endif\n"
                + value["original_source"]
            )
        elif mutation == "inverted_guard":
            edit["replacement"] = edit["replacement"].replace(" /= 0) then", " == 0) then", 1)
        elif mutation == "no_fatal":
            edit["replacement"] = edit["replacement"].replace(
                "if (fort_close_status /= FORT_SCOPE_OK) error stop 'lexical owner publication failed'\n", ""
            )
    value["identity"] = _digest({k: v for k, v in value.items() if k != "identity"})
    returned["identity"] = _digest({k: v for k, v in returned.items() if k != "identity"})
    with pytest.raises(ValueError, match="Compiler checked execution plan"):
        validate_checked_execution_plans(bad)


def test_provider_retained_domain_cannot_use_full_retirement_route(tmp_path):
    from compiler.scopes.conditional_return import _preserved_native_call

    _path, builder, found, _outputs, _report = emit(tmp_path)
    member, patch = found[0]
    builder.provider_only_lifetime = object()
    assert _preserved_native_call(member, patch.originals) is None
    with pytest.raises(CompilationError, match="admission/capture/entry"):
        patch.lifecycle.validate()


@pytest.mark.parametrize("operation", ["call absent(a)", "!$omp task\na(1)=1\n!$omp end task"])
def test_unknown_or_unjoined_native_call_cannot_reenter_conditional_owner(tmp_path, operation):
    from compiler.tests.test_hierarchical_child_regions import fixture, owner_at

    source = source_fixture().replace("if(allocated(records))then", operation + "\nif(allocated(records))then")
    _path, builder = fixture(tmp_path, source)
    owner = owner_at(builder)
    with pytest.raises(CompilationError, match="cannot reenter after unproved lifetime/escape"):
        owner.run(builder.inline.prepare(owner.closed_selection))


def test_outer_writable_actual_requires_existing_capture_and_managed_family(tmp_path):
    from compiler.tests.test_hierarchical_child_regions import fixture, owner_at

    source = source_fixture().replace("real(8)::observed", "real(8)::observed\nreal(8)::unmanaged(4)")
    source = source.replace("call native_worker(left,right,n,scale)", "call native_worker(unmanaged,right,n,scale)")
    _path, builder = fixture(tmp_path, source)
    owner = owner_at(builder)
    with pytest.raises(CompilationError):
        owner.run(builder.inline.prepare(owner.closed_selection))


def test_mutating_callback_cannot_write_through_readonly_metadata_alias(tmp_path):
    source = SOURCE.replace("integer,intent(in)::i,j,k", "integer,intent(inout)::i,j,k")
    _path, _analysis, _selection, _graph, view = one(tmp_path, source)
    assert not view.public()["available"]
