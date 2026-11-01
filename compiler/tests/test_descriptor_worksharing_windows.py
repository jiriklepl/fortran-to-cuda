"""Complete called-source descriptor scans retain original orphan constraints."""
from copy import copy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.scopes.storage_epoch import (
    _worksharing_rows,
    require_original_worksharing_coverage,
    require_original_worksharing_edits,
    worksharing_descriptor_windows,
)
from compiler.tests.test_descriptor_worksharing import SOURCE

PARENT = "worksharing::parent"
WRAPPER = """subroutine parent(a,b,n)
integer,intent(in)::n
real(8),intent(in)::a(:,:,:)
real(8),intent(inout)::b(:,:,:)
call run(a,b,n)
end subroutine
"""


def fixture(tmp_path, source=SOURCE, sealed=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "original.f90"
    path.write_text(source.replace("end module", WRAPPER+"end module"))
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine = analysis.routines[PARENT]
    graph, selected = analysis._selected_source(PARENT, tuple(routine.execution.content))
    return path, analysis, selected, graph


def issue(parts):
    _path, analysis, selected, _graph = parts
    return prove_descriptor_window(analysis, PARENT, selected, ("argument::a", "argument::b"))


@pytest.mark.parametrize("sealed", [False, True])
def test_complete_callee_scan_has_conditional23_and_retains_structure_only(tmp_path, sealed):
    parts = fixture(tmp_path, sealed=sealed)
    proof = issue(parts)
    public = proof.public()
    assert public["schema_version"] == 23
    assert len(proof._worksharing) == 1
    requirement, = public["native_worksharing_storage_requirements"]
    assert requirement["structured_identity"] == public["source_call_closure"]["worksharing::run"]
    assert not requirement["work_completion_authority"]
    assert not public["execution_authorized"]
    assert proof.validate(parts[1], PARENT, parts[2]) is proof
    assert issue(parts) is proof
    (tmp_path / "descriptor-public.json").write_text(__import__('json').dumps(public, indent=2)+"\n")


@pytest.mark.parametrize("tail", ["call unknown_escape(b)", "allocate(b(3,3,3))"])
def test_complete_native_body_is_not_skipped_after_envelope(tmp_path, tail):
    source = SOURCE.replace("end subroutine", tail+"\nend subroutine")
    if tail.startswith("allocate"):
        source = source.replace("real(8),intent(inout)::b(:,:,:)", "real(8),allocatable,intent(inout)::b(:,:,:)")
    with pytest.raises(CompilationError):
        issue(fixture(tmp_path, source))


def owner_for(parts, proof):
    path, analysis, _selected, graph = parts
    routine = analysis.routines[PARENT]
    member = SimpleNamespace(routine=routine, units=[], device_tasks=[], source_device_tasks=[], storage_transfers=[])
    builder = SimpleNamespace(edits={}, analysis=analysis)
    return SimpleNamespace(position_epoch=SimpleNamespace(_descriptor=proof),
                           members={PARENT:member}, builder=builder), path


@pytest.mark.parametrize("boundary", ["first", "inside", "last"])
@pytest.mark.parametrize("insertion", [False, True])
def test_whole_original_native_callee_rejects_source_edits_including_empty_insertions(tmp_path, boundary, insertion):
    parts = fixture(tmp_path)
    proof = issue(parts)
    owner, path = owner_for(parts, proof)
    a,b = proof._worksharing[0].public()["callee_source"]["source_span"]
    first = {"first":a, "inside":a+2, "last":b}[boundary]
    owner.builder.edits[path] = [(first, first-1 if insertion else first, "! changed\n")]
    with pytest.raises(CompilationError):
        require_original_worksharing_edits(owner, owner.builder.edits)


def test_unrelated_sibling_edit_is_allowed_but_managed_callee_is_not(tmp_path):
    parts = fixture(tmp_path)
    proof = issue(parts)
    owner, path = owner_for(parts, proof)
    stop = proof._worksharing[0].public()["callee_source"]["source_span"][1]
    owner.builder.edits[path] = [(stop+2, stop+2, "! sibling\n")]
    require_original_worksharing_edits(owner, owner.builder.edits)
    member = copy(owner.members[PARENT])
    member.routine = parts[1].routines["worksharing::run"]
    owner.members["worksharing::run"] = member
    with pytest.raises(CompilationError):
        require_original_worksharing_edits(owner, owner.builder.edits)


@pytest.mark.parametrize("change", ["none", "missing", "clause", "span"])
def test_parent_coverage_uses_exact_original_regions_not_context_graph_policy(tmp_path, change):
    parts = fixture(tmp_path)
    parent = issue(parts)
    child = copy(parent)
    if change == "missing":
        empty = copy(parent)
        object.__setattr__(empty, "_record", {**dict(parent._record), "native_worksharing_storage_requirements": []})
        with pytest.raises(CompilationError):
            require_original_worksharing_coverage(empty, child)
        return
    if change in {"clause", "span"}:
        public = child.public()
        row = public["native_worksharing_storage_requirements"][0]
        if change == "clause":
            row["regions"][0]["collapse"] = 2
        else:
            row["callee_source"]["source_span"][0] += 1
        object.__setattr__(child, "_record", {key:value for key,value in public.items() if key != "proof_identity"})
        with pytest.raises(CompilationError):
            require_original_worksharing_coverage(parent, child)
    else:
        assert require_original_worksharing_coverage(parent, child) is child
        assert worksharing_descriptor_windows(parent) == (parent,)
        assert _worksharing_rows((parent,))[0]["worksharing_requirements"] == parent.public()["native_worksharing_storage_requirements"]


def emit_group(tmp_path, *, private=False):
    import json

    from compiler.tests.test_owner_local_call_groups import SOURCE as GROUP_SOURCE
    from compiler.tests.test_owner_local_call_groups import fixture as group_fixture
    from compiler.tests.test_owner_local_call_groups import publish

    native = """subroutine native_filter(b,n)
integer,intent(in)::n
real(8),intent(inout)::b(:)
integer::i
!$omp do
do i=1,n
b(i)=real(i,8)
enddo
!$omp end do nowait
end subroutine
"""
    source = GROUP_SOURCE.replace("subroutine second(a,b,n)", native+"subroutine second(a,b,n)")
    source = source.replace("integer::i\ndo i=1,n\nb(i)=real(i,8)",
                            "integer::i\nif(n<0)call native_filter(b,n)\ndo i=1,n\nb(i)=real(i,8)")
    if private:
        helper = """subroutine private_scratch(b,n)
integer,intent(in)::n
real(8),intent(in)::b(:)
real(8),allocatable,save::scratch(:)
scratch=b
end subroutine
"""
        source = source.replace(native, helper+native).replace(
            "if(n<0)call native_filter(b,n)",
            "if(n<0)call native_filter(b,n)\nif(n<0)call private_scratch(b,n)")
    path, builder = group_fixture(tmp_path, source)
    outputs, report = builder.run()
    publish(tmp_path, outputs, report)
    assert report["scope_count"] == 1, report["boundaries"]
    scope, = report["scopes"]
    epoch = scope["checked_execution_plan"]["storage_epoch"]
    assert epoch["version"] == 3
    assert epoch["original_worksharing_storage_obligations"]
    emitted = outputs[report["sources"][str(path)]["replacement"]]
    assert native in emitted
    assert not any("native_filter" in member["procedure"] for member in epoch["members"])
    (tmp_path/"epoch3.json").write_text(json.dumps(epoch, indent=2)+"\n")
    return path, builder, outputs, report


@pytest.fixture(scope="module")
def emitted_group(tmp_path_factory):
    return emit_group(tmp_path_factory.mktemp("worksharing_emitted"))


def test_real_owner_local_emission_preserves_the_whole_native_callee(emitted_group):
    from compiler.scopes.storage_epoch import require_original_worksharing_publication

    _path, builder, _outputs, _report = emitted_group
    assert getattr(builder, "_graph_storage_epoch", None) is None
    require_original_worksharing_publication(builder)


@pytest.mark.parametrize("change", ["edit", "tuple", "public"])
def test_final_hook_rejects_late_constraint_removal_after_capture_context_closed(emitted_group, change):
    from compiler.scopes.storage_epoch import require_original_worksharing_publication

    path, builder, _outputs, report = emitted_group
    retained = builder._original_worksharing_publications
    epoch = builder.scopes[0]["checked_execution_plan"]["storage_epoch"]
    obligations = epoch["original_worksharing_storage_obligations"]
    edits = builder.edits[path]
    original_edits = list(edits)
    requirement = obligations[0]["worksharing_requirements"][0]
    try:
        if change == "edit":
            first = requirement["callee_source"]["source_span"][0]
            edits.append((first, first-1, "! prohibited insertion\n"))
        elif change == "tuple":
            builder._original_worksharing_publications = ()
        else:
            del epoch["original_worksharing_storage_obligations"]
        with pytest.raises(CompilationError):
            require_original_worksharing_publication(builder)
    finally:
        builder._original_worksharing_publications = retained
        epoch["original_worksharing_storage_obligations"] = obligations
        edits[:] = original_edits


@pytest.mark.parametrize("change", ["edit", "tuple", "public", "token", "namespace"])
def test_final_live_callback_cannot_drop_or_change_retained_constraint(emitted_group, monkeypatch, change):
    from compiler.scopes import storage_epoch as service

    path, builder, _outputs, _report = emitted_group
    retained = builder._original_worksharing_publications
    owner = retained[0]._owner
    window = owner.position_epoch._descriptor
    tokens = window._worksharing
    routine = tokens[0]._routine
    imports = routine.scope.imports
    epoch = builder.scopes[0]["checked_execution_plan"]["storage_epoch"]
    obligations = epoch["original_worksharing_storage_obligations"]
    edits = builder.edits[path]
    original_edits = list(edits)
    original_inventory = service._worksharing_inventory
    original_live = type(builder.analysis.inputs).verify_live
    final_ready = False

    def inventory(current):
        rows, finish, checkpoint = original_inventory(current)

        def completed():
            nonlocal final_ready
            finish()
            final_ready = True

        return rows, completed, checkpoint

    def late(inputs, *args, **kwargs):
        result = original_live(inputs, *args, **kwargs)
        if final_ready and inputs is builder.analysis.inputs:
            if change == "edit":
                first = obligations[0]["worksharing_requirements"][0]["callee_source"]["source_span"][1]
                edits.append((first, first-1, "! prohibited insertion\n"))
            elif change == "tuple":
                builder._original_worksharing_publications = ()
            elif change == "public":
                del epoch["original_worksharing_storage_obligations"]
            elif change == "token":
                object.__setattr__(window, "_worksharing", ())
            else:
                object.__setattr__(routine.scope, "imports", {"foreign": ("unknown", "foreign")})
        return result

    monkeypatch.setattr(service, "_worksharing_inventory", inventory)
    monkeypatch.setattr(type(builder.analysis.inputs), "verify_live", late)
    try:
        with pytest.raises(CompilationError):
            service.require_original_worksharing_publication(builder)
    finally:
        builder._original_worksharing_publications = retained
        epoch["original_worksharing_storage_obligations"] = obligations
        edits[:] = original_edits
        object.__setattr__(window, "_worksharing", tokens)
        object.__setattr__(routine.scope, "imports", imports)


def test_scope_rollback_restores_exact_retained_tuple(emitted_group):
    from compiler.scopes.storage_epoch import require_original_worksharing_publication

    _path, builder, _outputs, _report = emitted_group
    retained = builder._original_worksharing_publications
    checkpoint = builder.scope_checkpoint()
    builder._original_worksharing_publications = (*retained, retained[0])
    builder.restore_scope_checkpoint(checkpoint)
    assert builder._original_worksharing_publications is retained
    require_original_worksharing_publication(builder)


def test_real_orphan_epoch_retains_private21_exclusions(tmp_path):
    from compiler.scopes.storage_epoch import require_original_worksharing_publication

    _path, builder, _outputs, report = emit_group(tmp_path, private=True)
    epoch = report["scopes"][0]["checked_execution_plan"]["storage_epoch"]
    assert epoch["version"] == 3
    rows = epoch["private_allocation_obligations"]
    assert any(item["resource"] == "local_groups::private_scratch::scratch"
               for row in rows for item in row["unregistered_private_allocations"])
    require_original_worksharing_publication(builder)


@pytest.mark.native
def test_real_orphan_whole_generated_module_compiles(emitted_group):
    import shutil
    import subprocess

    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    assert fortran is not None
    path, _builder, _outputs, report = emitted_group
    directory = path.parent/"generated"
    sources = [item["path"] for role in ("common_runtime", "shared_entry", "original_source")
               for item in report["build_sources"] if item["role"] == role and item["language"] == "fortran"]
    for index, name in enumerate(sources):
        result = subprocess.run([fortran, "-fopenmp", "-ffp-contract=off", "-J", str(directory),
            "-I", str(directory), "-c", str(directory/name), "-o", str(directory/f"orphan-{index}.o")],
            capture_output=True, text=True, timeout=30)
        (directory/f"orphan-{index}.stderr").write_text(result.stderr)
        assert result.returncode == 0, result.stderr


@dataclass(frozen=True)
class ConsumedNativePayload:
    _descriptor_window: object


def test_reached_consumer_without_complete_parent_epoch_fails_closed(tmp_path):
    from compiler.scopes.source import ScopeBuilder

    parts = fixture(tmp_path)
    proof = issue(parts)
    consumer = SimpleNamespace(analysis=parts[1])
    with pytest.raises(CompilationError, match="complete graph storage epoch"):
        ScopeBuilder._check_private_descriptor_window(consumer, proof)


@pytest.mark.parametrize("route", ["handoff", "returned"])
def test_consumed_native_or_returned_proof_cannot_hide_orphan_requirement(tmp_path, route):
    from compiler.scopes.storage_epoch import _worksharing_inventory

    parts = fixture(tmp_path)
    proof = issue(parts)
    owner, _path = owner_for(parts, proof)
    owner.position_epoch = None
    payload = ConsumedNativePayload(proof)
    handoffs = (SimpleNamespace(proof=payload),) if route == "handoff" else ()
    owner.execution_plan = SimpleNamespace(_proof_handoffs=handoffs)
    owner.return_contract = payload if route == "returned" else None
    with pytest.raises(CompilationError, match="complete parent storage epoch"):
        _worksharing_inventory(owner)


@pytest.mark.parametrize("sealed", [False, True])
def test_native_private_context_has_distinct_graph_and_parent_source_coverage(tmp_path, sealed):
    from compiler.scopes.segments import _native_analysis
    from compiler.scopes.storage_epoch import _worksharing_inventory

    parts = fixture(tmp_path, sealed=sealed)
    parent = issue(parts)
    native = _native_analysis(parts[1])
    native._native_metadata = True
    routine = native.routines[PARENT]
    _graph, selection = native._selected_source(PARENT, tuple(routine.execution.content))
    child = prove_descriptor_window(native, PARENT, selection, ("argument::a", "argument::b"))
    assert child.public()["structured_identity"] != parent.public()["structured_identity"]
    assert require_original_worksharing_coverage(parent, child) is child
    owner, _path = owner_for(parts, parent)
    retained = (SimpleNamespace(proof=ConsumedNativePayload(child)),)
    owner.execution_plan = SimpleNamespace(_proof_handoffs=retained)
    rows, finish, checkpoint = _worksharing_inventory(owner)
    assert rows == _worksharing_rows((parent,))
    finish()
    owner.execution_plan._proof_handoffs = tuple(list(retained))
    with pytest.raises(CompilationError, match="attachment changed"):
        checkpoint()


def test_exploratory_registry_proof_is_not_a_consumed_epoch_obligation(tmp_path):
    from compiler.scopes.storage_epoch import _worksharing_inventory

    parts = fixture(tmp_path)
    proof = issue(parts)
    owner, _path = owner_for(parts, proof)
    owner.position_epoch = None
    owner.execution_plan = SimpleNamespace(_proof_handoffs=())
    rows, finish, checkpoint = _worksharing_inventory(owner)
    assert rows == ()
    finish()
    checkpoint()
