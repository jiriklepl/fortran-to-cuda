"""A path selector inventories original edges and grants no retention."""
from copy import copy
from dataclasses import replace

import pytest

from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.supported_paths import prove_supported_path
from compiler.ir import CompilationError

SOURCE = """module path_fixture
implicit none
contains
subroutine advance(a,n,mode,debug)
real(8),intent(inout)::a(:)
integer,intent(in)::n,mode
logical,intent(in)::debug
call produce(a,n)
if(mode==1)then
 call selected(a,n)
else
 call unavailable(a)
endif
if(debug)call native_observer(a)
call finish(a,n)
write(*,*)a(1)
end subroutine
subroutine produce(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
a(1)=2.0_8
end subroutine
subroutine selected(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
a(1)=a(1)+1.0_8
end subroutine
subroutine finish(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
a(1)=a(1)*3.0_8
end subroutine
end module
"""


def fixture(tmp_path, *, sealed=False, source=SOURCE):
    path = tmp_path / "source.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine = analysis.routines["path_fixture::advance"]
    originals = tuple(routine.execution.content[:-1])
    roots = tuple("path_fixture::" + name for name in ("produce", "selected", "finish"))
    return path, analysis, routine, originals, roots


def issue(parts):
    _path, analysis, routine, originals, roots = parts
    return prove_supported_path(analysis, routine, originals, roots)


@pytest.mark.parametrize("sealed", [False, True])
def test_whole_original_conditional_interval_preserves_unselected_unknown_edges(tmp_path, sealed):
    parts = fixture(tmp_path, sealed=sealed)
    proof = issue(parts)
    assert proof.validate() is proof
    assert issue(parts) is proof
    record = proof.public()
    assert len(record["calls"]) == 5
    assert sum(row["device_discovery_selected"] for row in record["calls"]) == 3
    assert sum(row["resolved"] is None for row in record["calls"]) == 2
    assert [row["parser_kind"] for row in record["controls"]] == ["If_Then_Stmt", "Else_Stmt", "If_Stmt"]
    assert record["source_interval"]["following_original"]["parser_kind"] == "Write_Stmt"
    for flag in ("lifetime_authorized", "entry_definition_authority", "payload_access_authorized",
                 "execution_authorized", "placement_authorized", "motion_authorized", "cross_frontier_retention_authorized"):
        assert record[flag] is False
    assert "independent retirement/noescape proof" in record["required_future_epoch_policy"]


@pytest.mark.parametrize("selection", ["gap", "copy", "prefix_only", "wrong_endpoint"])
def test_original_interval_cannot_drop_branch_tail_or_substitute_parser_objects(tmp_path, selection):
    _path, analysis, routine, originals, roots = fixture(tmp_path)
    if selection == "gap":
        originals = (originals[0], originals[-1])
    elif selection == "copy":
        originals = (copy(originals[0]), *originals[1:])
    elif selection == "prefix_only":
        originals = (originals[0],)
    else:
        originals = (*originals, routine.execution.content[-1])
    with pytest.raises(CompilationError):
        prove_supported_path(analysis, routine, originals, roots)


@pytest.mark.parametrize("mutation", ["copy", "registry", "namespace", "call_actuals", "association", "source"])
def test_registered_selection_never_retags_changed_authority(tmp_path, mutation):
    parts = fixture(tmp_path)
    path, analysis, routine, _originals, _roots = parts
    proof = issue(parts)
    if mutation == "copy":
        proof = copy(proof)
    elif mutation == "registry":
        analysis._supported_path_selections.clear()
    elif mutation == "namespace":
        routine.scope.imports["changed"] = ("other", "a")
    elif mutation == "source":
        path.write_text(path.read_text() + "! changed\n")
    else:
        edge = next(value for value in analysis._original_call_associations.values()
                    if value.procedure == "path_fixture::produce")
        if mutation == "call_actuals":
            object.__setattr__(edge, "actuals", tuple(copy(value) for value in edge.actuals))
        else:
            object.__setattr__(edge, "associations", (replace(edge.associations[0]), *edge.associations[1:]))
    with pytest.raises(CompilationError):
        proof.validate()


def test_unproved_root_loop_control_is_not_selected_by_this_initial_slice(tmp_path):
    source = SOURCE.replace("if(mode==1)then", "do while(mode==1)").replace("else\n call unavailable(a)\nendif", "enddo")
    with pytest.raises(CompilationError, match="structured control"):
        issue(fixture(tmp_path, source=source))


@pytest.mark.parametrize("kind", ["content", "foreign"])
def test_path_registry_cannot_retag_content_or_lend_original_analysis_authority(tmp_path, kind):
    parts = fixture(tmp_path)
    proof = issue(parts)
    if kind == "content":
        object.__setattr__(proof, "identity", "0"*64)
        parts[1]._supported_path_selections[proof.identity] = proof
        with pytest.raises(CompilationError):
            proof.validate()
    else:
        with pytest.raises(CompilationError):
            prove_supported_path(parts[1].private_projection(), *parts[2:])


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("mutation", ["source", "input_metadata", "call_actuals", "registry"])
def test_publication_fence_cannot_retag_stale_source_or_consumed_edges(tmp_path, monkeypatch, sealed, existing, mutation):
    from compiler.frontend import supported_paths as service

    parts = fixture(tmp_path, sealed=sealed)
    path, analysis, _routine, _originals, _roots = parts
    if existing:
        issue(parts)
    encode = service._encoded
    fired = False

    def alter(value):
        nonlocal fired
        result = encode(value)
        if value.get("kind") == "original_supported_path_selection" and not fired:
            fired = True
            if mutation == "source":
                path.write_text(path.read_text() + "! changed at publication\n")
            elif mutation == "input_metadata":
                inputs = copy(analysis.inputs)
                inputs.configuration = {"changed_at_publication": True}
                analysis.inputs = inputs
            elif mutation == "call_actuals":
                edge = next(value for value in analysis._original_call_associations.values()
                            if value.procedure == "path_fixture::produce")
                object.__setattr__(edge, "actuals", tuple(copy(value) for value in edge.actuals))
            else:
                analysis._supported_path_selections = {}
        return result

    monkeypatch.setattr(service, "_encoded", alter)
    with pytest.raises(CompilationError):
        issue(parts)
    assert fired


def test_public_json_can_never_populate_the_selection_registry(tmp_path):
    parts = fixture(tmp_path)
    _path, analysis, _routine, _originals, _roots = parts
    public = issue(parts).public()
    analysis._supported_path_selections = {public["identity"]: public}
    with pytest.raises(CompilationError, match="public metadata"):
        issue(parts)
