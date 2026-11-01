"""Only scalar-local control under a proved standard PRESENT is projected."""
import sys
from copy import copy

import pytest

from compiler.frontend.optional_scalar_control import original_optional_scalar_control
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

SOURCE = """module optional_control
implicit none
contains
subroutine advance(a,toggle)
real(8),intent(inout)::a(:)
logical,optional,intent(in)::toggle
logical::update
if(present(toggle))then
update=.not.toggle
else
update=.true.
endif
a(:)=a(:)+1.0_8
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE, *, sealed=False):
    path = tmp_path/"optional.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine = analysis.routines["optional_control::advance"]
    graph = analysis.structure(routine.qualified)
    node, = graph.source_nodes(graph.nodes[graph.root_id].children[0])
    return path, analysis, routine, node


@pytest.mark.parametrize("sealed", [False, True])
def test_both_original_scalar_arms_remain_without_array_definition_or_execution_authority(tmp_path, sealed):
    path, analysis, routine, node = fixture(tmp_path, sealed=sealed)
    source = path.read_bytes()
    proof = original_optional_scalar_control(analysis, routine.qualified, node)
    assert proof.validate(analysis, routine.qualified, node) is proof
    assert original_optional_scalar_control(analysis, routine.qualified, node) is proof
    record = proof.public()
    assert record["optional_dummy"]["resource"] == "argument::toggle"
    assert record["local_output"]["resource"] == "optional_control::advance::update"
    assert record["incoming_array_definition_changes"] == []
    assert record["scalar_accesses_require_independent_native_coherence"] is True
    for flag in ("new_definition_authority", "execution_authorized", "payload_access_authorized",
                 "descriptor_lifetime_authority", "alias_authority", "source_motion_authorized"):
        assert record[flag] is False
    assert path.read_bytes() == source


@pytest.mark.parametrize(("old", "new"), [
    ("update=.true.", "update=toggle"),
    ("update=.not.toggle", "update=any(a>0)"),
    ("update=.not.toggle", "a(1)=1.0_8"),
    ("update=.not.toggle", "call update_flag(toggle,update)"),
    ("logical::update", "logical,target::update"),
    ("logical::update", "logical,pointer::update"),
    ("logical::update", "logical::update\nexternal present"),
    ("logical::update", "logical::update\ncommon /shared/ update"),
    ("advance(a,toggle)", "advance(a,toggle,update)"),
    ("logical,optional,intent(in)::toggle", "logical,optional,target,intent(in)::toggle"),
    ("logical,optional,intent(in)::toggle", "logical,optional,intent(in)::toggle(:)"),
    ("update=.true.", "update=.true.\na(1)=2.0_8"),
    ("if(present(toggle))then", "if(.not.present(toggle))then"),
])
def test_payloads_calls_aliases_unreached_optional_reads_and_unknown_control_decline(tmp_path, old, new):
    _path, analysis, routine, node = fixture(tmp_path, SOURCE.replace(old, new))
    with pytest.raises(CompilationError):
        original_optional_scalar_control(analysis, routine.qualified, node)


@pytest.mark.parametrize("mutation", ["copy", "registry", "owner", "node"])
def test_exact_original_registry_and_node_are_required(tmp_path, mutation):
    _path, analysis, routine, node = fixture(tmp_path)
    proof = original_optional_scalar_control(analysis, routine.qualified, node)
    if mutation == "copy":
        proof = copy(proof)
    elif mutation == "registry":
        analysis._optional_scalar_controls.pop(proof.identity)
    elif mutation == "owner":
        analysis._optional_scalar_controls_owner = object()
    else:
        node = copy(node)
    with pytest.raises(CompilationError):
        proof.validate(analysis, routine.qualified, node)


@pytest.mark.parametrize("mutation", ["output", "registry", "token", "source"])
@pytest.mark.parametrize("sealed", [False, True])
def test_final_publication_cannot_replace_original_control_or_local_storage(tmp_path, monkeypatch, mutation, sealed):
    path, analysis, routine, node = fixture(tmp_path, sealed=sealed)
    proof = original_optional_scalar_control(analysis, routine.qualified, node)
    original = analysis.verify_publication
    callbacks = []

    def changed():
        original()
        if sys._getframe(1).f_code is type(proof).validate.__code__:
            callbacks.append("optional-final")
            if mutation == "output":
                object.__setattr__(routine.scope.bindings["update"], "attributes", frozenset({"target"}))
            elif mutation == "registry":
                analysis._optional_scalar_controls.pop(proof.identity)
            elif mutation == "token":
                object.__setattr__(proof, "_node", copy(node))
            else:
                path.write_text(path.read_text()+"! after final source publication\n")

    monkeypatch.setattr(analysis, "verify_publication", changed)
    with pytest.raises(CompilationError):
        proof.validate(analysis, routine.qualified, node)
    assert callbacks == ["optional-final"]
