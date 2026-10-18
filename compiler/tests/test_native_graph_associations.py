"""One joined native graph owns pointer confinement and optional associations."""
from copy import copy

import pytest

from compiler.frontend.native_demand import prove_native_demand
from compiler.frontend.resource_reachability import prove_resource_reachability
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

SOURCE = """module work
implicit none
type record
real(8)::flux,aux
end type
type(record),target::points(4)
real(8)::old(4),future(4)
contains
subroutine owner()
call update(points,old)
end subroutine
subroutine update(items,input)
type(record),target,intent(inout)::items(:)
real(8),intent(in)::input(:)
type(record),pointer::p
integer::i
!$omp parallel do private(i,p) schedule(guided,5)
do i=1,size(items)
p=>items(i)
call helper(p%flux,input(i),aux=p%aux)
enddo
!$omp end parallel do
end subroutine
subroutine helper(value,input,discard,aux)
real(8),intent(inout)::value
real(8),intent(in)::input
real(8),optional,intent(out)::discard,aux
value=value+twice(input)
if(present(discard)) discard=3.0_8
if(present(aux)) aux=input
end subroutine
pure real(8) function twice(value)
real(8),intent(in)::value
twice=2.0_8*value
end function
end module
"""


def setup(tmp_path, source=SOURCE, *, seal=True):
    path = tmp_path / "native.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    if seal:
        analysis.seal_session()
    routine = analysis.routines["work::owner"]
    selected = tuple(routine.execution.content)
    demands = tuple(analysis.modules["work"].bindings[name] for name in ("old", "future"))
    proof = prove_resource_reachability(analysis, routine.qualified, selected, demands)
    return path, analysis, selected, demands, proof


@pytest.mark.parametrize("seal", [False, True])
def test_confined_pointer_optional_helper_and_guided_join_form_one_native_graph(tmp_path, seal):
    _path, analysis, selected, demands, proof = setup(tmp_path, seal=seal)
    assert not proof.barriers, proof.public()["barriers"]
    paths = {item.storage.resource for item in proof.accesses}
    assert {"work::points%flux", "work::points%aux", "work::old"} <= paths
    assert not any("::p%" in name or "discard" in name for name in paths)
    effects = prove_native_demand(analysis, "work::owner", selected, demands)
    assert effects.validate(analysis, "work::owner", selected, demands) is effects
    assert effects.effects == ((demands[0], frozenset({"read"})),)
    pointer = next(item for template in proof.native_graph.templates for item in template.confined_pointers)
    assert pointer.binding.name == "p"
    assert pointer.target.binding.name == "items"
    completion, = [item for _name, _selected, item in proof._completion
                   if type(item).__name__ == "NativeGraphCompletionProof"]
    record = completion.public()
    assert record["private_pointer_descriptors"] == ["work::update::p"]
    assert record["native_only"]
    assert not record["internal_cuts_authorized"]
    assert not record["private_pointer_payload_authority"]
    assert not record["gpu_legality_established"]
    assert analysis.numerical_helpers["work::twice"].scope.bindings["twice"].declaring_scope is not None


@pytest.mark.parametrize(("before", "after", "reason"), [
    ("p=>items(i)", "p%flux=0.0_8\np=>items(i)", "pointer"),
    ("call helper(p%flux,input(i),aux=p%aux)", "call helper(p%flux,input(i),aux=p%aux)\np=>items(i)", "pointer"),
    ("enddo\n!$omp end parallel do", "enddo\n!$omp end parallel do\np%flux=0.0_8", "pointer"),
    ("private(i,p)", "private(i)", "PRIVATE"),
    ("schedule(guided,5)", "schedule(guided,i)", "SCHEDULE"),
    ("if(present(discard)) discard=3.0_8", "discard=3.0_8", "absent optional"),
    ("value=value+twice(input)", "call unknown(value)", "unresolved"),
    ("p=>items(i)", "p=>items(i)\nreturn", "nonlocal completion"),
])
def test_unknown_associations_or_completion_remain_barriers(tmp_path, before, after, reason):
    _path, _analysis, _selected, _demands, proof = setup(tmp_path, SOURCE.replace(before, after))
    assert proof.barriers
    assert reason.lower() in " ".join(item.reason for item in proof.barriers).lower()


def test_optional_absence_is_forwarded_without_fictitious_payload(tmp_path):
    source = SOURCE.replace("if(present(discard)) discard=3.0_8", "call output(discard)")
    source = source.replace("pure real(8) function twice(value)", """subroutine output(value)
real(8),optional,intent(out)::value
if(present(value)) value=3.0_8
end subroutine
pure real(8) function twice(value)""")
    _path, _analysis, _selected, _demands, proof = setup(tmp_path, source)
    assert not proof.barriers, proof.public()["barriers"]
    assert not any("discard" in access.storage.resource for access in proof.accesses)


def test_optional_allocation_dependent_presence_is_not_assumed_supplied(tmp_path):
    source = SOURCE.replace("call update(points,old)", "call output(maybe)")
    source = source.replace("real(8)::old(4),future(4)", "real(8)::old(4),future(4)\nreal(8),allocatable::maybe(:)")
    source = source.replace("subroutine update(items,input)", """subroutine output(value)
real(8),optional,intent(in)::value(:)
if(present(value)) future(1)=value(1)
end subroutine
subroutine update(items,input)""")
    _path, _analysis, _selected, _demands, proof = setup(tmp_path, source)
    assert any("descriptor-dependent presence" in item.reason for item in proof.barriers)


def test_absent_actual_cannot_be_forwarded_to_required_dummy(tmp_path):
    source = SOURCE.replace("if(present(discard)) discard=3.0_8", "call output(discard)")
    source = source.replace("pure real(8) function twice(value)", """subroutine output(value)
real(8),intent(in)::value
end subroutine
pure real(8) function twice(value)""")
    _path, _analysis, _selected, _demands, proof = setup(tmp_path, source)
    assert any("absent actual requires" in item.reason for item in proof.barriers)


def test_graph_completion_cannot_be_copied_or_borrowed_by_other_analysis(tmp_path):
    path, analysis, selected, demands, proof = setup(tmp_path)
    assert not proof.barriers, proof.public()["barriers"]
    completion, = [item for _name, _selected, item in proof._completion
                   if type(item).__name__ == "NativeGraphCompletionProof"]
    with pytest.raises(CompilationError, match="exact registered"):
        copy(completion).validate(analysis, completion.procedure, completion.selected_node_ids)
    with pytest.raises(CompilationError, match="exact registered"):
        completion.validate(SourceEffects([path]), completion.procedure, completion.selected_node_ids)
    path.write_text(SOURCE + "\n! changed source\n")
    with pytest.raises(CompilationError):
        analysis.verify_publication()
    assert proof.validate(analysis, "work::owner", selected, demands) is proof


def test_failed_nested_join_cannot_lend_outer_completion(tmp_path):
    source = SOURCE.replace("value=value+twice(input)", """!$omp parallel do schedule(dynamic)
do j=1,2
value=value+twice(input)
enddo
!$omp end parallel do""").replace("real(8),optional,intent(out)::discard,aux",
                                    "real(8),optional,intent(out)::discard,aux\ninteger::j")
    _path, analysis, _selected, _demands, proof = setup(tmp_path, source)
    assert proof.barriers
    assert not getattr(analysis, "_native_graph_completions", {})
