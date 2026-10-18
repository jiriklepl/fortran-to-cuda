"""Dormant native state retains definitions without active coherence rules."""

from copy import copy

import pytest

from compiler.frontend.call_bindings import SourceDependency
from compiler.frontend.native_preservation import prove_native_preservation
from compiler.frontend.native_sections import NativeAxis, NativeBound, NativeRectangle
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

SOURCE = """module state
implicit none
real(8),allocatable::a(:),b(:)
integer::n
contains
subroutine owner()
call correction(a,b,n)
end subroutine
subroutine correction(x,y,count)
real(8),intent(inout)::x(:),y(:)
integer,intent(inout)::count
x(1)=y(1)+x(1)
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE):
    path = tmp_path / "state.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    routine = analysis.routines["state::owner"]
    originals = tuple(routine.execution.content)
    demanded = tuple(analysis.modules["state"].bindings[name] for name in ("a", "b", "n"))
    proof = prove_native_preservation(analysis, routine.qualified, originals, demanded)
    return path, analysis, routine, originals, demanded, proof


def rectangle(binding=None, *, descriptor=False):
    if binding is None:
        upper = NativeBound("literal", value=8)
    else:
        kind = "descriptor_read" if descriptor else "scalar_read"
        upper = NativeBound("size" if descriptor else "scalar", resource=binding.root,
                            dimension=1 if descriptor else None,
                            dependencies=(SourceDependency(binding.root, kind, binding),))
    return NativeRectangle((NativeAxis(NativeBound("literal", value=1), upper),), "entry")


def test_complete_native_writes_preserve_existing_points_without_granting_new_ones(tmp_path):
    _path, _analysis, _routine, _originals, (a, b, n), proof = fixture(tmp_path)
    existing = rectangle(n)
    result = proof.transfer({a.root: (existing,), b.root: ()})
    assert proof.complete
    assert proof.reachability.obligations  # Active coherence would require all of these.
    assert result.definitions == {a.root: (existing,), b.root: ()}
    assert not result.invalidations
    assert not result.public()["new_definition_authority"]
    assert not proof.public()["execution_authorized"]


def test_out_actual_loses_coverage_without_destroying_independent_owned_storage(tmp_path):
    source = SOURCE.replace("real(8),intent(inout)::x(:),y(:)",
                            "real(8),intent(inout)::x(:)\nreal(8),intent(out)::y(:)")
    source = source.replace("x(1)=y(1)+x(1)", "y(1)=0.d0\nx(1)=x(1)+1.d0")
    _path, _analysis, _routine, _originals, (a, b, n), proof = fixture(tmp_path, source)
    box = rectangle(n)
    assert proof.transfer({a.root: (box,), b.root: (box,)}).definitions == {a.root: (box,), b.root: ()}
    assert dict(proof.invalidations) == {b.root: ("entry undefinition",)}


def test_scalar_change_invalidates_geometry_but_not_fixed_rectangles(tmp_path):
    source = SOURCE.replace("x(1)=y(1)+x(1)", "count=count+1")
    _path, _analysis, _routine, _originals, (a, _b, n), proof = fixture(tmp_path, source)
    fixed = rectangle()
    result = proof.transfer({a.root: (rectangle(n), fixed)})
    assert result.definitions == {a.root: (fixed,)}
    assert dict(result.invalidations)[n.root] == ("scalar value write",)


def test_payload_write_and_out_leave_other_resources_descriptor_geometry_current(tmp_path):
    source = SOURCE.replace("real(8),intent(inout)::x(:),y(:)",
                            "real(8),intent(inout)::x(:)\nreal(8),intent(out)::y(:)")
    source = source.replace("x(1)=y(1)+x(1)", "y(1)=0.d0\nx(1)=x(1)+1.d0")
    _path, _analysis, _routine, _originals, (a, b, _n), proof = fixture(tmp_path, source)
    box = rectangle(b, descriptor=True)
    assert proof.transfer({a.root: (box,)}).definitions == {a.root: (box,)}


@pytest.mark.parametrize("statement", ["deallocate(b)", "allocate(b(8))", "call unavailable(a)",
                                       "b=a"])
def test_descriptor_changes_and_unknown_execution_clear_all_coverage(tmp_path, statement):
    source = SOURCE.replace("call correction(a,b,n)", statement)
    _path, _analysis, _routine, _originals, (a, b, n), proof = fixture(tmp_path, source)
    result = proof.transfer({a.root: (rectangle(b, descriptor=True), rectangle(n))})
    assert not proof.complete
    assert not result.complete
    assert result.definitions == {a.root: ()}
    assert proof.reachability.barriers


def test_potential_alias_of_out_actual_cannot_preserve_other_dummy(tmp_path):
    source = SOURCE.replace("subroutine owner()", "subroutine owner(a,b,n)\n"
        "real(8),intent(inout)::a(:),b(:)\ninteger,intent(inout)::n")
    source = source.replace("real(8),intent(inout)::x(:),y(:)",
                            "real(8),intent(inout)::x(:)\nreal(8),intent(out)::y(:)")
    source = source.replace("x(1)=y(1)+x(1)", "y(1)=0.d0")
    path = tmp_path / "state.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    routine = analysis.routines["state::owner"]
    demanded = tuple(routine.scope.bindings[name] for name in ("a", "b", "n"))
    proof = prove_native_preservation(analysis, routine.qualified, tuple(routine.execution.content), demanded)
    assert proof.complete
    assert proof.transfer({demanded[0].root: (rectangle(),)}).definitions == {demanded[0].root: ()}


def test_unseparated_dummy_payload_writes_preserve_points_but_invalidate_possible_bound_alias(tmp_path):
    source = SOURCE.replace("subroutine owner()", "subroutine owner(a,b,n)\n"
        "real(8),intent(inout)::a(:),b(:)\ninteger,intent(inout)::n")
    path = tmp_path / "state.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    routine = analysis.routines["state::owner"]
    demanded = tuple(routine.scope.bindings[name] for name in ("a", "b", "n"))
    proof = prove_native_preservation(analysis, routine.qualified, tuple(routine.execution.content), demanded)
    fixed = rectangle()
    dependent = rectangle(demanded[2])
    result = proof.transfer({demanded[0].root: (fixed, dependent), demanded[1].root: (fixed,)})
    assert proof.complete
    assert result.definitions == {demanded[0].root: (fixed,), demanded[1].root: (fixed,)}
    assert dict(result.invalidations) == {demanded[2].root: ("scalar value write",)}


@pytest.mark.parametrize("mutation", ["copy", "source", "payload", "demand", "selection"])
def test_preservation_requires_registered_exact_source_authority(tmp_path, mutation):
    path, analysis, routine, originals, demanded, proof = fixture(tmp_path)
    if mutation == "copy":
        proof = copy(proof)
    elif mutation == "source":
        path.write_text(path.read_text() + "\n! changed\n")
    elif mutation == "payload":
        object.__setattr__(proof, "invalidations", ((demanded[0].root, ("invented",)),))
    elif mutation == "demand":
        demanded = (copy(demanded[0]), *demanded[1:])
    else:
        originals = (copy(originals[0]),)
    with pytest.raises(CompilationError):
        proof.validate(analysis, routine.qualified, originals, demanded)


def test_foreign_or_undeclared_bound_dependency_cannot_be_filtered_as_original(tmp_path):
    _path, _analysis, _routine, _originals, (a, _b, n), proof = fixture(tmp_path)
    with pytest.raises(CompilationError, match="exact demanded dependency"):
        proof.transfer({a.root: (rectangle(copy(n)),)})


@pytest.mark.parametrize("empty_registry", [False, True])
def test_private_analysis_cannot_share_preservation_registry(tmp_path, empty_registry):
    _path, analysis, routine, originals, demanded, proof = fixture(tmp_path)
    if empty_registry:
        analysis._native_preservation_proofs.clear()
    projected = copy(analysis)
    # A source projection creates its own registered reachability authorities;
    # the preservation registry must not borrow an inherited owner token.
    projected._resource_reachability_proofs = {}
    projected._native_effect_graphs = {}
    projected._native_effect_templates = {}
    other = prove_native_preservation(projected, routine.qualified, originals, demanded)
    assert other is not proof
    assert analysis._native_preservation_proofs is not projected._native_preservation_proofs
    assert projected._native_preservation_registry_owner is projected
    if empty_registry:
        assert not analysis._native_preservation_proofs
    else:
        assert proof.validate(analysis, routine.qualified, originals, demanded) is proof
