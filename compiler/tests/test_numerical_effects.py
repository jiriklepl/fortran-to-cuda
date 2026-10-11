"""Prepared physical effects retain source coordinates and guarded definitions."""

import json
from copy import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.native_prefix import _fork
from compiler.frontend.native_sections import NativeAxis, NativeRectangle
from compiler.frontend.numerical_effects import (
    NumericalEffectProof,
    NumericalEffectUnsupported,
    _Mapper,
    prove_numerical_effects,
)
from compiler.frontend.numerical_preparation import NumericalPreparationUnsupported, prepare_original_region
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import Binary, CompilationError, Literal, ScalarType
from compiler.offload.config import OffloadConfig

ENTRY = "retained_cells::sweep"
SOURCE = """module retained_cells
implicit none
contains
subroutine sweep(a,b,c,n)
real(8),intent(in)::a(-4:)
real(8),intent(inout)::b(-4:),c(-4:)
integer,intent(in)::n
integer::i
!$omp parallel do private(i)
do i=-3,n
b(i)=a(i)+real(i,8)
enddo
!$omp end parallel do
end subroutine
end module
"""


def fixture(directory, source=SOURCE):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "cells.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    routine = analysis.routines[ENTRY]
    groups = [group for group in grouped_original_sequence(routine.execution.content, {})
              if isinstance(group, NativeGroupCandidate)]
    assert len(groups) == 1
    selection = groups[0].original_nodes
    prepared = prepare_original_region(analysis, ENTRY, selection,
        options=CompilerOptions(gpu_policy="sections", memory_model="scoped", scope_execution="reached"),
        config=OffloadConfig(policy="sections", scope_execution="reached"), environment_root=ENTRY)
    return path, analysis, selection, prepared


def resource(proof, name):
    return next(item for item in proof.sections.resources if item.resource == "argument::" + name)


def test_original_negative_coordinates_and_iteration_domain_are_retained(tmp_path):
    path, analysis, selection, prepared = fixture(tmp_path)
    before = path.read_bytes()
    proof = prove_numerical_effects(analysis, ENTRY, selection, prepared)
    assert isinstance(proof, NumericalEffectProof)
    assert proof.prepared is prepared
    output = resource(proof, "b")
    box, = output.overwrites
    axis, = box.axes
    assert axis.lower.kind == "literal"
    assert axis.lower.value == -3
    assert axis.upper.kind == "scalar"
    assert axis.upper.resource == "argument::n"
    iteration, = box.iterations
    assert iteration.lower.value == -3
    assert iteration.upper.resource == "argument::n"
    assert iteration.step == 1
    assert output.lower_bounds == (-4,)
    assert {item.resource for item in output.dependencies} == {"argument::n"}
    result = proof.transfer()
    assert not result.reads_covered
    assert {item.resource for item in result.uncovered_reads} == {"argument::a"}
    assert proof.transfer({"argument::a": resource(proof, "a").reads}).reads_covered
    assert result.definitions["argument::b"] == output.overwrites
    assert path.read_bytes() == before
    public = proof.public()
    assert json.loads(json.dumps(public)) == public
    assert public["source_demands"] == [prepared.identity]
    assert public["execution_authorized"] is False
    assert public["input_definitions_authorized"] is False
    assert public["descriptor_lifetime_authorized"] is False
    assert public["unconditional_return_definitions_authorized"] is False


def test_conditional_preservation_uploads_are_not_semantic_reads_or_definitions(tmp_path):
    # This original descriptor inquiry requires b's lower-bound operand at
    # the same native condition, without reading any b payload. A synthetic
    # operand protected only by the write branch remains unsupported below.
    source = SOURCE.replace("b(i)=a(i)+real(i,8)",
                            "if(i>=lbound(b,1).and.a(i)>0.0_8)then\nb(i)=a(i)+real(i,8)\nendif")
    _path, analysis, selection, prepared = fixture(tmp_path, source)
    footprint = next(item for unit in prepared.offload.units for item in unit.footprints
                     if item.symbol.name == "b")
    assert not footprint.exact
    assert footprint.uploads
    proof = prove_numerical_effects(analysis, ENTRY, selection, prepared)
    output = resource(proof, "b")
    assert output.reads == ()
    assert output.writes
    assert output.overwrites == ()
    result = proof.transfer({"argument::a": resource(proof, "a").reads})
    assert result.reads_covered
    assert result.definitions["argument::b"] == ()


def test_conditional_only_original_lower_operand_remains_protected(tmp_path):
    source = SOURCE.replace("b(i)=a(i)+real(i,8)", "if(a(i)>0.0_8)then\nb(i)=a(i)+real(i,8)\nendif")
    with pytest.raises(NumericalPreparationUnsupported, match="protected.*fort_region_lb_b_1"):
        fixture(tmp_path, source)


def test_tiled_private_helper_definitions_remain_guard_conditional(tmp_path):
    source = SOURCE.replace("sweep(a,b,c,n)", "sweep(a,b,c,n,tile)")
    source = source.replace("integer,intent(in)::n", "integer,intent(in)::n,tile")
    source = source.replace("integer::i", "integer::i,block\nreal(8)::scratch(2),value")
    source = source.replace("parallel do private(i)", "parallel do private(i,block,scratch,value)")
    source = source.replace("do i=-3,n\nb(i)=a(i)+real(i,8)\nenddo", """do block=-3,n,tile
do i=block,min(block+tile-1,n)
call packed(scratch,value,a(i),i)
b(i)=value
enddo
enddo""")
    source = source.replace("end subroutine\nend module", """contains
pure subroutine packed(g,s,x,index)
real(8),intent(out)::g(2),s
real(8),intent(in)::x
integer,intent(in)::index
g(1)=x
g(2)=real(index,8)
s=dot_product(g,g)
end subroutine
end subroutine
end module""")
    _path, analysis, selection, prepared = fixture(tmp_path, source)
    proof = prove_numerical_effects(analysis, ENTRY, selection, prepared)
    assert {item.role for item in proof.preconditions} == {"positive_tile_step", "tile_integer_range"}
    assert "scratch" in prepared.extraction.private_arrays
    assert {item.resource for item in proof.sections.resources} == {"argument::a", "argument::b"}
    assert proof.transfer({"argument::a": resource(proof, "a").reads}).reads_covered
    assert resource(proof, "b").overwrites
    assert "precondition" in proof.public()["transfer_scope"]
    assert proof.public()["unconditional_return_definitions_authorized"] is False


def test_joined_nowait_units_cannot_supply_each_others_entry_reads(tmp_path):
    opening = SOURCE.index("!$omp parallel do")
    closing = SOURCE.index("end subroutine")
    source = SOURCE[:opening] + """!$omp parallel private(i)
!$omp do
do i=-3,n
b(i)=a(i)
enddo
!$omp end do nowait
!$omp do
do i=-3,n
c(i)=b(i)
enddo
!$omp end do
!$omp end parallel
""" + SOURCE[closing:]
    _path, analysis, selection, prepared = fixture(tmp_path, source)
    proof = prove_numerical_effects(analysis, ENTRY, selection, prepared)
    result = proof.transfer({"argument::a": resource(proof, "a").reads})
    assert {item.resource for item in result.uncovered_reads} == {"argument::b"}
    assert result.definitions["argument::b"]
    assert proof.transfer({"argument::a": resource(proof, "a").reads,
                           "argument::b": resource(proof, "b").reads}).reads_covered


def test_dynamic_original_dummy_origin_and_descriptor_dependencies(tmp_path):
    source = SOURCE.replace("sweep(a,b,c,n)", "sweep(a,b,c,n,lo)")
    source = source.replace("integer,intent(in)::n\n", "")
    source = source.replace("real(8),intent(in)::a", "integer,intent(in)::n,lo\nreal(8),intent(in)::a")
    source = source.replace("a(-4:)", "a(lo:)").replace("b(-4:),c(-4:)", "b(lo:),c(lo:)")
    _path, analysis, selection, prepared = fixture(tmp_path, source)
    proof = prove_numerical_effects(analysis, ENTRY, selection, prepared)
    for item in proof.sections.resources:
        assert item.descriptor_lower_bounds
        assert item.descriptor_origin == "original_dummy"
        assert item.lower_bounds == (None,)
    # Numerical indices are original logical indices. Original descriptor
    # lower parameters cancel algebraically, rather than rebasing index data.
    assert resource(proof, "b").overwrites[0].axes[0].lower.value == -3


def test_default_assumed_shape_origin_is_one(tmp_path):
    _path, analysis, selection, prepared = fixture(tmp_path,
        SOURCE.replace("a(-4:)", "a(:)").replace("b(-4:),c(-4:)", "b(:),c(:)").replace("do i=-3,n", "do i=1,n"))
    proof = prove_numerical_effects(analysis, ENTRY, selection, prepared)
    assert all(item.lower_bounds == (1,) and not item.descriptor_lower_bounds for item in proof.sections.resources)


@pytest.mark.parametrize("mutation", [None, "binding", "kind"])
def test_imported_constant_lower_origin_is_authenticated(tmp_path, mutation):
    source = """module origin_values
implicit none
integer,parameter::floor_value=-4
end module
""" + SOURCE.replace("module retained_cells\n", "module retained_cells\nuse origin_values,only:floor_alias=>floor_value\n")
    source = source.replace("a(-4:)", "a(floor_alias:)").replace("b(-4:),c(-4:)", "b(floor_alias:),c(floor_alias:)")
    _path, analysis, selection, prepared = fixture(tmp_path, source)
    proof = prove_numerical_effects(analysis, ENTRY, selection, prepared)
    assert all(item.lower_bounds == (-4,) and not item.descriptor_lower_bounds for item in proof.sections.resources)
    if mutation is None:
        return
    scope = analysis.modules["origin_values"]
    if mutation == "binding":
        scope.bindings["floor_value"] = replace(scope.bindings["floor_value"], declaring_scope=None)
    else:
        scope.kinds.values["floor_value"] = -3
    with pytest.raises(CompilationError) as failure:
        proof.validate(analysis, ENTRY, selection, prepared)
    assert not isinstance(failure.value, NumericalEffectUnsupported)


@pytest.mark.parametrize("after_proof", [False, True])
def test_hidden_module_lower_node_metadata_cannot_replace_original_declaration(tmp_path, after_proof):
    source = SOURCE.replace("implicit none\ncontains", "implicit none\nreal(8)::b(-4:8192)\ncontains")
    source = source.replace("sweep(a,b,c,n)", "sweep(a,c,n)")
    source = source.replace("real(8),intent(inout)::b(-4:),c(-4:)", "real(8),intent(inout)::c(-4:)")
    _path, analysis, selection, prepared = fixture(tmp_path, source)
    proof = prove_numerical_effects(analysis, ENTRY, selection, prepared) if after_proof else None
    binding = analysis.modules["retained_cells"].bindings["b"]
    public = binding.public()
    # Reuse a different authentic original expression while preserving the
    # declaration's public spelling and shape AST. It must not become b's
    # original origin merely because it is a valid INTEGER expression.
    binding.lower_bound_nodes = (binding.shape_nodes[0].items[1],)
    assert binding.public() == public
    def invalid():
        if after_proof:
            proof.validate(analysis, ENTRY, selection, prepared)
        else:
            prove_numerical_effects(analysis, ENTRY, selection, prepared)

    with pytest.raises(CompilationError, match="shape/lower-bound node authority") as failure:
        invalid()
    assert not isinstance(failure.value, NumericalEffectUnsupported)


def test_fixed_component_array_effects_use_original_field_bound_authority(tmp_path):
    source = SOURCE.replace("implicit none\ncontains", """implicit none
type retained_weights
real(8)::samples(-4:8192)
real(8),allocatable::unrelated(:)
end type
contains""")
    source = source.replace("sweep(a,b,c,n)", "sweep(a,b,c,n,weights)\ntype(retained_weights),intent(in)::weights")
    source = source.replace("b(i)=a(i)+real(i,8)", "b(i)=a(i)+weights%samples(i)+real(i,8)")
    _path, analysis, selection, prepared = fixture(tmp_path, source)
    proof = prove_numerical_effects(analysis, ENTRY, selection, prepared)
    field = next(item for item in proof.sections.resources if item.resource == "argument::weights%samples")
    assert field.lower_bounds == (-4,)
    assert not field.descriptor_lower_bounds
    assert field.reads
    assert not field.writes
    assert proof.validate(analysis, ENTRY, selection, prepared) is proof
    view = next(item for item in prepared.extraction.bindings if item.root == field.resource)
    before = view.public()
    view.lower_bound_nodes = (view.shape_nodes[0].items[1],)
    assert view.public() == before
    with pytest.raises(CompilationError, match="shape/lower-bound node authority") as failure:
        proof.validate(analysis, ENTRY, selection, prepared)
    assert not isinstance(failure.value, NumericalEffectUnsupported)


def test_cross_resource_descriptor_bound_requires_a_supported_renderer(tmp_path):
    _path, analysis, selection, prepared = fixture(tmp_path, SOURCE.replace("do i=-3,n", "do i=-3,size(a,1)-4"))
    with pytest.raises(NumericalEffectUnsupported, match="cross-resource descriptor"):
        prove_numerical_effects(analysis, ENTRY, selection, prepared)


def test_empty_iteration_does_not_define_an_unconditional_point(tmp_path):
    _path, analysis, selection, prepared = fixture(tmp_path)
    proof = prove_numerical_effects(analysis, ENTRY, selection, prepared)
    defined = resource(proof, "b").overwrites[0]
    from compiler.frontend.native_definitions import rectangle_covers

    point = NativeRectangle((NativeAxis(defined.axes[0].lower, defined.axes[0].lower, True),), selection[0])
    assert not rectangle_covers(defined, point)


@pytest.mark.parametrize("kind", ["active", "full_read", "full_write", "operator", "overflow", "depth"])
def test_unsupported_physical_geometry_never_gains_whole_authority(tmp_path, kind):
    _path, analysis, selection, prepared = fixture(tmp_path)
    mapper = _Mapper(analysis, analysis.routines[ENTRY], prepared, None)
    unit, = prepared.offload.units
    footprint = next(item for item in unit.footprints if item.symbol.name == "b")
    if kind in {"full_read", "full_write"}:
        with pytest.raises(NumericalEffectUnsupported, match="bounded physical"):
            mapper.resource(replace(footprint, **{kind: True}), unit, selection[0], mapper.iterations(unit))
    elif kind == "active":
        box, = footprint.writes
        with pytest.raises(NumericalEffectUnsupported, match="active-unit"):
            mapper.rectangles((replace(box, active_units=(unit.index, unit.index + 1)),), unit,
                              mapper.binding("argument::b"), selection[0], mapper.iterations(unit))
    else:
        expression = Literal(str(2**31), ScalarType.INTEGER) if kind == "overflow" else Literal("1", ScalarType.INTEGER)
        if kind == "operator":
            expression = Binary("/", expression, expression)
        elif kind == "depth":
            for _ in range(analysis.depth_limit + 2):
                expression = Binary("+", expression, Literal("1", ScalarType.INTEGER))
        with pytest.raises(NumericalEffectUnsupported):
            mapper.bound(expression)


def test_registry_cache_reuse_has_same_charged_work(tmp_path):
    _path, analysis, selection, prepared = fixture(tmp_path)
    cold, reused = [], []
    proof = prove_numerical_effects(analysis, ENTRY, selection, prepared, consume=cold.append)
    assert prove_numerical_effects(analysis, ENTRY, selection, prepared, consume=reused.append) is proof
    assert sum(cold) == sum(reused) > 0
    assert analysis._numerical_effect_proofs[proof.identity] is proof
    projected = _fork(analysis, ())
    assert projected._numerical_effect_proofs == {}
    assert projected._numerical_preparations == {}
    assert projected._numerical_preparation_providers == {}
    with pytest.raises(CompilationError, match="authority"):
        proof.validate(projected, ENTRY, selection, prepared)


@pytest.mark.parametrize("mutation", ["cloned_source", "file", "binding", "prepared", "proof"])
def test_foreign_or_changed_source_authority_is_hard(tmp_path, mutation):
    path, analysis, selection, prepared = fixture(tmp_path)
    proof = prove_numerical_effects(analysis, ENTRY, selection, prepared)
    if mutation == "cloned_source":
        selection = tuple(copy(node) if type(node).__name__ == "Block_Nonlabel_Do_Construct" else node
                          for node in selection)
    elif mutation == "file":
        path.write_text(SOURCE.replace("a(i)+", "2.0_8*a(i)+"))
    elif mutation == "binding":
        binding = analysis.routines[ENTRY].scope.bindings["a"]
        analysis.routines[ENTRY].scope.bindings["a"] = replace(binding, declaring_scope=None)
    elif mutation == "prepared":
        prepared = SimpleNamespace(**vars(prepared))
    else:
        proof = replace(proof, sections=replace(proof.sections, resources=()))
    with pytest.raises(CompilationError) as failure:
        proof.validate(analysis, ENTRY, selection, prepared)
    assert not isinstance(failure.value, NumericalEffectUnsupported)


def test_shared_and_local_budget_failures_stay_hard(tmp_path):
    _path, analysis, selection, prepared = fixture(tmp_path)

    def exhausted(_count):
        raise CompilationError("shared numerical work exhausted")

    with pytest.raises(CompilationError, match="shared numerical work") as failure:
        prove_numerical_effects(analysis, ENTRY, selection, prepared, consume=exhausted)
    assert not isinstance(failure.value, NumericalEffectUnsupported)
    mapper = _Mapper(analysis, analysis.routines[ENTRY], prepared, None)
    mapper.work = 65536
    with pytest.raises(CompilationError, match="traversal budget") as failure:
        mapper.tick()
    assert not isinstance(failure.value, NumericalEffectUnsupported)
