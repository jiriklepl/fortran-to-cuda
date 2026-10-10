"""Outlined fixed vectors retain lexical constants and physical coordinates."""

from __future__ import annotations

import copy
import re
import shutil
import subprocess
from hashlib import sha256

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.driver.options import CompilerOptions
from compiler.frontend.source_effects import SourceEffects, _children, _kind
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.regions import extract_region
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT


def selected(routine):
    nodes = tuple(_children(routine.execution))
    position = next(index for index, node in enumerate(nodes) if _kind(node) == "Block_Nonlabel_Do_Construct")
    return nodes[position], nodes[:position], nodes[position + 1:]


def fixture(tmp_path, *, precision=8, expression="sum(coeff*a(i-2:i+1,j,k))*gain", joined=False):
    constants = tmp_path / "constants.f90"
    constants.write_text(f"""module renamed_coefficients
implicit none
integer,parameter::first=-1,last=2
integer,parameter::basis(first:last)=[-1,9,9,-1]
real(4),parameter::unit=1._4/10._4
real({precision}),parameter::scale=real(unit,{precision})
end module
""")
    source = tmp_path / "caller.f90"
    source.write_text(f"""module renamed_windows
use renamed_coefficients, only: coeff=>basis,gain=>scale
implicit none
contains
subroutine advance(a,out,n)
real({precision}),intent(in)::a(-4:,-2:,0:)
real({precision}),intent(inout)::out(-4:,-2:,0:)
integer,intent(in)::n
integer::i,j,k
real({precision})::unused_runtime_line(-4:ubound(a,1))
{'!$omp parallel do collapse(3) private(i,j,k)' if joined else ''}
do k=2,ubound(a,3)-1
do j=0,ubound(a,2)-1
do i=-1,n
out(i,j,k)={expression}+real(i+13*j+7*k,{precision})
enddo
enddo
enddo
{'!$omp end parallel do' if joined else ''}
end subroutine
end module
""")
    analysis = SourceEffects([constants, source])
    routine = analysis.routines["renamed_windows::advance"]
    node, preceding, following = selected(routine)
    if joined:
        from compiler.tests.test_inline_source_regions import original_joined_group
        node = original_joined_group(routine)
        preceding = following = ()
    region = extract_region(analysis, routine, node, preceding=preceding, following=following)
    return analysis, routine, region, (constants, source)


@pytest.mark.parametrize("precision", [4, 8])
@pytest.mark.parametrize("joined", [False, True])
def test_imported_vectors_are_immutable_declarations_not_captures_or_private_arrays(tmp_path, precision, joined):
    _, _, region, _ = fixture(tmp_path, precision=precision, joined=joined)
    assert set(region.immutable_constants) == {
        "renamed_coefficients::basis", "renamed_coefficients::scale", "renamed_coefficients::unit"}
    assert not region.private_arrays
    assert not any("coeff" in item.name or "gain" in item.name for item in region.parameters)
    text = region.source.lower()
    assert "parameter :: fort_region_constant_" in text
    assert "(-1:2) = [- 1, 9, 9, - 1]" in text
    assert "1._4 / 10._4" in text
    assert "unused_runtime_line" not in text
    assert "use renamed_coefficients" not in text
    assert region.requires_numerical_environment
    assert region.public()["immutable_constants"] == list(region.immutable_constants)


@pytest.mark.parametrize(("expression", "fragments"), [
    ("sum(a(i-2:i+1,j,k))", ("(i - 2) - fort_region_lb_a_1 + 1", "(i + 1) - fort_region_lb_a_1 + 1")),
    ("sum(a(i,j-2:j+1,k))", ("(j - 2) - fort_region_lb_a_2 + 1", "(j + 1) - fort_region_lb_a_2 + 1")),
    ("sum(a(i,j,k+1:k-2:-1))", ("(k + 1) - fort_region_lb_a_3 + 1", "(k - 2) - fort_region_lb_a_3 + 1 : - 1")),
    ("sum(coeff(2:-1:-1)*a(i+1:i-2:-1,j,k))", ("fort_region_constant_0(2 : - 1 : - 1)",)),
])
def test_sections_rebase_both_endpoints_and_preserve_order(tmp_path, expression, fragments):
    _, _, region, _ = fixture(tmp_path, expression=expression)
    compact = " ".join(region.source.replace("&", " ").split())
    assert all(fragment in compact for fragment in fragments), compact
    assert "REAL(i + 13 * j + 7 * k, 8)" in compact


@pytest.mark.parametrize("expression", ["sum(a(i:i+1:0,j,k))", "sum(a(i:i+1:n,j,k))"])
def test_zero_or_runtime_slice_stride_is_not_authorized(tmp_path, expression):
    with pytest.raises(CompilationError, match="constant nonzero"):
        fixture(tmp_path, expression=expression)


def test_changed_constant_initializer_cannot_reuse_original_source_authority(tmp_path):
    analysis, routine, _, _ = fixture(tmp_path)
    module = analysis.modules["renamed_coefficients"]
    declaration = next(node for node in walk(module.node, F.Entity_Decl) if str(node.items[0]).lower() == "basis")
    declaration.items[3].items = ("=", F.Expr("[1,1,1,1]"))
    node, preceding, following = selected(routine)
    with pytest.raises(CompilationError, match="original source-backed module"):
        extract_region(analysis, routine, node, preceding=preceding, following=following)


def test_copied_constant_bounds_do_not_replace_original_declaration_proof(tmp_path):
    analysis, routine, _, _ = fixture(tmp_path)
    binding = analysis.modules["renamed_coefficients"].bindings["basis"]
    binding.shape_nodes = tuple(copy.copy(axis) for axis in binding.shape_nodes)
    node, preceding, following = selected(routine)
    with pytest.raises(CompilationError, match="descriptor differs"):
        extract_region(analysis, routine, node, preceding=preceding, following=following)


def test_foreign_scope_cannot_supply_an_imported_constant_initializer(tmp_path):
    analysis, routine, _, paths = fixture(tmp_path)
    binding = analysis.modules["renamed_coefficients"].bindings["unit"]
    foreign = copy.copy(binding.declaring_scope)
    foreign.node = SourceEffects(paths).modules["renamed_coefficients"].node
    declaration = next(node for node in walk(foreign.node, F.Entity_Decl) if str(node.items[0]).lower() == "unit")
    declaration.items[3].items = ("=", F.Expr("7._4"))
    binding.declaring_scope = foreign
    node, preceding, following = selected(routine)
    with pytest.raises(CompilationError, match="original declaring scope"):
        extract_region(analysis, routine, node, preceding=preceding, following=following)


def test_helper_free_vectors_keep_native_minmax_ordering_restriction(tmp_path):
    vector = "a(i-2:i+1,j,k)"
    with pytest.raises(CompilationError, match="real MIN/MAX with more than two operands"):
        fixture(tmp_path, expression=f"sum(max({vector},{vector},{vector}))")


@pytest.mark.parametrize("observer", [False, True])
def test_lowered_vector_environment_controls_original_source_guard_and_flag_observers(tmp_path, observer):
    _, _, _, paths = fixture(tmp_path)
    source = paths[1]
    text = source.read_text().replace("integer::i,j,k", "integer::i,j,k\nreal(8)::g(4)")
    text = text.replace("out(i,j,k)=sum(coeff*a(i-2:i+1,j,k))*gain",
                        "g=sqrt(a(i-2:i+1,j,k))\nout(i,j,k)=g(1)")
    if observer:
        text = text.replace("end module", """subroutine observer
use,intrinsic::ieee_exceptions,only:ieee_get_flag,ieee_invalid
logical::seen
call ieee_get_flag(ieee_invalid,seen)
end subroutine
end module""")
    source.write_text(text)
    facts = {"schema_version": 1, "participation": "serial",
             "captures": {"argument::a": FACT, "argument::out": FACT},
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths}}
    outputs, manifest = ScopeBuilder(paths, "renamed_windows::advance", facts=facts,
        options=CompilerOptions(), config=OffloadConfig("sections")).run()
    if observer:
        assert manifest["scope_count"] == 0
        assert "source-observable floating-point exception flags" in str(manifest["boundaries"])
    else:
        assert manifest["scope_count"] == 1, manifest["boundaries"]
        region, = manifest["inline_numerical_regions"]["regions"]
        assert region["numerical_environment"]["required"]
        original = outputs[manifest["sources"][str(source)]["replacement"]]
        alias = re.search(r"(fort_scope_environment_\w+)\s*=>\s*fort_scope_numerical_environment_supported", original).group(1)
        assert alias + "()/=0" in "".join(original.replace("&", "").split())


def test_helper_dummy_cannot_shadow_generated_captured_bounds(tmp_path):
    _, _, _, paths = fixture(tmp_path)
    source = paths[1]
    text = source.read_text().replace("sum(coeff*a(i-2:i+1,j,k))*gain", "shift(7)")
    text = text.replace("end subroutine", """contains
pure function shift(fort_region_lb_a_1) result(value)
integer,intent(in)::fort_region_lb_a_1
real(8)::value
value=a(i,j,k)+real(fort_region_lb_a_1,8)
end function
end subroutine""")
    source.write_text(text)
    analysis = SourceEffects(paths)
    routine = analysis.routines["renamed_windows::advance"]
    node, preceding, following = selected(routine)
    with pytest.raises(CompilationError, match="lower-bound parameter namespace conflicts"):
        extract_region(analysis, routine, node, preceding=preceding, following=following)


def test_imported_constant_changes_invalidate_region_identity(tmp_path):
    _, _, before, paths = fixture(tmp_path)
    constants, source = paths
    constants.write_text(constants.read_text().replace("[-1,9,9,-1]", "[-2,9,9,-2]"))
    analysis = SourceEffects([constants, source])
    routine = analysis.routines["renamed_windows::advance"]
    node, preceding, following = selected(routine)
    after = extract_region(analysis, routine, node, preceding=preceding, following=following)
    assert before.source_identity != after.source_identity
    assert before.source != after.source


def test_minimum_integer_constant_bound_retains_valid_fortran_spelling(tmp_path):
    _, _, _, paths = fixture(tmp_path)
    constants = paths[0]
    constants.write_text(constants.read_text().replace("first=-1,last=2", "first=(-2147483647-1),last=-2147483645"))
    analysis = SourceEffects(paths)
    routine = analysis.routines["renamed_windows::advance"]
    node, preceding, following = selected(routine)
    region = extract_region(analysis, routine, node, preceding=preceding, following=following)
    assert "((-2147483647 - 1):-2147483645)" in region.source
    assert "-2147483648" not in region.source


@pytest.mark.native
@pytest.mark.parametrize("precision", [4, 8])
@pytest.mark.parametrize("expression", [
    "sum(coeff*a(i-2:i+1,j,k))*gain", "sum(a(::-1,j,k))",
    "sum(a(i-1::-1,j,k))", "sum(a(:i+1:-1,j,k))",
])
def test_outlined_vectors_match_native_complete_fields_with_negative_bounds(tmp_path, precision, expression):
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    if fc is None:
        pytest.skip("requires native Fortran")
    _, _, region, paths = fixture(tmp_path, precision=precision, expression=expression)
    normalized = region.write(tmp_path / "normalized")
    module, entry = region.entry.split("::")
    actuals = []
    for parameter in region.parameters:
        if parameter.lower_bound_dimension is not None:
            actuals.append(str((-4, -2, 0)[parameter.lower_bound_dimension - 1]))
        else:
            actuals.append("expected" if parameter.name == "out" else parameter.name)
    driver = tmp_path / "driver.f90"
    driver.write_text(f"""program verify
use renamed_windows,only:advance
use {module},only:{entry}
implicit none
real({precision}),allocatable::a(:,:,:),out(:,:,:),expected(:,:,:)
integer::i,j,k,n,shape
do shape=0,2
n=3*shape-2
allocate(a(-4:n+2,-2:5,0:5),out(-4:n+2,-2:5,0:5),expected(-4:n+2,-2:5,0:5))
do k=0,5
do j=-2,5
do i=-4,n+2
a(i,j,k)=real(i+17*j+31*k,{precision})/32._{precision}
enddo
enddo
enddo
out=-117._{precision}
expected=out
call advance(a,out,n)
call {entry}({','.join(actuals)})
if(any(out/=expected)) error stop 'complete fields or halos differ'
deallocate(a,out,expected)
enddo
print *,'FIELDS_OK'
end program
""")
    command = [fc, "-O3", "-fopenmp", "-fcheck=all", *(str(path) for path in paths),
               str(normalized), str(driver), "-o", str(tmp_path / "verify")]
    compiled = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=90, check=False)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    checked = subprocess.run([str(tmp_path / "verify")], capture_output=True, text=True, timeout=30, check=False)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "FIELDS_OK" in checked.stdout, checked.stdout + checked.stderr
