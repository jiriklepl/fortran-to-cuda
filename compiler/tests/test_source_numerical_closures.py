"""Source-backed private closures retain logical coordinates and native guards."""

from hashlib import sha256
import re

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.frontend import lower_source
from compiler.frontend.source_effects import SourceEffects, _children, _kind
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.regions import extract_region
from compiler.tests.test_inline_source_regions import original_joined_group
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT


SOURCE = """module renamed_numerics
implicit none
contains
subroutine advance(a,n,tile)
real(8),intent(inout)::a(-3:)
integer,intent(in)::n,tile
integer::i,block
real(8)::g(2),s
do block=-3,n,tile
 do i=block,min(block+tile-1,n)
  call derivative(g,i)
  call invariant(s,g)
  a(i)=s+real(i,8)
 enddo
enddo
contains
pure subroutine derivative(g,i)
real(8),intent(out)::g(2)
integer,intent(in)::i
g(1)=a(i)
g(2)=real(i,8)
end subroutine
pure subroutine invariant(s,g)
real(8),intent(out)::s
real(8),intent(in)::g(2)
s=dot_product(g,g)
end subroutine
end subroutine
end module
"""


def extraction(tmp_path, source=SOURCE):
    path = tmp_path / "original.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    routine = analysis.routines["renamed_numerics::advance"]
    nodes = tuple(_children(routine.execution))
    position = next(i for i, node in enumerate(nodes) if _kind(node) == "Block_Nonlabel_Do_Construct")
    return analysis, routine, extract_region(analysis, routine, nodes[position],
                                            preceding=nodes[:position], following=nodes[position + 1:])


def test_pure_private_closure_is_rebased_and_tiled_domain_is_proved(tmp_path):
    analysis, routine, region = extraction(tmp_path)
    assert analysis._candidates(routine.scope, "derivative") == ["renamed_numerics::advance::derivative"]
    assert set(region.numerical_helpers) == {"renamed_numerics::advance::derivative",
                                             "renamed_numerics::advance::invariant"}
    assert region.private_arrays == ("g",)
    assert region.tile_domains[0]["lower"] == "- 3"
    assert len(region.runtime_guards) == 2
    compact = "".join(region.source.lower().split())
    assert "doblock=" not in compact
    assert "doi=-3,n" in compact
    assert "a((i)-fort_region_lb_a_1+1)" in compact
    function, plan = prepare_function(lower_source(region.source, region.entry, source_name="borrowed.f90"),
                                      options=CompilerOptions())
    assert len(plan.regions) == 1
    assert not any(symbol.rank for symbol in function.symbols if not symbol.parameter)


def test_guarded_public_owner_keeps_internal_helper_native_fallback_at_caller(tmp_path):
    path = tmp_path / "original.f90"
    path.write_text(SOURCE)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::a": FACT}}
    outputs, report = ScopeBuilder([path], "renamed_numerics::advance", facts=facts,
                                  options=CompilerOptions(), config=OffloadConfig(policy="sections")).run()
    assert report["scope_count"] == 1, report["boundaries"]
    original = outputs[report["sources"][str(path)]["replacement"]]
    kind_alias = re.search(r"(fort_scope_guard_i64_\w+)\s*=>\s*c_int64_t", original).group(1)
    compact = "".join(original.replace("&", "").split())
    assert f"int(tile,kind={kind_alias})>0_{kind_alias}" in compact
    assert f"+int(tile,kind={kind_alias})<=2147483647_{kind_alias}" in compact
    environment_alias = re.search(r"(fort_scope_environment_\w+)\s*=>\s*fort_scope_numerical_environment_supported", original).group(1)
    assert environment_alias + "()/=0" in compact
    assert "if (fort_native_" in original
    assert original.count("pure subroutine derivative") == 1
    scope, = report["scopes"]
    assert scope["runtime_preflight"]["position"] == "original reached segment after allocation and bounds guards"
    assert report["inline_numerical_regions"]["regions"][0]["numerical_environment"]["exceptions"] == "host traps disabled"


def test_unrelated_type_and_contains_do_not_reject_a_supported_candidate(tmp_path):
    source = SOURCE.replace("integer::i,block", "type unrelated\n integer::unused\nend type\ninteger::i,block")
    _, _, region = extraction(tmp_path, source)
    assert region.numerical_helpers


@pytest.mark.parametrize("replacement,reason", [
    ("a(i)=s+real(block,8)", "tile coordinates are observed"),
    ("g(2)=g(2)+real(i,8)", "per-iteration definition|before definition"),
])
def test_observed_tile_coordinates_or_undefined_private_elements_are_not_offloaded(tmp_path, replacement, reason):
    source = SOURCE.replace("a(i)=s+real(i,8)", replacement) if replacement.startswith("a(") else SOURCE.replace("g(2)=real(i,8)", replacement)
    with pytest.raises(CompilationError, match=reason):
        _, _, region = extraction(tmp_path, source)
        prepare_function(lower_source(region.source, region.entry, source_name="borrowed.f90"), options=CompilerOptions())


def test_identically_named_helpers_resolve_in_their_original_lexical_owner(tmp_path):
    second = SOURCE[SOURCE.index("subroutine advance"):SOURCE.rindex("end module")]
    source = SOURCE.replace("end module", second.replace("subroutine advance", "subroutine other") + "end module")
    path = tmp_path / "original.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    for name in ("advance", "other"):
        routine = analysis.routines["renamed_numerics::" + name]
        assert analysis._candidates(routine.scope, "derivative") == [routine.qualified + "::derivative"]


def test_combined_parallel_do_with_private_arrays_and_runtime_schedule_is_joined(tmp_path):
    source = SOURCE.replace("do block=-3,n,tile", "i=0\n!$omp parallel do private(g,s,i,block) schedule(runtime)\ndo block=-3,n,tile").replace(
        "enddo\ncontains", "enddo\n!$omp end parallel do\ncontains")
    path = tmp_path / "original.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    routine = analysis.routines["renamed_numerics::advance"]
    group = original_joined_group(routine)
    region = extract_region(analysis, routine, group)
    assert region.completion["available"]
    assert region.completion["has_openmp_in_closure"]
    assert region.private_arrays == ("g",)
    _, plan = prepare_function(lower_source(region.source, region.entry, source_name="borrowed.f90"),
                                options=CompilerOptions())
    assert len(plan.regions) == 1


def test_unused_intrinsic_ieee_import_keeps_host_capture_resolution_without_authorizing_flags(tmp_path):
    source = SOURCE.replace("pure subroutine derivative(g,i)", "pure subroutine derivative(g,i)\nuse,intrinsic::ieee_arithmetic")
    _, _, region = extraction(tmp_path, source)
    assert "a((i) - fort_region_lb_a_1 + 1)" in region.source.lower()
    changed = source.replace("g(1)=a(i)", "call ieee_set_flag(ieee_invalid,.false.)\ng(1)=a(i)")
    with pytest.raises(CompilationError, match="unresolved source helper"):
        extraction(tmp_path, changed)
    renamed = SOURCE.replace("pure subroutine derivative(g,i)",
                             "pure subroutine derivative(g,i)\nuse,intrinsic::ieee_arithmetic,only:sin=>ieee_value").replace(
                                 "g(1)=a(i)", "g(1)=sin(a(i))")
    with pytest.raises(CompilationError, match="shadowed"):
        extraction(tmp_path, renamed)


@pytest.mark.parametrize("observer", ["ieee_get_flag(ieee_invalid,seen)", "ieee_get_status(status)", "peek(ieee_invalid,seen)"])
def test_source_intrinsic_flag_observers_and_resolved_aliases_prevent_new_gpu_closure(tmp_path, observer):
    source = SOURCE.replace("end module", """subroutine caller
use,intrinsic::ieee_exceptions,only:ieee_get_flag,ieee_get_status,ieee_invalid,ieee_status_type,peek=>ieee_get_flag
logical::seen
type(ieee_status_type)::status
call """ + observer + "\nend subroutine\nend module")
    with pytest.raises(CompilationError, match="source-observable floating-point exception flags"):
        extraction(tmp_path, source)


def test_same_spelling_user_procedure_does_not_establish_intrinsic_flag_observation(tmp_path):
    source = SOURCE.replace("end module", """subroutine ieee_get_flag(value)
logical,intent(out)::value
value=.false.
end subroutine
subroutine caller
logical::seen
call ieee_get_flag(seen)
end subroutine
end module""")
    _, _, region = extraction(tmp_path, source)
    assert region.numerical_helpers


def test_guard_kind_import_keeps_original_c_int64_t_binding_visible_to_native_fallback(tmp_path):
    source = SOURCE.replace("real(8)::g(2),s", "real(8)::g(2),s,c_int64_t").replace(
        "do block=-3,n,tile", "c_int64_t=17.d0\ndo block=-3,n,tile").replace(
        "a(i)=s+real(i,8)", "a(i)=s+c_int64_t+real(i,8)")
    path = tmp_path / "original.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()}, "captures": {"argument::a": FACT}}
    outputs, report = ScopeBuilder([path], "renamed_numerics::advance", facts=facts,
                                  options=CompilerOptions(), config=OffloadConfig(policy="sections")).run()
    assert report["scope_count"] == 1, report["boundaries"]
    text = outputs[report["sources"][str(path)]["replacement"]]
    imports = [line for line in re.sub(r"&\s*\n\s*&?", "", text).splitlines()
               if line.startswith("use iso_c_binding, only:") and "c_int64_t" in line]
    assert imports
    assert all("=> c_int64_t" in line for line in imports)
    assert "a(i)=s+c_int64_t+real(i,8)" in text  # Unchanged native caller fallback.


def test_shadowed_int_conversion_is_an_explicit_tile_guard_boundary(tmp_path):
    source = SOURCE.replace("integer::i,block", "integer::i,block,int")
    with pytest.raises(CompilationError, match="tile guard conversion intrinsic is shadowed"):
        extraction(tmp_path, source)


def test_modified_helper_ast_cannot_borrow_original_source_authority(tmp_path):
    analysis, routine, _ = extraction(tmp_path)
    helper = analysis.numerical_helpers[routine.qualified + "::derivative"]
    helper.execution.content.pop()
    node = next(node for node in _children(routine.execution) if _kind(node) == "Block_Nonlabel_Do_Construct")
    with pytest.raises(CompilationError, match="source-backed procedure authority"):
        extract_region(analysis, routine, node)


def test_new_real_helper_varargs_keep_unproved_native_nan_ordering(tmp_path):
    source = SOURCE.replace("s=dot_product(g,g)", "s=max(g(1),g(2),0.d0)")
    with pytest.raises(CompilationError, match="real MIN/MAX with more than two operands"):
        extraction(tmp_path, source)
    _, _, region = extraction(tmp_path, source.replace("max(g(1),g(2),0.d0)", "max(-1.d0,2.d0,0.d0)"))
    assert "MAX(" in region.source


def test_multifile_typed_function_and_transitive_source_identity(tmp_path):
    kernel = tmp_path / "kernel.f90"
    helper = tmp_path / "helper.f90"
    kernel.write_text("""module numerical_owner
use numerical_library,only:twice
implicit none
contains
subroutine step(a)
real(8),intent(inout)::a(:)
integer::i
do i=1,size(a)
a(i)=twice(a(i))
enddo
end subroutine
end module
""")
    helper_source = """module numerical_library
implicit none
contains
pure real(8) function twice(x)
real(8),intent(in)::x
twice=x*2.d0
end function
end module
"""
    helper.write_text(helper_source)

    def selected():
        analysis = SourceEffects([kernel, helper])
        routine = analysis.routines["numerical_owner::step"]
        node, = _children(routine.execution)
        region = extract_region(analysis, routine, node)
        _, plan = prepare_function(lower_source(region.source, region.entry, source_name="borrowed.f90"),
                                   options=CompilerOptions())
        assert len(plan.regions) == 1
        return region

    first = selected()
    helper.write_text(helper_source.replace("x*2.d0", "x*3.d0"))
    second = selected()
    assert first.source_identity != second.source_identity
    assert first.numerical_helpers == second.numerical_helpers == ("numerical_library::twice",)


@pytest.mark.parametrize("change,reason", [
    (lambda source: source.replace("pure subroutine derivative", "subroutine derivative"), "explicit PURE"),
    (lambda source: source.replace("g(1)=a(i)", "call derivative(g,i)\ng(1)=a(i)"), "recursive"),
    (lambda source: source.replace("real(8)::g(2),s", "real(8),save::g(2)\nreal(8)::s"), "original owner|PRIVATE|live|whole-array"),
])
def test_unproved_helper_effects_and_persistent_private_storage_remain_native(tmp_path, change, reason):
    with pytest.raises(CompilationError, match=reason):
        extraction(tmp_path, change(SOURCE))
