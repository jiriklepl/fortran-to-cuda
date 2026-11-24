"""Regression tests for soundness, coherence, and build fixes from the compiler audit."""

from __future__ import annotations

from pathlib import Path

import pytest

from compiler.analysis import ParallelizationError, build_execution_plan
from compiler.emission.common.c_family import render_expression
from compiler.emission.cuda.scoped import generate_scoped
from compiler.frontend import lower_file
from compiler.frontend.component_bindings import register_associates, source_scope_for
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.syntax import parse_fortran
from compiler.ir import Assignment, CompilationError, Literal, Reference, ScalarType, Unary
from compiler.offload.analysis import Unit, _unit_footprints
from compiler.offload.config import OffloadConfig
from compiler.resident.array_statements import lower_array_statements
from compiler.resident.build import Executable, ObjectLibrary, cmake_lists, cmake_path
from compiler.resident.helpers import _element_pointer, _select_case
from compiler.resident.inlining import inline_calls
from compiler.resident.procedures import declarations_of, procedures_of, renamed_lines
from compiler.resident.reductions import lower_reductions
from compiler.resident.specialize import integer_parameters, specialize
from compiler.scopes.regions import _declared_pure
from compiler.transforms.fusion import optimize_function


def _lower(tmp_path: Path, body: str, name: str = "k"):
    path = tmp_path / "src.f90"
    path.write_text(body)
    return lower_file(path, name)


# --- 1. Analysis, dependence proof & fusion ---


def test_stale_definitions_do_not_certify_square_waw_race(tmp_path: Path):
    fn = _lower(
        tmp_path,
        """module m
contains
  subroutine k(a, b, n)
    real(8), intent(in) :: a(:)
    real(8), intent(inout) :: b(:)
    integer, intent(in) :: n
    integer :: i, k1, m1
    do i = 1, n
      if (a(i) > 0.0d0) then
        k1 = 1
      else
        k1 = 2
      end if
      m1 = k1
      k1 = i
      b(m1 * m1) = 1.0d0
    end do
  end subroutine
end module
""",
    )
    with pytest.raises(ParallelizationError):
        build_execution_plan(fn)


def test_guarded_division_is_not_hoisted_into_entry_assumptions(tmp_path: Path):
    fn = _lower(
        tmp_path,
        """module m
contains
  subroutine k(a, n, x, d)
    real(8), intent(inout) :: a(:)
    integer, intent(in) :: n, x, d
    integer :: i
    do i = 1, n
      if (d /= 0) then
        a(i) = a(x / d) + 1.0d0
      end if
    end do
  end subroutine
end module
""",
    )
    with pytest.raises(ParallelizationError):
        build_execution_plan(fn)


def test_out_of_range_isl_constraint_does_not_emit_overflowing_literal(tmp_path: Path):
    fn = _lower(
        tmp_path,
        """module m
contains
  subroutine k(a, step_k)
    real(8), intent(inout) :: a(:)
    integer, intent(in) :: step_k
    integer :: i
    do i = -10, 2147483647
      a(i + step_k) = a(i)
    end do
  end subroutine
end module
""",
    )
    with pytest.raises(ParallelizationError):
        build_execution_plan(fn)


def test_shifted_origin_fusion_skips_when_bound_scalar_is_written_in_body(tmp_path: Path):
    fn = _lower(
        tmp_path,
        """module m
contains
  subroutine k(a, b)
    real(8), intent(inout) :: a(:), b(:)
    integer :: i, j, s
    s = 10
    do i = s, s + 9
      s = 999
      a(i) = dble(s)
    end do
    do j = 1, 10
      b(j) = dble(j)
    end do
  end subroutine
end module
""",
    )
    _opt, plan = optimize_function(fn)
    assert len(plan.regions) == 2


# --- 2. Frontend lowering, source_effects & component_bindings ---


def test_pure_copy_back_rejects_mutated_array_subscript(tmp_path: Path):
    with pytest.raises(CompilationError, match="same one after the call"):
        _lower(
            tmp_path,
            """module m
contains
  pure subroutine mutate_and_set(x, idx)
    real(8), intent(out) :: x
    integer, intent(inout) :: idx(:)
    idx(1) = 2
    x = 42.0d0
  end subroutine
  subroutine k(a, idx)
    real(8), intent(inout) :: a(:)
    integer, intent(inout) :: idx(:)
    call mutate_and_set(a(idx(1)), idx)
  end subroutine
end module
""",
        )


def test_lhs_vector_section_snapshots_self_referencing_subscripts(tmp_path: Path):
    fn = _lower(
        tmp_path,
        """module m
contains
  subroutine k(idx)
    integer, intent(inout) :: idx(:)
    idx(idx(1):idx(1)+2) = [10, 20, 30]
  end subroutine
end module
""",
    )
    assigns = [s for s in fn.body.statements if isinstance(s, Assignment)]
    idx_snaps = [s for s in assigns if isinstance(s.target, Reference) and s.target.symbol.name.startswith("fort_idx")]
    assert len(idx_snaps) == 1


def test_pure_cannot_call_nonpure_and_inferred_intent_cannot_write_intent_in(tmp_path: Path):
    with pytest.raises(CompilationError, match="PURE procedure cannot call non-pure"):
        _lower(
            tmp_path,
            """module m
contains
  subroutine mutate(x)
    real(8) :: x(3)
    x(1) = 99.0d0
  end subroutine
  pure subroutine middle(p)
    real(8), intent(in) :: p(3)
    call mutate(p)
  end subroutine
  subroutine k(a)
    real(8), intent(inout) :: a(:)
    real(8) :: v(3)
    v = [1.0d0, 2.0d0, 3.0d0]
    call middle(v)
    a(1) = v(1)
  end subroutine
end module
""",
        )


def test_array_inquiry_rejects_vector_dim(tmp_path: Path):
    with pytest.raises(CompilationError, match="dimension must be INTEGER"):
        _lower(
            tmp_path,
            """module m
contains
  subroutine k(a, out)
    real(8), intent(in) :: a(:)
    integer, intent(out) :: out(:)
    out(1) = size(a, [1])
  end subroutine
end module
""",
        )


def test_positional_after_keyword_argument_rejected(tmp_path: Path):
    with pytest.raises(CompilationError, match="positional argument cannot follow keyword"):
        _lower(
            tmp_path,
            """module m
contains
  pure subroutine h(x, y)
    real(8), intent(out) :: x
    real(8), intent(in) :: y
    x = y + 1.0d0
  end subroutine
  subroutine k(a, b)
    real(8), intent(out) :: a(:)
    real(8), intent(in) :: b(:)
    call h(y=b(1), a(1))
  end subroutine
end module
""",
        )


def test_empty_use_only_is_not_wildcard(tmp_path: Path):
    p = tmp_path / "u.f90"
    p.write_text(
        """module m1
  integer :: g
end module
module m2
  use m1, only:
  use, intrinsic :: ieee_exceptions, only:
contains
  subroutine s()
  end subroutine
end module
"""
    )
    se = SourceEffects([p])
    mod = se.modules["m2"]
    assert "m1" not in mod.wildcards
    assert "ieee_exceptions" not in mod.intrinsic_wildcards


def test_unsupported_associate_does_not_leak_enclosing_scope(tmp_path: Path):
    p = tmp_path / "assoc.f90"
    p.write_text(
        """module m
contains
  subroutine s(x, y)
    real(8), intent(inout) :: x, y
    associate (x => y + 1.0d0)
      y = x
    end associate
  end subroutine
end module
"""
    )
    se = SourceEffects([p])
    routine = se.routines["m::s"]
    register_associates(se, routine)
    # Inside the unsupported ASSOCIATE block, no enclosing scope should be registered for the assignment.
    assoc = routine.execution.content[0]
    inner_assign = next(c for c in assoc.content if type(c).__name__ == "Assignment_Stmt")
    assert source_scope_for(se, inner_assign, None) is None


# --- 3. Resident passes & scopes ---


def test_specialize_fold_does_not_duplicate_inline_comments_or_semicolons():
    src = """module m
contains
  subroutine s(flag, x)
    integer, intent(in) :: flag
    real, intent(inout) :: x
    if (flag == 1) then ! comment on if
      x = x + 1.0 ! increment x
    end if
    x = x + 2.0; x = x + 3.0
  end subroutine
end module
"""
    out = specialize(src, ["m::s"], ("flag == 1",), {})
    then_part = out[out.index("if ((flag == 1)) then"):out.index("else\n")]
    assert then_part.count("x = x + 1.0") == 1
    assert then_part.count("x = x + 2.0; x = x + 3.0") == 1
    assert "if (flag == 1) then ! comment on if" not in then_part


def test_specialize_does_not_fold_runtime_dummy_shadowing_global_parameter():
    src = """module m1
  integer, parameter :: mode = 1
end module
module m2
contains
  subroutine step(flag, mode, x)
    integer, intent(in) :: flag, mode
    real, intent(inout) :: x
    if (flag == 1) then
      if (mode == 1) then
        x = 1.0
      else
        x = 2.0
      end if
    end if
  end subroutine
end module
"""
    constants = integer_parameters([src])
    out = specialize(src, ["m2::step"], ("flag == 1",), constants)
    then_part = out[out.index("if ((flag == 1)) then"):out.index("\n    else\n")]
    assert "if (mode == 1) then" in then_part
    assert "x = 2.0" in then_part


def test_declarations_of_preserves_attributes_after_dimension_and_renamed_lines_keeps_dot_ops():
    src = """module m
contains
  subroutine s(a, p)
    real, dimension(:), allocatable, intent(inout) :: a
    real, dimension(:), pointer, intent(inout) :: p
  end subroutine
end module
"""
    scope = next(procedures_of(parse_fortran(src)))
    decls = declarations_of(scope.node)
    assert "allocatable" in decls["a"][1]
    assert decls["a"][2] == ":"
    assert "pointer" in decls["p"][1]
    assert decls["p"][2] == ":"

    lines = ["  if (a .ne. b) ne = ne + 1\n"]
    renamed = renamed_lines(lines, {"ne": "fort_inl1_ne"})
    assert renamed == ["  if (a .ne. b) fort_inl1_ne = fort_inl1_ne + 1\n"]


def test_lower_array_statements_rejects_scalar_element_self_aliasing():
    src = """module m
contains
  subroutine s(a)
    real, intent(inout) :: a(:)
    a = a / a(1)
  end subroutine
end module
"""
    out, records = lower_array_statements(src, ["m::s"])
    assert out == src
    assert records
    assert "scalar elements" in records[0]["reason"]


def test_inlining_preserves_real_conversion_and_one_line_if_and_external_callers():
    src = """module m
contains
  subroutine callee(a, n)
    real, intent(inout) :: a(:)
    integer, intent(in) :: n
    real :: x
    integer :: m
    x = n
    m = 1
    if (n > 0) m = 2
    a(m) = x / 2
  end subroutine
  subroutine caller(a, n)
    real, intent(inout) :: a(:)
    integer, intent(in) :: n
    call callee(a, n)
  end subroutine
end module
"""
    out, _records = inline_calls(src, ["m::caller"], ["m::callee"])
    assert "1 = 2" not in out
    assert "n / 2" not in out

    # External caller has empty scope.hosts and must not raise IndexError.
    ext = """subroutine ext(a)
  real, intent(inout) :: a(:)
  a(1) = 1.0
end subroutine
"""
    assert inline_calls(ext, ["$external::ext"], ["m::callee"]) == (ext, [])


def test_reductions_do_not_drop_intent_out_dummy():
    src = """module m
contains
  subroutine calc(a, n, s, last_val)
    real, intent(in) :: a(:)
    integer, intent(in) :: n
    real, intent(inout) :: s
    real, intent(out) :: last_val
    integer :: i
    do i = 1, n
      last_val = a(i)
      s = max(s, last_val)
    end do
  end subroutine
end module
"""
    out, records = lower_reductions(src, ["m::calc"])
    assert out == src
    assert records
    assert "not local to the procedure" in records[0]["reason"]


def test_element_pointer_skips_nested_reuse_and_select_case_deduplicates_semicolons():
    src = """module m
  type :: pt
    real :: v
  end type
contains
  subroutine s(a, b, res, n, sel)
    type(pt), target, intent(in) :: a(:), b(:)
    real, intent(inout) :: res(:)
    integer, intent(in) :: n, sel
    type(pt), pointer :: p
    integer :: i, j
    do i = 1, n
      p => a(i)
      res(i) = p%v
      do j = 1, n
        p => b(j)
        res(i) = res(i) + p%v
      end do
    end do
    select case (sel)
    case (1)
      res(1) = 1.0; res(2) = 2.0
    end select
  end subroutine
end module
"""
    lines = src.splitlines(keepends=True)
    scope = next(procedures_of(parse_fortran(src)))
    assert _element_pointer(scope, lines) is None
    _first, _last, case_lines = _select_case(scope, lines)
    assert "".join(case_lines).count("res(1) = 1.0; res(2) = 2.0") == 1


def test_declared_pure_rejects_impure_elemental(tmp_path: Path):
    p = tmp_path / "imp.f90"
    p.write_text(
        """module m
contains
  impure elemental subroutine f(x)
    real(8), intent(inout) :: x
    x = x + 1.0d0
  end subroutine
end module
"""
    )
    se = SourceEffects([p])
    assert _declared_pure(se.routines["m::f"]) is False


# --- 4. Emission, offload & build ---


def test_c_family_unary_negative_literal_parenthesized():
    rendered = render_expression(Unary("-", Literal("-5", ScalarType.INTEGER)))
    assert "--" not in rendered
    assert rendered == "(-(-5))"


def test_scoped_entry_fixes_cpu_worker_invariants_exact_false_and_wide_assumptions(tmp_path: Path):
    # 1) Guarded stencil inside If (exact=False) clamps/falls back rather than rejecting with FORT_SCOPE_ARGUMENT.
    fn_guard = _lower(
        tmp_path,
        """module m
contains
  subroutine k(a, b, n)
    real(8), intent(in) :: a(:)
    real(8), intent(out) :: b(:)
    integer, intent(in) :: n
    integer :: i
    do i = 1, n
      if (i > 1) b(i) = a(i - 1)
    end do
  end subroutine
end module
""",
    )
    plan_guard = build_execution_plan(fn_guard)
    entry = generate_scoped(fn_guard, plan_guard, OffloadConfig(), "common.cuh")
    assert "fallback" in entry.cuda
    assert "std::size_t fort_internal_total, int tid, int team" in entry.cuda
    assert "flat += team" not in entry.cuda
    assert "#ifdef _OPENMP\n#include <omp.h>\n#endif" in entry.cuda

    # 2) HostBlock integer local overwritten inside ParallelRegion is not treated as invariant.
    fn_inv = _lower(
        tmp_path,
        """module m
contains
  subroutine k(a, b, idx, n)
    real(8), intent(in) :: a(:)
    real(8), intent(out) :: b(:)
    integer, intent(in) :: idx(:), n
    integer :: i, k1
    k1 = 1
    do i = 1, n
      k1 = idx(i)
      b(i) = a(k1)
    end do
  end subroutine
end module
""",
    )
    plan_inv = build_execution_plan(fn_inv)
    entry_inv = generate_scoped(fn_inv, plan_inv, OffloadConfig(), "common.cuh")
    assert "FORT_SCOPE_READ_ALL" in entry_inv.cuda

    # 3) Non-safe loop bound intrinsic (abs(n)) falls back to full footprint without crashing.
    fn_abs = _lower(
        tmp_path,
        """module m
contains
  subroutine k(a, n)
    real(8), intent(out) :: a(:)
    integer, intent(in) :: n
    integer :: i
    do i = 1, abs(n)
      a(i) = 1.0d0
    end do
  end subroutine
end module
""",
    )
    plan_abs = build_execution_plan(fn_abs)
    unit_abs = _unit_footprints(Unit(0, plan_abs.regions[0], ()), frozenset(fn_abs.parameters))
    assert unit_abs.footprints[0].full_write is True


def test_cmake_lists_quotes_source_options_and_keeps_single_file_links(tmp_path: Path):
    src = tmp_path / "main.f90"
    src.write_text("program p\nend program\n")
    text = cmake_lists(
        "app",
        (ObjectLibrary("single", (), link=("mylib",)),),
        (Executable("app", src, "single"),),
        include_directories=(tmp_path,),
        source_options={src: ("-O0", "-fno-fast-math")},
    )
    assert 'COMPILE_OPTIONS "-O0;-fno-fast-math"' in text
    assert "target_link_libraries(app PRIVATE OpenMP::OpenMP_Fortran mylib)" in text
    assert "target_include_directories(app PRIVATE" in text
    with pytest.raises(ValueError, match="Unsupported CMake path"):
        cmake_path(Path("/tmp/bad\\path"))
