"""Regression tests for soundness, coherence, and build fixes from the compiler audit."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from compiler.analysis import ParallelizationError, build_execution_plan
from compiler.emission.common.c_family import render_expression
from compiler.emission.cuda.scoped import generate_scoped
from compiler.frontend import lower_file
from compiler.frontend.component_bindings import register_associates, source_scope_for
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.syntax import parse_fortran
from compiler.ir import Assignment, CompilationError, Literal, Reference, ScalarType, Unary, format_ir
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
from compiler.scopes.resident_entries import export_resident_entries
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


def test_pure_copy_back_goes_to_the_element_passed_when_the_call_changes_its_subscript(tmp_path: Path):
    fn = _lower(
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
    # The element is the one idx(1) selected when the call began, as in Fortran.
    ir = format_ir(fn)
    taken = re.search(r"(fort_v\d+_fort_idx_\d+) = fort_v\d+_idx\(1\)", ir)
    assert taken
    assert re.search(rf"fort_v\d+_a\({taken.group(1)}\) = ", ir)


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
    # Inside the unsupported ASSOCIATE block the alias names nothing: not the enclosing scope's x.
    assoc = routine.execution.content[0]
    inner_assign = next(c for c in assoc.content if type(c).__name__ == "Assignment_Stmt")
    inner = source_scope_for(se, inner_assign, None)
    assert inner is not None
    assert se._binding(inner, "x") is None
    assert se._binding(inner, "y") is se._binding(routine.scope, "y")


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


def test_renamed_lines_renames_names_next_to_dotted_operators_and_keeps_numbers():
    # A name touches an operator written without spaces; an exponent and a kind are parts of their numbers.
    lines = ["  if (n.eq.1.and.k.lt.n) x = 1.e+5 * e + 2.5d-3 + 1_k\n", "  if (k.and..not.n) n = s%n\n"]
    assert renamed_lines(lines, {"n": "nx", "k": "kk", "e": "ee"}) == [
        "  if (nx.eq.1.and.kk.lt.nx) x = 1.e+5 * ee + 2.5d-3 + 1_k\n", "  if (kk.and..not.nx) nx = s%n\n"]


def test_an_inlined_callee_reads_its_argument_where_it_compares_without_spaces():
    src = """module m
contains
  subroutine callee(a, n)
    real, intent(inout) :: a(:)
    integer, intent(in) :: n
    if (n.gt.0) a(1) = 1.0
  end subroutine
  subroutine caller(a, nx, n)
    real, intent(inout) :: a(:)
    integer, intent(in) :: nx, n
    call callee(a, nx)
  end subroutine
end module
"""
    out, records = inline_calls(src, ["m::caller"], ["m::callee"], {})
    assert [record["inlined"] for record in records] == [True]
    assert "if (nx.gt.0) a(1) = 1.0" in out


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


# --- 5. What the first pass left: regressions and incomplete fixes ---


def test_specialize_writes_each_line_once_and_refuses_a_decided_if_that_shares_one():
    src = """module m
contains
  subroutine s(flag, x, y, n)
    integer, intent(in) :: flag, n
    real, intent(inout) :: x, y
    integer :: i
    do i = 1, n
      if (flag == 1) then
        y = y + 1.0
      end if
      x = x + 2.0; y = y * 3.0
    end do
    if (flag == 1) then
      x = x + 1.0
    else
      x = x - 1.0
    end if ! done
    x = x + &
        1.0 ! continued
  end subroutine
end module
"""
    out = specialize(src, ["m::s"], ("flag == 1",), {})
    version = out[out.index("if ((flag == 1)) then"):out.index("\n    else\n")]
    # The second statement of a line in a loop, the comment after a folded END IF and the last line of a
    # continued statement are written with what they belong to, once.
    assert version.count("x = x + 2.0; y = y * 3.0") == 1
    assert "end if" not in version
    assert version.count("1.0 ! continued") == 1
    with pytest.raises(ValueError, match="shares a line"):
        specialize(src.replace("    if (flag == 1) then\n      x = x + 1.0\n", "    if (flag == 1) then; x = x + 1.0\n"),
                   ["m::s"], ("flag == 1",), {})


def test_select_case_writes_a_continued_statement_once_and_leaves_a_case_that_shares_its_line():
    src = """module m
contains
  subroutine s(k, x)
    integer, intent(in) :: k
    real, intent(inout) :: x
    select case (k)
    case (1)
      x = x + &
          1.0 ! continued
    case default
      x = 0.0
    end select
  end subroutine
end module
"""
    _first, _last, out = _select_case(next(procedures_of(parse_fortran(src))), src.splitlines(keepends=True))
    assert "".join(out).count("1.0 ! continued") == 1
    shared = src.replace("    case (1)\n      x = x + &\n", "    case (1); x = x + &\n")
    assert _select_case(next(procedures_of(parse_fortran(shared))), shared.splitlines(keepends=True)) is None


def test_specialize_does_not_fold_a_host_variable_that_shadows_a_parameter():
    src = """module m2
  integer :: mode
contains
  subroutine step(flag, x)
    integer, intent(in) :: flag
    real, intent(inout) :: x
    if (flag == 1) then
      if (mode == 1) then
        x = 2.0
      end if
    end if
  end subroutine
end module
"""
    out = specialize(src, ["m2::step"], ("flag == 1",), {"mode": 1})
    assert "if (mode == 1) then" in out[out.index("if ((flag == 1)) then"):out.index("\n    else\n")]


def test_inlining_keeps_an_integer_local_whose_value_the_assignment_converts():
    src = """module m
contains
  subroutine callee(a, x)
    real, intent(inout) :: a(:)
    real, intent(in) :: x
    integer :: k
    k = x * 2
    a(1) = k * 0.5
  end subroutine
  subroutine caller(a, x)
    real, intent(inout) :: a(:)
    real, intent(in) :: x
    call callee(a, x)
  end subroutine
end module
"""
    out, records = inline_calls(src, ["m::caller"], ["m::callee"], {})
    assert [record["inlined"] for record in records] == [True]
    # k holds x * 2 truncated: substituted, a(1) would be x itself.
    assert re.search(r"(\w+) = x \* 2\n\s*a\(1\) = \1 \* 0\.5", out)


def test_lower_array_statements_rejects_an_element_of_the_target_inside_a_subscript():
    src = """module m
contains
  subroutine s(x, y, c)
    real, intent(inout) :: x(:)
    real, intent(in) :: y(:), c(:)
    x = y * c(nint(x(1)))
  end subroutine
end module
"""
    out, records = lower_array_statements(src, ["m::s"])
    assert out == src
    assert "scalar elements of x" in records[0]["reason"]


def test_element_pointer_leaves_a_pointer_that_is_the_function_result():
    src = """module m
  type :: pt
    real :: v
  end type
contains
  function first_pos(a, n) result(p)
    type(pt), target, intent(in) :: a(:)
    integer, intent(in) :: n
    type(pt), pointer :: p
    integer :: i
    do i = 1, n
      p => a(i)
      if (p%v > 0) exit
    end do
  end function
end module
"""
    assert _element_pointer(next(procedures_of(parse_fortran(src))), src.splitlines(keepends=True)) is None


def test_a_dummy_without_intent_that_only_reads_a_read_only_private_array_keeps_its_nest_parallel(tmp_path: Path):
    source = """module m
contains
  subroutine reader(x, y, j)
    real(8) :: x(3)
    real(8) :: y(:)
    integer :: j
    y(j) = x(1) + x(2) + x(3)
  end subroutine
  subroutine middle(p, r, j)
    real(8), intent(in) :: p(3)
    real(8), intent(inout) :: r(:)
    integer, intent(in) :: j
    call reader(p, r, j)
  end subroutine
  subroutine k(a, n)
    real(8), intent(inout) :: a(:)
    integer, intent(in) :: n
    real(8) :: v(3)
    integer :: i
    do i = 1, n
      v(1) = a(i)
      v(2) = 2.0d0
      v(3) = 3.0d0
      call middle(v, a, i)
    end do
  end subroutine
end module
"""
    plan = build_execution_plan(_lower(tmp_path, source))
    assert len(plan.regions) == 1


def test_a_copied_element_goes_back_to_the_element_the_call_was_given(tmp_path: Path):
    fn = _lower(tmp_path, """module m
contains
  pure subroutine set2(j, x)
    integer, intent(inout) :: j
    real(8), intent(out) :: x
    j = 2
    x = 42.0d0
  end subroutine
  subroutine k(a, idx)
    real(8), intent(inout) :: a(:)
    integer, intent(inout) :: idx(:)
    call set2(idx(1), a(idx(1)))
  end subroutine
end module
""")
    ir = format_ir(fn)
    # idx(1) goes back first; a's element is the one idx(1) selected before the call.
    taken = re.search(r"(fort_v\d+_fort_idx_\d+) = fort_v\d+_idx\(1\)", ir)
    assert taken
    assert re.search(rf"fort_v\d+_a\({taken.group(1)}\) = ", ir)


def test_scoped_footprints_do_not_take_a_loop_variable_from_before_the_loop(tmp_path: Path):
    fn = _lower(tmp_path, """module m
  integer, parameter :: knd = kind(1.0d0)
contains
  subroutine k(a, b, n)
    real(knd), contiguous, intent(in) :: a(:)
    real(knd), contiguous, intent(inout) :: b(:)
    integer, intent(in) :: n
    integer :: i
    i = 3
    b(1) = a(i)
    do i = 1, n
      b(i) = a(i)
    end do
  end subroutine
end module
""")
    entry = generate_scoped(fn, build_execution_plan(fn), OffloadConfig(), "common.cuh")
    # The region reads and writes elements 1 to n, not the element i = 3 the host left.
    assert not re.search(r"offload::index\(static_cast<long double>\(fort_v\d+_i\)", entry.cuda)


def test_a_nest_inside_an_unsupported_associate_is_not_exported_as_the_dummy_it_shadows(tmp_path: Path):
    source = tmp_path / "assoc.f90"
    source.write_text("""module m
contains
  subroutine s(a, x, n)
    real(8), intent(inout) :: a(:)
    real(8), intent(inout) :: x(:)
    integer, intent(in) :: n
    integer :: i
    associate (x => a(1:n))
      do i = 1, n
        x(i) = 2.0d0
      end do
    end associate
  end subroutine
end module
""")
    exported = export_resident_entries([source], ["m::s"], tmp_path / "out")
    assert not exported["entries"]
    assert len(exported["rejected"]) == 1



def test_a_subscript_divided_by_a_literal_is_one_value_the_entry_checks(tmp_path: Path):
    source = """module m
contains
  subroutine k(a, n, m)
    real(8), intent(inout) :: a(:)
    integer, intent(in) :: n, m
    integer :: i
    do i = 1, n
      a(i) = a(m / 2) + 1.0d0
    end do
  end subroutine
end module
"""
    # m / 2 cannot trap: the nest is independent where it reads no element it writes, and checks that.
    fn = _lower(tmp_path, source)
    plan = build_execution_plan(fn)
    region, = plan.regions
    assert region.assumptions
    # The entry checks it in 64-bit arithmetic, with the division Fortran's.
    condition = next(line for line in generate_scoped(fn, plan, OffloadConfig(), "common.cuh").cuda.splitlines()
                     if "return FORT_SCOPE_DECLINED;" in line and "if (!" in line)
    assert re.search(r"static_cast<long long>\(fort_v\d+_m\) / 2LL", condition)
    # A divisor that may be zero stays a value the analysis does not know.
    with pytest.raises(ParallelizationError):
        build_execution_plan(_lower(tmp_path, source.replace("m / 2", "2 / m")))


def test_a_call_that_may_change_a_component_array_is_a_host_write_of_it(tmp_path: Path):
    source = tmp_path / "flows.f90"
    source.write_text("""module flows
type state
  real(8) :: u(100)
end type
type(state) :: flow
integer :: n
contains
subroutine fill(u)
  real(8) :: u(:)
  u = 1.0d0
end subroutine
subroutine step()
  integer :: i
  do i = 1, n
    flow%u(i) = 2 * flow%u(i)
  end do
  call fill(flow%u)
  do i = 1, n
    flow%u(i) = flow%u(i) + 1
  end do
end subroutine
end module
""")
    exported = export_resident_entries([source], ["flows::step"], tmp_path / "out")
    assert len(exported["entries"]) == 2
    call, = (item for item in exported["host_statements"] if item["kind"] == "Call_Stmt")
    # fill does not take its argument INTENT(IN): the device copy of flow%u must not outlive the call.
    assert [(row["name"], row["access"]) for row in call["arrays"]] == [("flow%u", "write")]
