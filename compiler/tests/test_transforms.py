"""Check fusion legality rather than particular emitted string formatting."""

from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend import lower_file
from compiler.transforms import optimize_function

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(("case", "filename"), [("CDU", "cdu.f90"), ("CDV", "cvd.f90"), ("CDW", "cdw.f90")])
def test_stencil_scalar_motion_and_fusion(case: str, filename: str) -> None:
    function = lower_file(ROOT / "benchmarks" / "cases" / case / "Fortran" / filename, case)
    original, baseline = optimize_function(function, options=CompilerOptions(opt_level=0))
    optimized, plan = optimize_function(function)
    assert original is function
    assert len(baseline.regions) == 4
    assert len(plan.regions) == 1
    assert optimized.parameters == function.parameters
    assert any("moved 3 scalar" in report for report in plan.reports)


def _function(tmp_path: Path, first: str, second: str, *, middle="", header="1, n", setup="factor=2.0_knd"):
    source = tmp_path / "fusion.f90"
    source.write_text(f"""! kernels
module fusion_cases
  integer, parameter :: knd=kind(1.0d0)
contains
  ! kernel
  subroutine entry(a,b,n,stride)
    real(knd), intent(inout) :: a(:),b(:)
    integer, intent(in) :: n,stride
    integer :: i
    real(knd) :: factor
    {setup}
    do i={header}
      {first}
    end do
    {middle}
    do i={header}
      {second}
    end do
  end subroutine
end module
""")
    return lower_file(source, "entry")


@pytest.mark.parametrize("header", ["1, n", "n, 1, -1", "1, n, 2", "1, n, 1+1", "n, 1, -(4/2)"])
def test_same_point_dependencies_preserve_order(tmp_path: Path, header: str) -> None:
    function = _function(tmp_path, "a(i)=i", "b(i)=a(i)+factor", header=header)
    _, plan = optimize_function(function)
    assert len(plan.regions) == 1


def test_explicit_scoped_pipeline_honors_requested_optimization(tmp_path):
    from compiler.driver.pipeline import prepare_function

    function = _function(tmp_path, "a(i)=i", "b(i)=a(i)+factor")
    _, source = prepare_function(function, options=CompilerOptions(
        memory_model="scoped", gpu_policy="sections", opt_level=0))
    _, optimized = prepare_function(function, options=CompilerOptions(
        memory_model="scoped", gpu_policy="sections", opt_level=1))
    assert len(source.regions) == 2
    assert len(optimized.regions) == 1


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("a(i)=i", "b(i)=a(i-1)"),
        ("b(i)=a(i+1)", "a(i)=i"),
        ("a(i)=i", "a(i+1)=2*i"),
    ],
    ids=["RAW", "WAR", "WAW"],
)
def test_cross_pass_dependencies_prevent_fusion(tmp_path: Path, first: str, second: str) -> None:
    function = _function(tmp_path, first, second, header="2, n")
    unchanged, plan = optimize_function(function)
    assert unchanged is function
    assert len(plan.regions) == 2
    assert any("skipped" in report for report in plan.reports)


@pytest.mark.parametrize("middle", ["factor=3.0_knd", "factor=a(1)", "b(1)=3.0_knd"])
def test_unsafe_setup_motion_is_not_performed(tmp_path: Path, middle: str) -> None:
    function = _function(tmp_path, "a(i)=factor", "b(i)=a(i)", middle=middle)
    _, plan = optimize_function(function)
    assert len(plan.regions) == 2


def test_runtime_strides_retain_original_regions(tmp_path: Path) -> None:
    function = _function(tmp_path, "a(i)=i", "b(i)=a(i)", header="1,n,stride")
    _, plan = optimize_function(function)
    assert len(plan.regions) == 2


def test_unequal_domains_retain_original_regions(tmp_path: Path) -> None:
    from dataclasses import replace

    from compiler.ir import Block, Literal, ScalarType

    function = _function(tmp_path, "a(i)=i", "b(i)=a(i)")
    setup, first, second = function.body.statements
    second = replace(second, lower=Literal("2", ScalarType.INTEGER))
    function = replace(function, body=Block((setup, first, second)))
    unchanged, plan = optimize_function(function)
    assert unchanged is function
    assert len(plan.regions) == 2
    assert any("different domains" in report for report in plan.reports)


@pytest.mark.parametrize("second", ["b(j+3)=j", "b(j+3)=a(j+3)"])
def test_constant_width_shifted_domains_preserve_source_coordinates(tmp_path, second):
    source = tmp_path / "shifted.f90"
    source.write_text(f"""module shifted
contains
subroutine entry(a,b,n)
integer,intent(inout)::a(:),b(:)
integer,intent(in)::n
integer::i,j
do i=n+1,n+3
  a(i-n)=i
enddo
do j=-2,0
  {second}
enddo
end subroutine
end module
""")
    function = lower_file(source, "entry")
    _, baseline = optimize_function(function, options=CompilerOptions(opt_level=0))
    transformed, plan = optimize_function(function)
    assert len(baseline.regions) == 2
    assert len(plan.regions) == 1
    # Re-run the ordinary semantic/dependency checker on the translated body.
    from compiler.analysis import build_execution_plan
    assert len(build_execution_plan(transformed).regions) == 1


@pytest.mark.parametrize("second", ["b(j+3)=a(j+2)", "a(j+4)=j"])
def test_shifted_domain_cross_point_dependencies_still_prevent_fusion(tmp_path, second):
    source = tmp_path / "shifted_conflict.f90"
    source.write_text(f"""module shifted_conflict
contains
subroutine entry(a,b,n)
integer,intent(inout)::a(:),b(:)
integer,intent(in)::n
integer::i,j
do i=n+1,n+3
  a(i-n)=i
enddo
do j=-2,0
  {second}
enddo
end subroutine
end module
""")
    function = lower_file(source, "entry")
    unchanged, plan = optimize_function(function)
    assert unchanged is function
    assert len(plan.regions) == 2


def test_conservative_cross_pass_accesses_prevent_fusion(tmp_path: Path) -> None:
    function = _function(tmp_path, "a(i)=i", "b(i)=a(i*i+i)")
    unchanged, plan = optimize_function(function)
    assert unchanged is function
    assert len(plan.regions) == 2
    assert any(region.report.conservative for region in plan.regions)
    assert any("skipped" in report for report in plan.reports)


def test_hoisted_definitions_retain_their_order(tmp_path: Path) -> None:
    function = _function(tmp_path, "a(i)=i", "b(i)=a(i)*factor", middle="factor=3.0_knd\n    factor=factor+2.0_knd")
    _, plan = optimize_function(function)
    assert len(plan.regions) == 1
    assert any("moved 2 scalar" in report for report in plan.reports)


def test_host_branch_fusion_stays_inside_each_branch_and_preserves_surrounding_loops(tmp_path: Path) -> None:
    from compiler.ir import Assignment, ConditionalRegion, If

    source = tmp_path / "branch_fusion.f90"
    source.write_text("""! kernels
module branch_fusion
contains
  ! kernel
  subroutine entry(a,b,n,flag)
    integer, intent(inout) :: a(:),b(:)
    integer, intent(in) :: n
    logical, intent(in) :: flag
    integer :: i,factor
    do i=1,n
      a(i)=a(i)+1
    end do
    if (flag) then
      do i=1,n
        a(i)=i
      end do
      factor=3
      do i=1,n
        b(i)=a(i)*factor
      end do
    else if (n>2) then
      do i=1,n
        a(i)=2*i
      end do
      factor=5
      do i=1,n
        b(i)=a(i)*factor
      end do
    else
      do i=1,n
        a(i)=3*i
      end do
      do i=1,n
        b(i)=a(i)
      end do
    end if
    do i=1,n
      b(i)=b(i)+1
    end do
  end subroutine
end module
""")
    function = lower_file(source, "entry")
    _, baseline = optimize_function(function, options=CompilerOptions(opt_level=0))
    optimized, plan = optimize_function(function)
    assert len(baseline.regions) == 8
    assert len(plan.regions) == 5
    assert optimized.body.statements[0] is function.body.statements[0]
    assert optimized.body.statements[-1] is function.body.statements[-1]
    conditional = optimized.body.statements[1]
    assert isinstance(conditional, If)
    assert isinstance(conditional.then_body.statements[0], Assignment)
    assert len(conditional.then_body.statements) == 2
    host_condition = plan.steps[1]
    assert isinstance(host_condition, ConditionalRegion)
    assert len(host_condition.then_plan.regions) == 1
    assert len(host_condition.else_plan.regions) == 2


def test_conflicting_branch_passes_remain_separate(tmp_path: Path) -> None:
    function = _function(tmp_path, "a(i)=i", "b(i)=a(i-1)", header="2,n")
    from dataclasses import replace

    from compiler.ir import Binary, Block, If, Literal, Reference, ScalarType

    n = next(symbol for symbol in function.parameters if symbol.name == "n")
    loops = function.body.statements[1:]
    conditional = If(
        Binary(">", Reference(n), Literal("0", ScalarType.INTEGER)),
        Block(loops),
        Block(()),
        loops[0].location,
    )
    function = replace(function, body=Block((function.body.statements[0], conditional)))
    unchanged, plan = optimize_function(function)
    assert unchanged is function
    assert len(plan.regions) == 2
    assert any("skipped" in report for report in plan.reports)


def test_branch_fusion_keeps_outer_liveness_checks(tmp_path: Path) -> None:
    from compiler.ir import ConditionalRegion, SequentialRegion

    source = tmp_path / "branch_liveout.f90"
    source.write_text("""! kernels
module branch_liveout
contains
  ! kernel
  subroutine entry(a,b,n,flag)
    integer, intent(inout) :: a(:),b(:)
    integer, intent(in) :: n
    logical, intent(in) :: flag
    integer :: i
    i=0
    if (flag) then
      do i=1,n
        a(i)=i
      end do
      do i=1,n
        b(i)=a(i)
      end do
    end if
    b(1)=i
  end subroutine
end module
""")
    function = lower_file(source, "entry")
    unchanged, plan = optimize_function(function, options=CompilerOptions(fallback="host"))
    assert unchanged is function
    conditional = next(step for step in plan.steps if isinstance(step, ConditionalRegion))
    assert isinstance(conditional.then_plan.steps[-1], SequentialRegion)
    assert "live after" in conditional.then_plan.steps[-1].reason


def test_fusion_does_not_capture_retained_serial_iterator(tmp_path: Path) -> None:
    source = tmp_path / "iterator_capture.f90"
    source.write_text("""! kernels
module iterator_capture
contains
  ! kernel
  subroutine entry(a,b,n)
    integer, intent(inout) :: a(:,:),b(:,:)
    integer, intent(in) :: n
    integer :: i,j
    do i=1,n
      a(i,1)=i
    end do
    do j=1,n
      b(j,1)=1
      do i=1,2
        b(j,i)=2
      end do
    end do
  end subroutine
end module
""")
    function = lower_file(source, "entry")
    unchanged, plan = optimize_function(function)
    assert unchanged is function
    assert len(plan.regions) == 2
    assert any("mapped iterator conflicts" in report for report in plan.reports)


@pytest.mark.parametrize(("first", "second"), [("i,n", "j,n"), ("1,i", "1,j")])
def test_header_induction_inputs_are_not_unified(tmp_path: Path, first: str, second: str) -> None:
    source = tmp_path / "header_inputs.f90"
    source.write_text(f"""! kernels
module header_inputs
contains
  ! kernel
  subroutine entry(a,b,n)
    integer, intent(inout) :: a(:),b(:)
    integer, intent(in) :: n
    integer :: i,j
    i=1
    j=3
    do i={first}
      a(i)=i
    end do
    do j={second}
      b(j)=j
    end do
  end subroutine
end module
""")
    function = lower_file(source, "entry")
    unchanged, plan = optimize_function(function)
    assert unchanged is function
    assert len(plan.regions) == 2
    assert any("iterator/array-valued bounds" in report for report in plan.reports)


def _private_scalar_function(tmp_path, first, second, *, tail=""):
    path = tmp_path / "reuse.f90"
    path.write_text(f"""module reuse
contains
subroutine entry(a,b,n)
integer,intent(in)::n
real(8),intent(inout)::a(:),b(:)
real(8)::t
integer::i
t=0
do i=1,n
 {first}
enddo
do i=1,n
 {second}
enddo
{tail}
end subroutine
end module
""")
    return lower_file(path, "entry")


@pytest.mark.parametrize("guarded", [False, True])
def test_reused_definite_private_temporary_can_fuse(tmp_path, guarded):
    first = "t=a(i)+1; a(i)=t"
    second = "t=b(i)+2; b(i)=t+a(i)"
    if guarded:
        first = "if(a(i)>0)then\n" + first + "\nendif"
        second = "if(b(i)>0)then\n" + second + "\nendif"
    function = _private_scalar_function(tmp_path, first, second)
    _, original = optimize_function(function, options=CompilerOptions(opt_level=0))
    _, fused = optimize_function(function)
    assert len(original.regions) == 2 and len(fused.regions) == 1


def test_private_temporary_does_not_hide_array_dependence(tmp_path):
    function = _private_scalar_function(tmp_path, "t=a(i)+1; a(i)=t", "t=b(i)+2; b(i)=t+a(i+1)")
    _, plan = optimize_function(function)
    assert len(plan.regions) == 2


@pytest.mark.parametrize(("second", "tail"), [
    ("b(i)=t", ""),
    ("t=b(i)+2; b(i)=t", "b(1)=t"),
    ("if(b(i)>0) t=b(i)+2\n b(i)=t", ""),
])
def test_escaping_or_carried_scalar_does_not_become_private(tmp_path, second, tail):
    from compiler.ir import SequentialRegion

    function = _private_scalar_function(tmp_path, "t=a(i)+1; a(i)=t", second, tail=tail)
    unchanged, plan = optimize_function(function, options=CompilerOptions(fallback="host"))
    assert unchanged is function
    reasons = [step.reason for step in plan.steps if isinstance(step, SequentialRegion)]
    assert any("live after" in reason or "before its per-iteration definition" in reason for reason in reasons)
