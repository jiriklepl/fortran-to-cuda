"""Adjacent regions that are not known to depend on each other share one queued launch."""

import re

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission.cuda.scoped import generate_scoped
from compiler.frontend import lower_source
from compiler.offload.config import OffloadConfig


def emit(body, *, declarations="real(8),intent(inout)::u(:),v(:),w(:)\nreal(8),intent(in)::x(:)"):
    source = f"""module faces
contains
subroutine update(u,v,w,x,n)
{declarations}
integer,intent(in)::n
integer::i
{body}
end subroutine
end module
"""
    function, plan = prepare_function(lower_source(source, "update", source_name="faces.f90"),
                                      options=CompilerOptions())
    return plan, generate_scoped(function, plan, OffloadConfig("sections"), "common_functions.cuh")


def loop(target, value, upper):
    return f"do i=1,{upper}\n{target}(i)={value}\nenddo"


def groups(emitted):
    return [group["regions"] for group in emitted.report["graph_execution"]["groups"] if len(group["regions"]) > 1]


def test_independent_regions_of_different_extent_share_one_launch():
    # Different trip counts keep loop fusion away; the arrays are distinct.
    plan, emitted = emit("\n".join((loop("u", "2*u(i)", "n"), loop("v", "x(i)+v(i)", "n-1"), loop("w", "x(i)", "n-2"))))
    assert len(plan.regions) == 3
    assert groups(emitted) == [[0, 1, 2]]
    cuda = emitted.cuda
    assert cuda.count("static __global__ void kernel_group_0_1_2_device(") == 1
    # Each member keeps its own kernel for the one-region-per-launch path.
    for region in range(3):
        assert f"static __global__ void kernel_region_{region}_device(" in cuda
        assert f"kernel_region_{region}_points(" in cuda
    assert cuda.count("fort_scope_graph_enqueue_members_v1(") == 1
    # The group launch passes every member's arguments, in member order.
    kernel = cuda.split("static __global__ void kernel_group_0_1_2_device(", 1)[1].split(") {", 1)
    parameters = [line.strip().rstrip(",") for line in kernel[0].strip().splitlines()]
    assert [name.rsplit(" ", 1)[1].split("_")[1] for name in parameters] == sorted(
        name.rsplit(" ", 1)[1].split("_")[1] for name in parameters)
    count = int(re.search(r"fort_graph_kernel\.argument_count = (\d+);", cuda).group(1))
    assert count == len(parameters)
    # Every thread walks each member from the same starting point.
    calls = re.findall(r"kernel_region_(\d)_points\(.*fort_internal_grid_stride, fort_internal_point\);", kernel[1])
    assert calls == ["0", "1", "2"]


@pytest.mark.parametrize("body", [
    # The second region reads what the first writes.
    "\n".join((loop("u", "x(i)", "n"), loop("v", "u(i)", "n-1"))),
    # Both write the same array, and the second section lies inside the first whenever both loops run.
    "\n".join((loop("u", "x(i)", "n"), "do i=2,n-1\nu(i)=0\nenddo")),
    # The second region writes what the first reads.
    "\n".join((loop("u", "v(i)", "n"), loop("v", "x(i)", "n-1"))),
    # A condition does not help sections whose bounds do meet.
    "\n".join(("do i=1,n\nif (n>=3) then\nu(i)=x(i)\nendif\nenddo",
               "do i=2,n-1\nif (n>=3) then\nu(i)=0\nendif\nenddo")),
])
def test_regions_that_may_depend_on_each_other_keep_their_own_launches(body):
    plan, emitted = emit(body)
    assert len(plan.regions) == 2
    assert groups(emitted) == []
    assert "kernel_group_" not in emitted.cuda
    assert "fort_scope_graph_enqueue_members_v1" not in emitted.cuda
    assert emitted.cuda.count("fort_scope_graph_enqueue_v1(") == 2


@pytest.mark.parametrize("body", [
    # The sections never meet: the ghost layers of two opposite faces are like this.
    "\n".join((loop("u", "x(i)", "n-2"), "do i=n-1,n\nu(i)=0\nenddo")),
    # Whether they meet depends on n; the runtime decides on the actual sections.
    "\n".join((loop("u", "x(i)", "3"), "do i=n-1,n\nu(i)=0\nenddo")),
    # Under a condition the sections are upper bounds, and bounds that never meet are enough: this
    # is what a selection's guard makes of the ghost layers of two opposite faces.
    "\n".join(("do i=1,n-2\nif (n>=3) then\nu(i)=x(i)\nendif\nenddo",
               "do i=n-1,n\nif (n>=3) then\nu(i)=0\nendif\nenddo")),
])
def test_regions_that_write_sections_of_one_array_not_known_to_meet_share_a_launch(body):
    plan, emitted = emit(body)
    assert len(plan.regions) == 2
    assert groups(emitted) == [[0, 1]]
    assert "kernel_group_0_1_device" in emitted.cuda
    # The members' own tasks stay behind the group launch for the case that the runtime refuses it.
    assert emitted.cuda.count("fort_scope_graph_enqueue_v1(") == 2


def test_a_dependent_region_splits_the_run_and_later_regions_regroup():
    body = "\n".join((loop("u", "x(i)", "n"), loop("v", "u(i)", "n-1"), loop("w", "x(i)", "n-2")))
    plan, emitted = emit(body)
    assert len(plan.regions) == 3
    # The first cannot join the second, which reads its result; the second and third can share.
    assert groups(emitted) == [[1, 2]]
    assert "kernel_group_1_2_device" in emitted.cuda
    assert emitted.cuda.count("fort_scope_graph_enqueue_members_v1(") == 1


def test_members_keep_their_own_code_as_the_fallback():
    _plan, emitted = emit("\n".join((loop("u", "2*u(i)", "n"), loop("v", "x(i)", "n-1"))))
    cuda = emitted.cuda
    attempt = cuda.index("fort_scope_graph_enqueue_members_v1(")
    fallback = cuda.index("if (!fort_graph_done) {", attempt)
    # One queued task per member remains behind the refused or unavailable group launch.
    assert cuda.count("fort_scope_graph_enqueue_v1(", fallback) == 2
    # Only forced device placement attempts the group; automatic placement decides per region.
    assert "if (fort_gpu_requested) {" in cuda[:attempt]
    assert "if (fort_graph_queued) {" in cuda[attempt:fallback]
