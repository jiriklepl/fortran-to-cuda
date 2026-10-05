"""Keep transfer costs and host-visible validation inside the right regions."""

import pytest

from benchmarks.harness.drivers import correctness_driver, timing_driver
from benchmarks.harness.paths import CASES


@pytest.mark.parametrize("case", ["CDU", "CDV", "CDW"])
def test_call_drivers_preserve_canonical_workload(case):
    canonical = CASES / case / "test_main.f90"
    assert correctness_driver(case) == canonical.read_text()
    source = timing_driver(case, (37, 11, 5), 7, 2)
    assert "integer, parameter :: nx=37, ny=11, nz=5" in source
    assert "integer, parameter :: niter=7, nwarmup=2" in source
    assert source.count(f"call {case}(result,u,v,w,dxmin,dymin,dzmin,nx,ny,nz)") == 2
    assert "!$acc" not in source


def test_resident_timing_includes_transfers_and_reads_result_after_copyout():
    source = timing_driver("CDU", (37, 11, 5), 7, 2, resident=True)
    positions = [
        source.index(marker)
        for marker in (
            "do iteration = 1, nwarmup",
            "call system_clock(begin_count,rate)",
            "!$acc data copyin(u,v,w) copyout(result)",
            "do iteration = 1, niter",
            "!$acc end data",
            "call system_clock(end_count)",
            "checksum = sum(abs(result(",
        )
    ]
    assert positions == sorted(positions)
    assert sum("!$acc" in line for line in source.splitlines()) == 2
    without_directives = "\n".join(line for line in source.splitlines() if "!$acc" not in line) + "\n"
    assert without_directives == timing_driver("CDU", (37, 11, 5), 7, 2)


@pytest.mark.parametrize(("case", "output"), [("CDU", "U2"), ("CDV", "V2"), ("CDW", "W2")])
def test_resident_correctness_repeats_call_before_result_copyout(case, output):
    source = correctness_driver(case, resident=True)
    call = f"    call {case}({output}, U, V, W, dxmin, dymin, dzmin, NX, NY, NZ)"
    opening = f"    !$acc data copyin(U,V,W) copyout({output})"
    region = f"{opening}\n{call}\n{call}\n    !$acc end data"
    assert region in source
    assert source.index("!$acc end data") < source.index("write(*,'(g0.17)')")
    assert source.replace(region, call) == correctness_driver(case)


@pytest.mark.parametrize("case", ["CDU", "CDV", "CDW"])
def test_local_session_timing_includes_lifetime_and_explicit_result(case):
    source = timing_driver(case, (37, 11, 5), 7, 2, resident=True, session=True)
    positions = [
        source.index(marker)
        for marker in (
            "do iteration = 1, nwarmup",
            "call system_clock(begin_count,rate)",
            f"call {case}_create(",
            "do iteration = 1, niter",
            f"call {case}_run(",
            f"call {case}_update_host(",
            f"call {case}_destroy(",
            "call system_clock(end_count)",
            "checksum = sum(abs(result(",
        )
    ]
    assert positions == sorted(positions)
    assert "!$acc" not in source
    assert source.count(f"call {case}(result,") == 1  # Warmup remains host-visible.
    correctness = correctness_driver(case, resident=True, session=True)
    assert correctness.count(f"call {case}_run(") == 2
    assert correctness.index(f"call {case}_update_host(") < correctness.index("write(*,'(g0.17)')")
