"""Exact original phase geometry, separated from unmeasured placement costs."""

import shutil
import subprocess
from dataclasses import replace

import pytest

from compiler.emission.cuda.tile_costs import generate_tile_work_function, tile_work_support
from compiler.frontend.tile_private import prove_tile_private_team
from compiler.ir import CompilationError
from compiler.offload.tile_work import TileWorkUnavailable, axis_visits, tile_team_work
from compiler.tests.test_tile_private import SOURCE, SOURCE_3D
from compiler.tests.test_tile_private_emission import original_source
from compiler.tests.test_tile_private_team import team_fixture


def issued(tmp_path, source=SOURCE, mapping="cooperative"):
    _path, analysis, routine, nodes = team_fixture(tmp_path, source)
    return tile_team_work(prove_tile_private_team(analysis, routine, nodes), mapping=mapping)


@pytest.mark.parametrize(("lower", "upper", "width"), [(-9, 7, 3), (-2, -2, 2), (1, 0, 3), (1, 19, 5), (1, 19, 100)])
def test_closed_geometry_matches_all_original_min_clipped_loop_visits(lower, upper, width):
    for before in (-3, -1, 0, 2):
        for clip in (-3, -1, 0, 4):
            for after in (-2, 0, 2):
                tiles = list(range(lower, upper+1, width))
                actual = sum(len(range(tile+before, min(tile+width+clip, upper)+after+1)) for tile in tiles)
                assert axis_visits(lower, upper, width, before, clip, after) == (len(tiles), actual)


def test_original_producer_halos_scratch_traffic_and_odd_batch_are_charged(tmp_path):
    work = issued(tmp_path)
    result = work.evaluate(lambda _axis: (1, 7, 3), batch_tiles=lambda: 2, scratch_lease_bytes=lambda: 112)
    assert (result.tiles, result.producer_visits, result.consumer_visits, result.launches) == (3, 16, 7, 2)
    assert result.work.real_work_proxy == 30
    assert result.work.external_read_bytes == 184
    assert result.work.external_write_bytes == 56
    assert result.work.scratch_read_bytes == 112
    assert result.work.scratch_write_bytes == 128
    assert result.work.conditional_upper_bound is False
    assert result.cooperative_phase_barriers == 6
    assert result.original_teams == 1
    assert result.original_worksharing_units == 1
    assert result.scratch_lease_bytes == 112
    record = work.public()
    assert record["original_fortran_family_attestation"] is False
    assert record["placement_authorized"] is False
    assert record["speedup_guaranteed"] is False
    assert "source transaction and scratch lifecycle" in record["unpriced"]


@pytest.mark.parametrize("kind", [4, 8])
def test_multiple_original_units_and_conditional_consumers_keep_all_work(tmp_path, kind):
    source = original_source(kind, branch=True)
    first = source.index("!$omp do ")
    last = source.index("!$omp end do", first)+len("!$omp end do\n")
    source = source[:last]+source[first:last]+source[last:]
    work = issued(tmp_path, source)
    result = work.evaluate(lambda _axis: (1, 19, 3), batch_tiles=lambda: 2, scratch_lease_bytes=lambda: 112)
    assert (result.tiles, result.producer_visits, result.consumer_visits, result.launches) == (14, 80, 38, 8)
    assert result.work.real_work_proxy == 156
    assert result.work.external_read_bytes == kind*(80+38)
    assert result.work.external_write_bytes == kind*38
    assert result.work.scratch_write_bytes == kind*80
    assert result.work.scratch_read_bytes == 2*kind*38
    assert result.work.conditional_upper_bound is True
    assert result.original_teams == 1
    assert result.original_worksharing_units == 2


def test_three_dimensional_product_preserves_expanded_producer_axis(tmp_path):
    work = issued(tmp_path, SOURCE_3D)
    domains = ((1, 5, 2), (1, 7, 3), (1, 9, 4))
    result = work.evaluate(domains.__getitem__, batch_tiles=lambda: 4, scratch_lease_bytes=lambda: 4096)
    assert result.tiles == 27
    # SOURCE_3D expands the original innermost x producer by three per tile.
    assert result.producer_visits == 5*7*(9+3*3)
    assert result.consumer_visits == 5*7*9
    assert result.launches == 7
    assert result.work.real_work_proxy == result.producer_visits+2*result.consumer_visits


@pytest.mark.parametrize("empty_axis", [0, 1, 2])
def test_zero_domain_never_reads_later_axis_or_batch_and_private_descriptor(tmp_path, empty_axis):
    work = issued(tmp_path, SOURCE_3D)
    reached = []

    def domain(axis):
        assert axis <= empty_axis
        reached.append(axis)
        return (1, 0, 3) if axis == empty_axis else (1, 7, 3)

    def forbidden():
        raise AssertionError("unreached private descriptor or batch expression")

    result = work.evaluate(domain, batch_tiles=forbidden, scratch_lease_bytes=forbidden)
    assert reached == list(range(empty_axis+1))
    assert result.launches == 0
    assert result.work.real_work_proxy == 0
    assert result.scratch_lease_bytes == 0


def test_overflow_unknown_width_and_unplanned_scratch_remain_unpriced(tmp_path):
    work = issued(tmp_path, SOURCE_3D)
    with pytest.raises(TileWorkUnavailable, match="uint64"):
        work.evaluate(lambda _axis: (-(1 << 31), (1 << 31)-1, 1),
                      batch_tiles=lambda: 1, scratch_lease_bytes=lambda: 1)
    for domain in ((1, 7, 0), (1, 7, -1), (1, 7, True), (1, 7, 1.5)):
        with pytest.raises(TileWorkUnavailable):
            work.evaluate(lambda _axis, domain=domain: domain, batch_tiles=lambda: 1, scratch_lease_bytes=lambda: 1)
    with pytest.raises(TileWorkUnavailable, match="preplanned"):
        work.evaluate(lambda _axis: (1, 1, 1), batch_tiles=lambda: 0, scratch_lease_bytes=lambda: 1)


def test_source_substitution_or_modified_work_counts_never_supply_cost_authority(tmp_path):
    work = issued(tmp_path)
    unit = work.units[0]
    forged = replace(work, units=(replace(unit, producer=replace(unit.producer, real_work_proxy=0)),))
    with pytest.raises(CompilationError, match="exact issued source"):
        forged.validate()
    source = tmp_path / "source.f90"
    source.write_text(source.read_text()+"! changed original\n")
    with pytest.raises(CompilationError):
        work.validate()


def test_compiled_prework_counts_match_source_visits_and_decline_overflow(tmp_path):
    cxx = shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler required for real work-count emission")
    work = issued(tmp_path, original_source(8, branch=True))
    function = generate_tile_work_function(work, name="quote_work", parameters=("int n", "int t"),
        domains=(("1", "n", "t"),), batch_tiles="2", scratch_lease_bytes="112")
    code = tmp_path / "counts.cpp"
    code.write_text(tile_work_support()+"\n"+function+r'''
#include <cassert>
#include <cstdlib>
#include <iostream>
int main(int argc,char**argv) {
 fort_tile_cost::totals out{};
 const bool ok=quote_work(std::atoi(argv[1]),std::atoi(argv[2]),out);
 std::cout<<ok<<' '<<out.tiles<<' '<<out.producer_visits<<' '<<out.consumer_visits<<' '<<out.launches
 <<' '<<out.real_work_proxy<<' '<<out.external_read_bytes<<' '<<out.external_write_bytes
 <<' '<<out.scratch_read_bytes<<' '<<out.scratch_write_bytes<<' '<<out.conditional_upper_bound;
}
''')
    executable = tmp_path / "counts"
    subprocess.run([cxx, "-std=c++17", "-O2", str(code), "-o", str(executable)], check=True, capture_output=True, timeout=30)
    for n, t in ((0, 3), (1, 2), (7, 3), (13, 2), (13, 5), (19, 3)):
        result = work.evaluate(lambda _axis, n=n, t=t: (1, n, t), batch_tiles=lambda: 2, scratch_lease_bytes=lambda: 112)
        raw = subprocess.run([str(executable), str(n), str(t)], check=True, capture_output=True, text=True, timeout=10)
        expected = [1, result.tiles, result.producer_visits, result.consumer_visits, result.launches,
                    result.work.real_work_proxy, result.work.external_read_bytes, result.work.external_write_bytes,
                    result.work.scratch_read_bytes, result.work.scratch_write_bytes, int(result.work.conditional_upper_bound)]
        assert list(map(int, raw.stdout.split())) == expected


def test_compiled_zero_guard_and_overflow_never_publish_partial_counts(tmp_path):
    cxx = shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler required for real work-count emission")
    work = issued(tmp_path, SOURCE_3D)
    zero = generate_tile_work_function(work, name="zero_work", parameters=(), domains=(
        ("1", "0", "3"), ("unreached()", "unreached()", "unreached()"), ("1", "7", "3")),
        batch_tiles="unreached()", scratch_lease_bytes="unreached()")
    overflow = generate_tile_work_function(work, name="overflow_work", parameters=(),
        domains=(("INT_MIN", "INT_MAX", "1"),)*3, batch_tiles="1", scratch_lease_bytes="1")
    code = tmp_path / "guard.cpp"
    code.write_text(tile_work_support()+r'''
#include <cassert>
#include <cstdlib>
static int unreached(){std::abort();}
'''+zero+"\n"+overflow+r'''
int main() {
 for(int lower:{-9,1})for(int upper:{0,1,7,19})for(int width:{1,3,5})
 for(int before:{-3,-1,0,2})for(int clip:{-3,-1,0,4})for(int after:{-2,0,2}) {
  std::uint64_t tiles=0,visits=0,actual_tiles=0,actual_visits=0;
  assert(fort_tile_cost::axis(lower,upper,width,before,clip,after,tiles,visits));
  for(int tile=lower;tile<=upper;tile+=width) {
   ++actual_tiles;
   for(int point=tile+before;point<=std::min(tile+width+clip,upper)+after;++point)++actual_visits;
  }
  assert(tiles==actual_tiles);assert(visits==actual_visits);
 }
 fort_tile_cost::totals out{};out.launches=123;
 assert(zero_work(out));assert(out.launches==0);assert(out.scratch_lease_bytes==0);
 out.launches=456;assert(!overflow_work(out));assert(out.launches==0);assert(out.producer_visits==0);
}
''')
    executable = tmp_path / "guard"
    subprocess.run([cxx, "-std=c++17", "-O2", str(code), "-o", str(executable)], check=True, capture_output=True, timeout=30)
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=10)
