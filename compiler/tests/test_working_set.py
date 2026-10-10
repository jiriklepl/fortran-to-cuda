"""Unique cost footprints retain geometry, root identities and query validity."""

import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.emission.cuda.working_set import working_set_lines


def compile_and_run(tmp_path, body):
    cxx = shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler required")
    runtime = Path(__file__).resolve().parents[1] / "runtime/offload.hpp"
    emitted = "\n".join(working_set_lines(["first", "second"], data="metadata"))
    unknown = "\n".join(working_set_lines(["first", "second"], data="metadata", exact=False))
    # Run the emitted code directly against the actual offload geometry API;
    # the fixture never includes CUDA or calls a runtime context interface.
    source = f'''#include <cassert>
#include <limits>
#include <utility>
#include "{runtime}"
using namespace generated_kernels;
using namespace generated_kernels::offload;
std::pair<bool,std::size_t> measure(const Data &metadata,std::uint64_t first=1,std::uint64_t second=2) {{
{emitted}
return {{fort_working_set_available,fort_working_set_bytes}};
}}
std::pair<bool,std::size_t> unknown(const Data &metadata,std::uint64_t first=1,std::uint64_t second=2) {{
(void)metadata; (void)first; (void)second;
{unknown}
return {{fort_working_set_available,fort_working_set_bytes}};
}}
Data make(Array first,Array second) {{
Data d; d.arrays={{first,second}}; d.units.resize(1); d.units[0].iterations=1; d.units[0].arrays.resize(2);
return d;
}}
int main() {{
{body}
}}
'''
    (tmp_path / "check.cpp").write_text(source)
    result = subprocess.run([cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "check.cpp", "-o", "check"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(tmp_path / "check")], cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


def test_opposite_faces_and_read_write_overlap_count_once(tmp_path):
    compile_and_run(tmp_path, r'''
Array a{nullptr,8,8*12*10*6,{12,10,6}};
auto d=make(a,a);
d.units[0].arrays[0].upload={{{0,0,0},{0,9,5}},{{11,0,0},{11,9,5}}};
d.units[0].arrays[0].download={{{0,0,0},{0,9,5}}};
d.units[0].memory_bytes=3*10*6*8; // Traffic remains a distinct quantity.
const auto original=d.units[0].arrays;
const auto result=measure(d);
assert(result.first && result.second==2*10*6*8);
assert(d.valid && d.units[0].memory_bytes==3*10*6*8);
assert(d.units[0].arrays[0].upload==original[0].upload);
assert(d.units[0].arrays[0].download==original[0].download);
// A partial corner intersection requires an exact nonrectangular union.
auto plane=make({nullptr,8,8*5*5,{5,5}},a);
plane.units[0].arrays[0].upload={{{0,0},{3,3}}};
plane.units[0].arrays[0].download={{{2,2},{4,4}}};
assert(measure(plane)==std::make_pair(true,std::size_t((16+9-4)*8)));
''')


def test_duplicate_touched_roots_decline_but_untouched_aliases_do_not(tmp_path):
    compile_and_run(tmp_path, r'''
Array a{nullptr,8,8*8,{8}};
auto d=make(a,a);
d.units[0].arrays[0].upload={{{0},{4}}};
assert(measure(d,7,7)==std::make_pair(true,std::size_t(5*8)));
d.units[0].arrays[1].upload={{{3},{7}}};
assert(measure(d,7,7)==std::make_pair(false,std::size_t(0)));
assert(d.valid);
assert(measure(d,7,8)==std::make_pair(true,std::size_t(10*8)));
assert(measure(d,0,8)==std::make_pair(false,std::size_t(0)));
// A scalar-only unit has a known zero array working set.
d.units[0].arrays[0]={}; d.units[0].arrays[1]={};
assert(measure(d,0,0)==std::make_pair(true,std::size_t(0)));
''')


def test_combined_rectangle_and_fragmentation_limits_are_cost_only(tmp_path):
    compile_and_run(tmp_path, r'''
Array a{nullptr,8,8*80,{80}};
auto d=make(a,a);
for(std::size_t k=0;k<16;++k) {
 d.units[0].arrays[0].upload.push_back({{4*k},{4*k}});
 d.units[0].arrays[0].download.push_back({{4*k+2},{4*k+2}});
}
assert(measure(d)==std::make_pair(true,std::size_t(32*8)));
d.units[0].arrays[0].download.push_back({{79},{79}});
assert(measure(d)==std::make_pair(false,std::size_t(0)));
assert(d.valid);
// The existing union work/output bound also rejects crossing strips.
auto strips=make({nullptr,1,256,{16,16}},a);
for(std::size_t k=0;k<16;k+=2) strips.units[0].arrays[0].upload.push_back({{k,0},{k,15}});
for(std::size_t k=1;k<16;k+=2) strips.units[0].arrays[0].download.push_back({{0,k},{15,k}});
assert(measure(strips)==std::make_pair(false,std::size_t(0)));
assert(strips.valid);
''')


def test_checked_coordinates_products_and_total_overflow_leave_metadata_valid(tmp_path):
    compile_and_run(tmp_path, r'''
const auto max=std::numeric_limits<std::size_t>::max();
Array huge{nullptr,1,max,{max}};
auto d=make(huge,huge);
d.units[0].arrays[0].upload={{{0},{max-1}}};
assert(measure(d)==std::make_pair(true,max));
d.units[0].arrays[1].upload={{{0},{0}}};
assert(measure(d)==std::make_pair(false,std::size_t(0)));
assert(d.valid);
auto invalid=make({nullptr,8,64,{8}},huge);
for(const Box &box : std::vector<Box>{{{0},{8}},{{2},{1}},{{0,0},{1,1}}}) {
 invalid.units[0].arrays[0].upload={box};
 assert(measure(invalid)==std::make_pair(false,std::size_t(0)));
 assert(invalid.valid);
}
auto product=make({nullptr,8,max,{max}},huge);
product.units[0].arrays[0].upload={{{0},{max-1}}};
assert(measure(product)==std::make_pair(false,std::size_t(0)));
assert(product.valid);
''')


def test_unknown_empty_or_malformed_metadata_does_not_poison_definition_query(tmp_path):
    compile_and_run(tmp_path, r'''
auto d=make({nullptr,8,0,{0}},{nullptr,8,0,{0}});
d.units[0].iterations=0;
assert(measure(d)==std::make_pair(true,std::size_t(0)));
assert(unknown(d)==std::make_pair(false,std::size_t(0)));
assert(d.valid);
d.units[0].arrays.resize(1);
assert(measure(d)==std::make_pair(false,std::size_t(0)));
assert(d.valid);
d.units.clear();
assert(measure(d)==std::make_pair(false,std::size_t(0)));
assert(d.valid);
d.valid=false;
assert(measure(d)==std::make_pair(false,std::size_t(0)));
assert(!d.valid);
''')


def test_distinct_validated_view_roots_use_logical_cardinality_not_root_pitch(tmp_path):
    compile_and_run(tmp_path, r'''
// A validated rectangular view of a rank-reduced 100x80x60 root can retain
// logical axes (z,x), extents(4,3), and original pitches(8000,1). Its mapping
// remains injective: union cardinality uses logical boxes, not an enclosing
// 4*8000-byte physical address interval. The second view has a distinct root.
auto d=make({nullptr,8,4*3*8,{4,3}},{nullptr,8,5*8,{5}});
d.units[0].arrays[0].upload={{{0,0},{3,2}}};
d.units[0].arrays[0].download={{{1,0},{2,2}}};
d.units[0].arrays[1].upload={{{0},{4}}};
assert(measure(d,11,12)==std::make_pair(true,std::size_t((12+5)*8)));
// Logical boxes of views sharing one root cannot establish physical union.
assert(measure(d,11,11)==std::make_pair(false,std::size_t(0)));
assert(d.valid);
''')


@pytest.mark.parametrize("kwargs", [{"prefix": "bad-prefix"}, {"unit_index": -1},
                                    {"unit_index": True}, {"exact": "unknown"}])
def test_generator_requires_explicit_bounded_metadata_contract(kwargs):
    with pytest.raises(ValueError, match="working-set"):
        working_set_lines(["root"], **kwargs)


def test_unknown_static_footprints_emit_no_runtime_metadata_reads():
    lines = "\n".join(working_set_lines(["untouchable_root"], data="untouchable_data", exact=False))
    assert "fort_working_set_available = false" in lines
    assert "untouchable_root" not in lines
    assert "untouchable_data" not in lines
