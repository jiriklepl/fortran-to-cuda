"""Computation bytes count exact read/write unions, preserving both directions."""
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission.common.abi import dimension_name
from compiler.emission.cuda.offload import _metadata
from compiler.frontend import lower_source
from compiler.offload.analysis import analyze_offload

RUNTIME = Path(__file__).resolve().parents[1] / 'runtime'
DRIVER = r'''
#include "offload.hpp"
#include <cassert>
#include <random>
#include <set>
using namespace generated_kernels::offload;
size_t enumerated(const std::vector<Box> &boxes) {
    std::set<std::vector<size_t>> touched;
    for (const auto &box:boxes) {
        auto point=box.lower;
        while (true) {
            touched.insert(point);
            size_t axis=0;
            while (axis<point.size() && point[axis]==box.upper[axis]) {
                point[axis]=box.lower[axis]; ++axis;
            }
            if (axis==point.size()) break;
            ++point[axis];
        }
    }
    return touched.size()*8;
}
int main(int argc,char **argv) {
    assert(argc==2); std::string mode=argv[1]; bool valid=true;
    Array a{nullptr,8,8*8*8*8,{8,8,8}};
    if (mode=="cross") {
        std::vector<Box> boxes={{{0,2,2},{7,5,5}},{{2,0,2},{5,7,5}},{{2,2,0},{5,5,7}}};
        assert(footprint_bytes(a,boxes,valid)==enumerated(boxes));
        assert(valid && footprint_bytes(a,boxes,valid)==256*8);
    } else if (mode=="slabs") {
        std::vector<Box> boxes={{{0,0,0},{1,7,7}},{{1,0,0},{2,7,7}},{{5,0,0},{6,7,7}}};
        assert(footprint_bytes(a,boxes,valid)==5*8*8*8 && valid);
    } else if (mode=="read_write") {
        std::vector<Box> read={{{0,0,0},{5,7,7}},{{2,0,0},{7,7,7}}};
        std::vector<Box> write={{{3,0,0},{4,7,7}}};
        const auto bytes=footprint_bytes(a,read,valid)+footprint_bytes(a,write,valid);
        assert(valid && bytes==(8+2)*8*8*8);
        assert(bytes>enumerated(read)); // R/W overlap still pays a read AND a write.
    } else if (mode=="duplicates") {
        std::vector<Box> boxes={{{0,0,0},{7,7,7}},{{1,1,1},{2,2,2}},{{0,0,0},{7,7,7}}};
        assert(footprint_bytes(a,boxes,valid)==8*8*8*8 && valid);
        assert(footprint_bytes(a,{},valid)==0 && valid);
    } else if (mode=="random") {
        std::mt19937 rng(19371);
        for (size_t rank=1;rank<=4;++rank) for (int trial=0;trial<50;++trial) {
            Array b{nullptr,8,0,std::vector<size_t>(rank,6)};std::vector<Box> boxes;
            for (int k=0;k<4;++k) {
                Box box;
                for (size_t axis=0;axis<rank;++axis) {
                    size_t lo=rng()%6,hi=rng()%6;
                    box.lower.push_back(std::min(lo,hi));box.upper.push_back(std::max(lo,hi));
                }
                boxes.push_back(box);
            }
            valid=true;const auto bytes=footprint_bytes(b,boxes,valid);
            assert(valid && bytes==enumerated(boxes));
        }
    } else if (mode=="invalid") {
        const std::vector<Box> rank={{{0,0},{1,1}}},outside={{{0,0,0},{8,0,0}}},reversed={{{2,0,0},{1,0,0}}};
        assert(footprint_bytes(a,rank,valid)==0 && !valid);
        valid=true;assert(footprint_bytes(a,outside,valid)==0 && !valid);
        valid=true;assert(footprint_bytes(a,reversed,valid)==0 && !valid);
        valid=false;assert(footprint_bytes(a,{},valid)==0 && !valid);
    } else if (mode=="overflow") {
        const auto max=std::numeric_limits<size_t>::max();
        Array b{nullptr,8,0,{max}};
        const std::vector<Box> boxes={{{0},{max-1}}};
        assert(footprint_bytes(b,boxes,valid)==0 && !valid);
    } else if (mode=="budget") {
        Array b{nullptr,8,1024,{128}};std::vector<Box> boxes;
        for(size_t i=0;i<=section_union_limit;++i) boxes.push_back({{2*i},{2*i}});
        assert(footprint_bytes(b,boxes,valid)==0 && !valid);
    } else return 2;
}
'''


def compile_and_run(path, source, *arguments):
    cxx = shutil.which('g++')
    if cxx is None:
        pytest.skip('C++ compiler required')
    (path / 'check.cpp').write_text(source)
    built = subprocess.run([cxx, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-I', str(RUNTIME), str(path / 'check.cpp'), '-o', str(path / 'check')],
                           capture_output=True, text=True, timeout=120)
    assert built.returncode == 0, built.stdout + built.stderr
    result = subprocess.run([str(path / 'check'), *arguments], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope='module')
def union_binary(tmp_path_factory):
    path = tmp_path_factory.mktemp('footprint-union')
    compile_and_run(path, DRIVER, 'cross')
    return path / 'check'


@pytest.mark.parametrize('mode', ['cross', 'slabs', 'read_write', 'duplicates', 'random',
                                 'invalid', 'overflow', 'budget'])
def test_computation_memory_unions(union_binary, mode):
    result = subprocess.run([str(union_binary), mode], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('budget_exhausted', [False, True])
def test_actual_generated_stencil_metadata_counts_union_and_both_rw_directions(tmp_path, budget_exhausted):
    source = '''module memory_case
contains
subroutine stencil(a,b)
real(8),intent(in)::a(:,:,:)
real(8),intent(inout)::b(:,:,:)
integer::i,j,k
do k=2,size(a,3)-1
 do j=2,size(a,2)-1
  do i=2,size(a,1)-1
   b(i,j,k)=b(i,j,k)+a(i,j,k)+a(i-1,j,k)+a(i+1,j,k)+a(i,j-1,k)+a(i,j+1,k)+a(i,j,k-1)+a(i,j,k+1)
  end do
 end do
end do
end subroutine
end module'''
    dimensions = (10, 9, 8)
    if budget_exhausted:
        dimensions = (36, 36, 36)
        source = source.replace('do k=2,size(a,3)-1', 'do k=1,2')
        source = source.replace('do j=2,size(a,2)-1', 'do j=1,2')
        source = source.replace('do i=2,size(a,1)-1', 'do i=1,2')
        expression = '+'.join(f'a(i+{offset},j+{offset},k+{offset})' for offset in range(33))
        before = source.index('   b(i,j,k)=')
        after = source.index('\n', before)
        source = source[:before] + '   b(i,j,k)=b(i,j,k)+' + expression + source[after:]
    function, plan = prepare_function(lower_source(source, 'stencil', source_name='memory_case.f90'),
                                      options=CompilerOptions(opt_level=0))
    analysis = analyze_offload(function, plan)
    assert analysis.available, analysis.reason
    declarations, actuals = [], []
    for position, symbol in enumerate(function.parameters):
        assert symbol.rank == 3
        declarations.append(f'const double *{symbol.cpp_name}')
        elements = '*'.join(map(str, dimensions))
        actuals.append(f'storage+{position}*{elements}')
        for axis, extent in enumerate(dimensions, 1):
            declarations.append(f'std::size_t {dimension_name(symbol, axis)}')
            actuals.append(str(extent))
    helper = '\n'.join(_metadata(function, analysis, ', '.join(declarations), 'metadata'))
    assert helper.count('offload::footprint_bytes(') == 2
    driver = '#include "offload.hpp"\n#include <cassert>\n#include <climits>\nnamespace offload=generated_kernels::offload;\n'
    driver += helper
    driver += f'\nint main(){{double storage[2*{elements}]{{}};auto data=metadata(' + ','.join(actuals) + ');'
    # a's read union: interior plus six faces. b is read and written separately.
    driver += 'assert(data.valid && data.units.size()==1);'
    if budget_exhausted:
        driver += 'assert(data.units[0].iterations==8);assert(std::isnan(data.units[0].memory_bytes));'
        driver += 'offload::Profile profile;profile.valid=true;'
        driver += 'assert(offload::select(data,profile,false).has_gpu());'
        driver += 'assert(!offload::select(data,profile,true).has_gpu());}\n'
    else:
        driver += 'assert(data.units[0].iterations==8*7*6);'
        driver += 'assert(data.units[0].memory_bytes==8*(3*8*7*6+2*(8*7+8*6+7*6)));}\n'
    compile_and_run(tmp_path, driver)


def test_unavailable_memory_cost_preserves_scoped_definition_proof(tmp_path):
    source = r'''
#include "scoped_planning.hpp"
#include <cassert>
using namespace fort_scoped::planning;
int main(){
    Inputs in;in.graph_mode=true;
    Resource b;b.handle=1;b.element_bytes=8;b.bytes=32;b.extents={4};
    b.initialized=b.host_current={Box{{0},{4}}};in.resources={b};
    Operation op;op.kind=FORT_SCOPE_PLAN_WORKER;op.unit=1;op.flops=8;
    op.memory_bytes=std::numeric_limits<double>::quiet_NaN();op.gpu_available=true;
    Region full=b.initialized;op.bindings={{1,{full,full,full}}};in.operations={op};
    assert(validate_definitions(in).status==FORT_SCOPE_OK);
    fort_scope_plan_costs c{};c.version=FORT_SCOPE_PLANNING_ABI_VERSION;c.valid=1;c.max_allocation_bytes=1024;
    for(double *value:{&c.cpu_flops,&c.cpu_bandwidth,&c.gpu_flops,&c.gpu_bandwidth,
        &c.h2d_bandwidth,&c.d2h_bandwidth,&c.create_seconds,&c.register_seconds,
        &c.host_access_seconds,&c.device_access_seconds,&c.gpu_setup_seconds,
        &c.cold_driver_startup_seconds,&c.allocation_seconds,&c.release_seconds,
        &c.wait_seconds,&c.launch_enqueue_seconds,&c.planning_operation_seconds})*value=1;
    const auto result=select(in,c);
    assert(!result.decision.available && result.reason=="unknown_or_overflowed_work");
    for(const Region *r:{&in.operations[0].bindings[0].effects.writes,
                        &in.resources[0].initialized,&in.resources[0].host_current}) {
        assert(r->size()==1 && r->front().lo==full.front().lo && r->front().hi==full.front().hi);
    }
}
'''
    compile_and_run(tmp_path, source)
