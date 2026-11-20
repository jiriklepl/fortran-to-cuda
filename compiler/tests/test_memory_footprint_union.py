"""Computation bytes count exact read/write unions, preserving both directions."""
import shutil
import subprocess
from pathlib import Path

import pytest

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
