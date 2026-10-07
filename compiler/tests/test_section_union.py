"""Bounded physical unions, checked independently by enumerating coordinates."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.emission.common.resources import read_common_header


def run(command, directory):
    result = subprocess.run(command, cwd=directory, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.native
def test_bounded_union_coverage_copy_budget_and_overflow(tmp_path):
    cxx = shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler required")
    runtime = Path(__file__).resolve().parents[1] / "runtime/offload.hpp"
    (tmp_path / "union.cpp").write_text(r'''
#include <cassert>
#include <cstdlib>
#include <numeric>
#include <random>
''' + f'#include "{runtime}"\n' + r'''
using namespace generated_kernels::offload;

std::vector<unsigned> cover(const Array &array,const std::vector<Box> &boxes) {
    std::size_t elements=array.bytes/array.element_bytes;
    std::vector<unsigned> counts(elements);
    for(std::size_t i=0;i<elements;++i) {
        for(const auto &box:boxes) {
            bool inside=true; auto rest=i;
            for(std::size_t axis=0;axis<array.dimensions.size();++axis) {
                const auto at=rest%array.dimensions[axis]; rest/=array.dimensions[axis];
                inside &= box.lower[axis]<=at && at<=box.upper[axis];
            }
            counts[i]+=inside;
        }
    }
    return counts;
}
void same_coverage(const std::vector<unsigned> &a,const std::vector<unsigned> &b,bool disjoint) {
    assert(a.size()==b.size());
    for(std::size_t i=0;i<a.size();++i) {
        assert(bool(a[i])==bool(b[i]));
        if(disjoint) assert(b[i]<=1);
    }
}
int main() {
    std::mt19937 random(117);
    unsigned successful=0, optimized=0;
    for(unsigned rank=1;rank<=4;++rank) {
        Array a{nullptr,8,8,{7,6,5,4}}; a.dimensions.resize(rank);
        for(auto dimension:a.dimensions) a.bytes*=dimension;
        for(unsigned sample=0;sample<180;++sample) {
            std::vector<Box> boxes;
            for(unsigned n=0;n<2+sample%8;++n) {
                Box b;
                for(auto dimension:a.dimensions) {
                    const auto lo=random()%dimension;
                    b.lower.push_back(lo); b.upper.push_back(lo+random()%(dimension-lo));
                }
                boxes.push_back(b);
            }
            const auto expected=cover(a,boxes);
            std::vector<Box> geometry{{{99},{99}}};
            if(disjoint_union(boxes,geometry)) {
                assert(geometry.size()<=section_union_limit);
                same_coverage(expected,cover(a,geometry),true); ++successful;
            } else assert((geometry==std::vector<Box>{{{99},{99}}}));
            auto normalized=boxes;
            const bool calibrated=sample%2;
            const bool changed=calibrated ? deduplicate_boxes(a,normalized,1,8) : deduplicate_boxes(a,normalized);
            same_coverage(expected,cover(a,normalized),changed);
            if(!changed) assert(normalized==boxes);
            else {
                std::size_t old_bytes=0,old_copies=0,new_bytes=0,new_copies=0;
                assert(transfer_metrics(a,boxes,old_bytes,old_copies));
                assert(transfer_metrics(a,normalized,new_bytes,new_copies));
                assert(new_bytes<old_bytes);
                if(calibrated) assert(new_bytes/8.0+new_copies<old_bytes/8.0+old_copies);
                else assert(new_copies<=old_copies);
                ++optimized;
            }
        }
    }
    assert(successful>100 && optimized>100);
    // Common stencil geometry: an expanded x box and four overlapping y/z
    // boxes. Keep the large box and send only the newly exposed faces.
    Array a{nullptr,8,12*12*12*8,{12,12,12}};
    std::vector<Box> stencil{{{0,1,1},{11,10,10}},
                            {{1,0,1},{10,9,10}},{{1,2,1},{10,11,10}},
                            {{1,1,0},{10,10,9}},{{1,1,2},{10,10,11}}};
    const auto original=stencil;
    assert(deduplicate_boxes(a,stencil));
    assert(stencil.size()==5);
    std::size_t bytes=0,copies=0;
    assert(transfer_metrics(a,stencil,bytes,copies));
    assert(bytes==(1200+4*100)*8 && copies==5);
    same_coverage(cover(a,original),cover(a,stencil),true);
    // Apply exactly the same normalization to uploads/downloads across units.
    Data data; data.arrays={a}; data.units.resize(2);
    for(auto &u:data.units) { u.iterations=1; u.arrays.resize(1); }
    data.units[0].arrays[0].upload={original[0]};
    data.units[1].arrays[0].upload={original.begin()+1,original.end()};
    data.units[0].arrays[0].download=original;
    Profile p; p.valid=true; p.h2d_latency=1; p.d2h_latency=2; p.h2d_bandwidth=p.d2h_bandwidth=8;
    const auto uncalibrated=interval(data,0,2)[0];
    assert(uncalibrated.upload.size()==3);
    same_coverage(cover(a,original),cover(a,uncalibrated.upload),false);
    const auto footprint=interval(data,0,2,p)[0];
    same_coverage(cover(a,original),cover(a,footprint.upload),true);
    same_coverage(cover(a,original),cover(a,footprint.download),true);
    assert(gpu_seconds(data,0,2,p)==5*3+2*1600);
    // Never enclose opposite faces in a volume-sized bounding rectangle.
    std::vector<Box> faces{{{0,0,0},{0,11,11}},{{11,0,0},{11,11,11}}};
    const auto original_faces=faces;
    assert(!deduplicate_boxes(a,faces) && faces==original_faces);
    // Small corner overlaps would require extra copies: preserve the old plan.
    Array plane{nullptr,8,5*5*8,{5,5}};
    std::vector<Box> corner{{{0,0},{3,3}},{{2,2},{4,4}}}, separate;
    assert(disjoint_union(corner,separate) && separate.size()==3);
    const auto original_corner=corner;
    assert(!deduplicate_boxes(plane,corner) && corner==original_corner);
    // Extra calls require calibrated savings; ties, bad rates, and expensive
    // launch latency retain the original list.
    assert(!deduplicate_boxes(plane,corner,4,8) && corner==original_corner);
    assert(!deduplicate_boxes(plane,corner,5,8) && corner==original_corner);
    assert(!deduplicate_boxes(plane,corner,0,0) && corner==original_corner);
    assert(!deduplicate_boxes(plane,corner,std::numeric_limits<double>::quiet_NaN(),8));
    assert(!deduplicate_boxes(plane,corner,0,std::numeric_limits<double>::infinity()));
    assert(deduplicate_boxes(plane,corner,1,8));
    same_coverage(cover(plane,original_corner),cover(plane,corner),true);
    // A grid of crossing strips exceeds the fragmentation limit. Failed
    // normalization must leave the complete original transfer list untouched.
    std::vector<Box> strips;
    for(std::size_t i=0;i<16;i+=2) strips.push_back({{i,0},{i,15}});
    for(std::size_t j=1;j<16;j+=2) strips.push_back({{0,j},{15,j}});
    separate={{{99},{99}}};
    assert(!disjoint_union(strips,separate));
    assert((separate==std::vector<Box>{{{99},{99}}}));
    Array grid{nullptr,1,256,{16,16}};
    const auto original_strips=strips;
    assert(!deduplicate_boxes(grid,strips) && strips==original_strips);
    std::vector<Box> too_many(section_union_limit+1,{{0},{0}});
    assert(!disjoint_union(too_many,separate));
    assert(!disjoint_union({{{0},{0}},{{0,0},{1,1}}},separate));
    // Endpoint arithmetic must not wrap, even at the size_t boundary.
    const auto max=std::numeric_limits<std::size_t>::max();
    assert(disjoint_union({{{1},{max-1}},{{0},{max}}},separate));
    assert((separate==std::vector<Box>{{{0},{max}}}));
    Array huge{nullptr,1,max,{max}};
    std::vector<Box> overflow{{{0},{max-1}},{{1},{max-1}}};
    const auto original_overflow=overflow;
    assert(!deduplicate_boxes(huge,overflow) && overflow==original_overflow);
}
''')
    run([cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "union.cpp", "-o", "union"], tmp_path)
    run([str(tmp_path / "union")], tmp_path)


@pytest.mark.native
@pytest.mark.cuda
def test_disjoint_transfers_preserve_holes_on_device_and_host(tmp_path):
    nvcc = shutil.which(os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc"))
    if nvcc is None:
        pytest.skip("CUDA compiler required")
    (tmp_path / "common.hpp").write_text(read_common_header())
    (tmp_path / "copies.cu").write_text(r'''
#include <cassert>
#include <numeric>
#include "common.hpp"
int main() {
    using namespace generated_kernels::offload;
    const std::vector<Box> original{{{0,1,1},{11,10,10}},
        {{1,0,1},{10,9,10}},{{1,2,1},{10,11,10}},
        {{1,1,0},{10,10,9}},{{1,1,2},{10,10,11}}};
    std::vector<double> host(12*12*12),actual(host.size()),expected(host.size()),untouched(host.size());
    std::iota(untouched.begin(),untouched.end(),-10000.0);
    Array array{host.data(),8,host.size()*8,{12,12,12}};
    void *device=nullptr; CUCH(cudaMalloc(&device,array.bytes));
    for(int rep=0;rep<3;++rep) for(bool upload:{true,false}) {
        std::iota(host.begin(),host.end(),rep*1000.0);
        expected=untouched;
        for(std::size_t k=0;k<12;++k) for(std::size_t j=0;j<12;++j) for(std::size_t i=0;i<12;++i) {
            bool touched=false;
            for(const auto &box:original)
                touched |= box.lower[0]<=i && i<=box.upper[0] && box.lower[1]<=j && j<=box.upper[1] &&
                           box.lower[2]<=k && k<=box.upper[2];
            if(touched) expected[i+12*(j+12*k)]=host[i+12*(j+12*k)];
        }
        CUCH(cudaMemcpy(device,upload ? untouched.data() : host.data(),array.bytes,cudaMemcpyHostToDevice));
        if(!upload) host=untouched;
        auto boxes=original;
        assert(deduplicate_boxes(array,boxes));
        for(const auto &box:boxes) copy_box(array,box,device,upload);
        if(upload) CUCH(cudaMemcpy(actual.data(),device,array.bytes,cudaMemcpyDeviceToHost));
        else actual=host;
        assert(actual==expected);
    }
    CUCH(cudaFree(device));
}
''')
    command = [nvcc, "-std=c++17", "-O2", "-DFORT_OFFLOAD_ENABLED", "-Xcompiler=-fopenmp"]
    host = shutil.which(os.environ.get("CUDAHOSTCXX", "g++-14"))
    if host:
        command += ["-ccbin", host]
    run([*command, "copies.cu", "-o", "copies"], tmp_path)
    run([str(tmp_path / "copies")], tmp_path)
