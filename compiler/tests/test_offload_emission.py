"""Runtime partition and section-copy checks against independent Fortran."""

import shutil
import subprocess
from pathlib import Path

import pytest


def run(command, path, environment=None):
    result = subprocess.run(command, cwd=path, env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr


def test_partition_costs_preserve_faces_and_choose_cpu_gpu_gpu_cpu(tmp_path):
    cxx = shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler required")
    header = Path(__file__).resolve().parents[1] / "runtime" / "offload.hpp"
    (tmp_path / "check.cpp").write_text('''
#include <cstdlib>
#include <cassert>
#include "''' + str(header) + '''"
using namespace generated_kernels::offload;
int main() {
    Profile p; p.valid=true; p.launch_seconds=10;
    p.cpu_flops=1; p.gpu_flops=100;
    Data d; d.units.resize(4); d.arrays.push_back({nullptr,8,8,{1}});
    for(auto &u:d.units) { u.iterations=1; u.arrays.resize(1); }
    d.units[1].arrays[0].upload=d.units[2].arrays[0].upload={{{0},{0}}};
    p.h2d_latency=1;
    d.units[0].flops=d.units[3].flops=1;
    d.units[1].flops=d.units[2].flops=1000;
    auto choice=select(d,p,true);
    assert(choice.choices.size()==3);
    assert(!choice.choices[0].gpu && choice.choices[0].end==1);
    assert(choice.choices[1].gpu && choice.choices[1].begin==1 && choice.choices[1].end==3);
    assert(!choice.choices[2].gpu && choice.choices[2].begin==3);
    p.valid=false; assert(!select(d,p,true).has_gpu());
    d.valid=false; assert(!select(d,p,false).has_gpu());
    std::vector<Box> faces;
    append_box(faces,{{0,0,0},{0,19,29}});
    append_box(faces,{{9,0,0},{9,19,29}});
    assert(faces.size()==2);
    Array a{nullptr,8,10*20*30*8,{10,20,30}};
    bool valid=true;
    assert(box_bytes(a,faces[0],valid)+box_bytes(a,faces[1],valid)==2*20*30*8);
    assert(copy_operations(a,{{1,1,1},{8,18,28}},valid)==1);
    std::size_t result=0;
    assert(!mul(std::numeric_limits<std::size_t>::max(),2,result));
    index(1.0e50L,valid); assert(!valid);
}
''')
    run([cxx, "-std=c++17", "-Wall", "-Werror", "check.cpp", "-o", "check"], tmp_path)
    run([str(tmp_path / "check")], tmp_path)


SOURCE = """module section_case
contains
subroutine advance(a,b,n,m,l)
real(8),intent(inout)::a(:,:,:)
real(8),intent(in)::b(:,:,:)
integer,intent(in)::n,m,l
integer::i,j,k
do k=1,l
 do j=2,m-1
  do i=-3,n-4
   if(b(i+4,j,k)>0.0_8) a(i+4,j,k)=b(i+4,j,k)+real(i+100*j+10000*k,8)
  end do
 end do
end do
do k=1,l
 do j=1,m
  a(1,j,k)=b(1,j,k)*2.0_8
  a(n,j,k)=b(n,j,k)*3.0_8
 end do
end do
end subroutine
end module
"""
BOUNDARY = """module section_case
contains
subroutine advance(a,b,n,m,l)
real(8),intent(inout)::a(:,:,:)
real(8),intent(in)::b(:,:,:)
integer,intent(in)::n,m,l
integer::j,k
do k=1,l
 do j=1,m
  a(1,j,k)=b(1,j,k)*2.0_8
  a(n,j,k)=b(n,j,k)*3.0_8
 end do
end do
end subroutine
end module
"""
MIXED = """module section_case
contains
subroutine advance(a,b,n,m,l)
real(8),intent(inout)::a(:,:,:)
real(8),intent(in)::b(:,:,:)
integer,intent(in)::n,m,l
integer::i,j,k
do k=1,l
 do j=1,m
  a(1,j,k)=b(1,j,k)*2.0_8
 end do
end do
do k=1,l
 do j=1,m
  do i=1,n
   a(i,j,k)=b(i,j,k)*2.0_8
  end do
 end do
end do
do k=1,l
 do j=1,m
  do i=1,n
   a(i,j,k)=a(i,j,k)*3.0_8
  end do
 end do
end do
do k=1,l
 do j=1,m
  a(n,j,k)=b(n,j,k)*3.0_8
 end do
end do
end subroutine
end module
"""
