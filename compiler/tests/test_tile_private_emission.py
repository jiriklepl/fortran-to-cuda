"""Compile the emitted ordered body and compare complete original Fortran data."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.emission.common.abi import abi_arguments
from compiler.emission.cuda.tile_private import generate_tile_function, scratch_batch_helper
from compiler.frontend.lowering import lower_tile_private
from compiler.frontend.tile_private import prove_tile_private
from compiler.tests.test_tile_private import SOURCE, fixture


def original_arguments(lowered, *, array_pointer=".data()", array_extent=".size()", lower=-3):
    """Bind the issued original ABI rather than assuming symbol IDs or order."""
    bindings = {symbol: binding for binding, symbol in lowered.bindings}
    lower_bounds = {symbol for _, _, symbol in lowered.lower_bounds}
    result = []
    for argument in abi_arguments(lowered.region.parameters):
        symbol = argument.symbol
        if symbol in lower_bounds:
            result.append(str(lower))
        else:
            name = bindings[symbol].name
            result.append(name + (array_extent if argument.dimension else array_pointer if symbol.rank else ""))
    return result


def original_source(kind, lower=-3, branch=False):
    source = SOURCE.replace("real(8)", f"real({kind})").replace("3.0_8", f"3.0_{kind}")
    if branch:
        source = source.replace("  b(i)=b(i)+(tmp(li+1)-tmp(li-1))", """  if(mod(i,2)==0)then
   b(i)=b(i)+(tmp(li+1)-tmp(li-1))
  else
   b(i)=b(i)-(tmp(li+1)-tmp(li-1))
  endif""")
    if lower != -3:
        offset = lower + 3
        source = source.replace("(-3:)", f"({lower}:)")
        source = source.replace("a(i)", f"a(i+{offset})").replace("b(i)", f"b(i+{offset})")
    return source


@pytest.fixture(scope="module", params=[(kind, lower, branch) for kind in (4, 8) for lower in (-3, 2) for branch in (False, True)])
def compiled(request, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("tile-private-" + "-".join(map(str, request.param)))
    fc, cxx = shutil.which("gfortran"), shutil.which("g++")
    if fc is None or cxx is None:
        pytest.skip("gfortran and g++ required for original numerical oracle")
    kind, lower, branch = request.param
    source = original_source(kind, lower, branch)
    _, analysis, routine, node = fixture(tmp_path, source)
    proof = prove_tile_private(analysis, routine, node)
    lowered = lower_tile_private(proof)
    region = lowered.region
    function = "\n".join(generate_tile_function(region, addressing=lowered.addressing, device=False))
    arguments = ",".join(original_arguments(lowered, lower=lower))
    element = "float" if kind == 4 else "double"
    header = Path("compiler/emission/common/templates/common_functions.cuh").resolve()
    numeric = Path("compiler/runtime/numeric.hpp").resolve()
    cpp = tmp_path / "probe.cpp"
    cpp.write_text(f'''#include "{header}"
#include "{numeric}"
#include <fstream>
using generated_kernels::indexing::F_IDX;
{function}
int main(int argc,char**argv){{
 const int n=std::atoi(argv[1]),t=std::atoi(argv[2]);
 std::vector<{element}>a(n+8),b(n+8);
 for(int i=-3;i<=n+4;++i){{a[i+3]=static_cast<{element}>(17*i-3)/static_cast<{element}>(7);b[i+3]=static_cast<{element}>(500-i)/static_cast<{element}>(32);}}
 const std::size_t tiles=n?((n-1)/t+1):0;
 const std::size_t stride=(static_cast<std::size_t>(t+4)*sizeof({element})+7)&~std::size_t(7);
 const std::size_t batch=tiles<3?tiles:3;
 std::vector<unsigned char>scratch(batch*stride);
 for(std::size_t first=0;first<tiles;first+=batch){{
  const std::size_t count=tiles-first<batch?tiles-first:batch;
  for(std::size_t slot=0;slot<count;++slot)
   fort_tile_one_0({arguments},1,t,tiles,scratch.data(),stride,0,-1,t+4,slot,first+slot);
 }}
 std::ofstream out(argv[3],std::ios::binary);
 out.write(reinterpret_cast<const char*>(a.data()),a.size()*sizeof({element}));
 out.write(reinterpret_cast<const char*>(b.data()),b.size()*sizeof({element}));
}}
''')
    driver = tmp_path / "driver.f90"
    driver.write_text(f"""program driver
use stencil
implicit none
integer::n,t,i,j
character(512)::arg,out
real({kind}),allocatable::a(:),b(:)
call get_command_argument(1,arg)
read(arg,*)n
call get_command_argument(2,arg)
read(arg,*)t
call get_command_argument(3,out)
allocate(a({lower}:{lower}+n+7),b({lower}:{lower}+n+7))
do i={lower},{lower}+n+7
 j=i-{lower}-3
 a(i)=real(17*j-3,{kind})/real(7,{kind})
 b(i)=real(500-j,{kind})/real(32,{kind})
enddo
call advance(a,b,n,t)
open(10,file=trim(out),access='stream',form='unformatted',status='replace')
write(10)a,b
close(10)
end program
""")
    subprocess.run(
        [
            fc,
            "-O2",
            "-fopenmp",
            "-ffp-contract=off",
            "-J",
            str(tmp_path),
            str(tmp_path / "source.f90"),
            str(driver),
            "-o",
            str(tmp_path / "native"),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    subprocess.run(
        [cxx, "-O2", "-std=c++17", "-ffp-contract=off", str(cpp), "-o", str(tmp_path / "emitted")],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return tmp_path


@pytest.mark.parametrize(("n", "t"), [(0, 3), (1, 2), (7, 3), (13, 2), (13, 5), (19, 3)])
def test_emitted_original_phases_match_complete_fortran_fields_and_untouched_halos(compiled, n, t):
    paths = []
    for name in ("native", "emitted"):
        out = compiled / (name + ".bin")
        subprocess.run(
            [str(compiled / name), str(n), str(t), str(out)],
            check=True,
            capture_output=True,
            timeout=30,
            env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static"},
        )
        paths.append(out.read_bytes())
    assert paths[0] == paths[1]


def test_lease_batches_preserve_prework_failure_and_never_replay_accepted_prefix(tmp_path):
    cxx = shutil.which("g++")
    if cxx is None:
        pytest.skip("g++ required for actual lease-adapter compile")
    header = Path("compiler/runtime/scoped_runtime.h").resolve()
    p = tmp_path / "lease.cpp"
    p.write_text(f'''#include "{header}"
#include <cstddef>
#include <cstdlib>
#include <cassert>
static int acquired=0,released=0,submits=0,resource=0;
int fort_scope_scratch_acquire_v1(fort_scope_t,std::size_t bytes,fort_scope_scratch_lease_v1* p){{
 ++acquired;if(resource)return FORT_SCOPE_RESOURCE;p->device=std::malloc(bytes);p->token=1;return FORT_SCOPE_OK;}}
int fort_scope_scratch_release_v1(fort_scope_t,uint64_t){{++released;return FORT_SCOPE_OK;}}
{chr(10).join(scratch_batch_helper())}
int main(){{
 auto submit=[](std::size_t,std::size_t,void*){{++submits;return FORT_SCOPE_OK;}};
 assert(fort_tile_submit_batches(1,0,0,0,submit)==FORT_SCOPE_OK&&acquired==0);
 assert(fort_tile_submit_batches(1,5,3,16,submit)==FORT_SCOPE_OK&&acquired==1&&released==1&&submits==2);
 acquired=released=submits=0;resource=1;
 assert(fort_tile_submit_batches(1,5,3,16,submit)==FORT_SCOPE_RESOURCE&&acquired==1&&released==0&&submits==0);
 acquired=released=submits=0;resource=0;
 auto fail=[](std::size_t,std::size_t,void*){{++submits;return FORT_SCOPE_RESOURCE;}};
 assert(fort_tile_submit_batches(1,5,3,16,fail)==FORT_SCOPE_EXECUTION&&acquired==1&&released==1&&submits==1);
}}
''')
    subprocess.run(
        [cxx, "-std=c++17", "-O2", str(p), "-o", str(tmp_path / "lease")], check=True, capture_output=True, timeout=120
    )
    subprocess.run([str(tmp_path / "lease")], check=True, capture_output=True, timeout=30)


@pytest.fixture(scope="module", params=[4, 8])
def compiled_three_dimensional(request, tmp_path_factory):
    from compiler.tests.test_tile_private import SOURCE_3D

    kind = request.param
    directory = tmp_path_factory.mktemp(f"tile-private-3d-real{kind}")
    fc, cxx = shutil.which("gfortran"), shutil.which("g++")
    if fc is None or cxx is None:
        pytest.skip("gfortran and g++ required for original numerical oracle")
    source = SOURCE_3D.replace("real(8)", f"real({kind})").replace("3.0_8", f"3.0_{kind}")
    source = source.replace("(:,:,:)", "(-3:,-3:,-3:)")
    _, analysis, routine, node = fixture(directory, source)
    lowered = lower_tile_private(prove_tile_private(analysis, routine, node))
    arguments = []
    bindings = {symbol: binding for binding, symbol in lowered.bindings}
    lower = {symbol for _, _, symbol in lowered.lower_bounds}
    for argument in abi_arguments(lowered.region.parameters):
        if argument.symbol in lower:
            arguments.append("-3")
        elif argument.dimension:
            arguments.append(("ex", "ey", "ez")[argument.dimension - 1])
        else:
            name = bindings[argument.symbol].name
            arguments.append(name + (".data()" if argument.symbol.rank else ""))
    arguments += ["1", "tz", "ntz", "1", "ty", "nty", "1", "tx", "ntx", "scratch.data()", "stride",
                  "0", "-1", "tx+4", "-1", "ty+4", "-1", "tz+4", "slot", "first+slot"]
    element = "float" if kind == 4 else "double"
    function = "\n".join(generate_tile_function(lowered.region, addressing=lowered.addressing, device=False))
    code = directory / "probe.cpp"
    code.write_text(f'''#include "{Path('compiler/emission/common/templates/common_functions.cuh').resolve()}"
#include "{Path('compiler/runtime/numeric.hpp').resolve()}"
#include <fstream>
using generated_kernels::indexing::F_IDX;
{function}
int main(int argc,char**argv){{
 const int nx=std::atoi(argv[1]),ny=std::atoi(argv[2]),nz=std::atoi(argv[3]);
 const int tx=3,ty=2,tz=3;const std::size_t ex=nx+8,ey=ny+8,ez=nz+8;
 std::vector<{element}>a(ex*ey*ez),b(a.size());
 for(std::size_t i=0;i<a.size();++i){{a[i]=static_cast<{element}>(17*static_cast<int>(i)-3)/{element}(7);b[i]=static_cast<{element}>(500-static_cast<int>(i))/{element}(32);}}
 const std::size_t ntx=nx?(nx-1)/tx+1:0,nty=ny?(ny-1)/ty+1:0,ntz=nz?(nz-1)/tz+1:0;
 const std::size_t total=ntx*nty*ntz,batch=total<3?total:3;
 const std::size_t stride=(std::size_t(tx+4)*(ty+4)*(tz+4)*sizeof({element})+7)&~std::size_t(7);
 std::vector<unsigned char>scratch(batch*stride);
 for(std::size_t first=0;first<total;first+=batch){{
  const std::size_t count=total-first<batch?total-first:batch;
  for(std::size_t slot=0;slot<count;++slot)fort_tile_one_0({','.join(arguments)});
 }}
 std::ofstream out(argv[4],std::ios::binary);
 out.write(reinterpret_cast<const char*>(a.data()),a.size()*sizeof({element}));
 out.write(reinterpret_cast<const char*>(b.data()),b.size()*sizeof({element}));
}}
''')
    driver = directory / "driver.f90"
    driver.write_text(f'''program driver
use stencil
implicit none
integer::nx,ny,nz,i,j,k,q
character(512)::arg,out
real({kind}),allocatable::a(:,:,:),b(:,:,:)
call get_command_argument(1,arg)
read(arg,*)nx
call get_command_argument(2,arg)
read(arg,*)ny
call get_command_argument(3,arg)
read(arg,*)nz
call get_command_argument(4,out)
allocate(a(-3:nx+4,-3:ny+4,-3:nz+4),b(-3:nx+4,-3:ny+4,-3:nz+4))
q=0
do k=-3,nz+4
 do j=-3,ny+4
  do i=-3,nx+4
   a(i,j,k)=real(17*q-3,{kind})/real(7,{kind})
   b(i,j,k)=real(500-q,{kind})/real(32,{kind})
   q=q+1
  enddo
 enddo
enddo
call advance(a,b,nx,ny,nz,3,2,3)
open(10,file=trim(out),access='stream',form='unformatted',status='replace')
write(10)a,b
close(10)
end program
''')
    subprocess.run([fc, "-O2", "-fopenmp", "-ffp-contract=off", "-J", str(directory),
                    str(directory / "source.f90"), str(driver), "-o", str(directory / "native")],
                   check=True, capture_output=True, timeout=120)
    subprocess.run([cxx, "-O2", "-std=c++17", "-ffp-contract=off", str(code), "-o", str(directory / "emitted")],
                   check=True, capture_output=True, timeout=120)
    return directory


@pytest.mark.parametrize("shape", [(0, 7, 5), (7, 0, 5), (7, 5, 0), (1, 1, 1), (7, 5, 4), (13, 7, 5)])
def test_three_dimensional_original_phase_order_and_all_halos_are_bitwise(compiled_three_dimensional, shape):
    values = []
    for mode in ("native", "emitted"):
        path = compiled_three_dimensional / (mode + ".bin")
        subprocess.run([str(compiled_three_dimensional / mode), *map(str, shape), str(path)], check=True,
                       capture_output=True, timeout=30,
                       env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static"})
        values.append(path.read_bytes())
    assert values[0] == values[1]
