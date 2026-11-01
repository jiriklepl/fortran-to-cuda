"""Original Fortran versus emitted serial line work, with checked metadata."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.emission.common.abi import abi_arguments
from compiler.emission.cuda.line_snapshot import generate_line_metadata_check
from compiler.emission.cuda.tile_private import generate_tile_function
from compiler.frontend.line_snapshot import prove_line_snapshot
from compiler.frontend.lowering import lower_line_snapshot
from compiler.tests.test_line_snapshot import fixture, source


@pytest.fixture(
    scope="module", params=[(axis, kind, lower) for axis in (1, 2, 3) for kind in (4, 8) for lower in (-2, -5)]
)
def compiled(request, tmp_path_factory):
    axis, kind, lower = request.param
    root = tmp_path_factory.mktemp(f"line-cpu-{axis}-{kind}-{lower}")
    fc, cxx = shutil.which("gfortran-15"), shutil.which("g++")
    if fc is None or cxx is None:
        pytest.skip("GNU Fortran15 and C++ required for original numerical oracle")
    text = source(axis, kind).replace("-2:", f"{lower}:")
    _, analysis, routine, nodes = fixture(root, text)
    lowered = lower_line_snapshot(prove_line_snapshot(analysis, routine, nodes))
    by_symbol = {symbol: binding for binding, symbol in lowered.bindings}
    lows = {symbol for _, _, symbol in lowered.lower_bounds}
    arguments = []
    for arg in abi_arguments(lowered.region.parameters):
        symbol = arg.symbol
        if symbol in lows:
            value = "lb"
        else:
            binding = by_symbol[symbol]
            value = (
                f"extent[{arg.dimension - 1}]"
                if arg.dimension
                else binding.name + ".data()"
                if symbol.rank
                else binding.name
            )
        arguments.append(value)
    args = ",".join(arguments)
    geometry = []
    for loop in lowered.region.outer_loops:
        name = loop.iterator.name
        count = {"i": "nx", "j": "ny", "k": "nz"}[name]
        geometry += ["1", "1", count]
    kernel_args = ",".join(
        [*arguments, *geometry, "scratch.data()", "stride", "0", "lb", "scratch_extent", "slot", "first+slot"]
    )
    function = "\n".join(generate_tile_function(lowered.region, addressing=lowered.addressing, device=False))
    metadata = "\n".join(generate_line_metadata_check(lowered))
    header = Path("compiler/emission/common/templates/common_functions.cuh").resolve()
    numeric = Path("compiler/runtime/numeric.hpp").resolve()
    element = "float" if kind == 4 else "double"
    code = root / "probe.cpp"
    code.write_text(f'''#include "{header}"
#include "{numeric}"
#include <fstream>
#include <cassert>
using generated_kernels::indexing::F_IDX;
{metadata}
{function}
int main(int argc,char**argv){{
 int nx=std::atoi(argv[1]),ny=std::atoi(argv[2]),nz=std::atoi(argv[3]);
 const int mode=std::atoi(argv[4]),lb={lower};
 std::size_t extent[3]={{static_cast<std::size_t>(nx+4-lb),static_cast<std::size_t>(ny+4-lb),static_cast<std::size_t>(nz+4-lb)}};
 std::vector<{element}>a(extent[0]*extent[1]*extent[2]);std::vector<int>mask(a.size());
 for(int k=lb;k<=nz+3;++k)for(int j=lb;j<=ny+3;++j)for(int i=lb;i<=nx+3;++i){{
  const auto at=static_cast<std::size_t>(i-lb)+extent[0]*(static_cast<std::size_t>(j-lb)+extent[1]*static_cast<std::size_t>(k-lb));
  a[at]=static_cast<{element}>(i*i*i*i+3*j*j*j*j+5*k*k*k*k+17*i*j)/static_cast<{element}>(7);
  const int residue=((i+2*j+3*k+mode)%5+5)%5;
  mask[at]=mode==0?0:residue==0?1:residue==1?-1:0;
 }}
 const std::size_t scratch_extent=extent[{axis - 1}];std::size_t total=999;
 if(mode==10){{assert(fort_line_metadata_check({args},lb,scratch_extent+1,&total)==2);assert(total==0);return 0;}}
 if(mode==11){{assert(fort_line_metadata_check({args},lb+1,scratch_extent,&total)==2);assert(total==0);return 0;}}
 if(mode==12){{{"nx" if axis == 1 else "ny" if axis == 2 else "nz"}=INT_MAX;assert(fort_line_metadata_check({args},lb,scratch_extent,&total)==2);assert(total==0);return 0;}}
 if(mode==13){{{ {"i": "nx", "j": "ny", "k": "nz"}[lowered.region.outer_loops[0].iterator.name] }=0;
  extent[0]=extent[1]=extent[2]=0;
  assert(fort_line_metadata_check({args},INT_MAX,SIZE_MAX,&total)==1);assert(total==0);return 0;}}
 if(mode==14){{{"nx" if axis == 1 else "ny" if axis == 2 else "nz"}=0;assert(fort_line_metadata_check({args},lb,scratch_extent,&total)==2);assert(total==0);return 0;}}
 assert(fort_line_metadata_check({args},lb,scratch_extent,&total)==0);
 const std::size_t stride=(scratch_extent*sizeof({element})+7)&~std::size_t(7),batch=std::min<std::size_t>(3,total);
 std::vector<unsigned char>scratch(batch*stride);
 for(std::size_t first=0;first<total;first+=batch){{
  const auto count=std::min(batch,total-first);
  for(std::size_t slot=0;slot<count;++slot)fort_tile_one_0({kernel_args});
 }}
 std::ofstream out(argv[5],std::ios::binary);
 out.write(reinterpret_cast<const char*>(a.data()),a.size()*sizeof({element}));
 out.write(reinterpret_cast<const char*>(mask.data()),mask.size()*sizeof(int));
}}
''')
    driver = root / "driver.f90"
    driver.write_text(f"""program driver
use lines
implicit none
integer::nx,ny,nz,i,j,k,mode,residue
character(512)::arg,out
real({kind}),allocatable::a(:,:,:)
integer,allocatable::mask(:,:,:)
call get_command_argument(1,arg)
read(arg,*)nx
call get_command_argument(2,arg)
read(arg,*)ny
call get_command_argument(3,arg)
read(arg,*)nz
call get_command_argument(4,arg)
read(arg,*)mode
call get_command_argument(5,out)
allocate(a({lower}:nx+3,{lower}:ny+3,{lower}:nz+3),mask({lower}:nx+3,{lower}:ny+3,{lower}:nz+3))
do k={lower},nz+3
 do j={lower},ny+3
  do i={lower},nx+3
   a(i,j,k)=real(i*i*i*i+3*j*j*j*j+5*k*k*k*k+17*i*j,{kind})/real(7,{kind})
   residue=modulo(i+2*j+3*k+mode,5)
   mask(i,j,k)=0
   if(mode/=0)then
    if(residue==0)mask(i,j,k)=1
    if(residue==1)mask(i,j,k)=-1
   endif
  enddo
 enddo
enddo
call advance(a,mask,nx,ny,nz)
open(10,file=trim(out),access='stream',form='unformatted',status='replace')
write(10)a,mask
close(10)
end program
""")
    subprocess.run(
        [
            fc,
            "-O2",
            "-fopenmp",
            "-ffp-contract=off",
            "-fcheck=bounds",
            "-J",
            str(root),
            str(root / "source.f90"),
            str(driver),
            "-o",
            str(root / "native"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    subprocess.run(
        [cxx, "-O2", "-std=c++17", "-ffp-contract=off", str(code), "-o", str(root / "emitted")],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return root


@pytest.mark.parametrize(("dimensions", "mask"), [((5, 3, 7), 0), ((8, 5, 3), 1), ((7, 4, 5), 2)])
def test_original_fortran_and_serial_emitted_line_fields_masks_and_halos_match(compiled, dimensions, mask):
    root = compiled
    args = [*map(str, dimensions), str(mask)]
    native = root / ("native-" + str(mask) + ".bin")
    emitted = root / ("emitted-" + str(mask) + ".bin")
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "4",
        "OMP_THREAD_LIMIT": "4",
        "OMP_DYNAMIC": "FALSE",
        "OMP_SCHEDULE": "static",
    }
    subprocess.run(
        [str(root / "native"), *args, str(native)], check=True, capture_output=True, text=True, timeout=20, env=env
    )
    subprocess.run([str(root / "emitted"), *args, str(emitted)], check=True, capture_output=True, text=True, timeout=20)
    assert native.read_bytes() == emitted.read_bytes()


@pytest.mark.parametrize("mode", [10, 11, 12, 13, 14])
def test_line_metadata_rejects_shape_integer_and_empty_consumer_before_work(compiled, mode):
    subprocess.run(
        [str(compiled / "emitted"), "5", "3", "7", str(mode), str(compiled / "unused.bin")],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert not (compiled / "unused.bin").exists()


@pytest.mark.parametrize(("factor", "expected"), [(1, 0), (1073741824, 2)])
def test_original_integer_predicate_is_checked_before_payload_work(tmp_path, factor, expected):
    cxx = shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler required for emitted metadata check")
    original = source().replace("mask(i,j,k)==0", f"mask(i,j,k)==0.and.i*{factor}>0")
    _, analysis, routine, nodes = fixture(tmp_path, original)
    lowered = lower_line_snapshot(prove_line_snapshot(analysis, routine, nodes))
    lower_symbols = {symbol for _, _, symbol in lowered.lower_bounds}
    args = []
    for arg in abi_arguments(lowered.region.parameters):
        args.append(
            "12" if arg.dimension else "nullptr" if arg.symbol.rank else "-2" if arg.symbol in lower_symbols else "3"
        )
    checker = "\n".join(generate_line_metadata_check(lowered))
    probe = tmp_path / "predicate.cpp"
    probe.write_text(
        "#include <cstddef>\n#include <cstdint>\n"
        + checker
        + f"\nint main(){{std::size_t total=0; return fort_line_metadata_check({','.join(args)},-2,12,&total)=={expected}?0:1;}}\n"
    )
    executable = tmp_path / "predicate"
    subprocess.run(
        [cxx, "-O2", "-std=c++17", str(probe), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=10)


def test_payload_integer_arithmetic_declines_metadata_only_path(tmp_path):
    from compiler.ir import CompilationError

    original = source().replace("mask(i,j,k)==0", "mask(i,j,k)+1==1")
    _, analysis, routine, nodes = fixture(tmp_path, original)
    lowered = lower_line_snapshot(prove_line_snapshot(analysis, routine, nodes))
    with pytest.raises(CompilationError, match="payload-dependent integer"):
        generate_line_metadata_check(lowered)
