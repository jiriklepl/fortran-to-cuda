"""Section-index conversion preserves boundary decisions and host IEEE state."""
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("sanitize", [False, True])
def test_section_index_round_trip_preserves_results_validity_and_exception_flags(tmp_path, sanitize):
    cxx = shutil.which("g++-14") or shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler required")
    source = tmp_path / "check.cpp"
    source.write_text(SOURCE)
    command = [cxx, "-O3", "-std=c++17", "-frounding-math", "-ffp-contract=off",
               "-I", str(Path(__file__).resolve().parents[1] / "runtime"),
               str(source), "-o", str(tmp_path / "check")]
    if sanitize:
        command += ["-fsanitize=undefined,float-cast-overflow", "-fno-sanitize-recover=all"]
    build = subprocess.run(command, text=True, capture_output=True, timeout=60)
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run([str(tmp_path / "check")], text=True, capture_output=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.strip() == "CHECKS 640168 FLAG_DIFFERENCES 0"


SOURCE = r'''
#include <cassert>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>
#include "offload.hpp"
__attribute__((noinline)) long long old_index(long double value,bool &valid) {
 if (!std::isfinite(value) || value < static_cast<long double>(std::numeric_limits<long long>::min()) ||
     value > static_cast<long double>(std::numeric_limits<long long>::max()) || std::trunc(value)!=value) {
  valid=false;return 0;
 }
 return static_cast<long long>(value);
}
__attribute__((noinline)) long long new_index(long double value,bool &valid) {
 return generated_kernels::offload::index(value,valid);
}
int main(){
 const long double low=static_cast<long double>(std::numeric_limits<long long>::min()),high=static_cast<long double>(std::numeric_limits<long long>::max());
 std::vector<long double> values{0.L,-0.L,1.L,-1.L,.5L,-.5L,1.25L,-1.25L,low,high,low-1,high+1,
  std::nextafter(low,-INFINITY),std::nextafter(low,INFINITY),std::nextafter(high,INFINITY),std::nextafter(high,-INFINITY),
  std::numeric_limits<long double>::infinity(),-std::numeric_limits<long double>::infinity(),std::numeric_limits<long double>::quiet_NaN(),
  std::numeric_limits<long double>::denorm_min(),-std::numeric_limits<long double>::denorm_min()};
 uint64_t rng=0x123456789abcdefULL;
 for(int i=0;i<40000;++i){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;long long n;std::memcpy(&n,&rng,sizeof n);values.push_back(static_cast<long double>(n));values.push_back(std::ldexp(static_cast<long double>(n),-(i%67)));}
 size_t checked=0,flag_differences=0;
 for(int mode:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}){
  assert(!std::fesetround(mode));
  for(auto value:values)for(bool initial:{false,true}){
   bool a=initial,b=initial;std::feclearexcept(FE_ALL_EXCEPT);auto x=old_index(value,a);int fa=std::fetestexcept(FE_ALL_EXCEPT);std::feclearexcept(FE_ALL_EXCEPT);auto y=new_index(value,b);int fb=std::fetestexcept(FE_ALL_EXCEPT);
   if(x!=y || a!=b){std::fprintf(stderr,"value mismatch at %zu\n",checked);return 2;}
   if(fa!=fb){if(flag_differences<8)std::fprintf(stderr,"flags old=%d new=%d value=%La mode=%d\n",fa,fb,value,mode);++flag_differences;}++checked;
  }
 }
 std::fesetround(FE_TONEAREST);std::printf("CHECKS %zu FLAG_DIFFERENCES %zu\n",checked,flag_differences);if(flag_differences)return 3;
 return 0;
}
'''
