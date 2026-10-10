"""Fresh native proofs use only checked, safely available numerical metadata."""

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from compiler.analysis import build_execution_plan
from compiler.emission import generate_sources
from compiler.emission.cuda.native_preflight import generate_native_preflight
from compiler.frontend import lower_source
from compiler.offload.config import OffloadConfig
from compiler.offload.preparation import prepare_offload
from compiler.tests.test_source_compute_costs import SOURCE, profile_v2


def make(source=SOURCE, **options):
    function = lower_source(source, "evaluate", source_name="renamed_preflight.f90")
    plan = build_execution_plan(function)
    analysis = prepare_offload(function, plan).analysis
    units = tuple(replace(unit, compute_model={"item_range": [4, 8]}) for unit in analysis.units)
    arguments = dict(source_compute=True, collective=False, planning_available=True,
                     profile_available=True, protected={})
    arguments.update(options)
    result = generate_native_preflight(function, plan, units, "preflight", **arguments)
    return function, plan, units, result


def run_cpp(tmp_path, preflight, main):
    cxx = shutil.which("g++")
    if not cxx:
        pytest.skip("requires a C++ compiler")
    runtime = Path(__file__).resolve().parents[1] / "runtime/offload.hpp"
    source = f'''#include <cassert>
#include <climits>
#include <sys/mman.h>
#include "{runtime}"
using namespace generated_kernels;
{chr(10).join(preflight.cpp)}
int main() {{
{main}
}}
'''
    (tmp_path / "check.cpp").write_text(source)
    build = subprocess.run([cxx, "-std=c++17", "-O2", "check.cpp", "-o", "check"],
                           cwd=tmp_path, text=True, capture_output=True, timeout=60)
    assert build.returncode == 0, build.stderr
    executed = subprocess.run([str(tmp_path / "check")], text=True, capture_output=True, timeout=10)
    assert executed.returncode == 0, executed.stderr


def test_public_metadata_has_no_payload_or_context_arguments():
    function, _, _, result = make()
    scalar = next(symbol for symbol in function.parameters if symbol.name == "n")
    assert result.available
    public = result.public("preflight")
    assert public["parameters"] == [{"name": scalar.cpp_name, "kind": "integer_scalar", "parameter": "n",
                                      "passing": "reference", "integer_kind": "c_int"}]
    assert public["proofs"] == ["empty domain", "outside validated compute item range"]
    assert public["contexts_created"] == public["registrations"] == 0
    assert not public["payload_reads"]
    assert "deferred" in public["startup_cost_proof"]
    cpp = "\n".join(result.cpp)
    assert "fort_scope_" not in cpp
    assert "query_load" not in cpp
    assert "offload::Data" not in cpp


def test_empty_and_out_of_range_domains_prove_native_but_range_endpoints_do_not(tmp_path):
    _, _, _, result = make()
    run_cpp(tmp_path, result, '''
for (int n : {-1, 0, 1, 3, 9, 100}) assert(preflight(&n) == 1);
for (int n : {4, 5, 8}) assert(preflight(&n) == 0);
assert(preflight(nullptr) == 0);
''')


@pytest.mark.parametrize(("loop", "counts"), [
    ("do i=-2,n", [(-3, 1), (0, 1), (1, 0), (5, 0), (6, 1)]),
    ("do i=n,1,-1", [(-1, 1), (0, 1), (3, 1), (4, 0), (8, 0), (9, 1)]),
])
def test_negative_bounds_and_descending_trip_counts(tmp_path, loop, counts):
    _, _, _, result = make(SOURCE.replace("do i=1,n", loop))
    assert result.available
    run_cpp(tmp_path, result, "\n".join(f"{{ int n={n}; assert(preflight(&n)=={expected}); }}" for n, expected in counts))


def test_descriptor_extent_conversion_is_checked_without_reading_arrays(tmp_path):
    _, _, _, result = make(SOURCE.replace("do i=1,n", "do i=1,size(a)"))
    assert result.available
    assert result.parameters[0]["kind"] == "array_extent"
    assert result.parameters[0]["parameter"] == "a"
    assert result.parameters[0]["dimension"] == 1
    run_cpp(tmp_path, result, '''
assert(preflight(0) == 1);
assert(preflight(4) == 0);
assert(preflight(8) == 0);
assert(preflight(9) == 1);
assert(preflight(static_cast<std::size_t>(INT_MAX)+1) == 0);
''')


def test_overflow_unknown_strides_and_divide_by_zero_continue_ordinary_planning(tmp_path):
    source = SOURCE.replace("integer,intent(in)::n", "integer,intent(in)::n,denominator,stride")
    source = source.replace("evaluate(a,n)", "evaluate(a,n,denominator,stride)")
    source = source.replace("do i=1,n", "do i=1,(n+1)/denominator,stride")
    _, _, _, result = make(source)
    assert result.available
    run_cpp(tmp_path, result, '''
int n=3, denominator=1, stride=1;
assert(preflight(&n,&denominator,&stride) == 0);
n=INT_MAX; assert(preflight(&n,&denominator,&stride) == 0);
n=3; denominator=0; assert(preflight(&n,&denominator,&stride) == 0);
denominator=1; stride=0; assert(preflight(&n,&denominator,&stride) == 0);
''')


def test_empty_outer_domain_does_not_read_inner_bounds(tmp_path):
    source = SOURCE.replace("evaluate(a,n)", "evaluate(a,n,m)")
    source = source.replace("a(:)", "a(:,:)").replace("::n", "::n,m").replace("::i", "::i,j")
    source = source.replace("do i=1,n", "do j=1,n\ndo i=1,m")
    source = source.replace("a(i)", "a(i,j)").replace("enddo", "enddo\nenddo")
    _, _, _, result = make(source)
    assert result.available
    run_cpp(tmp_path, result, '''
void *page=mmap(nullptr,4096,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
assert(page != MAP_FAILED);
int n=0; assert(preflight(&n,static_cast<const int*>(page)) == 1);
int m=4; n=1; assert(preflight(&n,&m) == 0);
munmap(page,4096);
''')


def test_item_product_overflow_never_proves_a_native_decision(tmp_path):
    source = SOURCE.replace("a(:)", "a(:,:,:)").replace("::i", "::i,j,k")
    source = source.replace("do i=1,n", "do k=1,n\ndo j=1,n\ndo i=1,n")
    source = source.replace("a(i)", "a(i,j,k)").replace("enddo", "enddo\nenddo\nenddo")
    _, _, _, result = make(source)
    assert result.available
    run_cpp(tmp_path, result, '''int n=INT_MAX; assert(preflight(&n) == 0);''')


@pytest.mark.parametrize(("options", "reason"), [
    ({"source_compute": False}, "original Fortran"),
    ({"collective": True}, "existing-team"),
    ({"planning_available": False}, "complete numerical estimates"),
    ({"profile_available": False}, "complete numerical estimates"),
    ({"protected": {0: frozenset({"n"})}}, "protected"),
])
def test_unproved_participation_or_bounds_do_not_export_preflight(options, reason):
    function, plan, units, _ = make()
    if "protected" in options:
        options = {"protected": {units[0].region.id: frozenset({"n"})}}
    arguments = dict(source_compute=True, collective=False, planning_available=True,
                     profile_available=True, protected={})
    arguments.update(options)
    result = generate_native_preflight(function, plan, units, "preflight", **arguments)
    assert not result.available
    assert reason in result.reason
    assert not result.cpp
    assert not result.fortran


@pytest.mark.parametrize("source", [
    SOURCE.replace("do i=1,n", "if(n>0)then\ndo i=1,n").replace("enddo", "enddo\nendif"),
    SOURCE.replace("do i=1,n", "do i=1,n").replace("enddo", "enddo\ndo i=1,n\na(i)=2*a(i)\nenddo"),
    SOURCE.replace("integer::i", "integer::i,limit").replace("do i=1,n", "limit=n\ndo i=1,limit"),
    SOURCE.replace("evaluate(a,n)", "evaluate(a,n,bounds)").replace("integer::i", "integer,intent(in)::bounds(:)\ninteger::i")
          .replace("do i=1,n", "do i=1,bounds(1)"),
])
def test_multiunit_control_preparation_and_payload_bounds_remain_ordinary_queries(source):
    _, _, _, result = make(source)
    assert not result.available


def test_scoped_interface_exports_the_metadata_only_public_preflight():
    function = lower_source(SOURCE, "evaluate", source_name="independent.f90")
    plan = build_execution_plan(function)
    generated = generate_sources(function, plan, memory_model="scoped",
        offload_config=OffloadConfig(policy="auto", profile=profile_v2(), native_participation="serial"))
    public = generated.scoped["native_preflight"]
    assert public["available"]
    assert public["item_range"] == [65536, 1048576]
    assert public["parameters"][0]["parameter"] == "n"
    interface = next(text for name, text in generated.artifacts.items() if name.endswith("shared_interface.f90"))
    assert ", native_preflight" in interface
    assert "function native_preflight(" in interface
    assert f"bind(C, name='{public['entry']}')" in interface


def test_fortran_metadata_abi_calls_the_cpu_only_preflight(tmp_path):
    cxx, fc = shutil.which("g++"), shutil.which("gfortran")
    if not cxx or not fc:
        pytest.skip("requires native C++ and Fortran compilers")
    _, _, _, result = make(SOURCE.replace("do i=1,n", "do i=1,min(n,size(a))"))
    assert result.available
    runtime = Path(__file__).resolve().parents[1] / "runtime/offload.hpp"
    (tmp_path / "preflight.cpp").write_text(f'#include "{runtime}"\nusing namespace generated_kernels;\n' + "\n".join(result.cpp))
    (tmp_path / "check.f90").write_text("program check\nuse iso_c_binding\nimplicit none\ninterface\n" + "\n".join(result.fortran) +
        "\nend interface\ninteger(c_int)::n\nn=4\nif(native_preflight(8_c_size_t,n)/=0)stop 1\n"
        "n=3\nif(native_preflight(8_c_size_t,n)/=1)stop 2\nend program\n")
    for argv in ([cxx,"-std=c++17","-O2","-c","preflight.cpp","-o","preflight.o"],
                 [fc,"check.f90","preflight.o","-lstdc++","-o","check"], [str(tmp_path / "check")]):
        executed = subprocess.run(argv, cwd=tmp_path, text=True, capture_output=True, timeout=60)
        assert executed.returncode == 0, executed.stdout + executed.stderr
