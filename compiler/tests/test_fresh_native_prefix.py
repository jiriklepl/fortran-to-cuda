"""Fresh native metadata preparation does not create a coherent context."""

import json
import re
import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder

SOURCE = """module metadata
implicit none
integer,protected::tile(3)=[2,3,4]
end module
module fresh_prefix
use metadata,only:tile
implicit none
contains
subroutine advance(a,b,n,seen,flag)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer,intent(in)::n
integer,intent(out)::seen
logical,intent(in)::flag
integer::i,stride,extra
stride=tile(1)
extra=tile(2)
seen=stride+extra
do i=1,n
b(i)=a(i)+real(stride+extra,8)
enddo
end subroutine
end module
"""


def emit(directory, source=SOURCE, *, entry="fresh_prefix::advance", captures=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "source.f90"
    path.write_text(source)
    fact = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {name: dict(fact) for name in ("argument::a", "argument::b", "metadata::tile")}}
    facts["captures"].update(captures or {})
    builder = ScopeBuilder([path], entry, facts=facts,
        options=CompilerOptions(), config=OffloadConfig(policy="sections", scope_execution="graph"))
    outputs, report = builder.run()
    text = outputs[report["sources"][str(path)]["replacement"]]
    return path, outputs, report, text


def prefixes(report):
    return [segment for owner in report["scopes"] for segment in owner["planning_segments"]
            if segment.get("fresh_native_prefix") is not None]


def test_exact_readonly_prefix_delays_create_and_preserves_active_context_protocol(tmp_path):
    original, _, report, text = emit(tmp_path)
    assert original.read_text() == SOURCE
    selected = prefixes(report)
    assert len(selected) == 2
    assert all(item["fresh_native_prefix"]["resources"] == ["metadata::tile"] for item in selected)
    assert all(not item["fresh_native_prefix"]["definitions_authorized"] for item in selected)
    assert all(not item["fresh_native_prefix"]["estimates_available"] for item in selected)
    owner = report["scopes"][0]
    assert not owner["estimate_available"]
    assert owner["planning_reason"] == "original native prefix before context creation is unpriced"
    guard = f"if ({owner['owner']} == 0 .and. .not. "
    assert text.count(guard) == 2
    for assignment in ("stride=tile(1)", "extra=tile(2)"):
        start = text.index(guard, text.index(assignment) - len(guard) - 100)
        raw, managed = text[start:].split("else\n", 1)
        assert assignment in raw
        assert "fort_scope" not in raw.split("then\n", 1)[1]
        assert "fort_scope_create" in managed
        assert "fort_scope_plan_validate" in managed
        assert "fort_scope_access_begin_v1" in managed
    numerical = [item for item in owner["planning_segments"] if item["gpu_leaves"]]
    assert len(numerical) == 1
    assert numerical[0]["operations"]["planning_segments"][0]["source_read_requirements"]


def test_numerical_use_of_same_root_keeps_managed_native_read(tmp_path):
    source = SOURCE.replace("real(stride+extra,8)", "real(stride+extra+tile(i),8)")
    _, _, report, text = emit(tmp_path, source)
    assert report["scopes"]
    assert not prefixes(report)
    assert "fort_scope_access_begin_v1" in text


@pytest.mark.parametrize("prefix", ["if(flag) stride=tile(1)", "call opaque()\nstride=tile(1)"])
def test_conditional_or_call_prefix_keeps_reached_guards_and_hooks(tmp_path, prefix):
    source = SOURCE.replace("stride=tile(1)", prefix)
    if "opaque" in prefix:
        source = source.replace("integer::i,stride,extra", "integer::i,stride,extra\nexternal opaque")
    _, _, report, text = emit(tmp_path, source)
    assert report["scopes"]
    assert not prefixes(report)
    assert "fort_scope_access_begin_v1" in text
    if "if(flag)" in prefix:
        assert "if (flag) then" in text.lower()


def test_array_write_in_original_prefix_is_not_unmanaged(tmp_path):
    source = SOURCE.replace("integer,protected::tile", "integer::tile").replace(
        "stride=tile(1)", "tile(1)=n\nstride=tile(1)")
    _, _, report, text = emit(tmp_path, source)
    assert report["scopes"]
    assert not prefixes(report)
    assert "fort_scope_host_end" in text


def test_borrowed_helper_never_uses_fresh_owner_route(tmp_path):
    source = SOURCE.rsplit("end module", 1)[0] + """subroutine outer(a,b,n,seen,flag)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer,intent(in)::n
integer,intent(out)::seen
logical,intent(in)::flag
call advance(a,b,n,seen,flag)
end subroutine
end module
"""
    _, _, report, text = emit(tmp_path, source, entry="fresh_prefix::outer")
    assert report["scopes"][0]["module_coordinators"]
    assert not prefixes(report)
    assert "fort_scope_access_begin_v1" in text


def test_numerical_child_use_of_metadata_root_prevents_prefix_bypass(tmp_path):
    loop = "do i=1,n\nb(i)=a(i)+real(stride+extra,8)\nenddo"
    child = """subroutine compute(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer,intent(in)::n
integer::i
do i=1,n
b(i)=a(i)+real(tile(i),8)
enddo
end subroutine
"""
    source = SOURCE.replace(loop, "call compute(a,b,n)").rsplit("end module", 1)[0] + child + "end module\n"
    _, _, report, _ = emit(tmp_path, source)
    assert report["scopes"]
    assert report["scopes"][0]["module_coordinators"]
    assert not prefixes(report)


@pytest.mark.native
@pytest.mark.parametrize("case", ["zero-trip", "later-descriptor-fallback", "borrowed"])
def test_repeated_original_metadata_and_local_fallback_match_native(tmp_path, case):
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    cxx = shutil.which("g++-14") or shutil.which("g++")
    if not fc or not cxx:
        pytest.skip("Fortran and C++ compilers required")
    source = SOURCE.replace("end module\nmodule fresh_prefix", """contains
subroutine update(value)
integer,intent(in)::value
tile=[value,value+1,value+2]
end subroutine
end module
module fresh_prefix""", 1).replace("do i=1,n", "if(n>0)then\ndo i=1,n").replace(
        "enddo\nend subroutine", "enddo\nendif\nend subroutine")
    entry = "fresh_prefix::advance"
    if case == "borrowed":
        entry = "fresh_prefix::outer"
        source = source.rsplit("end module", 1)[0] + """subroutine outer(a,b,n,seen,flag)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer,intent(in)::n
integer,intent(out)::seen
logical,intent(in)::flag
integer,save::visits=0
visits=visits+1
call advance(a,b,n,seen,flag)
seen=seen+100*visits
end subroutine
end module
"""
    else:
        source = source.replace("integer::i,stride,extra", "integer::i,stride,extra\ninteger,save::visits=0")
        source = source.replace("stride=tile(1)", "visits=visits+1\nstride=tile(1)")
        source = source.replace("seen=stride+extra", "seen=stride+extra+100*visits")
    captures = None
    if case == "later-descriptor-fallback":
        captures = {"argument::b": {"storage": "stable", "initialized": "whole", "escapes": False,
                    "allocation_changes": False, "expected_descriptor": {"lower_bounds": [1], "extents": [8]}}}
    original, outputs, report, _ = emit(tmp_path / "generated", source, entry=entry, captures=captures)
    assert len(prefixes(report)) == (0 if case == "borrowed" else 2)
    build = tmp_path / "build"
    build.mkdir()
    for name, text in outputs.items():
        target = build / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    driver = build / "driver.f90"
    driver.write_text("""program verify
use iso_c_binding,only:c_int
use fresh_prefix,only:ENTRY
use metadata,only:update
implicit none
real(8),allocatable::a(:),b(:)
integer::seen,iteration,i,unit
#ifdef FORT_CANDIDATE
interface
integer(c_int) function prefix_contexts() bind(C)
import c_int
end function
end interface
#endif
allocate(a(-2:4),b(5:11))
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do iteration=1,3
call update(iteration)
do i=-2,4
a(i)=real(i+iteration,8)*0.25_8
enddo
b=-99.0_8
call ENTRY(a,b,COUNT,seen,.false.)
if(seen/=2*iteration+1+100*iteration) error stop 'prefix replay or stale metadata'
do i=0,COUNT-1
if(b(5+i)/=a(-2+i)+real(2*iteration+1,8)) error stop 'local native fallback result'
enddo
if(any(b(5+COUNT:)/=-99.0_8)) error stop 'untouched halo changed'
write(unit)seen,a,b
enddo
#ifdef FORT_CANDIDATE
if(prefix_contexts()/=CREATES) error stop 'wrong owning or borrowed context path'
#endif
close(unit)
print *, 'fresh metadata and zero-trip source preserved'
end program
""".replace("ENTRY", entry.split("::")[-1]).replace(
        "COUNT", "3" if case == "later-descriptor-fallback" else "0").replace(
        "CREATES", "3" if case == "borrowed" else "0"))
    commands = build / "commands.jsonl"

    def run(argv):
        result = subprocess.run(list(map(str, argv)), cwd=build, capture_output=True, text=True, timeout=90)
        with commands.open("a") as stream:
            stream.write(json.dumps({"argv": list(map(str, argv)), "returncode": result.returncode,
                                     "stdout": result.stdout, "stderr": result.stderr}) + "\n")
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    flags = [fc, "-O2", "-cpp", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", "-ffp-contract=off"]
    native = build / "native"
    run([*flags, original, driver, "-o", native])
    reference = run([native])
    expected_fields = (build / "fields.bin").read_bytes()
    objects = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, item in enumerate(report["build_sources"]):
            if item["language"] != "fortran" or item["role"] != role:
                continue
            target = build / f"fortran-{index}.o"
            run([*flags, "-I", build, "-J", build, "-c", build / item["path"], "-o", target])
            objects.append(target)
    runtime = build / "runtime.o"
    run([cxx, "-O1", "-std=c++17", "-fopenmp", "-DFORT_SCOPE_CPU_TEST", "-x", "c++",
         "-I", build, "-c", build / "scoped_runtime.cu", "-o", runtime])
    names = {name for path, text in outputs.items() if path.endswith("shared_interface.f90")
             for name in re.findall(r"bind\s*\(\s*C\s*,\s*name\s*=\s*['\"]([^'\"]+)['\"]", text, re.I)}
    assert names
    configurations = {name: body for path, text in outputs.items() if path.endswith("shared_entry.cu")
                      for body, name in re.findall(
                          r'(extern "C" int (\w+_configure)\(fort_scope_t fort_context\) \{.*?\n\})',
                          text, re.S)}
    assert configurations
    assert set(configurations) <= names
    stubs = build / "unreached.cpp"
    create = ('++contexts; return __real_fort_scope_create(device,out);'
              if case == "borrowed" else 'std::abort();')
    stubs.write_text('#include <cstdlib>\n#include "scoped_runtime.h"\n'
        + '#define FORT_SHARED_CHECK(expression) do { int status=(expression); if(status) return status; } while(0)\n'
        + '\n'.join(configurations.values()) + '\n' + '\n'.join(
        f'extern "C" int {name}(...) {{ std::abort(); }}' for name in sorted(names - configurations.keys()))
        + '\nstatic int contexts=0;\nextern "C" int prefix_contexts(){return contexts;}\n'
        + 'extern "C" int __real_fort_scope_create(int,fort_scope_t*);\n'
        + 'extern "C" int __wrap_fort_scope_create(int device,fort_scope_t*out){' + create + '}\n')
    stubs_object = build / "unreached.o"
    run([cxx, "-O2", "-c", stubs, "-o", stubs_object])
    candidate = build / "candidate"
    run([*flags, "-DFORT_CANDIDATE", driver, *objects, runtime, stubs_object, "-lstdc++", "-Wl,--wrap=fort_scope_create",
         "-o", candidate])
    result = run([candidate])
    assert result.stdout == reference.stdout
    assert (build / "fields.bin").read_bytes() == expected_fields
    assert "array temporary" not in result.stderr.lower()
