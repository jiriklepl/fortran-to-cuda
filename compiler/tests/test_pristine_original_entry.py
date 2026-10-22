"""Original helper entries keep their original team without managed hooks."""

import os
import re
import shutil
import subprocess
from hashlib import sha256
from unittest.mock import patch

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes import segments
from compiler.scopes.graph import GraphOwner
from compiler.scopes.joined_team import original_team_source
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_joined_team_source import write_fortran
from compiler.tests.test_scoped_planning_entries import calibration


def source(style):
    producer = "!$omp do\ndo i=-2,n-3\nb(i)=2*a(i)+real(i,8)\nenddo\n!$omp end do\n"
    correction = "!$omp do\ndo i=-2,n-3\nb(i)=b(i)+sum(a)\nenddo\n!$omp end do\n"
    consumer = "!$omp do\ndo i=-2,n-3\nc(i)=a(i)+b(i)\nenddo\n!$omp end do\n"
    team = ("!$omp parallel private(i) shared(a,b,c,n,flag) default(none)\n" + producer
            + ("if(flag)then\n" + correction + "endif\n" if style == "joined" else "")
            + consumer + "!$omp end parallel\n")
    declarations = ("real(8),intent(in)::a(-2:)\nreal(8),intent(inout)::b(-2:),c(-2:)\n"
                    "integer,intent(in)::n\nlogical,intent(in)::flag\n")
    return ("module original_entry_fixture\nimplicit none\ninteger::visits=0\ncontains\n"
            "subroutine child(a,b,c,n,flag)\n" + declarations + "integer::i\n"
            "visits=visits+1\n" + team + "visits=visits+11\nend subroutine\n"
            "subroutine advance(a,b,c,n,flag)\n" + declarations
            + "call child(a,b,c,n,flag)\nend subroutine\nend module\n")


def emit(directory, style, *, decline_native=False):
    directory.mkdir(parents=True, exist_ok=True)
    original = directory / "original.f90"
    original.write_text(source(style))
    builder = ScopeBuilder([original], "original_entry_fixture::advance",
        facts={"schema_version": 1, "participation": "serial",
               "sources": {str(original): sha256(original.read_bytes()).hexdigest()},
               "captures": {"argument::" + name: dict(FACT) for name in ("a", "b", "c")}},
        options=CompilerOptions(opt_level=0), config=OffloadConfig("auto", calibration()))
    retained = []
    initialize, fragment = GraphOwner.__init__, segments.fragment

    def retain(owner, *args, **kwargs):
        initialize(owner, *args, **kwargs)
        retained.append(owner)

    def independent_fragment(*args, **kwargs):
        if decline_native and kwargs.get("retain_proof_analysis"):
            raise CompilationError("test unavailable independent native sections")
        return fragment(*args, **kwargs)

    with patch.object(GraphOwner, "__init__", retain), patch.object(segments, "fragment", independent_fragment):
        outputs, manifest = builder.run()
    owner, = retained
    child = owner.members["original_entry_fixture::child"]
    text = outputs[manifest["sources"][str(original)]["replacement"]]
    return original, outputs, manifest, text, owner, child


@pytest.mark.parametrize("style", ["combined", "joined"])
def test_original_entry_routes_directly_to_authenticated_whole_team(tmp_path, style):
    _original, _outputs, manifest, text, _owner, child = emit(tmp_path, style)
    scope, = [unit.scope for unit in child.units if unit.leaves]
    public = scope.public()["combined_numerical_team" if style == "combined" else "joined_team"]
    assert public["pristine_original_entry"]["source_eligible"]
    guard = f".not. ({child.control_guard})"
    if style == "combined":
        select = text.index(f"if ({guard}) fort_team_pristine = .true.")
        branch = text.index("if (fort_team_pristine) then", select)
        fork = text.index("!$omp parallel", branch)
        join = text.index("!$omp end parallel", fork)
        assert "!$omp master" not in text[branch:join]
        assert "!$omp barrier" not in text[branch:join]
        after = text[join:]
        assert after.index("if (fort_team_native) then") < after.index("fort_scope_host_end(")
    else:
        branch = text.index(f"if ({guard}) then")
        fork = text.index("!$omp parallel", branch)
        join = text.index("!$omp end parallel", fork)
        assert "!$omp master" not in text[branch:join]
        assert "!$omp barrier" not in text[branch:join]
        assert "fort_" not in text[fork:join]
        assert text.index("else\n", join) < text.index("!$omp master", join)
    assert child.control_guard + " = .false." in text[:branch]
    assert "fort_team_owned = .false." in text[:branch]
    assert manifest["scope_execution"] == "graph"


def test_unmanaged_team_does_not_require_optional_native_acquisition_proof(tmp_path):
    _original, _outputs, _manifest, text, _owner, child = emit(tmp_path, "combined", decline_native=True)
    scope, = [unit.scope for unit in child.units if unit.leaves]
    assert scope.cpu_native is None
    assert scope.public()["combined_numerical_team"]["pristine_original_entry"]["source_eligible"]
    assert f"if (.not. ({child.control_guard})) fort_team_pristine = .true." in text
    assert "fort_team_native_begin" not in text
    assert "if (fort_team_pristine) then" in text
    assert "=> run_team" in text  # The managed alternative remains available.


@pytest.mark.parametrize("style", ["combined", "joined"])
def test_pristine_source_hash_cannot_be_replaced(tmp_path, style):
    _original, _outputs, _manifest, _text, _owner, child = emit(tmp_path, style)
    scope, = [unit.scope for unit in child.units if unit.leaves]
    with pytest.raises(CompilationError, match="authenticated source span"):
        original_team_source(child.builder, scope.original_nodes, scope.first, scope.last, "0" * 64)


def test_labeled_team_is_not_duplicated_for_an_optional_fastpath(tmp_path):
    original = source("joined").replace("b(i)=2*a(i)+real(i,8)", "100 b(i)=2*a(i)+real(i,8)")
    path = tmp_path / "labeled.f90"
    path.write_text(original)
    builder = ScopeBuilder([path], "original_entry_fixture::advance",
        facts={"schema_version": 1, "participation": "serial",
               "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
               "captures": {"argument::" + name: dict(FACT) for name in ("a", "b", "c")}},
        options=CompilerOptions(opt_level=0), config=OffloadConfig("auto", calibration()))
    outputs, report = builder.run()
    replacement = report["sources"].get(str(path), {}).get("replacement")
    text = outputs[replacement] if replacement else original
    assert len(re.findall(r"^\s*100\s+b\(", text, re.M)) == 1


DRIVER = """program verify
use original_entry_fixture, only:child,visits
use iso_c_binding, only:c_int64_t
implicit none
interface
 function parallel_count() bind(c) result(n)
 import c_int64_t
 integer(c_int64_t)::n
 end function
 function barrier_count() bind(c) result(n)
 import c_int64_t
 integer(c_int64_t)::n
 end function
end interface
real(8),allocatable::a(:),b(:),c(:)
integer,parameter::sizes(3)=[0,1,8]
integer::shape,n,mode,repeat,i,j,unit
logical::flag
real(8)::extra
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=1,size(sizes)
n=sizes(shape)
allocate(a(-3:n+4),b(-7:n),c(-9:n-2))
do i=lbound(a,1),ubound(a,1)
a(i)=real(i+5,8)*0.25d0
enddo
do mode=1,2
flag=mode==2
do repeat=1,2
b=-99.d0
c=-77.d0
call child(a,b,c,n,flag)
extra=0
!EXTRA
do j=0,n-1
if(b(-7+j)/=2*a(-3+j)+real(-2+j,8)+extra) error stop 'native producer'
if(c(-9+j)/=a(-3+j)+b(-7+j)) error stop 'native consumer'
enddo
write(unit)n,mode,repeat,lbound(a),ubound(a),lbound(b),ubound(b),lbound(c),ubound(c),a,b,c,visits
enddo
enddo
deallocate(a,b,c)
enddo
if(visits/=144) error stop 'original helper did not execute exactly once'
write(unit)parallel_count(),barrier_count()
close(unit)
print *, 'PRISTINE_ORIGINAL_ENTRY_OK'
end program
"""


@pytest.mark.native
@pytest.mark.parametrize(("style", "decline_native", "root_hit", "strided"), [
    ("combined", False, False, False), ("combined", True, False, False), ("joined", False, False, False),
    ("combined", False, True, False), ("combined", True, True, False), ("joined", False, True, False),
    ("combined", False, True, True), ("joined", False, True, True)])
def test_original_helper_runtime_fields_and_openmp_calls_match_original(tmp_path, style, decline_native, root_hit, strided):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    cc = shutil.which("gcc")
    if not fortran or not cc:
        pytest.skip("C and Fortran compilers required")
    original, outputs, report, _text, _owner, _child = emit(tmp_path, style, decline_native=decline_native)
    build = tmp_path / "native-entry"
    paths = write_fortran(outputs, report, build)
    pattern = (r"\b(?:function|subroutine)\s+(\w+)\s*\([^)]*\)\s*bind\s*\(\s*c"
               r"(?:\s*,\s*name\s*=\s*['\"]([^'\"]+)['\"])?\s*\)")
    symbols = set()
    for path in paths:
        symbols.update(name or procedure.lower() for procedure, name in re.findall(pattern,
            re.sub(r"&\s*\n\s*&?", "", path.read_text()), re.I))
    assert symbols
    if root_hit:
        # A source-control test, not a model/cache test: force the optional
        # performance hint to select its always-legal original CPU route.
        # Runtime cache/generation/TTL guards have separate C ABI coverage.
        symbols -= {"fort_scope_native_hint_begin_v1", "fort_scope_native_hint_finish_v1", "fort_scope_serial_caller"}
    allowed = ("#include <stddef.h>\nint fort_scope_serial_caller(void) {return 1;}\n"
        "int fort_scope_native_hint_begin_v1(const long long *words,size_t n,void *observation,int *hit) {"
        "(void)words;(void)n;(void)observation;*hit=1;return 0;}\n"
        "int fort_scope_native_hint_finish_v1(void *observation) {"
        "(void)observation;fputs(\"unexpected finish on native hit\\n\",stderr);abort();}\n" if root_hit else "")
    sentinels = build / "sentinels.c"
    sentinels.write_text("#include <stdio.h>\n#include <stdlib.h>\n#include <stdatomic.h>\n"
        + "\n".join(f'void {name}(void) {{ fputs("unexpected native ABI call: {name}\\n",stderr); abort(); }}'
                    for name in sorted(symbols)) + "\n"
        + allowed + "static _Atomic long long forks,barriers;\n"
        + "void __real_GOMP_parallel(void (*fn)(void *),void *data,unsigned threads,unsigned flags);\n"
        + "void __wrap_GOMP_parallel(void (*fn)(void *),void *data,unsigned threads,unsigned flags) {"
          "atomic_fetch_add(&forks,1);__real_GOMP_parallel(fn,data,threads,flags);}\n"
        + "void __real_GOMP_barrier(void);\nvoid __wrap_GOMP_barrier(void) {"
          "atomic_fetch_add(&barriers,1);__real_GOMP_barrier();}\n"
        + "long long parallel_count(void) {return atomic_load(&forks);}\n"
        + "long long barrier_count(void) {return atomic_load(&barriers);}\n")
    driver = build / "driver.f90"
    driver_source = DRIVER.replace("!EXTRA", "if(flag)extra=sum(a)" if style == "joined" else "")
    if strided:
        driver_source = (driver_source.replace("call child(a,b,c,n,flag)", "call child(a(::2),b(::2),c(::2),n,flag)")
                         .replace("sum(a)", "sum(a(::2))").replace("(-7+j)", "(-7+2*j)")
                         .replace("(-3+j)", "(-3+2*j)").replace("(-9+j)", "(-9+2*j)"))
    if root_hit:
        driver_source = driver_source.replace("only:child,visits", "only:advance,visits").replace("call child(", "call advance(")
    driver.write_text(driver_source)
    flags = [fortran, "-O2", "-std=f2018", "-cpp", "-fopenmp", "-ffree-line-length-none",
             "-fcheck=all,array-temps", "-finit-logical=true", "-ffp-contract=off"]

    def checked(command):
        result = subprocess.run(command, cwd=build, capture_output=True, text=True, timeout=60,
            env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"})
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    checked([cc, "-std=c11", "-c", str(sentinels), "-o", str(build / "sentinels.o")])
    link = [str(driver), str(build / "sentinels.o"), "-Wl,--wrap=GOMP_parallel", "-Wl,--wrap=GOMP_barrier"]
    checked([*flags, str(original), *link, "-o", str(build / "original")])
    checked([str(build / "original")])
    expected = (build / "fields.bin").read_bytes()
    checked([*flags, *map(str, paths), *link, "-o", str(build / "candidate")])
    result = checked([str(build / "candidate")])
    assert "PRISTINE_ORIGINAL_ENTRY_OK" in result.stdout
    assert "array temporary" not in result.stderr.lower()
    assert (build / "fields.bin").read_bytes() == expected
