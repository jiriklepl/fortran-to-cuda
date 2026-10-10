"""Original-team regions use a bounded shared dispatcher with team workers."""

from dataclasses import replace
from hashlib import sha256
import shutil
import subprocess

from fparser.two import Fortran2003 as F
from fparser.two.utils import walk
import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT


SOURCE = """module mixed
implicit none
contains
subroutine step(a,b,n)
real(8),intent(inout)::a(:)
real(8),intent(in)::b(:)
integer,intent(in)::n
integer::i
do i=1,n
a(i)=2*b(i)
enddo
!$omp parallel private(i)
!$omp do
do i=1,n
a(i)=2*b(i)
enddo
!$omp end do
!$omp do
do i=1,n
a(i)=2*b(i)
enddo
!$omp end do
!$omp end parallel
end subroutine
end module
"""


def example(tmp_path, source=SOURCE):
    path = tmp_path / "mixed.f90"
    path.write_text(source)
    builder = ScopeBuilder([path], "mixed::step", options=CompilerOptions(),
                           config=OffloadConfig(policy="sections"), facts={
                               "schema_version": 1, "participation": "serial",
                               "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
                               "captures": {"argument::a": FACT, "argument::b": FACT}})
    originals = tuple(builder.entry.execution.content)
    loops = tuple(walk(builder.entry.execution, F.Block_Nonlabel_Do_Construct))
    # fparser may attach PARALLEL to the first associated DO. Use the exact
    # original suffix, retaining every directive for the joined proof.
    joined = originals[originals.index(loops[0]) + 1:]
    return builder, loops, joined


def test_original_worksharing_regions_share_a_team_dispatcher_and_computation(tmp_path):
    builder, loops, joined = example(tmp_path)
    inline = builder.inline
    serial, = inline.prepare((loops[0],))
    team = [inline.prepare_worksharing(loop, builder.analysis.worksharing_completion(
                builder.entry.qualified, joined, (loop,)), preceding=(loops[0],)) for loop in loops[1:]]
    serial_name = serial.fort_inline_region
    first, second = (node.fort_inline_region for node in team)
    assert inline.entries[first] == inline.entries[second]
    assert inline.entries[first] != inline.entries[serial_name]
    assert "team" not in inline.generated[serial_name].scoped
    assert inline.generated[first].scoped["team"]["available"]
    assert inline.generated[first].scoped["team"]["host_threads"] == 4

    handles = {root: "handle_" + str(index) for index, root in enumerate(inline.arrays)}
    values = {root: binding.name for root, binding in inline.bindings.items()}
    imports = []
    for node in (serial, *team):
        call = inline.call(node)
        inline.emit_call(call, handles, values, imports, query=True)
        collective = node is not serial
        execution = inline.emit_call(call, handles, values, imports, query=False,
                                     collective=collective, status="returned", check_status=False)
        assert execution[0].startswith("returned = ")
        assert not any("error stop" in line for line in execution)
    inline.finish()
    source = builder.outputs[inline.path]
    assert "public :: run, query, run_team" in source
    serial_body = source.split("function run(", 1)[1].split("end function run", 1)[0]
    team_body = source.split("function run_team(", 1)[1].split("end function run_team", 1)[0]
    assert "case (1)" in serial_body and "case (2)" not in serial_body
    assert "case (1)" not in team_body and "case (2)" in team_body and "case (3)" in team_body
    assert "fort_entry_run_team_2 => run_team" in source
    assert inline.public()["team_dispatcher"].endswith("::run_team")
    assert [item["execution_participation"] for item in inline.public()["regions"]] == [
        "serial_coordinator", "qualified_original_team", "qualified_original_team"]
    variants, = builder.variants.public()["procedures"]
    assert len(variants["variants"]) == 1  # Team dispatch adds no per-region wrapper variant.


def test_worksharing_dispatch_cannot_execute_with_serial_participation(tmp_path):
    builder, loops, joined = example(tmp_path)
    proof = builder.analysis.worksharing_completion(builder.entry.qualified, joined, (loops[1],))
    node = builder.inline.prepare_worksharing(loops[1], proof)
    call = builder.inline.call(node)
    with pytest.raises(CompilationError, match="proven serial or original-team participation"):
        builder.inline.emit_call(call, {}, {}, [], query=False)


def test_worksharing_preparation_rejects_copied_proof_before_lowering(tmp_path, monkeypatch):
    from compiler.scopes import region_dispatch

    builder, loops, joined = example(tmp_path)
    proof = builder.analysis.worksharing_completion(builder.entry.qualified, joined, (loops[1],))
    attempts = []
    monkeypatch.setattr(region_dispatch, "lower_source", lambda *args, **kwargs: attempts.append(args))
    with pytest.raises(CompilationError, match="original-team authority"):
        builder.inline.prepare_worksharing(loops[1], replace(proof))
    assert not attempts
    assert not builder.inline.regions


def test_serial_and_worksharing_candidates_share_the_attempt_budget(tmp_path):
    builder, loops, joined = example(tmp_path)
    proof = builder.analysis.worksharing_completion(builder.entry.qualified, joined, (loops[1],))
    builder.inline.attempts = 32
    with pytest.raises(CompilationError, match="bounded inline region/operation budget exhausted"):
        builder.inline.prepare_worksharing(loops[1], proof)
    assert not builder.inline.regions


def test_native_worksharing_corrections_do_not_exhaust_later_generation(tmp_path):
    native = '!$omp do\ndo i=1,n\na(i)=a(i)+sum(b)\nenddo\n!$omp end do\n'
    position = SOURCE.rfind('!$omp do\n')
    source = SOURCE[:position] + native*34 + SOURCE[position:]
    builder, loops, joined = example(tmp_path, source)
    for loop in loops[2:-1]:
        proof = builder.analysis.worksharing_completion(builder.entry.qualified, joined, (loop,))
        with pytest.raises(CompilationError, match='array section requires bounded constant cardinality'):
            builder.inline.prepare_worksharing(loop, proof)
    assert builder.inline.attempts == 0
    proof = builder.analysis.worksharing_completion(builder.entry.qualified, joined, (loops[-1],))
    node = builder.inline.prepare_worksharing(loops[-1], proof)
    assert builder.inline.generated[node.fort_inline_region].scoped['team']['available']
    assert builder.inline.attempts == 1


def test_reached_dispatcher_identity_never_demands_flattened_owner_summary(tmp_path, monkeypatch):
    builder, loops, joined = example(tmp_path)
    builder.config = replace(builder.config, scope_execution='reached')
    proof = builder.analysis.worksharing_completion(builder.entry.qualified, joined, (loops[1],))
    node = builder.inline.prepare_worksharing(loops[1], proof)
    def summarize(*args, **kwargs):
        raise AssertionError('reached dispatcher demanded an eagerly flattened owner summary')
    monkeypatch.setattr(builder.analysis, 'summarize', summarize)
    handles = {root: 'handle' for root in builder.inline.arrays}
    values = {root: binding.name for root, binding in builder.inline.bindings.items()}
    builder.inline.emit_call(builder.inline.call(node), handles, values, [], query=False, collective=True)
    variant, = builder.variants.public()['procedures'][0]['variants']
    assert variant['summary_identity'] == builder.analysis.structure(builder.entry.qualified).identity


@pytest.mark.native
def test_native_dispatcher_routes_worker_kinds_and_preserves_distinct_bounds(tmp_path):
    """Compile the public Fortran interfaces against independent ABI probes."""
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    cc = shutil.which("cc")
    if not fortran or not cc:
        pytest.skip("native Fortran and C compilers required")
    builder, loops, joined = example(tmp_path)
    inline = builder.inline
    serial, = inline.prepare((loops[0],))
    team = inline.prepare_worksharing(loops[1], builder.analysis.worksharing_completion(
        builder.entry.qualified, joined, (loops[1],)))
    handles = {root: "unused" for root in inline.arrays}
    values = {root: binding.name for root, binding in inline.bindings.items()}
    for node in (serial, team):
        inline.emit_call(inline.call(node), handles, values, [], query=True)
    inline.finish()
    (tmp_path / "memory.f90").write_text("""module fort_scoped_memory
use iso_c_binding
integer(c_int),parameter::FORT_SCOPE_BOUNDARY=7
type,bind(C)::fort_scope_plan_decision
integer(c_int)::unused
end type
end module
""")
    sources = [tmp_path / "memory.f90"]
    probes = ["#include <stdint.h>"]
    common = "int64_t c,int64_t a,int64_t b,const int *lo_a,const int *lo_b,const int *n"
    check = "c==77 && a==101 && b==202 && *lo_a==-7 && *lo_b==13 && *n==19"
    for index, node in enumerate((serial, team)):
        generated = inline.generated[node.fort_inline_region]
        path = tmp_path / ("interface" + str(index) + ".f90")
        path.write_text(generated.artifacts["shared_interface.f90"])
        sources.append(path)
        public = generated.scoped
        probes.append(f"int {public['planning']['entry']}({common}) {{return {check} ? 51 : 99;}}")
        name = public["entry"] if index == 0 else public["team"]["entry"]
        signature = common.replace("int64_t c,", "int64_t c,int mode,")
        probes.append(f"int {name}({signature}) {{return mode==1 && {check} ? {31+10*index} : 99;}}")
    (tmp_path / "probes.c").write_text("\n".join(probes) + "\n")
    subprocess.run([cc, "-c", "probes.c", "-o", "probes.o"], cwd=tmp_path, check=True, capture_output=True)
    dispatcher = tmp_path / "dispatcher.f90"
    dispatcher.write_text(builder.outputs[inline.path])
    sources.append(dispatcher)
    arguments = "101_c_int64_t,202_c_int64_t,19_c_int,-7_c_int,13_c_int"
    driver = tmp_path / "driver.f90"
    driver.write_text(f"""program verify
use iso_c_binding
use {inline.name}
implicit none
if(query(77_c_int64_t,1_c_int,{arguments})/=51) error stop 1
if(query(77_c_int64_t,2_c_int,{arguments})/=51) error stop 2
if(run(77_c_int64_t,1_c_int,1_c_int,{arguments})/=31) error stop 3
if(run_team(77_c_int64_t,1_c_int,2_c_int,{arguments})/=41) error stop 4
if(run(77_c_int64_t,1_c_int,2_c_int,{arguments})/=7) error stop 5
if(run_team(77_c_int64_t,1_c_int,1_c_int,{arguments})/=7) error stop 6
end program
""")
    result = subprocess.run([fortran, "-std=f2018", "-fcheck=all", *map(str, sources), str(driver), "probes.o", "-o", "verify"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    subprocess.run([str(tmp_path / "verify")], cwd=tmp_path, check=True, capture_output=True)
