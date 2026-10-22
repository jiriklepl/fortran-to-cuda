"""Metadata-only entered-arm queries retain existing execution authorities."""

import shutil
import subprocess
from dataclasses import replace
from hashlib import sha256
from unittest.mock import patch

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes import query_regions
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_scoped_planning_entries import calibration

SOURCE = """module tilemeta
implicit none
integer,protected::tiles(2)=[2,3]
end module
module query_fixture
use tilemeta
implicit none
contains
subroutine first(a,b,n,scale)
real(8),intent(in)::a(:),scale
real(8),intent(inout)::b(:)
integer,intent(in)::n
integer::i,tnx
real(8)::coefficient
tnx=tiles(1)
coefficient=scale/2.0_8
do i=1,n
b(i)=a(i)*coefficient
enddo
end subroutine
subroutine second(b,c,n)
real(8),intent(in)::b(:)
real(8),intent(inout)::c(:)
integer,intent(in)::n
integer::i
do i=1,n
c(i)=b(i)+1.0_8
enddo
end subroutine
subroutine advance(a,b,c,n,flag)
real(8),intent(inout)::a(:),b(:),c(:)
integer,intent(in)::n
logical,intent(in)::flag
if(flag)then
call first(a,b,n,2.0_8)
call second(b,c,n)
else
call second(a,c,n)
endif
end subroutine
end module
"""

SINGLE_TEAM = """module tilemeta
implicit none
integer,protected::tiles(2)=[2,3]
end module
module query_fixture
implicit none
contains
subroutine advance(a,b,c,n,flag)
real(8),intent(inout)::a(:),b(:),c(:)
integer,intent(in)::n
logical,intent(in)::flag
integer::i
if(flag)then
!$omp parallel private(i) shared(a,b,c,n) default(none)
!$omp do
do i=1,n
b(i)=a(i)+1.0_8
enddo
!$omp end do
!$omp do
do i=1,n
c(i)=2.0_8*b(i)
enddo
!$omp end do
!$omp end parallel
endif
end subroutine
end module
"""


def emit(directory, source=SOURCE):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "query.f90"
    path.write_text(source)
    builder = ScopeBuilder([path], "query_fixture::advance",
        facts={"schema_version": 1, "participation": "serial",
               "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
               "captures": {**{"argument::" + name: dict(FACT) for name in ("a", "b", "c")},
                            "tilemeta::tiles": dict(FACT)}},
        options=CompilerOptions(opt_level=0), config=OffloadConfig("auto", calibration()))
    proofs = []
    seal = query_regions.seal_query_regions

    def retain(*args, **kwargs):
        result = seal(*args, **kwargs)
        proofs.extend(result)
        return result

    with patch.object(query_regions, "seal_query_regions", side_effect=retain):
        outputs, report = builder.run()
    text = outputs[report["sources"][str(path)]["replacement"]]
    return outputs, report, text, proofs


def test_maximal_query_is_inside_entered_arm_and_crosses_native_preparation(tmp_path):
    _outputs, report, text, proofs = emit(tmp_path)
    proof, = proofs
    assert proof.public()["workers"] == 2
    assert {step.member.routine.qualified for step in proof.steps} == {"query_fixture::first", "query_fixture::second"}
    assert any(step.call is None and step.recipes for step in proof.steps)
    assert text.lower().replace(" ", "").index("if(flag)then") < text.index("fort_query_owned_")
    assert "fort_scope_plan_abandon_remaining_v1" in text
    assert "future_source_read_admission" in str(report["scopes"][0]["query_regions"])
    assert "early_payload_transfers" in str(report["scopes"][0]["query_regions"])
    proof.validate()


def test_one_original_source_team_records_its_two_registered_physical_workers(tmp_path):
    _outputs, _report, text, proofs = emit(tmp_path, SINGLE_TEAM)
    proof, = proofs
    assert len(proof.steps) == 1
    assert proof.public()['source_calls'] == 1
    assert proof.public()['source_steps'] == 1
    assert proof.public()['workers'] == 2
    assert text.index('fort_query_owned_') < text.lower().index('!$omp parallel')
    assert text.lower().replace(' ', '').index('if(flag)then') < text.index('fort_query_owned_')
    assert not proof.public()['future_source_read_admission']
    proof.validate()


def test_one_physical_worker_still_uses_its_original_local_query(tmp_path):
    single = SINGLE_TEAM.replace('!$omp do\ndo i=1,n\nc(i)=2.0_8*b(i)\nenddo\n!$omp end do\n', '')
    _outputs, _report, text, proofs = emit(tmp_path, single)
    assert not proofs
    assert 'fort_query_owned_' not in text


@pytest.mark.parametrize('mutation', ['worker_count', 'registration'])
def test_single_source_query_rejects_changed_registered_physical_authority(tmp_path, mutation):
    _outputs, _report, _text, proofs = emit(tmp_path, SINGLE_TEAM)
    proof, = proofs
    step, = proof.steps
    inline = step.member.builder.inline_for(step.call.procedure)
    prepared = inline.prepared_regions[step.call.procedure]
    if mutation == 'registration':
        inline.prepared_regions.pop(step.call.procedure)
    else:
        object.__setattr__(prepared, 'offload', replace(prepared.offload, units=prepared.offload.units[:1]))
    with pytest.raises(CompilationError):
        proof.validate()


def test_tiling_recipe_uses_host_current_probe_before_exact_payload_snapshot(tmp_path):
    source = SOURCE.replace("do i=1,n\nb(i)", "do i=1,tnx\nb(i)", 1)
    _outputs, _report, text, proofs = emit(tmp_path, source)
    proof, = proofs
    assert proof.table_reads == ("tilemeta::tiles",)
    probe = text.index("fort_scope_plan_host_current(")
    snapshot = text.index("= int(tiles(", probe)
    assert probe < snapshot
    assert "/= FORT_SCOPE_OK) exit" in text[probe:snapshot]
    assert "coefficient = scale / 2.0" in text or "coefficient=scale/2.0" in text


@pytest.mark.parametrize("mutation", ["mapping", "child_plan", "unit_order", "config", "copy", "call_path"])
def test_query_rejects_mutated_authority(tmp_path, mutation):
    _outputs, _report, _text, proofs = emit(tmp_path)
    proof, = proofs
    if mutation == "mapping":
        step = next(step for step in proof.steps if step.call is not None)
        object.__setattr__(step, "mappings", tuple(reversed(step.mappings)))
    elif mutation == "child_plan":
        object.__setattr__(proof.steps[0].member, "execution_plan", object())
    elif mutation == "unit_order":
        member = proof.steps[0].member
        object.__setattr__(member, "units", list(reversed(member.units)))
    elif mutation == "config":
        proof.owner.builder.config = replace(proof.owner.builder.config, policy="sections")
    elif mutation == "copy":
        proof = replace(proof)
    else:
        step = next(step for step in proof.steps if step.call_path)
        part = step.call_path[0]
        dict.__setitem__(part.site, "first_line", part.site["first_line"] + 1)
    with pytest.raises(CompilationError):
        proof.validate()


@pytest.mark.parametrize("team", [False, True, "recipe", "single"])
def test_query_entry_and_original_team_default_none_compile(tmp_path, team):
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran unavailable")
    source = SINGLE_TEAM if team == 'single' else SOURCE
    if team == "recipe":
        source = source.replace("do i=1,n\nb(i)", "do i=1,tnx\nb(i)", 1)
    elif team is True:
        source = source.replace("do i=1,n\nb(i)",
            "!$omp parallel do private(i) shared(a,b,n,coefficient) default(none)\ndo i=1,n\nb(i)")
        source = source.replace("b(i)=a(i)*coefficient\nenddo", "b(i)=a(i)*coefficient\nenddo\n!$omp end parallel do")
    outputs, report, _text, proofs = emit(tmp_path, source)
    assert proofs
    for name, text in outputs.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in report["build_sources"]:
            if item["role"] != role or item["language"] != "fortran":
                continue
            result = subprocess.run([fortran, "-O2", "-std=f2018", "-fopenmp", "-c",
                str(tmp_path / item["path"]), "-o", str(tmp_path / (item["path"].replace("/", "_") + ".o"))],
                cwd=tmp_path, capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, result.stdout + result.stderr
