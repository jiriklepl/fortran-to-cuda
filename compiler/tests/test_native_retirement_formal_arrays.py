"""Joined fixed-formal retirement never grants unknown mask payload authority."""
import json
from hashlib import sha256

import pytest
from fparser.two import Fortran2003
from fparser.two.utils import walk

from compiler.driver.options import CompilerOptions
from compiler.frontend.native_completion import native_call_storage_structures, prove_native_call_retirement
from compiler.frontend.resource_reachability import prove_native_effect_graph
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder

SOURCE = """module parameters
implicit none
integer,allocatable::Utype(:,:,:)
end module

module forcing
use parameters
implicit none
real(8),allocatable::u(:,:,:),v(:,:,:),w(:,:,:),output(:,:,:)
real(8)::mean(3)=0
contains
subroutine advance(n,go,masked)
integer,intent(in)::n
logical,intent(in)::go,masked
call middle(n,go,masked)
end subroutine
subroutine middle(n,go,masked)
integer,intent(in)::n
logical,intent(in)::go,masked
call producer(u,v,w,n)
if(go)call mean_uvw(mean,u,v,w,n,masked)
call consumer(output,u,v,w,n)
end subroutine

subroutine producer(a,b,c,n)
integer,intent(in)::n
real(8),intent(inout)::a(-2:,-2:,-2:),b(-2:,-2:,-2:),c(-2:,-2:,-2:)
integer::i,j,k
do k=1,n
do j=1,n
do i=1,n
a(i,j,k)=real(i,8)
b(i,j,k)=real(j,8)
c(i,j,k)=real(k,8)
enddo
enddo
enddo
end subroutine

subroutine consumer(a,b,c,d,n)
integer,intent(in)::n
real(8),intent(inout)::a(-2:,-2:,-2:)
real(8),intent(in)::b(-2:,-2:,-2:),c(-2:,-2:,-2:),d(-2:,-2:,-2:)
integer::i,j,k
do k=1,n
do j=1,n
do i=1,n
a(i,j,k)=b(i,j,k)+c(i,j,k)+d(i,j,k)
enddo
enddo
enddo
end subroutine

subroutine mean_uvw(result,a,b,c,n,masked)
integer,intent(in)::n
logical,intent(in)::masked
real(8),intent(out)::result(3)
real(8),contiguous,intent(in)::a(-2:,-2:,-2:),b(-2:,-2:,-2:),c(-2:,-2:,-2:)
integer::i,j,k
result=0
if(masked)then
!$omp parallel reduction(+:result)
!$omp do collapse(3)
do k=1,n
do j=1,n
do i=1,n
if(Utype(i,j,k)<=0)result(1)=result(1)+a(i,j,k)
enddo
enddo
enddo
!$omp do collapse(3)
do k=1,n
do j=1,n
do i=1,n
if(Utype(i,j,k)<=0)result(2)=result(2)+b(i,j,k)
enddo
enddo
enddo
!$omp do collapse(3)
do k=1,n
do j=1,n
do i=1,n
if(Utype(i,j,k)<=0)result(3)=result(3)+c(i,j,k)
enddo
enddo
enddo
!$omp end parallel
else
!$omp parallel reduction(+:result)
!$omp do collapse(3)
do k=1,n
do j=1,n
do i=1,n
result(1)=result(1)+a(i,j,k)
enddo
enddo
enddo
!$omp do collapse(3)
do k=1,n
do j=1,n
do i=1,n
result(2)=result(2)+b(i,j,k)
enddo
enddo
enddo
!$omp do collapse(3)
do k=1,n
do j=1,n
do i=1,n
result(3)=result(3)+c(i,j,k)
enddo
enddo
enddo
!$omp end parallel
endif
result=result/max(1,n)**3
end subroutine
end module
"""


def fixture(directory, source=SOURCE):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "mean.f90"
    path.write_text(source)
    fact = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"forcing::" + name: dict(fact) for name in ("u", "v", "w", "output")}}
    builder = ScopeBuilder([path], "forcing::advance", facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="graph"),
        contracts={"forcing::advance": {"native_ancestor": {"version": 1,
            "device_roots": ["forcing::producer", "forcing::consumer"]}}})
    return path, builder


def retirement(directory):
    path, builder = fixture(directory)
    analysis = builder.analysis
    routine = analysis.routines["forcing::middle"]
    selected = tuple(node for node in walk(routine.execution)
                     if type(node).__name__ == "If_Stmt" and "mean_uvw" in str(node).lower())
    assert len(selected) == 1
    native = prove_native_effect_graph(analysis, routine.qualified, selected)
    teams = native_call_storage_structures(analysis, native)
    storage = prove_storage_mutation_effects(analysis, routine.qualified, selection=selected,
                                            joined_storage=teams, native_graph=native)
    proof = prove_native_call_retirement(analysis, routine.qualified, selected, native, storage)
    return path, builder, selected, native, teams, storage, proof


def test_whole_guarded_call_uses_exact_formal_teams_only_for_retirement(tmp_path):
    _path, builder, selected, _native, teams, storage, proof = retirement(tmp_path)
    assert len(teams) == 2
    assert all(len(team._fixed_array_associations) == 1 for team in teams)
    assert storage._record["complete"] is True
    assert not storage._record["frontiers"]
    assert not storage._record["escapes"]
    assert proof._call_storage is storage
    assert proof._reduction_teams == ()
    public = proof.public()
    assert public["schema_version"] == 3
    assert public["requires_domain_retirement"] is True
    assert public["completion_scope"] == "normal return only"
    assert all(public[key] is False for key in ("payload_access_authorized", "stable_storage_authorized",
        "gpu_legality_established", "motion_authorized", "internal_cuts_authorized"))
    assert proof.validate(builder.analysis, "forcing::middle", proof.selected_node_ids) is proof
    assert "parameters::utype" not in builder.facts["captures"]
    assert str(selected[0]).startswith("IF (go)")


def test_missing_mask_fact_preserves_guarded_retirement_and_later_gpu_work(tmp_path):
    _path, builder = fixture(tmp_path)
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    scope, = report["scopes"]
    boundary, = report["boundaries"]
    assert boundary["conditional_close"] == "go"
    assert boundary["lifecycle"] == "proved_lifecycle"
    assert boundary["reopen"] is True
    plan = scope["checked_execution_plan"]
    returned = plan["storage_epoch"]["return_contract"]
    middle, = [row for row in returned["members"] if row["procedure"] == "forcing::middle"]
    assert any(row["kind"] == "reached_retirement" and row["rearm_allowed"] for row in middle["routes"])
    assert any(row["kind"] == "checked_child" and row["procedure"] == "forcing::consumer" for row in middle["routes"])
    assert all(row["resource"] != "parameters::utype" for row in plan["storage_epoch"]["resources"])
    binding = builder.visible_binding(builder.entry, "parameters::utype")
    with pytest.raises(CompilationError, match="missing stable storage/definition facts"):
        builder.capture_fact(binding)
    changed = "\n".join(text for name, text in outputs.items() if name.startswith("sources/"))
    assert "if (go) then" in changed.lower()
    assert "fort_child_retention_" in changed
    assert "parameters::utype" not in builder.facts["captures"]
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    validate_checked_execution_plans(json.loads(outputs["scope-manifest.json"]))


@pytest.mark.parametrize("change", ["missing", "foreign", "bounds", "arena", "guard"])
def test_formal_retirement_revalidates_original_call_receipts_at_publication(tmp_path, change):
    _path, builder, selected, _native, teams, _storage, proof = retirement(tmp_path / "subject")
    association = teams[0]._fixed_array_associations[0]
    if change == "missing":
        object.__setattr__(teams[0], "_fixed_array_associations", ())
    elif change == "foreign":
        _p, _b, _s, _n, foreign, _st, _pr = retirement(tmp_path / "foreign")
        object.__setattr__(teams[0], "_fixed_array_associations", foreign[0]._fixed_array_associations)
    elif change == "bounds":
        object.__setattr__(association, "actual_bounds", (-1, 1))
    elif change == "arena":
        object.__setattr__(association, "_arena", object())
    else:
        object.__setattr__(selected[0], "items", (Fortran2003.Name("another_guard"), selected[0].items[1]))
    with pytest.raises(CompilationError):
        proof.validate(builder.analysis, "forcing::middle", proof.selected_node_ids)


@pytest.mark.parametrize("effect", ["unknown_escape", "task"])
def test_joined_formal_reduction_does_not_waive_other_lifetime_frontiers(tmp_path, effect):
    if effect == "unknown_escape":
        source = SOURCE.replace("result=result/max(1,n)**3", "result=result/max(1,n)**3\ncall unknown_escape(result)")
    else:
        source = SOURCE.replace("result=result/max(1,n)**3",
            "result=result/max(1,n)**3\n!$omp task\nresult(1)=result(1)+1\n!$omp end task")
    _path, builder = fixture(tmp_path, source)
    _outputs, report = builder.run()
    assert report["scope_count"] == 0
    assert "cannot reenter after unproved lifetime/escape" in report["graph_owner_attempt"]["error"]
