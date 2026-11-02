"""Scalar formal reductions can retire one original CALL, never grant payload."""
import json
from copy import copy

import pytest
from fparser.two import Fortran2003
from fparser.two.utils import walk

from compiler.frontend.descriptor_teams import prove_descriptor_team
from compiler.frontend.native_completion import (
    _joined_completion_facts,
    native_call_storage_structures,
    prove_graph_joined_completion,
    prove_native_call_retirement,
)
from compiler.frontend.resource_reachability import prove_native_effect_graph
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.ir import CompilationError
from compiler.tests.test_native_retirement_formal_arrays import fixture

SOURCE = """module parameters
implicit none
integer,allocatable::Prtype(:,:,:)
integer::Prnx=2,Prny=2,Prnz=2,n_full_im_U=0,n_full_im_V=0,n_full_im_W=0
real(8)::n_free_domain=8
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
real(8)::tke
tke=1
call producer(u,v,w,n)
if(go)call mean_tke(tke,u,v,w,mean)
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

  subroutine mean_TKE(tke, U, V, W, Umean)
    real(8), intent(out) :: tke
    real(8), dimension(-2:,-2:,-2:), contiguous, intent(in) :: U,V,W
    real(8), intent(in) :: Umean(3)
    integer :: i, j, k

    tke = 0

    if (n_full_im_U==0.and. &
        n_full_im_V==0.and. &
        n_full_im_W==0) then

      !$omp parallel do private(i,j,k) reduction(+:tke) collapse(3)
      do k = 1, Prnz
        do j = 1, Prny
          do i = 1, Prnx
            tke = tke + (U(i,j,k) + U(i-1,j,k) - 2*Umean(1))**2 &
                      + (V(i,j,k) + V(i,j-1,k) - 2*Umean(2))**2 &
                      + (W(i,j,k) + W(i,j,k-1) - 2*Umean(3))**2
          end do
        end do
      end do
      !$omp end parallel do

      tke = tke / 8
    else
      !$omp parallel do private(i,j,k) reduction(+:tke) collapse(3)
      do k = 1, Prnz
        do j = 1, Prny
          do i = 1, Prnx
            if (Prtype(i,j,k)<=0) &
              tke = tke + (U(i,j,k) + U(i-1,j,k) - 2*Umean(1))**2 &
                        + (V(i,j,k) + V(i,j-1,k) - 2*Umean(2))**2 &
                        + (W(i,j,k) + W(i,j,k-1) - 2*Umean(3))**2
          end do
        end do
      end do
      !$omp end parallel do

      tke = tke / 8
    end if



    tke = tke / n_free_domain
  end subroutine

end module
"""


def retirement(directory, source=SOURCE):
    _path, builder = fixture(directory, source)
    analysis = builder.analysis
    routine = analysis.routines["forcing::middle"]
    selected = tuple(node for node in walk(routine.execution)
                     if type(node).__name__ == "If_Stmt" and "mean_tke" in str(node).lower())
    if not selected:
        selected = tuple(node for node in walk(routine.execution)
                         if type(node).__name__ == "Call_Stmt" and "mean_tke" in str(node).lower())
    assert len(selected) == 1
    native = prove_native_effect_graph(analysis, routine.qualified, selected)
    teams = native_call_storage_structures(analysis, native)
    storage = prove_storage_mutation_effects(analysis, routine.qualified, selection=selected,
                                            joined_storage=teams, native_graph=native)
    proof = prove_native_call_retirement(analysis, routine.qualified, selected, native, storage)
    return builder, selected, native, teams, storage, proof


@pytest.mark.parametrize("intent", ["out", "inout"])
def test_original_scalar_formal_only_authorizes_whole_call_retirement(tmp_path, intent):
    source = SOURCE.replace("intent(out) :: tke", "intent(" + intent + ") :: tke")
    builder, selected, native, teams, storage, proof = retirement(tmp_path, source)
    assert len(teams) == 2
    assert all(team._native_graph is native for team in teams)
    assert all(team.public()["native_lifetime_scalar_reductions"]["tke"] for team in teams)
    assert storage._record["complete"]
    assert not storage._record["escapes"]
    assert not storage._record["frontiers"]
    assert proof._call_storage is storage
    assert proof._reduction_teams == ()
    public = proof.public()
    assert public["schema_version"] == 3
    assert public["requires_domain_retirement"]
    assert not any(public[key] for key in ("payload_access_authorized", "stable_storage_authorized",
        "gpu_legality_established", "motion_authorized", "internal_cuts_authorized"))
    assert proof.validate(builder.analysis, "forcing::middle", proof.selected_node_ids) is proof
    assert str(selected[0]).startswith("IF (go)")


@pytest.mark.parametrize("guard", ["go", ".false.", ".true."])
def test_mean_tke_owner_preserves_original_guard_and_rearms_checked_consumer(tmp_path, guard):
    source = SOURCE.replace("if(go)call mean_tke", "if(" + guard + ")call mean_tke")
    _path, builder = fixture(tmp_path, source)
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    boundary, = report["boundaries"]
    assert boundary["conditional_close"].lower() == guard
    assert boundary["lifecycle"] == "proved_lifecycle"
    assert boundary["reopen"]
    scope, = report["scopes"]
    plan = scope["checked_execution_plan"]
    middle, = [row for row in plan["storage_epoch"]["return_contract"]["members"]
               if row["procedure"] == "forcing::middle"]
    assert any(row["kind"] == "reached_retirement" and row["rearm_allowed"] for row in middle["routes"])
    assert any(row["kind"] == "checked_child" and row["procedure"] == "forcing::consumer" for row in middle["routes"])
    assert all(row["resource"] != "parameters::prtype" for row in plan["storage_epoch"]["resources"])
    binding = builder.visible_binding(builder.entry, "parameters::prtype")
    with pytest.raises(CompilationError, match="missing stable storage/definition facts"):
        builder.capture_fact(binding)
    assert "parameters::prtype" not in builder.facts["captures"]
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    manifest = json.loads(outputs["scope-manifest.json"])
    validate_checked_execution_plans(manifest)
    from elmm_pipeline.local_adapter import _scope_package_build

    package = tmp_path / "package"
    for name, text in outputs.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _scope_package_build({"artifact_root": str(package), "scope_manifest": manifest})


def test_same_issued_scalar_team_cannot_authorize_strong_graph_or_payload_completion(tmp_path):
    builder, _selected, native, teams, _storage, _proof = retirement(tmp_path)
    team = teams[0]
    with pytest.raises(CompilationError, match="whole-CALL lifetime association"):
        prove_descriptor_team(builder.analysis, team.procedure, team._originals)
    with pytest.raises(CompilationError, match="whole-CALL lifetime association"):
        prove_graph_joined_completion(builder.analysis, team.procedure, team._originals, native)
    with pytest.raises(CompilationError, match="whole-call retirement only"):
        _joined_completion_facts(builder.analysis, team.procedure, team._originals, reduction_team=team)


@pytest.mark.parametrize("change", ["missing_graph", "foreign_graph", "missing_actual_type", "foreign_actual",
                                     "actual_kind", "actual_unknown_attribute", "formal_intent", "guard"])
def test_scalar_retirement_revalidates_exact_original_receipts_at_publication(tmp_path, change):
    builder, selected, native, teams, _storage, proof = retirement(tmp_path / "subject")
    call = next(instance.call.edge for instance in native.instances if instance.call is not None)
    mapping = next(item for item in call.resolved.associations if item.formal == "tke")
    if change == "missing_graph":
        object.__setattr__(teams[0], "_native_graph", None)
    elif change == "foreign_graph":
        _b, _s, foreign, _t, _st, _p = retirement(tmp_path / "foreign")
        object.__setattr__(teams[0], "_native_graph", foreign)
    elif change == "missing_actual_type":
        object.__setattr__(mapping, "actual_type", None)
    elif change == "foreign_actual":
        _b, _s, foreign, _t, _st, _p = retirement(tmp_path / "foreign")
        foreign_call = next(instance.call.edge for instance in foreign.instances if instance.call is not None)
        other = next(item for item in foreign_call.resolved.associations if item.formal == "tke")
        object.__setattr__(mapping, "actual_type", other.actual_type)
    elif change == "actual_kind":
        object.__setattr__(mapping.binding, "kind", 4)
    elif change == "actual_unknown_attribute":
        object.__setattr__(mapping.binding, "attributes", frozenset({"unproved_external_storage"}))
    elif change == "formal_intent":
        object.__setattr__(mapping.formal_binding, "intent", "in")
    else:
        object.__setattr__(selected[0], "items", (Fortran2003.Name("foreign_guard"), selected[0].items[1]))
    with pytest.raises(CompilationError):
        proof.validate(builder.analysis, "forcing::middle", proof.selected_node_ids)


@pytest.mark.parametrize("change", ["target", "save", "initializer", "data", "threadprivate", "cray", "namelist",
                                    "element", "duplicate", "unknown_escape", "task"])
def test_unproved_scalar_storage_and_async_effects_remain_native(tmp_path, change):
    if change in {"target", "save"}:
        source = SOURCE.replace("real(8)::tke", "real(8)," + change + "::tke")
    elif change == "initializer":
        source = SOURCE.replace("real(8)::tke", "real(8)::tke=1")
    elif change == "data":
        source = SOURCE.replace("real(8)::tke", "real(8)::tke\ndata tke /1.0d0/")
    elif change == "threadprivate":
        source = SOURCE.replace("real(8)::tke", "real(8),save::tke\n!$omp threadprivate(tke)")
    elif change == "cray":
        source = SOURCE.replace("real(8)::tke", "real(8)::tke\ninteger(8)::address\npointer(address,tke)")
    elif change == "namelist":
        source = SOURCE.replace("real(8)::tke", "real(8)::tke\nnamelist /hidden/ tke")
    elif change == "element":
        source = SOURCE.replace("real(8)::tke", "real(8)::tke(2)").replace("mean_tke(tke,u", "mean_tke(tke(1),u")
    elif change == "duplicate":
        source = SOURCE.replace("mean_tke(tke,u,v,w,mean)", "mean_tke(tke,u,v,w,mean,tke)")
        source = source.replace("subroutine mean_TKE(tke, U, V, W, Umean)",
                                "subroutine mean_TKE(tke, U, V, W, Umean, other)")
        source = source.replace("    real(8), intent(out) :: tke", "    real(8), intent(out) :: tke\n    real(8), intent(in) :: other")
    else:
        effect = "call unknown_escape(tke)" if change == "unknown_escape" else "!$omp task\ntke=tke+1\n!$omp end task"
        source = SOURCE.replace("    tke = tke / n_free_domain", "    tke = tke / n_free_domain\n" + effect)
    _path, builder = fixture(tmp_path, source)
    _outputs, report = builder.run()
    assert report["scope_count"] == 0, report
    assert "cannot reenter after unproved lifetime/escape" in report["graph_owner_attempt"]["error"]


def test_foreign_graph_scalar_team_cannot_be_forwarded_into_original_storage(tmp_path, monkeypatch):
    _foreign, _selection, _graph, teams, _storage, _proof = retirement(tmp_path / "foreign")
    import compiler.frontend.native_completion as completion

    monkeypatch.setattr(completion, "native_call_storage_structures", lambda _analysis, _native: teams)
    _path, builder = fixture(tmp_path / "subject", SOURCE)
    _outputs, report = builder.run()
    assert report["scope_count"] == 0
    assert "cannot reenter after unproved lifetime/escape" in report["graph_owner_attempt"]["error"]


@pytest.mark.parametrize("attachment", ["template", "call", "edge", "resolved"])
def test_scalar_team_final_fence_rejects_callback_receipt_replacement(tmp_path, monkeypatch, attachment):
    from compiler.frontend.native_lifetime_context import NativeLifetimeContext

    builder, _selected, native, teams, _storage, _proof = retirement(tmp_path)
    template = next(item for item in native.templates if item.calls)
    call = template.calls[0]
    original = NativeLifetimeContext.scalar_reduction_associations
    mutated = []

    def replaced(context, *args):
        rows = original(context, *args)
        if context.graph is native and not mutated:
            if attachment == "template":
                object.__setattr__(template, "calls", tuple(list(template.calls)))
            elif attachment == "call":
                object.__setattr__(call, "edge", copy(call.edge))
            elif attachment == "edge":
                object.__setattr__(call.edge, "original", copy(call.edge.original))
            else:
                object.__setattr__(call.edge.resolved, "actuals", tuple(list(call.edge.resolved.actuals)))
            mutated.append(True)
        return rows

    monkeypatch.setattr(NativeLifetimeContext, "scalar_reduction_associations", replaced)
    with pytest.raises(CompilationError, match="attachments changed|original|source|graph"):
        teams[0].validate(builder.analysis, teams[0].procedure, teams[0].selected_node_ids)
    assert mutated


@pytest.mark.parametrize("intent", ["out", "inout"])
def test_actual_mean_tke_plain_call_also_retires_and_rearms(tmp_path, intent):
    source = SOURCE.replace("if(go)call mean_tke", "call mean_tke")
    source = source.replace("intent(out) :: tke", "intent(" + intent + ") :: tke")
    builder, selected, _native, _teams, _storage, proof = retirement(tmp_path / "proof", source)
    assert type(selected[0]).__name__ == "Call_Stmt"
    assert proof.validate(builder.analysis, "forcing::middle", proof.selected_node_ids) is proof
    _path, builder = fixture(tmp_path / "owner", source)
    _outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    boundary, = report["boundaries"]
    assert boundary["lifecycle"] == "proved_lifecycle"
    assert boundary["reopen"]
    assert boundary["conditional_close"] is None
