"""Complete original numerical groups use one checked reached owner path."""

from copy import copy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.closed_selection import ClosedNumericalOwner, candidates, discover_closed_selections
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

SOURCE = """module selected_groups
implicit none
integer:: tiles(3)=[4,3,2], nx=15, ny=7, nz=3, order=2
contains
subroutine advance(a,b)
real(8),intent(in)::a(0:,0:,0:)
real(8),intent(out)::b(0:,0:,0:)
integer:: tx,ty,tz,i,j,k,bi,bj,bk
real(8):: coefficient
tx=tiles(1)
ty=tiles(2)
tz=tiles(3)
coefficient=2.0_8
if(order==4) then
!$omp parallel do private(i,j,k,bi,bj,bk) collapse(3) schedule(runtime)
do bk=0,nz,tz
do bj=0,ny,ty
do bi=0,nx,tx
do k=bk,min(bk+tz-1,nz)
do j=bj,min(bj+ty-1,ny)
do i=bi,min(bi+tx-1,nx)
b(i,j,k)=coefficient*a(i,j,k)+4.0_8
enddo
enddo
enddo
enddo
enddo
enddo
!$omp end parallel do
else
!$omp parallel do private(i,j,k,bi,bj,bk) collapse(3) schedule(runtime)
do bk=0,nz,tz
do bj=0,ny,ty
do bi=0,nx,tx
do k=bk,min(bk+tz-1,nz)
do j=bj,min(bj+ty-1,ny)
do i=bi,min(bi+tx-1,nx)
b(i,j,k)=coefficient*a(i,j,k)
enddo
enddo
enddo
enddo
enddo
enddo
!$omp end parallel do
endif
end subroutine
end module
"""


def builder(path, source=SOURCE, *, config=None, options=None, captures=None):
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::a": FACT, "argument::b": {**FACT, "initialized": "none"},
                          "selected_groups::tiles": FACT}}
    facts["captures"].update(captures or {})
    return ScopeBuilder([path], "selected_groups::advance", facts=facts,
                        options=options or CompilerOptions(gpu_policy="sections"),
                        config=config or OffloadConfig(policy="sections", scope_execution="bounded"))


def test_closed_original_groups_keep_prefix_branches_and_one_declaration_edit(tmp_path, monkeypatch):
    selected = builder(tmp_path / "original.f90")

    def no_synthetic_owner(*_args, **_kwargs):
        raise AssertionError("original complete group retried through old synthetic owner")

    monkeypatch.setattr(ScopeBuilder, "owner", no_synthetic_owner)
    outputs, report = selected.run()
    assert report["scope_count"] == 2
    assert not report["boundaries"]
    text = outputs[report["sources"][str(tmp_path / "original.f90")]["replacement"]]
    assert text.count("INTENT(IN), TARGET :: a") == 1
    assert text.count("INTENT(OUT), TARGET :: b") == 1
    assert "tx=tiles(1)\nty=tiles(2)\ntz=tiles(3)\ncoefficient=2.0_8\nif(order==4) then\nblock" in text
    assert "end block\nelse\nblock" in text
    assert text.count("b(i,j,k)=coefficient*a(i,j,k)+4.0_8") == 1
    assert text.count("b(i,j,k)=coefficient*a(i,j,k)\n") == 1
    assert text.count("!$omp parallel private(i,j,k,bi,bj,bk)") == 2
    assert text.count("error stop 'combined original team numerical execution failed'") == 2
    assert text.count("error stop 'combined original team completion failed'") == 2
    for scope in report["scopes"]:
        assert scope["checked_execution_plan"]["policy"]["lifetime"] == "closed original numerical selection"
        assert scope["checked_execution_plan"]["numerical_invocations"] == 1
        assert scope["closed_selection"]["first_line"] == scope["first_line"]
        assert scope["resource_bindings"]["resources"][1]["resource"] == "argument::b"
        assert not scope["estimate_available"]


def test_exact_original_selection_prefix_authority_is_retained(tmp_path):
    selected = builder(tmp_path / "original.f90")
    group, prefix, controls = next(candidates(selected))
    owner = ClosedNumericalOwner(selected, group, prefix, controls)
    assert owner.run(group)
    assert {item.name for item in owner.closed_initializations} == {"tx", "ty", "tz", "coefficient"}
    proof = owner.emission.selection_prefix
    assert proof is not None
    owner.execution_plan.validate()
    owner.emission.selection_prefix = copy(proof)
    with pytest.raises(CompilationError, match="prefix authority"):
        owner.execution_plan.validate()
    owner.emission.selection_prefix = proof
    owner.execution_plan.validate()
    with pytest.raises(CompilationError):
        owner.closed_prefix = ()


@pytest.mark.parametrize("prefix", ["b=0.0_8", "nx=nx+1", "call opaque()", "if(order==2) tx=1"])
def test_unsupported_prefix_declines_without_retry_or_edits(tmp_path, prefix):
    selected = builder(tmp_path / "original.f90", SOURCE.replace("tx=tiles(1)", prefix + "\ntx=tiles(1)"))
    assert discover_closed_selections(selected)
    outputs, report = selected.finish()
    assert report["scope_count"] == 0
    assert not report["source_edits"]
    assert not report["build_sources"]
    assert not outputs.keys() - {"scope-manifest.json"}
    assert len(report["boundaries"]) == 2
    assert all(row["kind"] == "closed_numerical_selection_decline" for row in report["boundaries"])


def test_closed_groups_do_not_hide_unrelated_numerical_candidates(tmp_path):
    source = SOURCE.replace("subroutine advance(a,b)", "subroutine advance(a,b,c,d,n)\n"
                            "real(8),intent(in)::c(:)\nreal(8),intent(out)::d(:)\ninteger,intent(in)::n")
    source = source.replace("endif\nend subroutine", "endif\ncall tail(c,d,n)\ncall tail(c,d,n)\nend subroutine")
    source = source.replace("end module", """subroutine tail(c,d,n)
real(8),intent(in)::c(:)
real(8),intent(out)::d(:)
integer,intent(in)::n
integer::i
do i=1,n
d(i)=3*c(i)
enddo
end subroutine
end module""")
    selected = builder(tmp_path / "original.f90", source,
                       captures={"argument::c": FACT, "argument::d": {**FACT, "initialized": "none"}})
    outputs, report = selected.run()
    closed = [scope for scope in report["scopes"] if "closed_selection" in scope]
    ordinary = [scope for scope in report["scopes"] if "closed_selection" not in scope]
    assert len(closed) == 2
    assert len(ordinary) == 1
    assert ordinary[0]["first_line"] > max(scope["last_line"] for scope in closed)
    text = outputs[report["sources"][str(tmp_path / "original.f90")]["replacement"]]
    assert "compiler-owned closed selection scan boundary" not in text
    assert text.count("b(i,j,k)=coefficient*a(i,j,k)\n") == 1
    assert ordinary[0]["calls"] == ["selected_groups::tail", "selected_groups::tail"]


def test_closed_group_selection_requires_exact_original_nodes(tmp_path):
    selected = builder(tmp_path / "original.f90")
    group, prefix, controls = next(candidates(selected))
    owner = ClosedNumericalOwner(selected, group, prefix, controls)
    altered = (copy(group[0]), *group[1:])
    with pytest.raises(CompilationError, match="selection changed"):
        owner.scan(altered)
    assert not selected.edits


def test_unavailable_automatic_team_pricing_discards_all_numerical_workers(tmp_path):
    from compiler.tests.test_source_compute_costs import profile_v2

    selected = builder(tmp_path / "original.f90", options=CompilerOptions(gpu_policy="auto"),
                       config=OffloadConfig(policy="auto", scope_execution="bounded", profile=profile_v2()))
    outputs, report = selected.run()
    assert len(report["scopes"]) == 2
    assert all(scope["automatic_preflight"]["selection"] == "native" for scope in report["scopes"])
    assert not report["automatic_estimate_available"]
    assert outputs[report["sources"][str(tmp_path / "original.f90")]["replacement"]] == SOURCE
    assert all(row["role"] in {"original_source", "common_runtime"} for row in report["build_sources"])
    assert not any(name.startswith(("regions/", "leaves/", "exact/")) for name in outputs)


def test_fresh_closed_plans_share_portable_identity_and_reject_publication_change(tmp_path):
    path = tmp_path / "original.f90"
    results = [builder(path).run()[1] for _ in range(2)]
    assert [scope["checked_execution_plan"]["execution_identity"] for scope in results[0]["scopes"]] == [
        scope["checked_execution_plan"]["execution_identity"] for scope in results[1]["scopes"]]
    selected = builder(path)
    group, prefix, controls = next(candidates(selected))
    owner = ClosedNumericalOwner(selected, group, prefix, controls)
    assert owner.run(group)
    path.write_text(SOURCE + "! changed after admission\n")
    with pytest.raises(CompilationError, match="changed"):
        owner.execution_plan.validate()
    with pytest.raises(CompilationError, match="changed"):
        selected.finish()
