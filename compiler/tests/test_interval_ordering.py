"""Search priority exposes useful reuse without becoming placement authority."""

from hashlib import sha256

import pytest

from compiler._legacy.config import LegacyCompilerOptions as CompilerOptions
from compiler._legacy.config import LegacyOffloadConfig as OffloadConfig
from compiler.scopes.closed_selection import interval_candidates
from compiler.scopes.interval_ordering import order_intervals
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_selection_cost import build_validation, process_fixture  # noqa: F401
from compiler.tests.test_source_scopes import FACT

SOURCE = """module stream_fields
implicit none
real(8)::old(0:15),middle(0:15),star(0:15),out(0:15),unrelated(0:15)
contains
subroutine advance(n)
integer,intent(in)::n
integer::i
call tune(unrelated,n)
call tune(unrelated,n)
call tune(unrelated,n)
call make_field(old,middle,n)
middle(0)=middle(0)+old(0)
call diffuse(middle,star,n)
!$omp parallel do private(i) schedule(static)
do i=0,n-1
out(i)=out(i)+star(i)
enddo
!$omp end parallel do
end subroutine
subroutine tune(a,n)
integer,intent(in)::n
real(8),intent(inout)::a(0:)
integer::i
do i=0,n-1
a(i)=a(i)+1.d0
enddo
end subroutine
subroutine make_field(a,b,n)
integer,intent(in)::n
real(8),intent(in)::a(0:)
real(8),intent(inout)::b(0:)
integer::i
do i=0,n-1
b(i)=2.d0*a(i)
enddo
end subroutine
subroutine diffuse(b,c,n)
integer,intent(in)::n
real(8),intent(in)::b(0:)
real(8),intent(inout)::c(0:)
integer::i
do i=0,n-1
c(i)=3.d0*b(i)
enddo
end subroutine
end module
"""


def subject(tmp_path, source=SOURCE, *, capture=None):
    path = tmp_path / "stream.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"stream_fields::" + name: dict(FACT)
                          for name in ("old", "middle", "star", "out", "unrelated")}}
    request = {"coverage": "checked_interval", "workloads": process_fixture()[0]["workloads"],
               "build_contract": {"checked_fixture": True}, "validation_policy": {"checked_policy_fixture": True}}
    if capture:
        request["calibration"] = capture
    config = OffloadConfig(policy="auto", scope_execution="reached", exact_window=request)
    return ScopeBuilder([path], "stream_fields::advance", facts=facts,
                        options=CompilerOptions(gpu_policy="auto"), config=config)


def test_reuse_priority_keeps_producer_across_partial_native_correction(tmp_path):
    builder = subject(tmp_path)
    candidates = interval_candidates(builder)
    ordered, record = order_intervals(builder, candidates)
    assert len(ordered) == len(candidates) <= 16
    selected = record["candidates"][record["ordered_ordinals"][0]]
    assert selected["first_line"] == SOURCE.splitlines().index("call make_field(old,middle,n)") + 1
    assert {row["resource"] for row in selected["possible_reuse"]} == {
        "stream_fields::middle", "stream_fields::star"}
    assert len(selected["units"]) == 4
    assert "middle(0)=middle(0)+old(0)" in "\n".join(map(str, ordered[0].originals)).lower().replace(" ", "")
    assert not builder.edits
    assert not builder.scopes
    assert not builder.generated
    assert not builder.inline.original_numerical_regions
    assert "no effect, legality or cost authority" in record["authority"]


@pytest.mark.parametrize("capture", [None, "native", "resident"])
def test_public_discovery_uses_same_later_window_before_native_or_resident_placement(
        tmp_path, build_validation, capture):  # noqa: F811
    builder = subject(tmp_path, capture=capture)
    outputs, report = builder.run()
    scope, = report["scopes"]
    record = report["original_interval_discovery"]
    assert record["selected"]["first_line"] == SOURCE.splitlines().index("call make_field(old,middle,n)") + 1
    assert scope["checked_execution_plan"]["numerical_invocations"] == 3
    assert not scope["boundaries"]
    source = outputs[report["sources"][str(builder.entry.scope.path)]["replacement"]]
    assert source.count("call tune(unrelated,n)") == 3
    assert source.count("middle(0)=middle(0)+old(0)") == 1
    if capture is None:
        assert scope["automatic_preflight"]["selection"] == "native"
        assert scope["automatic_preflight"]["registrations"] == 0
        assert not any(name.startswith("entries/") for name in outputs)


def test_unknown_source_calls_remain_hints_and_do_not_grant_effects(tmp_path):
    source = SOURCE.replace("middle(0)=middle(0)+old(0)", "call external_correction(middle)")
    builder = subject(tmp_path, source)
    _ordered, record = order_intervals(builder, interval_candidates(builder))
    assert any(unit["unresolved_calls"] for row in record["candidates"] for unit in row["units"])
    assert not builder.edits
    assert not builder.scopes
    assert not builder.generated
    assert not builder.inline.original_numerical_regions


def test_hint_budget_exhaustion_preserves_original_bounded_search(tmp_path, monkeypatch):
    builder = subject(tmp_path)
    sites = interval_candidates(builder)
    monkeypatch.setattr("compiler.scopes.interval_ordering.NATIVE_GROUP_SCAN_LIMIT", 1)
    ordered, record = order_intervals(builder, sites)
    assert all(left is right for left, right in zip(ordered, sites, strict=True))
    assert "budget exhausted" in record["reason"]
    assert not builder.edits
    assert not builder.generated
