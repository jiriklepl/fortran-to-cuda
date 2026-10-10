"""Known negative view candidates do not expand unrelated effect closures."""

from hashlib import sha256
from unittest.mock import Mock

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder


def builder(tmp_path, *, attributes="", body="a=2*a", internal=False):
    child = f"""subroutine child(a)
real(8),intent(inout){attributes}::a(:)
{body}
end subroutine
"""
    contents = ("subroutine owner()\ncontains\n" + child + "end subroutine\n" if internal else child)
    path = tmp_path / "renamed_views.f90"
    path.write_text("module renamed_views\ncontains\n" + contents + "end module\n")
    qualified = "renamed_views::owner::child" if internal else "renamed_views::child"
    facts = {"schema_version": 1, "participation": "serial", "captures": {},
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()}}
    result = ScopeBuilder([path], qualified, facts=facts, options=CompilerOptions(), config=OffloadConfig("sections"))
    return result, qualified


@pytest.mark.parametrize(("attributes", "body", "internal"), [
    ("", "a=2*a", False), (",optional", "call unknown(a)", False),
    (",allocatable", "call unknown(a)", False), (",contiguous", "call unknown(a)", False),
    ("", "call unknown(a)", True), ("", "", False),
])
def test_wrapper_known_boundaries_skip_transitive_summary(tmp_path, monkeypatch, attributes, body, internal):
    owner, child = builder(tmp_path, attributes=attributes, body=body, internal=internal)
    summary = Mock(side_effect=AssertionError("known negative candidate expanded its effect closure"))
    monkeypatch.setattr(owner.analysis, "summarize", summary)
    assert not owner.view_wrapper_supported(child)
    summary.assert_not_called()


@pytest.mark.parametrize(("attributes", "body"), [
    ("", "call unknown(a)"), (",optional", "a=2*a"),
    (",allocatable", "a=2*a"), (",contiguous", "a=2*a"),
])
def test_native_known_boundaries_skip_transitive_summary(tmp_path, monkeypatch, attributes, body):
    owner, child = builder(tmp_path, attributes=attributes, body=body)
    summary = Mock(side_effect=AssertionError("known negative candidate expanded its effect closure"))
    monkeypatch.setattr(owner.analysis, "summarize", summary)
    assert not owner.view_native_supported(child)
    summary.assert_not_called()


@pytest.mark.parametrize("kind", ["wrapper", "native"])
def test_remaining_candidates_still_require_complete_effect_proof(tmp_path, monkeypatch, kind):
    owner, child = builder(tmp_path, body="call unknown(a)" if kind == "wrapper" else "a=2*a")
    summary = Mock(return_value={"complete": False})
    monkeypatch.setattr(owner.analysis, "summarize", summary)
    assert not getattr(owner, "view_" + kind + "_supported")(child)
    summary.assert_called_once_with(child)
