"""Numerical lowering consumes captured inputs even during transient file edits."""

from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes import source
from compiler.tests.test_numerical_classification import ENTRY, SOURCE
from compiler.tests.test_numerical_sources import package
from compiler.tests.test_source_scopes import FACT


def builders(tmp_path, normalized):
    if normalized:
        original, lowered, document = package(tmp_path)
        entry = "original::producer"
        roots = ("argument::a", "argument::b", "settings::weights")
    else:
        original = lowered = tmp_path / "original.f90"
        original.write_text(SOURCE)
        document, entry, roots = None, ENTRY, ("argument::a",)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(original): sha256(original.read_bytes()).hexdigest()},
             "captures": {root: FACT for root in roots}}

    def build():
        return source.ScopeBuilder([original], entry, facts=facts, options=CompilerOptions(),
                                   config=OffloadConfig("sections"), numerical_sources=document)

    return lowered, entry, build(), build()


@pytest.mark.parametrize("normalized", [False, True])
def test_transient_live_file_changes_never_enter_numerical_lowering(tmp_path, monkeypatch, normalized):
    path, entry, control, candidate = builders(tmp_path, normalized)
    expected = control.numerical(entry)
    assert expected is not None, control.numerical_reasons
    captured = path.read_bytes()
    lower_file = source.lower_file
    calls = []

    def changed_file(*args, **kwargs):
        # Restoring before publication defeats a mere final-file hash check.
        # The parser must never consume this transient replacement.
        path.write_text("this is not the captured Fortran source\n")
        try:
            calls.append(kwargs.get("source_text"))
            return lower_file(*args, **kwargs)
        finally:
            path.write_bytes(captured)

    monkeypatch.setattr(source, "lower_file", changed_file)
    actual = candidate.numerical(entry)
    assert calls == [captured.decode()]
    assert actual is not None, candidate.numerical_reasons
    assert actual.artifacts == expected.artifacts
    candidate.finish()


@pytest.mark.parametrize("normalized", [False, True])
def test_live_change_after_capture_still_prevents_publication(tmp_path, normalized):
    path, entry, _control, candidate = builders(tmp_path, normalized)
    path.write_text("this live file differs from the captured source\n")
    assert candidate.numerical(entry) is not None, candidate.numerical_reasons
    with pytest.raises(CompilationError, match="changed"):
        candidate.finish()
    assert candidate.outputs == {}
