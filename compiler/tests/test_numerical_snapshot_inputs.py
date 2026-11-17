"""Numerical lowering consumes captured inputs even during transient file edits."""

from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.collective_calibration import fixture_facts, fixture_source
from compiler.offload.config import OffloadConfig
from compiler.scopes import collective, source
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


def collective_builders(tmp_path, normalized):
    original = lowered = tmp_path / "original.f90"
    original.write_text(fixture_source("memory", 4, 64, owner_calls=1))
    facts = fixture_facts(original, 4)
    entry, document = "team_operations::work", None
    if normalized:
        lowered = tmp_path / "normalized.f90"
        lowered.write_text("""module extracted
implicit none
contains
subroutine work(a,b,c,n)
real(8),intent(in)::a(:,:,:),c(:,:,:)
real(8),intent(inout)::b(:,:,:)
integer,intent(in)::n
integer::i
do i=1,n
b(i,1,1)=a(i,1,1)+0.25*c(i,1,1)
enddo
end subroutine
end module
""")
        document = {"schema_version": 1, "source_inputs": facts["sources"], "entries": [{
            "procedure": entry, "source_sha256": facts["sources"][str(original)],
            "path": str(lowered), "sha256": sha256(lowered.read_bytes()).hexdigest(),
            "entry": "extracted::work", "normalization": "whole_storage_rebased_v1",
            "participation": "serial_coordinator", "capture_safe": True, "preserves_source_order": True,
            "parameters": [{"name": name, "resource": "argument::" + name,
                            **({"physical_origin": [0, 0, 0]} if name != "n" else {})}
                           for name in ("a", "b", "c", "n")]}]}

    def build():
        return collective.CollectiveScopeBuilder(
            [original], "team_operations::step", facts=facts, options=CompilerOptions(),
            config=OffloadConfig("sections", collective=True), numerical_sources=document)

    return lowered, entry, build(), build()


@pytest.mark.parametrize("collective_owner", [False, True], ids=["serial", "collective"])
@pytest.mark.parametrize("normalized", [False, True])
def test_transient_live_file_changes_never_enter_numerical_lowering(tmp_path, monkeypatch, normalized,
                                                                  collective_owner):
    build = collective_builders if collective_owner else builders
    path, entry, control, candidate = build(tmp_path, normalized)
    expected = control.numerical(entry)
    assert expected is not None, control.numerical_reasons
    captured = path.read_bytes()
    module = collective if collective_owner else source
    lower_file = module.lower_file
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

    monkeypatch.setattr(module, "lower_file", changed_file)
    actual = candidate.numerical(entry)
    assert calls == [captured.decode()]
    assert actual is not None, candidate.numerical_reasons
    assert actual.artifacts == expected.artifacts
    candidate.finish()


@pytest.mark.parametrize("collective_owner", [False, True], ids=["serial", "collective"])
@pytest.mark.parametrize("normalized", [False, True])
def test_live_change_after_capture_still_prevents_publication(tmp_path, normalized, collective_owner):
    build = collective_builders if collective_owner else builders
    path, entry, _control, candidate = build(tmp_path, normalized)
    path.write_text("this live file differs from the captured source\n")
    assert candidate.numerical(entry) is not None, candidate.numerical_reasons
    with pytest.raises(CompilationError, match="changed"):
        candidate.finish()
    assert candidate.outputs == {}
