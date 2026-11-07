"""AUTO placement uses the same detached emission without changing decisions."""
import gc
from dataclasses import replace

import pytest

from compiler.frontend import source_effects
from compiler.frontend.summary_cache import SummaryCache
from compiler.ir import CompilationError
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.lexical import LexicalOwner
from compiler.scopes.storage_admission import active_admission
from compiler.tests.test_owned_emission_image import check_package, emit_branch, source_builder
from compiler.tests.test_scoped_planning_entries import calibration


def test_auto_owned_emission_keeps_every_artifact_and_runtime_selector(tmp_path, monkeypatch):
    monkeypatch.setattr(source_effects, "_DEFAULT_SUMMARY_CACHE", SummaryCache())
    path, parent = source_builder(tmp_path)
    parent.config = replace(parent.config, policy="auto", profile=calibration())
    original = path.read_bytes()
    warmups = []
    with monkeypatch.context() as patch:
        patch.setattr(LexicalOwner, "supports_owned_emission", lambda *_: False)
        patch.setattr(LexicalOwner, "supports_owned_unit_emission", lambda *_: False)
        for _ in range(2):
            with GenerationTransaction(parent) as branch:
                warmups.append(emit_branch(branch.builder))
        gc.collect()
        with GenerationTransaction(parent) as branch:
            _owner, expected, report = emit_branch(branch.builder)
            check_package(tmp_path / "contextual", branch.builder, expected, report)

    published = []
    publish = LexicalOwner._publish_owned_emission

    def retain(targets, image, tickets):
        assert active_admission() is None
        published.append(tickets)
        return publish(targets, image, tickets)

    monkeypatch.setattr(LexicalOwner, "_publish_owned_emission", staticmethod(retain))
    gc.collect()
    with GenerationTransaction(parent) as branch:
        owner, actual, report = emit_branch(branch.builder)
        check_package(tmp_path / "owned", branch.builder, actual, report)
        assert len(published) == 1
        assert actual == expected
        assert all(member.unit_emission(unit).prepared is not None
                   for member in owner.execution_plan.members for unit in owner.execution_plan.units(member))
        text = "\n".join(value for name, value in actual.items() if name.startswith("sources/"))
        assert "fort_mode = 2" in text
        assert "fort_choose" in text
    assert path.read_bytes() == original
    assert not parent.outputs


def test_missing_auto_estimates_do_not_construct_an_empty_image(tmp_path, monkeypatch):
    path, parent = source_builder(tmp_path)
    parent.config = replace(parent.config, policy="auto", profile=None)
    original = path.read_bytes()

    def forbidden(*_args):
        pytest.fail("unpriced native owner must not publish numerical unit tickets")

    monkeypatch.setattr(LexicalOwner, "_publish_owned_emission", staticmethod(forbidden))
    with GenerationTransaction(parent) as branch:
        _owner, _outputs, report = emit_branch(branch.builder)
        scope, = report["scopes"]
        assert scope["automatic_preflight"]["selection"] == "native"
        assert scope["automatic_preflight"]["contexts_created"] == 0
    assert path.read_bytes() == original
    assert not parent.outputs


@pytest.mark.parametrize("fault", ["profile", "policy"])
def test_auto_image_rejects_changed_placement_inputs(tmp_path, monkeypatch, fault):
    _path, parent = source_builder(tmp_path)
    parent.config = replace(parent.config, policy="auto", profile=calibration())
    prepare = LexicalOwner.prepare_unit
    changed = []

    def prepare_then_change(self, unit, index):
        prepared = prepare(self, unit, index)
        if not changed:
            changed.append(fault)
            if fault == "profile":
                self.builder.config.profile["scoped"]["costs"]["allocation_seconds"] = 100.0
            else:
                self.builder.config = replace(self.builder.config, policy="sections")
        return prepared

    monkeypatch.setattr(LexicalOwner, "prepare_unit", prepare_then_change)

    def attempt():
        with GenerationTransaction(parent) as branch:
            emit_branch(branch.builder)
            branch.adopt()

    with pytest.raises(CompilationError):
        attempt()
    assert changed == [fault]
    assert not parent.outputs
    assert not parent.edits
    assert not parent.scopes
