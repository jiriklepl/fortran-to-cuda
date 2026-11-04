"""Owned team emission keeps original participation and exact package bytes."""

import json
import sys
from copy import copy, deepcopy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.combined_team import CombinedNumericalTeam
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.lexical import LexicalOwner
from compiler.scopes.source import ScopeBuilder
from compiler.tests.support.numerical_regions import (
    CHILD,
    CHILD_SOURCE,
    ENTRY,
    FACT,
    MATH_SOURCE,
    OWNER_SOURCE,
    environment_source,
    team_text,
)
from compiler.tests.test_ordinary_numerical_regions import ORDINARY_CHILD_SOURCE
from compiler.tests.test_owned_emission_image import emit_branch, emit_member_branch, member_fixture


def team_fixture(directory, kind="combined", *, companion=False):
    """Reuse the source fixtures, with the ordinary graph owner being tested."""
    child = environment_source(CHILD_SOURCE) if kind == "combined" else ORDINARY_CHILD_SOURCE
    paths = tuple(directory / name for name in ("math.f90", "child.f90", "owner.f90"))
    for path, source in zip(paths, (MATH_SOURCE, child, OWNER_SOURCE), strict=True):
        path.write_text(source)
    facts = {
        "schema_version": 1,
        "participation": "serial",
        "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths},
        "captures": {"argument::" + name: deepcopy(FACT) for name in ("a", "b", "c")},
    }
    if kind == "separate":
        facts["captures"]["quartz_numerics::coefficient"] = deepcopy(FACT)
    builder = ScopeBuilder(
        paths, ENTRY if companion else CHILD, facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="graph", host_threads=4),
    )
    return paths, builder


def teams(owner):
    return tuple((member, unit) for member in owner.execution_plan.members
                 for unit in member.units if isinstance(unit.scope, CombinedNumericalTeam))


def strict_package(directory, owner, outputs, report):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build

    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    manifest = json.loads(outputs["scope-manifest.json"])
    validate_checked_execution_plans(manifest, capture_facts=owner.builder.facts)
    for name, content in outputs.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    _scope_package_build({"artifact_root": str(directory), "scope_manifest": manifest})
    expected = {identity for units in owner.graph_unit_ids.values() for identity in units.values()}
    assert expected
    assert owner.emission.graph_emitted_units == expected
    return manifest


@pytest.mark.parametrize(("kind", "companion"), [
    ("combined", False), ("separate", False), ("combined", True),
])
def test_original_team_images_preserve_every_artifact(tmp_path, monkeypatch, kind, companion):
    from compiler.frontend import source_effects
    from compiler.frontend.summary_cache import SummaryCache
    from compiler.scopes.storage_admission import active_admission

    monkeypatch.setattr(source_effects, "_DEFAULT_SUMMARY_CACHE", SummaryCache())
    paths, parent = team_fixture(tmp_path, kind, companion=companion)
    originals = tuple(path.read_bytes() for path in paths)
    facts = deepcopy(parent.facts)
    with monkeypatch.context() as patch:
        patch.setattr(LexicalOwner, "supports_owned_emission", lambda _self, _plan: False)
        patch.setattr(LexicalOwner, "supports_owned_unit_emission", lambda _self, _unit: False)
        for _ in range(2):
            with GenerationTransaction(parent) as branch:
                emit_branch(branch.builder)
        with GenerationTransaction(parent) as branch:
            old_owner, old_outputs, old_report = emit_branch(branch.builder)
            assert teams(old_owner)
            old_manifest = strict_package(tmp_path / "legacy", old_owner, old_outputs, old_report)

    published = []
    original_publish = LexicalOwner._publish_owned_emission

    def publish(targets, image, tickets):
        assert active_admission() is None
        published.append((targets, tickets))
        return original_publish(targets, image, tickets)

    monkeypatch.setattr(LexicalOwner, "_publish_owned_emission", staticmethod(publish))
    with GenerationTransaction(parent) as branch:
        owner, outputs, report = emit_branch(branch.builder)
        manifest = strict_package(tmp_path / "owned", owner, outputs, report)
        assert outputs == old_outputs
        assert manifest == old_manifest
        assert len(published) == 1, "the authentic team owner did not activate owned emission"
        _targets, tickets = published[0]
        plan = owner.execution_plan
        assert len(tickets) == sum(len(plan.units(member)) for member in plan.members)
        assert len({ticket[5] for ticket in tickets}) == len(tickets)
        for ticket in tickets:
            member = plan.members[ticket[2]]
            assert ticket[3] == member.routine.qualified
            assert ticket[5] == owner.graph_unit_ids[ticket[3]][str(ticket[4])]
            assert ticket[8][0] == str(member.routine.scope.path)
            for local, canonical, slot in ticket[6]:
                binding, actual_slot = member.resources[local]
                assert canonical == member.canonical(binding).root
                assert slot == actual_slot
        current_teams = teams(owner)
        assert current_teams
        for member, unit in current_teams:
            state = member.unit_emission(unit)
            assert state.prepared is not None
            public = unit.scope.public()["combined_numerical_team"]
            assert public["host_threads"] == 4
            assert public["expected_omp_level"] == 1
            assert public["closed_is_absorbing"]
            assert not public["nested_team_created"]
            assert not public["native_effects_authority"]
            assert public["completion"]["proof_identity"] == unit.scope.prepared.completion_proof.identity
        text = outputs[report["sources"][str(paths[1])]["replacement"]]
        assert text.count("call ieee_set_halting_mode(ieee_invalid,.false.)") == 1
        assert "error stop 'combined original team numerical execution failed'" in text
        assert "error stop 'inherited collective entry executed or invalid state'" in text
        if kind == "combined":
            helper_member, helper_unit = next(
                (member, unit) for member, unit in current_teams
                if unit.scope.prepared.extraction.numerical_helpers
            )
            prepared = helper_member.unit_emission(helper_unit).prepared
            replacement, = (replacement for first, last, replacement in prepared.finished_edits
                            if (first, last) == (helper_unit.scope.first, helper_unit.scope.last))
            # Use the exact source-bound replacement. A companion retains its
            # pristine native body before the generated team's sole vote.
            assert replacement.count("fort_team_environment_supported = .true.") == 1
            assert replacement.count("call cell_formula(g,s,a(i),i)") == (2 if companion else 1)
            vote = replacement.index("fort_team_environment_supported = .true.")
            group = replacement.rfind("!$omp parallel", 0, vote)
            assert group >= 0
            body = team_text(replacement[group:])
            assert body.index("fort_team_omp_level() /= 1") < body.index("fort_scope_register(")
            execution = body.index("fort_inline_team_run(")
            failure = body.index("error stop 'inherited collective entry executed or invalid state'")
            native = body.index("if (.not. fort_team_run) then", failure)
            assert execution < failure < native
            assert "!$omp barrier" in body[failure:native]
            # Graph mode retains the checked completion dependency above;
            # its legacy emitter does not insert a reached-mode per-team wait.
            assert "fort_scope_wait(" not in body
            assert body.count("call cell_formula(g,s,a(i),i)") == 1
        else:
            assert "!$omp end do nowait" in text
            assert text.count("call touch_face(b)") == 1
    assert tuple(path.read_bytes() for path in paths) == originals
    assert parent.facts == facts
    assert not parent.outputs
    assert not parent.edits
    assert not parent.scopes


def test_team_image_publication_never_calls_live_team_authorities(tmp_path, monkeypatch):
    from compiler.scopes.execution_plan import CheckedExecutionPlan
    from compiler.scopes.storage_epoch import StorageEpochProof

    _paths, parent = team_fixture(tmp_path)
    original_publish = LexicalOwner._publish_owned_emission
    publications = []

    def forbidden(*_args, **_kwargs):
        pytest.fail("owned team publication revisited live source or provider authority")

    def publish(targets, image, tickets):
        publications.append(image.identity)
        with monkeypatch.context() as patch:
            patch.setattr(CombinedNumericalTeam, "_validate", forbidden)
            patch.setattr(CombinedNumericalTeam, "public", forbidden)
            patch.setattr(CombinedNumericalTeam, "emit_original", forbidden)
            patch.setattr(CheckedExecutionPlan, "validate", forbidden)
            patch.setattr(StorageEpochProof, "validate", forbidden)
            patch.setattr(ScopeBuilder, "capture_fact", forbidden)
            patch.setattr(ScopeBuilder, "entry_artifacts", forbidden)
            patch.setattr(LexicalOwner, "canonical", forbidden)
            return original_publish(targets, image, tickets)

    monkeypatch.setattr(LexicalOwner, "_publish_owned_emission", staticmethod(publish))
    with GenerationTransaction(parent) as branch:
        owner, outputs, report = emit_branch(branch.builder)
        assert teams(owner)
        strict_package(tmp_path / "package", owner, outputs, report)
    assert len(publications) == 1


@pytest.mark.parametrize("fault", ["source", "canonical_slot", "completion"])
def test_team_image_seal_rejects_changed_source_or_receipt(tmp_path, monkeypatch, fault):
    paths, parent = team_fixture(tmp_path)
    original_seal = LexicalOwner._seal_owned_emission
    changed = []

    def seal(self, plan, preparation):
        member, unit = teams(self)[0]
        if fault == "source":
            paths[1].write_text(paths[1].read_text() + "! changed before team seal\n")
        elif fault == "canonical_slot":
            root, (binding, slot) = next(iter(member.resources.items()))
            dict.__setitem__(member.resources, root, (binding, slot + 1))
        else:
            proof = unit.scope.prepared.completion_proof
            object.__setattr__(unit.scope.prepared, "completion_proof", copy(proof))
        changed.append(fault)
        return original_seal(self, plan, preparation)

    monkeypatch.setattr(LexicalOwner, "_seal_owned_emission", seal)

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


@pytest.mark.parametrize("failure", ["prepare", "publish"])
def test_team_preparation_or_publication_failure_adopts_nothing(tmp_path, monkeypatch, failure):
    _paths, parent = team_fixture(tmp_path)
    reached = []
    if failure == "prepare":
        original = LexicalOwner.prepare_unit

        def prepare(self, unit, index):
            image = original(self, unit, index)
            if isinstance(unit.scope, CombinedNumericalTeam):
                reached.append(failure)
                raise CompilationError("injected owned team preparation failure")
            return image

        monkeypatch.setattr(LexicalOwner, "prepare_unit", prepare)
    else:
        original = LexicalOwner._publish_owned_emission

        def publish(targets, image, tickets):
            original(targets, image, tickets)
            reached.append(failure)
            raise CompilationError("injected owned team publication failure")

        monkeypatch.setattr(LexicalOwner, "_publish_owned_emission", staticmethod(publish))

    def attempt():
        with GenerationTransaction(parent) as branch:
            emit_branch(branch.builder)
            branch.adopt()

    with pytest.raises(CompilationError, match="injected owned team"):
        attempt()
    assert reached == [failure]
    assert not parent.outputs
    assert not parent.edits
    assert not parent.scopes


@pytest.mark.parametrize("count", [1, 7])
def test_global_admission_checks_are_constant_for_original_teams(tmp_path, monkeypatch, count, record_property):
    from compiler.scopes.storage_admission import StorageAdmission
    from compiler.tests.test_separate_numerical_team import BODY, SOURCE

    path = tmp_path / "separate.f90"
    path.write_text(SOURCE.replace(BODY, BODY * count))
    facts = {
        "schema_version": 1,
        "participation": "serial",
        "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
        "captures": {"argument::" + name: deepcopy(FACT) for name in ("a", "b", "c")},
    }
    parent = ScopeBuilder(
        [path], "bounded_team::step", facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="graph", host_threads=4),
    )
    original = StorageAdmission._check
    checks = []

    def checked(self):
        checks.append(self._image_preparation)
        return original(self)

    monkeypatch.setattr(StorageAdmission, "_check", checked)
    with GenerationTransaction(parent) as branch:
        owner, outputs, report = emit_branch(branch.builder)
        assert len(teams(owner)) == count
        strict_package(tmp_path / "package", owner, outputs, report)
    assert len(checks) == 5
    assert all(token is not None for token in checks)
    record_property("original_team_count", count)
    record_property("owned_global_admission_checks", len(checks))


def test_finish_retains_its_guard_before_the_first_callback(tmp_path, monkeypatch):
    from compiler.frontend.source_effects import SourceEffects

    _path, parent = member_fixture(tmp_path, "readonly")
    with GenerationTransaction(parent) as branch:
        emit_member_branch(branch.builder, "readonly")
        branch.adopt()
    guard = parent._owned_emission_prepared_channels
    name = next(name for name in parent.outputs if name.endswith(".cu") and "runtime" not in name)
    code = ScopeBuilder.finish.__code__
    original = SourceEffects.verify_publication
    calls = []

    def verify(self):
        result = original(self)
        if sys._getframe(1).f_code is code:
            calls.append(self)
            if len(calls) == 1:
                parent._owned_emission_prepared_channels = None
            elif len(calls) == 2:
                parent.outputs[name] += "\n! output changed after guard removal\n"
        return result

    monkeypatch.setattr(SourceEffects, "verify_publication", verify)
    with pytest.raises(CompilationError, match="publication changed its entry guard"):
        parent.finish(materialize_native_effects=False)
    assert len(calls) == 2
    assert parent._owned_emission_prepared_channels is guard


@pytest.mark.parametrize("stage", ["initial_verify", "report_callback"])
def test_failed_finish_restores_guard_and_rejects_corrupt_retry(tmp_path, monkeypatch, stage):
    from compiler.frontend.source_effects import SourceEffects

    _path, parent = member_fixture(tmp_path, "readonly")
    with GenerationTransaction(parent) as branch:
        emit_member_branch(branch.builder, "readonly")
        branch.adopt()
    guard = parent._owned_emission_prepared_channels
    name = next(name for name in parent.outputs if name.endswith(".cu") and "runtime" not in name)
    code = ScopeBuilder.finish.__code__
    original_verify = SourceEffects.verify_publication
    original_report = SourceEffects.report
    callbacks = []

    def corrupt_and_fail():
        callbacks.append(stage)
        parent._owned_emission_prepared_channels = None
        parent.outputs[name] += "\n! changed by a failed finish callback\n"
        if stage == "report_callback":
            raise CompilationError("injected finish callback failure")

    def verify(self):
        result = original_verify(self)
        if stage == "initial_verify" and sys._getframe(1).f_code is code:
            corrupt_and_fail()
        return result

    def report(self, *args, **kwargs):
        result = original_report(self, *args, **kwargs)
        if stage == "report_callback" and sys._getframe(1).f_code is code:
            corrupt_and_fail()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(SourceEffects, "verify_publication", verify)
        patch.setattr(SourceEffects, "report", report)
        reason = "sealed generated artifact" if stage == "initial_verify" else "injected finish callback failure"
        with pytest.raises(CompilationError, match=reason):
            parent.finish(materialize_native_effects=False)
    assert callbacks == [stage]
    assert parent._owned_emission_prepared_channels is guard
    # The failed callback cannot turn a later ordinary finish into an
    # unguarded publisher of its corrupted provider.
    with pytest.raises(CompilationError, match="sealed generated artifact"):
        parent.finish(materialize_native_effects=False)
    assert parent._owned_emission_prepared_channels is guard
