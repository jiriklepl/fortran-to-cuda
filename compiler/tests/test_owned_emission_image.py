"""Owned emission preserves authentic source bytes without live render inputs."""

import json
from contextlib import contextmanager
from copy import copy, deepcopy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.lexical import LexicalOwner
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_lexical_source_owner import SOURCE
from compiler.tests.test_source_scopes import FACT


def source_builder(directory, *, source=None):
    path = directory / "owner.f90"
    # The guarded native SUM consumes the produced array before the consumer.
    path.write_text(source or SOURCE.replace("call opaque(b,n)", "b(-2)=b(-2)+sum(b)"))
    facts = {
        "schema_version": 1,
        "participation": "serial",
        "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
        "captures": {"argument::" + name: deepcopy(FACT) for name in ("a", "b", "out")},
    }
    builder = ScopeBuilder(
        [path], "local_owner::step", facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="graph"),
    )
    return path, builder


def check_package(directory, builder, outputs, report):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build

    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    manifest = json.loads(outputs["scope-manifest.json"])
    plan = manifest["scopes"][0]["checked_execution_plan"]
    assert plan["numerical_invocations"] == 2
    native = [operation for procedure in plan["procedures"] for unit in procedure["units"]
              for operation in unit["operations"]["native_operations"]]
    assert any(operation["resource_effects"].get("argument::b") == ["read", "write"]
               for operation in native)
    validate_checked_execution_plans(manifest, capture_facts=builder.facts)
    for name, content in outputs.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    _scope_package_build({"artifact_root": str(directory), "scope_manifest": manifest})
    return manifest


def emit_branch(builder):
    """The normal graph-owner body inside the existing transaction boundary."""
    from compiler.frontend.source_effects import _children
    from compiler.scopes.conditional_return import prepare as prepare_return
    from compiler.scopes.graph import GraphOwner
    from compiler.scopes.native_ancestor import prepare

    owner = GraphOwner(builder)
    prepare_return(owner)
    nodes = prepare(builder, _children(builder.entry.execution))
    builder.generated.update(builder.inline.generated)
    builder.numerical_ir.update(builder.inline.ir)
    builder.numerical_reasons.update({name: None for name in builder.inline.regions})
    assert owner.run(nodes)
    outputs, report = builder.finish(materialize_native_effects=False)
    return owner, outputs, report


def test_owned_unit_preparation_preserves_all_files_and_strict_package(tmp_path, monkeypatch):
    path, parent = source_builder(tmp_path)
    original = path.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(LexicalOwner, "supports_owned_unit_emission", lambda _self, _unit: False)
        for _ in range(2):
            with GenerationTransaction(parent) as branch:
                emit_branch(branch.builder)
        with GenerationTransaction(parent) as branch:
            _owner, old_outputs, old_report = emit_branch(branch.builder)
            check_package(tmp_path / "old", branch.builder, old_outputs, old_report)
    with GenerationTransaction(parent) as branch:
        _owner, new_outputs, new_report = emit_branch(branch.builder)
        check_package(tmp_path / "owned", branch.builder, new_outputs, new_report)
    assert new_outputs == old_outputs
    assert path.read_bytes() == original
    assert not parent.outputs
    assert not parent.scopes


def test_prepared_unit_render_does_not_consume_live_compiler_objects(tmp_path, monkeypatch):
    from compiler.frontend.numerical_preparation import NumericalPreparationProvider
    from compiler.scopes import source_device_tasks, unit_emission
    from compiler.scopes.storage_epoch import StorageEpochProof

    path, builder = source_builder(tmp_path)
    images = []
    original_prepare = LexicalOwner.prepare_unit

    def retain(self, unit, index):
        image = original_prepare(self, unit, index)
        images.append(image)
        return image

    monkeypatch.setattr(LexicalOwner, "prepare_unit", retain)
    outputs, report = builder.run()
    check_package(tmp_path / "package", builder, outputs, report)
    assert len(images) >= 3
    expected = [unit_emission.render_unit(image) for image in images]

    def forbidden(*_args, **_kwargs):
        pytest.fail("detached rendering consulted a live compiler authority")

    with monkeypatch.context() as patch:
        patch.setattr(ScopeBuilder, "capture_fact", forbidden)
        patch.setattr(ScopeBuilder, "entry_artifacts", forbidden)
        patch.setattr(LexicalOwner, "canonical", forbidden)
        patch.setattr(LexicalOwner, "prepare_unit", forbidden)
        patch.setattr(StorageEpochProof, "validate", forbidden)
        patch.setattr(source_device_tasks, "_provider", forbidden)
        for proof in builder.analysis._numerical_preparations.values():
            patch.setattr(type(proof), "validate", forbidden)
        for name in ("prepare", "lookup"):
            if hasattr(NumericalPreparationProvider, name):
                patch.setattr(NumericalPreparationProvider, name, forbidden)
        for binding in builder.entry.scope.bindings.values():
            patch.setattr(type(binding), "public", forbidden)
        assert [unit_emission.render_unit(image) for image in images] == expected
    assert path.read_bytes()


@pytest.mark.parametrize("fault", ["mutable_edits", "boolean_span", "mutable_lines", "live_diagnostics"])
def test_prepared_unit_rejects_unowned_render_inputs(fault):
    from compiler.scopes.unit_emission import PreparedUnitEmission

    edits, diagnostics = ((1, 1, ("continue",)),), b"{}"
    if fault == "mutable_edits":
        edits = list(edits)
    elif fault == "boolean_span":
        edits = ((True, 1, ("continue",)),)
    elif fault == "mutable_lines":
        edits = ((1, 1, ["continue"]),)
    else:
        diagnostics = {"provider": object()}
    with pytest.raises(CompilationError):
        PreparedUnitEmission(edits, diagnostics)


@contextmanager
def prepared_image_case(directory):
    path, parent = source_builder(directory)
    with GenerationTransaction(parent) as branch:
        # These API controls publish the one image manually. Prepare their
        # authentic plan through ordinary full admission, without first
        # publishing an automatic image in the same transaction.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(LexicalOwner, "supports_owned_emission", lambda _self, _plan: False)
            owner, outputs, report = emit_branch(branch.builder)
        plan = owner.execution_plan
        token = plan.emission_preparation()
        units = []
        for member_index, member in enumerate(plan.members):
            for ordinal, unit in enumerate(plan.units(member)):
                state = member.unit_emission(unit)
                edits = tuple((str(member.routine.scope.path), first, last, text, False)
                              for first, last, text in state.source_edits)
                diagnostics = json.dumps({
                    "compute_identity": state.compute_identity,
                    "native_preflight": state.native_preflight,
                    "fresh_native_prefix": state.fresh_native_prefix,
                }, separators=(",", ":"), allow_nan=False).encode()
                units.append((token.ticket(member_index, ordinal), edits, diagnostics))
        payload = {
            "units": tuple(units),
            "edits": tuple((str(source), first, last, text, False)
                           for source, rows in branch.builder.edits.items()
                           for first, last, text in rows),
            "artifacts": tuple((name, text.encode()) for name, text in outputs.items()),
            "scopes": tuple(json.dumps(scope, separators=(",", ":"), allow_nan=False).encode()
                            for scope in report["scopes"]),
        }
        yield path, parent, branch, owner, plan, token, payload
    assert not parent.outputs
    assert not parent.edits
    assert not parent.scopes


@pytest.mark.parametrize("consumer", ["tickets", "publication"])
def test_owned_image_consumes_authentic_edits_after_live_inputs_are_poisoned(tmp_path, monkeypatch, consumer):
    from compiler.scopes.execution_plan import CheckedExecutionPlan
    from compiler.scopes.storage_epoch import StorageEpochProof

    with prepared_image_case(tmp_path) as (_path, _parent, _branch, owner, plan, token, payload):
        image = plan.seal_emission_image(token, **payload)
        assert token.sealed
        assert type(image.payload) is bytes
        assert sha256(image.payload).hexdigest() == image.identity

        def forbidden(*_args, **_kwargs):
            pytest.fail("owned image rendering consulted live source or authority")

        with monkeypatch.context() as patch:
            patch.setattr(CheckedExecutionPlan, "validate", forbidden)
            patch.setattr(StorageEpochProof, "validate", forbidden)
            patch.setattr(ScopeBuilder, "capture_fact", forbidden)
            patch.setattr(ScopeBuilder, "entry_artifacts", forbidden)
            patch.setattr(LexicalOwner, "canonical", forbidden)
            patch.setattr(LexicalOwner, "prepare_unit", forbidden)
            patch.setattr(type(owner.builder.analysis), "verify_publication", forbidden)
            for binding in owner.routine.scope.bindings.values():
                patch.setattr(type(binding), "public", forbidden)
            if consumer == "tickets":
                for ticket, edits, diagnostics in payload["units"]:
                    assert image.consume(ticket) == (edits, diagnostics)
                assert image.finish() == (payload["edits"], payload["artifacts"], payload["scopes"])
            else:
                LexicalOwner._publish_owned_emission(owner.builder, image, token.tickets)
                assert owner.builder.outputs == {name: text.decode() for name, text in payload["artifacts"]}
                assert owner.builder.scopes == [json.loads(record) for record in payload["scopes"]]
        with pytest.raises(CompilationError, match="completed"):
            image.finish()


@pytest.mark.parametrize("fault", ["duplicate", "missing", "reordered", "slot", "source"])
def test_seal_rejects_changed_ticket_inventory(tmp_path, fault):
    with prepared_image_case(tmp_path) as (_path, _parent, _branch, _owner, plan, token, payload):
        units = list(payload["units"])
        assert len(units) >= 3
        if fault == "duplicate":
            units[1] = units[0]
        elif fault == "missing":
            units.pop()
        elif fault == "reordered":
            units.reverse()
        else:
            ticket, edits, diagnostics = units[0]
            ticket = list(ticket)
            if fault == "slot":
                slots = list(ticket[6])
                assert slots
                root, canonical, number = slots[0]
                slots[0] = (root, canonical, number + 1)
                ticket[6] = tuple(slots)
            else:
                geometry = list(ticket[8])
                geometry[0] += ".foreign"
                ticket[8] = tuple(geometry)
            units[0] = (tuple(ticket), edits, diagnostics)
        with pytest.raises(CompilationError):
            plan.seal_emission_image(token, **{**payload, "units": tuple(units)})
        assert not token.sealed


@pytest.mark.parametrize("fault", ["drop", "replace", "duplicate", "empty_unit"])
def test_seal_binds_unit_edits_to_exact_global_publication(tmp_path, fault):
    with prepared_image_case(tmp_path) as (_path, _parent, _branch, _owner, plan, token, payload):
        unit_edit = payload["units"][0][1][0]
        edits = list(payload["edits"])
        position = edits.index(unit_edit)
        if fault == "drop":
            edits.pop(position)
        elif fault == "replace":
            source, first, last, text, prepend = unit_edit
            edits[position] = (source, first, last, text + "! altered publication\n", prepend)
        elif fault == "duplicate":
            edits.append(unit_edit)
        else:
            units = list(payload["units"])
            ticket, unit_edits, diagnostics = units[0]
            for edit in unit_edits:
                edits.remove(edit)
            units[0] = (ticket, (), diagnostics)
            payload = {**payload, "units": tuple(units)}
        with pytest.raises(CompilationError):
            plan.seal_emission_image(token, **{**payload, "edits": tuple(edits)})
        assert not token.sealed


@pytest.mark.parametrize("fault", ["source", "generation", "owner", "configuration"])
def test_seal_rejects_changed_live_preparation_context(tmp_path, fault):
    from dataclasses import replace

    with prepared_image_case(tmp_path) as (path, _parent, branch, owner, plan, token, payload):
        original_source = path.read_bytes()
        original_generation = branch.builder._generation_transaction_marker
        original_config = branch.builder.config
        try:
            if fault == "source":
                path.write_bytes(original_source + b"! changed before image seal\n")
            elif fault == "generation":
                branch.builder._generation_transaction_marker = object()
            elif fault == "owner":
                object.__setattr__(owner, "execution_plan", copy(plan))
            else:
                branch.builder.config = replace(original_config, host_threads=2)
            with pytest.raises(CompilationError):
                plan.seal_emission_image(token, **payload)
        finally:
            path.write_bytes(original_source)
            branch.builder._generation_transaction_marker = original_generation
            branch.builder.config = original_config
            object.__setattr__(owner, "execution_plan", plan)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "reordered", "replaced"])
def test_image_rejects_incomplete_or_changed_publication_order(tmp_path, fault):
    with prepared_image_case(tmp_path) as (_path, _parent, _branch, _owner, plan, token, payload):
        image = plan.seal_emission_image(token, **payload)
        first, second = token.tickets[:2]

        def changed_publication():
            if fault == "missing":
                image.finish()
            elif fault == "duplicate":
                image.consume(first)
                image.consume(first)
            elif fault == "reordered":
                image.consume(second)
            else:
                image.consume((*first[:-1], ("foreign.f90", 1, 1)))

        with pytest.raises(CompilationError):
            changed_publication()


@pytest.mark.parametrize("kind", ["preparation", "image"])
def test_forged_capsule_never_invokes_foreign_callback(kind):
    from compiler.scopes.emission_image import EmissionPreparation, OwnedEmissionImage

    called = []

    def hostile(*args):
        called.append(args)
        raise AssertionError("foreign callback executed")

    if kind == "preparation":
        forged = tuple.__new__(EmissionPreparation, (object(), (), hostile))
        with pytest.raises(CompilationError):
            forged.check(object())
    else:
        forged = tuple.__new__(OwnedEmissionImage, ("foreign", b"{}", hostile))
        with pytest.raises(CompilationError):
            forged.finish()
    assert called == []


@pytest.mark.parametrize("phase", ["prepare", "publish"])
def test_failure_in_owned_route_never_adopts_candidate_artifacts(tmp_path, monkeypatch, phase):
    from compiler.scopes.storage_admission import active_admission

    _path, parent = source_builder(tmp_path)
    reached = []
    if phase == "prepare":
        original = LexicalOwner.prepare_unit

        def fail_prepare(self, unit, index):
            original(self, unit, index)
            reached.append("prepare")
            raise RuntimeError("owned preparation failed")

        monkeypatch.setattr(LexicalOwner, "prepare_unit", fail_prepare)
    else:
        original = LexicalOwner._publish_owned_emission

        def fail_publish(builder, image, tickets):
            original(builder, image, tickets)
            assert builder.edits
            assert builder.scopes
            reached.append("publish")
            raise RuntimeError("owned publication failed")

        monkeypatch.setattr(LexicalOwner, "_publish_owned_emission", staticmethod(fail_publish))
    def attempt():
        with GenerationTransaction(parent) as branch:
            emit_branch(branch.builder)
            branch.adopt()

    with pytest.raises(RuntimeError, match="owned .* failed"):
        attempt()
    assert reached == [phase]
    assert active_admission() is None
    assert not parent.outputs
    assert not parent.edits
    assert not parent.scopes
    assert not parent.resolved_calls


def test_preparation_pins_absent_generation_marker(tmp_path):
    from compiler.tests.test_checked_execution_plan import scanned

    with scanned(tmp_path, config=OffloadConfig(policy="sections", scope_execution="graph")) as (owner, plan):
        assert getattr(owner.builder, "_generation_transaction_marker", None) is None
        token = plan.emission_preparation()
        owner.builder._generation_transaction_marker = object()
        try:
            with pytest.raises(CompilationError, match="generation"):
                token.check(plan)
        finally:
            del owner.builder._generation_transaction_marker


@pytest.mark.parametrize("kind", ["preparation", "image"])
def test_copying_genuine_control_does_not_issue_a_second_capsule(tmp_path, kind):
    with prepared_image_case(tmp_path) as (_path, _parent, _branch, _owner, plan, token, payload):
        genuine = token if kind == "preparation" else plan.seal_emission_image(token, **payload)
        copied = tuple.__new__(type(genuine), tuple(genuine))

        def consume_copy():
            if kind == "preparation":
                copied.check(plan)
            else:
                copied.finish()

        with pytest.raises(CompilationError):
            consume_copy()


def test_ticket_type_spoof_cannot_invoke_custom_equality(tmp_path):
    calls = []

    class SpoofMeta(type):
        def __eq__(cls, _other):
            calls.append("type equality")
            return True

        __hash__ = type.__hash__

    class Spoof(metaclass=SpoofMeta):
        def __eq__(self, _other):
            calls.append("value equality")
            return True

    with prepared_image_case(tmp_path) as (_path, _parent, _branch, _owner, plan, token, payload):
        image = plan.seal_emission_image(token, **payload)
        with pytest.raises(CompilationError, match="primitive"):
            image.consume((Spoof(),))
    assert calls == []


@pytest.mark.parametrize("count", [1, 7, 64])
def test_owned_batch_global_admission_checks_do_not_scale_with_units(tmp_path, monkeypatch, count, record_property):
    from compiler.scopes.storage_admission import StorageAdmission

    body = "\n".join(
        f"if(escape)then\ndo i=-2,n-3\nb(i)=a(i)+{index + 1}.0_8\nenddo\nendif"
        for index in range(count)
    )
    source = SOURCE[:SOURCE.index("visits=visits+1")] + body + "\nend subroutine\nend module\n"
    _path, parent = source_builder(tmp_path, source=source)
    original = StorageAdmission._check
    checks, admissions, preparations = [], [], []

    def counted(self):
        checks.append(self)
        preparations.append(self._image_preparation)
        if not admissions or admissions[-1] is not self:
            admissions.append(self)
        return original(self)

    monkeypatch.setattr(StorageAdmission, "_check", counted)
    with monkeypatch.context() as patch:
        patch.setattr(LexicalOwner, "supports_owned_emission", lambda _self, _plan: False)
        with GenerationTransaction(parent) as branch:
            old_owner, _outputs, _report = emit_branch(branch.builder)
            assert sum(len(member.units) for member in old_owner.execution_plan.members) == count
    old_checks = len(checks)
    assert old_checks > 5
    assert all(token is None for token in preparations)
    checks.clear()
    admissions.clear()
    preparations.clear()
    with GenerationTransaction(parent) as branch:
        owner, outputs, report = emit_branch(branch.builder)
        assert sum(len(member.units) for member in owner.execution_plan.members) == count
        assert len(admissions) == 1
        assert all(token is not None for token in preparations)
        # Entry once, seal before/after callbacks, exit before/after callbacks.
        assert len(checks) == 5
        assert report["scope_count"] == 1
        assert outputs["scope-manifest.json"]
        record_property("original_unit_count", count)
        record_property("legacy_global_admission_checks", old_checks)
        record_property("owned_global_admission_checks", len(checks))


def member_fixture(directory, kind):
    from compiler.tests.test_hierarchical_child_regions import fixture
    from compiler.tests.test_inherited_lexical_namespace import source_fixture
    from compiler.tests.test_lexical_source_owner import MODULE_SOURCE

    if kind == "module":
        return source_builder(directory, source=MODULE_SOURCE)
    return fixture(directory, source_fixture(native_call=kind == "native_write"))


def emit_member_branch(builder, kind):
    from compiler.tests.test_hierarchical_child_regions import owner_at

    if kind == "module":
        return emit_branch(builder)
    owner = owner_at(builder)
    nodes = builder.inline.prepare(owner.closed_selection)
    builder.generated.update(builder.inline.generated)
    builder.numerical_ir.update(builder.inline.ir)
    builder.numerical_reasons.update({name: None for name in builder.inline.regions})
    assert owner.run(nodes)
    outputs, report = builder.finish(materialize_native_effects=False)
    return owner, outputs, report


def check_member_package(directory, owner, outputs, report):
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
    plan = owner.execution_plan
    assert len(plan.members) > 1
    assert all(not member.device_tasks and not member.source_device_tasks for member in plan.members)
    return manifest


@pytest.mark.parametrize("kind", ["readonly", "native_write", "module"])
def test_real_hierarchical_and_companion_images_preserve_all_artifacts(tmp_path, monkeypatch, kind):
    from compiler.frontend import source_effects
    from compiler.frontend.summary_cache import SummaryCache
    from compiler.scopes.storage_admission import active_admission

    monkeypatch.setattr(source_effects, "_DEFAULT_SUMMARY_CACHE", SummaryCache())
    path, parent = member_fixture(tmp_path, kind)
    original, facts = path.read_bytes(), deepcopy(parent.facts)
    with monkeypatch.context() as patch:
        patch.setattr(LexicalOwner, "supports_owned_emission", lambda _self, _plan: False)
        patch.setattr(LexicalOwner, "supports_owned_unit_emission", lambda _self, _unit: False)
        for _ in range(2):
            with GenerationTransaction(parent) as branch:
                emit_member_branch(branch.builder, kind)
        with GenerationTransaction(parent) as branch:
            owner, old_outputs, old_report = emit_member_branch(branch.builder, kind)
            old_manifest = check_member_package(tmp_path / "legacy", owner, old_outputs, old_report)

    published, sealed = [], []
    original_publish = LexicalOwner._publish_owned_emission
    original_seal = LexicalOwner._seal_owned_emission

    def seal(self, plan, preparation):
        image = original_seal(self, plan, preparation)
        assert preparation.sealed
        sealed.append(preparation)
        return image

    def publish(targets, image, tickets):
        assert active_admission() is None, "all callback-bearing exit checks must precede publication"
        assert len(sealed) == 1
        assert sealed[0].sealed
        published.append((targets, tickets))
        return original_publish(targets, image, tickets)

    monkeypatch.setattr(LexicalOwner, "_seal_owned_emission", seal)
    monkeypatch.setattr(LexicalOwner, "_publish_owned_emission", staticmethod(publish))
    with GenerationTransaction(parent) as branch:
        owner, new_outputs, new_report = emit_member_branch(branch.builder, kind)
        new_manifest = check_member_package(tmp_path / "owned", owner, new_outputs, new_report)
        assert len(published) == 1, "the authentic member owner did not use owned emission"
        targets, tickets = published[0]
        assert type(targets) is tuple
        assert len(targets) > 1
        assert all(target.outputs is branch.builder.outputs for target in targets)
        assert all(target.edits is branch.builder.edits for target in targets)
        assert all(target.scopes is branch.builder.scopes for target in targets)
        plan = owner.execution_plan
        assert len(tickets) == sum(len(plan.units(member)) for member in plan.members)
        for ticket in tickets:
            member = plan.members[ticket[2]]
            assert ticket[3] == member.routine.qualified
            assert ticket[5] == owner.graph_unit_ids[ticket[3]][str(ticket[4])]
            for local, canonical, slot in ticket[6]:
                binding, actual_slot = member.resources[local]
                assert canonical == member.canonical(binding).root
                assert slot == actual_slot
        scope, = new_manifest["scopes"]
        if kind == "module":
            assert {row["procedure"] for row in scope["module_coordinators"]} == {"stage_child::adjust"}
        else:
            assert scope["selected_end_retirement"]["context"] == scope["owner"]
            assert scope["checked_execution_plan"]["storage_epoch"]["return_contract"]["version"] == 4
            coordinator, = scope["internal_coordinators"]
            assert coordinator["lexical_namespace"]["namespace_owner"] == "hierarchical::middle"
        assert new_outputs == old_outputs
        assert new_manifest == old_manifest
    assert path.read_bytes() == original
    assert parent.facts == facts
    assert not parent.outputs
    assert not parent.edits
    assert not parent.scopes


@pytest.mark.parametrize("kind", ["readonly", "module"])
def test_hierarchical_publication_does_not_revisit_live_member_authority(tmp_path, monkeypatch, kind):
    from compiler.scopes import source_device_tasks
    from compiler.scopes.execution_plan import CheckedExecutionPlan
    from compiler.scopes.storage_epoch import StorageEpochProof

    _path, parent = member_fixture(tmp_path, kind)
    original_publish = LexicalOwner._publish_owned_emission
    publications = []

    def forbidden(*_args, **_kwargs):
        pytest.fail("sealed member publication consulted live source or provider authority")

    def publish(targets, image, tickets):
        assert type(targets) is tuple
        publications.append(image.identity)
        with monkeypatch.context() as patch:
            patch.setattr(CheckedExecutionPlan, "validate", forbidden)
            patch.setattr(StorageEpochProof, "validate", forbidden)
            patch.setattr(ScopeBuilder, "capture_fact", forbidden)
            patch.setattr(ScopeBuilder, "entry_artifacts", forbidden)
            patch.setattr(LexicalOwner, "canonical", forbidden)
            patch.setattr(LexicalOwner, "prepare_unit", forbidden)
            patch.setattr(source_device_tasks, "_provider", forbidden)
            for target in targets:
                patch.setattr(type(target.analysis), "verify_publication", forbidden)
                for binding in target.entry.scope.bindings.values():
                    patch.setattr(type(binding), "public", forbidden)
                for proof in target.analysis._numerical_preparations.values():
                    patch.setattr(type(proof), "validate", forbidden)
            original_publish(targets, image, tickets)

    monkeypatch.setattr(LexicalOwner, "_publish_owned_emission", staticmethod(publish))
    with GenerationTransaction(parent) as branch:
        owner, outputs, report = emit_member_branch(branch.builder, kind)
        check_member_package(tmp_path / "package", owner, outputs, report)
    assert len(publications) == 1
    assert not parent.outputs
    assert not parent.edits
    assert not parent.scopes


@pytest.mark.parametrize("kind", ["readonly", "module"])
def test_member_image_survives_normal_transaction_adoption(tmp_path, monkeypatch, kind):
    path, parent = member_fixture(tmp_path, kind)
    source, facts = path.read_bytes(), deepcopy(parent.facts)
    publications = []
    original_publish = LexicalOwner._publish_owned_emission

    def publish(targets, image, tickets):
        publications.append(image.identity)
        return original_publish(targets, image, tickets)

    monkeypatch.setattr(LexicalOwner, "_publish_owned_emission", staticmethod(publish))
    with GenerationTransaction(parent) as branch:
        owner, expected_outputs, report = emit_member_branch(branch.builder, kind)
        expected_manifest = check_member_package(tmp_path / "package", owner, expected_outputs, report)
        branch.adopt()
    assert len(publications) == 1
    outputs = parent.outputs
    assert outputs == expected_outputs
    assert json.loads(outputs["scope-manifest.json"]) == expected_manifest
    assert path.read_bytes() == source
    assert parent.facts == facts


@pytest.mark.parametrize("kind", ["readonly", "module"])
@pytest.mark.parametrize("failure", ["prepare", "publish"])
def test_member_image_failure_leaves_no_adopted_artifact(tmp_path, monkeypatch, kind, failure):
    from compiler.scopes.storage_admission import active_admission

    path, parent = member_fixture(tmp_path, kind)
    source, facts = path.read_bytes(), deepcopy(parent.facts)
    reached = []
    if failure == "prepare":
        original = LexicalOwner.prepare_unit

        def prepare(self, unit, index):
            image = original(self, unit, index)
            if self is not self.owner:
                reached.append(image)
                raise CompilationError("injected child preparation failure")
            return image

        monkeypatch.setattr(LexicalOwner, "prepare_unit", prepare)
    else:
        original = LexicalOwner._publish_owned_emission

        def publish(targets, image, tickets):
            original(targets, image, tickets)
            assert type(targets) is tuple
            assert all(target.outputs and target.edits and target.scopes for target in targets)
            reached.append(image)
            raise CompilationError("injected companion publication failure")

        monkeypatch.setattr(LexicalOwner, "_publish_owned_emission", staticmethod(publish))

    def attempt():
        with GenerationTransaction(parent) as branch:
            emit_member_branch(branch.builder, kind)
            branch.adopt()

    with pytest.raises(CompilationError, match="injected"):
        attempt()
    assert reached
    assert active_admission() is None
    assert path.read_bytes() == source
    assert parent.facts == facts
    assert not parent.outputs
    assert not parent.edits
    assert not parent.scopes
    assert not parent.resolved_calls


@pytest.mark.parametrize("stage", ["source_callback", "transaction_commit"])
@pytest.mark.parametrize("channel", ["artifact", "edit", "scope"])
def test_late_validation_cannot_adopt_changes_to_sealed_publication(tmp_path, monkeypatch, stage, channel):
    from compiler.frontend.source_effects import SourceEffects

    _path, parent = member_fixture(tmp_path, "readonly")
    published, changed = [], []
    original_publish = LexicalOwner._publish_owned_emission
    original_verify = SourceEffects.verify_publication
    original_regions = ScopeBuilder._validate_numerical_regions
    committing = False

    def publish(targets, image, tickets):
        original_publish(targets, image, tickets)
        assert type(targets) is tuple
        target = targets[0]
        if channel == "artifact":
            key = next(iter(target.outputs))
            expected = target.outputs[key]
        elif channel == "edit":
            key = next(iter(target.edits))
            expected = target.edits[key][0]
        else:
            key = "owner"
            expected = target.scopes[0][key]
        published.append((target, key, expected))

    def tamper():
        if published and not changed:
            target, key, _expected = published[0]
            if channel == "artifact":
                target.outputs[key] += "\n! changed after owned image publication\n"
            elif channel == "edit":
                first, last, text = target.edits[key][0]
                target.edits[key][0] = (first, last, text + "\n! changed after publication\n")
            else:
                target.scopes[0][key] += "_changed_after_publication"
            changed.append(key)

    def verify(self):
        result = original_verify(self)
        if stage == "source_callback":
            tamper()
        return result

    def regions(self):
        result = original_regions(self)
        if stage == "transaction_commit" and committing:
            tamper()
        return result

    monkeypatch.setattr(LexicalOwner, "_publish_owned_emission", staticmethod(publish))
    monkeypatch.setattr(SourceEffects, "verify_publication", verify)
    monkeypatch.setattr(ScopeBuilder, "_validate_numerical_regions", regions)
    rejected = False
    try:
        with GenerationTransaction(parent) as branch:
            emit_member_branch(branch.builder, "readonly")
            branch.adopt()
            committing = True
    except CompilationError:
        rejected = True
    assert len(published) == 1
    assert len(changed) == 1, "the intended late validator callback was not reached"
    if rejected:
        assert not parent.outputs
        assert not parent.edits
        assert not parent.scopes
    else:
        _target, key, expected = published[0]
        if channel == "artifact":
            actual = parent.outputs[key]
        elif channel == "edit":
            actual = parent.edits[key][0]
        else:
            actual = parent.scopes[0][key]
        assert actual == expected, "adoption accepted output outside the sealed image"


def test_final_package_callback_cannot_publish_changed_owned_provider(tmp_path, monkeypatch):
    import sys

    from compiler.frontend.source_effects import SourceEffects

    _path, parent = member_fixture(tmp_path, "readonly")
    with GenerationTransaction(parent) as branch:
        owner, outputs, report = emit_member_branch(branch.builder, "readonly")
        check_member_package(tmp_path / "before", owner, outputs, report)
        branch.adopt()
    name = next(name for name in parent.outputs if name.endswith(".cu") and "runtime" not in name)
    expected = parent.outputs[name]
    finish_code = ScopeBuilder.finish.__code__
    original = SourceEffects.verify_publication
    direct_calls = []

    def verify(self):
        result = original(self)
        if sys._getframe(1).f_code is finish_code:
            direct_calls.append(self)
            # The final package callback follows report construction. Earlier
            # plan/receipt checks retain their ordinary implementations.
            if len(direct_calls) == 2:
                parent.outputs[name] += "\n! late final-package mutation\n"
        return result

    monkeypatch.setattr(SourceEffects, "verify_publication", verify)
    try:
        finished, _report = parent.finish(materialize_native_effects=False)
    except CompilationError:
        pass  # No completed package was returned by the failing publisher.
    else:
        assert finished[name] == expected
    assert len(direct_calls) == 2


@pytest.mark.parametrize("artifact", ["unexpected.cu", "sources/foreign.f90"])
def test_first_finish_callback_cannot_add_unsealed_code(tmp_path, monkeypatch, artifact):
    import sys

    from compiler.frontend.source_effects import SourceEffects

    _path, parent = member_fixture(tmp_path, "readonly")
    candidate = None
    finish_code = ScopeBuilder.finish.__code__
    original = SourceEffects.verify_publication
    direct_calls = []

    def verify(self):
        result = original(self)
        if sys._getframe(1).f_code is finish_code:
            direct_calls.append(self)
            if len(direct_calls) == 1:
                assert candidate._owned_emission_prepared_channels
                assert artifact not in candidate.outputs
                candidate.outputs[artifact] = "! code absent from the sealed image\n"
        return result

    def attempt():
        nonlocal candidate
        with GenerationTransaction(parent) as branch:
            candidate = branch.builder
            emit_member_branch(candidate, "readonly")
            branch.adopt()

    monkeypatch.setattr(SourceEffects, "verify_publication", verify)
    with pytest.raises(CompilationError, match="unsealed generated artifact"):
        attempt()
    assert len(direct_calls) == 1
    assert not parent.outputs
    assert not parent.edits
    assert not parent.scopes


def test_final_report_serialization_callback_cannot_change_owned_code(tmp_path, monkeypatch):
    from compiler.frontend.source_effects import SourceEffects

    _path, parent = member_fixture(tmp_path, "readonly")
    with GenerationTransaction(parent) as branch:
        owner, outputs, report = emit_member_branch(branch.builder, "readonly")
        check_member_package(tmp_path / "before", owner, outputs, report)
        branch.adopt()
    name = next(name for name in parent.outputs if name.endswith(".cu") and "runtime" not in name)
    original_report = SourceEffects.report
    callbacks = []

    class MutatingReport(dict):
        def items(self):
            callbacks.append(name)
            parent.outputs[name] += "\n! changed by final JSON serialization\n"
            return super().items()

    def report(self, *args, **kwargs):
        return MutatingReport(original_report(self, *args, **kwargs))

    monkeypatch.setattr(SourceEffects, "report", report)
    with pytest.raises(CompilationError, match="output changed during final report publication"):
        parent.finish(materialize_native_effects=False)
    assert callbacks == [name]


def test_late_adoption_copy_failure_does_not_commit_summary_cache(tmp_path, monkeypatch):
    from compiler.frontend import source_effects
    from compiler.frontend.summary_cache import SummaryCache
    from compiler.scopes.generation_transaction import _GenerationCopy

    monkeypatch.setattr(source_effects, "_DEFAULT_SUMMARY_CACHE", SummaryCache())
    _path, parent = member_fixture(tmp_path, "readonly")
    cache = parent.analysis._summary_cache
    memory, stats = deepcopy(cache._memory), cache.stats
    original_copy = _GenerationCopy.__call__
    candidate, artifact = None, None
    armed, changed = False, []

    def copy_value(self, value):
        result = original_copy(self, value)
        if armed and not changed and value is candidate.outputs:
            assert result is not value
            result[artifact] += "\n! changed only in the final adoption copy\n"
            changed.append(artifact)
        return result

    def attempt():
        nonlocal candidate, artifact, armed
        with GenerationTransaction(parent) as branch:
            candidate = branch.builder
            owner, outputs, report = emit_member_branch(candidate, "readonly")
            check_member_package(tmp_path / "before", owner, outputs, report)
            private = candidate.analysis._summary_cache
            assert private is not cache
            assert private._memory != memory
            assert private.stats != stats
            artifact = next(name for name in outputs if name.endswith(".cu") and "runtime" not in name)
            branch.adopt()
            armed = True

    monkeypatch.setattr(_GenerationCopy, "__call__", copy_value)
    with pytest.raises(CompilationError, match="copying atomic adoption"):
        attempt()
    assert changed == [artifact]
    assert parent.analysis._summary_cache is cache
    assert cache._memory == memory
    assert cache.stats == stats
    assert not parent.outputs
    assert not parent.edits
    assert not parent.scopes


@pytest.mark.parametrize("count", [1, 7])
def test_global_admission_checks_remain_constant_across_real_children(tmp_path, monkeypatch, count, record_property):
    from compiler.scopes.storage_admission import StorageAdmission
    from compiler.tests.test_hierarchical_child_regions import fixture
    from compiler.tests.test_inherited_lexical_namespace import source_fixture

    source = source_fixture(native_call=False)
    start = source.index("subroutine leaf(n)")
    leaf = source[start:source.index("end subroutine", start) + len("end subroutine")]
    children = "\n".join(leaf.replace("subroutine leaf(n)", f"subroutine leaf_{index}(n)")
                         for index in range(count))
    source = source.replace(leaf, children).replace(
        "call leaf(n)", "\n".join(f"call leaf_{index}(n)" for index in range(count)))
    path, parent = fixture(tmp_path, source)
    original = StorageAdmission._check
    checks, tokens = [], []

    def counted(self):
        checks.append(self)
        tokens.append(self._image_preparation)
        return original(self)

    monkeypatch.setattr(StorageAdmission, "_check", counted)
    with monkeypatch.context() as patch:
        patch.setattr(LexicalOwner, "supports_owned_emission", lambda _self, _plan: False)
        with GenerationTransaction(parent) as branch:
            emit_member_branch(branch.builder, "readonly")
    old_count = len(checks)
    assert old_count > 5
    assert all(token is None for token in tokens)
    checks.clear()
    tokens.clear()
    with GenerationTransaction(parent) as branch:
        owner, outputs, report = emit_member_branch(branch.builder, "readonly")
        check_member_package(tmp_path / "package", owner, outputs, report)
        assert {member.routine.qualified for member in owner.execution_plan.members
                if member.routine.qualified.startswith("hierarchical::leaf_")} == {
                    f"hierarchical::leaf_{index}" for index in range(count)}
        assert all(token is not None for token in tokens)
        assert len(checks) == 5
        record_property("original_child_count", count)
        record_property("legacy_global_admission_checks", old_count)
        record_property("owned_global_admission_checks", len(checks))
    assert path.read_text() == source
