"""Original invocation facts survive only their exact independently proved prefix."""
import json
from copy import copy

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.supported_paths import requested_supported_path
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.scopes.supported_paths import OwnerLocalSupportedPath
from compiler.tests.test_supported_path_scope import SOURCE, fixture


def incoming_fixture(tmp_path, *, initialized="whole", lifetime="invocation", private_whole=False, block_scope=False):
    source = SOURCE.replace("subroutine produce(a,b,n)", "subroutine produce(x,a,b,n)\nreal(8),intent(in)::x(:)")
    source = source.replace("a(i)=real(i,8)", "a(i)=x(i)")
    source = source.replace("subroutine run(n,mode)", "subroutine run(x,n,mode)\nreal(8),intent(in)::x(:)")
    source = source.replace("call produce(a,b,n)", "call produce(x,a,b,n)")
    # An explicit excluded route need not contain an unknown escape. Its
    # independent shared-input footprint must itself be source-proved.
    source = source.replace("call unknown_private_escape(a)", "a=a+b")
    if block_scope:
        source = source.replace("end module", "subroutine unrelated()\nblock\ninteger::local_value\nlocal_value=1\nend block\nend subroutine\nend module")
    path, facts, contracts = fixture(tmp_path, source)
    facts["captures"]["argument::x"] = {"storage": "stable", "initialized": initialized,
        "escapes": False, "allocation_changes": False, "lifetime": lifetime}
    if private_whole:
        for name in ("a", "b"):
            facts["captures"]["supported_scope::run::" + name].update(initialized="whole", lifetime="invocation")
    builder = ScopeBuilder([path], "supported_scope::run", facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections"), contracts=contracts)
    return path, builder


def owner_fixture(tmp_path, **options):
    path, builder = incoming_fixture(tmp_path, **options)
    selection = builder._native_ancestor_selection.public()
    original, gate = requested_supported_path(builder.analysis, builder.entry, selection["path_request"],
        tuple(item["procedure"] for item in selection["device_roots"]))
    return path, builder, OwnerLocalSupportedPath(builder, original, gate)


def emit_incoming_supported_path_fixture(tmp_path):
    """An actual producer sample with explicit independent invocation-WHOLE premise."""
    path, builder = incoming_fixture(tmp_path)
    outputs, report = builder.run()
    (tmp_path / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (tmp_path / "facts.json").write_text(json.dumps(builder.facts, indent=2) + "\n")
    for name, text in outputs.items():
        target = tmp_path / "generated" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return path, builder, outputs, report


def test_independent_incoming_read_uses_position1_entry4_prefix3_and_private_none(tmp_path):
    _path, builder, owner = owner_fixture(tmp_path, private_whole=True)
    epoch = owner.position_epoch
    public = epoch.public()
    assert public["version"] == 3
    assert public["gate_metadata_edges"] == []
    assert public["reached_position"]["version"] == 1
    assert public["entry_preservation"]["version"] == 4
    assert public["entry_preservation"]["prefix_projection"]["version"] == 3
    assert public["entry_preservation"]["preserved_resources"] == ["argument::x"]
    facts = dict(zip((binding.root for binding in epoch._bindings),
                     epoch.capture_many(builder, epoch._bindings), strict=True))
    assert facts["argument::x"]["initialized"] == "whole"
    assert facts["supported_scope::run::a"]["initialized"] == "none"
    assert facts["supported_scope::run::b"]["initialized"] == "none"
    assert all(item["lifetime"] == "selected_window" for item in facts.values())
    assert all("sections" not in item and "source_sections" not in item for item in facts.values())


@pytest.mark.parametrize(("initialized", "lifetime"), [("none", "invocation"), ("whole", "selected_window")])
def test_intent_in_or_selected_window_whole_never_infers_invocation_definitions(tmp_path, initialized, lifetime):
    _path, builder, owner = owner_fixture(tmp_path, initialized=initialized, lifetime=lifetime)
    epoch = owner.position_epoch
    assert epoch.public()["version"] == 1
    assert "entry_preservation" not in epoch.public()
    assert all(fact["initialized"] == "none" for fact in epoch.capture_many(builder, epoch._bindings))


def test_real_emitted_epoch3_uses_the_same_source_proof_in_scope_and_checked_plan(tmp_path):
    _path, _builder, _outputs, report = emit_incoming_supported_path_fixture(tmp_path)
    assert report["scope_count"] == 1, report["boundaries"]
    scope, = report["scopes"]
    epoch = scope["supported_path_epoch"]
    assert epoch == scope["checked_execution_plan"]["storage_epoch"]["supported_path_epoch"]
    assert epoch["version"] == 3
    assert epoch["entry_preservation"]["preserved_resources"] == ["argument::x"]


@pytest.mark.parametrize("mutation", ["fact", "binding", "entry", "position", "descriptor", "registry", "owner_marker", "scope", "config", "selector", "members"])
def test_synchronous_capture_batch_does_not_reseed_prior_source_or_fact_authority(tmp_path, mutation, monkeypatch):
    _path, builder, owner = owner_fixture(tmp_path)
    epoch = owner.position_epoch
    capture_fact = builder.capture_fact
    changed = False

    def inspect(binding):
        nonlocal changed
        result = capture_fact(binding)
        if changed:
            return result
        changed = True
        if mutation == "fact":
            builder.facts["captures"]["argument::x"]["initialized"] = "none"
        elif mutation == "binding":
            epoch._bindings[-1].kind = 4
        elif mutation == "entry":
            object.__setattr__(epoch, "_entry_preservation", copy(epoch._entry_preservation))
        elif mutation == "position":
            object.__setattr__(epoch._entry_preservation, "_position", copy(epoch._reached_position))
        elif mutation == "descriptor":
            object.__setattr__(epoch, "_descriptor", copy(epoch._descriptor))
        elif mutation == "registry":
            builder.analysis._entry_preservations[epoch._entry_preservation.identity] = copy(epoch._entry_preservation)
        elif mutation == "owner_marker":
            builder.analysis._entry_preservations_owner = object()
        elif mutation == "scope":
            scope = builder.entry.scope
            scope.bindings["called"] = copy(scope.bindings["called"])
        elif mutation == "config":
            object.__setattr__(builder.config, "policy", "auto")
        elif mutation == "selector":
            builder._native_ancestor_selection = copy(builder._native_ancestor_selection)
        else:
            owner.members = dict(owner.members)
        return result

    monkeypatch.setattr(builder, "capture_fact", inspect)
    with pytest.raises(CompilationError):
        epoch.capture_many(builder, epoch._bindings)


def test_capture_batch_preserves_declaration_state_agreement_across_all_edges(tmp_path):
    _path, builder, owner = owner_fixture(tmp_path)
    epoch = owner.position_epoch
    _node, component = epoch._descriptor._components[-1]
    public = component.public

    def different_state():
        value = public()
        value["resources"][0]["stable"] = False
        return value

    # Substitute only the inspection service; the underlying registered
    # declaration proof still validates, so the agreement fence is exercised.
    object.__setattr__(component, "public", different_state)
    with pytest.raises(CompilationError, match="descriptor state"):
        epoch.capture_many(builder, epoch._bindings)


def test_final_capture_checkpoint_keeps_original_registry_owner_marker(tmp_path):
    _path, builder, owner = owner_fixture(tmp_path)
    epoch = owner.position_epoch
    calls = 0

    def inspect():
        nonlocal calls
        calls += 1
        if calls == 2 * len(epoch._bindings) + 2:
            builder.analysis._entry_preservations_owner = object()

    with pytest.raises(CompilationError):
        epoch.capture_many(builder, epoch._bindings, _consumer_check=inspect)


@pytest.mark.parametrize("mutation", ["binding", "scope_entry", "scope_registry"])
def test_final_capture_checkpoint_keeps_original_block_local_declaration(tmp_path, mutation):
    _path, builder, owner = owner_fixture(tmp_path, block_scope=True)
    epoch = owner.position_epoch
    binding, = (binding for scope in builder.analysis._block_scopes.values()
                for binding in scope.bindings.values() if binding.name == "local_value")
    facts = epoch.capture_many(builder, epoch._bindings)
    assert any(fact["initialized"] == "whole" for fact in facts)
    calls = 0

    def inspect():
        nonlocal calls
        calls += 1
        if calls == 2 * len(epoch._bindings) + 2:
            if mutation == "binding":
                object.__setattr__(binding, "kind", 8)
            elif mutation == "scope_entry":
                original, scope = next(iter(builder.analysis._block_scopes.items()))
                dict.__setitem__(builder.analysis._block_scopes, original, copy(scope))
            else:
                original = builder.analysis._block_scopes
                replacement = dict(original)
                with pytest.raises(CompilationError, match="original source registry is immutable"):
                    builder.analysis._block_scopes = replacement
                assert builder.analysis._block_scopes is original
                object.__setattr__(builder.analysis, "_block_scopes", replacement)

    with pytest.raises(CompilationError, match="attachment changed"):
        epoch.capture_many(builder, epoch._bindings, _consumer_check=inspect)


def test_first_full_validation_cannot_replace_an_epoch_attachment_before_baselining(tmp_path, monkeypatch):
    from compiler.scopes.supported_paths import SupportedPathStorageEpoch

    _path, builder, owner = owner_fixture(tmp_path)
    epoch = owner.position_epoch
    validate = SupportedPathStorageEpoch.validate
    changed = False

    def inspect(self, builder=None):
        nonlocal changed
        value = validate(self, builder)
        if self is epoch and not changed:
            changed = True
            object.__setattr__(epoch, "_descriptor", copy(epoch._descriptor))
        return value

    monkeypatch.setattr(SupportedPathStorageEpoch, "validate", inspect)
    with pytest.raises(CompilationError):
        epoch.capture_many(builder, epoch._bindings)


@pytest.mark.parametrize("mutation", ["entry", "position", "missing"])
def test_epoch_rejects_untyped_or_incomplete_entry_attachments_before_public_projection(tmp_path, mutation):
    _path, _builder, owner = owner_fixture(tmp_path)
    epoch = owner.position_epoch
    field, value = ("_entry_preservation", object()) if mutation == "entry" else (
        "_reached_position", None if mutation == "missing" else object())
    object.__setattr__(epoch, field, value)
    with pytest.raises(CompilationError, match="typed|entry proof"):
        epoch.validate()


@pytest.mark.parametrize("consumer", ["validate", "capture"])
@pytest.mark.parametrize("mutation", ["epoch_entry", "entry_analysis", "entry_record", "registry", "projection_gate", "retained_local", "scalar_binding"])
def test_late_entry_validator_cannot_replace_the_consumed_epoch_attachment(tmp_path, monkeypatch, consumer, mutation):
    from compiler.frontend.entry_preservation import PreservedEntryDefinitions

    _path, builder, owner = owner_fixture(tmp_path)
    epoch = owner.position_epoch
    entry = epoch._entry_preservation
    validate = PreservedEntryDefinitions.validate

    def inspect(self, *args, **kwargs):
        result = validate(self, *args, **kwargs)
        if self is entry:
            if mutation == "epoch_entry":
                object.__setattr__(epoch, "_entry_preservation", copy(entry))
            elif mutation == "entry_analysis":
                object.__setattr__(entry, "_analysis", copy(builder.analysis))
            elif mutation == "entry_record":
                object.__setattr__(entry, "_record", "{}")
            elif mutation == "registry":
                builder.analysis._entry_preservations[entry.identity] = copy(entry)
            elif mutation == "projection_gate":
                object.__setattr__(entry._projection, "_gate", copy(epoch._gate))
            elif mutation == "retained_local":
                proof = next(proof for proof, name in entry._projection._retained if name == "_local_allocation_proofs")
                object.__setattr__(proof, "_record", "{}")
            else:
                object.__setattr__(builder.entry.scope.bindings["eligible"], "kind", 8)
        return result

    monkeypatch.setattr(PreservedEntryDefinitions, "validate", inspect)
    if consumer == "validate":
        operation = epoch.validate
    else:
        binding = next(binding for binding in epoch._bindings if binding.root == "argument::x")
        fact = builder.capture_fact(binding)

        def operation():
            return epoch.capture(builder, binding, fact)

    with pytest.raises(CompilationError, match="attachment changed"):
        operation()


def test_late_descriptor_callback_cannot_install_an_untyped_entry_before_public(tmp_path, monkeypatch):
    from compiler.frontend.supported_path_storage import SupportedPathDescriptorComposition

    _path, _builder, owner = owner_fixture(tmp_path)
    epoch = owner.position_epoch
    validate = SupportedPathDescriptorComposition.validate

    def inspect(self, *args, **kwargs):
        result = validate(self, *args, **kwargs)
        if self is epoch._descriptor:
            object.__setattr__(epoch, "_entry_preservation", object())
        return result

    monkeypatch.setattr(SupportedPathDescriptorComposition, "validate", inspect)
    with pytest.raises(CompilationError, match="attachment changed"):
        epoch.validate()
