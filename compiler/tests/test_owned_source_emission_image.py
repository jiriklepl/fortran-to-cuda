"""Source teams finish proof/provider work before detached graph publication."""
import gc
import json
from copy import deepcopy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend import source_effects
from compiler.frontend.summary_cache import SummaryCache
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.lexical import LexicalOwner
from compiler.scopes.source import ScopeBuilder
from compiler.scopes.storage_admission import active_admission
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_owned_emission_image import emit_branch
from compiler.tests.test_source_native_pricing import live_profile
from compiler.tests.test_tile_private import SOURCE


def builder(directory, policy, transfers, *, mixed=True):
    source = SOURCE.rsplit("end module", 1)[0] + """subroutine seed(a,b,n)
integer,intent(in)::n
real(8),intent(in)::a(-3:)
real(8),intent(inout)::b(-3:)
integer::i
do i=1,n
 b(i)=b(i)+a(i)*0.25_8
enddo
end subroutine
subroutine outer(a,b,n,t,twice)
integer,intent(in)::n,t
logical,intent(in)::twice
real(8),intent(in)::a(-3:)
real(8),intent(inout)::b(-3:)
call seed(a,b,n)
if(twice)then
 call advance(a,b,n,t)
endif
call advance(a,b,n,t)
end subroutine
end module
"""
    if not mixed:
        source = source.replace("call seed(a,b,n)\n", "")
    path = directory / "source.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: deepcopy(FACT) for name in ("a", "b")}}
    return path, ScopeBuilder([path], "stencil::outer", facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy=policy, scope_execution="graph", scope_transfers=transfers,
                             profile=live_profile() if policy == "auto" else None))


def check_package(directory, outputs, report):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build

    scope, = report["scopes"]
    task, = scope["source_device_tasks"]
    proof = task.get("source_proof", task)
    assert proof["kind"] == "original_tile_private_team"
    assert [item["name"] for item in proof["scratch"]] == ["tmp"]
    assert "tmp" not in {row["descriptor"]["name"] for row in scope["resource_bindings"]["resources"]}
    child, = [item for item in scope["module_coordinators"] if item["procedure"] == "stencil::advance"]
    assert len(child["calls"]) == 2
    validate_checked_execution_plans(report)
    for name, content in outputs.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    manifest = json.loads(outputs["scope-manifest.json"])
    _scope_package_build({"artifact_root": str(directory), "scope_manifest": manifest})


@pytest.mark.parametrize(("policy", "transfers", "mixed"), [
    ("sections", "direct", True), ("auto", "direct", True), ("auto", "auto", True),
    ("sections", "direct", False), ("auto", "direct", False)])
def test_owned_source_team_keeps_all_contextual_artifacts_and_borrowed_calls(tmp_path, monkeypatch, policy, transfers, mixed):
    from compiler.frontend.source_device_tasks import SourceDeviceTaskProof
    from compiler.offload.source_task_pricing import SourceTaskPricing
    from compiler.scopes import source_device_tasks

    monkeypatch.setattr(source_effects, "_DEFAULT_SUMMARY_CACHE", SummaryCache())
    path, parent = builder(tmp_path, policy, transfers, mixed=mixed)
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
            check_package(tmp_path / "contextual", expected, report)

    published = []
    publish = LexicalOwner._publish_owned_emission

    def forbidden(*_args, **_kwargs):
        pytest.fail("detached source publication consulted live proof/provider state")

    def detached(targets, image, tickets):
        assert active_admission() is None
        published.append(image.identity)
        with monkeypatch.context() as patch:
            patch.setattr(source_device_tasks, "_provider", forbidden)
            patch.setattr(SourceDeviceTaskProof, "validate", forbidden)
            patch.setattr(SourceTaskPricing, "validate", forbidden)
            return publish(targets, image, tickets)

    monkeypatch.setattr(LexicalOwner, "_publish_owned_emission", staticmethod(detached))
    gc.collect()
    with GenerationTransaction(parent) as branch:
        owner, actual, report = emit_branch(branch.builder)
        assert len(published) == 1
        assert actual == expected
        assert all(member.unit_emission(unit).prepared is not None
                   for member in owner.execution_plan.members for unit in owner.execution_plan.units(member))
        check_package(tmp_path / "owned", actual, report)
    assert path.read_bytes() == original
    assert not parent.outputs
    assert not parent.edits
    assert not parent.scopes


@pytest.mark.parametrize("fault", ["source_proof", "scratch_binding", "pricing"])
def test_source_callback_corruption_never_publishes_or_adopts_an_image(tmp_path, monkeypatch, fault):
    from compiler.scopes.source_device_tasks import SourceDevicePatch

    path, parent = builder(tmp_path, "auto", "auto")
    original = path.read_bytes()
    render = SourceDevicePatch.render
    changed = []

    def corrupt(task, member):
        result = render(task, member)
        if not changed:
            changed.append(fault)
            if fault == "source_proof":
                object.__setattr__(task.proof, "identity", "0" * 64)
            elif fault == "scratch_binding":
                object.__setattr__(task.proof.team.scratch_bindings[0], "kind", 4)
            else:
                object.__setattr__(task.pricing, "identity", "0" * 64)
        return result

    monkeypatch.setattr(SourceDevicePatch, "render", corrupt)

    def publish(*_args):
        pytest.fail("corrupted source state reached detached publication")

    monkeypatch.setattr(LexicalOwner, "_publish_owned_emission", staticmethod(publish))

    def generate():
        with GenerationTransaction(parent) as branch:
            emit_branch(branch.builder)
            branch.adopt()

    with pytest.raises(CompilationError):
        generate()
    assert changed == [fault]
    assert path.read_bytes() == original
    assert not parent.outputs
    assert not parent.edits
    assert not parent.scopes
