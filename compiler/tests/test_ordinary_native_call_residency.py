"""A real read-only native team between checked GPU children keeps replicas."""
import json

import pytest

from compiler.scopes.lexical import LexicalBoundaryPatch
from compiler.tests.test_hierarchical_child_regions import SOURCE, fixture


def source():
    old = "observed=0.0_8\ndo i=1,n\nobserved=observed+right(lbound(right,1)+i-1)\nenddo"
    worker = """subroutine sample()
real(8)::scratch(size(right))
integer::i
!$omp parallel do private(i)
do i=1,size(right)
scratch(i)=right(lbound(right,1)+i-1)
enddo
!$omp end parallel do
observed=sum(scratch)
end subroutine
"""
    assert SOURCE.count(old) == 1
    return SOURCE.replace(old, "call sample()").replace("end module", worker + "end module")


def emit(directory):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    original, builder = fixture(directory, source())
    outputs, report = builder.run()
    assert report["scope_count"] == 1
    member = builder._conditional_return_draft.owner.members["hierarchical::middle"]
    operation, = [op for unit in member.units for op in unit.scope.native if op.call_effects is not None]
    assert operation.coherence_effects == {"hierarchical::right": {"read"}}
    assert operation.preserve_original
    assert not operation.overwrites
    assert type(operation.nodes[0]).__name__ == "Call_Stmt"
    assert str(operation.nodes[0].items[0]).lower() == "sample"
    assert not any("sample" in str(patch.originals).lower() for _, _, patch in member.patches
                   if isinstance(patch, LexicalBoundaryPatch))
    proof = operation.call_effects
    assert proof.validate(operation.proof_analysis, member.routine.qualified, operation.nodes) is proof
    assert proof.reachability._completion
    validate_checked_execution_plans(report)
    (directory / "ordinary-call-proof.json").write_text(json.dumps(proof.public(), indent=2) + "\n")
    return original, outputs, report


def test_ordinary_joined_readonly_call_keeps_original_source_and_strict_transport(tmp_path):
    emit(tmp_path)


@pytest.fixture(scope="module")
def binaries(tmp_path_factory):
    from compiler.tests import test_graph_storage_epochs_cuda as base
    from compiler.tests import test_hierarchical_child_regions_cuda as hierarchy

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, "emit", emit)
        patch.setattr(base, "DRIVER", hierarchy.DRIVER)
        directory, candidate, native = base.epoch_binaries.__wrapped__(tmp_path_factory)
    manifest = json.loads((directory / "scope-manifest.json").read_text())
    provenance = {row["id"]: row for row in manifest["runtime_provenance"]["records"]}
    return "overwrite", directory, candidate, native, None, provenance


@pytest.mark.cuda
@pytest.mark.parametrize("hidden", [False, True], ids=["enabled", "hidden"])
def test_joined_readonly_call_preserves_fields_and_replica_ledger(binaries, hidden):
    from compiler.tests import test_hierarchical_child_regions_cuda as hierarchy

    # Reuse the independently defined six-case field/bounds/swap and physical
    # copy oracle. The CPU observation now runs through a separate joined CALL.
    hierarchy.test_child_host_read_replica_and_checked_retirement(binaries, hidden)
