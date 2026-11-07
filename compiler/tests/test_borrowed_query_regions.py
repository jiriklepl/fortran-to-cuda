"""A reached checked child can price its connected work in the live domain."""
from unittest.mock import patch

import pytest

from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes import query_regions
from compiler.tests.test_native_ancestor import SOURCE, fixture
from compiler.tests.test_scoped_planning_entries import calibration

NESTED = SOURCE.replace(
    "call producer(a,b,n)\nobserved=0.0_8\ndo i=1,n\nobserved=observed+b(i)\nenddo\n"
    "if(flag) call opaque(b,n)\ncall consumer(a,b,n)",
    "if(flag)then\ncall producer(a,b,n)\ncall consumer(a,b,n)\nendif").replace(
    "call middle(a,b,n,flag)", "call producer(a,b,n)\ncall middle(a,b,n,flag)", 1).replace(
    "a(i)=b(i)+1.0_8", "a(i)=a(i)+b(i)+1.0_8")


def emit(path, source=NESTED):
    original, builder = fixture(path, source=source)
    builder.config = OffloadConfig("auto", calibration())
    proofs = []
    seal = query_regions.seal_query_regions

    def retain(*args, **kwargs):
        result = seal(*args, **kwargs)
        proofs.extend(result)
        return result

    with patch.object(query_regions, "seal_query_regions", side_effect=retain):
        outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    text = outputs[report["sources"][str(original)]["replacement"]]
    return report, text, proofs


def test_reached_child_groups_workers_using_only_existing_canonical_handles(tmp_path):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    report, text, proofs = emit(tmp_path)
    validate_checked_execution_plans(report)
    proof, = [p for p in proofs if p.anchor.routine.qualified == "ancestor_fixture::middle"]
    assert proof.public()["workers"] == 2
    assert proof.public()["entry_definitions"] == "existing live handles only; no registration or new definitions"
    block = "fort_query_" + proof.identity[:12]
    query = text[text.index(block + ": block"):text.index("end block " + block)]
    assert "fort_scope_buffer_valid_v1" in query
    assert "fort_scope_resident_create_v1" not in query
    assert "fort_scope_register" not in query
    assert proof.anchor.epoch_pristine not in query
    for root, _binding in proof.arrays:
        assert "fort_buffer_" + str(proof.anchor.resources[root][1]) in query
    assert text.lower().replace(" ", "").index("if(flag)then") < text.index(block + ": block")
    proof.validate()


@pytest.mark.parametrize("fault", ["context", "enabled", "query_plan", "epoch_pristine", "control_guard",
                                  "slot", "missing"])
def test_child_query_keeps_its_original_context_controls_and_forwarded_slots(tmp_path, fault):
    _report, _text, proofs = emit(tmp_path)
    proof, = [p for p in proofs if p.anchor.routine.qualified == "ancestor_fixture::middle"]
    if fault in {"slot", "missing"}:
        resources = dict(proof.anchor.resources)
        root = proof.arrays[0][0]
        if fault == "missing":
            del resources[root]
        else:
            resources[root] = (resources[root][0], -1)
        object.__setattr__(proof.anchor, "resources", resources)
    else:
        object.__setattr__(proof.anchor, fault, "foreign_control")
    with pytest.raises(CompilationError, match="query region changed"):
        proof.validate()
