"""Independent original windows do not consume one shared call-closure budget."""
import pytest

from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

ENTRY = "capacity::run"
ROOT = "argument::a"


def fixture(tmp_path, *, statements, procedures=1, extra="", sealed=False):
    path = tmp_path / "capacity.f90"
    path.write_text("module capacity\ncontains\n" + extra + "subroutine run(a)\n"
                    "real(8),allocatable,intent(inout)::a(:)\n" + statements
                    + "\nend subroutine\nend module\n")
    analysis = SourceEffects([path], procedures=procedures)
    if sealed:
        analysis.seal_session()
    routine = analysis.routines[ENTRY]
    graph, selected = analysis._selected_source(ENTRY, tuple(routine.execution.content))
    return analysis, graph, selected


@pytest.mark.parametrize("sealed", [False, True])
def test_70_distinct_original_windows_keep_a_one_procedure_query_limit(tmp_path, sealed):
    analysis, graph, selected = fixture(tmp_path, sealed=sealed,
        statements="\n".join(f"a(1)=a(1)+{i}.0_8" for i in range(70)))
    proofs = [prove_descriptor_window(analysis, ENTRY, (identity,), (ROOT,)) for identity in selected]
    assert len(proofs) == len({proof.identity for proof in proofs}) == 70
    assert len(analysis._descriptor_window_proofs) == 70
    assert analysis.procedure_limit == 1
    for identity, proof in zip(selected, proofs, strict=True):
        record = proof.public()
        assert record["selected_node_ids"] == [identity]
        assert record["structured_identity"] == graph.identity
        assert record["resolution_validation"]["procedure_limit"] == 1
        assert record["source_call_closure"] == {}
        assert record["execution_authorized"] is False
        assert record["entry_definition_authority"] is False
    assert prove_descriptor_window(analysis, ENTRY, (selected[-1],), (ROOT,)) is proofs[-1]


@pytest.mark.parametrize(("procedures", "calls", "accepted"), [
    (1, "call work(a)", False),
    (2, "call work(a)", True),
    (2, "call work(a)\ncall other(a)", False),
])
@pytest.mark.parametrize("sealed", [False, True])
def test_each_query_still_requires_its_bounded_complete_call_closure(tmp_path, procedures, calls, accepted, sealed):
    extra = ("subroutine work(a)\nreal(8),intent(inout)::a(:)\na(1)=2.0_8\nend subroutine\n"
             "subroutine other(a)\nreal(8),intent(inout)::a(:)\na(1)=3.0_8\nend subroutine\n")
    analysis, _graph, selected = fixture(tmp_path, procedures=procedures, extra=extra,
                                          statements=calls, sealed=sealed)
    if accepted:
        proof = prove_descriptor_window(analysis, ENTRY, selected, (ROOT,))
        assert list(proof.public()["source_call_closure"]) == ["capacity::work"]
        assert proof.public()["resolution_validation"]["procedure_limit"] == procedures
    else:
        with pytest.raises(CompilationError, match="direct call closure budget"):
            prove_descriptor_window(analysis, ENTRY, selected, (ROOT,))
        assert not getattr(analysis, "_descriptor_window_proofs", {})
