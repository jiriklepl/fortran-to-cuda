"""Complete native CALL fingerprints retain original scalar-function callees."""
from copy import copy

import pytest

from compiler.frontend.native_call_effects import prove_native_call_effects
from compiler.frontend.native_demand import prove_native_demand
from compiler.ir import CompilationError
from compiler.scopes.execution_plan import _portable_proof_references
from compiler.tests.test_native_call_effects import issue
from compiler.tests.test_native_call_graph import generate
from compiler.tests.test_native_formal_host_reads import source
from compiler.tests.test_native_graph_associations import SOURCE, setup


def function_source():
    return source().replace("input(1,1,1)", "twice(input(1,1,1))").replace("end module", """pure real(8) function twice(value)
real(8),intent(in)::value
twice=2.0_8*value
end function
end module""")


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("kind", ["complete_call", "demand"])
def test_native_call_function_template_has_a_portable_original_selection(tmp_path, sealed, kind):
    _path, analysis, selection, demands, _reachability = setup(tmp_path, seal=sealed)
    proof = (prove_native_call_effects(analysis, "work::owner", selection) if kind == "complete_call" else
             prove_native_demand(analysis, "work::owner", selection, demands))
    template, = [item for item in proof.reachability.native_graph.templates if item.procedure == "work::twice"]
    assert template.procedure not in analysis.routines
    portable = _portable_proof_references(analysis, (proof,))
    assert portable(template.identity) != template.identity
    assert portable(proof.identity) != proof.identity
    assert template.validate(analysis) is template


@pytest.mark.parametrize("sealed", [False, True])
def test_native_function_fingerprint_is_stable_across_fresh_parse(tmp_path, sealed):
    def fingerprint():
        _, analysis, selection, _, _ = setup(tmp_path, seal=sealed)
        proof = prove_native_call_effects(analysis, "work::owner", selection)
        return analysis, proof

    first_analysis, first = fingerprint()
    second_analysis, second = fingerprint()
    assert first is not second
    assert _portable_proof_references(first_analysis, (first,))(first.identity) == (
        _portable_proof_references(second_analysis, (second,))(second.identity))


def test_portable_fingerprint_changes_with_original_function_body(tmp_path):
    analysis, _, proof = issue(tmp_path)
    before = _portable_proof_references(analysis, (proof,))(proof.identity)
    analysis, _, proof = issue(tmp_path, SOURCE.replace("twice=2.0_8*value", "twice=3.0_8*value"))
    after = _portable_proof_references(analysis, (proof,))(proof.identity)
    assert before != after


@pytest.mark.parametrize("fault", ["function", "template", "source"])
def test_portable_fingerprint_cannot_replace_original_function_authority(tmp_path, fault):
    analysis, _selection, proof = issue(tmp_path)
    template, = [item for item in proof.reachability.native_graph.templates if item.procedure == "work::twice"]
    if fault == "function":
        object.__setattr__(analysis, "numerical_helpers", {**analysis.numerical_helpers,
                                                         "work::twice": copy(template._routine)})
    elif fault == "template":
        object.__setattr__(template, "_routine", copy(template._routine))
    else:
        path = template._routine.scope.path
        path.write_text(path.read_text().replace("twice=2.0_8*value", "twice=3.0_8*value"))
        # Immutable source proofs do not reread files at every metadata query;
        # publication separately authenticates their original input bytes.
        with pytest.raises(CompilationError):
            analysis.verify_publication()
        return
    with pytest.raises(CompilationError):
        _portable_proof_references(analysis, (proof,))


def test_checked_mixed_graph_seals_original_native_function_closure(tmp_path):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    _path, builder, _outputs, report = generate(tmp_path, function_source())
    assert report["scopes"], report["graph_owner_attempt"]
    scope, = report["scopes"]
    assert not scope["boundaries"], scope["boundaries"]
    child, = [p for p in scope["checked_execution_plan"]["procedures"] if p["procedure"] == "native_graph::child"]
    assert not child["boundaries"], child["boundaries"]
    operation, = [op for u in child["units"] for op in u["operations"]["native_operations"]
                   if op["kind"] == "complete original native CALL"]
    assert operation["native_opaque_accesses"]["accesses"][0]["resource"] == "argument::coords"
    validate_checked_execution_plans(report, capture_facts=builder.facts)
    assert builder.finish()
