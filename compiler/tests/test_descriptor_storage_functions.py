"""Descriptor windows consume storage effects without authorizing native work."""
import pytest

from compiler.frontend.descriptor_windows import DescriptorWindowUnsupported, _derive
from compiler.ir import CompilationError
from compiler.tests.test_descriptor_source_functions import function_fixture
from compiler.tests.test_descriptor_window_proofs import ENTRY, ROOT, prove


@pytest.mark.parametrize(("declaration", "body", "extra"), [
    ("real(8),intent(in)::x", "print *,x\nok=x>0", ""),
    ("real(8),intent(inout)::x", "x=x+1\nok=x>0", ""),
    ("real(8),intent(out)::x", "x=1\nok=x>0", ""),
    ("real(8),intent(in)::x\nreal(8),allocatable::scratch(:)",
     "allocate(scratch(3))\nscratch(:)=x\nok=sum(scratch)>0", ""),
    ("real(8),intent(in)::x", "call update(x)\nok=x>0",
     "subroutine update(y)\nreal(8),intent(in)::y\nprint *,y\nend subroutine"),
])
def test_payload_mutation_output_and_private_lifetimes_do_not_require_readonly_numeric_proof(
        tmp_path, declaration, body, extra):
    _path, analysis, _graph, selected, caller, expression = function_fixture(tmp_path,
        declaration=declaration, body=body, extra=extra)
    summary = analysis.summarize(caller.qualified)
    public = prove(analysis, selected).public()
    assert public["schema_version"] == 9
    projection, = public["native_storage_function_requirements"]
    assert projection["version"] == 2
    assert projection["kind"] == "original_storage_function_mutation_effects"
    assert projection["original_expression"]["source"] == str(expression)
    assert projection["procedure"] == caller.qualified
    assert not projection["frontiers"]
    for flag in ("execution_authorized", "payload_access_authorized", "entry_definition_authority",
                 "capture_authorized", "motion_authorized", "gpu_legality_established", "alias_authority"):
        assert projection[flag] is False
    assert not analysis._native_functions
    assert not analysis._descriptor_functions
    assert analysis.summarize(caller.qualified) == summary
    assert public["entry_definition_authority"] is False
    assert public["execution_authorized"] is False
    assert prove(analysis, selected).resource_state(ROOT)["stable"]


def test_original_intent_out_definition_kill_is_retained_without_definition_credit(tmp_path):
    _path, analysis, _graph, selected, _caller, _expression = function_fixture(tmp_path,
        declaration="real(8),intent(out)::x", body="x=1\nok=x>0")
    projection, = prove(analysis, selected).public()["native_storage_function_requirements"]
    kill, = projection["definition_kills"]
    assert kill["kind"] == "original INTENT(OUT) entry kill"
    assert kill["resource"] == "intervals::read_modify::value"
    assert not projection["entry_definition_authority"]


def test_direct_consumer_cannot_issue_its_own_storage_authority(tmp_path):
    _path, analysis, _graph, selected, caller, expression = function_fixture(tmp_path,
        declaration="real(8),intent(inout)::x", body="x=x+1\nok=x>0")
    with pytest.raises(DescriptorWindowUnsupported, match="registered original readonly"):
        _derive(analysis, ENTRY, selected, (ROOT,))
    token = analysis.storage_mutation_effects(caller.qualified, expression=expression)
    public = _derive(analysis, ENTRY, selected, (ROOT,))
    assert public["native_storage_function_requirements"] == [token.public()]


@pytest.mark.parametrize(("declaration", "body", "extra"), [
    ("real(8),pointer::x", "ok=x>0", ""),
    ("real(8),target,intent(in)::x", "ok=x>0", ""),
    ("real(8),intent(in)::x\nreal(8)::p,q\nequivalence(p,q)", "ok=x>0", ""),
    ("real(8),intent(in)::x", "call absent(x)\nok=x>0", ""),
    ("real(8),intent(in)::x", "ok=x>0\ncall hidden()",
     "subroutine hidden()\ncall absent()\nend subroutine"),
    ("real(8),intent(in)::x", "!$omp parallel\nok=x>0\n!$omp end parallel", ""),
])
def test_storage_frontiers_and_full_callee_tail_still_decline(tmp_path, declaration, body, extra):
    _path, analysis, _graph, selected, _caller, _expression = function_fixture(tmp_path,
        declaration=declaration, body=body, extra=extra)
    with pytest.raises(CompilationError):
        prove(analysis, selected)


def test_hidden_module_allocation_mutation_cannot_be_excluded_by_distinct_spelling(tmp_path):
    path, _analysis, _graph, _selected, _caller, _expression = function_fixture(tmp_path,
        body="deallocate(hidden)\nok=x>0")
    path.write_text(path.read_text().replace("module intervals\n", "module intervals\nreal(8),allocatable::hidden(:)\n"))
    from compiler.frontend.source_effects import SourceEffects

    analysis = SourceEffects([path])
    graph = analysis.structure(ENTRY)
    routine = analysis.routines[ENTRY]
    loop = next(node for node in routine.execution.content if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    selected = (graph.node_id(loop.content[1]),)
    with pytest.raises(CompilationError, match="storage mutation/escape"):
        prove(analysis, selected)


def test_registered_readonly_record_and_no_new_schema_remain_exact(tmp_path):
    _path, analysis, _graph, selected, _caller, _expression = function_fixture(tmp_path)
    public = prove(analysis, selected).public()
    assert public["schema_version"] in {6, 7, 8}
    assert "native_storage_function_requirements" not in public
    assert len(public["native_descriptor_source_function_requirements"]) == 1
    assert not getattr(analysis, "_storage_mutation_effects", {})
