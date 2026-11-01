"""SELECT CASE contributes every original arm to storage effects."""
import pytest
from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.storage_mutation_effects import STORAGE_SELECT_CASE_VERSION
from compiler.ir import CompilationError


def source(tmp_path, body, *, helpers="", declarations="", module_declarations="", sealed=True):
    path = tmp_path / "cases.f90"
    path.write_text(f"""module cases
implicit none
{module_declarations}
contains
subroutine advance(a,choice)
real(8),allocatable,intent(inout)::a(:)
integer,intent(in)::choice
{declarations}
{body}
end subroutine
{helpers}
end module
""")
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    return path, analysis


def prove(analysis):
    return analysis.storage_mutation_effects("cases::advance")


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("default", ["", "case default\na(:)=3"])
def test_all_original_arms_and_optional_no_match_preserve_descriptors_only(tmp_path, sealed, default):
    _path, analysis = source(tmp_path, f"""select case(choice)
case(:-1,1,3:5,8:)
a(:)=1
case(2)
a(:)=2
{default}
end select""", sealed=sealed)
    proof = prove(analysis)
    record = proof.public()
    assert proof.preserves_descriptors()
    assert proof.descriptor_unchanged("argument::a")
    assert record["version"] == STORAGE_SELECT_CASE_VERSION == 13
    case, = record["original_select_cases"]
    assert case["procedure"] == "cases::advance"
    assert "SELECT CASE (choice)" in case["original_source"]
    assert case["coverage"] == "original selector, every CASE bound and every arm body"
    assert case["branch_selection_authorized"] is case["guaranteed_match_authorized"] is False
    assert record["definition_kills"] == record["mutations"] == record["escapes"] == record["frontiers"] == []
    for flag in ("payload_access_authorized", "entry_definition_authority", "capture_authorized",
                 "execution_authorized", "motion_authorized", "gpu_legality_established", "alias_authority"):
        assert record[flag] is False
    assert analysis._summary_authority()["storage_select_case_version"] == 13
    assert proof.validate() is proof


def test_non_case_record_keeps_existing_shape(tmp_path):
    _path, analysis = source(tmp_path, "a(:)=1")
    record = prove(analysis).public()
    assert record["version"] == 1
    assert "original_select_cases" not in record


@pytest.mark.parametrize("effect", ["deallocate(a)", "allocate(a(3))", "a=1", "call hidden(a)"])
def test_nonmatching_arm_and_tail_after_return_keep_all_mutations(tmp_path, effect):
    _path, analysis = source(tmp_path, f"""select case(1)
case(1)
a(:)=1
case(2)
return
{effect}
end select""", helpers="""subroutine hidden(x)
real(8),intent(inout)::x(:)
deallocate(global)
end subroutine""", module_declarations="real(8),allocatable::global(:)")
    proof = prove(analysis)
    assert not proof.preserves_descriptors()
    assert proof.public()["mutations"]
    if effect == "call hidden(a)":
        assert any(row["resource"] == "cases::global" for row in proof.public()["mutations"])
        assert {row["procedure"] for row in proof.public()["closure"]} == {"cases::advance", "cases::hidden"}
    assert any(row["kind"] == "original control/termination" for row in proof.public()["observables"])


@pytest.mark.parametrize("effect", ["call unknown(a)", "a(1)=unknown(a(1))", "p=>target"])
def test_unknown_calls_and_pointer_association_in_any_arm_remain_frontiers(tmp_path, effect):
    declarations = "real(8),pointer::p\nreal(8),target::target" if effect == "p=>target" else ""
    _path, analysis = source(tmp_path, f"""select case(choice)
case(1)
a(:)=1
case default
{effect}
end select""", declarations=declarations)
    proof = prove(analysis)
    assert not proof.descriptor_unchanged("argument::a")
    assert proof.public()["frontiers"] or proof.public()["escapes"]
    if effect == "p=>target":
        assert any(row["kind"] == "pointer association" for row in proof.public()["escapes"])


@pytest.mark.parametrize("position", ["selector", "body"])
@pytest.mark.parametrize("effect", ["deallocate(global)", "call unknown()"])
def test_source_functions_in_selector_or_body_keep_full_hidden_closure(tmp_path, position, effect):
    selector = "choose()" if position == "selector" else "choice"
    body = "a(:)=choose()" if position == "body" else "a(:)=1"
    _path, analysis = source(tmp_path, f"""select case({selector})
case(1)
{body}
end select""", helpers=f"""integer function choose()
{effect}
choose=1
end function""", module_declarations="real(8),allocatable::global(:)")
    proof = prove(analysis)
    record = proof.public()
    assert not proof.preserves_descriptors()
    assert {row["procedure"] for row in record["closure"]} == {"cases::advance", "cases::choose"}
    assert record["mutations"] or record["frontiers"]


def test_case_bound_names_and_intrinsic_constant_expressions_are_scanned(tmp_path):
    _path, analysis = source(tmp_path, """select case(choice)
case(lower:abs(-2))
a(:)=1
case(upper:)
a(:)=2
end select""", module_declarations="integer,parameter::lower=1,upper=3")
    proof = prove(analysis)
    assert proof.preserves_descriptors()
    assert {"cases::lower", "cases::upper"} <= set(proof.public()["resources"])


@pytest.mark.parametrize("label", ["unknown()", "abs(-2)"])
def test_unproved_case_constant_expression_cannot_hide_function_effects(tmp_path, label):
    # A source function is not a legal CASE constant expression. The parser
    # accepts this shape; the projection must still retain its effects rather
    # than treat a label or intrinsic spelling as an exemption.
    helpers = """integer function abs(x)
integer,intent(in)::x
deallocate(global)
abs=1
end function""" if label == "abs(-2)" else ""
    _path, analysis = source(tmp_path, f"""select case(choice)
case({label})
a(:)=1
end select""", helpers=helpers, module_declarations="real(8),allocatable::global(:)")
    proof = prove(analysis)
    assert not proof.preserves_descriptors()
    assert proof.public()["frontiers"] or proof.public()["mutations"]
    if helpers:
        assert any(row["resource"] == "cases::global" for row in proof.public()["mutations"])


def test_nested_cases_keep_all_arms_and_original_observable_order(tmp_path):
    _path, analysis = source(tmp_path, """select case(choice)
case(1)
select case(choice+1)
case(2)
write(*,*)1
case default
write(*,*)2
end select
case default
write(*,*)3
end select""")
    proof = prove(analysis)
    record = proof.public()
    assert proof.preserves_descriptors()
    assert len(record["original_select_cases"]) == 2
    assert [row["original_source"] for row in record["observables"]] == ["WRITE(*, *) 1", "WRITE(*, *) 2", "WRITE(*, *) 3"]


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("part", ["selector", "bound", "body"])
def test_original_case_syntax_cannot_change_at_final_publication(tmp_path, monkeypatch, sealed, part):
    _path, analysis = source(tmp_path, """select case(choice)
case(1)
a(:)=2
case default
a(:)=3
end select""", sealed=sealed)
    routine = analysis.routines["cases::advance"]
    parser_kind, spelling = {"selector": ("Name", "choice"), "bound": ("Int_Literal_Constant", "1"),
                             "body": ("Int_Literal_Constant", "2")}[part]
    node = next(item for item in walk(routine.execution) if type(item).__name__ == parser_kind and str(item) == spelling)
    original = analysis.verify_publication
    reached = []

    def changed():
        result = original()
        reached.append(node)
        if part == "selector":
            object.__setattr__(node, "string", "missing")
        else:
            object.__setattr__(node, "items", ("99", None))
        return result

    monkeypatch.setattr(analysis, "verify_publication", changed)
    with pytest.raises(CompilationError):
        prove(analysis)
    assert reached
    assert not getattr(analysis, "_storage_mutation_effects", {})


@pytest.mark.parametrize("fault", ["file", "namespace", "copy"])
def test_retained_case_projection_rejects_stale_or_foreign_source(tmp_path, fault):
    from copy import copy

    path, analysis = source(tmp_path, "select case(choice)\ncase(1)\na(:)=1\nend select", sealed=False)
    proof = prove(analysis)
    if fault == "file":
        path.write_text(path.read_text() + "! changed\n")
    elif fault == "namespace":
        analysis.routines["cases::advance"].scope.imports["choice"] = ("unknown_module", "other")
    else:
        proof = copy(proof)
    with pytest.raises(CompilationError):
        proof.validate()


@pytest.mark.parametrize("limit", [1, 2])
def test_all_case_function_closures_count_toward_real_procedure_budget(tmp_path, limit):
    _path, analysis = source(tmp_path, """select case(first())
case(1)
a(:)=first()
case default
a(:)=second()
end select""", helpers="""integer function first()
first=1
end function
integer function second()
second=2
end function""")
    analysis.procedure_limit = limit
    with pytest.raises(CompilationError, match="closure/recursion"):
        prove(analysis)
    assert not getattr(analysis, "_storage_mutation_effects", {})


def test_repeated_calls_across_arms_count_unique_callees_and_keep_each_handle(tmp_path):
    _path, analysis = source(tmp_path, """select case(first())
case(1)
a(:)=first()
case default
a(:)=first()
end select""", helpers="integer function first()\nfirst=1\nend function")
    analysis.procedure_limit = 2
    proof = prove(analysis)
    assert proof.preserves_descriptors()
    assert {row["procedure"] for row in proof.public()["closure"]} == {"cases::advance", "cases::first"}
    assert len(proof._calls) == 3
    assert len({id(call._original) for call in proof._calls}) == 3


def test_fixed_array_function_select_case_preserves_existing_result_receipt(tmp_path):
    _path, analysis = source(tmp_path, "a(:)=vector(a,choice)", helpers="""function vector(x,which) result(value)
real(8),intent(in)::x(:)
integer,intent(in)::which
real(8)::value(3)
select case(which)
case(1)
value=[x(1),x(2),x(3)]
case default
value=[x(3),x(2),x(1)]
end select
end function""")
    proof = prove(analysis)
    record = proof.public()
    assert proof.preserves_descriptors()
    assert record["version"] == 13
    result, = record["fixed_array_function_results"]
    assert result["bounds"] == [[1, 3]]
    assert record["original_select_cases"][0]["procedure"] == "cases::vector"


def test_selector_hidden_scalar_payload_write_is_not_descriptor_or_execution_credit(tmp_path):
    _path, analysis = source(tmp_path, "select case(choose())\ncase(1)\na(:)=1\nend select",
        module_declarations="integer::counter", helpers="""integer function choose()
counter=counter+1
choose=1
end function""")
    proof = prove(analysis)
    record = proof.public()
    assert proof.preserves_descriptors()
    assert "cases::counter" in record["resources"]
    assert {row["procedure"] for row in record["closure"]} == {"cases::advance", "cases::choose"}
    assert record["payload_access_authorized"] is record["execution_authorized"] is False


@pytest.mark.parametrize(("selector", "labels"), [
    (".true.", (".true.", ".false.")),
    ("'a'", ("'a'", "'b':'z'")),
])
def test_literal_logical_and_character_selectors_use_the_same_all_arm_scan(tmp_path, selector, labels):
    _path, analysis = source(tmp_path, f"""selection: select case({selector})
case({labels[0]}) selection
a(:)=1
case({labels[1]}) selection
a(:)=2
end select selection""")
    proof = prove(analysis)
    assert proof.preserves_descriptors()
    assert len(proof.public()["original_select_cases"]) == 1


@pytest.mark.parametrize("sealed", [False, True])
def test_case_in_record_function_keeps_owned_region_fixed_result_and_terminating_requirements(tmp_path, sealed):
    from compiler.tests.test_descriptor_record_calls import fixture
    from compiler.tests.test_descriptor_record_calls import prove as prove_window
    from compiler.tests.test_owned_record_whole_body import fixed_result_source
    from compiler.tests.test_record_terminating_calls import source as terminating_source

    text = fixed_result_source().replace("value=[a(i),a(i),a(i)]", """select case(mod(i,2))
case(0)
value=[a(i),a(i),a(i)]
case default
value=[a(i)+1,a(i)+2,a(i)+3]
end select""")
    text = terminating_source(base=text)
    window = prove_window(fixture(tmp_path, text, sealed=sealed)).public()
    assert window["schema_version"] == 27
    owned, = window["owned_record_call_requirements"]
    storage = owned["storage_mutation"]
    assert storage["version"] == 13
    assert storage["owned_record_region"]["version"] == 2
    assert storage["fixed_array_function_results"][0]["bounds"] == [[1, 3]]
    assert storage["original_terminating_calls"][0]["requires_original_call_execution"]
    assert storage["original_select_cases"][0]["procedure"] == "regions::sample"
    assert {row["procedure"] for row in storage["closure"]} == {
        "regions::touch", "regions::sample", "regions::update", "diagnostics::fail_message"}
    assert owned["complete_original_body_and_hidden_effects_scanned"]
    assert not storage["frontiers"]
    assert not storage["mutations"]
    assert not storage["escapes"]
    assert not storage["execution_authorized"]
    assert not owned["execution_authorized"]
