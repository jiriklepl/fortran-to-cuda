"""A typed record association never drops original prelude, tail, or callees."""
from copy import copy

import pytest

from compiler.frontend.descriptor_record_calls import original_record_call
from compiler.ir import CompilationError
from compiler.tests.test_descriptor_record_calls import fixture, prove
from compiler.tests.test_storage_record_regions import SOURCE

PRELUDE = """real(8)::delta(3),vec(3),mag
delta=[1.0_8,2.0_8,3.0_8]
"""
CALLEES = """subroutine update(value,vec,offset)
real(8),intent(inout)::value
real(8),intent(in)::vec(3),offset
value=value+vec(1)+offset
end subroutine
real(8) function sample(a,i) result(value)
real(8),intent(in)::a(:)
integer,intent(in)::i
value=a(i)
end function
"""
WHOLE = SOURCE.replace("!$omp parallel do private(i,p)", PRELUDE+"!$omp parallel do private(i,p,vec,mag)")
WHOLE = WHOLE.replace("p%value=p%value+a(i)", "vec=[a(i),a(i),a(i)]\nmag=real(mod(i,2),8)+sample(a,i)\ncall update(p%value,vec,offset=mag+delta(1))")
WHOLE = WHOLE.replace("end module", CALLEES+"end module")


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("norm", [False, True], ids=["mod", "norm2"])
def test_whole_original_body_uses_existing_scalar_function_vector_keyword_and_intrinsic(tmp_path, sealed, norm):
    source = WHOLE.replace("real(mod(i,2),8)", "norm2(vec)") if norm else WHOLE
    data = fixture(tmp_path, source, sealed=sealed)
    record = prove(data).public()
    assert record["schema_version"] == 27
    call, = record["owned_record_call_requirements"]
    assert call["owned_record_region"]["version"] == 2
    assert call["storage_mutation"]["version"] == 10
    assert call["complete_original_body_and_hidden_effects_scanned"]
    assert call["complete_original_callee_specification_and_lifecycle_scanned"]
    assert {row["procedure"] for row in call["storage_mutation"]["closure"]} == {
        "regions::touch", "regions::update", "regions::sample"}
    body = call["owned_record_region"]["complete_original_body"]
    assert body["source"].startswith("delta = [1.0_8, 2.0_8, 3.0_8]")
    assert len(body["selected_node_ids"]) > len(call["team_storage"]["selected_node_ids"])
    assert call["storage_mutation"]["original_selection"]["selected_node_ids"] == body["selected_node_ids"]
    assert not call["execution_authorized"]
    assert not call["alias_authority"]
    assert not call["storage_mutation"]["frontiers"]
    assert not call["storage_mutation"]["mutations"]
    assert not call["storage_mutation"]["escapes"]


def fixed_result_source():
    return WHOLE.replace("+sample(a,i)", "+sum(sample(a,i))").replace(
        "real(8) function sample(a,i) result(value)", "function sample(a,i) result(value)").replace(
        "integer,intent(in)::i\nvalue=a(i)", "integer,intent(in)::i\nreal(8)::value(3)\nvalue=[a(i),a(i),a(i)]")


@pytest.mark.parametrize("sealed", [False, True])
def test_whole_body_record_context_composes_with_fixed_array_function_result(tmp_path, sealed):
    owned, = prove(fixture(tmp_path, fixed_result_source(), sealed=sealed)).public()["owned_record_call_requirements"]
    storage = owned["storage_mutation"]
    assert storage["version"] == 11
    assert storage["owned_record_region"]["version"] == 2
    result, = storage["fixed_array_function_results"]
    assert result["procedure"] == "regions::sample"
    assert result["bounds"] == [[1, 3]]
    assert not storage["execution_authorized"]
    assert not storage["payload_access_authorized"]


@pytest.mark.parametrize("effect", ["call unknown(a)", "deallocate(field)"])
def test_original_tail_and_hidden_callee_effects_are_not_discarded(tmp_path, effect):
    source = WHOLE.replace("value=value+vec(1)+offset", "value=value+vec(1)+offset\n"+effect.replace("unknown(a)", "unknown(value)"))
    with pytest.raises(CompilationError):
        prove(fixture(tmp_path, source, sealed=True))


@pytest.mark.parametrize("placement", ["prefix", "tail"])
def test_pointer_component_cannot_be_used_outside_its_dominating_original_loop(tmp_path, placement):
    before = "delta=[1.0_8,2.0_8,3.0_8]"
    source = WHOLE.replace(before, before+"\nmag=p%value") if placement == "prefix" else WHOLE.replace("!$omp end parallel do", "!$omp end parallel do\nmag=p%value")
    with pytest.raises(CompilationError, match="dominat|inside.*loop"):
        prove(fixture(tmp_path, source, sealed=True))


def test_record_component_allocation_remains_unavailable(tmp_path):
    source = WHOLE.replace("real(8)::value", "real(8)::value\nreal(8),allocatable::extra(:)", 1)
    with pytest.raises(CompilationError):
        prove(fixture(tmp_path, source, sealed=True))


@pytest.mark.parametrize("nested", [False, True], ids=["shadowed_norm2", "nested_argument"])
def test_norm2_spelling_never_skips_resolved_or_nested_hidden_effects(tmp_path, nested):
    if nested:
        source = WHOLE.replace("real(mod(i,2),8)+sample(a,i)", "norm2([sample(a,i),1.0_8,2.0_8])")
        source = source.replace("value=a(i)", "deallocate(field)\nvalue=a(i)")
    else:
        source = WHOLE.replace("real(mod(i,2),8)", "norm2(vec)")
        shadow = "real(8) function norm2(vec) result(v)\nreal(8),intent(in)::vec(:)\ndeallocate(field)\nv=vec(1)\nend function\n"
        source = source.replace("end module", shadow+"end module")
    with pytest.raises(CompilationError, match="mutation|escape|incomplete"):
        prove(fixture(tmp_path, source, sealed=True))


def test_complete_body_selection_cannot_be_replaced_by_just_its_team(tmp_path):
    from compiler.frontend.descriptor_teams import prove_descriptor_team
    from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
    from compiler.frontend.storage_record_regions import prove_owned_record_region

    analysis, caller, _selected, _roots, _modules = fixture(tmp_path, WHOLE, sealed=True)
    call, = caller.execution.content
    region = prove_owned_record_region(analysis, caller.qualified, call, (analysis.modules["regions"].bindings["field"],))
    team = prove_descriptor_team(analysis, region._roles[1].qualified, region._roles[6], record_region=region)
    with pytest.raises(CompilationError, match="complete exact original callee body"):
        prove_storage_mutation_effects(analysis, region._roles[1].qualified,
                                      selection=team.selected_node_ids, joined_storage=team)


@pytest.mark.parametrize("sealed", [False, True])
def test_late_source_function_association_registry_cannot_be_replaced(tmp_path, monkeypatch, sealed):
    analysis, caller, _selected, _roots, _modules = fixture(tmp_path, WHOLE, sealed=sealed)
    call, = caller.execution.content
    proof = original_record_call(analysis, caller, call, (analysis.modules["regions"].bindings["field"],))
    function, = [row for row in proof._storage._calls if row.procedure == "regions::sample"]
    original = analysis.verify_publication
    fired = False

    def changed():
        nonlocal fired
        original()
        if not fired:
            fired = True
            analysis._original_call_associations[id(analysis), id(function._original)] = copy(function)

    monkeypatch.setattr(analysis, "verify_publication", changed)
    with pytest.raises(CompilationError):
        proof.validate()
    assert fired


def many_function_source(count=6):
    template = CALLEES[CALLEES.index("real(8) function sample"):]
    declarations = "\n".join(template.replace("sample(", f"sample{k}(") for k in range(count))
    source = WHOLE.replace("mag=real(mod(i,2),8)+sample(a,i)",
                           "mag="+"+".join(f"sample{k}(a,i)" for k in range(count)))
    return source.replace(template, declarations)


def test_every_direct_source_function_counts_in_the_complete_cumulative_closure(tmp_path):
    source = many_function_source()
    good = tmp_path/"good"
    good.mkdir()
    data = fixture(good, source)
    data[0].procedure_limit = 9
    record = prove(data).public()
    call, = record["owned_record_call_requirements"]
    assert {row["procedure"] for row in call["storage_mutation"]["closure"]} == {
        "regions::touch", "regions::update", *(f"regions::sample{k}" for k in range(6))}
    bad = tmp_path/"bad"
    bad.mkdir()
    data = fixture(bad, source)
    data[0].procedure_limit = 8
    with pytest.raises(CompilationError, match="composed original call closure budget"):
        prove(data)


def test_previous_record_function_inventory_counts_before_a_later_ordinary_call(tmp_path):
    source = many_function_source().replace("call touch(records,field)", "call touch(records,field)\ncall plain()")
    source = source.replace("end module", "subroutine plain()\nend subroutine\nend module")
    data = fixture(tmp_path, source)
    data[0].procedure_limit = 9
    with pytest.raises(CompilationError, match="direct call closure budget"):
        prove(data)


def test_root_caller_depth_is_added_to_the_complete_source_function_depth(tmp_path):
    data = fixture(tmp_path, WHOLE)
    data[0].depth_limit = 2
    with pytest.raises(CompilationError, match="composed original call depth budget"):
        prove(data)


@pytest.mark.parametrize("sealed", [False, True])
def test_final_storage_validation_cannot_detach_an_original_prefix_statement(tmp_path, monkeypatch, sealed):
    from compiler.frontend.storage_mutation_effects import StorageMutationEffects

    analysis, caller, _selected, _roots, _modules = fixture(tmp_path, WHOLE, sealed=sealed)
    call, = caller.execution.content
    proof = original_record_call(analysis, caller, call, (analysis.modules["regions"].bindings["field"],))
    callee = proof._region._roles[1]
    content = callee.execution.content
    if sealed:
        original_facts = proof.public()
        with pytest.raises(CompilationError, match="immutable"):
            callee.execution.content = (copy(content[0]), *content[1:])
        assert proof.validate() is proof
        assert proof.public() == original_facts
        return
    original = StorageMutationEffects.validate
    fired = False

    def changed(self, *args, **kwargs):
        nonlocal fired
        result = original(self, *args, **kwargs)
        if self is proof._storage and not fired:
            fired = True
            object.__setattr__(callee.execution, "content", (copy(content[0]), *content[1:]))
        return result

    monkeypatch.setattr(StorageMutationEffects, "validate", changed)
    with pytest.raises(CompilationError):
        proof.validate()
    assert fired


def test_authentic_emission_keeps_whole_callee_and_none_facts(tmp_path):
    import json

    from compiler.tests.test_descriptor_record_calls import emitted_record_retirement, emitted_source
    from compiler.tests.test_graph_storage_epochs import builder_at

    before = "subroutine touch"+SOURCE.split("subroutine touch", 1)[1].rsplit("end module", 1)[0]
    after = "subroutine touch"+fixed_result_source().split("subroutine touch", 1)[1].rsplit("end module", 1)[0]
    source = emitted_source()
    assert source.count(before) == 1
    source = source.replace(before, after)
    path, builder = builder_at(tmp_path, source)
    outputs, report = builder.run()
    storage, completion = emitted_record_retirement(path, builder, outputs, report)
    assert {row["procedure"] for row in storage["closure"]} == {
        "epochs::touch", "epochs::sample", "epochs::update"}
    result, = storage["fixed_array_function_results"]
    assert result["procedure"] == "epochs::sample"
    assert result["bounds"] == [[1, 3]]
    assert all(row["initialized"] == "none" for row in builder.facts["captures"].values())
    for name, content in outputs.items():
        target = tmp_path/"generated"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    for name, value in [("report.json", report), ("facts.json", builder.facts),
                        ("retirement-records.json", {"storage": storage, "completion": completion})]:
        (tmp_path/name).write_text(json.dumps(value, indent=2)+"\n")
    assert path.read_text() == source


def test_shadowed_norm2_uses_original_user_function_body(tmp_path):
    source = WHOLE.replace("real(mod(i,2),8)", "norm2(vec)")
    source = source.replace("end module", "real(8) function norm2(vec) result(v)\nreal(8),intent(in)::vec(:)\nv=vec(1)\nend function\nend module")
    owned, = prove(fixture(tmp_path, source, sealed=True)).public()["owned_record_call_requirements"]
    assert "regions::norm2" in {row["procedure"] for row in owned["storage_mutation"]["closure"]}


def test_builtin_norm2_cannot_issue_original_user_function_authority(tmp_path):
    from fparser.two.utils import walk

    from compiler.frontend.call_bindings import resolve_original_function

    analysis, _caller, _selected, _roots, _modules = fixture(tmp_path, WHOLE.replace("real(mod(i,2),8)", "norm2(vec)"), sealed=True)
    routine = analysis.routines["regions::touch"]
    expression, = [node for node in walk(routine.execution) if type(node).__name__ == "Intrinsic_Function_Reference"
                   and str(node.items[0]).lower() == "norm2"]
    with pytest.raises(CompilationError):
        resolve_original_function(analysis, analysis.source_scope_for(expression, routine.scope), expression)


@pytest.mark.parametrize("sealed", [False, True])
def test_late_original_prelude_name_cannot_change_after_complete_effect_validation(tmp_path, monkeypatch, sealed):
    from fparser.two.utils import walk

    from compiler.frontend.storage_mutation_effects import StorageMutationEffects

    analysis, caller, _selected, _roots, _modules = fixture(tmp_path, WHOLE, sealed=sealed)
    call, = caller.execution.content
    proof = original_record_call(analysis, caller, call, (analysis.modules["regions"].bindings["field"],))
    callee = proof._region._roles[1]
    target, = [node.items[0] for node in walk(callee.execution) if type(node).__name__ == "Assignment_Stmt"
                and str(node.items[0]).lower() == "delta"]
    if sealed:
        original_facts = proof.public()
        with pytest.raises(CompilationError, match="immutable"):
            target.string = "changed_delta"
        assert str(target).lower() == "delta"
        assert proof.validate() is proof
        assert proof.public() == original_facts
        return
    original = StorageMutationEffects.validate
    reached = []

    def changed(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if self is proof._storage and not reached:
            reached.append(target)
            object.__setattr__(target, "string", "changed_delta")
        return result

    monkeypatch.setattr(StorageMutationEffects, "validate", changed)
    with pytest.raises(CompilationError):
        proof.validate()
    assert reached
