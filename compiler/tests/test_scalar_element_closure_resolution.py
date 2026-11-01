"""Original typed overloads and lexical nesting preserve scalar associations."""

from copy import copy

import pytest

from compiler.frontend import scalar_elements
from compiler.frontend.native_call_coordinates import prove_native_call_coordinates
from compiler.ir import CompilationError
from compiler.tests.test_scalar_element_inputs import NUMERICAL_SOURCE, scalar_element_call


def closure(tmp_path, source):
    paths, analysis, caller, original, resolved = scalar_element_call(tmp_path, numerical_source=source)
    result, _calls = scalar_elements.check_scalar_element_closure(analysis, resolved.procedure, resolved.mappings)
    return paths, analysis, caller, original, resolved, result


def overloaded_source(*, selected_write=False, ambiguous=False, recursive=False):
    interface = "interface combine\nmodule procedure combine3,combine4,combine_scalar3\nend interface\n"
    helpers = """subroutine combine3(to,from)
real(8),intent(inout)::to(:,:,:)
real(8),intent(in)::from(:,:,:)
to=to+from
end subroutine
subroutine combine4(to,from)
real(8),intent(inout)::to(:,:,:,:)
real(8),intent(in)::from(:,:,:,:)
call combine(to(:,:,:,1),from(:,:,:,1))
end subroutine
subroutine combine_scalar3(to,from)
real(8),intent(inout)::to(:,:,:)
real(8),intent(in)::from
to=to+from
end subroutine
"""
    source = NUMERICAL_SOURCE.replace("contains\n", interface + "contains\n", 1)
    source = source.replace("bridge_visits=bridge_visits+1", "real(8)::work(1,1,1,1),input(1,1,1,1)\n"
                            "work=1.0_8\ninput=2.0_8\ncall combine(work,input)\nbridge_visits=bridge_visits+1")
    if selected_write:
        helpers = helpers.replace("subroutine combine3(to,from)\n", "subroutine combine3(to,from)\n"
                                  "use tide_values,only:weights\n").replace("to=to+from\n", "weights(-3)=0.0_8\nto=to+from\n", 1)
    if ambiguous:
        source = source.replace("combine_scalar3\nend interface", "combine_scalar3,other3\nend interface")
        helpers += "subroutine other3(to,from)\nreal(8),intent(inout)::to(:,:,:)\nreal(8),intent(in)::from(:,:,:)\nend subroutine\n"
    if recursive:
        helpers = helpers.replace("call combine(to(:,:,:,1),from(:,:,:,1))", "call combine(to,from)")
    return source.replace("end module\n", helpers + "end module\n")


def test_rank_four_generic_calls_rank_three_without_false_recursion(tmp_path):
    _paths, analysis, caller, original, _resolved, record = closure(tmp_path, overloaded_source())
    assert record["version"] == 2
    names = [item["procedure"] for item in record["procedures"]]
    assert names.count("tide_numerics::combine4") == 1
    assert names.count("tide_numerics::combine3") == 1
    assert "tide_numerics::combine_scalar3" not in names
    assert record["effect_authority"] is record["definition_authority"] is record["completion_authority"] is False
    proof = prove_native_call_coordinates(analysis, caller.qualified, original)
    assert proof.public()["scalar_element_closure"] == record


@pytest.mark.parametrize(("change", "match"), [("selected_write", "backing storage"), ("ambiguous", "ambiguous"),
                                           ("recursive", "call depth or recursion")])
def test_selected_generic_body_ambiguity_and_true_recursion_remain_barriers(tmp_path, change, match):
    with pytest.raises(CompilationError, match=match):
        closure(tmp_path, overloaded_source(**{change: True}))


def nested_source(*, hidden_write=False, count=12):
    statement = "y(i)=scale*x(i)+real(i,8)"
    nested = "\n".join(["if(n>0) then"] * count + [statement] + ["endif"] * count)
    if hidden_write:
        nested = nested.replace(statement, statement + "\nelse\nweights(-3)=0.0_8")
    source = NUMERICAL_SOURCE.replace(statement, nested)
    if hidden_write:
        source = source.replace("only:selector", "only:selector,weights")
    return source


def test_lexical_nesting_does_not_consume_interprocedural_depth(tmp_path):
    _paths, analysis, caller, original, _resolved, record = closure(tmp_path, nested_source())
    assert analysis.depth_limit == 8
    assert record["operations"] < analysis.operation_limit
    assert prove_native_call_coordinates(analysis, caller.qualified, original).public()["scalar_element_closure"] == record


def test_deep_unreached_alternative_still_scanned(tmp_path):
    with pytest.raises(CompilationError, match="backing storage"):
        closure(tmp_path, nested_source(hidden_write=True))


@pytest.mark.parametrize("which", ["node", "operation", "procedure", "source_depth"])
def test_distinct_budgets_remain_hard(tmp_path, monkeypatch, which):
    _paths, analysis, _caller, _original, resolved = scalar_element_call(tmp_path, numerical_source=nested_source())
    if which == "node":
        monkeypatch.setattr(scalar_elements, "_SCAN_LIMIT", 100)
        message = "source scan budget"
    elif which == "operation":
        analysis.operation_limit = 24
        message = "operation budget"
    elif which == "procedure":
        analysis.procedure_limit = 1
        message = "procedure budget"
    else:
        analysis.depth_limit = 0
        message = "native source function resolution budget exhausted"
    with pytest.raises(CompilationError, match=message):
        scalar_elements.check_scalar_element_closure(analysis, resolved.procedure, resolved.mappings)


@pytest.mark.parametrize("change", ["missing", "copy", "foreign"])
def test_original_association_registration_is_rechecked_before_publication(tmp_path, monkeypatch, change):
    _paths, analysis, _caller, _original, resolved = scalar_element_call(tmp_path / "one")
    _paths, other, _caller, _original, _resolved = scalar_element_call(tmp_path / "two")
    real = scalar_elements.resolve_original_call
    foreign = next(node for node in other.routines["tide_numerics::bridge"].execution.content
                   if type(node).__name__ == "Call_Stmt")
    otherproof = real(other, other.routines["tide_numerics::bridge"].scope, foreign)

    def replaced(owner, scope, node):
        proof = real(owner, scope, node)
        key = id(owner), id(node)
        if change == "missing":
            del owner._original_call_associations[key]
        else:
            owner._original_call_associations[key] = copy(proof) if change == "copy" else otherproof
        return proof

    monkeypatch.setattr(scalar_elements, "resolve_original_call", replaced)
    with pytest.raises(CompilationError, match="exact issued source association"):
        scalar_elements.check_scalar_element_closure(analysis, resolved.procedure, resolved.mappings)


def test_changed_original_callee_file_rejected_at_artifact_publication(tmp_path):
    from compiler.tests.test_scalar_element_inputs import scalar_element_fixture

    paths, builder, _proof, _originals = scalar_element_fixture(tmp_path, numerical_source=overloaded_source())
    analysis = builder.analysis
    caller = analysis.routines["tide_numerics::bridge"]
    call = next(node for node in caller.execution.content if type(node).__name__ == "Call_Stmt")
    scalar_elements.resolve_original_call(analysis, caller.scope, call)
    paths[1].write_text(paths[1].read_text().replace("to=to+from", "to=to-from"))
    with pytest.raises(CompilationError):
        analysis.verify_publication()


@pytest.mark.parametrize("consumer", ["validate", "issuer"])
@pytest.mark.parametrize("change", ["missing", "copy"])
def test_coordinate_token_does_not_heal_missing_nested_association(tmp_path, consumer, change):
    _paths, analysis, caller, original, _resolved = scalar_element_call(tmp_path)
    proof = prove_native_call_coordinates(analysis, caller.qualified, original)
    assert proof._element_calls
    nested = proof._element_calls[0]
    key = id(analysis), id(nested._original)
    if change == "missing":
        del analysis._original_call_associations[key]
    else:
        analysis._original_call_associations[key] = copy(nested)
    operation = proof.validate if consumer == "validate" else prove_native_call_coordinates
    with pytest.raises(CompilationError, match="exact issued source association"):
        operation(analysis, caller.qualified, original)
    if change == "missing":
        assert key not in analysis._original_call_associations
    else:
        assert analysis._original_call_associations[key] is not nested


def test_interprocedural_depth_is_still_bounded(tmp_path):
    source = NUMERICAL_SOURCE.replace("selector=1", "selector=1\ncall stage0(n)")
    helpers = []
    for index in range(10):
        body = f"call stage{index + 1}(n)\n" if index < 9 else ""
        helpers.append(f"subroutine stage{index}(n)\ninteger,intent(in)::n\n{body}end subroutine\n")
    source = source.replace("end module\n", "".join(helpers) + "end module\n")
    with pytest.raises(CompilationError, match="call depth or recursion budget"):
        closure(tmp_path, source)


@pytest.mark.parametrize("hidden_write", [False, True])
def test_original_generic_function_resolves_only_its_typed_target(tmp_path, hidden_write):
    source = NUMERICAL_SOURCE.replace("contains\n", "interface factor\nmodule procedure real_factor,integer_factor\n"
                                    "end interface\ncontains\n", 1)
    source = source.replace("y(i)=scale*x(i)+real(i,8)", "y(i)=factor(scale)*x(i)+real(i,8)")
    helpers = """real(8) function real_factor(value)
real(8),intent(in)::value
real_factor=value
end function
integer function integer_factor(value)
integer,intent(in)::value
integer_factor=value
end function
"""
    if hidden_write:
        helpers = helpers.replace("real(8),intent(in)::value", "use tide_values,only:weights\nreal(8),intent(in)::value")
        helpers = helpers.replace("real_factor=value", "weights(-3)=0.0_8\nreal_factor=value")
    source = source.replace("end module\n", helpers + "end module\n")
    if hidden_write:
        with pytest.raises(CompilationError, match="backing storage"):
            closure(tmp_path, source)
    else:
        *_prefix, record = closure(tmp_path, source)
        assert "tide_numerics::real_factor" in {item["procedure"] for item in record["procedures"]}
        assert "tide_numerics::integer_factor" not in {item["procedure"] for item in record["procedures"]}
