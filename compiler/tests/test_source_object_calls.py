"""Fixed source objects preserve original type/storage without a numeric ABI."""

from copy import copy

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.frontend.call_bindings import resolve_source_call
from compiler.frontend.source_effects import SOURCE_SUMMARY_VERSION, SourceEffects
from compiler.frontend.source_objects import SOURCE_OBJECT_VERSION
from compiler.ir import CompilationError


def inputs(tmp_path, *sources, **options):
    paths = []
    for index, text in enumerate(sources):
        path = tmp_path / f"unit{index}.f90"
        path.write_text(text)
        paths.append(path)
    return SourceEffects(paths, **options)


def resolve(analysis, procedure="clients::step", ordinal=0):
    routine = analysis.routines[procedure]
    calls = [node for node in walk(routine.execution) if type(node).__name__ == "Call_Stmt"]
    return resolve_source_call(analysis, routine.scope, calls[ordinal])


def same_module(*, component="real(8)::value=0", attributes="", formal="type(state),intent(inout)::x",
                actual="type(state)::s", body="x%value=x%value+1", call="call leaf(x=s)"):
    return f"""module clients
implicit none
type{attributes}::state
{component}
end type
contains
subroutine leaf(x)
{formal}
{body}
end subroutine
subroutine step()
{actual}
{call}
end subroutine
end module
"""


def test_fixed_object_call_keeps_original_keyword_storage_and_component_effects(tmp_path):
    analysis = inputs(tmp_path, same_module())
    call = resolve(analysis)
    mapping = call.public()["resource_mappings"][0]
    assert call.render_original_arguments() == ("x = s",)
    assert mapping["storage"] == "whole"
    assert mapping["resource"] == "clients::step::s"
    proof = mapping["source_object"]
    assert proof["schema_version"] == SOURCE_OBJECT_VERSION == 1
    assert proof["declaring_type"] == "clients::state"
    assert proof["source_only"]
    assert not proof["numerical_abi"]
    assert not proof["whole_object_copy"]
    assert not proof["native_continuation_authority"]
    summary = analysis.summarize("clients::step")
    assert summary["complete"], summary["reasons"]
    assert {effect["resource"] for effect in summary["ordered_effects"]} == {"clients::step::s%value"}
    assert not summary["cloneable"]
    assert not summary["native_completion"]["available"]
    assert "do not prove native continuation" in summary["native_completion"]["reason"]
    # Public mutation is not a source proof mutation.
    proof["type_definition"]["fields"].clear()
    assert call.public()["resource_mappings"][0]["source_object"]["type_definition"]["fields"]


def test_reached_projection_preserves_original_object_storage_and_type(tmp_path):
    analysis = inputs(tmp_path, same_module().replace("subroutine step()", "subroutine step(s)")
                      .replace("type(state)::s", "type(state),intent(inout)::s"))
    call = next(node for node in walk(analysis.routines["clients::step"].execution)
                if type(node).__name__ == "Call_Stmt")
    summary = analysis.segment_summary("clients::step", (call,))
    assert summary["complete"], summary["reasons"]
    assert {effect["resource"] for effect in summary["ordered_effects"]} == {"argument::s%value"}
    assert not summary["native_completion"]["available"]


def test_original_object_descriptor_inquiries_do_not_require_value_copy_semantics(tmp_path):
    analysis = inputs(tmp_path, """module clients
type::state
real(8)::value
end type
contains
subroutine inspect(items,n)
type(state),allocatable,intent(in)::items(:)
integer,intent(out)::n
n=0
if(allocated(items)) n=size(items)
end subroutine
end module
""")
    summary = analysis.summarize("clients::inspect")
    assert summary["complete"], summary["reasons"]
    effects = [operation for operation in summary["operations"]
               if operation.get("resource") == "argument::items"]
    assert [operation["kind"] for operation in effects] == ["descriptor_read", "descriptor_read"]
    assert effects[0]["guard"] == ()
    assert effects[1]["guard"] == ("ALLOCATED(items)",)


def test_renamed_multifile_type_is_resolved_in_each_original_declaring_scope(tmp_path):
    analysis = inputs(tmp_path, """module types
type::state
real(8)::value
end type
end module
""", """module workers
use types,only:formal_type=>state
contains
subroutine leaf(x)
type(formal_type),intent(inout)::x
x%value=x%value+1
end subroutine
end module
""", """module clients
use types,only:actual_type=>state
use workers,only:renamed=>leaf
contains
subroutine step(s)
type(actual_type),intent(inout)::s
call renamed(x=s)
end subroutine
end module
""")
    call = resolve(analysis)
    assert call.procedure == "workers::leaf"
    assert call.mappings[0].formal_binding.dtype != call.mappings[0].binding.dtype
    assert call.mappings[0].source_object.public()["declaring_type"] == "types::state"
    summary = analysis.summarize("clients::step")
    assert summary["complete"], summary["reasons"]
    assert {effect["resource"] for effect in summary["ordered_effects"]} == {"argument::s%value"}


@pytest.mark.parametrize("actual_type", ["left_type", "right_type"])
def test_same_spelling_distinct_modules_select_generic_by_canonical_type(tmp_path, actual_type):
    analysis = inputs(tmp_path, """module first
type::state
real(8)::value
end type
end module
module second
type::state
real(8)::value
end type
end module
""", f"""module clients
use first,only:left_type=>state
use second,only:right_type=>state
interface apply
module procedure left,right
end interface
contains
subroutine left(x)
type(left_type),intent(inout)::x
x%value=1
end subroutine
subroutine right(x)
type(right_type),intent(inout)::x
x%value=2
end subroutine
subroutine step(s)
type({actual_type}),intent(inout)::s
call apply(s)
end subroutine
end module
""")
    assert resolve(analysis).procedure == "clients::" + ("left" if actual_type == "left_type" else "right")


def test_identically_spelled_different_types_are_not_structurally_interchangeable(tmp_path):
    analysis = inputs(tmp_path, """module first
type::state
real(8)::value
end type
contains
subroutine leaf(x)
type(state),intent(in)::x
end subroutine
end module
""", """module clients
use first,only:leaf
type::state
real(8)::value
end type
contains
subroutine step(s)
type(state),intent(in)::s
call leaf(s)
end subroutine
end module
""")
    with pytest.raises(CompilationError, match="canonical declaring type mismatch"):
        resolve(analysis)


def test_diamond_reexports_share_one_original_declaring_type(tmp_path):
    analysis = inputs(tmp_path, """module types
type::state
real(8)::value
end type
end module
module left
use types,only:state
end module
module right
use types,only:state
end module
""", """module clients
use left
use right
contains
subroutine leaf(x)
type(state),intent(in)::x
end subroutine
subroutine step(s)
type(state),intent(in)::s
call leaf(s)
end subroutine
end module
""")
    assert resolve(analysis).mappings[0].source_object.public()["declaring_type"] == "types::state"


def test_nested_fixed_fields_numeric_out_and_original_guards_compose(tmp_path):
    analysis = inputs(tmp_path, """module clients
integer,parameter::width=3
type::inner
integer::count
real(8)::values(-1:width)
end type
type::state
type(inner)::nested
logical::ready
end type
contains
subroutine store(y)
real(8),intent(out)::y(-1:3)
y=1
end subroutine
subroutine leaf(x,flag)
type(state),intent(inout)::x
logical,intent(in)::flag
if(flag) call store(x%nested%values)
end subroutine
subroutine step(s,flag)
type(state),intent(inout)::s
logical,intent(in)::flag
if(flag) call leaf(flag=flag,x=s)
end subroutine
end module
""")
    summary = analysis.summarize("clients::step")
    assert summary["complete"], summary["reasons"]
    effects = [effect for effect in summary["ordered_effects"] if effect["resource"] == "argument::s%nested%values"]
    assert {effect["kind"] for effect in effects} == {"definition_change", "overwrite"}
    assert all(len(effect["guard_frames"]) == 2 for effect in effects)
    assert effects[0]["view_chain"][0]["source_object"]["source_only"]
    fields = resolve(analysis).mappings[0].source_object.public()["type_definition"]["fields"]
    assert fields[0]["type"]["fields"][1]["shape"] == [{"lower": -1, "upper": 3}]


@pytest.mark.parametrize("declaration", [
    "type(state),pointer::s", "type(state),allocatable::s", "type(state),optional::s",
    "type(state),volatile::s", "type(state),asynchronous::s", "type(state)::s(2)",
])
def test_uncertain_or_array_actual_storage_is_rejected_even_when_unused(tmp_path, declaration):
    # OPTIONAL is legal only on a dummy; all others also work as dummy declarations.
    text = same_module(actual=declaration, body="").replace("subroutine step()", "subroutine step(s)")
    analysis = inputs(tmp_path, text)
    with pytest.raises(CompilationError, match="fixed nonoptional scalar original storage"):
        resolve(analysis)


@pytest.mark.parametrize("formal", [
    "type(state),intent(out)::x", "type(state),value,intent(in)::x",
    "type(state),optional,intent(in)::x", "type(state),pointer,intent(in)::x",
    "type(state),allocatable,intent(in)::x", "type(state),intent(in)::x(2)",
    "class(state),intent(in)::x",
])
def test_unsupported_formal_semantics_do_not_disappear_when_unused(tmp_path, formal):
    analysis = inputs(tmp_path, same_module(formal=formal, body=""))
    with pytest.raises(CompilationError):
        resolve(analysis)


@pytest.mark.parametrize("component", [
    "real(8),allocatable::value(:)", "real(8),pointer::value", "class(*),allocatable::value",
    "character(4)::value", "complex(8)::value", "procedure(),pointer,nopass::value",
    "real(8)::value\ncontains\nprocedure,nopass::method=>operation",
    "real(8)::value\ncontains\nfinal::finish",
])
def test_complete_type_checks_reject_unsupported_unused_components(tmp_path, component):
    text = same_module(component=component, body="")
    if "finish" in component:
        text = text.replace("subroutine leaf", "subroutine finish(x)\ntype(state)::x\nend subroutine\nsubroutine leaf", 1)
    if "operation" in component:
        text = text.replace("subroutine leaf", "subroutine operation()\nend subroutine\nsubroutine leaf", 1)
    analysis = inputs(tmp_path, text)
    with pytest.raises(CompilationError):
        resolve(analysis)


@pytest.mark.parametrize("definition", [
    "type::base\nreal(8)::value\nend type\ntype,extends(base)::state\ninteger::extra\nend type",
    "type::state(k)\ninteger,kind::k\nreal(k)::value\nend type",
    "type::inner\nreal(8),allocatable::value(:)\nend type\ntype::state\ntype(inner)::nested\nend type",
    "type::inner\nreal(8)::value\nend type\ntype::state\ntype(inner)::nested(2)\nend type",
])
def test_unsupported_type_features_are_rejected_through_the_entire_type(tmp_path, definition):
    text = f"""module clients
{definition}
contains
subroutine leaf(x)
type(state),intent(in)::x
end subroutine
subroutine step(s)
type(state),intent(in)::s
call leaf(s)
end subroutine
end module
"""
    with pytest.raises(CompilationError):
        resolve(inputs(tmp_path, text))


@pytest.mark.parametrize("actual", ["state(1.0_8)", "s%value"])
def test_expressions_and_component_actuals_do_not_gain_object_mapping(tmp_path, actual):
    analysis = inputs(tmp_path, same_module(call=f"call leaf({actual})"))
    with pytest.raises(CompilationError, match="whole scalar names"):
        resolve(analysis)


def test_whole_object_assignment_with_defined_assignment_is_not_a_field_effect(tmp_path):
    text = same_module(body="x=x")
    text = text.replace("contains\nsubroutine leaf", "interface assignment(=)\nmodule procedure assign\nend interface\ncontains\nsubroutine assign(a,b)\ntype(state),intent(inout)::a\ntype(state),intent(in)::b\ncall opaque()\nend subroutine\nsubroutine leaf", 1)
    summary = inputs(tmp_path, text).summarize("clients::step")
    assert not summary["complete"]
    assert any("whole derived-object effects" in reason for reason in inputs(tmp_path, text).summarize("clients::leaf")["reasons"])


@pytest.mark.parametrize("mutation", ["source", "schema", "imports", "binding", "kinds"])
def test_original_source_and_declaration_authority_cannot_be_replaced(tmp_path, mutation):
    analysis = inputs(tmp_path, same_module())
    resolve(analysis)
    if mutation == "source":
        path = analysis.inputs.paths[0]
        path.write_text(path.read_text().replace("real(8)::value", "real(4)::value"))
    elif mutation == "schema":
        schema = analysis.modules["clients"].component_types["state"]
        analysis.modules["clients"].component_types["state"] = copy(schema)
    elif mutation == "imports":
        analysis.routines["clients::leaf"].scope.imports["state"] = ("other", "state")
    elif mutation == "binding":
        analysis.routines["clients::leaf"].scope.bindings["x"].attributes = frozenset({"intent", "optional"})
    else:
        analysis.modules["clients"].kinds.values["unrelated"] = 7
    with pytest.raises(CompilationError, match="source changed|schema changed|scope changed|storage declaration|nonoptional"):
        resolve(analysis)


def test_copied_call_syntax_cannot_issue_source_object_mapping(tmp_path):
    analysis = inputs(tmp_path, same_module())
    routine = analysis.routines["clients::step"]
    with pytest.raises(CompilationError, match="exact unchanged original call"):
        resolve_source_call(analysis, routine.scope, F.Call_Stmt("call leaf(x=s)"))


def test_source_object_proofs_are_rebuilt_and_versioned_not_imported_as_cache_authority(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    first = inputs(tmp_path, same_module(), summary_cache=cache)
    before = first.report("clients::step")
    import compiler.frontend.source_objects as objects
    original, rebuilt = objects.source_object_mapping, []

    def checked(*args):
        rebuilt.append(args)
        return original(*args)

    monkeypatch.setattr(objects, "source_object_mapping", checked)
    second = SourceEffects(first.inputs.paths, summary_cache=cache)
    after = second.report("clients::step")
    assert before["complete"]
    assert after["complete"]
    assert after["summary_version"] == SOURCE_SUMMARY_VERSION == 16
    assert rebuilt  # Unchanged numeric/component leaf summaries may still be cached.
    assert first._summary_authority()["source_object_version"] == SOURCE_OBJECT_VERSION
    assert first.summarize("clients::step")["summary_identity"] == second.summarize("clients::step")["summary_identity"]


def test_fixed_type_proof_respects_existing_operation_budget(tmp_path):
    analysis = inputs(tmp_path, same_module(component="\n".join(f"real(8)::x{i}" for i in range(8)), body=""), operations=4)
    with pytest.raises(CompilationError, match="budget exhausted"):
        resolve(analysis)
