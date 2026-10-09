"""Call mappings retain source order, guards and logical section coordinates."""

import re
from dataclasses import FrozenInstanceError
from hashlib import sha256

import pytest
from fparser.two.utils import walk

from compiler.driver.options import CompilerOptions
from compiler.frontend.call_bindings import resolve_source_call
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import form_source_scopes


def write(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return path


def resolve(analysis, routine="clients::step", position=0):
    owner = analysis.routines[routine]
    calls = [node for node in walk(owner.execution) if type(node).__name__ == "Call_Stmt"]
    return resolve_source_call(analysis, owner.scope, calls[position])


def source(actuals, *, declarations="real(8)::a(-3:8,-2:9)\ninteger::n", formals=None, body="field=real(value,8)"):
    formals = formals or "real(8),intent(inout)::field(:,:)\ninteger,intent(in)::value"
    return f"""module clients
implicit none
contains
subroutine leaf(field,value)
{formals}
{body}
end subroutine
subroutine step()
{declarations}
call leaf({actuals})
end subroutine
end module
"""


def test_keyword_formal_order_and_original_syntax_are_separate(tmp_path):
    analysis = SourceEffects([write(tmp_path, "calls.f90", source("value=n,field=a"))])
    call = resolve(analysis)
    assert call.procedure == "clients::leaf"
    assert tuple(map(str, call.actuals)) == ("a", "n")
    assert call.formals == ("field", "value")
    assert call.resource_mapping == {"argument::field": "clients::step::a", "argument::value": "clients::step::n"}
    assert call.render_original_arguments() == ("value = n", "field = a")
    assert call.render_original_arguments(("array_view", "scalar_view")) == (
        "value = scalar_view", "field = array_view")
    public = call.public()
    assert [item["formal"] for item in public["original_arguments"]] == ["value", "field"]
    public["resource_mapping"]["argument::field"] = "changed"
    assert call.resource_mapping["argument::field"] == "clients::step::a"
    with pytest.raises(FrozenInstanceError):
        call.procedure = "changed"


def test_mixed_positional_and_keyword_arguments(tmp_path):
    analysis = SourceEffects([write(tmp_path, "calls.f90", source("a,value=n"))])
    call = resolve(analysis)
    assert call.render_original_arguments(("view", "extent")) == ("view", "value = extent")


@pytest.mark.parametrize(("actuals", "reason"), [
    ("a,value=n,value=2", "duplicate source-call argument"),
    ("array=a,value=n", "unknown source-call keyword"),
    ("field=a,n", "positional source-call argument follows a keyword"),
    ("field=a", "required source-call argument is missing"),
    ("a,n,2", "too many source-call arguments"),
    ("n,a", "type, kind or rank mismatch"),
])
def test_invalid_keyword_or_signature_mappings_are_boundaries(tmp_path, actuals, reason):
    analysis = SourceEffects([write(tmp_path, "calls.f90", source(actuals))])
    with pytest.raises(CompilationError, match=reason):
        resolve(analysis)


def test_renamed_multifile_generic_is_selected_by_keyword_signature(tmp_path):
    library = write(tmp_path, "library.f90", """module workers
interface apply
module procedure real_leaf,integer_leaf
end interface
contains
subroutine real_leaf(field,value)
real(8),intent(inout)::field(:,:)
real(8),intent(in)::value
field=value
end subroutine
subroutine integer_leaf(field,value)
real(8),intent(inout)::field(:,:)
integer,intent(in)::value
field=real(value,8)
end subroutine
end module
""")
    caller = write(tmp_path, "caller.f90", """module clients
use workers,only:renamed=>apply
contains
subroutine step(a,n)
real(8)::a(-2:,-3:)
integer::n
call renamed(value=n,field=a)
end subroutine
end module
""")
    call = resolve(SourceEffects([caller, library]))
    assert call.procedure == "workers::integer_leaf"
    assert call.target == "renamed"
    assert call.resource_mapping["argument::field"] == "argument::a"


def test_omitted_and_forwarded_optional_arguments_keep_presence(tmp_path):
    path = write(tmp_path, "optional.f90", """module clients
contains
subroutine leaf(field,value)
real(8),intent(in)::field(:,:)
integer,optional,intent(in)::value
end subroutine
subroutine step(a,n)
real(8),intent(in)::a(:,:)
integer,optional,intent(in)::n
call leaf(field=a)
call leaf(value=n,field=a)
end subroutine
end module
""")
    analysis = SourceEffects([path])
    omitted, forwarded = resolve(analysis), resolve(analysis, position=1)
    assert omitted.actuals[1] is None
    assert omitted.mappings[1].presence == "omitted"
    assert omitted.public()["actual_arguments"] == ["a", None]
    assert omitted.render_original_arguments() == ("field = a",)
    assert forwarded.mappings[1].presence == "forwarded_optional"
    assert forwarded.public()["resource_mappings"][1]["requirements"]["presence_preserving_forwarding"]


def test_optional_actual_to_required_formal_remains_conservative(tmp_path):
    path = write(tmp_path, "optional.f90", source("a,n", declarations="real(8)::a(4,4)\ninteger,optional::n"))
    with pytest.raises(CompilationError, match="optional actual requires an optional callee formal"):
        resolve(SourceEffects([path]))


def test_readonly_allocatable_forwarding_keeps_original_descriptor_facts(tmp_path):
    path = write(tmp_path, "allocatable.f90", source("value=n,field=a", declarations="real(8),allocatable::a(:,:)\ninteger::n",
                                                        formals="real(8),allocatable,intent(in)::field(:,:)\ninteger,intent(in)::value", body="continue"))
    call = resolve(SourceEffects([path]))
    assert call.mappings[0].section is None
    public = call.mappings[0].public()
    assert public["requirements"]["original_allocation_descriptor"]
    assert "allocatable" in public["actual_descriptor"]["attributes"]
    assert public["resource"] == "clients::step::a"


def test_rectangular_actual_keeps_negative_logical_bounds_and_dependencies(tmp_path):
    path = write(tmp_path, "rectangle.f90", source("value=n,field=a(-2:n+1,lbound(a,2):ubound(a,2))"))
    call = resolve(SourceEffects([path]))
    section = call.mappings[0].section
    assert section.resource == "clients::step::a"
    assert len(section.axes) == 2
    assert section.axes[0].lower.kind == "unary"
    assert section.axes[0].lower.children[0].value == 2
    assert section.axes[0].upper.operator == "+"
    assert section.axes[1].lower.kind == "lbound"
    assert section.axes[1].upper.dimension == 2
    assert {(item.resource, item.kind) for item in section.dependencies} == {
        ("clients::step::n", "scalar_read"), ("clients::step::a", "descriptor_read")}
    public = section.public()
    assert "original caller logical indices" in public["coordinate_system"]
    assert "physical" not in public["axes"][0]
    assert public["bounds_evaluation"] == "at the original call under its original guards"


def test_omitted_rectangle_bounds_are_descriptor_dependencies(tmp_path):
    call = resolve(SourceEffects([write(tmp_path, "rectangle.f90", source("a(:, -2:),n"))]))
    section = call.mappings[0].section
    assert section.axes[0].lower.kind == "lbound"
    assert section.axes[0].upper.kind == "ubound"
    assert section.axes[1].upper.dimension == 2
    assert call.public()["resource_mapping"]["argument::field"] == "clients::step::a"


@pytest.mark.parametrize(("actual", "reason"), [
    ("a(1,:)", "type, kind or rank mismatch"),
    ("a([1,2],:)", "affine scalar arithmetic"),
    ("a(::2,:)", "requires unit stride"),
    ("a(::-1,:)", "requires unit stride"),
    ("a(:n*n,:)", "affine scalar arithmetic"),
    ("a(:unknown(n),:)", "affine scalar arithmetic"),
])
def test_unsupported_section_geometry_remains_a_boundary(tmp_path, actual, reason):
    analysis = SourceEffects([write(tmp_path, "rectangle.f90", source(actual + ",n"))])
    with pytest.raises(CompilationError, match=reason):
        resolve(analysis)


def test_rank_reduced_actual_retains_physical_axes_and_scalar_dependencies(tmp_path):
    text = source("field=a(:,n,:),value=n", declarations="real(8)::a(-3:8,-2:9,11:17)\ninteger::n")
    call = resolve(SourceEffects([write(tmp_path, "plane.f90", text)]))
    mapping = call.mappings[0]
    assert mapping.binding.rank == 3 and mapping.formal_binding.rank == 2
    assert mapping.section.logical_rank == 2
    assert [axis.scalar for axis in mapping.section.axes] == [False, True, False]
    assert mapping.section.axes[1].lower is mapping.section.axes[1].upper
    assert mapping.section.axes[1].lower.resource == "clients::step::n"
    record = mapping.section.public()
    assert record["rank"] == 3 and record["logical_rank"] == 2 and record["retained_axes"] == [0, 2]
    assert record["axes"][1]["kind"] == "scalar_coordinate"
    assert call.resource_mapping["argument::field"] == "clients::step::a"


def test_fixed_component_plane_and_bound_follow_original_component_resources(tmp_path):
    text = source("field=state%field(:,state%plane,:),value=n",
                  declarations="type(fields)::state\ninteger::n")
    text = text.replace("implicit none", "implicit none\ntype fields\nreal(8)::field(-3:8,-2:9,11:17)\ninteger::plane\nend type")
    call = resolve(SourceEffects([write(tmp_path, "components.f90", text)]))
    mapping = call.mappings[0]
    assert mapping.resource == "clients::step::state%field"
    assert mapping.section.logical_rank == 2
    assert mapping.section.axes[1].lower.resource == "clients::step::state%plane"
    assert any(item.resource == "clients::step::state%plane" and item.kind == "scalar_read"
               for item in mapping.section.dependencies)


def test_writable_expression_actual_is_rejected(tmp_path):
    path = write(tmp_path, "literal.f90", source("a,2", formals="real(8),intent(inout)::field(:,:)\ninteger,intent(inout)::value"))
    with pytest.raises(CompilationError, match="requires original storage"):
        resolve(SourceEffects([path]))


def test_direct_rectangular_execution_requires_a_borrowed_root_view(tmp_path):
    path = write(tmp_path, "rectangle.f90", """module clients
contains
subroutine leaf(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
a(i)=2*a(i)
enddo
end subroutine
subroutine step(a,n)
real(8),intent(inout)::a(-3:)
integer,intent(in)::n
call leaf(n=n,a=a(-2:n))
call leaf(n=n,a=a(-2:n))
end subroutine
end module
""")
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::a": {"storage": "stable", "initialized": "whole", "escapes": False,
                                         "allocation_changes": False}}}
    outputs, report = form_source_scopes([path], "clients::step", facts=facts, options=CompilerOptions(),
                                        config=OffloadConfig(policy="sections"))
    assert report["scope_count"] == 1, report["boundaries"]
    assert report["scopes"][0]["borrowed_views"]["abi_version"] == 2
    assert report["resolved_calls"][0]["resource_mappings"][0]["storage"] == "rectangle"
    assert report["source_edits"]
    assert any(path.endswith("_views/shared_entry.cu") for path in outputs)


def test_keyword_execution_uses_formal_workers_and_original_native_calls(tmp_path):
    path = write(tmp_path, "keyword_scope.f90", """module clients
contains
subroutine leaf(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
a(i)=2*a(i)
enddo
end subroutine
subroutine inspect(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
if(n>0) a(1)=sum(a)
end subroutine
subroutine step(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
call leaf(n=n,a=a)
if(n>0) call inspect(n=n,a=a)
call leaf(a=a,n=n)
end subroutine
end module
""")
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::a": {"storage": "stable", "initialized": "whole", "escapes": False,
                                         "allocation_changes": False}}}
    outputs, report = form_source_scopes([path], "clients::step", facts=facts, options=CompilerOptions(),
                                        config=OffloadConfig(policy="sections"))
    assert report["scope_count"] == 1
    edited = "\n".join(value for name, value in outputs.items() if name.startswith("sources/"))
    continuous = re.sub(r"&\s*\n\s*&?", "", edited.lower())
    native_calls = re.findall(r"call inspect\(([^)]*)\)", continuous)
    assert native_calls
    assert all([argument.split("=")[0].strip() for argument in call.split(",")] == ["n", "a"]
               for call in native_calls)
    assert any(item["procedure"] == "clients::inspect" for item in report["resolved_calls"])


def test_diamond_calls_preserve_distinct_caller_sections_and_nested_out_events(tmp_path):
    leaf = write(tmp_path, "leaf.f90", """module leaf_owner
contains
subroutine make(field)
real(8),intent(out)::field(-2:,:)
field=4.d0
end subroutine
end module
""")
    branches = write(tmp_path, "branches.f90", """module branches
use leaf_owner,only:renamed=>make
contains
subroutine left(field)
real(8),intent(inout)::field(-3:,:)
call renamed(field=field(-2:,:))
end subroutine
subroutine right(field)
real(8),intent(inout)::field(:,:)
call renamed(field=field(2:,:))
end subroutine
end module
""")
    clients = write(tmp_path, "clients.f90", """module clients
use branches,only:l=>left,r=>right
contains
subroutine middle(field)
real(8),intent(inout)::field(:,:)
field(1,1)=field(1,1)+1.d0
end subroutine
subroutine step(field)
real(8),intent(inout)::field(-7:,-3:)
call l(field=field(-6:,:))
call middle(field(-5:,:))
call r(field=field(-4:,:))
call l(field=field(-3:,:))
end subroutine
end module
""")
    analysis = SourceEffects([clients, branches, leaf])
    calls = [resolve(analysis, position=index) for index in range(4)]
    assert [call.procedure for call in calls] == ["branches::left", "clients::middle", "branches::right", "branches::left"]
    assert all(call.resource_mapping == {"argument::field": "argument::field"} for call in calls)
    assert [str(call.mappings[0].section.axes[0].lower.node) for call in calls] == ["- 6", "- 5", "- 4", "- 3"]
    left, right = resolve(analysis, "branches::left"), resolve(analysis, "branches::right")
    assert left.procedure == right.procedure == "leaf_owner::make"
    assert left.mappings[0].formal_binding.lower_bounds == ("- 2", "1")
    summary = analysis.summarize("clients::step")
    assert summary["complete"], summary["reasons"]
    assert analysis.summarize("leaf_owner::make")["definition_changes"] == ["argument::field"]
    child_calls = [operation for operation in summary["operations"] if operation["kind"] == "call"]
    assert len(child_calls) == 4
    assert all(operation["resource_mapping"] == {"argument::field": "argument::field"} for operation in child_calls)
    assert all(operation["resource_mappings"][0]["storage"] == "rectangle" for operation in child_calls)


def test_signed_constant_coefficients_remain_affine(tmp_path):
    call = resolve(SourceEffects([write(tmp_path, "rectangle.f90", source("a(-2*n:n+3,:),n"))]))
    lower = call.mappings[0].section.axes[0].lower
    assert lower.operator == "-"
    assert lower.children[0].operator == "*"
    assert lower.children[0].children[0].value == 2
    assert lower.children[0].children[1].resource == "clients::step::n"


def test_mixed_keyword_array_inquiry_retains_descriptor_identity(tmp_path):
    call = resolve(SourceEffects([write(tmp_path, "rectangle.f90", source("a(:,lbound(a,dim=2):ubound(dim=2,array=a)),n"))]))
    section = call.mappings[0].section
    assert section.axes[1].lower.dimension == section.axes[1].upper.dimension == 2
    assert section.axes[1].lower.resource == "clients::step::a"


def test_optional_nondescriptor_formal_has_allocation_dependent_presence(tmp_path):
    path = write(tmp_path, "optional_allocation.f90", source("a,n", declarations="real(8),allocatable::a(:,:)\ninteger::n",
                                                            formals="real(8),optional,intent(in)::field(:,:)\ninteger,intent(in)::value", body="continue"))
    mapping = resolve(SourceEffects([path])).mappings[0]
    assert mapping.presence == "allocation_dependent"
    assert mapping.public()["requirements"]["descriptor_dependent_presence"]


def test_partial_generic_interface_cannot_select_available_overload(tmp_path):
    path = write(tmp_path, "partial.f90", """module clients
interface choose
module procedure available,unavailable
end interface
contains
subroutine available(value)
integer,intent(in)::value
end subroutine
subroutine step(value)
integer,intent(in)::value
call choose(value=value)
end subroutine
end module
""")
    analysis = SourceEffects([path])
    with pytest.raises(CompilationError, match="interface closure unavailable: clients::unavailable"):
        resolve(analysis)


@pytest.mark.parametrize(("declaration", "call", "leaf", "reason"), [
    ("real(8),intent(inout)::a(:)", "call inner(a(2:))", "real(8),intent(out)::a(:)\na=7.d0", "canonical root views"),
    ("real(8),intent(inout)::a(:)", "call inner(a)", "real(8),intent(out)::a(2)\na=7.d0", "explicit dummy extents"),
])
def test_complete_nested_summaries_do_not_enable_unsupported_native_views(tmp_path, declaration, call, leaf, reason):
    inner_arguments = "a,offset" if "optional" in leaf else "a"
    path = write(tmp_path, "nested_native.f90", f"""module clients
contains
subroutine gpu(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
a(i)=2*a(i)
enddo
end subroutine
subroutine inner({inner_arguments})
{leaf}
end subroutine
subroutine middle(a)
{declaration}
a(1)=a(1)+1.d0
{call}
end subroutine
subroutine step(a,n)
{declaration}
integer,intent(in)::n
call gpu(a,n)
call middle(a)
call gpu(a,n)
end subroutine
end module
""")
    summary = SourceEffects([path]).summarize("clients::middle")
    assert summary["complete"], summary["reasons"]
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::a": {"storage": "stable", "initialized": "whole", "escapes": False,
                                         "allocation_changes": False}}}
    _outputs, report = form_source_scopes([path], "clients::step", facts=facts, options=CompilerOptions(),
                                         config=OffloadConfig(policy="sections"))
    assert any(reason in boundary["reason"] for boundary in report["boundaries"])
    assert all("clients::middle" not in scope["calls"] for scope in report["scopes"])


def test_native_nested_call_retains_its_hidden_readonly_allocation_descriptor(tmp_path):
    path = write(tmp_path, "hidden_descriptor.f90", """module clients
real(8),allocatable::hidden(:)
contains
subroutine gpu(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
a(i)=2*a(i)
enddo
end subroutine
subroutine inner(field)
real(8),allocatable,intent(in)::field(:)
integer::extent
extent=size(field)
end subroutine
subroutine middle(a)
real(8),intent(inout)::a(:)
a(1)=a(1)+1.d0
call inner(hidden)
end subroutine
subroutine step(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
call gpu(a,n)
call middle(a)
call gpu(a,n)
end subroutine
end module
""")
    analysis = SourceEffects([path])
    analysis.authorize_stable_module_allocatables({"clients::hidden"})
    summary = analysis.summarize("clients::middle")
    assert summary["complete"], summary["reasons"]
    capture = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::a": capture, "clients::hidden": capture}}
    _outputs, report = form_source_scopes([path], "clients::step", facts=facts, options=CompilerOptions(),
                                         config=OffloadConfig(policy="sections"))
    assert report["scope_count"] == 1, report["boundaries"]
    assert {resource["resource"] for resource in report["scopes"][0]["resources"]} == {"argument::a"}
    assert report["native_effects"]["complete"]
    assert report["scopes"][0]["calls"] == ["clients::gpu", "clients::middle", "clients::gpu"]
