"""Available free procedure source supplies effects, not missing caller ABIs."""

from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import form_source_scopes


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def records(report):
    return {summary["procedure"]: summary for summary in report["procedures"]}


def test_multifile_external_scalar_calls_keep_ordered_definitions_and_identity(tmp_path):
    leaf = write(tmp_path, "free.f90", """subroutine replace(value)
real(8),intent(out)::value
value=4.d0
end subroutine
""")
    caller = write(tmp_path, "caller.f90", """module caller
contains
subroutine step(value)
real(8),intent(inout)::value
external replace
call replace(value)
call replace(value)
end subroutine
end module
""")
    report = SourceEffects([caller, leaf]).report("caller::step")
    assert report["complete"], report
    summaries = records(report)
    assert set(summaries) == {"caller::step", "$external::replace"}
    external = summaries["$external::replace"]
    assert external["source_kind"] == "external"
    assert not external["cloneable"]
    assert external["native_completion"]["available"]
    assert external["call_interface"]["available"]
    calls = [operation for operation in summaries["caller::step"]["operations"] if operation["kind"] == "call"]
    assert len(calls) == 2
    assert all(operation["resource_mapping"] == {"argument::value": "argument::value"} for operation in calls)
    assert all(operation["summary_identity"] == external["summary_identity"] for operation in calls)
    assert all(operation["definition_events"][0]["resource"] == "argument::value" for operation in calls)


def test_explicit_shape_external_arrays_have_original_logical_view_metadata(tmp_path):
    path = write(tmp_path, "external.f90", """subroutine scale(field,n)
integer,intent(in)::n
real(8),intent(inout)::field(-2:n-3)
integer::i
do i=-2,n-3
field(i)=2*field(i)
enddo
end subroutine
module caller
contains
subroutine step(field,n)
real(8),intent(inout)::field(-7:)
integer,intent(in)::n
call scale(field,n)
end subroutine
end module
""")
    report = SourceEffects([path]).report("caller::step")
    assert report["complete"], report
    summary = records(report)["$external::scale"]
    assert summary["arguments"][0]["lower_bounds"] == ("- 2",)
    assert summary["call_interface"]["available"]
    mapped = records(report)["caller::step"]["ordered_effects"]
    assert any(operation["resource"] == "argument::field" and operation["view_chain"] for operation in mapped)


@pytest.mark.parametrize("shape", ["*", "-2:*", "n,*", "n,-2:*"])
def test_assumed_size_external_shape_is_indexed_without_descriptor_assumptions(tmp_path, shape):
    indices = "1,1" if "," in shape else "1"
    path = write(tmp_path, "assumed_size.f90", f"""subroutine touch(field,n)
integer,intent(in)::n
real(8),intent(inout)::field({shape})
field({indices})=7.d0
end subroutine
module caller
contains
subroutine step(field,n)
integer,intent(in)::n
real(8),intent(inout)::field(n,n)
call touch(field,n)
end subroutine
end module
""")
    analysis = SourceEffects([path])
    rank = 2 if "," in shape else 1
    if rank == 1:
        path.write_text(path.read_text().replace("::field(n,n)", "::field(n)"))
        analysis = SourceEffects([path])
    report = analysis.report("caller::step")
    assert report["complete"], report
    external = records(report)["$external::touch"]
    binding = external["arguments"][0]
    assert binding["rank"] == rank
    assert binding["shape"][-1]["kind"] == "Assumed_Size_Spec"
    assert binding["shape"][-1]["bounds"][1] == "*"
    assert binding["lower_bounds"][-1] == ("- 2" if "-2:" in shape else "1")
    assert external["call_interface"]["available"]


def test_module_and_external_identical_names_remain_distinct(tmp_path):
    path = write(tmp_path, "names.f90", """subroutine operation(value)
integer,intent(out)::value
value=99
end subroutine
module caller
contains
subroutine operation(value)
integer,intent(out)::value
value=1
end subroutine
subroutine step(value)
integer,intent(out)::value
call operation(value)
end subroutine
subroutine explicit_external(value)
integer,intent(out)::value
external operation
call operation(value)
end subroutine
end module
""")
    analysis = SourceEffects([path])
    local = analysis.report("caller::step")
    assert local["complete"]
    assert set(records(local)) == {"caller::step", "caller::operation"}
    external = analysis.report("caller::explicit_external")
    assert external["complete"]
    assert set(records(external)) == {"caller::explicit_external", "$external::operation"}


@pytest.mark.parametrize("declaration", [
    "real(8),intent(in)::field(:)",
    "real(8),optional,intent(in)::field(3)",
    "real(8),allocatable,intent(in)::field(:)",
    "real(8),pointer,intent(in)::field(:)",
    "real(8),value,intent(in)::field",
])
def test_external_descriptor_and_value_abis_require_original_explicit_interface(tmp_path, declaration):
    actual_declaration = "real(8)::field(3)" if "field(" in declaration else "real(8)::field"
    path = write(tmp_path, "requires_interface.f90", f"""subroutine inspect(field)
{declaration}
end subroutine
module caller
contains
subroutine step(field)
{actual_declaration}
call inspect(field)
end subroutine
end module
""")
    report = SourceEffects([path]).report("caller::step")
    assert not report["complete"]
    root = records(report)["caller::step"]
    assert any("requires a proven explicit interface" in reason for reason in root["reasons"])
    assert not any(operation["kind"] == "call" for operation in root["operations"])


def test_external_keyword_call_is_not_inferred_from_definition_source(tmp_path):
    path = write(tmp_path, "keyword.f90", """subroutine inspect(value)
integer,intent(in)::value
end subroutine
module caller
contains
subroutine step(value)
integer,intent(in)::value
call inspect(value=value)
end subroutine
end module
""")
    report = SourceEffects([path]).report("caller::step")
    assert not report["complete"]
    assert any("external keyword arguments require a proven explicit interface" in reason
               for reason in records(report)["caller::step"]["reasons"])


def test_unknown_wildcard_does_not_fall_back_to_external_source(tmp_path):
    path = write(tmp_path, "unknown_exports.f90", """subroutine inspect(value)
integer,intent(in)::value
end subroutine
module caller
use unavailable_library
contains
subroutine step(value)
integer,intent(in)::value
call inspect(value)
end subroutine
end module
""")
    report = SourceEffects([path]).report("caller::step")
    assert not report["complete"]
    assert "$external::inspect" not in records(report)


def test_procedure_dummy_is_dynamic_even_when_same_named_external_source_exists(tmp_path):
    path = write(tmp_path, "dynamic.f90", """subroutine callback(value)
integer,intent(out)::value
value=1
end subroutine
module caller
contains
subroutine step(callback,value)
external callback
integer,intent(out)::value
call callback(value)
end subroutine
end module
""")
    report = SourceEffects([path]).report("caller::step")
    assert not report["complete"]
    assert "$external::callback" not in records(report)


def test_external_recursion_remains_a_bounded_boundary(tmp_path):
    path = write(tmp_path, "recursive.f90", """recursive subroutine recurse(value)
integer,intent(inout)::value
if(value>0) call recurse(value)
end subroutine
""")
    report = SourceEffects([path]).report("$external::recurse")
    assert not report["complete"]
    assert any("recurs" in reason for summary in report["procedures"] for reason in summary["reasons"])


def test_external_uses_module_resource_and_completion_without_fake_module_owner(tmp_path):
    path = write(tmp_path, "hidden.f90", """module state
integer::value
end module
subroutine inspect()
use state,only:value
value=value+1
end subroutine
""")
    report = SourceEffects([path]).report("inspect")
    assert report["complete"], report
    summary = records(report)["$external::inspect"]
    assert summary["native_completion"]["available"]
    assert any(operation.get("resource") == "state::value" for operation in summary["operations"])


def test_duplicate_external_definitions_are_rejected(tmp_path):
    first = write(tmp_path, "first.f90", "subroutine duplicate()\nend subroutine\n")
    second = write(tmp_path, "second.f90", "subroutine duplicate()\nend subroutine\n")
    with pytest.raises(CompilationError, match="duplicate source procedure"):
        SourceEffects([first, second])


def test_free_entry_and_nested_native_external_calls_remain_execution_boundaries(tmp_path):
    path = write(tmp_path, "native_boundaries.f90", """subroutine inspect(value)
integer,intent(inout)::value
value=value+1
end subroutine
module caller
contains
subroutine gpu(field,n)
real(8),intent(inout)::field(:)
integer,intent(in)::n
integer::i
do i=1,n
field(i)=2*field(i)
enddo
end subroutine
subroutine middle(field,value)
real(8),intent(inout)::field(:)
integer,intent(inout)::value
field(1)=field(1)+1
call inspect(value)
end subroutine
subroutine step(field,value,n)
real(8),intent(inout)::field(:)
integer,intent(inout)::value
integer,intent(in)::n
call gpu(field,n)
call middle(field,value)
call gpu(field,n)
end subroutine
end module
""")
    capture = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::field": capture}}
    _outputs, report = form_source_scopes([path], "caller::step", facts=facts, options=CompilerOptions(),
                                         config=OffloadConfig(policy="sections"))
    assert report["native_effects"]["complete"]
    assert any("external source entry execution requires standalone procedure variants" in boundary["reason"]
               for boundary in report["boundaries"])
    assert all("caller::middle" not in scope["calls"] for scope in report["scopes"])
    _outputs, external = form_source_scopes([path], "$external::inspect", facts=facts, options=CompilerOptions(),
                                           config=OffloadConfig(policy="sections"))
    assert external["scope_count"] == 0
    assert external["native_effects"]["complete"]
    assert external["boundaries"][0]["reason"].startswith("external source entry execution")


def test_module_argument_cannot_collide_with_formal_resource_namespace(tmp_path):
    path = write(tmp_path, "namespace.f90", """module argument
real(8)::field(3)
contains
subroutine update(field)
real(8),intent(inout)::field(:)
field=2*field
end subroutine
end module
module caller
use argument,only:renamed=>field,update
contains
subroutine step(field)
real(8),intent(inout)::field(:)
call update(field)
field=field+renamed
end subroutine
end module
""")
    report = SourceEffects([path]).report("caller::step")
    assert not report["complete"]
    assert any("collides with canonical dummy resources" in reason for summary in report["procedures"]
               for reason in summary["reasons"])


@pytest.mark.parametrize("operation", ["x=x+hidden", "call update(hidden)"])
def test_imported_argument_module_global_is_not_a_same_named_dummy(tmp_path, operation):
    path = write(tmp_path, "imported_namespace.f90", f"""module argument
real(8)::x(3)
end module
module caller
use argument,only:hidden=>x
contains
subroutine update(field)
real(8),intent(inout)::field(:)
field=2*field
end subroutine
subroutine step(x)
real(8),intent(inout)::x(:)
{operation}
end subroutine
end module
""")
    analysis = SourceEffects([path])
    summary = analysis.report("caller::step")
    assert not summary["complete"]
    root = records(summary)["caller::step"]
    assert any("collides with canonical dummy resources" in reason for reason in root["reasons"])
    assert not any(operation["kind"] == "call" for operation in root["operations"])
    assert analysis.resource_identity_boundary(analysis.routines["caller::step"].scope.bindings["x"]) is None
