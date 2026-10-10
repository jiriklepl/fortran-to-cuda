"""Native read-only storage keeps original guards without becoming defined GPU data."""

import re
from copy import copy
from hashlib import sha256
from types import SimpleNamespace

import pytest
from fparser.two.utils import walk

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.segments import fragment
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_lexical_source_owner import SOURCE, emit
from compiler.tests.test_source_scopes import FACT

READ = """if(allocated(edge_data)) then
b(-2)=b(-2)+sum(edge_data)
endif"""
HOST_SOURCE = SOURCE.replace("integer::visits=0", "integer::visits=0\nreal(8),allocatable::edge_data(:)").replace(
    "if(escape) then\ncall opaque(b,n)\nendif", READ)


def operations(owner):
    return [operation for segment in owner["planning_segments"]
            for operation in segment["operations"]["native_operations"]]


def source_builder(tmp_path, source=HOST_SOURCE):
    path = tmp_path / "readonly.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: FACT for name in ("a", "b", "out")}}
    return ScopeBuilder([path], "local_owner::step", facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="reached"))


def test_native_read_range_keeps_original_effects_but_no_managed_definition(tmp_path):
    path, outputs, report = emit(tmp_path, HOST_SOURCE)
    owner, = report["scopes"]
    assert not owner["boundaries"], owner["boundaries"]
    assert len(owner["gpu_leaves"]) == 2
    operation, = [operation for operation in operations(owner) if operation["host_only_native_reads"]]
    proof = operation["host_only_native_reads"]
    assert proof["schema_version"] == 1
    assert proof["resources"] == ["local_owner::edge_data"]
    assert not proof["device_capture"]
    assert not proof["managed_definitions"]
    assert not proof["calls_or_escapes"]
    assert not proof["allocation_changes"]
    assert len(proof["identity"]) == 64
    assert len(proof["analysis_identity"]) == 64
    assert "local_owner::edge_data" in operation["resources"]
    assert "local_owner::edge_data" not in operation["managed_resources"]
    assert operation["host_metadata"]["local_owner::edge_data"]["device_capture"] is False
    for segment in owner["planning_segments"]:
        definitions = segment["ordered_definitions"]
        assert "local_owner::edge_data" not in definitions["required_whole"]
        assert "local_owner::edge_data" not in definitions["whole_on_return"]
    text = outputs[report["sources"][str(path)]["replacement"]]
    reservation = text.index("fort_meta_extents_")
    # The original condition/payload remain original references. Only a guarded
    # descriptor-range reservation is emitted; its initialized flag is zero.
    assert "if (allocated(edge_data)) then" in text[reservation:].lower()
    shape = text.index("shape(edge_data, kind=c_size_t)")
    assert text.rfind("if (allocated(edge_data)) then", 0, shape) >= 0
    assert text.rfind("if (allocated(edge_data)) then", 0, text.index("lbound(edge_data, kind=c_int64_t)")) >= 0
    assert "storage_size(edge_data, kind=c_size_t)" in text
    assert "c_loc(edge_data)" in text
    assert "sum(edge_data)" in text.lower()
    assert "FORT_SCOPE_BYTES" in text
    assert "fort_layout, 0_c_int, fort_buffer_" in text
    assert "allocated(fort_array" not in text.lower()


@pytest.mark.parametrize("dtype", ["real(4)", "real(8)", "integer(4)", "integer(8)"])
def test_numeric_native_read_types_require_no_gpu_capture_fact(tmp_path, dtype):
    _, _, report = emit(tmp_path, HOST_SOURCE.replace("real(8),allocatable::edge_data", dtype + ",allocatable::edge_data"))
    owner, = report["scopes"]
    assert not owner["boundaries"]
    assert any(operation["host_only_native_reads"] for operation in operations(owner))


@pytest.mark.parametrize("action", ["edge_data(-2)=1", "deallocate(edge_data)", "call opaque(edge_data,n)",
                                    "call move_alloc(edge_data,other_data)"])
def test_later_writes_lifetime_changes_and_escapes_close_without_replaying_prefix(tmp_path, action):
    source = HOST_SOURCE.replace("real(8),allocatable::edge_data(:)", "real(8),allocatable::edge_data(:),other_data(:)")
    source = source.replace(READ, READ + "\n" + action)
    path, outputs, report = emit(tmp_path, source)
    owner, = report["scopes"]
    assert owner["boundaries"]
    assert all(not boundary["reopen"] for boundary in owner["boundaries"])
    assert any(operation["host_only_native_reads"] for operation in operations(owner))
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert text.lower().count(action.lower()) == 1


def test_existing_capture_facts_keep_the_resource_managed(tmp_path):
    _, _, report = emit(tmp_path, HOST_SOURCE, captures={"local_owner::edge_data": FACT})
    owner, = report["scopes"]
    assert not owner["boundaries"]
    assert not any(operation["host_only_native_reads"] for operation in operations(owner))
    assert any("local_owner::edge_data" in operation["managed_resources"] for operation in operations(owner))


def test_managed_numerical_use_and_native_read_share_real_capture_facts(tmp_path):
    source = HOST_SOURCE.replace("b(i)=2*a(i)+real(i,8)", "b(i)=2*a(i)+real(i,8)+edge_data(i)")
    _, _, report = emit(tmp_path, source, captures={"local_owner::edge_data": FACT})
    owner, = report["scopes"]
    assert not owner["boundaries"]
    assert len(owner["gpu_leaves"]) == 2
    assert not any(operation["host_only_native_reads"] for operation in operations(owner))
    assert any("local_owner::edge_data" in segment["resources"] for segment in owner["planning_segments"])


def test_automatic_execution_does_not_guess_unpriced_range_preparation(tmp_path):
    path, outputs, report = emit(tmp_path, HOST_SOURCE, policy="auto")
    owner, = report["scopes"]
    assert not owner["estimate_available"]
    assert owner["planning_reason"] == "host-only native range preparation lacks compatible offline cost estimates"
    assert owner["automatic_preflight"]["contexts_created"] == 0
    assert outputs[report["sources"][str(path)]["replacement"]] == HOST_SOURCE


def test_imported_native_read_keeps_original_canonical_module_binding(tmp_path):
    source = HOST_SOURCE.replace("module local_owner\nimplicit none", """module boundary_values
real(8),allocatable::edge_values(:)
end module
module local_owner
use boundary_values,only:edge_data=>edge_values
implicit none""").replace("real(8),allocatable::edge_data(:)\n", "")
    _, _, report = emit(tmp_path, source)
    owner, = report["scopes"]
    assert not owner["boundaries"]
    operation, = [operation for operation in operations(owner) if operation["host_only_native_reads"]]
    assert operation["host_only_native_reads"]["resources"] == ["boundary_values::edge_values"]


def test_later_uncaptured_numerical_write_closes_at_reached_operation(tmp_path):
    source = HOST_SOURCE.replace(READ, READ + "\ndo i=-2,n-3\nedge_data(i)=b(i)\nenddo")
    path, outputs, report = emit(tmp_path, source)
    owner, = report["scopes"]
    assert any(operation["host_only_native_reads"] for operation in operations(owner))
    boundary, = owner["boundaries"]
    assert "read-only complete effects" in boundary["reason"] or "managed write" in boundary["reason"]
    assert "edge_data(i)=b(i)" in outputs[report["sources"][str(path)]["replacement"]].lower()


@pytest.mark.parametrize("intent", ["in", "inout", "out"])
@pytest.mark.parametrize("guarded", [False, True])
def test_host_range_closes_before_numeric_child_formal_and_original_call_once(tmp_path, intent, guarded):
    call = "call child(edge_data,n)"
    action = "visits=visits+int(sum(x))" if intent == "in" else "x(i)=real(i,8)"
    source = HOST_SOURCE.replace(READ, READ + "\n" + ("if(escape) " if guarded else "") + call).replace("end module", f"""subroutine child(x,n)
real(8),intent({intent})::x(-2:)
integer,intent(in)::n
integer::i
do i=-2,n-3
{action}
enddo
end subroutine
end module""")
    path, outputs, report = emit(tmp_path, source)
    owner, = report["scopes"]
    assert len(owner["gpu_leaves"]) == 2
    boundary, = owner["boundaries"]
    assert "host-only native range cannot be forwarded to a managed child formal" in boundary["reason"]
    assert "local_owner::edge_data" in boundary["reason"]
    assert not owner["module_coordinators"]
    assert any(operation["host_only_native_reads"] for operation in operations(owner))
    text = outputs[report["sources"][str(path)]["replacement"]].lower()
    assert len(re.findall(r"call\s+child\s*\(\s*edge_data\s*,\s*n\s*\)", text)) == 1


def test_invalid_existing_capture_facts_cannot_be_replaced_by_host_authority(tmp_path):
    _, _, report = emit(tmp_path, HOST_SOURCE, captures={"local_owner::edge_data": {**FACT, "storage": "unstable"}})
    owner, = report["scopes"]
    assert owner["boundaries"]
    assert not any(operation["host_only_native_reads"] for operation in operations(owner))


@pytest.mark.parametrize("attribute", ["pointer", "volatile", "asynchronous"])
def test_uncertain_native_storage_remains_a_boundary(tmp_path, attribute):
    declaration = "real(8),pointer::edge_data(:)" if attribute == "pointer" else "real(8),allocatable," + attribute + "::edge_data(:)"
    _, _, report = emit(tmp_path, HOST_SOURCE.replace("real(8),allocatable::edge_data(:)", declaration))
    owner, = report["scopes"]
    assert owner["boundaries"]
    assert not any(operation["host_only_native_reads"] for operation in operations(owner))


def test_threadprivate_original_storage_cannot_be_reserved_as_one_shared_range(tmp_path):
    _, _, report = emit(tmp_path, HOST_SOURCE.replace("real(8),allocatable::edge_data(:)",
        "real(8),allocatable::edge_data(:)\n!$omp threadprivate(edge_data)"))
    owner, = report["scopes"]
    assert owner["boundaries"]


def test_native_lifetime_projection_does_not_mutate_parent_authority(tmp_path):
    builder = source_builder(tmp_path)
    selected = next(node for node in walk(builder.entry.execution) if type(node).__name__ == "Assignment_Stmt"
                    and "SUM(edge_data)" in str(node))
    graph = builder.analysis.structure(builder.entry.qualified)
    parent_summary = builder.analysis.segment_summary(builder.entry.qualified, (selected,), capture_locals=True)
    assert not parent_summary["complete"]
    cache_names = ("_structures", "_segments", "_joined_completions", "_worksharing_native_completions",
                   "_native_sections", "_descriptor_proofs", "_allocation_authorizations", "_source_provenance")
    previous = {name: copy(getattr(builder.analysis, name)) for name in cache_names}
    cache_stats = builder.analysis._summary_cache.stats
    stable = builder.analysis.stable_module_allocatables
    authority = builder.analysis._summary_authority()
    native = fragment(builder, (selected,), native_host_reads=True)
    assert native.host_only_reads
    assert native.summary["complete"]
    assert builder.analysis._summary_authority() == authority
    assert builder.analysis.stable_module_allocatables == stable
    assert builder.analysis._summary_cache.stats == cache_stats
    assert builder.analysis.structure(builder.entry.qualified) is graph
    for name in cache_names:
        assert getattr(builder.analysis, name) == previous[name]
    assert builder.facts["captures"].keys() == {"argument::a", "argument::b", "argument::out"}
    with pytest.raises(CompilationError, match="stable storage/definition facts"):
        fragment(builder, (selected,))
    assert not builder.analysis.segment_summary(builder.entry.qualified, (selected,), capture_locals=True)["complete"]


def test_canonical_borrowed_formal_cannot_promote_prior_host_only_range(tmp_path):
    """Reconciliation uses actual mappings, including a not-yet-added child.

    This isolates role reconciliation using original bindings and an actual
    source call. It grants no capture/alias proof and emits no numerical code.
    """
    from compiler.scopes.lexical import LexicalOwner

    source = HOST_SOURCE.replace(READ, READ + "\ncall child(edge_data)").replace("end module", """subroutine child(x)
real(8),intent(inout)::x(:)
x=x+1
end subroutine
end module""")
    builder = source_builder(tmp_path, source)
    selected = next(node for node in walk(builder.entry.execution) if type(node).__name__ == "Assignment_Stmt"
                    and "SUM(edge_data)" in str(node))
    native = fragment(builder, (selected,), native_host_reads=True)
    host_scope = SimpleNamespace(native=[native])
    empty_scope = SimpleNamespace(native=[])
    parent = LexicalOwner(builder)
    child = copy(parent)
    child.owner, child.parent = parent, parent
    call = next(node for node in walk(builder.entry.execution) if type(node).__name__ == "Call_Stmt"
                and "child" in str(node))
    resolved = builder.execution_source_call(builder.entry, call)
    mapping, = resolved.mappings
    assert mapping.binding.root == "local_owner::edge_data"
    child.canonical_bindings = {mapping.formal_binding.root: mapping.binding}
    child.units = []
    managed = {mapping.formal_binding.root: mapping.formal_binding}
    parent.units = [SimpleNamespace(scope=host_scope, arrays={})]
    assert child.host_read_conflicts(empty_scope, managed) == {"local_owner::edge_data"}
    # Reverse order: a managed borrowed child already belongs to the owner.
    parent.units = []
    child.units = [SimpleNamespace(scope=empty_scope, arrays=managed)]
    parent.members[resolved.procedure] = child
    assert parent.host_read_conflicts(host_scope, {}) == {"local_owner::edge_data"}
    # The companion constructor has not yet inserted itself in owner.members.
    del parent.members[resolved.procedure]
    child.units = [SimpleNamespace(scope=host_scope, arrays={})]
    assert child.host_read_conflicts(empty_scope, managed) == {"local_owner::edge_data"}
    # A mixed current unit also closes before either role is registered.
    child.units = []
    assert child.host_read_conflicts(host_scope, managed) == {"local_owner::edge_data"}
