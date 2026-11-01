"""Opaque host state remains original while exact array effects use a provider."""
import json
import shutil
import subprocess
from hashlib import sha256

import pytest
from fparser.two.utils import walk

from compiler.driver.options import CompilerOptions
from compiler.frontend.device_tasks import external_device_task, task_scope_identity
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_external_device_tasks import PROVIDER_INTERFACE, SOURCE, digest, fixture, requalify
from compiler.tests.test_graph_source_owner import FACT

TYPES = """module external_types
implicit none
type state_type
real(8)::scratch
end type
end module
"""
OPAQUE_SOURCE = TYPES + SOURCE.replace("module external_fixture", "module external_fixture\nuse external_types", 1)
OPAQUE_SOURCE = OPAQUE_SOURCE.replace("subroutine original_transform(input,output)", """function make_state() result(value)
type(state_type)::value
value%scratch=3.0_8
end function
subroutine original_transform(d,input,output)
type(state_type),intent(inout)::d""")
OPAQUE_SOURCE = OPAQUE_SOURCE.replace("intent(inout)::output", "intent(out)::output")
OPAQUE_SOURCE = OPAQUE_SOURCE.replace("output=3.0_8*input", "d%scratch=3.0_8\noutput=d%scratch*input")
OPAQUE_SOURCE = OPAQUE_SOURCE.replace("integer::i\ndo i=1,n", """integer::i
type(state_type),save::cache
logical,save::called=.false.
if(.not.called)then
cache=make_state()
called=.true.
endif
do i=1,n""")
OPAQUE_SOURCE = OPAQUE_SOURCE.replace("call original_transform(b,a)", "call original_transform(cache,b,a)")


def opaque_fixture(tmp_path, source=OPAQUE_SOURCE):
    path, analysis, routine, node, task, manifest, implementation = fixture(tmp_path, source)
    constructor, = [item for item in walk(routine.execution) if type(item).__name__ == "Assignment_Stmt"
                    and str(item.items[0]).lower() == "cache"]
    provider = json.loads(manifest.read_text())
    task.update(version=2, completion="wait_without_export", opaque_arguments=[{
        "formal": "d", "constructor": {"span": list(constructor.item.span),
            "statement_sha256": sha256(str(constructor).encode()).hexdigest()},
        "qualification": provider["numerical_contract"]["qualification"]}])
    task["configuration"] = task["configuration"][:1]
    provider.update(version=2, metadata_query={"module": "external_metadata", "procedure": "metadata_query",
        "opaque_formals": ["d"], "configuration_argument": "config", "read_only": True,
        "configuration_components": [{"component": "coefficient", "kind": "real64", "count": 1}]})
    provider["numerical_contract"]["task_contracts"] = [task_scope_identity(task)]
    manifest.write_text(json.dumps(provider))
    task["provider_manifest"]["sha256"] = digest(manifest)
    return path, analysis, routine, node, task, manifest, implementation


def test_v2_separates_original_out_definition_kill_from_physical_partial_overwrite(tmp_path):
    _, analysis, routine, node, _, _, _ = opaque_fixture(tmp_path)
    proof = external_device_task(analysis, routine, node)
    record = proof.public()
    assert record["version"] == 2
    assert tuple(binding.root for binding in proof.bindings) == ("argument::b", "argument::a")
    assert record["source_effects"]["entry_definition_kills"] == ["output"]
    state, = record["source_effects"]["opaque_state"]
    assert state["resource"] == "external_fixture::advance::cache"
    assert state["capture_authorized"] is False
    assert state["gpu_access_authorized"] is False
    assert record["definitions_authorized"] is False


@pytest.mark.parametrize("change", ["receiver", "completion", "overlap", "query", "intent", "query_kind", "query_formals"])
def test_v2_rejects_unqualified_receiver_query_configuration_and_original_intent_changes(tmp_path, change):
    _, analysis, routine, node, task, manifest, _ = opaque_fixture(tmp_path)
    provider = json.loads(manifest.read_text())
    if change == "receiver":
        task["opaque_arguments"][0]["formal"] = "input"
    if change == "completion":
        task["completion"] = "unobserved_async"
    if change == "overlap":
        task["configuration"].append({"component": "coefficient", "kind": "real64", "values": [3.0]})
    if change == "query":
        provider["metadata_query"]["read_only"] = False
    if change == "query_kind":
        provider["metadata_query"]["configuration_components"][0]["kind"] = []
    if change == "query_formals":
        provider["metadata_query"]["opaque_formals"] = [{}]
    if change == "intent":
        task["resources"][1]["effects"][0]["kind"] = "read"
    manifest.write_text(json.dumps(provider))
    requalify(task, manifest)
    with pytest.raises(CompilationError):
        external_device_task(analysis, routine, node)


def emit(tmp_path, *, child=False):
    source = OPAQUE_SOURCE
    if child:
        prefix, separator, suffix = source.rpartition("end module")
        source = prefix + """subroutine outer(a,b,n)
real(8),intent(inout)::a(:),b(:)
integer,intent(in)::n
call advance(a,b,n)
end subroutine
""" + separator + suffix
    path, analysis, _, _, _, _, _ = opaque_fixture(tmp_path, source)
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): digest(path)},
             "captures": {"argument::a": dict(FACT), "argument::b": dict(FACT)}}
    builder = ScopeBuilder([path], "external_fixture::" + ("outer" if child else "advance"), facts=facts,
        options=CompilerOptions(), config=OffloadConfig(policy="sections", scope_execution="graph"),
        contracts=analysis.contracts)
    outputs, report = builder.run()
    return path, outputs, report


@pytest.mark.parametrize("child", [False, True])
def test_v2_emits_typed_query_out_kill_completion_and_owner_lifetime_controls(tmp_path, child):
    path, outputs, report = emit(tmp_path, child=child)
    owner, = report["scopes"]
    task, = owner["external_device_tasks"]
    binding = task["execution_binding"]
    assert binding["owner"] == "external_fixture::" + ("outer" if child else "advance")
    assert binding["canonical_resources"] == {"input": "argument::b", "output": "argument::a"}
    assert binding["configuration_words"] == 2
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert "fort_external_metadata(fort_external_config,cache)" in text
    kill = text.index("fort_scope_forget_definition(", text.index("fort_external_report ="))
    execute = text.index("fort_external_status = fort_external_execute(")
    wait = text.index("fort_external_status = fort_scope_wait(", execute)
    assert kill < execute < wait
    assert "transfer(fort_external_config%coefficient,0_c_int64_t)" in text
    assert "fort_external_controls_" not in text
    if child:
        assert owner["module_coordinators"][0]["qualified_opaque_storage"] == ["external_fixture::advance::cache"]
        assert text.count("type(fort_external_pointer)") >= 2


def test_v2_changed_physical_association_does_not_reuse_a_single_binding_record(tmp_path):
    prefix, separator, suffix = OPAQUE_SOURCE.rpartition("end module")
    source = prefix + """subroutine outer(a,b,c,d,n)
real(8),intent(inout)::a(:),b(:),c(:),d(:)
integer,intent(in)::n
call advance(a,b,n)
call advance(c,d,n)
end subroutine
""" + separator + suffix
    path, analysis, _, _, _, _, _ = opaque_fixture(tmp_path, source)
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): digest(path)},
             "captures": {"argument::" + name: dict(FACT) for name in ("a", "b", "c", "d")}}
    builder = ScopeBuilder([path], "external_fixture::outer", facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="graph"), contracts=analysis.contracts)
    _, report = builder.run()
    assert any("external provider member requires one canonical original association" in item["reason"]
               for item in report["boundaries"])
    assert all(not scope.get("external_device_tasks") for scope in report["scopes"])


def test_v2_borrowed_provider_retirement_stays_inside_original_observable_guard(tmp_path):
    source = OPAQUE_SOURCE.replace("use external_types", "use external_types\ninteger::debuglevel=0", 1)
    source = source.replace("call original_transform(cache,b,a)",
                            "call original_transform(cache,b,a)\nif(debuglevel>0)write(*,*)'original debug'")
    prefix, separator, suffix = source.rpartition("end module")
    source = prefix + """subroutine outer(a,b,n)
real(8),intent(inout)::a(:),b(:)
integer,intent(in)::n
call advance(a,b,n)
call advance(a,b,n)
end subroutine
""" + separator + suffix
    path, analysis, _, _, _, _, _ = opaque_fixture(tmp_path, source)
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): digest(path)},
             "captures": {"argument::a": dict(FACT), "argument::b": dict(FACT)}}
    builder = ScopeBuilder([path], "external_fixture::outer", facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="graph"), contracts=analysis.contracts)
    outputs, report = builder.run()
    scope, = report["scopes"]
    child, = scope["module_coordinators"]
    boundary, = [item for item in child["boundaries"] if item["conditional_close"] is not None]
    assert boundary["conditional_close"] == "debuglevel > 0"
    text = outputs[report["sources"][str(path)]["replacement"]]
    start = text.index("if (debuglevel > 0) then")
    write = text.index("WRITE(*, *) 'original debug'", start)
    guarded = text[start:write]
    assert guarded.count("if (debuglevel > 0) then") == 1
    assert "fort_external_destroy_status = fort_external_destroy(" in guarded
    assert "fort_scope_resident_close_v1(" in guarded
    task, = scope["external_device_tasks"]
    assert task["execution_binding"]["ownership"] == (
        "owning invocation controls forwarded by reference through every checked ENTRY")


def test_external_device_access_checks_generation_without_mutating_storage_lifetime(tmp_path):
    from compiler.tests.test_graph_source_owner import walk_graph

    _, _, report = emit(tmp_path)
    scope, = report["scopes"]
    procedure, = scope["graph_execution"]["procedures"]
    external, = [task for task in walk_graph(procedure["task_graph"]) if ":external:" in task["id"]]
    assert external["implementations"] == ["gpu"]
    assert external["metadata_reads"] == ["argument::a", "argument::b"]
    assert external["reads"] == ["argument::b"]
    assert external["may_writes"] == ["argument::a"]
    assert external["lifetime_resources"] == []
    assert external["ordered"] is True


@pytest.mark.native
@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("clock", [False, True])
def test_v2_owner_and_original_entry_abis_compile_with_actual_typed_fortran_interfaces(tmp_path, child, clock, monkeypatch):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    if clock:
        source = ("module clock_kinds\nuse iso_fortran_env\nend module\n"
                  "module clock_reexport\nuse clock_kinds\nend module\n" +
                  OPAQUE_SOURCE.replace("module external_fixture", "module external_fixture\nuse clock_reexport", 1))
        source = source.replace("integer::i\n", "integer::i\ninteger(int64)::t1,t2\n")
        source = source.replace("call original_transform(cache,b,a)",
                               "call system_clock(t1)\ncall original_transform(cache,b,a)\ncall system_clock(t2)")
        monkeypatch.setattr("compiler.tests.test_external_device_tasks_v2.OPAQUE_SOURCE", source)
    _, outputs, report = emit(tmp_path, child=child)
    build = tmp_path / "build"
    build.mkdir()
    for name, text in outputs.items():
        destination = build / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
    types, interface, query = (build / name for name in ("types.f90", "provider.f90", "query.f90"))
    types.write_text(TYPES)
    interface.write_text(PROVIDER_INTERFACE)
    query.write_text("""module external_metadata
use iso_c_binding
use external_types
use external_provider
implicit none
contains
integer(c_int) function metadata_query(config,d) result(status)
type(provider_config),intent(inout)::config
type(state_type),intent(in)::d
config%coefficient=d%scratch
status=0
end function
end module
""")
    artifacts = [item for item in report["build_sources"] if item["language"] == "fortran"]
    paths = [types, *(build / item["path"] for item in artifacts if item["role"] == "common_runtime"),
             interface, query, *(build / item["path"] for item in artifacts if item["role"] != "common_runtime")]
    for index, path in enumerate(paths):
        result = subprocess.run([fortran, "-std=f2018", "-fopenmp", "-fcheck=all", "-J", str(build), "-I", str(build),
                                 "-c", str(path), "-o", str(build / f"source-{index}.o")], cwd=build,
                                text=True, capture_output=True, timeout=30)
        (build / f"source-{index}.stderr").write_text(result.stderr)
        assert result.returncode == 0, result.stderr
