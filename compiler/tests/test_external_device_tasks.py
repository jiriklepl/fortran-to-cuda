"""External adapters cannot substitute for exact source or storage authority."""
import json
import shutil
import subprocess
from copy import copy
from hashlib import sha256

import pytest
from fparser.two.utils import walk

from compiler.driver.options import CompilerOptions
from compiler.frontend.device_tasks import external_device_task, task_scope_identity
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_source_owner import FACT

SOURCE = """module external_fixture
implicit none
contains
subroutine original_transform(input,output)
real(8),intent(in)::input(:)
real(8),intent(inout)::output(:)
output=3.0_8*input
end subroutine
subroutine advance(a,b,n)
real(8),intent(inout)::a(:),b(:)
integer,intent(in)::n
integer::i
do i=1,n
b(i)=2.0_8*a(i)
enddo
call original_transform(b,a)
do i=1,n
b(i)=a(i)+1.0_8
enddo
end subroutine
end module
"""


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def requalify(task, manifest):
    """A malformed typed contract still cannot bypass compiler validation."""
    provider = json.loads(manifest.read_text())
    provider["numerical_contract"]["task_contracts"] = [task_scope_identity(task)]
    manifest.write_text(json.dumps(provider))
    task["provider_manifest"]["sha256"] = digest(manifest)


def fixture(tmp_path, source=SOURCE):
    path = tmp_path / "source.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    routine = analysis.routines["external_fixture::advance"]
    node, = [item for item in walk(routine.execution) if type(item).__name__ == "Call_Stmt"
             and str(item.items[0]).lower() == "original_transform"]
    body_digest = sha256(str(analysis.routines["external_fixture::original_transform"].scope.node).encode()).hexdigest()
    implementation = tmp_path / "provider.f90"
    implementation.write_text("! Immutable implementation supplied by the caller\n")
    qualification = tmp_path / "qualification.json"
    qualification.write_text(json.dumps({"original_body": body_digest, "test_evidence": "explicit caller evidence"}))
    provider = {
        "version": 1, "kind": "external_device_provider",
        "fortran": {"module": "external_provider", "config_type": "provider_config", "create": "provider_create",
                    "execute": "provider_execute", "destroy": "provider_destroy"},
        "artifacts": {str(implementation): digest(implementation)},
        "numerical_contract": {"identity": digest(qualification), "qualification": {
            "path": str(qualification), "sha256": digest(qualification)}, "original_bodies": [body_digest]},
        "execution": {"stream": "supplied", "completion": "same_stream", "workspace": "domain_claim",
                      "fallback": "prework_only"},
    }
    manifest = tmp_path / "provider.json"
    manifest.write_text(json.dumps(provider))
    extent = {"descriptor": "size", "formal": "input", "axis": 1, "offset": 0}
    task = {"version": 1, "source": {"sha256": digest(path), "span": list(node.item.span),
        "call_sha256": sha256(str(node).encode()).hexdigest(), "callee": "external_fixture::original_transform",
        "callee_body_sha256": body_digest}, "provider_manifest": {"path": str(manifest), "sha256": digest(manifest)},
        "resources": [{"formal": "input", "effects": [{"kind": "read", "lower": [0], "upper": [extent]}]},
                      {"formal": "output", "effects": [{"kind": "overwrite", "lower": [0], "upper": [extent]}]}],
        "configuration": [{"component": "extent", "kind": "int64", "values": [extent]},
                          {"component": "coefficient", "kind": "real64", "values": [3.0]}]}
    provider["numerical_contract"]["task_contracts"] = [task_scope_identity(task)]
    manifest.write_text(json.dumps(provider))
    task["provider_manifest"]["sha256"] = digest(manifest)
    analysis.contracts[routine.qualified] = {"device_tasks": [task]}
    return path, analysis, routine, node, task, manifest, implementation


def test_exact_call_and_explicit_qualification_do_not_grant_definitions_or_lifetime(tmp_path):
    _, analysis, routine, node, _, _, _ = fixture(tmp_path)
    proof = external_device_task(analysis, routine, node)
    public = proof.public()
    assert proof.validate(analysis, routine, node) is proof
    assert public["canonical_resources"] == {"input": "argument::b", "output": "argument::a"}
    assert public["definitions_authorized"] is False
    assert public["placement_authorized"] is False
    assert public["source_association_authorized"] is True
    public["definitions_authorized"] = True
    assert proof.public()["definitions_authorized"] is False
    with pytest.raises(CompilationError, match="exact compiler-issued"):
        copy(proof).validate(analysis, routine, node)
    with pytest.raises(CompilationError, match="exact compiler-issued"):
        proof.validate(analysis, routine, copy(node))


@pytest.mark.parametrize("field", ["sha256", "call_sha256", "callee_body_sha256", "callee"])
def test_wrong_original_source_or_callee_cannot_be_substituted(tmp_path, field):
    _, analysis, routine, node, task, _, _ = fixture(tmp_path)
    task["source"][field] = "wrong"
    with pytest.raises(CompilationError, match="original CALL/body"):
        external_device_task(analysis, routine, node)


def test_provider_and_sidecar_mutation_invalidate_already_issued_proof(tmp_path):
    _, analysis, routine, node, task, _, implementation = fixture(tmp_path)
    proof = external_device_task(analysis, routine, node)
    before = implementation.read_bytes()
    implementation.write_text("! Changed code\n")
    with pytest.raises(CompilationError, match="artifact changed"):
        proof.validate(analysis, routine, node)
    implementation.write_bytes(before)
    task["configuration"][0]["values"][0]["offset"] = 1
    with pytest.raises(CompilationError, match="contract changed"):
        proof.validate(analysis, routine, node)


@pytest.mark.parametrize("expression", ["size(input)", {"descriptor": "size", "formal": "missing", "axis": 1, "offset": 0},
    {"descriptor": "size", "formal": "input", "axis": 2, "offset": 0},
    {"descriptor": "payload", "formal": "input", "axis": 1, "offset": 0}])
def test_configuration_never_accepts_free_expressions_payload_or_foreign_descriptors(tmp_path, expression):
    _, analysis, routine, node, task, manifest, _ = fixture(tmp_path)
    task["configuration"][0]["values"] = [expression]
    requalify(task, manifest)
    with pytest.raises(CompilationError):
        external_device_task(analysis, routine, node)


@pytest.mark.parametrize("change", ["alias", "optional", "rank", "effects", "component", "width"])
def test_unproved_alias_presence_layout_effect_and_configuration_fail_closed(tmp_path, change):
    source = SOURCE.replace("original_transform(b,a)", "original_transform(a,a)") if change == "alias" else SOURCE
    if change == "optional":
        source = source.replace("intent(inout)::a(:),b(:)", "intent(inout),optional::a(:),b(:)")
    _, analysis, routine, node, task, manifest, _ = fixture(tmp_path, source)
    if change == "rank":
        task["resources"][0]["effects"][0]["lower"] = [0, 0]
    if change == "effects":
        task["resources"][0]["effects"][0]["kind"] = "all_initialized"
    if change == "component":
        task["configuration"][0]["component"] = "extent; call hidden()"
    if change == "width":
        task["configuration"][0].update(kind="int32", values=[1 << 40])
    requalify(task, manifest)
    with pytest.raises(CompilationError):
        external_device_task(analysis, routine, node)


def test_no_contract_does_not_reinterpret_an_original_call(tmp_path):
    _, analysis, routine, node, _, _, _ = fixture(tmp_path)
    analysis.contracts.clear()
    assert external_device_task(analysis, routine, node) is None


@pytest.mark.parametrize("extra", ["write(*,*) 'host effect'", "call opaque(output)",
    "block\ninteger,save::counter=0\ncounter=counter+1\nend block"])
def test_qualification_hash_cannot_erase_original_host_effects(tmp_path, extra):
    _, analysis, routine, node, _, _, _ = fixture(tmp_path, SOURCE.replace("output=3.0_8*input",
                                                                                      extra+"\noutput=3.0_8*input"))
    with pytest.raises(CompilationError, match="external device first slice"):
        external_device_task(analysis, routine, node)


def emit(tmp_path, *, source=SOURCE, policy="sections"):
    path, analysis, _, _, _, _, _ = fixture(tmp_path, source)
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): digest(path)},
             "captures": {"argument::a": dict(FACT), "argument::b": dict(FACT)}}
    builder = ScopeBuilder([path], "external_fixture::advance", facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy=policy, scope_execution="graph"), contracts=analysis.contracts)
    outputs, report = builder.run()
    return path, outputs, report


def test_graph_emits_exact_external_call_with_unpriced_provider_and_prework_only_fallback(tmp_path):
    path, outputs, report = emit(tmp_path)
    owner, = report["scopes"]
    assert owner["boundaries"] == []
    assert owner["estimate_available"] is False
    task, = owner["external_device_tasks"]
    epoch, = owner["checked_execution_plan"]["storage_epoch"]["members"]
    assert epoch["device_tasks"] == [task]
    graph, = owner["graph_execution"]["procedures"]
    external, = [item for item in graph["task_graph"]["tasks"] if ":external:" in item["id"]]
    assert external["may_writes"] == ["argument::a"]
    assert not external.get("must_overwrites")
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert text.count("call original_transform(b,a)") == 1
    assert "fort_scope_domain_get_v1(" in text
    assert "external device work failed; replay prohibited" in text
    assert "fort_external_report%started /= 0" in text
    assert text.index("fort_external_destroy_status = fort_external_destroy(") < text.index("fort_scope_resident_close_v1(")


def test_v1_provider_cannot_outlive_cleanup_control_in_another_borrowed_member(tmp_path):
    child = """subroutine local_child(v,n)
real(8),intent(inout)::v(:)
integer,intent(in)::n
integer,save::counter=0
integer::i
counter=counter+1
do i=1,n
v(i)=v(i)+0.0_8
enddo
end subroutine
"""
    source = SOURCE.replace("call original_transform(b,a)", "call original_transform(b,a)\ncall local_child(a,n)")
    source = source.replace("end module", child+"end module")
    _, _, report = emit(tmp_path, source=source)
    assert any("v1 external provider cleanup" in item["reason"] for item in report["boundaries"])
    assert all(not item.get("external_device_tasks") for item in report["scopes"])


PROVIDER_INTERFACE = """module external_provider
use iso_c_binding
use fort_scoped_memory,only:fort_scope_external_report_v1
implicit none
type,bind(C)::provider_config
integer(c_int64_t)::extent=0
real(c_double)::coefficient=0
end type
interface
function provider_create(domain,config,handle)bind(C)result(status)
import c_int,c_int64_t,c_ptr,provider_config
integer(c_int64_t),value::domain
type(provider_config),intent(in)::config
type(c_ptr),intent(out)::handle
integer(c_int)::status
end function
function provider_execute(context,handle,input,output,report)bind(C)result(status)
import c_int,c_int64_t,c_ptr,fort_scope_external_report_v1
integer(c_int64_t),value::context,input,output
type(c_ptr),value::handle
type(fort_scope_external_report_v1),intent(out)::report
integer(c_int)::status
end function
function provider_destroy(handle)bind(C)result(status)
import c_int,c_ptr
type(c_ptr),intent(inout)::handle
integer(c_int)::status
end function
end interface
end module
"""


@pytest.mark.native
def test_external_lifecycle_and_exact_original_fallback_compile_in_real_fortran(tmp_path):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    _, outputs, report = emit(tmp_path)
    build = tmp_path / "build"
    build.mkdir()
    for name, text in outputs.items():
        destination = build / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
    interface = build / "external_provider.f90"
    interface.write_text(PROVIDER_INTERFACE)
    artifacts = [item for item in report["build_sources"] if item["language"] == "fortran"]
    runtime = [item for item in artifacts if item["role"] == "common_runtime"]
    remainder = [item for item in artifacts if item["role"] != "common_runtime"]
    paths = [*(build / item["path"] for item in runtime), interface, *(build / item["path"] for item in remainder)]
    for index, path in enumerate(paths):
        result = subprocess.run([fortran, "-std=f2018", "-fopenmp", "-fcheck=all", "-J", str(build), "-I", str(build),
                                 "-c", str(path), "-o", str(build / f"source-{index}.o")],
                                cwd=build, text=True, capture_output=True, timeout=30)
        (build / f"source-{index}.stderr").write_text(result.stderr)
        assert result.returncode == 0, result.stderr
