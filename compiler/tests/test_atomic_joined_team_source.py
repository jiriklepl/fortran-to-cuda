"""Whole-source admission/emission preserves one atomic original alternative."""
import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_atomic_worksharing_team import BODY
from compiler.tests.test_source_scopes import FACT

SOURCE = ("module atomic_owner\ncontains\nsubroutine step(a,b,p,n,mode,flag)\n"
          "real(8),intent(inout)::a(:),b(:)\nreal(8),intent(in)::p(0:)\n"
          "integer,intent(in)::n,mode\nlogical,intent(in)::flag\ninteger::i\n"
          "!$omp parallel private(i)\n" + BODY + "!$omp end parallel\nend subroutine\nend module\n")


def emit(tmp_path, *, policy="sections", source=SOURCE, extra_captures=None):
    path = tmp_path / "owner.f90"
    path.write_text(source)
    builder = ScopeBuilder([path], "atomic_owner::step", options=CompilerOptions(opt_level=0),
        config=OffloadConfig(policy=policy, scope_execution="graph", host_threads=4),
        facts={"schema_version": 1, "participation": "serial",
               "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
               "captures": {**{"argument::" + name: dict(FACT) for name in ("a", "b", "p")},
                            **(extra_captures or {})}})
    outputs, report = builder.run()
    replacement = report["sources"].get(str(path), {}).get("replacement")
    return outputs, report, outputs[replacement] if replacement else source


def test_complete_conditional_team_uses_one_source_task_and_three_normal_variants(tmp_path):
    outputs, report, text = emit(tmp_path)
    scope, = report["scopes"]
    task, = scope["source_device_tasks"]
    assert task["kind"] == "original_atomic_conditional_joined_team"
    assert len(task["alternatives"]) == 3
    assert task["source_proof"]["alternatives"] == [[0, 1], [2, 3], [2, 4]]
    assert scope["estimate_available"] is False
    assert "select case(fort_atomic_variant)" in text
    assert text.index("IF (mode == 4) THEN") < text.index("select case(fort_atomic_variant)")
    assert "atomic original team attempted work; replay prohibited" in text
    assert text.count("!$omp parallel private(i)") == 1
    providers = [value for name, value in outputs.items() if name.startswith("fort_atomic_") and name.endswith(".cu")]
    assert len(providers) == 3
    assert all(value.count("fort_scope_source_execute_v1(") == 1 for value in providers)
    assert all("fort_access.begin(false)" not in value for value in providers)
    assert "bind(C, name='fort_scope_task_graph_enable_v1')" in text
    setup = text.index(" = fort_enable_checked_graph(")
    assert setup < text.index("fort_scope_register") < text.index("fort_atomic_status=")


def test_every_speculative_child_publishes_its_atomic_providers_to_the_owner(tmp_path):
    head = "module atomic_owner\ncontains\n"
    child = SOURCE.removeprefix(head).removesuffix("end module\n")
    source = (head + "subroutine run(a,b,p,n,mode,flag)\n"
              "real(8),intent(inout)::a(:),b(:)\nreal(8),intent(in)::p(0:)\n"
              "integer,intent(in)::n,mode\nlogical,intent(in)::flag\n"
              + "".join(f"call {name}(a,b,p,n,mode,flag)\n" for name in ("first", "second", "third"))
              + "end subroutine\n"
              + "".join(child.replace("subroutine step(", f"subroutine {name}(")
                        for name in ("first", "second", "third")) + "end module\n")
    path = tmp_path / "owner.f90"
    path.write_text(source)
    builder = ScopeBuilder([path], "atomic_owner::run", options=CompilerOptions(opt_level=0),
        config=OffloadConfig(policy="sections", scope_execution="graph", host_threads=4),
        facts={"schema_version": 1, "participation": "serial",
               "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
               "captures": {"argument::"+name: dict(FACT) for name in ("a", "b", "p")}})
    outputs, report = builder.run()
    scope, = report["scopes"]
    tasks = scope["source_device_tasks"]
    assert len(tasks) == 3
    providers = [alternative["provider"] for task in tasks for alternative in task["alternatives"]]
    assert len({provider["entry"] for provider in providers}) == 9
    for provider in providers:
        for name, digest in ((provider["entry"]+".cu", provider["cuda_source_sha256"]),
                             (provider["module"]+".f90", provider["fortran_source_sha256"])):
            assert sha256(outputs[name].encode()).hexdigest() == digest
            assert report["artifacts_sha256"][name] == digest
    assert {"common_functions.cuh", "numeric.hpp"} <= outputs.keys()

    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    if fc is None:
        pytest.skip("Fortran compilation toolchain required")
    for name, text in outputs.items():
        target = tmp_path / "generated" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    generated = tmp_path / "generated"
    roles = {name: index for index, name in enumerate(("common_runtime", "shared_entry", "original_source"))}
    artifacts = sorted(enumerate(report["build_sources"]), key=lambda pair: roles[pair[1]["role"]])
    for index, artifact in artifacts:
        if artifact["language"] != "fortran":
            continue
        result = subprocess.run([fc, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-ffree-line-length-none",
            "-J", str(generated), "-I", str(generated), "-c", str(generated/artifact["path"]),
            "-o", str(generated/f"host-{index}.o")], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr


def test_auto_without_atomic_cost_proof_retains_original_source(tmp_path):
    outputs, report, text = emit(tmp_path, policy="auto")
    assert not any(name.startswith("fort_atomic_") for name in outputs)
    assert not any(scope.get("source_device_tasks") for scope in report["scopes"])
    assert "!$omp end do nowait" in text


def test_dependency_decline_never_admits_a_partial_team(tmp_path):
    source = SOURCE.replace("b(i)=b(i)+p(i+1)", "b(i)=a(i)+p(i+1)")
    outputs, report, text = emit(tmp_path, source=source)
    assert not any(name.startswith("fort_atomic_") for name in outputs)
    assert not any(scope.get("source_device_tasks") for scope in report["scopes"])
    assert "b(i)=a(i)+p(i+1)" in text


def test_branch_only_allocatable_field_remains_in_complete_epoch_without_early_registration(tmp_path):
    source = (SOURCE.replace("step(a,b,p,n,mode,flag)", "step(a,b,p,n,mode,flag,r)")
        .replace("integer::i", "integer::i\nreal(8),allocatable,intent(in)::r(:)")
        .replace("b(i)=b(i)+p(i)", "b(i)=b(i)+r(i)"))
    _outputs, report, text = emit(tmp_path, source=source,
        extra_captures={"argument::r": {**FACT, "initialized": "none"}})
    scope, = report["scopes"]
    epoch = scope["checked_execution_plan"]["storage_epoch"]
    field, = [r for r in epoch["resources"] if r["resource"] == "argument::r"]
    assert field["device_replica_authorized"] is True
    assert "initialized" not in field
    assert len(scope["source_device_tasks"][0]["alternatives"]) == 3
    first, selected, last = text.split("select case(fort_atomic_variant)", 1)[1].split("case(", 3)[1:]
    assert "allocated(r)" not in first
    assert "allocated(r)" in selected
    assert "allocated(r)" not in last.split("end select", 1)[0]
