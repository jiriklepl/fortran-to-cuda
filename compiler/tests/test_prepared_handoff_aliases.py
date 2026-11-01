"""Two checked groups may share one immutable original numerical authority."""
import json
import re
import shutil
import subprocess
from contextlib import contextmanager
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes import execution_plan
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_owner_local_call_groups_runtime import SOURCE


@contextmanager
def original_epoch_context(plan):
    builder = plan._owner.builder
    previous = getattr(builder, "_graph_storage_epoch", None)
    builder._graph_storage_epoch = plan._owner.position_epoch
    try:
        yield
    finally:
        builder._graph_storage_epoch = previous


def compile_published_fortran(directory, outputs):
    compiler = shutil.which("gfortran-15")
    if compiler is None:
        pytest.skip("GNU Fortran required for complete generated module validation")
    directory.mkdir()
    pending, modules = [], {}
    for name, source in outputs.items():
        path = directory/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        if name.endswith(".f90"):
            declared = {name.lower() for name in re.findall(r"(?im)^\s*module\s+(?!procedure\b)(\w+)", source)}
            used = {name.lower() for name in re.findall(r"(?im)^\s*use(?:\s*,\s*intrinsic)?\s*(?:::\s*)?(\w+)", source)}
            pending.append((path, declared, used))
            for module in declared:
                assert module not in modules
                modules[module] = path
    compiled, commands = set(), []
    while pending:
        ready = [item for item in pending if not {name for name in item[2] if name in modules} - compiled - item[1]]
        assert ready, "generated Fortran module dependency cycle"
        for path, declared, _used in ready:
            argv = [compiler, "-fopenmp", "-ffree-line-length-none", "-I", str(directory), "-J", str(directory),
                    "-c", str(path), "-o", str(directory/f"unit-{len(commands)}.o")]
            result = subprocess.run(argv, cwd=directory, capture_output=True, text=True, timeout=60)
            commands.append({"argv": argv, "returncode": result.returncode,
                             "stdout": result.stdout, "stderr": result.stderr})
            (directory/"commands.json").write_text(json.dumps(commands, indent=2)+"\n")
            assert result.returncode == 0, result.stdout+result.stderr
            compiled.update(declared)
            pending.remove((path, declared, _used))
    assert compiled


@pytest.mark.parametrize("swap_actuals", [False, True])
def test_two_original_groups_retain_prepared_aliases_and_both_checked_plans(tmp_path, monkeypatch, swap_actuals):
    calls = "call first(b,c,n)\ncall second(a,b,c,n)\ncall third(a,c,n)"
    alternative = "call first(a,c,n)\ncall second(b,a,c,n)\ncall third(b,c,n)" if swap_actuals else calls
    source = SOURCE.replace(calls+"\nendif", calls+"\nelse\n"+alternative+"\nendif")
    path = tmp_path/"two_groups.f90"
    path.write_text(source)
    fact = {"storage": "stable", "initialized": "whole", "escapes": False,
            "allocation_changes": False, "lifetime": "invocation"}
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::"+name: dict(fact) for name in ("a", "b", "c")}}
    contracts = {"live_groups::run": {"native_ancestor": {"version": 2,
        "lifetime": "owner_local_call_groups", "device_roots": ["live_groups::"+name
            for name in ("first", "second", "third")]}}}
    plans = []
    original = execution_plan.build_execution_plan

    def retain(owner):
        plan = original(owner)
        plans.append(plan)
        return plan

    monkeypatch.setattr(execution_plan, "build_execution_plan", retain)
    builder = ScopeBuilder([path], "live_groups::run", facts=facts, contracts=contracts,
                           options=CompilerOptions(), config=OffloadConfig(policy="sections"))
    outputs, report = builder.run()
    assert report["scope_count"] == 2, report["boundaries"]
    assert report["boundaries"] == []
    assert outputs
    assert len(plans) == 2
    shared = {id(proof): proof for proof in plans[0]._numerical_authorities}
    shared = [proof for proof in plans[1]._numerical_authorities if shared.get(id(proof)) is proof]
    assert shared
    for plan in plans:
        for handoff in plan._proof_handoffs:
            assert all(id(getattr(handoff.proof, name)) == identity for name, identity in handoff.state)
        with original_epoch_context(plan):
            plan.validate()  # The second seal must not invalidate the first plan.
    for proof in shared:
        assert proof.completion is proof.extraction.completion
        assert proof.preconditions is proof.extraction.preconditions
        proof.validate(proof._provider._analysis, proof.procedure, proof._selection)
        with pytest.raises(CompilationError, match="immutable"):
            proof.completion["injected"] = True
    compile_published_fortran(tmp_path/"compiled", outputs)
    proof = shared[0]
    object.__setattr__(proof, "completion", dict(proof.completion))
    with pytest.raises(CompilationError, match="authority changed"):
        proof.validate(proof._provider._analysis, proof.procedure, proof._selection)
    with original_epoch_context(plans[0]), pytest.raises(CompilationError):
        plans[0].validate()
