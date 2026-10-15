"""Public lexical ownership uses demanded native effects without a full closure."""

from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_offload_profile import scoped_profile
from compiler.tests.test_source_scopes import FACT

SOURCE = """module demand_owner
implicit none
type profile
real(8),allocatable::flux(:)
integer::visits=0
end type
type(profile)::metadata
real(8)::a(-1:6),b(-1:6),star(-1:6),out(-1:6)
contains
subroutine advance()
integer::i
do i=0,5
b(i)=2*a(i)
enddo
call correct(-0.5d0*a(0))
do i=0,5
star(i)=b(i)+a(i)
enddo
do i=0,5
out(i)=out(i)+star(i)
enddo
end subroutine
subroutine correct(scale)
real(8),intent(in)::scale
b(0)=b(0)+scale+metadata%flux(1)
metadata%flux(1)=metadata%flux(1)+1.d0
metadata%visits=metadata%visits+1
end subroutine
end module
"""


def generate(directory, source=SOURCE, *, policy="sections", captures=None):
    path = directory / "owner.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"demand_owner::" + name: dict(FACT) for name in ("a", "b", "star", "out")}}
    facts["captures"].update(captures or {})
    profile = scoped_profile() if policy == "auto" else None
    builder = ScopeBuilder([path], "demand_owner::advance", facts=facts,
                           options=CompilerOptions(gpu_policy=policy, memory_model="scoped"),
                           config=OffloadConfig(policy=policy, scope_execution="reached", profile=profile))
    outputs, report = builder.run()
    return path, builder, outputs, report


def demanded_operations(report):
    return [operation for scope in report["scopes"]
            for procedure in scope.get("checked_execution_plan", {}).get("procedures", [])
            for unit in procedure["units"] for operation in unit["operations"]["native_operations"]
            if operation.get("demand_relative_effects") is not None]


def test_public_owner_retains_original_call_and_later_resources(tmp_path):
    path, builder, outputs, report = generate(tmp_path)
    scope, = report["scopes"]
    operation, = demanded_operations(report)
    assert not scope["boundaries"]
    assert scope["checked_execution_plan"]["numerical_invocations"] == 3
    assert operation["preserves_original_source"]
    assert operation["resource_effects"] == {"demand_owner::a": ["read"],
                                             "demand_owner::b": ["read", "write"]}
    assert not operation["sections"]["available"]
    assert operation["demand_relative_effects"]["native_completion"]["available"]
    assert not operation["demand_relative_effects"]["cost_estimate_available"]
    prospective, = (demand for demand in builder._resource_demands.values() if demand._plan is None)
    assert {"demand_owner::star", "demand_owner::out"} <= {
        binding.root for binding in prospective.managed_bindings}
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert text.count("call correct(-0.5d0*a(0))") == 1
    assert text.count("metadata%flux(1)=metadata%flux(1)+1.d0") == 1
    assert "fort_scope_host_begin" in text
    assert "fort_scope_host_end" in text
    assert all("metadata" not in row["resource"] for row in scope["checked_execution_plan"]["resources"])


def test_auto_requires_native_interval_observations_and_preserves_source(tmp_path):
    path, _builder, outputs, report = generate(tmp_path, policy="auto")
    scope, = report["scopes"]
    assert demanded_operations(report)
    assert scope["automatic_preflight"]["selection"] == "native"
    assert not scope["estimate_available"]
    assert "native" in scope["planning_reason"].lower()
    requirements = scope["checked_execution_plan"]["cost_requirements"]
    assert any("source" in item["kind"] and "native" in item["kind"] for item in requirements)
    assert outputs[report["sources"][str(path)]["replacement"]] == SOURCE


@pytest.mark.parametrize("mutation", ["pointer", "reallocate", "unknown_call"])
def test_unproved_native_storage_or_execution_keeps_original_boundary(tmp_path, mutation):
    source = SOURCE
    if mutation == "pointer":
        source = source.replace("real(8),allocatable::flux(:)", "real(8),pointer::flux(:)")
    elif mutation == "reallocate":
        source = source.replace("metadata%flux(1)=metadata%flux(1)+1.d0", "metadata%flux=[1.d0,2.d0]")
    else:
        source = source.replace("metadata%visits=metadata%visits+1", "call opaque(metadata)")
    path, _builder, outputs, report = generate(tmp_path, source)
    assert not demanded_operations(report)
    assert any(scope.get("boundaries") for scope in report["scopes"])
    assert outputs[report["sources"][str(path)]["replacement"]].count("call correct(-0.5d0*a(0))") == 1


def test_partial_entry_definitions_cannot_use_conservative_whole_native_access(tmp_path):
    _path, _builder, _outputs, report = generate(tmp_path, captures={
        "demand_owner::b": {**FACT, "initialized": "none"}})
    assert not demanded_operations(report)
    assert any("whole initialized" in item["reason"] for scope in report["scopes"]
               for item in scope.get("boundaries", []))


def test_future_child_resource_omission_rejects_entire_demanded_candidate(tmp_path):
    source = SOURCE.replace("real(8)::a(-1:6)", "real(8)::extra(-1:6)\nreal(8)::a(-1:6)").replace(
        "call correct(-0.5d0*a(0))", "call correct(-0.5d0*a(0))\ncall later()").replace(
        "end module", """subroutine later()
integer::j
do j=0,5
extra(j)=2*a(j)
enddo
end subroutine
end module""")
    path, _builder, outputs, report = generate(tmp_path, source,
        captures={"demand_owner::extra": dict(FACT)})
    assert not demanded_operations(report)
    assert "resource demand misses final" in report["lexical_owner_attempt"]["error"]
    assert outputs[report["sources"][str(path)]["replacement"]].count("call correct(-0.5d0*a(0))") == 1
