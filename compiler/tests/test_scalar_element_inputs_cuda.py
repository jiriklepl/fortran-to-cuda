"""An original element association survives selector changes and GPU workers."""

import os
import re
from copy import deepcopy
from hashlib import sha256

import pytest

from compiler._legacy.config import LegacyCompilerOptions as CompilerOptions
from compiler._legacy.config import LegacyOffloadConfig as OffloadConfig
from compiler.frontend.numerical_preparation import NumericalPreparationProvider
from compiler.frontend.reached_windows import plan_reached_window
from compiler.scopes.reached_owner import ReachedWindowOwner
from compiler.scopes.source import ScopeBuilder
from compiler.tests.support.numerical_cuda import (
    _transfers,
    build_reached_numerical_binaries,
    check_prepared_chain,
)
from compiler.tests.support.numerical_regions import (
    CHILD_SOURCE,
    ENTRY,
    FACT,
    MATH_SOURCE,
    OWNER_SOURCE,
    ROOTS,
)
from compiler.tests.test_source_scopes import run

LOOKUP_SOURCE = """module quartz_lookup
implicit none
real(8)::weights(-1:1)=[0.5_8,1.25_8,2.0_8]
integer::selector=0
end module
"""


def emit_scalar_element_fixture(directory, *, precision=8, device_budget=None, alias_prefix=False):
    directory.mkdir(parents=True, exist_ok=True)
    child = CHILD_SOURCE.replace("module quartz_numerics\n",
        "module quartz_numerics\nuse quartz_lookup,only:selector\n").replace(
        "subroutine bridge(a,b,c,n,tile)", "subroutine bridge(a,b,c,n,tile,ratio)").replace(
        "integer::i,block", "real(8),intent(in)::ratio\ninteger::i,block").replace(
        "bridge_visits=bridge_visits+1", "bridge_visits=bridge_visits+1\nselector=-1").replace(
        "b(i)=s+real(i,8)", "b(i)=s+ratio*real(i,8)")
    parent = OWNER_SOURCE.replace("module quartz_owner\n",
        "module quartz_owner\nuse quartz_lookup,only:weights,selector,fort_actual_status=>weights\n").replace(
        "prefix_visits=prefix_visits+1", "prefix_visits=prefix_visits+1\nselector=mod(prefix_visits,3)-1").replace(
        "call renamed_bridge(a,b,c,n,tile)", "call renamed_bridge(a,b,c,n,tile,weights(selector))")
    if alias_prefix:
        parent = parent.replace("prefix_visits=prefix_visits+1", "integer::i\nprefix_visits=prefix_visits+1").replace(
            "selector=mod(prefix_visits,3)-1", "selector=0").replace(
            "call renamed_bridge(a,b,c,n,tile,weights(selector))",
            "do i=-3,n\nb(i)=b(i)+a(i)+1.0_8\nenddo\ncall renamed_bridge(a,b,c,n,tile,weights(selector))")
        child = child.replace("b(i)=s+ratio*real(i,8)", "b(i)=b(i)+s+ratio*real(i,8)")
    paths = tuple(directory / name for name in (
        "quartz_lookup.f90", "quartz_math.f90", "quartz_numerics.f90", "quartz_owner.f90"))
    for path, source in zip(paths, (LOOKUP_SOURCE, MATH_SOURCE, child, parent), strict=True):
        path.write_text(source.replace("real(8)", f"real({precision})")
                        .replace("_8", f"_{precision}").replace(",8)", f",{precision})"))
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths},
             "captures": {ROOTS[0]: deepcopy(FACT), ROOTS[1]: deepcopy(FACT),
                          ROOTS[2]: {**deepcopy(FACT), "initialized": "none", "lifetime": "selected_window"}}}
    if device_budget is not None:
        facts["device_budget_bytes"] = device_budget
    builder = ScopeBuilder(paths, ENTRY, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="reached"))
    originals = tuple(node for node in builder.entry.execution.content if type(node).__name__ in (
        {"Block_Nonlabel_Do_Construct", "Call_Stmt"} if alias_prefix else {"Call_Stmt"}))
    assert len(originals) == (2 if alias_prefix else 1)
    assert "weights(selector)" in str(originals[-1]).replace(" ", "").lower()
    provider = NumericalPreparationProvider(builder.analysis, options=builder.options, config=builder.config,
        environment_root=builder._numerical_environment_root, immutable_inputs={})
    proof = plan_reached_window(builder.analysis, ENTRY, originals, ROOTS, facts, numerical_provider=provider)
    owner = ReachedWindowOwner(builder, proof, originals)
    assert owner.run()
    outputs, report = owner.builder.finish()
    resources = report["scopes"][0]["resource_bindings"]["resources"]
    assert {item["resource"] for item in resources
            if item.get("role") != "opaque_scalar_element_alias_reservation"} == set(ROOTS)
    reservation, = (item for item in resources if item["resource"] == "quartz_lookup::weights")
    assert reservation["role"] == "opaque_scalar_element_alias_reservation"
    assert reservation["managed_payload"] is False
    assert reservation["definition_authority"] is False
    return paths, outputs, report, owner


@pytest.fixture(scope="module", params=((4, False), (4, True), (8, False), (8, True)))
def scalar_element_binaries(tmp_path_factory, request):
    precision, resource_fallback = request.param
    return build_reached_numerical_binaries(tmp_path_factory, precision, resource_fallback,
        emit_fixture=emit_scalar_element_fixture, prefix="scalar_element_")


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("path", ["eligible", "negative", "overflow", "empty", "late"])
def test_original_element_association_stays_coherent_without_replay(scalar_element_binaries, disabled, path):
    check_prepared_chain(scalar_element_binaries, disabled, path)


def alias_driver(precision):
    return f"""program verify
use iso_fortran_env,only:error_unit
use quartz_lookup,only:weights
use quartz_owner,only:advance,prefix_visits,tail_visits
use quartz_numerics,only:bridge_visits,middle_visits
implicit none
real({precision})::b(-4:-2),c(-4:-2)
integer::unit,repeat
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do repeat=1,2
weights=[0.5_{precision},1.25_{precision},2.0_{precision}]
b=-77._{precision}
c=-91._{precision}
write(error_unit,'(a,i0)') 'BEGIN_CALL ',repeat
! The table and a are read-only aliases throughout advance. The GPU prefix
! changes disjoint b; the child consumes that value, exposing any replay.
call advance(weights,b,c,-3,3)
write(error_unit,'(a,i0)') 'END_CALL ',repeat
if(prefix_visits/=repeat.or.bridge_visits/=repeat.or.tail_visits/=repeat) error stop 'prefix replayed'
if(middle_visits/=repeat) error stop 'native child replayed'
if(weights(0)/=1.25_{precision}) error stop 'read-only table changed'
if(b(-4)/=-77._{precision}.or.b(-2)/=-77._{precision}) error stop 'producer halo changed'
if(c(-4)/=-91._{precision}.or.c(-2)/=-91._{precision}) error stop 'consumer halo changed'
write(unit)repeat,prefix_visits,bridge_visits,middle_visits,tail_visits
write(unit)weights,b,c
enddo
close(unit)
end program
"""


@pytest.fixture(scope="module", params=(4, 8))
def scalar_element_alias_binaries(tmp_path_factory, request):
    def emit(directory, **options):
        return emit_scalar_element_fixture(directory, alias_prefix=True, **options)

    return build_reached_numerical_binaries(tmp_path_factory, request.param, False,
        emit_fixture=emit, prefix="scalar_element_alias_", driver_source=alias_driver, expected_gpu_leaves=4)


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
def test_physical_alias_publishes_gpu_prefix_before_original_call(scalar_element_alias_binaries, disabled):
    directory, candidate, native, precision, _fallback, resources = scalar_element_alias_binaries
    environment = {**os.environ, "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    reference = run([str(native)], cwd=directory, env=environment)
    expected = (directory / "fields.bin").read_bytes()
    result = run([str(candidate)], cwd=directory, env=environment)
    actual = (directory / "fields.bin").read_bytes()
    label = "alias-disabled" if disabled else "alias-cuda"
    (directory / (label + "-native.bin")).write_bytes(expected)
    (directory / (label + "-candidate.bin")).write_bytes(actual)
    (directory / (label + ".stderr")).write_text(result.stderr)
    assert actual == expected
    assert "array temporary" not in reference.stderr.lower() + result.stderr.lower()
    traces = re.findall(r"BEGIN_CALL (\d+)\n(.*?)END_CALL \1\n", result.stderr, re.DOTALL)
    assert [int(ordinal) for ordinal, _trace in traces] == [1, 2]
    input_identity = resources["argument::a"]["registration_identity"]
    output_identity = resources["argument::b"]["registration_identity"]
    for _ordinal, trace in traces:
        assert trace.count("FORT_SCOPED launch ") == (0 if disabled else 1)
        transfers = _transfers(trace)
        expected_upload = {} if disabled else {input_identity: precision, output_identity: precision}
        expected_download = {} if disabled else {output_identity: precision}
        assert dict(transfers["upload"]) == expected_upload
        assert dict(transfers["download"]) == expected_download
