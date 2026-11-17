"""Reachable original IEEE state restricts numerical work before outlining."""

from hashlib import sha256
from types import SimpleNamespace

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.lexical import immutable_numerical_unit
from compiler.scopes.segments import Segment
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_lexical_source_owner import MODULE_SOURCE, SOURCE, emit
from compiler.tests.test_reached_counted_loops import builder_for
from compiler.tests.test_reached_stage_prerequisites_cuda import sources
from compiler.tests.test_source_scopes import FACT


def numerical_regions(report):
    registries = [report["inline_numerical_regions"], *report["child_numerical_regions"]]
    return [region for registry in registries for region in registry["regions"] if region["used"]]


def numerical_segments(owner):
    members = [owner, *owner["module_coordinators"], *owner["internal_coordinators"]]
    return [segment for member in members for unit in member["planning_segments"]
            for segment in unit["operations"]["planning_segments"]]


def trap_source():
    # With a=+Inf, the producer writes -Inf while traps are disabled. The
    # original consumer must retain the restored invalid-operation trap.
    return (SOURCE.replace("subroutine step(a,b,out,n,escape)",
                           "subroutine step(a,b,out,n,escape)\nuse ieee_exceptions")
            .replace("visits=visits+1", "visits=visits+1\ncall ieee_set_halting_mode(ieee_invalid,.false.)")
            .replace("b(i)=2*a(i)+real(i,8)", "b(i)=-a(i)")
            .replace("if(escape) then\ncall opaque(b,n)\nendif",
                     "call ieee_set_halting_mode(ieee_invalid,.true.)"))


def test_immutable_unit_accepts_only_identical_registered_fallback():
    call = SimpleNamespace(node=object())
    fallback = SimpleNamespace(kind="guarded original numerical fallback")
    scope = SimpleNamespace(calls=[call], tree=[Segment([call])], native=[fallback],
                            guarded={id(call.node): fallback})
    assert immutable_numerical_unit(scope)
    scope.native.append(SimpleNamespace(kind="native source"))
    assert not immutable_numerical_unit(scope)
    scope.native = [SimpleNamespace(kind=fallback.kind)]
    assert not immutable_numerical_unit(scope)
    scope.native = [fallback]
    scope.guarded = {id(object()): fallback}
    assert not immutable_numerical_unit(scope)
