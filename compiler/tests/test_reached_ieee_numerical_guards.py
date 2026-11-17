"""Reachable original IEEE state restricts numerical work before outlining."""

from types import SimpleNamespace

from compiler.scopes.lexical import immutable_numerical_unit
from compiler.scopes.segments import Segment


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
