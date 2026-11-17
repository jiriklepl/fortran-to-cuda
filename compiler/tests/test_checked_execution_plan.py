"""Placement and emission consume one admitted original operation inventory."""

from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.execution_plan import build_execution_plan
from compiler.scopes.lexical import LexicalOwner
from compiler.scopes.ordinary_consumption import _OrdinaryPairConsumption
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_lexical_source_owner import SOURCE
from compiler.tests.test_source_scopes import FACT


@contextmanager
def scanned(directory, before_plan=None, *, config=None, source=SOURCE):
    path = directory / "owner.f90"
    path.write_text(source.replace("call opaque(b,n)", "b(-2)=b(-2)+sum(a)"))
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: deepcopy(FACT) for name in ("a", "b", "out")}}
    builder = ScopeBuilder([path], "local_owner::step", facts=facts, options=CompilerOptions(),
                           config=config or OffloadConfig(policy="sections"))
    nodes = builder.inline.prepare(builder.entry.execution.content)
    builder.generated.update(builder.inline.generated)
    builder.numerical_ir.update(builder.inline.ir)
    builder.numerical_reasons.update({name: None for name in builder.inline.regions})
    with _OrdinaryPairConsumption(builder):
        owner = LexicalOwner(builder)
        owner.scan(nodes)
        owner.validate_plan(nodes)
        assert not owner.boundaries
        if before_plan is not None:
            before_plan(owner)
        yield owner, build_execution_plan(owner)


def test_immutable_handoff_avoids_rederiving_geometry_and_rechecks_registry(tmp_path, monkeypatch):
    from compiler.frontend.numerical_preparation import PreparedOriginalRegion

    with scanned(tmp_path, config=OffloadConfig(policy="sections")) as (_owner, plan):
        proof = plan._numerical_authorities[0]

        def unexpected(*_args, **_kwargs):
            pytest.fail("immutable emission handoff recomputed original numerical geometry")

        monkeypatch.setattr(PreparedOriginalRegion, "validate", unexpected)
        assert plan.validate() is plan
        assert plan.validate() is plan
        registry = proof._provider._analysis._numerical_preparations
        assert registry.pop(proof.identity) is proof
        try:
            with pytest.raises(CompilationError, match="registered immutable proof handoff"):
                plan.validate()
        finally:
            registry[proof.identity] = proof
        assert plan.validate() is plan
