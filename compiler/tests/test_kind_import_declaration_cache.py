"""Declaration reuse preserves repeated import order and fresh values."""
from collections import Counter

import pytest

import compiler.frontend.source_effects as effects
from compiler.frontend.lowering import _KindScope
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import SourceLocation

SOURCE = """module original
integer,parameter::seed=4,wide=2*seed
real(wide)::marker
end module
module left
use original
end module
module right
use original
end module
module consumer
use left
use right
contains
subroutine work(a)
real(wide),intent(in)::a
end subroutine
end module
"""


def test_repeated_imports_resolve_each_original_declaration_once_per_pass(tmp_path, monkeypatch):
    path = tmp_path / "diamond.f90"
    path.write_text(SOURCE)
    original_import = SourceEffects._import_kinds
    original_binding = SourceEffects._binding
    original_walk = effects.walk
    stack, transactions = [], {}

    def importing(self, *args, **kwargs):
        context = kwargs.get("_lookups")
        stack.append((self, context))
        try:
            return original_import(self, *args, **kwargs)
        finally:
            stack.pop()

    def transaction():
        return next(((analysis, context) for analysis, context in reversed(stack)
                     if context is not None), None)

    def binding(self, scope, name, visited=frozenset()):
        active = transaction()
        if active is not None and not visited:
            _analysis, context = active
            row = transactions.setdefault(id(context), (context, Counter(), Counter()))
            row[1][id(scope), str(name)] += 1
        return original_binding(self, scope, name, visited)

    def walking(node, *args, **kwargs):
        active = transaction()
        if active is not None:
            analysis, context = active
            if any(expression is node for _binding, _owner, expression in analysis.kind_expressions):
                row = transactions.setdefault(id(context), (context, Counter(), Counter()))
                row[2][id(node)] += 1
        return original_walk(node, *args, **kwargs)

    monkeypatch.setattr(SourceEffects, "_import_kinds", importing)
    monkeypatch.setattr(SourceEffects, "_binding", binding)
    monkeypatch.setattr(effects, "walk", walking)
    analysis = SourceEffects([path])
    assert analysis.routines["consumer::work"].scope.bindings["a"].kind == 8
    assert analysis.modules["original"].kinds.integer("wide", SourceLocation(str(path))) == 8
    assert transactions
    assert any(row[1] for row in transactions.values())
    assert any(row[2] for row in transactions.values())
    for _context, bindings, selectors in transactions.values():
        assert all(count == 1 for count in bindings.values())
        assert all(count == 1 for count in selectors.values())


def test_parameter_values_are_evaluated_again_for_each_alias_in_same_pass(tmp_path, monkeypatch):
    path = tmp_path / "values.f90"
    path.write_text("module original\ninteger,parameter::wide=4\nend module\n"
                    "module consumer\nuse original,only:first=>wide,second=>wide\nend module\n")
    analysis = SourceEffects([path])
    owner = analysis.modules["original"]
    consumer = analysis.modules["consumer"]
    original_integer = _KindScope.integer
    seen = []

    def integer(self, node, location, active=frozenset()):
        value = original_integer(self, node, location, active)
        if self is owner.kinds and node == "wide" and not active:
            seen.append(value)
            self.values["wide"] = 8
        return value

    monkeypatch.setattr(_KindScope, "integer", integer)
    analysis._import_kinds(consumer)
    assert seen == [4, 8]
    assert consumer.kinds.values["first"] == 4
    assert consumer.kinds.values["second"] == 8


@pytest.mark.parametrize("missing_first", [False, True])
def test_later_pass_does_not_reuse_binding_or_selector_results(tmp_path, missing_first):
    path = tmp_path / "fresh.f90"
    path.write_text("module original\ninteger,parameter::wide=8\nend module\n"
                    "module short\ninteger,parameter::wide=4\nend module\n"
                    "module consumer\nuse original,only:wide\ncontains\n"
                    "subroutine work(a)\nreal(wide),intent(in)::a\nend subroutine\nend module\n")
    analysis = SourceEffects([path])
    scope = analysis.routines["consumer::work"].scope
    assert scope.bindings["a"].kind == 8
    scope.imports["wide"] = "absent" if missing_first else "short", "wide"
    analysis._import_kinds(scope)
    assert scope.kinds.values["wide"] == (None if missing_first else 4)
    scope.imports["wide"] = "original", "wide"
    analysis._import_kinds(scope)
    assert scope.kinds.values["wide"] == 8


def test_recursive_owner_imports_still_run_for_repeated_parameter_aliases(tmp_path, monkeypatch):
    path = tmp_path / "recursive.f90"
    path.write_text("module original\ninteger,parameter::wide=8\nend module\n"
                    "module consumer\nuse original,only:first=>wide,second=>wide\nend module\n")
    analysis = SourceEffects([path])
    original_import = SourceEffects._import_kinds
    owner = analysis.modules["original"]
    calls = []

    def importing(self, scope, active=frozenset(), **kwargs):
        if scope is owner:
            calls.append(active)
        return original_import(self, scope, active, **kwargs)

    monkeypatch.setattr(SourceEffects, "_import_kinds", importing)
    analysis._import_kinds(analysis.modules["consumer"])
    assert len(calls) == 2
    assert all(id(analysis.modules["consumer"]) in active for active in calls)
    assert analysis.modules["consumer"].kinds.values["first"] == 8
    assert analysis.modules["consumer"].kinds.values["second"] == 8
