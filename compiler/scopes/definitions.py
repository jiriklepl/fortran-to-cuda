"""Ordered whole-definition requirements for bounded reached source trees.

This proof complements exact runtime coverage. It does not turn a may-write
envelope into a definition, flatten descendants, or assert host freshness.
"""

from dataclasses import dataclass

from compiler.ir import CompilationError


@dataclass(frozen=True)
class DefinitionSummary:
    required_whole: frozenset
    whole_on_return: frozenset
    changes: frozenset

    def public(self):
        return {"version": 1, "required_whole": sorted(self.required_whole),
                "whole_on_return": sorted(self.whole_on_return),
                "possible_definition_changes": sorted(self.changes),
                "authority": "ordered reached operations and reusable child interfaces"}


def summarize(scope, *, entry_undefined=()):
    from compiler.scopes.segments import Association, Branch, Delegated, Segment

    builder = scope.builder
    required, changes = set(), set(entry_undefined)

    def require(roots, whole, undefined):
        for root in roots:
            if root in whole:
                continue
            if root in undefined:
                raise CompilationError("reached whole-array requirement follows incomplete definition: " + root)
            required.add(root)
            whole.add(root)

    def transfer(whole, undefined, kills, writes):
        changes.update(kills)
        return (whole - kills) | writes, (undefined | kills) - writes

    def native(operation, whole, undefined):
        if not operation.sections.available:
            # A whole overwrite removes preservation reads, never explicit RHS
            # reads. The original source must define those before this point.
            reads = {root for root, actions in operation.effects.items()
                     if "read" in actions or root not in operation.overwrites}
            require(reads, whole, undefined)
        return transfer(whole, undefined, set(), operation.overwrites)

    def call_effect(call, whole, undefined):
        effects, kills, writes = builder.roots_for(call)
        fallback = scope.guarded.get(id(call.node))
        if fallback is not None:
            native(fallback, whole, undefined)
        if call.region is not None:
            # Coverage comes from the original assignment syntax, not lowered
            # coordinates or an enclosing box of numerical accesses.
            writes = builder.inline_for(call.procedure).whole_definitions(call)
        elif not builder.closure(call.procedure)[0]:
            builder.check_native_definitions(call, *builder.call_effects(call))
            physical = builder.analysis.native_sections(call.procedure)
            whole, undefined = transfer(whole, undefined, kills, set())
            if not physical.available:
                require({root for root, actions in effects.items()
                         if "read" in actions or root not in writes}, whole, undefined)
        return transfer(whole, undefined, kills, writes)

    def visit(items, whole, undefined):
        for item in items:
            if isinstance(item, Segment):
                require(item.payload, whole, undefined)
                for call in item.calls:
                    whole, undefined = call_effect(call, whole, undefined)
            elif isinstance(item, Delegated):
                child, call = item.coordinator, item.call
                child.check_aliases(call)
                mapping = {formal: binding.root for formal, binding in call.bindings.items()}
                mapped = lambda roots: {mapping.get(root, root) for root in roots}
                proof = child.definition_summary
                require(mapped(proof.required_whole), whole, undefined)
                whole, undefined = transfer(whole, undefined, mapped(proof.changes), mapped(proof.whole_on_return))
            elif isinstance(item, Branch):
                paths = []
                for condition, body in item.alternatives:
                    if condition is not None:
                        whole, undefined = native(condition, whole, undefined)
                    paths.append(visit(body, set(whole), set(undefined)))
                if not item.alternatives or item.alternatives[-1][0] is not None:
                    paths.append((whole, undefined))
                whole = set.intersection(*(path[0] for path in paths))
                undefined = set.union(*(path[1] for path in paths))
            elif isinstance(item, Association):
                whole, undefined = visit(item.body, whole, undefined)
            else:
                whole, undefined = native(item, whole, undefined)
        return whole, undefined

    # First infer entry obligations; then propagate those common assumptions
    # through every branch to compute a sound, reusable return guarantee.
    visit(scope.tree, set(), set(entry_undefined))
    whole, _ = visit(scope.tree, set(required), set(entry_undefined))
    return DefinitionSummary(frozenset(required), frozenset(whole), frozenset(changes))
