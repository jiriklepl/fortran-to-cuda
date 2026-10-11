"""One bounded generation pass over a freshly authenticated reached plan.

This is a consumer view, not proof caching. Original syntax remains read-only
during speculative generation, and fresh source validation gates both ends.
GenerationTransaction retains responsibility for rollback and adoption.
"""

from __future__ import annotations

from compiler.frontend.reached_windows import _hash, _ReachedPlanView
from compiler.ir import CompilationError


class _ReachedPlanConsumption:
    def __init__(self, builder):
        self.builder = builder
        self.view = None
        self._builders = {}
        self._child_inputs = {}
        self._parents = []
        self._active = self._closed = False

    def __enter__(self):
        if self._active or self._closed or getattr(self.builder, "_reached_consumption", None) is not None:
            raise CompilationError("reached plan consumption requires a new nonoverlapping pass")
        if getattr(self.builder, "_reached_consumption_failed", False):
            raise CompilationError("a failed reached consumption candidate cannot be reused")
        if getattr(self.builder, "_mapped_reached_child", None) is not None:
            raise CompilationError("reached plan consumption must start at its original root owner")
        parent = getattr(self.builder, "_reached_window_parent", None)
        seen = {id(self.builder)}
        while parent is not None and id(parent) not in seen:
            if (getattr(parent, "_reached_consumption", None) is not None
                    or getattr(parent, "_reached_consumption_failed", False)):
                raise CompilationError("reached parent publication requires a new successful consumption pass")
            if len(self._parents) >= self.builder.analysis.procedure_limit:
                raise CompilationError("reached plan parent projection budget exhausted")
            seen.add(id(parent))
            self._parents.append(parent)
            parent = getattr(parent, "_reached_window_parent", None)
        try:
            proof = self.builder._validate_reached_window_fresh()
            if proof is None:
                raise CompilationError("reached plan consumption requires a registered bound window")
            view = _ReachedPlanView.create(proof)
        except BaseException:
            self._fail()
            self._closed = True
            raise
        self.view, self._active = view, True
        self._builders[id(self.builder)] = self.builder
        for builder in (self.builder, *self._parents):
            builder._reached_consumption = self
        return self

    def _check_view(self, view):
        if (not self._active or self._closed or view is not self.view
                or getattr(self.builder, "_reached_consumption", None) is not self):
            raise CompilationError("reached plan consumption view is copied, foreign or expired")

    def check(self, builder):
        self._check_view(self.view)
        if self._builders.get(id(builder)) is not builder or getattr(builder, "_reached_consumption", None) is not self:
            raise CompilationError("reached plan consumer is not an enrolled original owner")
        proof = self.view.proof
        binding = getattr(builder, "_reached_window_binding", None)
        if (binding is None or binding[0] is not proof or builder.analysis is not proof._analysis
                or len(binding[1]) != len(proof._selection)
                or any(left is not right for left, right in zip(binding[1], proof._selection, strict=True))
                or getattr(builder.analysis, "_reached_window_proofs", {}).get(proof.identity) is not proof):
            raise CompilationError("reached plan consumer lost its exact registered root authority")
        child = getattr(builder, "_mapped_reached_child", None)
        if builder is self.builder:
            if child is not None or builder.entry.qualified != proof.procedure:
                raise CompilationError("reached plan root consumer changed procedure authority")
        elif (child is None or self.view.children.get(getattr(builder, "_mapped_reached_path", ())) is not child
              or child.procedure != builder.entry.qualified):
            raise CompilationError("mapped child consumer lost its exact registered call path")
        if child is not None:
            inputs = getattr(builder, "immutable_inputs", {})
            expected = self._child_inputs[id(builder)]
            if len(inputs) != len(expected) or any(inputs.get(root) is not token for root, token in expected):
                raise CompilationError("mapped child immutable input associations changed during consumption")
        if _hash(builder.facts) != proof._record["entry_facts_identity"]:
            raise CompilationError("reached owner admitted entry facts changed")
        provider = proof._numerical_provider
        if provider is not None:
            # Compare mutable configuration contents on every use, without
            # rescanning source namespaces already authenticated by this pass.
            from compiler.frontend.numerical_preparation import _Budget, _configuration

            immutable = (provider.immutable_inputs if child is not None else
                         getattr(builder, "immutable_inputs", {}))
            if (getattr(builder.analysis, "_numerical_preparation_providers", {}).get(id(provider)) is not provider
                    or _configuration(builder.options, builder.config, builder._numerical_environment_root,
                                      immutable, _Budget()) != provider.identity):
                raise CompilationError("prepared original numerical region compiler policy changed")
        return proof

    def bind_child(self, builder):
        self.check(self.builder)
        child = getattr(builder, "_mapped_reached_child", None)
        if (builder is self.builder or child is None or builder.analysis is not self.builder.analysis
                or getattr(builder, "_reached_consumption", None) is not self
                or self.view.children.get(getattr(builder, "_mapped_reached_path", ())) is not child):
            raise CompilationError("reached plan child enrollment requires its exact original call path")
        if len(self._builders) >= self.builder.analysis.procedure_limit:
            raise CompilationError("reached plan consumer procedure budget exhausted")
        inputs = getattr(builder, "immutable_inputs", {})
        if len(inputs) > self.builder.analysis.operation_limit:
            raise CompilationError("reached plan immutable input association budget exhausted")
        self._child_inputs[id(builder)] = tuple(inputs.items())
        self._builders[id(builder)] = builder
        self.check(builder)

    def numerical_region(self, builder, procedure, selection):
        self.check(builder)
        region = self.view.numerical_region(self, procedure, selection)
        if region is not None:
            from compiler.frontend.numerical_preparation import _Budget, _configuration

            if _configuration(builder.options, builder.config, builder._numerical_environment_root,
                              getattr(builder, "immutable_inputs", {}), _Budget()) != region.prepared._provider.identity:
                raise CompilationError("prepared original numerical region compiler policy changed")
        return region

    def retain_children(self, builders):
        """Drop only child projections discarded by existing scope rollback."""
        self.check(self.builder)
        retained = {id(builder) for builder in builders} | {id(self.builder)}
        for identity, builder in tuple(self._builders.items()):
            if identity not in retained:
                if getattr(builder, "_reached_consumption", None) is self:
                    del builder._reached_consumption
                del self._builders[identity]
                self._child_inputs.pop(identity, None)

    def refresh(self, proof, children):
        """Expire the old view before atomically replacing staged child tokens."""
        self.check(self.builder)
        self.view = None
        try:
            proof.validate(self.builder.analysis, self.builder.entry.qualified, proof._selection)
            view = _ReachedPlanView.create(proof)
            replacements = []
            for builder in children:
                if self._builders.get(id(builder)) is not builder:
                    raise CompilationError("reached plan refresh contains an unenrolled child")
                old = builder._mapped_reached_child
                child = view.children.get(builder._mapped_reached_path)
                if child is None or child.original_call is not old.original_call or child.public() != old.public():
                    raise CompilationError("generation close changes an already staged mapped child; owner remains native")
                replacements.append((builder, child))
            if {id(builder) for builder, _child in replacements} != set(self._builders) - {id(self.builder)}:
                raise CompilationError("reached plan refresh omitted an enrolled mapped child")
            self.builder._reached_window_binding = proof, proof._selection
            for builder, child in replacements:
                builder._reached_window_binding = proof, proof._selection
                builder._mapped_reached_child = child
            self.view = view
            for builder in self._builders.values():
                self.check(builder)
        except BaseException:
            # An unsuccessful refresh cannot expose the discarded view again.
            self._fail()
            self._expire()
            raise

    def _expire(self):
        self._active, self._closed = False, True
        for builder in (*self._builders.values(), *self._parents):
            if getattr(builder, "_reached_consumption", None) is self:
                del builder._reached_consumption

    def _fail(self):
        for builder in (self.builder, *self._builders.values(), *self._parents):
            builder._reached_consumption_failed = True

    def __exit__(self, kind, value, traceback):
        try:
            if kind is not None:
                self._fail()
            else:
                for builder in self._builders.values():
                    self.check(builder)
                self.builder._validate_reached_window_fresh()
                # Check retained typed tokens too; a fresh derivation cannot
                # bless a mutated old child token or physical-effects object.
                _ReachedPlanView.create(self.view.proof)
        except BaseException:
            self._fail()
            raise
        finally:
            self._expire()
        return False
