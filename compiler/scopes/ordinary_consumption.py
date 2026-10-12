"""Private pair reuse during one read-only original lexical generation pass.

Source-issued pairs are validated on first use (or enrolled after their fresh
issuance), then revalidated before the pass returns. Public proof accessors and
publication remain fresh demands. This is not source-analysis caching.
"""

from types import SimpleNamespace

from compiler.frontend.numerical_preparation import _Budget, _configuration
from compiler.frontend.reached_windows import _hash, _ResolvedNumericalRegion
from compiler.ir import CompilationError


class _OrdinaryPairConsumption:
    def __init__(self, builder):
        self.builder = builder
        self.state = builder._ordinary_pair_state
        self.records = {}
        self._active = self._closed = False

    def __enter__(self):
        self.builder._check_reached_publication()
        if self._active or self._closed or getattr(self.builder, "_reached_window_binding", None) is not None:
            raise CompilationError("ordinary numerical consumption requires a new original lexical pass")
        self.facts = _hash(self.builder.facts)
        self.policy = self._policy(self.builder)
        self.checkpoint = self.builder.scope_checkpoint()
        self.scope_count, self.boundary_count = len(self.builder.scopes), len(self.builder.boundaries)
        self.state.active, self._active = self, True
        return self

    @staticmethod
    def _policy(builder):
        return _configuration(builder.options, builder.config, builder._numerical_environment_root,
                              getattr(builder, "immutable_inputs", {}), _Budget())

    def _fail(self):
        if self.state.active is self:
            self.state.failed = True

    def _check(self, registry):
        builder = registry.builder
        if (not self._active or self._closed or self.state.active is not self or self.state.failed
                or builder._ordinary_pair_state is not self.state
                or builder.analysis is not self.builder.analysis
                or self.builder.region_owners.get(builder.entry.qualified) is not registry
                or builder.region_owners.get(builder.entry.qualified) is not registry
                or builder.analysis.routines.get(builder.entry.qualified) is not builder.entry
                or getattr(builder, "_reached_window_binding", None) is not None):
            raise CompilationError("ordinary numerical consumer is copied, foreign, failed or expired")
        if _hash(builder.facts) != self.facts or _hash(self.builder.facts) != self.facts:
            raise CompilationError("ordinary numerical consumer admitted facts changed")
        if self._policy(self.builder) != self.policy:
            raise CompilationError("ordinary numerical consumer compiler policy changed")

    def _pair(self, registry, region, selection):
        self._check(registry)
        from compiler.frontend.numerical_effects import NumericalEffectProof
        from compiler.frontend.numerical_preparation import NumericalPreparationProvider, PreparedOriginalRegion

        builder, analysis = registry.builder, registry.builder.analysis
        if (type(region) is not _ResolvedNumericalRegion or region.procedure != builder.entry.qualified
                or len(selection) != len(region.selection)
                or any(left is not right for left, right in zip(selection, region.selection, strict=True))
                or registry.original_numerical_regions.get(tuple(map(id, selection))) is not region
                or type(region.prepared) is not PreparedOriginalRegion
                or type(region.effects) is not NumericalEffectProof
                or region.effects.prepared is not region.prepared
                or region.effects._analysis is not analysis
                or region.prepared.procedure != builder.entry.qualified
                or region.effects.procedure != builder.entry.qualified
                or getattr(analysis, "_numerical_preparations", {}).get(region.prepared.identity) is not region.prepared
                or getattr(analysis, "_numerical_effect_proofs", {}).get(region.effects.identity) is not region.effects):
            raise CompilationError("ordinary numerical consumer lost its exact registered pair")
        provider = region.prepared._provider
        if (type(provider) is not NumericalPreparationProvider or provider._analysis is not analysis
                or getattr(analysis, "_numerical_preparation_providers", {}).get(id(provider)) is not provider
                or self._policy(builder) != provider.identity):
            raise CompilationError("ordinary numerical consumer prepared policy changed")
        return builder

    def enroll(self, registry, region):
        """Called only after successful fresh issuance or first-use validation."""
        try:
            builder = self._pair(registry, region, region.selection)
            key = registry, tuple(map(id, region.selection))
            if key not in self.records and len(self.records) >= (
                    builder.analysis.procedure_limit * min(32, builder.analysis.operation_limit)):
                raise CompilationError("ordinary numerical consumer pair budget exhausted")
            self.records[key] = region, tuple(getattr(builder, "immutable_inputs", {}).items())
        except BaseException:
            self._fail()
            raise

    def resolve(self, registry, selection):
        try:
            self._check(registry)
            selection = tuple(selection)
            key = registry, tuple(map(id, selection))
            region = registry.original_numerical_regions.get(key[1])
            if region is None:
                if key in self.records:
                    raise CompilationError("ordinary numerical consumer registered pair was removed")
                return None
            record = self.records.get(key)
            if record is not None and record[0] is not region:
                raise CompilationError("ordinary numerical consumer registered pair was replaced")
            if record is None:
                region = registry.resolved_numerical_region(selection)
                self.enroll(registry, region)
                record = self.records[key]
            builder = self._pair(registry, region, selection)
            inputs = getattr(builder, "immutable_inputs", {})
            if len(inputs) != len(record[1]) or any(inputs.get(root) is not token for root, token in record[1]):
                raise CompilationError("ordinary numerical consumer immutable associations changed")
            return region
        except BaseException:
            self._fail()
            raise

    def retain(self):
        """Forget pairs discarded by the existing explicit scope rollback."""
        if not self._active or self._closed or self.state.active is not self:
            raise CompilationError("ordinary numerical consumer rollback is copied or expired")
        self.records = {key: record for key, record in self.records.items()
                        if key[0].original_numerical_regions.get(key[1]) is record[0]}

    def __exit__(self, kind, value, traceback):
        if not self._active or self._closed or self.state.active is not self:
            raise CompilationError("ordinary numerical consumer is copied or expired")
        try:
            if kind is None:
                self._check(self.builder.inline)
                for (registry, _selection), (region, inputs) in self.records.items():
                    builder = self._pair(registry, region, region.selection)
                    current = getattr(builder, "immutable_inputs", {})
                    if len(current) != len(inputs) or any(current.get(root) is not token for root, token in inputs):
                        raise CompilationError("ordinary numerical consumer immutable associations changed")
                # Rollback can discard speculative pairs. Validate all retained
                # owner registries, including pairs not consulted by this pass.
                self.builder._validate_numerical_regions()
        except BaseException:
            self.state.failed = True
            raise
        finally:
            try:
                if kind is not None or self.state.failed:
                    self.state.failed = True
                    self.builder.restore_scope_checkpoint(self.checkpoint)
                    del self.builder.scopes[self.scope_count:]
                    del self.builder.boundaries[self.boundary_count:]
                    for registry in self.builder.region_owners.values():
                        registry.refresh(self.builder)
            finally:
                self.state.active = None
                self._active, self._closed = False, True
                self.records.clear()
        return False


def _ordinary_pair_state():
    # Shallow original/child builder projections share this publication fence;
    # GenerationTransaction copies it into the isolated candidate graph.
    return SimpleNamespace(active=None, failed=False)
