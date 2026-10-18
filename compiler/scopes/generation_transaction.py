"""Serialized speculative generation without copying original source authority.

Generation containers and coordinator graphs belong to the candidate. Original
parser nodes, bindings, routines and registered proofs keep their identities.
The exact SourceEffects object is shared: its cache/registry containers are
temporarily detached while the transaction runs, then restored on rejection.
Consequently transactions using one analysis must not overlap. Parsed source
objects remain read-only, just as they do during ordinary source generation.

Persistent summary-cache writes are disabled for speculative work. Accepted
memory entries and counters are committed to the original cache only on adopt.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import copy
from dataclasses import fields, is_dataclass
from enum import Enum
from types import CellType, FunctionType, MappingProxyType, MethodType, SimpleNamespace
from weakref import WeakSet

from compiler.frontend.source_effects import SourceEffects, _Closure
from compiler.frontend.summary_cache import SummaryCache
from compiler.ir import CompilationError

_ACTIVE_ANALYSES = WeakSet()
_OWNED_MODULES = ("compiler.scopes.", "compiler.emission.", "compiler.ir.", "compiler.offload.")


class _GenerationCopy:
    """Copy mutable generated graphs, preserving source and proof identities."""

    def __init__(self, analysis, memo=None):
        self.analysis = analysis
        self.memo = dict(memo or {})

    def function(self, value):
        previous = self.memo.get(id(value))
        if previous is not None:
            return previous
        # Bound capture helpers close over their coordinator/lexical owner.
        # Rebinding only MethodType.__self__ would retain the old owner there.
        cells = tuple(CellType() for _ in value.__closure__ or ())
        result = FunctionType(value.__code__, value.__globals__, value.__name__,
                              value.__defaults__, cells or None)
        self.memo[id(value)] = result
        for source, target in zip(value.__closure__ or (), cells, strict=True):
            try:
                contents = source.cell_contents
            except ValueError:  # Preserve an empty closure cell.
                continue
            target.cell_contents = self(contents)
        result.__defaults__ = self(value.__defaults__)
        result.__kwdefaults__ = self(value.__kwdefaults__)
        result.__annotations__ = self(value.__annotations__)
        result.__dict__.update(self(value.__dict__))
        result.__qualname__, result.__module__ = value.__qualname__, value.__module__
        return result

    def __call__(self, value):
        identity = id(value)
        if identity in self.memo:
            return self.memo[identity]
        if isinstance(value, SourceEffects):
            self.analysis(value)
            return value
        from compiler.frontend.immutable_graph import FrozenDict, FrozenList, sealed
        if isinstance(value, FrozenDict):
            result = FrozenDict()
            self.memo[identity] = result
            dict.update(result, ((self(key), self(item)) for key, item in value.items()))
            return result
        if isinstance(value, FrozenList):
            result = FrozenList()
            self.memo[identity] = result
            list.extend(result, (self(item) for item in value))
            return result
        if isinstance(value, Enum):
            return value
        kind = type(value)
        if isinstance(value, dict):
            result = copy(value)
            result.clear()
            self.memo[identity] = result
            result.update((self(key), self(item)) for key, item in value.items())
            return result
        if kind is list:
            result = []
            self.memo[identity] = result
            result.extend(self(item) for item in value)
            return result
        if kind is set:
            result = set()
            self.memo[identity] = result
            result.update(self(item) for item in value)
            return result
        if kind in {tuple, frozenset}:
            result = kind(self(item) for item in value)
            self.memo[identity] = result
            return result
        if kind is MappingProxyType:
            result = MappingProxyType(self(dict(value)))
            self.memo[identity] = result
            return result
        if kind is MethodType:
            result = MethodType(self.function(value.__func__), self(value.__self__))
            self.memo[identity] = result
            return result
        if kind in {SimpleNamespace, _Closure} or kind.__module__.startswith(_OWNED_MODULES):
            if not hasattr(value, "__dict__") and not is_dataclass(value):
                return value
            result = SimpleNamespace() if kind is SimpleNamespace else object.__new__(kind)
            self.memo[identity] = result
            if hasattr(value, "__dict__"):
                state = {name: self(item) for name, item in vars(value).items()}
                if sealed(value):
                    # Retain immutability while rebinding builder/owner edges.
                    # Ordinary copy() intentionally returns mutable projections;
                    # transactional adoption must preserve the checked handoff.
                    object.__setattr__(result, "__dict__", state)
                else:
                    result.__dict__.update(state)
            else:
                for field in fields(value):
                    object.__setattr__(result, field.name, self(getattr(value, field.name)))
            return result
        # In particular, no frontend object, fparser syntax node or source
        # Binding is copied. A registered proof cannot acquire a new identity.
        return value


class GenerationTransaction:
    """Fork a ScopeBuilder, committing only an explicitly adopted candidate.

    Use ``with GenerationTransaction(parent) as transaction`` and generate
    through ``transaction.builder``. ``adopt()`` schedules a commit on normal
    context exit; an exception still rejects it. A window emitter may return a
    builder projection; pass that builder to ``adopt(result_builder)``. Source
    analysis and original entry must be identical. No execution or capture
    authority is created by the transaction itself.
    """

    def __init__(self, parent):
        self.parent = parent
        self.builder = None
        self._analyses = {}
        self._winner = None
        self._active = False
        self._closed = False
        self._marker = object()

    def _analysis(self, analysis):
        if analysis in self._analyses:
            return
        if analysis in _ACTIVE_ANALYSES:
            raise CompilationError("generation transactions cannot overlap on one source analysis")
        _ACTIVE_ANALYSES.add(analysis)
        original = dict(vars(analysis))
        self._analyses[analysis] = original
        session = getattr(analysis, "_source_session", None)
        clone = _GenerationCopy(self._analysis, memo=(
            {id(value): value for value in session.roots.values()} if session is not None else None))
        # Register first: cached proof objects can refer to this same analysis.
        detached = {name: clone(value) for name, value in original.items() if name != "_summary_cache"}
        cache = original["_summary_cache"]
        private_cache = SummaryCache(max_entries=cache.max_entries)
        with cache._lock:
            private_cache._memory = OrderedDict(cache._memory)
            private_cache._stats = dict(cache._stats)
        detached["_summary_cache"] = private_cache
        vars(analysis).clear()
        vars(analysis).update(detached)

    def __enter__(self):
        if self._active or self._closed:
            raise CompilationError("generation transaction is single-use")
        self.parent._check_reached_publication()
        self._active = True
        self._parent_state = dict(vars(self.parent))
        try:
            self.builder = _GenerationCopy(self._analysis)(self.parent)
            self.builder._generation_transaction_marker = self._marker
        except BaseException:
            self._restore()
            self._active, self._closed = False, True
            raise
        return self

    def adopt(self, builder=None):
        if not self._active or self._closed:
            raise CompilationError("generation adoption requires an active transaction")
        builder = self.builder if builder is None else builder
        if (builder is self.parent or getattr(builder, "analysis", None) is not self.parent.analysis
                or getattr(builder, "entry", None) is not self.parent.entry
                or getattr(builder, "_generation_transaction_marker", None) is not self._marker):
            raise CompilationError("generation adoption requires the same original analysis and entry")
        builder._check_reached_publication()
        self._winner = builder

    def _restore(self):
        for analysis, original in reversed(tuple(self._analyses.items())):
            vars(analysis).clear()
            vars(analysis).update(original)
            _ACTIVE_ANALYSES.discard(analysis)

    def _commit(self):
        winner = self._winner
        winner._check_reached_publication()
        if winner.facts != self._parent_state["facts"]:
            raise CompilationError("generation candidate changed independently admitted source facts")
        winner._validate_reached_window_fresh()
        winner.validate_selection_activation()
        winner._validate_numerical_regions()
        winner.analysis.verify_source()
        # Rebind every nested builder, inline registry and bound closure to the
        # adopted root while retaining their internal alias relationships.
        copier = _GenerationCopy(self._analysis, {id(self.builder): self.parent, id(winner): self.parent})
        state = {name: copier(value) for name, value in vars(winner).items()
                 if name != "_generation_transaction_marker"}
        for analysis, original in self._analyses.items():
            cache, private = original["_summary_cache"], analysis._summary_cache
            with cache._lock:
                cache._memory = OrderedDict(private._memory)
                cache._stats = dict(private._stats)
            analysis._summary_cache = cache
        vars(self.parent).clear()
        vars(self.parent).update(state)
        for analysis in self._analyses:
            _ACTIVE_ANALYSES.discard(analysis)

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None and self._winner is not None:
                self._commit()
            else:
                self._restore()
        except BaseException:
            self._restore()
            raise
        finally:
            self._active, self._closed = False, True
        return False
