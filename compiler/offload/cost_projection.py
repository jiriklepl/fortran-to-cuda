"""Detached cost evidence for one synchronous source-attachment operation.

Raw reconstruction is shared only within this operation. The caller's profile
and current implementation identities are checked again before publication.
There is no process-global cache or reuse across compiler/application calls.
"""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path

from compiler.offload.numerical_calibration import NumericalCalibrationError

_PROFILE_LIMIT = 32 * 2**20


def _json(value):
    """Copy JSON without silently normalizing invalid tuple/key/value types."""
    active = set()

    def check(item, depth=0):
        if depth > 256:
            raise NumericalCalibrationError("cost projection JSON depth exceeded")
        kind = type(item)
        if kind in (str, int, bool) or item is None:
            return
        if kind is float:
            if not math.isfinite(item):
                raise NumericalCalibrationError("cost projection requires finite JSON values")
            return
        if kind not in (dict, list):
            raise NumericalCalibrationError("cost projection requires plain JSON data")
        key = id(item)
        if key in active:
            raise NumericalCalibrationError("cost projection JSON is cyclic")
        active.add(key)
        if kind is dict:
            if any(type(name) is not str for name in item):
                raise NumericalCalibrationError("cost projection JSON keys must be strings")
            children = item.values()
        else:
            children = item
        for child in children:
            check(child, depth + 1)
        active.remove(key)

    try:
        check(value)
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(text.encode()) > _PROFILE_LIMIT:
            raise NumericalCalibrationError("cost projection JSON exceeds 32 MiB")
        return text
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        if isinstance(error, NumericalCalibrationError):
            raise
        raise NumericalCalibrationError("cost projection JSON unavailable") from error


def _reader_identity():
    # Some reader dependencies are deliberately outside the numerical worker
    # identity. They must still remain fixed during an attachment transaction.
    folder = Path(__file__).parent
    names = ("cost_projection.py", "dependency_source.py", "source_compute.py",
             "numerical_calibration.py", "profile.py", "cpu_protocol_calibration.py")
    try:
        readers = {name: sha256((folder / name).read_bytes()).hexdigest() for name in names}
        return readers
    except OSError as error:
        raise NumericalCalibrationError("cost projection source identity unavailable") from error


def _cpu_identity():
    from compiler.offload.cpu_dependency_calibration import dependency_generator_identity

    return dependency_generator_identity(), _reader_identity()


def _gpu_identity(precision):
    from compiler.offload.numerical_gpu_execution import current_gpu_identities
    from compiler.offload.numerical_gpu_execution_costs import cost_sources

    return current_gpu_identities(precision), cost_sources(), _reader_identity()


def _identity_stamp(value):
    # Internal providers use tuples as well as dictionaries. Serialize their
    # values so a provider cannot mutate the saved witness through an alias.
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise NumericalCalibrationError("cost projection identity unavailable") from error


def _without_holdouts(value):
    if isinstance(value, dict):
        return {key: _without_holdouts(child) for key, child in value.items() if key != "holdouts"}
    if isinstance(value, list):
        return [_without_holdouts(child) for child in value]
    return value


class _SourceCostProjection:
    """Lazy, single-use snapshot; only owned cost dictionaries leave it."""

    def __init__(self, profile):
        self._original = profile
        self._snapshot = None
        self._precision = None
        self._cpu = self._gpu = None
        self._cpu_stamp = self._gpu_stamp = None
        self._cpu_error = self._gpu_error = None
        self._closed = False

    def _check_open(self):
        if self._closed:
            raise NumericalCalibrationError("cost projection has expired")

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_exception):
        self.finish()

    def _profile(self):
        self._check_open()
        if self._snapshot is None:
            self._snapshot = _json(self._original)
        profile = json.loads(self._snapshot)
        if type(profile) is not dict:
            raise NumericalCalibrationError("cost projection requires a JSON object profile")
        return profile

    def cpu_costs(self, graph, backend, **arguments):
        from compiler.offload.cpu_dependency_calibration import _cpu_costs_from_section, validate_dependency_profile
        from compiler.offload.numerical_calibration import require_numerical_profile_contract

        self._check_open()
        if self._cpu_error is not None:
            raise NumericalCalibrationError(self._cpu_error)
        if self._cpu is None:
            profile = self._profile()
            self._cpu_stamp = _identity_stamp(_cpu_identity())
            try:
                require_numerical_profile_contract(profile)
                section = validate_dependency_profile(profile)
                self._precision = profile["precision_bits"]
                self._cpu = _json(_without_holdouts({key: section[key]
                    for key in ("protocol_id", "generator_id", "backends")}))
            except NumericalCalibrationError as error:
                self._cpu_error = str(error)
                raise
        return _cpu_costs_from_section({"precision_bits": self._precision}, json.loads(self._cpu),
                                       graph, backend, **arguments)

    def gpu_costs(self, counts, **arguments):
        from compiler.offload.numerical_gpu_execution_costs import _current_gpu_cost_sections, _gpu_costs_from_sections

        self._check_open()
        if self._gpu_error is not None:
            raise NumericalCalibrationError(self._gpu_error)
        if self._gpu is None:
            profile = self._profile()
            self._precision = profile.get("precision_bits")
            if type(self._precision) is not int or self._precision not in (32, 64):
                raise NumericalCalibrationError("cost projection precision unavailable")
            self._gpu_stamp = _identity_stamp(_gpu_identity(self._precision))
            try:
                parent, section = _current_gpu_cost_sections(profile)
                self._precision = profile["precision_bits"]
                self._gpu = (_json({key: parent[key] for key in ("registry", "identity")}),
                    _json(_without_holdouts({key: section[key] for key in
                        ("identity", "startup", "families", "workload_validation", "domain_validation")})))
            except NumericalCalibrationError as error:
                self._gpu_error = str(error)
                raise
        return _gpu_costs_from_sections(*(json.loads(part) for part in self._gpu), counts, **arguments)

    def finish(self):
        """Expire even on failure; callers must discard costs if the fence fails."""
        self._check_open()
        self._closed = True
        try:
            if self._snapshot is None:
                return
            if _json(self._original) != self._snapshot:
                raise NumericalCalibrationError("calibration profile changed during source cost attachment")
            if self._cpu_stamp is not None and _identity_stamp(_cpu_identity()) != self._cpu_stamp:
                raise NumericalCalibrationError("CPU source/backend identity changed during source cost attachment")
            if self._gpu_stamp is not None and _identity_stamp(_gpu_identity(self._precision)) != self._gpu_stamp:
                raise NumericalCalibrationError("GPU source/backend identity changed during source cost attachment")
        finally:
            self._original = None
            self._snapshot = self._cpu = self._gpu = None
