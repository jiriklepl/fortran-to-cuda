"""Verify source-bound application comparisons independently of timing evidence.

Raw payloads and binary legacy VTK are streamed with their real SHA256 hashes.
The approved dump source is an explicit input contract, not a semantic proof
that arbitrary instrumentation preserves a program. No application names or
claimed comparison-success flags participate in this verifier.
"""

from __future__ import annotations

import json
import math
import os
import stat
import sys
from array import array
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from compiler.offload.collective_calibration import normalize_fortran_options

POLICY_PROTOCOL = "full-float64-comparison-v1"
RECEIPT_PROTOCOL = "checked-selection-numerics-v1"
VTK_FORMAT = "legacy-binary-rectilinear-vtk-v1"
OBSERVATION_PREFIX = b"FORT_SELECTION_OBSERVATION "
CHUNK_BYTES = 64 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_LINE_BYTES = 64 * 1024


def _check(condition, message):
    if not condition:
        raise ValueError("selection numerical validation: " + message)


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _relative(value):
    return (type(value) is str and bool(value) and "\0" not in value and "\\" not in value
            and not value.startswith("/") and all(part not in {"", ".", ".."} for part in value.split("/")))


def normalize_policy(requested, *, build_contract):
    """Bind an explicit comparison policy to an original executed source unit."""
    _check(type(requested) is dict, "explicit validation policy required")
    policy = deepcopy(requested)
    supplied_identity = policy.pop("identity", None)
    _check(set(policy) == {"schema_version", "protocol", "dtype", "fields", "metadata_format", "atol", "rtol",
                          "finite_only", "exact_layouts", "outputs", "driver"}, "policy fields differ")
    _check(type(policy["schema_version"]) is int and policy["schema_version"] == 1
           and policy["protocol"] == POLICY_PROTOCOL and policy["dtype"] in {"<f8", ">f8"}
           and policy["metadata_format"] == "storage-shape-bounds-v1"
           and policy["finite_only"] is True and policy["exact_layouts"] is True, "unsupported comparison policy")
    for name in ("atol", "rtol"):
        _check(type(policy[name]) in {int, float} and math.isfinite(policy[name]) and policy[name] >= 0,
               "tolerances must be finite and nonnegative")
    fields = policy["fields"]
    _check(type(fields) is dict and 0 < len(fields) <= 128, "explicit full-field inventory required")
    paths = []
    for name, field in fields.items():
        _check(type(name) is str and 0 < len(name) <= 256 and "\0" not in name
               and type(field) is dict and set(field) == {"payload", "metadata"}
               and all(_relative(path) for path in field.values()), "invalid full-field inventory")
        paths.extend(field.values())
    outputs = policy["outputs"]
    _check(type(outputs) is dict and set(outputs) == {"format", "files"} and outputs["format"] == VTK_FORMAT
           and type(outputs["files"]) is dict and 0 < len(outputs["files"]) <= 32,
           "explicit output format and inventory required")
    for path, names in outputs["files"].items():
        _check(_relative(path) and type(names) is list and bool(names) and len(names) <= 128
               and all(type(name) is str and name and len(name) <= 256 and not any(c.isspace() for c in name)
                       and "\0" not in name and name not in {"X_COORDINATES", "Y_COORDINATES", "Z_COORDINATES"}
                       for name in names) and len(set(names)) == len(names), "invalid output field inventory")
        paths.append(path)
    _check(len(set(paths)) == len(paths), "comparison paths must be distinct")
    driver = policy["driver"]
    _check(type(driver) is dict and set(driver) == {"original_source", "original_sha256", "replacement_sha256"},
           "one approved diagnostic source replacement required")
    original = driver["original_source"]
    _check(type(original) is str and Path(original).is_absolute()
           and original in build_contract["original_units"]
           and build_contract["original_inputs"].get(original) == driver["original_sha256"]
           and _sha(driver["original_sha256"]) and _sha(driver["replacement_sha256"]),
           "diagnostic driver is not bound to an original executed source")
    identity = _digest(policy)
    _check(supplied_identity is None or supplied_identity == identity, "validation policy identity differs")
    return {**policy, "identity": identity}


class _Reader:
    """Hash exactly the bytes being interpreted, with bounded reads."""

    def __init__(self, reference):
        _check(type(reference) is dict and set(reference) == {"path", "sha256"}
               and type(reference["path"]) is str and Path(reference["path"]).is_absolute()
               and _sha(reference["sha256"]), "invalid file reference")
        self.reference = reference
        descriptor = os.open(reference["path"], os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        try:
            self.before = os.fstat(descriptor)
            _check(stat.S_ISREG(self.before.st_mode), "comparison input must be a regular file")
            self.stream = os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise
        self.hasher = sha256()
        self.consumed = 0

    def __enter__(self):
        return self

    def read(self, count):
        _check(0 <= count <= MAX_JSON_BYTES, "unbounded input read")
        value = self.stream.read(count)
        self.hasher.update(value)
        self.consumed += len(value)
        return value

    def line(self):
        value = self.stream.readline(MAX_LINE_BYTES + 1)
        _check(len(value) <= MAX_LINE_BYTES, "input line exceeds supported bound")
        self.hasher.update(value)
        self.consumed += len(value)
        return value

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None:
                _check(self.read(1) == b"", "unexpected trailing input")
                after = os.fstat(self.stream.fileno())
                _check((self.before.st_dev, self.before.st_ino, self.before.st_size, self.before.st_mtime_ns,
                        self.before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                                                  after.st_ctime_ns), "comparison input changed while reading")
                _check(self.hasher.hexdigest() == self.reference["sha256"], "comparison input hash differs")
        finally:
            self.stream.close()


def _json(reference):
    with _Reader(reference) as reader:
        _check(reader.before.st_size <= MAX_JSON_BYTES, "receipt exceeds supported bound")
        value = json.loads(reader.read(MAX_JSON_BYTES))
    _check(type(value) is dict, "receipt must be an object")
    return value


def _hash_file(path):
    hasher = sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


def _within(reference, directory, relative=None):
    _check(type(reference) is dict and type(reference.get("path")) is str, "missing comparison file reference")
    path, root = Path(reference["path"]).resolve(), Path(directory).resolve()
    _check(path.is_relative_to(root) and (relative is None or path == (root / relative).resolve()),
           "comparison file is outside the recorded run or differs from policy")


def _layout(reference):
    with _Reader(reference) as reader:
        _check(reader.before.st_size <= 4096, "shape metadata exceeds supported bound")
        text = reader.read(4096).decode("ascii")
        try:
            values = [int(word) for word in text.split()]
        except ValueError as error:
            raise ValueError("selection numerical validation: invalid shape metadata") from error
    rank, remainder = divmod(len(values) - 1, 3)
    _check(remainder == 0 and 1 <= rank <= 15 and values[0] == 64
           and all(-(2**63) <= value < 2**63 for value in values), "shape metadata rank or precision differs")
    shape, lower, upper = (values[1 + offset * rank:1 + (offset + 1) * rank] for offset in range(3))
    _check(all(extent >= 0 and extent == max(0, hi - lo + 1)
               for extent, lo, hi in zip(shape, lower, upper, strict=True)), "shape metadata extents differ")
    return {"bits": 64, "shape": shape, "lower_bounds": lower, "upper_bounds": upper}


def _values(reader, count, code, endian):
    size = 4 if code == "f" else 8
    data = reader.read(count * size)
    _check(len(data) == count * size, "truncated numeric payload")
    result = array(code)
    _check(result.itemsize == size, "unsupported native array precision")
    result.frombytes(data)
    if endian != sys.byteorder:
        result.byteswap()
    return result


def _compare_values(native, resident, count, *, code, endian, policy):
    maximum = 0.0
    remaining = count
    while remaining:
        take = min(remaining, CHUNK_BYTES // (4 if code == "f" else 8))
        expected = _values(native, take, code, endian)
        actual = _values(resident, take, code, endian)
        for reference, candidate in zip(expected, actual, strict=True):
            if not math.isfinite(reference) or not math.isfinite(candidate):
                raise ValueError("selection numerical validation: nonfinite numerical value")
            error = abs(candidate - reference)
            limit = policy["atol"] + policy["rtol"] * abs(reference)
            if not math.isfinite(error) or not math.isfinite(limit):
                raise ValueError("selection numerical validation: nonfinite comparison arithmetic")
            if error > limit:
                raise ValueError("selection numerical validation: numerical value exceeds source-bound tolerance")
            if error > maximum:
                maximum = error
        remaining -= take
    return {"values": count, "max_absolute_error": maximum}


def _compare_field(native, resident, policy):
    _check(type(native) is dict and set(native) == {"payload", "metadata"}
           and type(resident) is dict and set(resident) == {"payload", "metadata"}, "full-field references differ")
    layout = _layout(native["metadata"])
    _check(_layout(resident["metadata"]) == layout, "original storage bounds or shape differ")
    count = math.prod(layout["shape"])
    with _Reader(native["payload"]) as left, _Reader(resident["payload"]) as right:
        _check(left.before.st_size == right.before.st_size == count * 8, "full-field byte count differs from shape")
        result = _compare_values(left, right, count, code="d", endian="little" if policy["dtype"] == "<f8" else "big",
                                 policy=policy)
    return {**result, "layout": layout}


def _text_line(reader):
    try:
        return reader.line().decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ValueError("selection numerical validation: non-ASCII VTK header") from error


def _vtk_prefix(reader):
    _check(_text_line(reader).startswith("# vtk DataFile Version "), "not a legacy VTK file")
    _text_line(reader)  # Free-form title is not numerical data.
    _check(_text_line(reader) == "BINARY" and _text_line(reader) == "DATASET RECTILINEAR_GRID",
           "unsupported VTK encoding or dataset")
    fields = _text_line(reader).split()
    _check(len(fields) == 4 and fields[0] == "DIMENSIONS", "invalid VTK dimensions")
    shape = tuple(int(value) for value in fields[1:])
    _check(all(0 < value < 2**63 for value in shape), "invalid VTK dimension extent")
    return shape


def _compare_vtk(native, resident, expected_fields, policy):
    result = {}
    with _Reader(native) as left, _Reader(resident) as right:
        shape = _vtk_prefix(left)
        _check(_vtk_prefix(right) == shape, "VTK dimensions differ")
        for label, count in zip(("X_COORDINATES", "Y_COORDINATES", "Z_COORDINATES"), shape, strict=True):
            header = _text_line(left).split()
            _check(_text_line(right).split() == header and len(header) == 3 and header[0] == label
                   and int(header[1]) == count and header[2] in {"float", "double"}, "VTK coordinate layout differs")
            result[label] = _compare_values(left, right, count, code="f" if header[2] == "float" else "d",
                                            endian="big", policy=policy)
            _check(left.read(1) == right.read(1) == b"\n", "missing VTK coordinate delimiter")
        points = math.prod(shape)
        _check(_text_line(left).split() == _text_line(right).split() == ["POINT_DATA", str(points)],
               "VTK point count differs")
        seen = set()
        while True:
            left_header, right_header = _text_line(left), _text_line(right)
            if not left_header and not right_header:
                # A blank line is only accepted at EOF, not as permission to
                # ignore an unparsed trailing field.
                break
            header = left_header.split()
            _check(right_header.split() == header and len(header) in {3, 4}, "VTK field declaration differs")
            kind, name, dtype = header[:3]
            _check(name not in seen and name not in result and dtype in {"float", "double"},
                   "VTK data-field inventory differs")
            if kind == "SCALARS":
                components = 1 if len(header) == 3 else int(header[3])
                _check(1 <= components <= 4 and _text_line(left).split() == _text_line(right).split()
                       == ["LOOKUP_TABLE", "default"], "unsupported VTK scalar layout")
            else:
                _check(kind == "VECTORS" and len(header) == 3, "unsupported VTK data declaration")
                components = 3
            seen.add(name)
            result[name] = _compare_values(left, right, points * components,
                                          code="f" if dtype == "float" else "d", endian="big", policy=policy)
            _check(left.read(1) == right.read(1) == b"\n", "missing VTK field delimiter")
        _check(set(expected_fields) <= seen, "required VTK data field missing")
    return {"shape": list(shape), "fields": result}


def _run(reference, *, binary, binary_sha256, workload, identity, seconds=None):
    value = _json(reference)
    # Both binaries were already checked by the diagnostic/application build
    # verifier in this decision; its context rehashes them before acceptance.
    _check(type(value.get("argv")) is list and value["argv"] == [binary, *workload["argv"][1:]]
           and value.get("binary_sha256") == binary_sha256,
           "run executable or arguments differ")
    _check(type(value.get("exit_code")) is int and value["exit_code"] == 0 and value.get("timed_out") is False
           and value.get("workload_identity") == workload["identity"] and value.get("traced") is False
           and value.get("configuration_sha256") == workload["files"], "run workload or completion differs")
    _check(type(value.get("cwd")) is str and Path(value["cwd"]).is_absolute()
           and type(value.get("environment")) is dict
           and all(value["environment"].get(name) == expected for name, expected in workload["environment"].items())
           and value.get("affinity") == identity["affinity"] and value.get("threads") == identity["threads"],
           "run environment or placement differs")
    _check(type(value.get("completion")) is dict and all(value["completion"].get(name) == expected
           and type(value["completion"].get(name)) is type(expected) for name, expected in workload["completion"].items()),
           "run does not complete the authenticated workload")
    elapsed = value.get("diagnostic_complete_process_seconds")
    _check(type(elapsed) in {int, float} and math.isfinite(elapsed) and elapsed > 0
           and (seconds is None or elapsed == seconds), "run elapsed time differs from observed process")
    for name, expected in workload["files"].items():
        path = Path(value["cwd"]) / name
        _check(path.resolve().is_relative_to(Path(value["cwd"]).resolve()) and _hash_file(path) == expected,
               "run input differs from authenticated workload")
    return value


def _normalized_identity(value):
    result = deepcopy(value)
    _check(type(result) is dict and type(result.get("fortran_options")) is str, "observed execution identity missing")
    result["fortran_options"] = normalize_fortran_options(result["fortran_options"])
    return result


def _observations(reference, run, *, mode, contract, key_sequence, identity):
    _within(reference, run["cwd"])
    expected_cpu = {name: identity[name] for name in ("cpu_name", "fortran_version", "fortran_options", "affinity",
                                                    "effective_schedule", "environment")}
    expected_cpu["team"] = [identity[name] for name in ("threads", "dynamic", "proc_bind", "thread_limit", "max_active_levels")]
    observed = []
    with _Reader(reference) as reader:
        _check(reader.before.st_size <= MAX_JSON_BYTES, "observation log exceeds supported bound")
        while line := reader.line():
            if not line.startswith(OBSERVATION_PREFIX):
                continue
            row = json.loads(line[len(OBSERVATION_PREFIX):])
            ordinal = len(observed)
            _check(type(row) is dict and ordinal < len(key_sequence) and row.get("selection") == contract["selection_identity"]
                   and type(row.get("key")) is list and all(type(item) is int for item in row["key"])
                   and row["key"] == list(key_sequence[ordinal]) and row.get("outcome") == mode
                   and type(row.get("launches")) is int
                   and row["launches"] == (0 if mode == "native" else contract["expected_launches"]),
                   "diagnostic selected key, invocation or completion outcome differs")
            if mode == "native":
                _check(row.get("identity") is None and _normalized_identity(row.get("cpu_identity"))
                       == _normalized_identity(expected_cpu), "independent native identity differs")
            else:
                _check(row.get("cpu_identity") is None and _normalized_identity(row.get("identity"))
                       == _normalized_identity(identity), "resident execution identity differs")
            seconds = row.get("seconds")
            _check(type(seconds) in {int, float} and math.isfinite(seconds) and seconds > 0,
                   "invalid selected invocation time")
            observed.append({"ordinal": ordinal, **{name: row[name] for name in
                            ("selection", "key", "launches", "outcome", "seconds")}})
    _check(len(observed) == len(key_sequence) and sum(row["seconds"] for row in observed)
           <= run["diagnostic_complete_process_seconds"], "selected invocation sequence or total time differs")
    return observed


def _outputs(variants, runs, policy):
    expected = policy["outputs"]["files"]
    for mode in ("native", "resident"):
        _check(type(variants[mode].get("outputs")) is dict and set(variants[mode]["outputs"]) == set(expected)
               and type(runs[mode].get("output_hashes")) is dict
               and set(expected) <= set(runs[mode]["output_hashes"]), "required output file inventory differs")
        for name, reference in variants[mode]["outputs"].items():
            _within(reference, runs[mode]["cwd"], name)
            _check(runs[mode]["output_hashes"][name] == reference.get("sha256"), "output differs from run inventory")
    return {name: _compare_vtk(variants["native"]["outputs"][name], variants["resident"]["outputs"][name], fields, policy)
            for name, fields in expected.items()}


def validate_campaign_numerics(campaign, *, contract, workload, builds, key_sequence, identity):
    """Verify one complete dump pair and every timed pair's normal outputs."""
    from compiler.offload.application_build import validate_application_build_observation, validate_diagnostic_relink

    policy = normalize_policy(contract["validation_policy"], build_contract=contract["build_contract"])
    receipt = _json(campaign.get("numerical_validation"))
    _check(set(receipt) == {"schema_version", "protocol", "contract_sha256", "policy_identity", "workload_identity",
                            "selection_bundle_identity", "variants", "pairs"}
           and type(receipt["schema_version"]) is int and receipt["schema_version"] == 1
           and receipt["protocol"] == RECEIPT_PROTOCOL and receipt["contract_sha256"] == _digest(contract)
           and receipt["policy_identity"] == policy["identity"] and receipt["workload_identity"] == workload["identity"]
           and receipt["selection_bundle_identity"] == contract["selection_bundle"]["identity"],
           "numerical receipt source, policy or workload differs")
    variants = receipt["variants"]
    _check(type(variants) is dict and set(variants) == {"native", "resident"}, "independent diagnostic variants required")
    runs, diagnostic_builds = {}, {}
    for mode, variant in variants.items():
        _check(type(variant) is dict and set(variant) == {"diagnostic_build", "run", "observations", "fields", "outputs"},
               "diagnostic variant fields differ")
        validate_application_build_observation(builds[mode], contract=contract["build_contract"], mode=mode,
                                               selection_contract_identity=_digest(contract))
        diagnostic_builds[mode] = validate_diagnostic_relink(variant["diagnostic_build"], application_build=builds[mode],
            contract=contract["build_contract"], driver=policy["driver"], mode=mode)
        build = diagnostic_builds[mode]
        runs[mode] = _run(variant["run"], binary=build["binary"], binary_sha256=build["binary_sha256"],
                          workload=workload, identity=identity)
        _observations(variant["observations"], runs[mode], mode=mode, contract=contract,
                      key_sequence=key_sequence, identity=identity)
        _check(type(variant["fields"]) is dict and set(variant["fields"]) == set(policy["fields"]),
               "required full-field inventory differs")
        for name, specification in policy["fields"].items():
            field = variant["fields"][name]
            _check(type(field) is dict and set(field) == {"payload", "metadata"}, "full-field references differ")
            for kind, path in specification.items():
                _within(field[kind], runs[mode]["cwd"], path)
    fields = {name: _compare_field(variants["native"]["fields"][name], variants["resident"]["fields"][name], policy)
              for name in policy["fields"]}
    diagnostic_outputs = _outputs(variants, runs, policy)
    # Exact observations are authenticated by validate_diagnostic_relink above.
    # Resolve each immutable application attestation once, not once per pair.
    application_binaries = {
        mode: _json({"path": observation["attestation"], "sha256": observation["attestation_sha256"]})["binary"]
        for mode, observation in builds.items()
    }
    pairs, rows = receipt["pairs"], campaign["observations"]
    _check(type(pairs) is list and len(pairs) * 2 == len(rows), "every measured pair requires output validation")
    comparisons = []
    for ordinal, pair in enumerate(pairs):
        _check(type(pair) is dict and set(pair) == {"round", "native", "resident"}
               and type(pair["round"]) is int and pair["round"] == ordinal, "output pair order differs")
        runs = {}
        for mode in ("native", "resident"):
            variant = pair[mode]
            _check(type(variant) is dict and set(variant) == {"run", "observations", "outputs"}, "output pair fields differ")
            row = next((item for item in rows[2 * ordinal:2 * ordinal + 2] if item.get("mode") == mode), None)
            _check(row is not None and row.get("round") == ordinal, "timed output pair is missing")
            # The build observation is already authenticated by the cost caller.
            # Its attestation supplies the exact binary path, not a collector alias.
            runs[mode] = _run(variant["run"], binary=application_binaries[mode], binary_sha256=builds[mode]["binary_sha256"],
                              workload=workload, identity=identity, seconds=row["seconds"])
            observations = _observations(variant["observations"], runs[mode], mode=mode, contract=contract,
                                         key_sequence=key_sequence, identity=identity)
            _check(observations == row["invocations"], "timed invocation rows differ from original observation log")
        comparisons.append({"round": ordinal, "outputs": _outputs(pair, runs, policy)})
    return {"protocol": RECEIPT_PROTOCOL, "receipt_sha256": campaign["numerical_validation"]["sha256"],
            "policy_identity": policy["identity"], "diagnostic_builds": diagnostic_builds,
            "fields": fields, "diagnostic_outputs": diagnostic_outputs, "timed_pairs": comparisons}
