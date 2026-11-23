"""Validate application completion and numeric outputs without NumPy."""
from array import array
import math
from pathlib import Path
import re
import sys


def validate_log(text, expected_max_steps=None, *, expected_completed_steps=None):
    failures = ["ELMM Abort:", "Solution diverged.", "Maximum clock time exceeded.",
                "Fortran runtime error", "ERROR STOP", "SIGSEGV", "Floating point exception"]
    for marker in failures:
        if marker.lower() in text.lower():
            raise ValueError(f"Application reported failure: {marker}")
    steps = [int(s) for s in re.findall(r"tstep:\s*(\d+)", text)]
    if not steps or max(steps) <= 0 or not re.search(r"(?m)^\s*saved\.\s*$", text):
        raise ValueError("Application did not complete timesteps and save final results")
    natural_end = "Time limit reached." in text or "Steady state reached." in text
    if not natural_end and (expected_max_steps is None or max(steps) != expected_max_steps):
        raise ValueError("Application stopped before the requested step/time limit")
    if expected_completed_steps is not None and max(steps) != expected_completed_steps:
        raise ValueError(f"Workload differs from baseline: completed {max(steps)} timesteps, "
                         f"expected {expected_completed_steps}")
    times = {}
    for name, label in (("time_steps_seconds", "Total wall clock time for time steps"),
                        ("poisson_seconds", "Wall clock time for Poisson solver"),
                        ("initialization_seconds", "Wall clock time for initialization"),
                        ("loop_seconds", "Wall clock time for loop"),
                        ("finalization_seconds", "Wall clock time for finalization")):
        match = re.search(re.escape(label) + r":\s*([\d.]+)", text)
        times[name] = float(match[1]) if match else None
    if times["time_steps_seconds"] is None:
        raise ValueError("Missing application timestep timing")
    return {"completed_steps": max(steps), "natural_end": natural_end, **times}


def read_vtk(path, *, require_flow_fields=True):
    fields = {}
    with Path(path).open("rb") as stream:
        def line():
            return stream.readline().decode("ascii").strip()

        def values(count, dtype):
            if dtype not in {"float", "double"}:
                raise ValueError(f"Unsupported VTK type: {dtype}")
            data = array("f" if dtype == "float" else "d")
            raw = stream.read(count * data.itemsize)
            if len(raw) != count * data.itemsize:
                raise ValueError("Truncated VTK data")
            data.frombytes(raw)
            if sys.byteorder == "little":
                data.byteswap()
            if not all(math.isfinite(v) for v in data):
                raise ValueError("Non-finite VTK field")
            if stream.read(1) != b"\n":
                raise ValueError("Missing VTK field delimiter")
            return data

        if not line().startswith("# vtk DataFile"):
            raise ValueError("Not a legacy VTK file")
        line()
        if line() != "BINARY" or line() != "DATASET RECTILINEAR_GRID":
            raise ValueError("Expected binary RECTILINEAR_GRID VTK")
        dimensions = line().split()
        if dimensions[0] != "DIMENSIONS":
            raise ValueError("Missing VTK dimensions")
        shape = tuple(map(int, dimensions[1:]))
        for label, n in zip(("X_COORDINATES", "Y_COORDINATES", "Z_COORDINATES"), shape):
            parts = line().split()
            if parts[0] != label or int(parts[1]) != n:
                raise ValueError("Unexpected VTK coordinates")
            fields[label] = values(n, parts[2])
        parts = line().split()
        count = math.prod(shape)
        if parts != ["POINT_DATA", str(count)]:
            raise ValueError("Unexpected VTK point count")
        while True:
            header = line()
            if not header:
                if stream.peek(1) if hasattr(stream, "peek") else False:
                    continue
                break
            parts = header.split()
            if parts[0] == "SCALARS":
                components = int(parts[3]) if len(parts) == 4 else 1
                if not line().startswith("LOOKUP_TABLE"):
                    raise ValueError("Missing VTK scalar lookup table")
            elif parts[0] == "VECTORS":
                components = 3
            else:
                raise ValueError(f"Unsupported VTK declaration: {header}")
            if parts[1] in fields:
                raise ValueError("Duplicate VTK field")
            fields[parts[1]] = values(count * components, parts[2])
    if require_flow_fields and ("u" not in fields or "p" not in fields):
        raise ValueError("Validation requires final velocity u and pressure p fields")
    if len(fields) <= 3:
        raise ValueError("VTK requires at least one numeric field")
    return shape, fields


def compare_vtk(reference, candidate, *, atol=1e-6, rtol=1e-5):
    if any(not math.isfinite(t) or t < 0 for t in (atol, rtol)):
        raise ValueError("Validation tolerances must be finite and nonnegative")
    ref_shape, ref_fields = reference
    shape, fields = candidate
    if shape != ref_shape or fields.keys() != ref_fields.keys():
        raise ValueError("Final-state field names or dimensions differ")
    checks = {}
    for name, expected in ref_fields.items():
        actual = fields[name]
        if len(actual) != len(expected):
            raise ValueError(f"Field length mismatch: {name}")
        worst, failures = 0.0, 0
        for a, b in zip(expected, actual):
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError(f"Non-finite comparison field: {name}")
            error = abs(a - b)
            worst = max(worst, error)
            if error > atol + rtol * abs(a):
                failures += 1
        checks[name] = {"max_absolute_error": worst, "failed_values": failures,
                        "values": len(expected)}
    return {"passed": all(c["failed_values"] == 0 for c in checks.values()),
            "atol": atol, "rtol": rtol, "fields": checks,
            "note": "ELMM final VTK values are float32, including double-precision builds."}
