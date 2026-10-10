"""Public build semantics shared by generated numerical code and calibration.

This is a build requirement, not a numerical equivalence or IEEE-observer
proof. NVCC does not expose all of these options through preprocessor macros;
consumers must validate and record the actual compile command.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from hashlib import sha256

CONTRACT_ID = "separate-arithmetic-v1"
CUDA_OPTIONS = ("--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
HOST_OPTIONS = ("-ffp-contract=off",)


def numerical_build_contract() -> dict:
    """Return fresh, canonical, versioned metadata for independent consumers."""
    payload = {"schema_version": 1, "id": CONTRACT_ID,
               "required_cuda_options": list(CUDA_OPTIONS),
               "required_host_options": list(HOST_OPTIONS)}
    identity = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**payload, "identity": identity}


def require_numerical_build_contract(contract) -> None:
    """Absent/legacy or altered requirements cannot authorize current costs."""
    if (not isinstance(contract, dict) or type(contract.get("schema_version")) is not int
            or contract != numerical_build_contract()):
        raise ValueError("numerical build contract missing or incompatible; require " + CONTRACT_ID)


def cuda_compile_options() -> tuple[str, ...]:
    """Apply both device and generated host-worker requirements to NVCC."""
    return (*CUDA_OPTIONS, *("-Xcompiler=" + option for option in HOST_OPTIONS))


def require_explicit_cuda_environment(environment: Mapping[str, str] | None = None) -> None:
    """Hidden NVCC flags cannot silently override receipted requirements."""
    values = os.environ if environment is None else environment
    for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        if values.get(name):
            raise ValueError("numerical build contract requires explicit CUDA options; unset " + name)


def numerical_source_prologue() -> str:
    """Bind requirements into source/artifact identity without a false guard."""
    contract = numerical_build_contract()
    return ("// Compiler numerical build contract: " + contract["id"] + " " + contract["identity"] + "\n"
            "// Required NVCC options: " + " ".join(cuda_compile_options()) + "\n"
            "// Consumers must enforce the public contract in the actual compile command.\n")
