"""Find the CUDA compiler, its host compiler and the visible device.

Installations are discovered on PATH or in normal system locations; no sibling
checkout or globally installed Python package is used.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from pathlib import Path


def _executable(explicit: str | Path | None, names: list[str], candidates: list[Path]) -> str:
    choices = [str(explicit)] if explicit else [*names, *(str(path) for path in candidates)]
    for choice in choices:
        located = shutil.which(choice)
        if located:
            return str(Path(located).resolve())
    label = str(explicit) if explicit else " / ".join(names)
    raise RuntimeError(f"GPU compiler {label!r} was not found; install it or supply its explicit path.")


def _version(executable: str) -> str:
    result = subprocess.run([executable, "--version"], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Could not query GPU compiler {executable}: {result.stderr.strip()}")
    return (result.stdout + result.stderr).strip()


def probe_gpu(nvidia_smi: str | Path | None = None) -> dict:
    """Report visible NVIDIA devices without treating sandbox denial as absence.

    A failed probe records its diagnostic. Compilation is still possible with an
    explicit architecture; a successful query alone is not execution evidence.
    """
    command = shutil.which(str(nvidia_smi or "nvidia-smi"))
    if command is None:
        return {"available": False, "devices": [], "error": "nvidia-smi was not found"}
    argv = [command, "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(argv, text=True, capture_output=True, check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "devices": [], "error": str(error), "argv": argv}
    devices = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 3:
                devices.append(dict(zip(("name", "driver_version", "compute_capability"), fields)))
    return {"available": bool(devices), "devices": devices, "argv": argv,
            "error": (result.stderr or result.stdout).strip() if not devices else None}


def _architecture(value: object) -> str:
    architecture = str(value).removeprefix("sm_").removeprefix("cc").replace(".", "")
    if not re.fullmatch(r"[1-9][0-9]{1,2}[af]?", architecture):
        raise ValueError(f"Invalid NVIDIA GPU architecture {value!r}; use a compute capability such as 86.")
    return architecture


def _cuda_runtime_directory(cuda_root: Path, compiler: str, explicit: str | Path | None) -> tuple[Path, dict]:
    if explicit is not None:
        directory = Path(explicit).resolve()
        if not (directory / "libcudart.so").is_file():
            raise RuntimeError("Explicit CUDA runtime directory has no libcudart.so: " + str(directory))
        return directory, {"method": "explicit", "directory": str(directory)}
    argv = [compiler, "-dumpmachine"]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Could not query CUDA host compiler target; supply cuda_runtime_dir explicitly") from error
    target = result.stdout.strip()
    if (result.returncode or len(target) > 128
            or not re.fullmatch(r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_.]+)+", target)):
        raise RuntimeError("Invalid CUDA host compiler target; supply cuda_runtime_dir explicitly")
    parts = target.lower().split("-")
    # Red Hat's native GCC reports e.g. x86_64-redhat-linux, omitting the
    # GNU ABI suffix. It uses the same CUDA target as x86_64-linux-gnu.
    # Keep explicit x32/musl (and unknown ABI) targets out of this alias.
    linux_gnu = ("linux" in parts and parts[-1] == "gnu") or (
        len(parts) == 3 and parts[1:] == ["redhat", "linux"])
    targets = cuda_root / "targets"
    candidates = [targets / target / "lib"]
    # Architecture-only CUDA directories do not distinguish x32/musl ABIs.
    if linux_gnu:
        candidates.append(targets / (parts[0] + "-linux") / "lib")
    for directory in candidates:
        if (directory / "libcudart.so").is_file():
            return directory.resolve(), {"method": "host_target", "target": target, "argv": argv}
    # Older native toolkits have only lib64/lib. A cross compiler must not
    # silently pick the running host's runtime, nor a symlink to another target.
    if linux_gnu and parts[0] == platform.machine().lower() and platform.system() == "Linux":
        for directory in (cuda_root / "lib64", cuda_root / "lib"):
            resolved = directory.resolve()
            if not resolved.is_relative_to(targets.resolve()) and (resolved / "libcudart.so").is_file():
                return resolved, {"method": "native_flat_layout", "target": target, "argv": argv}
    raise RuntimeError(f"CUDA runtime libcudart.so for host target {target!r} was not found beneath "
                       f"{cuda_root}; supply cuda_runtime_dir explicitly")


def discover_toolchain(
    options: dict | None = None,
    languages: tuple[str, ...] | list[str] = ("cuda",),
    *,
    nvcc: str | Path | None = None,
    cuda_host: str | Path | None = None,
    arch: str | None = None,
    cuda_runtime_dir: str | Path | None = None,
) -> dict:
    """Return JSON-serializable paths, versions, architecture and device status.

    CUDA defaults to g++-14 when installed, because a newer system GNU compiler
    may exceed nvcc's supported host range. Supply ``cuda_host`` to override it.
    If device visibility is restricted, callers must provide ``arch`` explicitly.
    ``cuda_runtime_dir`` (also accepted in options) selects a custom runtime
    directory; otherwise the selected CUDA host compiler determines the target.
    """
    options = dict(options or {})
    unsupported = set(languages) - {"cuda"}
    if unsupported:
        raise ValueError(f"Unsupported GPU languages: {sorted(unsupported)}")
    device = probe_gpu(options.get("nvidia_smi"))
    architecture = arch or options.get("architecture") or options.get("arch")
    if architecture is None and device["devices"]:
        architecture = device["devices"][0]["compute_capability"]
    if architecture is None:
        raise RuntimeError("Cannot determine the GPU architecture from nvidia-smi; supply an explicit architecture (for example 86). "
                           + str(device["error"]))
    config: dict = {"architecture": _architecture(architecture), "device": device, "versions": {}}
    if "cuda" in languages:
        config["nvcc"] = _executable(nvcc or options.get("nvcc") or os.environ.get("NVCC"),
                                      ["nvcc"], [Path("/usr/local/cuda/bin/nvcc")])
        config["cuda_host_cxx"] = _executable(
            cuda_host or options.get("cuda_host_cxx") or options.get("cuda_host") or os.environ.get("CUDAHOSTCXX"),
            ["g++-14", "g++"], [])
        config["versions"]["nvcc"] = _version(config["nvcc"])
        config["versions"]["cuda_host_cxx"] = _version(config["cuda_host_cxx"])
        cuda_root = Path(config["nvcc"]).parent.parent
        override = cuda_runtime_dir if cuda_runtime_dir is not None else options.get(
            "cuda_runtime_dir", options.get("cuda_library_dir"))
        runtime_dir, discovery = _cuda_runtime_directory(cuda_root, config["cuda_host_cxx"], override)
        config["cuda_library_dir"] = str(runtime_dir)
        config["cuda_runtime_discovery"] = discovery
    return config
