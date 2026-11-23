"""Small container front end; numerical decisions stay in the compiler."""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("WORKSPACE_ROOT", HERE.parent)).resolve()
PIPELINE = Path(os.environ.get("PIPELINE_ROOT", WORKSPACE / "elmm-pipeline")).resolve()
COMPILER = Path(os.environ.get("COMPILER_ROOT", WORKSPACE)).resolve()
sys.path.insert(0, str(PIPELINE))


def source_identity(root):
    excluded = {".git", ".cache", ".venv", "work", "results", "__pycache__", ".pytest_cache"}
    hashes = {}
    for directory, children, files in os.walk(root):
        children[:] = sorted(set(children) - excluded)
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix != ".pyc":
                hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return {"commit": commit.stdout.strip() if commit.returncode == 0 else None, "files": hashes}


def visible_gpu_uuids():
    """Use CUDA ordinals, which may differ from NVML/SLURM physical indices."""
    cuda = ctypes.CDLL("libcuda.so.1")
    def check(status):
        if status:
            raise RuntimeError(f"CUDA driver device probe failed (status {status})")
    check(cuda.cuInit(0))
    count = ctypes.c_int()
    check(cuda.cuDeviceGetCount(ctypes.byref(count)))
    result = []
    for ordinal in range(count.value):
        device = ctypes.c_int()
        value = (ctypes.c_ubyte * 16)()
        check(cuda.cuDeviceGet(ctypes.byref(device), ordinal))
        get_uuid = getattr(cuda, "cuDeviceGetUuid_v2", cuda.cuDeviceGetUuid)
        check(get_uuid(ctypes.byref(value), device))
        result.append("GPU-" + str(uuid.UUID(bytes=bytes(value))))
    return result


def selected_gpu(requested=None):
    smi = shutil.which("nvidia-smi")
    if not smi:
        raise RuntimeError("nvidia-smi is unavailable; run inside the allocated GPU environment")
    result = subprocess.run([smi, "--query-gpu=index,uuid,name,compute_cap,driver_version",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    devices = [dict(zip(("index", "uuid", "name", "compute_capability", "driver"),
                        (part.strip() for part in row), strict=True)) for row in csv.reader(result.stdout.splitlines())]
    # nvidia-smi alone ignores CUDA_VISIBLE_DEVICES and can expose other jobs'
    # GPUs. Match driver UUIDs rather than guessing scheduler ordinal mappings.
    visible = visible_gpu_uuids()
    devices = [{**d, "cuda_ordinal": ordinal} for ordinal, identifier in enumerate(visible)
               for d in devices if d["uuid"] == identifier]
    if len(devices) != len(visible):
        raise RuntimeError("CUDA devices cannot be matched to NVIDIA GPU metadata (MIG is unsupported)")
    if requested is not None:
        devices = [d for d in devices if requested == str(d["cuda_ordinal"]) or d["uuid"].startswith(requested)]
    if not devices:
        raise RuntimeError("selected GPU is absent or excluded by CUDA_VISIBLE_DEVICES")
    if requested is not None and len(devices) != 1:
        raise ValueError("GPU selector must identify exactly one allocated GPU")
    return devices[0]


def doctor(args, *, require_gpu=False, fortran=None):
    require_pipeline()
    from memory import memory_headroom
    fortran = fortran or os.environ.get("FC", "gfortran")
    result = {"python": sys.version, "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "memory_headroom_bytes": memory_headroom(), "tools": {},
              "source_roots": {"compiler": str(COMPILER), "pipeline": str(PIPELINE)}}
    for name in (fortran, args.nvcc or "nvcc", args.cuda_host or "g++-14"):
        path = shutil.which(name)
        result["tools"][name] = {"path": path}
        if path:
            probe = subprocess.run([path, "--version"], capture_output=True, text=True, check=True)
            result["tools"][name]["version"] = probe.stdout.strip()
    try:
        result["gpu"] = selected_gpu(args.gpu)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        if require_gpu:
            raise
        result["gpu"] = {"available": False, "reason": str(error)}
    if require_gpu:
        if not shutil.which(fortran):
            raise RuntimeError("native Fortran compiler is unavailable")
        from gpu_toolchain import discover_toolchain
        gpu = result["gpu"]
        architecture = gpu["compute_capability"].replace(".", "")
        explicit = args.arch.removeprefix("sm_").replace(".", "")
        if explicit not in {"auto", "native", architecture}:
            raise ValueError(f"requested architecture {args.arch} differs from selected GPU sm_{architecture}")
        toolchain = discover_toolchain(languages=("cuda",), nvcc=args.nvcc,
                                      cuda_host=args.cuda_host, arch=architecture)
        supported = subprocess.run([toolchain["nvcc"], "--list-gpu-code"],
                                   capture_output=True, text=True, check=True).stdout
        if "sm_" + architecture not in supported.split():
            raise RuntimeError(f"selected nvcc cannot generate sm_{architecture}; use a compatible CUDA image/toolkit")
        result["toolchain"] = toolchain
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    return result


def require_pipeline():
    if not (PIPELINE / "pipeline.py").is_file() or not (PIPELINE / "elmm_pipeline/__init__.py").is_file():
        if PIPELINE != (WORKSPACE / "elmm-pipeline").resolve():
            hint = "Set PIPELINE_ROOT to a complete pipeline checkout."
        else:
            hint = (f"Initialize it with 'git -C {shlex.quote(str(WORKSPACE))} "
                    "submodule update --init -- elmm-pipeline', or set PIPELINE_ROOT to a complete checkout.")
        raise RuntimeError(f"ELMM pipeline checkout is missing or incomplete: {PIPELINE}. {hint}")


def plotting_python():
    """Choose plotting dependencies independently of the compiler interpreter."""
    explicit = os.environ.get("PLOT_PYTHON")
    candidates = ([explicit] if explicit is not None else
                  [sys.executable, "python3", "python3.12", "python3.11", "python3.10", "python3.9",
                   "/usr/bin/python3", "/usr/bin/python3.9"])
    failures, seen = [], set()
    probe = ("import sys; "
             "assert sys.version_info >= (3, 9), 'plotting needs Python 3.9+'; "
             "import matplotlib, numpy")
    for candidate in candidates:
        executable = shutil.which(candidate) if candidate else None
        if not executable:
            failures.append(str(candidate) + ": interpreter not found")
            continue
        if executable in seen:
            continue
        seen.add(executable)
        try:
            result = subprocess.run([executable, "-c", probe], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired) as error:
            failures.append(executable + ": " + str(error))
            continue
        if result.returncode == 0:
            return executable
        failures.append(executable + ": " + (result.stderr.strip() or "Matplotlib/NumPy probe failed"))
    prefix = "PLOT_PYTHON is unusable" if explicit is not None else "No plotting interpreter with Matplotlib/NumPy found"
    raise RuntimeError(prefix + "; set PLOT_PYTHON to a Python 3.9+ interpreter with those packages. "
                       + " | ".join(failures))


def plot_experiment(report, output):
    """Render with an independent plotting Python; compiler Python is unchanged."""
    interpreter = plotting_python()
    print("Plotting interpreter:", interpreter, flush=True)
    subprocess.run([interpreter, str(HERE / "plot_experiment.py"), "--report", str(report),
                    "--output-dir", str(output)], check=True)


def reproduce(args):
    plot_experiment(args.experiment, args.output_dir / "figures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, aliases in (("doctor", []), ("reproduce-plots", ["reproduce"])):
        child = sub.add_parser(name, aliases=aliases)
        child.add_argument("--output-dir", "--output", "-o", type=Path,
                           default=Path(os.environ.get("OUTPUT_DIR", "/output" if Path("/output").exists() else WORKSPACE / "out")))
        child.add_argument("--work-dir", type=Path, default=os.environ.get("WORK_DIR"))
        if name != "reproduce-plots":
            child.add_argument("--arch", default=os.environ.get("CUDA_ARCH", "auto"))
            child.add_argument("--gpu", help="Visible CUDA ordinal or GPU UUID; defaults to first visible GPU")
            child.add_argument("--nvcc", default=os.environ.get("NVCC"))
            child.add_argument("--cuda-host", default=os.environ.get("CUDAHOSTCXX", os.environ.get("CUDA_HOST_CXX")))
        if name == "reproduce-plots":
            child.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.work_dir = (args.work_dir or args.output_dir / "work").resolve()
    if args.command == "doctor":
        print(json.dumps(doctor(args), indent=2))
    else:
        reproduce(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
