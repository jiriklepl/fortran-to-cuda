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


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


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


def doctor(args, *, require_gpu=False):
    from elmm_pipeline.campaigns.guard import memory_headroom
    result = {"python": sys.version, "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "memory_headroom_bytes": memory_headroom(), "tools": {},
              "source_roots": {"compiler": str(COMPILER), "pipeline": str(PIPELINE)}}
    for name in (os.environ.get("FC", "gfortran"), args.nvcc or "nvcc", args.cuda_host or "g++-14"):
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
        if not shutil.which(os.environ.get("FC", "gfortran")):
            raise RuntimeError("native Fortran compiler is unavailable")
        from elmm_pipeline.gpu_toolchain import discover_toolchain
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


def prepare(args):
    """Copy only maintained pipeline code into writable storage before setup."""
    target = args.work_dir / "pipeline"
    identity = source_identity(PIPELINE)
    marker = target / "container-source.json"
    if target.exists():
        if not marker.is_file() or json.loads(marker.read_text())["files"] != identity["files"]:
            raise ValueError("prepared pipeline differs; select a new --work-dir")
    else:
        shutil.copytree(PIPELINE, target, ignore=shutil.ignore_patterns(
            ".git", ".cache", "work", "results", ".venv", "__pycache__", ".pytest_cache"))
        write(marker, identity)
    command = [sys.executable, str(target / "pipeline.py"), "setup", "--tools", "baseline,local",
               "--local-compiler", str(COMPILER)]
    # Setup is explicit and repeatable; a prepared cache avoids worker-node downloads.
    subprocess.run(command, check=True, cwd=target)
    return target


def guarded(args):
    """Return True only in the guarded worker, never silently run locally."""
    if args.worker:
        return True
    from elmm_pipeline.campaigns.guard import available
    if available() < 3 * 2**30:
        raise RuntimeError("less than 3 GiB effective available memory; no job started")
    print("Output directory:", args.output_dir, flush=True)
    print("Progress log:", args.output_dir / "run.log", flush=True)
    command = [sys.executable, "-m", "elmm_pipeline.campaigns.guard",
               "--record", str(args.output_dir / "guard.json"),
               "--log", str(args.output_dir / "run.log"), "--",
               sys.executable, str(HERE / "workflow.py"), *sys.argv[1:], "--worker"]
    subprocess.run(command, env={**os.environ, "PYTHONPATH": str(PIPELINE)}, check=True)
    return False


def run_elmm(args):
    if args.dry_run:
        print(json.dumps({"action": "fresh original native versus SGS sections/direct", "case": args.case,
                          "config": str(args.config) if args.config else None, "steps": args.steps,
                          "threads": args.threads, "jobs": args.jobs, "repeats": args.repeats,
                          "warmups": args.warmups, "architecture": args.arch,
                          "allow_native_variation": args.allow_native_variation,
                          "output": str(args.output_dir / "experiment"), "work": str(args.work_dir),
                          "target_wall_ratio_to_original_native": .80, "executes_work": False}, indent=2))
        return
    if (args.output_dir / "experiment").exists():
        raise ValueError("existing experiment is immutable; choose a new --output-dir")
    if not guarded(args):
        print(f"Results: {args.output_dir / 'experiment/report.json'}")
        return
    machine = doctor(args, require_gpu=True)
    if min(machine["memory_headroom_bytes"].values()) < 3 * 2**30:
        raise RuntimeError("less than 3 GiB effective available memory")
    if max(args.threads, args.jobs) > len(machine["cpu_affinity"]):
        raise ValueError("jobs/threads exceed the allocated CPU affinity")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir / "machine.json", machine)
    target = PIPELINE
    if args.app_root is None:
        cache = PIPELINE / ".cache"
        prepared = all((cache / path).exists() for path in (
            "setup.json", "repos/elmm/src/SConstruct", "repos/PoisFFT/src/SConstruct",
            "envs/local/bin/python", "envs/build/bin/python", "native/lib"))
        if prepared:
            print("Using prepared pipeline cache:", cache, flush=True)
        else:
            target = prepare(args)
    native = target / ".cache/native/lib"
    compiler_python = args.compiler_python or target / ".cache/envs/local/bin/python"
    # Source analysis needs the prepared parser environment, even when the
    # node's newer system Python has no third-party packages installed.
    command = [str(compiler_python), str(HERE / "sgs_experiment.py")]
    for flag, value in {
        "app-root": args.app_root or target / ".cache/repos", "compiler-root": COMPILER,
        "pipeline-root": PIPELINE, "python": compiler_python,
        "scons-python": args.scons_python or target / ".cache/envs/build/bin/python",
        "native-library-dir": args.native_library_dir or native,
        "output": args.output_dir / "experiment", "work-dir": args.work_dir / "experiment",
        "case": args.case, "jobs": args.jobs, "threads": args.threads,
        "repeats": args.repeats, "warmups": args.warmups, "timeout": args.timeout,
        "nvcc": machine["toolchain"]["nvcc"], "cuda-host": machine["toolchain"]["cuda_host_cxx"],
        "gpu-arch": machine["toolchain"]["architecture"],
    }.items():
        command.extend(["--" + flag, str(value)])
    if args.config is not None:
        command.extend(["--config", str(args.config)])
    if args.steps is not None:
        command.extend(["--steps", str(args.steps)])
    if args.allow_native_variation:
        command.append("--allow-native-variation")
    print("Source-analysis interpreter:", compiler_python, flush=True)
    subprocess.run(command, check=True)
    result = json.loads((args.output_dir / "experiment/report.json").read_text())
    print(json.dumps(result.get("summary", result), indent=2))
    print("Results:", args.output_dir / "experiment/report.json", flush=True)
    plot_experiment(args.output_dir / "experiment/report.json", args.output_dir / "experiment/figures")


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
    if args.experiment:
        plot_experiment(args.experiment, args.output_dir / "figures")
    elif args.snapshot:
        from elmm_pipeline.campaigns.report import plot_snapshot
        plot_snapshot(args.snapshot, args.output_dir / "figures", gpu_label="SGS direct")
    elif args.campaign:
        from elmm_pipeline.campaigns.report import plot
        plot(args.campaign, args.output_dir / "figures", historical=True)
    else:
        if (args.output_dir / "ordinary-figures").exists():
            raise ValueError("existing figures are immutable; choose a new --output-dir")
        subprocess.run([sys.executable, str(WORKSPACE / "benchmarks/harness/plot_strategies.py"),
                        str(args.policy_report), "--output", str(args.output_dir / "ordinary-figures"),
                        "--label", "Historical measurements"], check=True)


def benchmarks(args):
    # Keep the older microbenchmark suite explicit and bounded; never swallow failures.
    if not guarded(args):
        print(f"Results: {args.output_dir / 'microbenchmarks.csv'}")
        return
    machine = doctor(args, require_gpu=True)
    os.environ.update(CUDA_ARCH="sm_" + machine["toolchain"]["architecture"],
                      OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE",
                      NVCC=shlex.quote(machine["toolchain"]["nvcc"]),
                      NVCCFLAGS=shlex.join(["-O3", "-ccbin", machine["toolchain"]["cuda_host_cxx"]]))
    if args.threads > len(machine["cpu_affinity"]):
        raise ValueError("threads exceed the allocated CPU affinity")
    work = args.work_dir / "microbenchmarks"
    work.mkdir(parents=True, exist_ok=False)
    for name in ("benchmarks", "compiler"):
        shutil.copytree((COMPILER if name == "compiler" else WORKSPACE) / name, work / name,
                        ignore=shutil.ignore_patterns("results", "bin", "__pycache__", "*.o", "*.mod"))
    code = ("from benchmarks.harness import run,check\n"
            "for c in run.FUNCTIONS:\n"
            "    reference = check.values(run.run_once(run.build(c,'Fortran',16,16,16,1,0,main_src='test_main.f90')),16**3)\n"
            "    for v in ['Fortran-OMP','CUDA']:\n"
            "        check.compare_values(reference,check.values(run.run_once(run.build(c,v,16,16,16,1,0,main_src='test_main.f90')),16**3))\n"
            "print(run.CSV_HEADER)\n"
            "for c in run.FUNCTIONS:\n"
            "    for v in ['Fortran-OMP','CUDA']:\n"
            "        print(run.run_combination(c,v,128,64,64,10),flush=True)\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "microbenchmarks.csv").open("x") as stream:
        subprocess.run([sys.executable, "-c", code], cwd=work,
                       env={**os.environ, "PYTHONPATH": str(work)}, stdout=stream, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, aliases in (("doctor", []), ("prepare", []), ("run-elmm", ["elmm"]),
                          ("reproduce-plots", ["reproduce"]), ("run-benchmarks", ["benchmark"])):
        child = sub.add_parser(name, aliases=aliases)
        child.add_argument("--output-dir", "--output", "-o", type=Path,
                           default=Path(os.environ.get("OUTPUT_DIR", "/output" if Path("/output").exists() else WORKSPACE / "out")))
        child.add_argument("--work-dir", type=Path, default=os.environ.get("WORK_DIR"))
        if name != "reproduce-plots":
            child.add_argument("--arch", default=os.environ.get("CUDA_ARCH", "auto"))
            child.add_argument("--gpu", help="Visible CUDA ordinal or GPU UUID; defaults to first visible GPU")
            child.add_argument("--nvcc", default=os.environ.get("NVCC"))
            child.add_argument("--cuda-host", default=os.environ.get("CUDAHOSTCXX", os.environ.get("CUDA_HOST_CXX")))
        if name == "run-elmm":
            child.add_argument("--case", choices=("channel-simple", "channel"), default="channel-simple")
            child.add_argument("--config", type=Path)
            child.add_argument("--steps", type=int, help="Explicit smoke-run cap; default preserves original duration")
            child.add_argument("--threads", type=int, default=4)
            child.add_argument("--jobs", type=int, default=8)
            child.add_argument("--repeats", type=int, default=7)
            child.add_argument("--warmups", type=int, default=1)
            child.add_argument("--timeout", type=int, default=900)
            child.add_argument("--allow-native-variation", action="store_true",
                               help="Complete unvalidated timings if native output varies; no target or promotion claim")
            child.add_argument("--app-root", type=Path, help="Explicit prepared source root with elmm/PoisFFT siblings")
            child.add_argument("--compiler-python", type=Path)
            child.add_argument("--scons-python", type=Path)
            child.add_argument("--native-library-dir", type=Path)
            child.add_argument("--dry-run", action="store_true")
            child.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
        if name in {"run-benchmarks", "prepare"}:
            child.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
        if name == "run-benchmarks":
            child.add_argument("--threads", type=int, default=4)
        if name == "reproduce-plots":
            select = child.add_mutually_exclusive_group(required=True)
            select.add_argument("--snapshot", type=Path)
            select.add_argument("--experiment", type=Path)
            select.add_argument("--campaign", type=Path)
            select.add_argument("--policy-report", type=Path)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.work_dir = (args.work_dir or args.output_dir / "work").resolve()
    if args.command in {"run-elmm", "elmm"}:
        if (min(args.threads, args.jobs, args.repeats, args.timeout) <= 0 or args.warmups < 0
                or args.steps is not None and args.steps <= 0):
            parser.error("budgets/repetitions/steps must be positive; warmups may be zero")
        run_elmm(args)
    elif args.command == "doctor":
        print(json.dumps(doctor(args), indent=2))
    elif args.command == "prepare":
        if guarded(args):
            print(prepare(args))
        else:
            print(f"Prepared pipeline: {args.work_dir / 'pipeline'}")
    elif args.command in {"reproduce-plots", "reproduce"}:
        reproduce(args)
    else:
        if args.threads <= 0:
            parser.error("threads must be positive")
        benchmarks(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
