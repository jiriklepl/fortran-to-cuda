#!/usr/bin/env python3
"""Evaluate speedup of New Build-06 Unified Resident vs Previous Dual Resident, SGS-only, and Native Baseline."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "elmm-pipeline"
sys.path.insert(0, str(PIPELINE))

from elmm_pipeline.validation import compare_vtk, read_vtk, validate_log

BENCH = ROOT / "benchmarks/results/reached-read-finalization-20261018"
CASES_DIR = ROOT / "benchmarks/results/checked-plan-application-20261014/cases"
BUILD_06 = Path("/tmp/fortran-startup-application-build-06/work/build/ELMM")
BUILD_OPT = Path("/tmp/ELMM_opt")

BINARIES = {
    "Native Baseline": BENCH / "native-origin-01/build/ELMM",
    "SGS-only Resident": BENCH / "resident-origin-01/build/ELMM",
    "Previous Dual Resident": BENCH / "build-simple-dual-01/build/ELMM",
    "Build-06 Unified Resident": BUILD_06,
    "New Unified Resident (Optimized)": BUILD_OPT,
}

OUTPUT_BASE = ROOT / "benchmarks/results/speedup-evaluation-20261102"


def setup_case_dir(case_name: str, target_dir: Path, steps: int | None = None) -> Path:
    process_dir = target_dir / "process"
    shutil.rmtree(target_dir, ignore_errors=True)
    process_dir.mkdir(parents=True, exist_ok=True)
    (process_dir / "output").mkdir(exist_ok=True)

    src_case = CASES_DIR / case_name
    for conf in src_case.glob("*.conf"):
        shutil.copy2(conf, process_dir / conf.name)
    if (src_case / "origin.json").is_file():
        shutil.copy2(src_case / "origin.json", process_dir / "origin.json")

    if steps is not None:
        conf_file = process_dir / "time_stepping.conf"
        content = conf_file.read_text()
        if "!  max_number_of_time_steps" in content:
            content = content.replace("!  max_number_of_time_steps = 3000,", f"  max_number_of_time_steps = {steps},")
        elif "max_number_of_time_steps" in content:
            content = re.sub(r"max_number_of_time_steps\s*=\s*\d+,", f"max_number_of_time_steps = {steps},", content)
        else:
            content += f"\n  max_number_of_time_steps = {steps},\n"
        conf_file.write_text(content)

    return process_dir


def run_single(name: str, binary: Path, run_dir: Path, expected_steps: int | None = None) -> dict:
    env = dict(os.environ)
    for k in list(env):
        if k.startswith(("ELMM_", "FORT_")):
            del env[k]
    env.update({
        "OMP_NUM_THREADS": "4",
        "OMP_DYNAMIC": "FALSE",
        "OMP_SCHEDULE": "static",
        "OMP_PROC_BIND": "false",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })

    cmd = [str(binary), "output_dir='output/'"]
    started = time.perf_counter()
    res = subprocess.run(cmd, cwd=run_dir, env=env, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started

    (run_dir / "stdout.log").write_text(res.stdout)
    (run_dir / "stderr.log").write_text(res.stderr)

    if res.returncode != 0:
        tail = "\n".join(res.stderr.splitlines()[-20:])
        raise RuntimeError(f"Simulation {name} failed with code {res.returncode}:\n{tail}")

    timing = validate_log(res.stdout, expected_steps or 1, expected_completed_steps=expected_steps)
    return {
        "config": name,
        "elapsed_wall": elapsed,
        "loop_time": timing.get("time_steps_seconds"),
        "poisson_time": timing.get("poisson_seconds"),
        "completed_steps": timing.get("completed_steps"),
        "run_dir": str(run_dir),
    }


def evaluate_case(case_name: str, steps: int | None = None) -> list[dict]:
    print(f"\n{'='*80}")
    print(f"  BENCHMARK SUITE: {case_name.upper()} ({steps or 'natural'} timesteps)")
    print(f"{'='*80}", flush=True)

    results = []
    case_out = OUTPUT_BASE / case_name

    # 1. Warmup run with 5 steps for each binary to stabilize CPU/GPU clocks and caches
    print("--> Warming up CPU/GPU...", flush=True)
    warmup_dir = case_out / "warmup"
    for name, bin_path in BINARIES.items():
        w_dir = setup_case_dir(case_name, warmup_dir / name.replace(" ", "_"), steps=5)
        run_single(f"warmup_{name}", bin_path, w_dir, expected_steps=5)

    # 2. Benchmark runs
    for name, bin_path in BINARIES.items():
        print(f"--> Running: {name} ...", end="", flush=True)
        r_dir = setup_case_dir(case_name, case_out / name.replace(" ", "_"), steps=steps)
        info = run_single(name, bin_path, r_dir, expected_steps=steps)
        results.append(info)
        print(f" done: loop={info['loop_time']:.3f}s, wall={info['elapsed_wall']:.3f}s", flush=True)

    # Baseline reference for speedups and VTK comparison
    base = results[0]
    base_run_dir = Path(base["run_dir"])
    base_out_vtk = read_vtk(base_run_dir / "output/out.vtk") if (base_run_dir / "output/out.vtk").is_file() else None

    for r in results:
        loop_time = r["loop_time"]
        r["speedup_loop"] = (base["loop_time"] / loop_time) if loop_time else 0.0
        r["speedup_wall"] = base["elapsed_wall"] / r["elapsed_wall"]

        # Numerical comparison
        cand_run_dir = Path(r["run_dir"])
        cand_out_vtk = read_vtk(cand_run_dir / "output/out.vtk") if (cand_run_dir / "output/out.vtk").is_file() else None
        if base_out_vtk and cand_out_vtk and r["config"] != "Native Baseline":
            cmp_res = compare_vtk(base_out_vtk, cand_out_vtk, atol=1e-5, rtol=1e-5)
            r["cmp_out"] = cmp_res
            max_err = max(f.get("max_absolute_error", 0.0) for f in cmp_res["fields"].values())
            r["numerical_status"] = f"PASSED (≤ {max_err:.1e})" if cmp_res["passed"] else "FAILED"
        else:
            r["numerical_status"] = "Baseline"

    # Print markdown table
    print(f"\n### Benchmark Comparison: {case_name} ({steps or results[0]['completed_steps']} Steps)\n")
    print("| Configuration | Timestep Loop Time | Total Wall Time | Speedup (Loop) | Speedup (Wall) | Poisson Time | Numerical ($p, u$) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        print(f"| **{r['config']}** | {r['loop_time']:.3f} s | {r['elapsed_wall']:.3f} s | **{r['speedup_loop']:.2f}x** | {r['speedup_wall']:.2f}x | {r['poisson_time']:.2f} s | {r['numerical_status']} |")

    return results


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # Case 1: channel-simple (109 steps)
    all_results["channel-simple"] = evaluate_case("channel-simple", steps=109)

    # Case 2: channel (100 calibrated steps)
    all_results["channel"] = evaluate_case("channel", steps=100)

    summary_file = OUTPUT_BASE / "summary.json"
    summary_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved summary to: {summary_file}")


if __name__ == "__main__":
    main()
