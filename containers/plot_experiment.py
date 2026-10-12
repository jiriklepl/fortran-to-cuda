#!/usr/bin/env python3
"""Render paired ELMM receipts with Python 3.9+ and Matplotlib/NumPy only.

This entry point deliberately imports neither the pipeline nor the compiler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def summarize(runs):
    """Recompute the receipt statistics before plotting recorded measurements."""
    summary = {}
    for mode in ("native", "current"):
        measured = [row for row in runs if row["mode"] == mode and not row["warmup"]]
        if any(row["traced"] for row in measured):
            raise ValueError("diagnostic execution cannot enter timing samples")
        samples = [row["wall_seconds"] for row in measured]
        if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
            raise ValueError("positive finite complete-wall measurements required")
        summary[mode] = {"median": statistics.median(samples), "min": min(samples), "max": max(samples),
                         "repetitions": len(samples), "samples": samples}
    for row in summary.values():
        row["ratio_to_native"] = row["median"] / summary["native"]["median"]
    return summary


def plot_experiment(report, output):
    """Plot only this paired experiment, with no dependency on archived runs."""
    report = Path(report).resolve(strict=True)
    value = json.loads(report.read_text())
    unvalidated = (value.get("status") == "completed_unvalidated"
                   and value.get("baseline_repeatability") is False
                   and value.get("correctness_qualified") is False
                   and value.get("target_met") is None
                   and value["protocol"].get("allow_native_variation") is True)
    if (not (value.get("status") == "passed" or unvalidated)
            or value["summary"] != summarize(value["runs"])
            or not value["gpu_evidence"]["verified"]
            or not unvalidated and not all(row["validation"]["passed"] for row in value["correctness"] + value["runs"])):
        raise ValueError("successful fresh timing and GPU/VTK evidence required")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = [value["summary"][mode] for mode in ("native", "current")]
    for relative in (False, True):
        baseline = rows[0]["median"] if relative else 1
        fig, ax = plt.subplots(figsize=(7, 7))
        medians = [row["median"] / baseline for row in rows]
        maximum = max(row["max"] / baseline for row in rows)
        ax.bar([0, 1], medians, width=.58, color=["#475569", "#0284c7"],
               yerr=[[median - row["min"] / baseline for median, row in zip(medians, rows)],
                     [row["max"] / baseline - median for median, row in zip(medians, rows)]], capsize=5)
        for position, row in enumerate(rows):
            ax.scatter(position + np.linspace(-.12, .12, len(row["samples"])),
                       np.array(row["samples"]) / baseline, s=30, color="white", edgecolor="#172033", zorder=3)
            ax.text(position, row["max"] / baseline + .03 * maximum,
                    f"{medians[position]:.3f}" + ("×" if relative else " s"), ha="center")
        if relative:
            ax.axhline(.8, color="#15803d", linestyle="--", label="20% lower wall time than original native")
            ax.legend(loc="upper center", fontsize=10)
        title = f"{value['case']} · {value['protocol']['threads']} host threads"
        if unvalidated:
            title += "\nUNVALIDATED: native outputs are not repeatable"
        policy = value.get("selection", {}).get("scope_transfers", "direct")
        ax.set(xticks=[0, 1], xticklabels=["Original native", "SGS " + policy], ylim=(0, 1.18 * maximum),
               ylabel="Complete wall time / original native" if relative else "Complete wall time (s)",
               title=title)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        eligible = value["performance_eligible"] and value.get("comparison_budget_eligible", True)
        heading = ("Unvalidated timings; no correctness or target claim" if unvalidated else
                   "Fresh paired measurements" if eligible else "Smoke/separate-budget run; no target claim")
        validation_note = ("Native repeatability failed; field mismatches retained. No promotion." if unvalidated else
                           "Separate GPU diagnostics; VTK validation only. Automatic policy not promoted.")
        fig.text(.5, .05, heading
                 + f" · n={rows[0]['repetitions']} · {value['protocol']['warmups']} excluded warmup(s)\n"
                 "Dots: samples; bars: medians; whiskers: min–max. Startup/output/teardown included.\n"
                 + validation_note,
                 ha="center", fontsize=10, linespacing=1.5)
        fig.subplots_adjust(bottom=.24, top=.90, left=.14, right=.97)
        name = "complete-wall-relative" if relative else "complete-wall-absolute"
        for extension in ("png", "svg", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=180)
        plt.close(fig)
    write(output / "source.json", {"report": str(report),
          "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
          "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "rendering_only": True, "measurement_started_utc": value["started_utc"],
          "plotting_python": sys.executable, "plotting_python_version": sys.version,
          "files_sha256": {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted(output.iterdir()) if path.is_file()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        plot_experiment(args.report, args.output_dir)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, "error: " + str(error) + "\n")
    print("Figures:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
