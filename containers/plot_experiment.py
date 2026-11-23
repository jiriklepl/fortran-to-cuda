#!/usr/bin/env python3
"""Render paired ELMM receipts with Python 3.9+ and Matplotlib/NumPy only.

This entry point deliberately does not import the compiler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


PHASE_SPECS = (
    ("initialization", "initialization_seconds", "Initialization", "#f59e0b", "#172033"),
    ("loop", "loop_seconds", "Loop", "#0284c7", "#ffffff"),
    ("finalization", "finalization_seconds", "Finalization", "#10b981", "#ffffff"),
)


def summarize_phases(rows):
    timings = [row.get("application_timing") or {} for row in rows]
    present = [any(timing.get(key) is not None for _phase, key, *_rest in PHASE_SPECS) for timing in timings]
    if not any(present):
        return None
    phases = {}
    for phase, key, *_rest in PHASE_SPECS:
        values = [timing.get(key) for timing in timings]
        if any(value is None or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("incomplete or invalid resident application phase timings")
        samples = [float(value) for value in values]
        phases[phase] = {"samples_seconds": samples, "median_seconds": statistics.median(samples)}
    return phases


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def plot_resident(value, report, output):
    policy = value["validation_policy"]
    rules = {"natural-channel-componentwise-relative-l2-1-percent": "Natural channel: 1% componentwise L2",
             "pointwise-1e-6-1e-5": "Pointwise field and VTK validation",
             "pointwise-sp-1.0-1e-2": "Single precision: pointwise field and VTK validation, 1% relative"}
    if policy not in rules:
        raise ValueError("unsupported resident validation policy")
    natural = policy == "natural-channel-componentwise-relative-l2-1-percent"
    diagnostics = value["diagnostics"]
    activity = (diagnostics[2].get("gpu_activity", {}).get("gpu_kernels_by_procedure", {})
                if len(diagnostics) == 3 else {})
    if (value.get("status") != "passed" or len(diagnostics) != 3
            or [row["mode"] for row in diagnostics] != ["native", "native", "resident"]
            or any(activity.get(name, 0) <= 0 for name in ("dynamics::convection", "subgrid::sgs_sigma_uniform"))):
        raise ValueError("successful resident diagnostics and GPU evidence required")
    for row in diagnostics[1:]:
        if not row.get("fields") or not all(result["passed"] for result in row["fields"].values()):
            raise ValueError("resident field validation failed")
        result = row.get("outputs" if natural else "vtk", {})
        if not result.get("passed"):
            raise ValueError("resident output validation failed")
    if any(row["traced"] or not row.get("outputs" if natural else "vtk", {}).get("passed")
           for row in value["runs"]):
        raise ValueError("all resident timing runs must be untraced and validated")
    summary = {}
    for mode in ("native", "resident"):
        rows = [row for row in value["runs"] if row["mode"] == mode and not row["warmup"]]
        if (not rows or any(row["traced"] or not row.get("outputs" if natural else "vtk", {}).get("passed")
                            or not math.isfinite(row["wall_seconds"]) or row["wall_seconds"] <= 0 for row in rows)):
            raise ValueError("validated, untraced resident timing samples required")
        samples = [row["wall_seconds"] for row in rows]
        entry = {"samples_seconds": samples, "median_seconds": statistics.median(samples)}
        phases = summarize_phases(rows)
        if phases is not None:
            entry["phases"] = phases
        summary[mode] = entry
    if ("phases" in summary["native"]) != ("phases" in summary["resident"]):
        raise ValueError("both modes must report phase timings together")
    summary["speedup"] = summary["native"]["median_seconds"] / summary["resident"]["median_seconds"]
    if "phases" in summary["native"] and summary["resident"]["phases"]["loop"]["median_seconds"] > 0:
        summary["loop_speedup"] = (
            summary["native"]["phases"]["loop"]["median_seconds"]
            / summary["resident"]["phases"]["loop"]["median_seconds"]
        )
    if value["summary"] != summary:
        raise ValueError("resident timing summary does not match samples")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = [summary[mode] for mode in ("native", "resident")]
    has_phases = "phases" in summary["native"]
    for relative in (False, True):
        baseline = rows[0]["median_seconds"] if relative else 1
        medians = [row["median_seconds"] / baseline for row in rows]
        if has_phases:
            phase_heights = {
                phase: [row["phases"][phase]["median_seconds"] / baseline for row in rows]
                for phase, *_rest in PHASE_SPECS
            }
            phase_totals = [
                sum(phase_heights[phase][idx] for phase, *_rest in PHASE_SPECS)
                for idx in range(len(rows))
            ]
            overheads = [max(0.0, medians[idx] - phase_totals[idx]) for idx in range(len(rows))]
            bar_tops = [max(medians[idx], phase_totals[idx]) for idx in range(len(rows))]
        else:
            phase_heights = {}
            overheads = [0.0, 0.0]
            bar_tops = medians
        maximum = max(
            max(max(row["samples_seconds"]) / baseline for row in rows),
            max(bar_tops),
        )
        show_overhead = has_phases and max(overheads) > 0.005 * maximum
        fig, ax = plt.subplots(figsize=(7.6, 7.2))
        if has_phases:
            bottoms = np.zeros(len(rows), dtype=float)
            segments = [(phase, label, color, text_color, phase_heights[phase])
                        for phase, _key, label, color, text_color in PHASE_SPECS]
            if show_overhead:
                segments.append(("overhead", "Process overhead", "#cbd5e1", "#172033", overheads))
            for _phase, label, color, text_color, heights in segments:
                arr = np.asarray(heights, dtype=float)
                ax.bar([0, 1], arr, bottom=bottoms, width=.58, color=color,
                       edgecolor="white", linewidth=1.2, label=label, zorder=2)
                for position, height in enumerate(arr):
                    if height >= 0.075 * maximum:
                        text = f"{height:.2f}x" if relative else (f"{height:.2f} s" if height >= 1.0 else f"{height:.3f} s")
                        ax.text(position, bottoms[position] + 0.5 * height, text,
                                ha="center", va="center", color=text_color, fontsize=9, fontweight="bold", zorder=3)
                bottoms = bottoms + arr
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01),
                      ncol=len(segments), frameon=False, fontsize=9.5, handlelength=1.2, columnspacing=1.1)
        else:
            ax.bar([0, 1], medians, width=.58, color=["#475569", "#0284c7"], zorder=2)
        for position, row in enumerate(rows):
            samples = np.asarray(row["samples_seconds"]) / baseline
            ax.scatter(position + np.linspace(-.12, .12, len(samples)), samples,
                       s=30, color="white", edgecolor="#172033", zorder=4)
            top = max(float(np.max(samples)), bar_tops[position])
            ax.text(position, top + .03 * maximum,
                    f"{medians[position]:.3f}" + ("x" if relative else " s"), ha="center", zorder=4)
        speedup_text = f"{summary['speedup']:.2f}x speedup"
        if "loop_speedup" in summary:
            speedup_text = f"{summary['speedup']:.2f}x wall speedup ({summary['loop_speedup']:.2f}x loop)"
        ax.set(xticks=[0, 1], xticklabels=["Native", "Resident GPU"],
               ylim=(0, (1.28 if has_phases else 1.18) * maximum),
               ylabel="Complete wall time / native" if relative else "Complete wall time (s)")
        ax.set_title(
            f"{value['case']} | {value['threads']} host threads, "
            f"{value.get('gpu_threads', value['threads'])} while the GPU run's fields are on the device\n"
            f"{speedup_text}",
            fontsize=11,
            pad=10,
        )
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        bar_note = "Dots: complete wall samples; stacked bars: phase medians" if has_phases else "Dots: samples; bars: medians"
        fig.text(.5, .07, f"{rules[policy]}; release qualification not established.\n"
                 f"{bar_note}; {value['warmups']} excluded warmup(s).",
                 ha="center", fontsize=10, linespacing=1.5)
        fig.subplots_adjust(bottom=.21, top=.88, left=.14, right=.97)
        name = "complete-wall-relative" if relative else "complete-wall-absolute"
        for extension in ("png", "svg", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=180)
        plt.close(fig)
    write(output / "source.json", {
        "report": str(report), "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rendering_only": True, "validation_policy": policy,
        "files_sha256": {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(output.iterdir()) if path.is_file()}})


def plot_experiment(report, output):
    """Plot one resident comparison report, with no dependency on archived runs."""
    report = Path(report).resolve(strict=True)
    value = json.loads(report.read_text())
    if "validation_policy" not in value:
        raise ValueError("only resident comparison reports can be plotted")
    return plot_resident(value, report, output)


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
