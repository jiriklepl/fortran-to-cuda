#!/usr/bin/env python3
"""Render generic cold-invocation evidence; never execute or recalibrate it."""
import argparse
import hashlib
import gzip
import json
import math
import os
import statistics
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/placement-checkpoint-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "svg.fonttype": "none", "svg.hashsalt": "placement-checkpoint-v1",
                     "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"native": "#607286", "automatic": "#2679ad"}


def save(fig, directory, stem):
    fig.savefig(directory / (stem + ".svg"), metadata={"Date": None, "Creator": "placement-checkpoint-v1"})
    fig.savefig(directory / (stem + ".png"), dpi=160, metadata={"Software": "placement-checkpoint-v1"})
    plt.close(fig)


def samples(case):
    rows = case["production_validation"]["observations"]
    result = {}
    for mode in ("native", "automatic"):
        selected = sorted((r for r in rows if r["mode"] == mode), key=lambda r: r["round"])
        if [r["round"] for r in selected] != list(range(9)):
            raise ValueError("Expected exactly seven fit and two holdout observations per mode")
        if any(r["role"] != ("fit" if r["round"] < 7 else "holdout") for r in selected):
            raise ValueError("Unexpected production observation roles")
        if any(r.get("agreement") is not True or not math.isfinite(r["seconds"]) or r["seconds"] <= 0
               for r in selected):
            raise ValueError("Invalid or numerically unvalidated production sample")
        result[mode] = ([r["seconds"] for r in selected[:7]], [r["seconds"] for r in selected[7:]])
    return result


def performance(cases, directory, digest):
    fig, axes = plt.subplots(2, len(cases), figsize=(max(8, 5 * len(cases)), 7.8), squeeze=False)
    for column, case in enumerate(cases):
        values = samples(case)
        denominator = statistics.median(values["native"][0])
        choice = case["decision"]["selection"]
        for row in range(2):
            ax = axes[row, column]
            for x, mode in enumerate(("native", "automatic")):
                fit, hold = ([v / (denominator if row else 1) for v in part] for part in values[mode])
                median = statistics.median(fit)
                ax.bar(x, median, width=.52, color=COLORS[mode], alpha=.16)
                ax.errorbar(x, median, yerr=[[median - min(fit)], [max(fit) - median]],
                            color=COLORS[mode], capsize=6, linewidth=1.7, zorder=3)
                ax.plot([x - .24, x + .24], [median, median], color=COLORS[mode], linewidth=2.8)
                ax.scatter([x + (i - 3) * .045 for i in range(7)], fit, color=COLORS[mode], s=25, zorder=4)
                ax.scatter([x - .11, x + .11], hold, facecolors="white", edgecolors=COLORS[mode],
                           marker="D", s=45, linewidths=1.5, zorder=5)
                ax.annotate(f"median {median:.4f}" + ("×" if row else " s"),
                            (x, max([*fit, *hold])), xytext=(0, 10), textcoords="offset points",
                            ha="center", fontsize=9, color=COLORS[mode])
            ax.set_xticks([0, 1], ["Native", "AUTO"])
            ax.set_xlim(-.6, 1.6)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
            ax.grid(axis="y", alpha=.18)
            ax.set_ylabel("Time / native fit median" if row else "Invocation wall time (seconds)")
            if row:
                ax.axhline(1, color="#778899", linestyle="--", linewidth=1, alpha=.7)
            else:
                miss = case["production_validation"].get("model_miss", False)
                ax.set_title(f"{case['size']}³ interior · AUTO selects {choice}" + ("\nProduction model miss" if miss else ""))
    fig.suptitle("Independent production: native versus automatic placement", fontsize=16, y=.98)
    handles = [Line2D([], [], color="#41576b", marker="o", linestyle="none", label="Seven fit samples"),
               Line2D([], [], color="#41576b", marker="D", markerfacecolor="white", linestyle="none", label="Two holdouts"),
               Line2D([], [], color="#41576b", linewidth=2.8, label="Fit median; whiskers = fit range")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .062), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .038, "Cold fixture invocation, not a full application: includes ownership and CUDA startup when selected;",
             ha="center", fontsize=9)
    fig.text(.5, .018, "excludes process startup, input initialization, field output and teardown. No application target.  Evidence " + digest[:12],
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .12, 1, .94))
    save(fig, directory, "performance")


def authority_map(directory):
    fig, ax = plt.subplots(figsize=(11, 6.3))
    ax.set(xlim=(0, 11), ylim=(0, 6.3))
    ax.axis("off")
    def box(x, y, width, height, label, fill="#edf2f8", edge="#8ba0b6", size=11):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.04,rounding_size=0.08",
                                   facecolor=fill, edgecolor=edge, linewidth=1.3))
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=size, linespacing=1.5)
    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#52677b", "lw": 1.5})
    ax.text(.2, 6, "One placement authority for all consumers", fontsize=18, weight="bold")
    ax.text(.25, 5.5, "BEFORE", fontsize=12, weight="bold", color="#536576")
    ax.text(4.35, 5.5, "AFTER", fontsize=12, weight="bold", color="#2679ad")
    box(.25, 4.2, 3.1, .85, "Prepared record present", fill="#fff1df", edge="#d5ac74")
    box(.25, 1.7, 3.1, 1.9, "Separate boolean gates\nCost gate · leaf pricing\nControl · diagnostics", fill="#fff1df", edge="#d5ac74")
    arrow((1.8, 4.2), (1.8, 3.6))
    box(4.35, 4.2, 2.65, .85, "Checked execution plan")
    box(7.65, 4.2, 2.65, .85, "Exact evidence request")
    arrow((5.7, 4.2), (7.3, 3.65)); arrow((8.95, 4.2), (7.3, 3.65))
    box(4.9, 2.7, 4.8, .9, "One exact registered attachment\nBound to this plan and request", fill="#e3f2eb", edge="#74a48f")
    for x, label in ((4.15, "Cost gate"), (5.85, "Leaf pricing"), (7.55, "Control"), (9.25, "Diagnostics")):
        arrow((7.3, 2.7), (x + .65, 2.05))
        box(x, 1.3, 1.35, .7, label, size=9)
    box(.25, .18, 10.35, .65, "Application selected-interval pricing remains unsupported. No channel target or application speedup claim.",
        fill="#fff1df", edge="#d5ac74", size=10)
    fig.subplots_adjust(left=.015, right=.995, top=.99, bottom=.02)
    save(fig, directory, "placement-authority")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    if args.report.suffix == ".gz":
        raw = gzip.decompress(raw)
    cases = json.loads(raw)
    if not isinstance(cases, list) or not cases:
        raise ValueError("Expected the exact-window producer's nonempty report list")
    args.output.mkdir(parents=True, exist_ok=True)
    performance(cases, args.output, hashlib.sha256(raw).hexdigest())
    authority_map(args.output)


if __name__ == "__main__":
    main()
