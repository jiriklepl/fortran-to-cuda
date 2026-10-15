"""Render frozen diagnostic observations; never build or run an application.

Usage: render.py --observations observations.json --size-receipt size.json --output DIR
The inputs are the checkpoint's observation list and emitter-size receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MODES = ("native", "sgs-direct", "resident-before", "resident-reuse")
LABELS = ("Native", "SGS direct\n(diffusion CPU)", "Diffusion before\n(SGS native)", "Diffusion reuse\n(SGS native)")
COLORS = ("#64748b", "#377bb5", "#ba7040", "#168078")
INK, MUTED, TARGET = "#24364b", "#596b7c", "#a53751"


def read(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def measured_samples(rows):
    """Reject partial protocols; never silently render them as three rounds."""
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError("Expected four modes, each with one warmup and three measured rounds")
    indexed, binaries = {}, {}
    for row in rows:
        mode, number = row.get("mode"), row.get("round")
        if mode not in MODES or type(number) is not int or number not in range(4):
            raise ValueError("Unexpected mode or round")
        key = mode, number
        if key in indexed or row.get("excluded_warmup") is not (number == 0):
            raise ValueError("Duplicate round or inconsistent warmup exclusion")
        seconds = row.get("seconds")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Complete-process seconds must be finite and positive")
        if row.get("qualification") is not False:
            raise ValueError("This plot requires explicitly diagnostic observations")
        digest = row.get("binary_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Each observation requires a binary SHA256")
        if binaries.setdefault(mode, digest) != digest:
            raise ValueError("Binary changed between rounds for " + mode)
        indexed[key] = float(seconds)
    if set(indexed) != {(mode, number) for mode in MODES for number in range(4)}:
        raise ValueError("Incomplete comparison rounds")
    return {mode: [indexed[mode, number] for number in (1, 2, 3)] for mode in MODES}


def source_metrics(receipt):
    totals, batches = receipt["totals"], receipt["batches"]
    if len(batches) != 3:
        raise ValueError("Expected the three source-bound wall batches")
    for direction in ("before", "after"):
        for metric in ("physical_reference_loops", "bytes"):
            values = [batch[direction][metric] for batch in batches]
            total = totals[direction][metric]
            if any(type(value) is not int or value <= 0 for value in [*values, total]):
                raise ValueError("Generated-source metrics must be positive integers")
            if sum(values) != total:
                raise ValueError("Generated-source totals differ from their batches")
    return totals


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e3e8ee", linewidth=.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#cbd4df")
    ax.tick_params(colors=INK, labelsize=9)


def save(fig, output, stem):
    for extension in ("svg", "png"):
        metadata = {"Date": None} if extension == "svg" else {"Software": "indirect-reuse-checkpoint-render"}
        fig.savefig(output / f"{stem}.{extension}", dpi=180, facecolor="white", metadata=metadata)
    plt.close(fig)


def performance(samples, digest, output):
    native = statistics.median(samples["native"])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.3))
    fig.suptitle("Inspector reuse: complete-process diagnostic", x=.055, ha="left", y=.97,
                 fontsize=19, fontweight="bold", color=INK)
    fig.text(.055, .905, "channel-simple · natural 109 steps · 4 threads · round 0 excluded; rounds 1–3 shown",
             fontsize=10.5, color=MUTED)
    for ax, denominator, title, ylabel in zip(
            axes, (1.0, native), ("Absolute wall time", "Relative to measured native median"),
            ("Complete-process wall time (s)", "Wall time / native median"), strict=True):
        style(ax)
        ax.set_title(title, loc="left", fontsize=12, pad=13, color=INK)
        largest = 9 / denominator
        for index, (mode, color) in enumerate(zip(MODES, COLORS, strict=True)):
            values = [value / denominator for value in samples[mode]]
            median, low, high = statistics.median(values), min(values), max(values)
            largest = max(largest, high)
            ax.errorbar(index, median, yerr=[[median - low], [high - median]],
                        fmt="none", ecolor=color, capsize=9, linewidth=2, zorder=2)
            ax.plot([index - .21, index + .21], [median, median], color=color, linewidth=3, zorder=4)
            for value, offset, marker in zip(values, (-.13, 0, .13), ("o", "s", "^"), strict=True):
                ax.scatter(index + offset, value, color=color, edgecolor="white", linewidth=.6,
                           marker=marker, s=49, zorder=3)
            suffix = " s" if denominator == 1.0 else "×"
            ax.annotate(f"{median:.2f}{suffix}", (index, high), xytext=(0, 12),
                        textcoords="offset points", ha="center", fontsize=10, color=color, fontweight="bold")
        if denominator != 1.0:
            ax.axhline(1, color=MUTED, linewidth=.8, linestyle=":")
        ax.axhline(9 / denominator, color=TARGET, linewidth=1.4, linestyle="--")
        ax.text(.98, 9 / denominator, "9 s target", ha="right", va="bottom", color=TARGET,
                fontsize=9, transform=ax.get_yaxis_transform())
        ax.set(xticks=range(4), xticklabels=LABELS, ylabel=ylabel, xlim=(-.5, 3.5), ylim=(0, largest * 1.22))
        ax.yaxis.label.set_color(INK)
    handles = [Line2D([], [], color=MUTED, marker=marker, linestyle="none", label=f"Round {number}")
               for number, marker in enumerate(("o", "s", "^"), 1)]
    handles += [Line2D([], [], color=MUTED, linewidth=3, label="Median; whiskers = min–max")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .18), ncol=4, frameon=False, fontsize=9)
    fig.text(.055, .135, "Resident modes: SGS native, same diffusion interval; complete-process times include observation logging.",
             fontsize=9.5, color=MUTED)
    fig.text(.055, .092, "Diagnostic sections policy, not AUTO or seven-round qualification. No channel performance qualification.",
             fontsize=9.5, color=MUTED)
    fig.text(.055, .045, f"Observations SHA256 {digest[:20]} · all measured samples normalized by the same native median",
             fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=.065, right=.975, bottom=.31, top=.79, wspace=.23)
    save(fig, output, "performance")


def inspector(totals, digest, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9))
    fig.suptitle("Less generated inspector work", x=.065, ha="left", y=.96,
                 fontsize=18, fontweight="bold", color=INK)
    fig.text(.065, .87, "Source metrics from the same three admitted native wall batches", fontsize=10, color=MUTED)
    for ax, metric, title in zip(axes, ("physical_reference_loops", "bytes"),
                                ("Generated reference loops", "Generated fragment bytes"), strict=True):
        values = [totals[direction][metric] for direction in ("before", "after")]
        style(ax)
        ax.bar([0, 1], values, color=[COLORS[2], COLORS[3]], width=.53, zorder=2)
        for index, value in enumerate(values):
            ax.text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=12,
                    color=INK, fontweight="bold")
        reduction = 100 * (1 - values[1] / values[0])
        ax.set_title(f"{title}\n{reduction:.1f}% reduction", loc="left", fontsize=11, pad=12, color=INK)
        ax.set(xticks=[0, 1], xticklabels=["Before", "Reuse"], ylim=(0, max(values) * 1.2))
        ax.ticklabel_format(axis="y", style="plain")
    fig.text(.065, .13, "Generated-source counts are not executed loop counts or a measured runtime improvement.",
             fontsize=9.5, color=MUTED)
    fig.text(.065, .075, f"Source-size receipt SHA256 {digest[:20]} · runtime evidence is shown separately",
             fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=.08, right=.98, bottom=.24, top=.72, wspace=.38)
    save(fig, output, "inspector")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--size-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows, observations_digest = read(args.observations)
    receipt, size_digest = read(args.size_receipt)
    samples, totals = measured_samples(rows), source_metrics(receipt)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "svg.hashsalt": "indirect-reuse-checkpoint-20261015"})
    args.output.mkdir(parents=True, exist_ok=True)
    performance(samples, observations_digest, args.output)
    inspector(totals, size_digest, args.output)


if __name__ == "__main__":
    main()
