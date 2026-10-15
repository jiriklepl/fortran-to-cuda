"""Render an archived native-demand scope checkpoint without running workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/native-demand-checkpoint-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, MUTED = "#24364b", "#586b7c"
GPU, NATIVE, CLOSED = "#e3f2ec", "#e8eff8", "#fff0df"


def validate(report):
    if report["schema_version"] != 1 or report["claims"]["qualification"] is not False:
        raise ValueError("Expected an explicitly non-qualification checkpoint")
    if report["automatic_placement"]["selection"] != "native":
        raise ValueError("This checkpoint has no automatic interval cost authority")
    if any(row["status"] != "unmet" for row in report["primary_targets"]):
        raise ValueError("No application qualification may be inferred from this fixture")
    ledger = report["transfer_ledger"]
    for label in ("prediction", "observations"):
        rows = ledger[label]
        if rows is None:
            continue
        if {row["resource"] for row in rows} != {"a", "b", "star", "out"} or len(rows) != 4:
            raise ValueError("Transfer ledger requires each fixture resource exactly once")
        if any(type(row[key]) is not int or row[key] < 0 for row in rows
               for key in ("upload_bytes", "download_bytes")):
            raise ValueError("Transfer bytes must be nonnegative integers")
    if ledger["observations"] is not None and not report["claims"]["fixture_cuda_execution"]:
        raise ValueError("Observed GPU transfers require a completed execution receipt")


def draw(report, digest, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "svg.hashsalt": "native-demand-checkpoint-20261015-v1"})
    fig, ax = plt.subplots(figsize=(13.8, 10.6))
    ax.set(xlim=(0, 14), ylim=(0, 11))
    ax.axis("off")

    def text(x, y, label, *, size=10, color=INK, weight="normal", **kwargs):
        ax.text(x, y, label, fontsize=size, color=color, weight=weight, **kwargs)

    def box(x, y, width, height, label, color=NATIVE, size=10):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
            boxstyle="round,pad=.035,rounding_size=.07", facecolor=color,
            edgecolor="#95a6b5", linewidth=1))
        text(x + width / 2, y + height / 2, label, size=size, ha="center", va="center", linespacing=1.4)

    def arrow(x1, y1, x2, y2, color=MUTED, style="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.4, "linestyle": style})

    text(.25, 10.55, "Keep the original native call inside one resident owner", size=19, weight="bold")
    runtime = report["fixture"]["runtime_validation"]
    subtitle = "Generic 3D fixture · original numerical order · " + (
        "runtime evidence pending" if runtime == "pending" else runtime)
    text(.25, 10.16, subtitle, size=11, color=MUTED)
    text(.25, 9.65, "BEFORE · native-call closure is unsupported", size=11, weight="bold")
    xs, widths = [.25, 3.05, 5.85, 8.65, 11.45], [2.3, 2.3, 2.3, 2.3, 2.25]
    before = ["Producer\nGPU alternative", "Publish + close\nOriginal native CALL",
              "Diffusion\nNative suffix", "Accumulation\nNative suffix", "Host outputs"]
    for index, (x, width, label) in enumerate(zip(xs, widths, before, strict=True)):
        box(x, 8.58, width, .82, label, GPU if index == 0 else CLOSED)
        if index < 4:
            arrow(x + width + .03, 8.99, xs[index + 1] - .08, 8.99)
    text(.25, 8.23, "Source-placement baseline; no measured before/after timing or transfer comparison is implied.",
         size=9.5, color=MUTED)
    text(.25, 7.76, "AFTER · source-bound demanded effects, static storage separation and checked final coverage", size=11,
         weight="bold")
    after = ["Producer\nGPU alternative", "Original native CALL\nCPU · once", "Diffusion\nGPU alternative",
             "Accumulation\nGPU alternative", "Publish outputs\nClose owner"]
    for index, (x, width, label) in enumerate(zip(xs, widths, after, strict=True)):
        box(x, 6.67, width, .82, label, NATIVE if index in {1, 4} else GPU)
        if index < 4:
            arrow(x + width + .03, 7.08, xs[index + 1] - .08, 7.08)
    lanes = [(6.17, "a · old input", "HOST + DEVICE CURRENT",
              "Host copy stays current at the CALL; device copy remains valid"),
             (5.74, "b · producer output", "DEVICE CURRENT",
              "Retained until diffusion; publication occurs at owner exit"),
             (5.31, "star · diffusion output", "INTERIOR ON DEVICE",
              "Interior retained through accumulation; original halos preserved")]
    for y, label, state, note in lanes:
        text(.25, y, label, size=9.5, weight="bold")
        text(3.05, y, state, size=9, weight="bold", color="#327c67")
        text(6.15, y, note, size=9.2, color=MUTED)
    text(.25, 4.88, "Metadata · native only", size=9.5, weight="bold")
    text(3.05, 4.88, "Original allocatable component updates · no device capture · no assumed branch outcome", size=9.5,
         color=MUTED)

    ledger = report["transfer_ledger"]
    expected = {row["resource"]: row for row in ledger["prediction"]}
    observed = {row["resource"]: row for row in ledger["observations"] or []}
    text(.25, 4.32, "TRANSFER LEDGER · bytes per invocation, including final publication", size=11, weight="bold")
    for x, label in [(0.4, "Resource"), (3.3, "Predicted upload / download"), (8.0, "Observed upload / download")]:
        text(x, 3.9, label, size=10, weight="bold")
    for index, resource in enumerate(("a", "b", "star", "out")):
        y = 3.5 - index * .34
        row, actual = expected[resource], observed.get(resource)
        text(.4, y, resource, size=10)
        text(3.3, y, f"{row['upload_bytes']:,} / {row['download_bytes']:,}", size=10)
        text(8.0, y, f"{actual['upload_bytes']:,} / {actual['download_bytes']:,}" if actual else "Pending receipt",
             size=10, color=INK if actual else MUTED)
    counts = ledger["observed_counts"]
    count_label = "Observed transfer counts pending" if counts is None else (
        f"Observed per invocation: {counts['uploads']} uploads · {counts['downloads']} downloads · "
        f"{counts['launches']} GPU launches · {counts['closes']} owner close")
    text(.4, 1.99, count_label, size=9.5, color=MUTED)
    if report.get("fixture_extensions"):
        text(.4, 1.72, "Also verified: stale native correction publication; pointer rejection closes after one GPU prefix; suffix runs once.",
             size=9, color=MUTED)
    box(.25, .91, 13.45, .72,
        "AUTO still selects native: native interval costs require compatible source-bound observations.\n"
        "No application speedup or qualification claim. Both channel targets remain unmet.", CLOSED, size=10.5)
    final = report["implementation"]["compiler_commit"]
    identity = "Compiler commit pending" if final is None else "Compiler " + final[:12]
    text(.25, .39, identity + " · Report SHA256 " + digest[:20] + " · Predicted bytes are not measurements",
         size=8.5, color=MUTED)
    fig.subplots_adjust(left=.015, right=.995, top=.995, bottom=.015)
    for extension in ("svg", "png"):
        metadata = ({"Date": None, "Creator": "native-demand-checkpoint-v1"} if extension == "svg" else
                    {"Software": "native-demand-checkpoint-v1"})
        fig.savefig(output / ("operation-resource-map." + extension), dpi=170, metadata=metadata)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("report.json"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    report = json.loads(raw)
    validate(report)
    args.output.mkdir(parents=True, exist_ok=True)
    draw(report, hashlib.sha256(raw).hexdigest(), args.output)


if __name__ == "__main__":
    main()
