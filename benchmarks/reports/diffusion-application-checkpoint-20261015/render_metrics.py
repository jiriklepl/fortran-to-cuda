"""Render frozen complete-process samples and separately traced transfer totals."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MODES = ("native", "sgs-direct", "sgs-plus-diffusion")
LABELS = ("Native", "SGS direct", "SGS + diffusion\n(forced diagnostic)")
COLORS = ("#596c81", "#1677a8", "#bc641b")


def save(fig, name):
    for extension in ("svg", "png", "pdf"):
        metadata = ({"Date": None} if extension == "svg" else
                    {"CreationDate": None, "ModDate": None} if extension == "pdf" else None)
        fig.savefig(HERE / f"{name}.{extension}", dpi=150, metadata=metadata)
    path = HERE / f"{name}.svg"
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    report = json.loads((HERE / "report.json").read_text())
    for row in report["evidence"]:
        raw = (HERE / row["archived"]).read_bytes()
        if row["compressed"]:
            raw = gzip.decompress(raw)
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Frozen evidence changed: " + row["archived"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "svg.hashsalt": "diffusion-checkpoint-metrics-20261015",
                         "axes.spines.top": False, "axes.spines.right": False})
    samples = [report["performance"][mode]["samples_seconds"] for mode in MODES]
    native_median = statistics.median(samples[0])
    target = report["targets"]["channel-simple"]["seconds_below"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, divisor, ylabel in zip(axes, (1, native_median),
                                  ("Complete wall time (seconds)", "Wall time / fresh native median"), strict=True):
        for i, (values, color) in enumerate(zip(samples, COLORS, strict=True)):
            points = [value / divisor for value in values]
            median = statistics.median(points)
            ax.vlines(i, min(points), max(points), color=color, linewidth=2)
            ax.scatter([i - .10, i, i + .10], points, color=color, s=42, zorder=3)
            ax.scatter([i], [median], color="black", marker="D", s=32, zorder=4)
            ax.annotate(f"median {median:.2f}", (i, max(points)), xytext=(0, 14),
                        textcoords="offset points", ha="center", fontsize=10)
        ax.axhline(target / divisor, color="#b02a37", linestyle="--", linewidth=1.2,
                   label="Target: below 9 seconds")
        if divisor != 1:
            ax.axhline(1, color="#596c81", linewidth=.7, alpha=.7)
        ax.set_xticks(range(3), LABELS)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(max(values) for values in samples) / divisor * 1.23)
        ax.set_xlim(-.5, 2.5)
        ax.grid(axis="y", alpha=.15)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("channel-simple: isolated GPU diffusion regresses complete-process time", fontsize=17)
    fig.text(.5, .91, "109 unchanged steps • 4 threads • one excluded warmup + 3 rotated rounds per mode",
             ha="center", fontsize=11)
    fig.text(.5, .025, "Dots: individual runs; diamond: median; line: range. Untraced binaries. Forced placement; no qualification claim.",
             ha="center", fontsize=10)
    fig.subplots_adjust(top=.83, bottom=.18, left=.07, right=.97, wspace=.25)
    save(fig, "performance")

    rows = [report["transfers"][mode]["observed"] for mode in MODES[1:]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, suffix, divisor, ylabel in zip(axes, ("bytes", "events"), (2**30, 1),
            ("Observed transfers (GiB)", "Logical transfer events"), strict=True):
        uploads = [row[f"upload_{suffix}"] / divisor for row in rows]
        downloads = [row[f"download_{suffix}"] / divisor for row in rows]
        ax.bar(range(2), uploads, color="#1677a8", width=.5, label="Upload")
        ax.bar(range(2), downloads, bottom=uploads, color="#bc641b", width=.5, label="Download")
        for i, (upload, download) in enumerate(zip(uploads, downloads, strict=True)):
            label = f"{upload + download:.2f} GiB" if suffix == "bytes" else f"{int(upload + download):,} events"
            ax.text(i, upload + download, label, ha="center", va="bottom", fontsize=11)
        ax.set_xticks(range(2), LABELS[1:])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(u + d for u, d in zip(uploads, downloads, strict=True)) * 1.2)
        ax.set_xlim(-.65, 1.65)
        ax.grid(axis="y", alpha=.15)
        ax.legend(frameon=False)
    fig.suptitle("Diffusion adds transfers instead of eliminating a round trip", fontsize=17)
    fig.text(.5, .91, "Separate traced 109-step runs • SGS: 328 launches • SGS + diffusion: 1,309 launches",
             ha="center", fontsize=11)
    fig.text(.5, .055, "Includes entry, native corrections, exit and cleanup. All observed contexts closed successfully.",
             ha="center", fontsize=10)
    fig.text(.5, .02, "Totals agree with pipeline parser; SGS canonical attribution incomplete. Predictions unavailable. Logical events ≠ CUDA copy calls.",
             ha="center", fontsize=9)
    fig.subplots_adjust(top=.83, bottom=.20, left=.07, right=.97, wspace=.27)
    save(fig, "transfers")


if __name__ == "__main__":
    main()
