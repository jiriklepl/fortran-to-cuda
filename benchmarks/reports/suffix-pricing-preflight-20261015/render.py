"""Render the frozen pricing scope and remaining application gates."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent


def main():
    report = json.loads((ROOT / "report.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "suffix-pricing"})
    fig, ax = plt.subplots(figsize=(15, 8.4))
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 15), ylim=(0, 8.4))
    ax.axis("off")
    dark, muted, teal = "#20364d", "#596c81", "#087e80"
    ax.text(.3, 8.0, "Pricing identity before resident-window expansion", color=dark, size=23, weight="bold")
    ax.text(.3, 7.57, "Runtime schedule binding improved; automatic channel residency remains unqualified", color=muted, size=13)
    ax.text(.3, 6.95, "EXISTING ADMITTED COMPLETE-WINDOW FIXTURE", color=teal, size=12, weight="bold")

    def box(x, y, w, h, title, body, *, ready=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.04,rounding_size=.09",
                                   facecolor="#e5f3ef" if ready else "#eef2f7",
                                   edgecolor=teal if ready else "#c7d2df", linewidth=1.5))
        ax.text(x+w/2, y+h-.32, title, color=dark, size=13, weight="bold", ha="center", va="center")
        ax.text(x+w/2, y+.43, body, color=dark, size=11.5, ha="center", va="center", linespacing=1.5)

    box(.3, 5.15, 4.1, 1.5, "Existing source and runtime checks", "Same checked owner and numerical proofs\nSource, build, hardware and environment")
    box(5.35, 5.15, 4.1, 1.5, "Effective schedule identity", "omp_get_schedule: kind + modifier + chunk\nRequire native/resident identity agreement", ready=True)
    box(10.4, 5.15, 4.1, 1.5, "Select before owner execution", "Matching evidence: measured choice\nChanged runtime state: original native")
    for left, right in ((4.45, 5.3), (9.5, 10.35)):
        ax.add_patch(FancyArrowPatch((left, 5.9), (right, 5.9), arrowstyle="-|>", mutation_scale=16,
                                    linewidth=1.7, color=teal))
    ax.text(.3, 4.6, "No new application interval is admitted by this identity change.", color=muted, size=12)
    ax.text(.3, 4.0, "CHANNEL DIFFUSION → ACCUMULATION: INDEPENDENT OPEN GATES", color=dark, size=12, weight="bold")
    gates = [
        ("Discovery", "Stable formals; complete original teams"),
        ("Definitions and prefix", "Stage interiors; exact native effects"),
        ("Selected team", "Registered completion and effect proofs"),
        ("Diffusion corrections", "Indirect CPU writes; section coherence"),
        ("Complete pricing", "Branches, children, tiles and build identity"),
        ("Activation and consumer", "Reached selector; application calibration"),
    ]
    for index, (title, body) in enumerate(gates):
        col, row = index % 3, index // 3
        box(.3+col*4.75, 2.55-row*1.45, 4.4, 1.15, title, body)
    ax.text(.3, .58, "No application runtime, transfer saving or target qualification is claimed.", color=muted, size=12)
    ax.text(.3, .15, report["figure_footer"], color=muted, size=9)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    for ext in ("svg", "png", "pdf"):
        metadata = {"Date": None} if ext == "svg" else {"CreationDate": None, "ModDate": None} if ext == "pdf" else None
        fig.savefig(ROOT / f"gate-map.{ext}", dpi=140, metadata=metadata)
    svg = ROOT / "gate-map.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
