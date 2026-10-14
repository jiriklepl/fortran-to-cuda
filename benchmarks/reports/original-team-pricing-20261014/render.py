"""Deterministic capability map; no performance data are plotted."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main():
    output = Path(__file__).resolve().parent
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "original-teams-v1"})
    figure, axis = plt.subplots(figsize=(14, 9), dpi=160)
    figure.patch.set_facecolor("#f6f8fb")
    axis.set(xlim=(0, 14), ylim=(0, 9))
    axis.axis("off")
    ink, muted = "#172b4d", "#52647a"

    def label(x, y, value, size=12, color=ink, weight="normal"):
        axis.text(x, y, value, fontsize=size, color=color, weight=weight, va="top", linespacing=1.5)

    def box(x, y, width, height, fill, edge="#d7dfe8"):
        axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.10",
                                    linewidth=1, edgecolor=edge, facecolor=fill))

    label(0.35, 8.8, "Original teams + native PRIVATE storage", 23, weight="bold")
    label(0.35, 8.28, "Compiler capability checkpoint  ·  2026-10-14  ·  No application performance claim", 12, muted)
    label(0.5, 7.66, "BEFORE  ·  62a1d02", 14, muted, "bold")
    label(7.55, 7.66, "AFTER  ·  9fe23c7", 14, "#166d54", "bold")
    rows = [
        (6.05, "Whole-window cost evidence", "Serial, branchless complete owner\nOriginal OpenMP teams rejected",
         "Whole-window cost evidence", "Proved complete original joined teams\nStartup, worksharing and joins included"),
        (4.75, "Native PRIVATE arrays", "Small fixed shapes\nRuntime extents rejected",
         "Native PRIVATE arrays", "Original automatic explicit-shape storage\nEntry-time bounds retained on the CPU"),
        (3.45, "Exact cost key", "Formal array descriptors\nInteger input controls",
         "Exact cost key", "+ PRIVATE sizes and lower bounds\nOnce per array, including unused storage"),
    ]
    for y, before_title, before, after_title, after in rows:
        box(0.4, y, 5.7, 1.13, "#ffffff")
        box(7.45, y, 6.15, 1.13, "#eaf6f0", "#bedccd")
        label(0.65, y + 0.95, before_title, 12, weight="bold")
        label(0.65, y + 0.62, before, 11, muted)
        label(7.7, y + 0.95, after_title, 12, weight="bold")
        label(7.7, y + 0.62, after, 11)
        axis.add_patch(FancyArrowPatch((6.35, y + 0.56), (7.15, y + 0.56), arrowstyle="-|>",
                                      mutation_scale=17, linewidth=1.6, color="#71859a"))
    box(0.4, 2.47, 13.2, 0.72, "#eaf0f7")
    label(0.65, 3.03, "Preserved: numerical and effect proofs · original order and joins · exact coherence · guards · no replay", 11)
    label(0.65, 2.72, "Integrated validation: 7 CUDA/native/source-probe tests passed; all 531 compiler source hashes stayed unchanged.", 10, muted)
    box(0.4, 0.55, 13.2, 1.65, "#fff5e4", "#ead5ad")
    label(0.65, 2.01, "APPLICATION BOUNDARIES REMAIN", 12, "#805516", "bold")
    label(0.65, 1.63, "Exact pricing still excludes reached suffixes, child owners, calls, branches and module/local numerical storage.", 11)
    label(0.65, 1.28, "The independent calibration producer is the supported build consumer; application build receipts are insufficient.", 11)
    label(0.65, 0.93, "channel-simple and channel: no resident application generation or target qualification established here.", 11, weight="bold")
    label(0.4, 0.28, "Synthetic profitable rows in guard tests are not measured calibration or performance evidence.", 10, muted)
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    figure.savefig(output / "capabilities.svg", metadata={"Date": None})
    figure.savefig(output / "capabilities.png", metadata={"Software": "original-team-pricing-report-v1"})
    plt.close(figure)


if __name__ == "__main__":
    main()
