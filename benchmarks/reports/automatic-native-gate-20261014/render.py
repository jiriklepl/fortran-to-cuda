"""Render the native cost-gate behavior without implying application speedup."""

import json
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
INK = "#203247"
MUTED = "#526478"
TEAL = "#157b76"
GREY = "#edf1f5"
GREEN = "#e6f4ef"
BLUE = "#edf4fc"


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "svg.hashsalt": "automatic-native-gate-20261014"})
    fig, ax = plt.subplots(figsize=(12, 7.1))
    fig.subplots_adjust(left=0.025, right=0.975, top=0.97, bottom=0.035)
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 12), ylim=(0, 7.1))
    ax.axis("off")

    def text(x, y, value, size=11, *, color=INK, weight="normal", align="left"):
        return ax.text(x, y, value, ha=align, va="center", fontsize=size,
                       color=color, weight=weight, linespacing=1.45)

    def box(x, y, width, height, title, body, fill=GREY, edge="#cad4df"):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
            boxstyle="round,pad=0.035,rounding_size=0.09", linewidth=1.1,
            facecolor=fill, edgecolor=edge))
        text(x + width / 2, y + height - 0.29, title, 10.8, weight="bold", align="center")
        text(x + width / 2, y + (height - 0.37) / 2, body, 9.5, align="center")

    def arrow(points, color=MUTED):
        for start, end in pairwise(points[:-1]):
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=1.4)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                    mutation_scale=13, linewidth=1.4, color=color))

    text(0.08, 6.86, "Early native selection when cost evidence is absent", 19, weight="bold")
    text(0.08, 6.48, "Fallback behavior only • no automatic GPU or application performance claim", 10.8,
         color=MUTED)

    text(0.08, 5.95, "BEFORE", 11, weight="bold", color=MUTED)
    box(0.10, 4.75, 2.45, 0.98, "Authenticate inputs", "Source · config · entry\nCapture / caller facts")
    box(3.02, 4.75, 2.65, 0.98, "Discover + lower", "Analyze candidate owners\nbefore rejecting missing costs")
    box(6.15, 4.75, 2.65, 0.98, "Native chosen late", "No usable estimate\nfor automatic placement")
    box(9.28, 4.75, 2.55, 0.98, "Authenticate publication", "Recheck live inputs\nthen publish manifest", BLUE)
    for start, end in ((2.55, 3.02), (5.67, 6.15), (8.8, 9.28)):
        arrow([(start, 5.24), (end, 5.24)])

    ax.plot([0.08, 11.9], [4.40, 4.40], color="#dce3ea", linewidth=1)
    text(0.08, 4.08, "NOW", 11, weight="bold", color=TEAL)
    box(0.10, 2.79, 2.45, 1.02, "Authenticate inputs", "Same constructor checks\nincluding collective callers", BLUE)
    box(3.02, 2.66, 2.65, 1.28, "Early cost gate", "AUTO policy and both absent:\nhardware profile +\nindependent exact-window request", GREEN, TEAL)
    box(6.15, 3.13, 2.65, 1.06, "Unchanged native source", "0 scopes · 0 source edits\nNo runtime or build units", GREEN, TEAL)
    box(6.15, 1.73, 2.65, 1.06, "Existing compiler path", "Discovery → legality → pricing\nExisting GPU / native decision", BLUE)
    box(9.28, 2.43, 2.55, 1.35, "Authenticate publication", "Recheck original / configured\nand normalized inputs\nthen publish manifest", BLUE)
    arrow([(2.55, 3.30), (3.02, 3.30)], TEAL)
    arrow([(5.67, 3.66), (6.15, 3.66)], TEAL)
    text(5.91, 3.85, "yes", 9, color=TEAL, align="center")
    arrow([(4.34, 2.66), (4.34, 2.26), (6.15, 2.26)])
    text(4.54, 2.43, "no", 9, color=MUTED)
    arrow([(8.8, 3.66), (9.28, 3.36)], TEAL)
    arrow([(8.8, 2.26), (9.03, 2.26), (9.28, 2.75)])

    ax.add_patch(FancyBboxPatch((0.10, 0.62), 11.73, 0.77,
        boxstyle="round,pad=0.035,rounding_size=0.08", linewidth=0,
        facecolor="#f1f5f8"))
    text(0.32, 1.13, "Collective lowering now uses captured text for ordinary and normalized inputs.",
         10.8, weight="bold")
    text(0.32, 0.82, "Transient file edits cannot enter lowering; persistent changes still block publication.",
         10.2, color=MUTED)

    metadata = json.loads((HERE / "report.json").read_text()) if (HERE / "report.json").exists() else {}
    footer = metadata.get("figure_footer", "DRAFT • source and validation binding pending in report.json")
    text(0.10, 0.22, footer, 8.5, color=MUTED)
    output = HERE / "native-selection-map"
    fig.savefig(output.with_suffix(".svg"), metadata={"Date": None})
    svg = output.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(output.with_suffix(".png"), dpi=160, metadata={"Software": "matplotlib"})
    fig.savefig(output.with_suffix(".pdf"),
                metadata={"CreationDate": None, "ModDate": None, "Creator": "matplotlib"})
    plt.close(fig)


if __name__ == "__main__":
    main()
