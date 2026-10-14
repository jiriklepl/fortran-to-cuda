"""Render frozen generation diagnostics; never run application or compiler analysis."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
INK = "#213047"
COLORS = {"checked": "#d2eee4", "scan": "#fff0c4", "blocked": "#f7d6cf", "context": "#e6eaf0"}


def box(ax, x, y, width, height, label, color="context", fontsize=11):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                               boxstyle="round,pad=0.008,rounding_size=0.012",
                               facecolor=COLORS[color], edgecolor="#a5afbb", linewidth=.8))
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center",
            fontsize=fontsize, color=INK, linespacing=1.4)


def save(fig, name):
    path = HERE / (name + ".svg")
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    fig.savefig(HERE / (name + ".png"), dpi=145, bbox_inches="tight", metadata={"Software": "matplotlib"})
    plt.close(fig)


def authentication_map(report):
    fig, ax = plt.subplots(figsize=(14, 9.4))
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    ax.text(0, .99, "Source ownership replaces repeated syntax scans", fontsize=23, weight="bold", color=INK)
    ax.text(0, .946, "Same authenticated Convection request · original application syntax preserved · no application qualification",
            fontsize=11, color=INK)
    rows = [
        (.735, "Baseline", "5d752da · measured", [("Source files /\nconfiguration", "checked"),
            ("Scan original\nitem / span provenance", "scan"), ("Compare original\nsyntax trees", "scan"),
            ("Check namespaces /\nbindings / proof", "checked")]),
        (.475, "Sealed prototype", "e335714 + frozen overlay · measured", [("Source files /\nconfiguration", "checked"),
            ("Still scan original\nitem / span provenance", "scan"), ("Original syntax\nidentity checks", "checked"),
            ("Check namespaces /\nbindings / proof", "checked")]),
        (.215, "Session refinement", "Implementation binding pending" if not report['implementation_commit'] else report['implementation_commit'][:12] + " · committed implementation",
            [("Source files /\nconfiguration", "checked"), ("Sealed item / span\nprovenance", "checked"),
             ("Original syntax\nidentity checks", "checked"), ("Check namespaces /\nbindings / proof", "checked")]),
    ]
    for y, title, note, nodes in rows:
        ax.text(0, y + .139, title, fontsize=16, weight="bold", color=INK)
        ax.text(.33, y + .139, note, fontsize=10.5, color=INK)
        for i, (label, color) in enumerate(nodes):
            x = i * .255
            box(ax, x, y, .225, .097, label, color)
            if i < 3:
                ax.annotate("", xy=(x + .25, y + .048), xytext=(x + .23, y + .048),
                            arrowprops={"arrowstyle": "->", "color": INK, "linewidth": 1.2})
    ax.text(0, .695, "Each demand traverses mutable original syntax and source provenance.", fontsize=10.5, color=INK)
    ax.text(0, .435, "Seal original AST + item syntax once; private copies thaw. Redundant provenance traversal remains.", fontsize=10.5, color=INK)
    ax.text(0, .175, "Remove duplicate full scan only for a verified sealed session; keep mutable standalone authentication.", fontsize=10.5, color=INK)
    ax.text(0, .09, "Unchanged proof boundary", fontsize=12, weight="bold", color=INK)
    ax.text(0, .055, "Fresh file/configuration checks, namespace and binding checks, invalidation, and source identity stay required.", fontsize=11, color=INK)
    ax.text(0, .012, "The refined path preserves source authority. Its separate bounded application checkpoint is reported alongside the prototype evidence.", fontsize=10.5, color=INK)
    save(fig, "authentication-map")


def bounded_outcomes(evidence):
    fig, ax = plt.subplots(figsize=(14, 7.9))
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    ax.text(0, .99, "Bounded Convection generation: no application qualification", fontsize=23, weight="bold", color=INK)
    ax.text(0, .94, "300-second generation cap unchanged · separate 180-second profiles of baseline / prototype · no application or GPU execution",
            fontsize=10.7, color=INK)
    report = json.loads((HERE / "report.json").read_text())
    final = report.get("attempts", {}).get("final", {})
    box(ax, 0, .735, .31, .145, "BASELINE  5d752da\n300 s cap reached · exit 124\nNo final manifest", "blocked", 11)
    box(ax, .34, .735, .31, .145, "PROTOTYPE  e335714 + overlay\n300 s cap reached · exit 124\nNo final manifest", "blocked", 11)
    final_label = "FINAL  62a1d02\n" + final.get("visual_outcome", "Bounded checkpoint pending")
    box(ax, .68, .735, .31, .145, final_label, "checked" if final.get("manifest_emitted") else "blocked", 11)
    ax.text(0, .65, "What the separate incomplete profiles show", fontsize=16, weight="bold", color=INK)
    ax.text(0, .61, "Cumulative profiler times overlap and include instrumentation. Attempts reached different work; CPU contention was present.",
            fontsize=10.5, color=INK)
    def select(label, function, line=None):
        return next(x for x in evidence[label]["profile_selected"]
                    if x["function"] == function and (line is None or x["line"] == line))
    baseline = select("baseline", "_require_original_global")
    prototype = select("sealed-prototype", "_require_original_global")
    provenance = select("sealed-prototype", "<genexpr>", 2013)
    box(ax, 0, .365, .475, .185,
        "Baseline authentication\n" + f"{baseline['calls']:,} global checks · {baseline['cumulative_seconds']:.2f} s cumulative\n" +
        "Repeated syntax-tree comparison is prominent.", "scan", 11.5)
    box(ax, .515, .365, .475, .185,
        "Prototype authentication\n" + f"{prototype['calls']:,} global checks · {prototype['cumulative_seconds']:.2f} s cumulative\n" +
        f"{provenance['calls']:,} provenance visits · {provenance['cumulative_seconds']:.2f} s",
        "scan", 11.5)
    ax.text(0, .30, "These are diagnostic hotspots, not a before/after speedup benchmark.", fontsize=13, weight="bold", color=INK)
    ax.text(0, .245, "The profile supports removing a duplicate provenance scan after immutable-session verification.", fontsize=11.5, color=INK)
    ax.text(0, .195, "It does not establish cheaper complete planning, admitted resident execution, transfer savings, or either wall-time target.", fontsize=11, color=INK)
    ax.text(0, .115, "Next decision: reassess the complete blocker map before adding capabilities; do not raise analysis budgets.", fontsize=11, color=INK)
    ax.text(0, .06, "Prototype predates Comment-cloning / serialization-alias repairs and includes the owning-placement change after baseline.", fontsize=10, color=INK)
    ax.text(0, .013, "Raw pstats, stack samples, source hashes, capture facts, request options, and corrected profile attribution are frozen beside this chart.", fontsize=10, color=INK)
    save(fig, "bounded-outcomes")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "source-session-20261014"})
    report = json.loads((HERE / "report.json").read_text())
    evidence = json.loads((HERE / "evidence/profile-evidence.json").read_text())
    authentication_map(report)
    bounded_outcomes(evidence)


if __name__ == "__main__":
    main()
