"""Render the architecture milestone; never run the compiler or applications.

Final evidence is supplied in report.json. Generation elapsed time is separate
from application performance; no application timings or transfer measurements
are present, and this map cannot imply qualification.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
INK = "#213047"
MUTED = "#526278"
EDGE = "#a5afbb"
COLORS = {
    "immutable": "#d2eee4",
    "plan": "#dceafb",
    "pending": "#fff0c4",
    "service": "#e6eaf0",
    "publication": "#edf4ec",
    "decline": "#f7d6cf",
}


def box(ax, x, y, width, height, title, body="", *, color="service", fontsize=11):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.007,rounding_size=0.009",
        facecolor=COLORS[color], edgecolor=EDGE, linewidth=.85))
    if body:
        title_fraction = .85 if body.count("\n") >= 2 else .73
        ax.text(x + width / 2, y + height * title_fraction, title, ha="center", va="center",
                fontsize=fontsize + .8, weight="bold", color=INK)
        ax.text(x + width / 2, y + height * .34, body, ha="center", va="center",
                fontsize=fontsize, color=INK, linespacing=1.25)
    else:
        ax.text(x + width / 2, y + height / 2, title, ha="center", va="center",
                fontsize=fontsize, weight="bold", color=INK, linespacing=1.25)


def arrow(ax, start, end, *, dashed=False, color=INK, width=1.2):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                               linewidth=width, color=color, linestyle="--" if dashed else "-",
                               shrinkA=4, shrinkB=5))


def footer(report):
    compiler = report.get("implementation_commit")
    pipeline = report.get("compatible_pipeline_commit")
    if compiler and pipeline:
        return f"Compiler {compiler[:12]} · compatible pipeline {pipeline[:12]} · source-bound evidence in README / report.json"
    return "DRAFT · final commit identities, validation receipts and capped application outcome are not yet bound"


def save(fig, name):
    fig.savefig(HERE / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    svg = HERE / f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(HERE / f"{name}.png", dpi=155, bbox_inches="tight", metadata={"Software": "matplotlib"})
    fig.savefig(HERE / f"{name}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None, "Creator": "matplotlib"})
    plt.close(fig)


def architecture_map(report):
    fig, ax = plt.subplots(figsize=(16, 11.5))
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(.015, .978, "One immutable source session, one checked execution plan",
            fontsize=24, weight="bold", color=INK, va="top")
    ax.text(.015, .917,
            "Architecture comparison only · original numerical order, proof requirements and fallback semantics remain required",
            fontsize=11.5, color=MUTED)
    ax.text(.025, .894, "BEFORE THIS SLICE", fontsize=13.5, weight="bold", color=INK)
    ax.text(.345, .894, "AFTER THIS SLICE", fontsize=13.5, weight="bold", color=INK)
    ax.plot([.314, .314], [.2, .875], color="#d4dae1", linewidth=1)

    box(ax, .025, .755, .265, .096, "Parser syntax already sealed",
        "Semantic namespaces and registries\nremain mutable", fontsize=10.7)
    box(ax, .025, .591, .265, .108, "Repeated source authentication",
        "Recheck files, scopes, bindings, kinds\nand namespace state during queries", color="pending", fontsize=10.5)
    box(ax, .025, .414, .265, .121, "Separate execution handoffs",
        "Discovery, placement and emission\nconsume mutable owner inventories", color="pending", fontsize=10.5)
    box(ax, .025, .247, .265, .111, "Coupled authority and work state",
        "Copies and speculative work must\npreserve original proof identities", fontsize=10.5)
    for top, bottom in [(.755, .699), (.591, .535), (.414, .358)]:
        arrow(ax, (.1575, top), (.1575, bottom))

    box(ax, .345, .786, .63, .065, "Authenticate inputs and parse their captured text",
        color="immutable", fontsize=12)
    ax.text(.66, .762, "Original / configured source + dependencies · uncaptured external INCLUDE blocks session sealing",
            ha="center", fontsize=9.8, color=MUTED)
    box(ax, .345, .621, .63, .108, "Immutable source session",
        "Original syntax, scopes, bindings, kinds, type schemas and registries\nQueries authenticate exact owned identities; private projections stay distinct",
        color="immutable", fontsize=11.2)
    arrow(ax, (.66, .754), (.66, .729))
    box(ax, .345, .449, .63, .117, "Checked ordered execution plan",
        "Operations, guards, mappings and definitions stay in admitted order\n"
        "execution_identity: complete-window calibration · identity: local proof tokens",
        color="plan", fontsize=10.7)
    arrow(ax, (.66, .621), (.66, .566))
    consumers = [(.345, "Placement", "Price this exact interval"),
                 (.565, "Emission", "Emit the admitted order"),
                 (.785, "Diagnostics", "Describe the same plan")]
    for x, title, body in consumers:
        box(ax, x, .303, .19, .086, title, body, color="plan", fontsize=10.5)
        arrow(ax, (.66, .449), (x + .095, .389))
    box(ax, .345, .19, .63, .066, "Publication: recheck live files, configuration and registered proofs",
        color="publication", fontsize=11.1)
    for x, *_ in consumers:
        arrow(ax, (x + .095, .303), (x + .095, .256), width=1)

    box(ax, .025, .052, .465, .091, "Mutable services remain separate",
        "Proof caches, generation builders, transactions and emission results\nCopies detach work state and rebind builder references",
        fontsize=10.8)
    box(ax, .515, .052, .46, .091, "Borrowed authorities retain their own validation",
        "Registered preparation, extraction, completion and source proofs\nA frozen copy does not gain original source authority",
        color="immutable", fontsize=10.6)
    ax.text(.015, .018, "No application speedup, transfer reduction or automatic qualification is claimed by this map.",
            fontsize=11, weight="bold", color=INK)
    ax.text(.015, -.032, footer(report), fontsize=9.2, color=MUTED)
    save(fig, "architecture-map")


def qualification_gates(report):
    fig, ax = plt.subplots(figsize=(16, 7.1))
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    checkpoint = report.get("application_checkpoint") or {}
    failed = checkpoint.get("status") == "timeout"
    ax.text(.015, .98, "Application generation remains blocked" if failed else
            "Architecture progress still needs an application checkpoint",
            fontsize=23, weight="bold", color=INK, va="top")
    ax.text(.015, .865, "Both channel cases remain primary · automatic placement and residency are the deliverable",
            fontsize=11.7, color=MUTED)
    validation = report.get("validation") or {}
    validation_status = validation.get("visual_status", "Validation binding pending")
    box(ax, .025, .563, .25, .245, "Compiler capability slice",
        "Immutable source session\nChecked execution plan\nShared placement / emission inventory\n\n" + validation_status,
        color="plan", fontsize=11)
    box(ax, .355, .563, .29, .245, "Integrated application checkpoint",
        ("300-second limit reached\nNo generated scope manifest\nNo application build or execution\nPricing and traffic remain unmeasured" if failed else
         "Capped generation + executable window\nNumerical fields, bounds and halos\nComplete transfer ledger\nCompatible automatic cost evidence"),
        color="decline" if failed else "pending", fontsize=11)
    box(ax, .725, .563, .25, .245, "Complete-process qualification",
        "Fresh native + SGS-direct controls\nAutomatic-resident candidate\n4 threads · unchanged workloads\n1 warmup + 7 rotated measured runs",
        color="pending", fontsize=11)
    arrow(ax, (.275, .686), (.355, .686))
    arrow(ax, (.645, .686), (.725, .686), dashed=failed, color=MUTED if failed else INK)
    ax.text(.15, .519, validation.get("visual_note", "Source / test receipts awaiting final binding"), ha="center", va="top", fontsize=10, color=MUTED)
    ax.text(.50, .519, checkpoint.get("visual_note", "Generation result pending coordinating task"), ha="center", va="top", fontsize=10, color=MUTED)
    ax.text(.85, .519, "No qualifying samples in this report", ha="center", va="top", fontsize=10, color=MUTED)

    box(ax, .025, .319, .465, .124, "channel-simple",
        "Complete wall-time median < 9 s\nUnchanged configuration · natural 109-step completion", color="service", fontsize=11.2)
    box(ax, .515, .319, .46, .124, "channel",
        "Median ≤ 74.4% of fresh native; faster than SGS direct\nFull duration · actual fourth-order advection and filtering", color="service", fontsize=10.8)
    ax.text(.025, .248, "Comfortably met: every candidate run meets its case target and beats SGS direct in the corresponding round.",
            fontsize=11.2, weight="bold", color=INK)
    ax.text(.025, .185, "Missing or incompatible automatic cost estimates select native. That fallback does not complete the automation objective.",
            fontsize=11.2, color=INK)
    ax.text(.025, .135, "No application performance or transfer measurements were produced at this checkpoint.",
            fontsize=11.2, color=INK)
    ax.text(.025, .082, "Case configurations and grids are unchanged; the diagnostic source includes six audited INTENT repairs.",
            fontsize=10.8, color=MUTED)
    ax.text(.025, .03, "Pressure ownership, cross-stage residency and mathematical reordering remain separate future work.",
            fontsize=10.8, color=MUTED)
    ax.text(.015, -.032, footer(report), fontsize=9.1, color=MUTED)
    save(fig, "qualification-gates")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "svg.hashsalt": "checked-execution-plan-20261014"})
    report = json.loads((HERE / "report.json").read_text())
    architecture_map(report)
    qualification_gates(report)


if __name__ == "__main__":
    main()
