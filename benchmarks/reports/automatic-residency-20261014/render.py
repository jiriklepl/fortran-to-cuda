"""Render this frozen source-preflight milestone without rerunning analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
COLORS = {"prepared": "#d2eee4", "unavailable": "#f7d6cf", "native": "#e6eaf0",
          "uninspected": "#fff0c4", "bypass": "#f4f5f7"}
INK = "#213047"


def box(ax, x, y, width, height, text, status, fontsize=10):
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.009,rounding_size=0.012",
                          linewidth=0.8, edgecolor="#a5afbb", facecolor=COLORS[status])
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", color=INK,
            fontsize=fontsize, linespacing=1.4)


def save(fig, name):
    vector = HERE / (name + ".svg")
    fig.savefig(vector, bbox_inches="tight", metadata={"Date": None})
    vector.write_text("\n".join(line.rstrip() for line in vector.read_text().splitlines()) + "\n")
    fig.savefig(HERE / (name + ".png"), dpi=140, bbox_inches="tight", metadata={"Software": "matplotlib"})
    plt.close(fig)


def scope_map():
    fig, ax = plt.subplots(figsize=(14, 17))
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0, 1.01, "The two primary momentum paths", fontsize=22, weight="bold", color=INK)
    ax.text(0, .984, "Source order and observed admission gaps · no resident owner admitted by this report",
            fontsize=11, color=INK)
    ax.text(.49, .945, "channel-simple", fontsize=15, weight="bold", ha="center", color=INK)
    ax.text(.82, .945, "channel", fontsize=15, weight="bold", ha="center", color=INK)
    ax.text(.49, .927, "advection 2 · filter off · wall model 1", fontsize=10, ha="center", color=INK)
    ax.text(.82, .927, "advection 4 · filter on · wall model 1", fontsize=10, ha="center", color=INK)
    rows = [
        ("1  Wall properties", "U/V/W, Pr, scalars → wall metadata",
         "ComputeViscsWM\noriginal native operation", "native", None, None, .049),
        ("2  SGS producer", "U/V/W → Viscosity",
         "Sigma, order-2 gradient\nnumerical group prepares; costs unavailable", "prepared", None, None, .057),
        ("3  Wall fluxes / halos", "U/V/W, Viscosity, wall metadata",
         "ComputeUVWFluxesWM → BoundViscosity\n→ immersed-boundary viscosity corrections", "native", None, None, .060),
        ("4  Stage initialization", "previous Ustar → U2, or zero fields",
         "Original RK_stage branch\ncomplete groups prepare; costs unavailable", "prepared", None, None, .055),
        ("5  Advection", "U/V/W → Ustar/Vstar/Wstar",
         "CDU → CDV → CDW\nleaf coverage still required", "uninspected",
         "CD4divU → CD4divV → CD4divW\nU leaf: dynamic scratch / multiply gaps", "unavailable", .061),
        ("6  Filter / halo updates", "star fields and Utype/Vtype/Wtype",
         "FilterUstar branch bypassed\nfiltertype = 0", "bypass",
         "BoundUVW → Filter each star field\n→ BoundUVW; Filter3ord scratch gap", "unavailable", .063),
        ("7  Source-ordered forces", "star fields; U/V/W; optional scalars",
         "Coriolis → buoyancy → resistance → linear\n→ Rayleigh damping → StressBoundaryFlux\nkeep original predicates and effects", "native", None, None, .075),
        ("8  Explicit diffusion", "Viscosity + U/V/W → star updates",
         "Second-order diffusion in BOTH cases\nU stencil → native U wall corrections\nV stencil → native V wall corrections\nW stencil → native W wall corrections\nwhole team: indirect component / effect gaps", "unavailable", None, None, .111),
        ("9  Accumulation", "U2/V2/W2 += beta(stage) × dt × star",
         "Original arithmetic order\ncomplete group prepares; costs unavailable", "prepared", None, None, .057),
        ("10  Next consumers", "publish stale host sections when needed",
         "ScalarRK3 → OtherTerms → boundaries\npressure correction remains native", "native", None, None, .060),
    ]
    top = .900
    for label, resource, left, status, right, right_status, height in rows:
        bottom = top - height
        ax.text(.0, top - .009, label, fontsize=11, weight="bold", color=INK, va="top")
        ax.text(.0, top - .027, resource, fontsize=8.8, color="#516075", va="top", wrap=True)
        if right is None:
            box(ax, .34, bottom, .64, height - .006, left, status, fontsize=10.2)
        else:
            box(ax, .34, bottom, .31, height - .006, left, status, fontsize=9.2)
            box(ax, .67, bottom, .31, height - .006, right, right_status, fontsize=9.2)
        top = bottom - .012
    ax.text(0, top - .014,
            "Diffusion formal U2/V2/W2 are mapped to the caller's Ustar/Vstar/Wstar; accumulation updates distinct U2/V2/W2.",
            fontsize=9.2, color=INK)
    ax.text(0, top - .037,
            "Any unsupported intervening consumer may require earlier publication. The map does not claim that native effects are proved.",
            fontsize=9.2, color=INK)
    legend = [("prepared", "Numerical preparation available; execution/cost proof remains"),
              ("unavailable", "Observed whole-source analysis limitation"),
              ("native", "Original native work or uninspected effects"),
              ("uninspected", "Leaf coverage remains incomplete")]
    for index, (status, label) in enumerate(legend):
        x, y = (0 if index % 2 == 0 else .51), top - .077 - .027 * (index // 2)
        box(ax, x, y, .025, .012, "", status)
        ax.text(x + .037, y + .006, label, fontsize=8.7, color=INK, va="center")
    save(fig, "scope-map")


def coverage(evidence):
    fig, ax = plt.subplots(figsize=(14, 7.0))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.text(0, .99, "Bounded diagnostics expose several prerequisites at once", fontsize=20, weight="bold", color=INK)
    ax.text(0, .935, "Six completed source-only snapshots · no captures asserted · no calibration supplied · no application execution",
            fontsize=10.5, color=INK)
    headings = [(0, "Entry"), (.43, "Operations"), (.55, "Prepared groups"), (.72, "Capture notices"), (.88, "Other notices")]
    for x, heading in headings:
        ax.text(x, .85, heading, fontsize=10, weight="bold", color=INK)
    names = {"dynamics::convection": "Convection (both cases)",
             "subgrid::sgs_sigma_uniform": "SGS sigma (both cases)",
             "momentumdiffusion::momentumdiffusion_nobranch_2ord": "Second-order diffusion (both)",
             "filters::filter": "Filter dispatcher (channel)",
             "momentumadvection::cd4divu": "Fourth-order advection U (channel)",
             "filters::filter3ord": "Filter3ord (channel)"}
    for index, snapshot in enumerate(evidence["snapshots"]):
        y = .76 - index * .095
        prepared = sum(row["numerical"]["status"] == "available" for row in snapshot["operations"])
        candidates = sum(row["kind"] in {"loop", "joined_team"} for row in snapshot["operations"])
        captures = snapshot["blockers_by_category"].get("capture", 0)
        ax.axhline(y - .031, color="#e5e9ee", linewidth=1)
        ax.text(0, y, names[snapshot["entry"]], fontsize=11, color=INK, va="center")
        for x, text in [(.46, snapshot["operation_count"]), (.61, f"{prepared} / {candidates}"),
                        (.78, captures), (.93, snapshot["blocker_count"] - captures)]:
            ax.text(x, y, str(text), fontsize=12, ha="center", va="center", color=INK)
    ax.text(0, .10, "Counts are diagnostic occurrences, not distinct defects. Both sides of runtime branches are inspected.", fontsize=10, color=INK)
    ax.text(0, .055, "Whole-team rejection does not disprove existing per-worksharing-unit support. A missing capture assertion is not a compiler defect.",
            fontsize=10, color=INK)
    ax.text(0, .01, "No measured transfers or performance improvements are represented here. Both application targets remain unqualified.", fontsize=10, color=INK)
    save(fig, "coverage")


def main():
    evidence = json.loads((HERE / "evidence.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "svg.hashsalt": "automatic-residency-20261014"})
    scope_map()
    coverage(evidence)


if __name__ == "__main__":
    main()
