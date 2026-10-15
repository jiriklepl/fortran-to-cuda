"""Render the generated diffusion ownership map; never run an application.

The two source-bound manifests are frozen inputs. Arrows describe conditional
coherence obligations, not observed copies or performance measurements.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
CHECKPOINT = HERE / "evidence"
MANIFESTS = {
    "diffusion": ("artifacts/scope-manifest.json", "cd0f856718bbaf6b6ffaf3c1f6ac182326db13519009c3dca8fec3fbb4f2a5eb"),
    "sgs": ("sgs-control/artifacts/scope-manifest.json", "e0ba7b0ba83a90ad807699309bea228d692ec489d780665222d02b2f09184943"),
}
COMPILER = "744a06a912da56f116831868c554b5e82f9ae4c1"
PIPELINE = "bbe44c1151ac90e0122df91022cceff9d4a48428"
INK, MUTED, BLUE, ORANGE, GREEN = "#20364d", "#596c81", "#1677a8", "#bc641b", "#087e80"
PALETTE = {
    "native": ("#eef2f7", "#bcc8d6"),
    "gpu": ("#e1f0fa", BLUE),
    "host": ("#e8f4ee", GREEN),
    "publish": ("#fff0dc", ORANGE),
}


def evidence():
    manifests = {}
    for name, (relative, expected) in MANIFESTS.items():
        path = CHECKPOINT / relative
        data = gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes())
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Frozen {name} manifest differs: {path}")
        manifests[name] = json.loads(data)
    diffusion, sgs = manifests["diffusion"], manifests["sgs"]
    assert diffusion["source_inputs"] == sgs["source_inputs"]
    owner, = diffusion["scopes"]
    assert owner["mode"] == "sections" and not owner["boundaries"]
    assert len(owner["gpu_leaves"]) == 3
    team_segments = [segment for segment in owner["planning_segments"]
                     if "joined_team" in segment["operations"]]
    segment, = team_segments
    corrections = [operation for operation in segment["operations"]["native_operations"]
                   if operation.get("indirect_inspector", {}).get("available")]
    assert len(corrections) == 3
    assert all(operation["completion"]["worksharing_units"] == 12 for operation in corrections)
    assert not diffusion["automatic_estimate_available"]
    request = json.loads((CHECKPOINT / "request.json").read_text())
    assert request["commits"] == {"compiler": COMPILER, "pipeline": PIPELINE}


def box(ax, x, y, width, height, title, body="", *, kind="native", size=12):
    face, edge = PALETTE[kind]
    ax.add_patch(FancyBboxPatch((x, y), width, height,
        boxstyle="round,pad=0.025,rounding_size=0.09", facecolor=face,
        edgecolor=edge, linewidth=1.1, zorder=2))
    ax.text(x + width / 2, y + height * (.67 if body else .5), title,
            ha="center", va="center", color=INK, fontsize=size, weight="bold", zorder=3)
    if body:
        ax.text(x + width / 2, y + height * .28, body, ha="center", va="center",
                color=INK, fontsize=size - 1, linespacing=1.3, zorder=3)


def arrow(ax, start, end, *, color=INK, dashed=False, width=1.5, zorder=3):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
        linewidth=width, color=color, linestyle=(0, (4, 3)) if dashed else "-",
        shrinkA=3, shrinkB=4, zorder=zorder))


def main():
    evidence()
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "svg.hashsalt": "diffusion-application-20261015"})
    fig, ax = plt.subplots(figsize=(18, 13.2))
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 18), ylim=(0, 13.2))
    ax.axis("off")
    ax.text(.45, 12.7, "Generated diffusion window: GPU kernels with native wall corrections",
            fontsize=23, weight="bold", color=INK)
    ax.text(.45, 12.25, "Diagnostic sections policy • channel-simple validated; channel candidate untested • automatic pricing unavailable",
            fontsize=12.5, color=MUTED)

    box(ax, .5, 10.75, 17, 1.0,
        "Both paths: SGS direct → native wall / viscosity work → stage initialization → advection, filtering and forces",
        "Original numerical order and wall model retained; channel's active filtering stays in the native prefix.", size=13)
    ax.text(.6, 10.36, "Separate ownership: SGS publishes before this native prefix. The new window starts inside the selected diffusion call.",
            fontsize=11.5, color=MUTED)

    for x, width, fill in ((.5, 4.55, "#fafbfd"), (5.4, 12.1, "#f7fbfa")):
        ax.add_patch(FancyBboxPatch((x, 2.3), width, 7.65,
            boxstyle="round,pad=.025,rounding_size=.12", facecolor=fill,
            edgecolor="#c7d2df", linewidth=1.0, zorder=0))
    ax.text(.8, 9.55, "SGS-DIRECT CONTROL", fontsize=14, weight="bold", color=INK)
    ax.text(.8, 9.15, "Diffusion remains native", fontsize=12, color=MUTED)
    ax.text(5.7, 9.55, "SGS DIRECT + GENERATED DIFFUSION", fontsize=14, weight="bold", color=GREEN)
    ax.text(5.7, 9.15, "One diffusion owner; one original OpenMP team; no owner boundary", fontsize=11.5, color=MUTED)
    arrow(ax, (2.75, 10.7), (2.75, 10.0))
    arrow(ax, (11.5, 10.7), (11.5, 10.0))

    box(ax, 5.75, 8.02, 5.0, .78, "Device: U / V / W + viscosity",
        "Shared read-only fields across all three kernels", kind="gpu", size=12)
    box(ax, 11.8, 8.02, 5.3, .78, "Host: U / V / W + viscosity",
        "Current mirrors retained for native readers", kind="host", size=12)
    ax.text(8.25, 7.66, "Entry: upload missing input sections, including stars", ha="center",
            fontsize=10.3, color=BLUE)
    ax.text(14.45, 7.66, "Read-only host mirrors need no download", ha="center",
            fontsize=10.3, color=GREEN)

    rows = (6.4, 4.9, 3.4)
    for index, (component, y) in enumerate(zip("UVW", rows, strict=True)):
        box(ax, .85, y, 3.85, 1.0, f"CPU {component} diffusion",
            "then 12 native wall corrections", size=12.2)
        box(ax, 6.65, y, 3.65, 1.0, f"{index * 2 + 1}  GPU {component} diffusion",
            f"Read {component}, viscosity and {component}*; update {component}*", kind="gpu", size=12)
        box(ax, 12.65, y, 4.45, 1.0, f"{index * 2 + 2}  CPU wall corrections",
            f"12 original worksharing loops update {component}*", size=12)
        arrow(ax, (10.32, y + .72), (12.62, y + .72))
        arrow(ax, (10.32, y + .23), (12.62, y + .23), color=ORANGE, dashed=True)
        ax.text(11.48, y + .4, f"Publish dirty {component}* sections", fontsize=9.4,
                ha="center", va="center", color=ORANGE)
        if index < 2:
            arrow(ax, (2.78, y), (2.78, rows[index + 1] + 1))
            arrow(ax, (14.8, y), (8.5, rows[index + 1] + 1), width=1.2)

    # The rails show reuse of the same registrations, not additional copies.
    for x, center, color in ((6.2, 8.2, BLUE), (17.3, 14.45, GREEN)):
        ax.plot([center, x, x], [8.02, 7.7, 3.9], color=color, linewidth=1.2, zorder=1)
    for y in rows:
        arrow(ax, (6.2, y + .5), (6.63, y + .5), color=BLUE, width=1.2)
        arrow(ax, (17.3, y + .5), (17.1, y + .5), color=GREEN, width=1.2)

    ax.text(.85, 2.86, "Original completion and barriers retained", fontsize=10.6, color=MUTED)
    ax.text(.85, 2.52, "No diffusion PCIe traffic in this control", fontsize=10.6, color=MUTED)
    box(ax, 5.8, 2.5, 11.25, .56,
        "Close: publish remaining dirty stars; preserve sections already corrected on the host",
        kind="publish", size=11.4)
    arrow(ax, (14.85, 3.4), (14.85, 3.07), color=ORANGE, dashed=True)

    box(ax, .5, 1.02, 17, .83, "Native accumulation and the remaining RK step stay outside the window",
        "Stars remain live after return. This checkpoint does not retain ownership through accumulation or across stages.", size=13)
    arrow(ax, (2.75, 2.28), (2.75, 1.87))
    arrow(ax, (11.4, 2.28), (11.4, 1.87))

    arrow(ax, (.6, .62), (1.2, .62))
    ax.text(1.3, .62, "Original operation order", va="center", fontsize=10.5, color=MUTED)
    arrow(ax, (5.0, .62), (5.6, .62), color=ORANGE, dashed=True)
    ax.text(5.7, .62, "Publication obligations; observed totals in transfer plot", va="center", fontsize=10.5, color=MUTED)
    ax.text(15.3, .62, "U* = Ustar", va="center", fontsize=10.5, color=MUTED)
    ax.text(.5, .16, f"Frozen manifests • compiler {COMPILER[:12]} / pipeline {PIPELINE[:12]} • no application speedup or target qualification claimed",
            fontsize=10, color=MUTED)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    for extension in ("svg", "png", "pdf"):
        metadata = ({"Date": None} if extension == "svg" else
                    {"CreationDate": None, "ModDate": None} if extension == "pdf" else None)
        fig.savefig(HERE / f"operation-map.{extension}", dpi=150, metadata=metadata)
    svg = HERE / "operation-map.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
