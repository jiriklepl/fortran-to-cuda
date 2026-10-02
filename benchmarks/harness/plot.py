#!/usr/bin/env python3
"""
Reads results/baseline.csv and writes one figure per unique (grid, iters) combination.
Output: results/figures/grid_{gx}x{gy}x{gz}_niter{niter}.{png,pdf}

Message the figure conveys:
  "Memory transfer dominates CUDA cost; the kernel itself is blazing fast."
"""

import argparse
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.legend_handler import HandlerPatch

from .paths import RESULTS

# ── LaTeX detection ──────────────────────────────────────────────────────────────
_USE_LATEX = bool(shutil.which("latex") or shutil.which("pdflatex"))

# ── Paths ───────────────────────────────────────────────────────────────────────
RESULTS_CSV = RESULTS / "baseline.csv"
FIGS_DIR = RESULTS / "figures"

# ── Publication style ───────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "sans-serif" if not _USE_LATEX else "Computer Modern",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 18,
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "xtick.labelsize": 15,
        "ytick.labelsize": 13,
        "legend.fontsize": 16,
        "legend.title_fontsize": 15,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#cccccc",
        "legend.borderpad": 0.6,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.linewidth": 0.4,  # Lightened grid
        "grid.color": "#E5E7EB",  # Softer grey grid
        "xtick.bottom": False,
        "ytick.left": True,
        "ytick.major.size": 3.0,
        "ytick.major.width": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)

if _USE_LATEX:
    plt.rcParams.update({"text.usetex": True})


def _t(s: str) -> str:
    """Replace Unicode chars with LaTeX-safe equivalents when usetex is active."""
    if not _USE_LATEX:
        return s
    for old, new in [("\u202f", r"\,"), ("×", r"$\times$"), ("—", "---"), ("↔", r"$\leftrightarrow$")]:
        s = s.replace(old, new)
    return s


_YLABEL = r"Execution time (s / Gcell$\cdot$iter)" if _USE_LATEX else "Execution time (s / Gcell\u00b7iter)"

# ── Layout ──────────────────────────────────────────────────────────────────────
VARIANT_ORDER = ["Fortran", "CPP", "Fortran-OMP", "CPP-OMP", "Fortran-ACC", "CUDA", "CUDA-pinned"]
FUNCTION_ORDER = ["CDW", "CDU", "CDV", "CVD"]

# DESIGN FIX: Slimmer bars, wider gap between groups
BAR_WIDTH = 0.05
GROUP_GAP = 0.22

# ── Palette — Professional, story-driven colors ─────────────────────────────────
# CPU variants — medium-bright blues so dark hatch lines remain visible
C_FORTRAN = "#3571B5"  # Medium Blue
C_CPP = "#70AED8"  # Light Sky Blue
C_FORTRAN_OMP = "#3571B5"  # Medium Blue (distinguished by hatch)
C_CPP_OMP = "#70AED8"  # Light Sky Blue (distinguished by hatch)
C_FORTRAN_ACC = "#4A90E2"  # Slightly different blue

# CUDA kernel gets a punchy accent color to draw the eye to the "blazing fast" part
C_KERNEL = "#D7263D"  # Sharp Crimson Red

# Memory remains muted to show weight without dominating visually
C_MEM_MOV = "#C4C4C4"  # Muted Blue-Grey
C_MEM_ALLOC = "#EAEAEA"  # Very Light Blue-Grey

# Hatches for accessibility and OpenMP distinction
H_FORTRAN = ""
H_CPP = ""
H_FORTRAN_OMP = "////"  # Denser hatch for visibility
H_CPP_OMP = "\\\\\\\\"  # Opposite direction for C++
H_FORTRAN_ACC = "oooo"  # GPU-accelerated Fortran
H_KERNEL = ""
H_MEM = "xxxx"  # Pageable memory
H_MEM_PINNED = "////"  # Pinned memory (distinct hatch)
H_ALLOC = "...."

ECOLOR = "#404040"  # Slightly softer error bar color


class _MemPatch(mpatches.Patch):
    """Marker subclass so the legend handler can identify memory-bar patches."""

    pass


class _WhiteHatchHandler(HandlerPatch):
    """Draws the legend swatch in two layers: white hatch lines + dark border."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        artists = super().create_artists(legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)
        artists[0].set_edgecolor("white")  # hatch lines → white
        border = mpatches.Rectangle(
            (-xdescent, -ydescent), width, height, linewidth=0.8, edgecolor="#1e1e1e", fill=False, transform=trans
        )
        return artists + [border]


LEGEND_PATCHES = [
    mpatches.Patch(facecolor=C_FORTRAN, hatch=H_FORTRAN, edgecolor="#1e1e1e", label="Fortran (serial)"),
    mpatches.Patch(facecolor=C_CPP, hatch=H_CPP, edgecolor="#1e1e1e", label="C++ (serial)"),
    mpatches.Patch(facecolor=C_FORTRAN_OMP, hatch=H_FORTRAN_OMP, edgecolor="#1e1e1e", label="Fortran (OpenMP)"),
    mpatches.Patch(facecolor=C_CPP_OMP, hatch=H_CPP_OMP, edgecolor="#1e1e1e", label="C++ (OpenMP)"),
    mpatches.Patch(facecolor=C_FORTRAN, hatch=H_FORTRAN_ACC, edgecolor="#1e1e1e", label="Fortran (OpenACC)"),
    mpatches.Patch(facecolor=C_KERNEL, hatch=H_KERNEL, edgecolor="#1e1e1e", label=_t("CUDA \u2014 kernel")),
    _MemPatch(facecolor=C_MEM_MOV, hatch=H_MEM, edgecolor="#1e1e1e", label=_t("CUDA \u2014 pageable transfer")),
    _MemPatch(facecolor=C_MEM_MOV, hatch=H_MEM_PINNED, edgecolor="#1e1e1e", label=_t("CUDA \u2014 pinned transfer")),
    _MemPatch(facecolor=C_MEM_ALLOC, hatch=H_ALLOC, edgecolor="#1e1e1e", label=_t("CUDA \u2014 malloc / free")),
]


# ── CSV helpers ─────────────────────────────────────────────────────────────────


def parse_cell(s: str) -> tuple:
    s = s.strip()
    if not s:
        return None, 0.0
    m = re.match(r"([\d.]+)\+-([\d.]+)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    try:
        return float(s), 0.0
    except ValueError:
        return None, 0.0


def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter=";")
        for raw in reader:
            r: dict = {
                "function": raw["function"],
                "variant": raw["variant"],
                "grid_x": int(raw["grid_x"]),
                "grid_y": int(raw["grid_y"]),
                "grid_z": int(raw["grid_z"]),
                "iters": int(raw["iters"]),
            }
            for col in ("cuda_malloc_ms", "cuda_h2d_ms", "cuda_kernel_run", "cuda_d2h_ms", "cuda_free_ms", "total_ms"):
                mean, std = parse_cell(raw.get(col, ""))
                r[col] = mean
                r[col + "_std"] = std
            rows.append(r)
    return rows


def group_by_scenario(rows: list[dict]) -> dict:
    groups: dict = defaultdict(list)
    for r in rows:
        key = (r["grid_x"], r["grid_y"], r["grid_z"], r["iters"])
        groups[key].append(r)
    return groups


# ── Plotting ────────────────────────────────────────────────────────────────────


def plot_scenario(
    key: tuple,
    rows: list[dict],
    out_stem: str,
    omit_title: bool = False,
    omit_x_description: bool = False,
    omit_error_bars: bool = False,
) -> None:
    gx, gy, gz, niter = key
    norm = (gx * gy * gz * niter) / 1e6  # s per Gcell·iter

    lookup = {(r["function"], r["variant"]): r for r in rows}
    present_funcs = {r["function"] for r in rows}
    funcs = [f for f in FUNCTION_ORDER if f in present_funcs]
    funcs += sorted(present_funcs - set(funcs))

    n_funcs = len(funcs)
    n_variants = len(VARIANT_ORDER)

    group_width = n_variants * BAR_WIDTH + GROUP_GAP
    group_ctrs = np.arange(n_funcs) * group_width
    offsets = (np.arange(n_variants) - (n_variants - 1) / 2) * BAR_WIDTH

    fig_w = 4.0 + n_funcs * 2.2
    fig, ax = plt.subplots(figsize=(fig_w, 3.4))

    for vi, variant in enumerate(VARIANT_ORDER):
        for fi, func in enumerate(funcs):
            row = lookup.get((func, variant))
            if row is None:
                continue

            x = group_ctrs[fi] + offsets[vi]
            ebar_kw = dict(elinewidth=0.8, ecolor=ECOLOR, capsize=2.5, capthick=0.8, zorder=6)

            if variant in ("CUDA", "CUDA-pinned"):
                kernel_ms = (row["cuda_kernel_run"] or 0.0) / norm
                mem_ms = ((row["cuda_h2d_ms"] or 0.0) + (row["cuda_d2h_ms"] or 0.0)) / norm
                alloc_ms = ((row["cuda_malloc_ms"] or 0.0) + (row["cuda_free_ms"] or 0.0)) / norm
                total_std = (row["total_ms_std"] or 0.0) / norm

                # Use the unique memory hatch based on the variant
                current_h_mem = H_MEM_PINNED if variant == "CUDA-pinned" else H_MEM

                ax.bar(
                    x,
                    kernel_ms,
                    BAR_WIDTH,
                    color=C_KERNEL,
                    hatch=H_KERNEL,
                    edgecolor="#1e1e1e",
                    linewidth=0.8,
                    zorder=3,
                )

                # Dynamic memory hatch applied here
                ax.bar(
                    x,
                    mem_ms,
                    BAR_WIDTH,
                    bottom=kernel_ms,
                    color=C_MEM_MOV,
                    hatch=current_h_mem,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3,
                )
                ax.bar(
                    x,
                    mem_ms,
                    BAR_WIDTH,
                    bottom=kernel_ms,
                    fill=False,
                    hatch="",
                    edgecolor="#1e1e1e",
                    linewidth=0.8,
                    zorder=4,
                )

                ax.bar(
                    x,
                    alloc_ms,
                    BAR_WIDTH,
                    bottom=kernel_ms + mem_ms,
                    color=C_MEM_ALLOC,
                    hatch=H_ALLOC,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3,
                    **({} if omit_error_bars else {"yerr": total_std, "error_kw": ebar_kw}),
                )
                ax.bar(
                    x,
                    alloc_ms,
                    BAR_WIDTH,
                    bottom=kernel_ms + mem_ms,
                    fill=False,
                    hatch="",
                    edgecolor="#1e1e1e",
                    linewidth=0.8,
                    zorder=4,
                )

            else:
                total_ms = (row["total_ms"] or 0.0) / norm
                total_std = (row["total_ms_std"] or 0.0) / norm
                palette = {
                    "Fortran": (C_FORTRAN, H_FORTRAN),
                    "Fortran-OMP": (C_FORTRAN_OMP, H_FORTRAN_OMP),
                    "Fortran-ACC": (C_FORTRAN, H_FORTRAN_ACC),
                    "CPP": (C_CPP, H_CPP),
                    "CPP-OMP": (C_CPP_OMP, H_CPP_OMP),
                }
                c, h = palette.get(variant, (C_FORTRAN, H_FORTRAN))
                ax.bar(
                    x,
                    total_ms,
                    BAR_WIDTH,
                    color=c,
                    hatch=h,
                    edgecolor="#1e1e1e",
                    linewidth=0.8,
                    zorder=3,
                    **({} if omit_error_bars else {"yerr": total_std, "error_kw": ebar_kw}),
                )

    # ── Axes decoration ────────────────────────────────────────────────────────
    ax.set_xticks(group_ctrs)
    ax.set_xticklabels(funcs)
    ax.set_ylabel(_YLABEL, labelpad=6)
    if not omit_x_description:
        ax.set_xlabel("Kernel", labelpad=6)

    # Pad limits so bars don't hug the edges of the plot
    ax.set_xlim(group_ctrs[0] - group_width * 0.55, group_ctrs[-1] + group_width * 0.55)
    ax.set_ylim(bottom=0)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:g}"))

    if not omit_title:
        ax.set_title(
            _t(f"Grid {gx}\u202f×\u202f{gy}\u202f×\u202f{gz},  {niter} iterations per call"),
            pad=10,
        )

    # ── Legend ─────────────────────────────────────────────────────────────────
    present_variants = {r["variant"] for r in rows}

    def _variant_in_label(label):
        if "OpenACC" in label:
            return "Fortran-ACC" in present_variants
        if "Fortran" in label and "OpenMP" not in label:
            return "Fortran" in present_variants
        if "Fortran (OpenMP)" in label:
            return "Fortran-OMP" in present_variants
        if "C++ (serial)" in label:
            return "CPP" in present_variants
        if "C++ (OpenMP)" in label:
            return "CPP-OMP" in present_variants
        if "pinned transfer" in label:
            return "CUDA-pinned" in present_variants
        if "pageable transfer" in label:
            return "CUDA" in present_variants
        # Kernel and Malloc/Free labels should show up if ANY CUDA variant is present
        if "CUDA" in label:
            return "CUDA" in present_variants or "CUDA-pinned" in present_variants
        return False

    visible_patches = [p for p in LEGEND_PATCHES if _variant_in_label(p.get_label())]

    ax.legend(
        handles=visible_patches,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Implementation",
        ncol=1,
        handlelength=1.6,
        handleheight=1.1,
        borderaxespad=0.0,
        handler_map={_MemPatch: _WhiteHatchHandler()},
    )

    fig.tight_layout()

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        dest = FIGS_DIR / f"{out_stem}.{ext}"
        fig.savefig(dest)
        print(f"  saved → {dest}")

    plt.close(fig)


# ── Entry point ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark results.")
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        metavar="FILE",
        help="Results CSV (default: benchmarks/results/baseline.csv)",
    )
    parser.add_argument("--omit-title", action="store_true", help="Omit the figure title")
    parser.add_argument(
        "--omit-x-description", action="store_true", help='Omit the x-axis "Kernel" label (keeps function names)'
    )
    parser.add_argument("--omit-error-bars", action="store_true", help="Omit error bars (±std) from all bars")
    args = parser.parse_args()

    results_csv = args.results if args.results is not None else RESULTS_CSV
    if not results_csv.exists():
        raise FileNotFoundError(f"Results file not found: {results_csv}")

    rows = load_csv(results_csv)
    groups = group_by_scenario(rows)

    print(f"Found {len(rows)} rows across {len(groups)} scenario(s).")

    for key in sorted(groups):
        gx, gy, gz, niter = key
        stem = f"grid_{gx}x{gy}x{gz}_niter{niter}"
        print(f"Plotting {stem} ...")
        plot_scenario(
            key,
            groups[key],
            stem,
            omit_title=args.omit_title,
            omit_x_description=args.omit_x_description,
            omit_error_bars=args.omit_error_bars,
        )

    print("Done.")


if __name__ == "__main__":
    main()
