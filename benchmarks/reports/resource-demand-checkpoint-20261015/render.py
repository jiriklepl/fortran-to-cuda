#!/usr/bin/env python3
"""Render archived reachability diagnostics; never run source analysis."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/resource-demand-report-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'svg.fonttype': 'none', 'svg.hashsalt': 'resource-demand-20261015-v1'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path(__file__).with_name('report.json'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    directory = args.output or args.report.parent
    directory.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.report.read_text())
    diagnostic_path = args.report.parent / report['runs'][1]['diagnostic']
    raw = diagnostic_path.read_bytes()
    rows = json.loads(raw)['classifications']
    if len(rows) != 4 or any(not r['proof_validate_completed'] for r in rows):
        raise ValueError('Expected exactly four individually validated diagnostic observations')
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set(xlim=(0, 12), ylim=(0, 9)); ax.axis('off')

    def box(x, y, w, h, text, color='#edf2f8', edge='#8da0b3', size=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.035,rounding_size=.07',
                                   facecolor=color, edgecolor=edge, linewidth=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size, linespacing=1.45)

    def arrow(start, end):
        ax.annotate('', xy=end, xytext=start, arrowprops={'arrowstyle': '->', 'color': '#536a7c', 'lw': 1.5})

    ax.text(.2, 8.65, 'Resource-directed analysis: first observations, no executable window',
            fontsize=17, weight='bold')
    implementation = report['implementation']
    final = implementation['final_production_commit']
    validation = implementation['validation']
    subtitle = (f"Final compiler {final[:12]} · {validation['compiler_host']['passed']} host + "
                f"{validation['compiler_cuda']['passed']} CUDA + {validation['pipeline']['passed']} pipeline tests passed "
                f"({validation['pipeline']['skipped']} skipped)")
    ax.text(.2, 8.28, subtitle, fontsize=10, color='#596b7a')
    box(.2, 7.2, 3.45, .8, 'Whole-interval resource inventory\nIncludes future uses and guards')
    box(4.2, 7.2, 3.5, .8, 'Exact original source selection\nSelective reachability demand')
    box(8.25, 7.2, 3.55, .8, 'Accesses + explicit barriers\nUnresolved storage obligations')
    arrow((3.7, 7.6), (4.15, 7.6)); arrow((7.75, 7.6), (8.2, 7.6))
    ax.text(.2, 6.86, 'The actual-source observations below used unsealed SourceEffects, not the immutable application pipeline.',
            fontsize=10, color='#596b7a')
    columns = [(.3, 'Original native operation'), (4.0, 'Old U / V / W'),
               (6.25, 'Viscosity'), (8.25, 'Coverage and remaining obligations')]
    for x, label in columns:
        ax.text(x, 6.38, label, weight='bold', fontsize=10)
    labels = ['StressBoundaryFlux', 'ComputeUVWFluxesWM', 'BoundViscosity', 'ComputeTDiff']
    velocity = ['None found*', 'Association only*', 'No direct access', 'Association only']
    viscosity = ['None found*', 'None found*', 'Associate / read / write', 'Read']
    explanations = ['Blocked: indexed scalar-actual payload',
                    'Blocked: 18 derived-array associations',
                    'Reachability covered; no execution proof',
                    'Reachability covered; no execution proof']
    for i, row in enumerate(rows):
        y = 5.36 - i * .8
        covered = row['selection_covered']
        box(.2, y, 11.6, .67, '', '#edf4f8' if covered else '#fff2e1',
            '#94aebe' if covered else '#d4ad75')
        ax.text(.35, y + .35, labels[i], va='center', weight='bold', fontsize=11)
        ax.text(4.0, y + .35, velocity[i], va='center', fontsize=10)
        ax.text(6.25, y + .35, viscosity[i], va='center', fontsize=9)
        ax.text(8.25, y + .43, explanations[i], va='center', fontsize=9)
        ax.text(8.25, y + .16, f"{row['storage_separation_obligations']} separation obligations unresolved",
                va='center', fontsize=9, color='#596b7a')
    ax.text(.2, 2.6, '* Incomplete selections: barriers may hide accesses. “None found” does not prove noninterference.',
            fontsize=10, color='#8b531e')
    first, second = report['runs']
    box(.2, 1.48, 5.65, .85,
        f"Pass 1 · ScopeBuilder · timeout {first['elapsed_seconds']:.3f} s\nNo completed session event; zero observations",
        '#f2f2f2', '#b1b6bb')
    box(6.15, 1.48, 5.65, .85,
        f"Pass 2 · frontend only · timeout {second['elapsed_seconds']:.3f} s\nSetup {second['source_effects_construction_seconds']:.3f} s; four validated observations",
        '#f2f2f2', '#b1b6bb')
    box(.2, .42, 11.6, .75,
        'Both batches stopped at the original 60-second cap. Remaining operations and full-summary comparisons were not reached.\n'
        'No final publication phase · no selective native hooks or application cost source added · no performance improvement',
        '#fff2e1', '#d4ad75', size=10)
    ax.text(.2, .12, 'Archived diagnostic SHA256 ' + hashlib.sha256(raw).hexdigest()[:16]
            + ' · 178 authenticated inputs unchanged · alias and execution authority remain false', fontsize=8, color='#596b7a')
    fig.subplots_adjust(left=.015, right=.995, bottom=.015, top=.995)
    fig.savefig(directory / 'operation-resource-map.svg', metadata={'Date': None, 'Creator': 'resource-demand-report-v1'})
    svg_path = directory / 'operation-resource-map.svg'
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
    fig.savefig(directory / 'operation-resource-map.png', dpi=160, metadata={'Software': 'resource-demand-report-v1'})
    plt.close(fig)


if __name__ == '__main__':
    main()
