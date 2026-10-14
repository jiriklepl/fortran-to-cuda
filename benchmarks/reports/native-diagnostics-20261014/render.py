"""Render frozen diagnostic aggregates; raw-field provenance is in data.json."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/native-diagnostic-report-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent

def main():
    data=json.loads((HERE/'data.json').read_text())
    rows=data['fields']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    figure = plt.figure(figsize=(12.8, 10.5), layout='constrained')
    grid = figure.add_gridspec(2, 2, height_ratios=[3.7, 1.3], width_ratios=[2.0, 1.35])
    axis = figure.add_subplot(grid[0, 0])
    y = np.arange(len(rows))
    axis.barh(y, [row['max_tolerance_fraction'] for row in rows],
              color=['#237955' if row['values'] else '#94a3b8' for row in rows], height=.65)
    axis.set_yticks(y, [row['field'] for row in rows])
    axis.invert_yaxis()
    axis.axvline(1, color='#9f3d30', linestyle='--', linewidth=1.4, label='unchanged failure boundary')
    axis.set_xlim(0, 1.05)
    axis.set_xlabel('Maximum |repeat − reference| / (1e−6 + 1e−5 × |reference|)')
    axis.set_title('Two native channel-simple diagnostic runs\n109 natural steps each · every stored element compared', loc='left')
    axis.grid(axis='x', color='#e5e7eb')
    axis.set_axisbelow(True)
    axis.legend(loc='lower right', frameon=False, fontsize=8)
    for index, row in enumerate(rows):
        text = f"{row['values']:,} values" if row['values'] else 'empty allocation · bounds checked'
        axis.text(max(.015, row['max_tolerance_fraction'] + .015), index, text, va='center', fontsize=8,
                  color='#334155')
    bounds = figure.add_subplot(grid[0, 1])
    bounds.axis('off')
    bounds.set_title('Complete storage coverage', loc='left')
    lines = []
    for row in rows:
        lower, upper = row['layout']['lower_bounds'], row['layout']['upper_bounds']
        extents = ', '.join(f'{lo}:{hi}' for lo, hi in zip(lower, upper, strict=True))
        lines.append(f"{row['field']}:  {extents}")
    bounds.text(0, .96, '\n'.join(lines), va='top', fontsize=9, linespacing=1.75, family='monospace')
    proof = figure.add_subplot(grid[1, :])
    proof.axis('off')
    proof.set_title('Private six-declaration repair · complete 72-object native rebuild', loc='left', pad=10)
    labels = [('72 / 72', 'Executable sections\nbyte-identical', '#1d648c'),
              ('72 / 72', 'Nondebug relocations\nbyte-identical', '#1d648c'),
              ('70 / 72', 'All allocated sections\nbyte-identical', '#1d648c'),
              ('2 / 72', 'Only source-path strings\nand their symbol offsets', '#8b6a23')]
    for index, (number, label, color) in enumerate(labels):
        x = .015 + index * .25
        proof.text(x, .72, number, color=color, fontsize=24, fontweight='bold', transform=proof.transAxes)
        proof.text(x, .34, label, color='#334155', fontsize=10, linespacing=1.5, transform=proof.transAxes)
    proof.text(.015, -.08,
        'The repaired application was not executed. Original source and binary remain unchanged.\n'
        'Native repeatability does not establish Fortran definedness or automatic-offload correctness. No performance qualification.',
        fontsize=9, color='#475569', transform=proof.transAxes)
    figure.suptitle('Native arrays and rebuild checkpoint · diagnostics only', fontsize=18, fontweight='bold', x=.015, ha='left')
    for extension in ['svg', 'png', 'pdf']:
        figure.savefig(HERE / ('native-checkpoint.' + extension), dpi=180, metadata={'Creator': 'native-diagnostic-visuals/render.py'})
    plt.close(figure)

if __name__ == '__main__':
    main()
