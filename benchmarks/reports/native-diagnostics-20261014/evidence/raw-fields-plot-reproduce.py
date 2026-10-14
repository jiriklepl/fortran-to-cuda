"""Render immutable native-array and private-rebuild diagnostic evidence."""
import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
sys.path.insert(0, str(WORK / 'elmm-pipeline'))
from elmm_pipeline.campaigns.halos import fields


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    os.sched_setaffinity(0, {4, 5, 10, 11})
    checkpoint = WORK / 'build-receipt-checkpoint-gfortran14/full-field-checkpoint-02'
    report_path = checkpoint / 'channel-simple/report.json'
    driver_path = checkpoint / 'driver.json'
    comparison_path = WORK / 'repaired-native-checkpoint-gfortran14/comparison.json'
    paths_audit_path = comparison_path.with_name('path-data-audit.json')
    report, metadata, comparison, path_audit = [json.loads(path.read_text())
        for path in [report_path, driver_path, comparison_path, paths_audit_path]]
    if report['status'] != 'passed' or len(report['runs']) != 2 or path_audit['status'] != 'passed':
        raise ValueError('Completed diagnostic evidence required')
    arrays = []
    source_hashes = {str(path): digest(path) for path in [report_path, driver_path,
                    comparison_path, paths_audit_path, Path(__file__)]}
    for index in range(2):
        directory = checkpoint / 'channel-simple' / f'native-{index}'
        for name, sha in report['runs'][index]['files_sha256'].items():
            if name.startswith('fields-'):
                path = directory / name
                if digest(path) != sha:
                    raise ValueError('Frozen field dump changed: ' + str(path))
                source_hashes[str(path)] = sha
        arrays.append(fields(directory, metadata['driver']))
    rows = []
    for name, (reference, layout) in arrays[0].items():
        actual, actual_layout = arrays[1][name]
        if layout != actual_layout:
            raise ValueError('Bounds differ')
        tolerance = metadata['driver']['atol'] + metadata['driver']['rtol'] * np.abs(reference)
        ratio = np.divide(np.abs(actual - reference), tolerance)
        rows.append({'field': name[7:-4], 'values': int(reference.size),
                     'max_tolerance_fraction': float(np.max(ratio, initial=0)),
                     'max_absolute_error': float(np.max(np.abs(actual-reference), initial=0)),
                     'layout': layout})
    rows.sort(key=lambda row: (not bool(row['values']), row['field']))
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
    data = {'schema_version': 1, 'performance_measured': False, 'qualification': False,
            'source_files_sha256': source_hashes, 'full_field_runs': 2, 'completed_steps': [109, 109],
            'array_count': len(rows), 'values': sum(row['values'] for row in rows),
            'tolerances': {'atol': metadata['driver']['atol'], 'rtol': metadata['driver']['rtol']},
            'fields': rows, 'original_build_identity': comparison['original_build_identity'],
            'repaired_build_identity': comparison['repaired_build_identity'],
            'repaired_application_executed': False,
            'plots_sha256': {str(path.name): digest(path) for path in HERE.glob('native-checkpoint.*')}}
    (HERE / 'data.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'arrays': len(rows), 'values': data['values'],
                      'max_tolerance_fraction': max(row['max_tolerance_fraction'] for row in rows)}, indent=2))


if __name__ == '__main__':
    main()
