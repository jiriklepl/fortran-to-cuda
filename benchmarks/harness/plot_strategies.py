"""Reproduce policy charts from saved public benchmark reports; never execute workloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/fortran-policy-plots')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MODES = ['native', 'current', 'sections', 'auto', 'chunked', 'hybrid']
LABELS = ['Native CPU', 'Whole-array CUDA', 'Sections', 'Scope selection', 'Chunked GPU', 'Hybrid policy']
COLORS = ['#737373', '#2166ac', '#67a9cf', '#1b7837', '#d95f02', '#762a83']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', default='Current compiler')
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report['status'] == 'passed'
    config = report['configuration']
    cases, grids = config['cases'], config['grids']
    modes = [mode for mode in MODES if mode in config['modes']]
    args.output.mkdir(parents=True, exist_ok=True)
    times = {(item['case'], item['mode'], tuple(item['grid'])): item for item in report['timings']}
    validation = {(item['case'], item['mode'], tuple(item['grid'])): item for item in report['validation']}
    assert all(item['passed'] for item in report['validation'])
    saved = []
    for metric, field, median, ylabel in [
        ('complete-wall-relative', 'process_seconds', 'process_seconds_median', 'Complete process time / native (lower is better)'),
        ('entry-time', 'seconds_per_call', 'seconds_per_call_median', 'Time per entry call (ms, lower is better)'),
        ('transfer-volume', None, None, 'Host/device transfer per entry (MiB)'),
    ]:
        # Transfer volumes grow with grid size. Separate linear axes keep the
        # smaller panels readable and avoid freezing all limits at panel one.
        fig, axes = plt.subplots(1, len(grids), figsize=(7*len(grids), 5), squeeze=False, sharey=bool(field))
        x = np.arange(len(cases)); width = 0.8/len(modes)
        for axis, grid in zip(axes[0], grids, strict=True):
            for index, mode in enumerate(modes):
                values, lower, upper = [], [], []
                for case in cases:
                    item = times[case, mode, tuple(grid)]
                    if field:
                        scale = 1/times[case, 'native', tuple(grid)]['process_seconds_median'] if metric == 'complete-wall-relative' else 1000
                        value = item[median]*scale
                        samples = [sample[field]*scale for sample in item['samples']]
                        lower.append(value-min(samples)); upper.append(max(samples)-value)
                    else:
                        check = validation[case, mode, tuple(grid)]
                        runtime = check['trace']['runtime']
                        value = (runtime.get('upload_bytes', 0)+runtime.get('download_bytes', 0))/check['calls']/2**20
                    values.append(value)
                    saved.append(dict(metric=metric, case=case, mode=mode, grid=grid, value=value))
                bars = axis.bar(x+(index-(len(modes)-1)/2)*width, values, width=width,
                         label=LABELS[MODES.index(mode)], color=COLORS[MODES.index(mode)],
                         **({'yerr':[lower,upper], 'capsize':2, 'error_kw':{'elinewidth':0.8}} if field else {}))
                for bar, case in zip(bars, cases, strict=True):
                    runtime = validation[case, mode, tuple(grid)]['trace']['runtime']
                    if mode in ('auto', 'hybrid') and not runtime.get('kernel_count', 0):
                        bar.set_hatch('///'); bar.set_edgecolor('#444444'); bar.set_linewidth(.3)
                if not field:
                    for at, value in zip(x+(index-(len(modes)-1)/2)*width, values, strict=True):
                        if value == 0: axis.plot(at, 0, marker='_', color=COLORS[MODES.index(mode)])
            if metric == 'complete-wall-relative':
                axis.axhline(1, color='black', linewidth=1)
                axis.axhline(1.05, color='#b35806', linestyle='--', linewidth=1)
            if field: axis.set_yscale('log')
            else: axis.set_ylim(bottom=0)
            axis.set_xticks(x, [case.capitalize() for case in cases])
            axis.set_title(' × '.join(map(str, grid)))
            axis.grid(axis='y', alpha=.2); axis.set_axisbelow(True)
        axes[0,0].set_ylabel(ylabel)
        handles = [matplotlib.patches.Patch(color=COLORS[MODES.index(mode)], label=LABELS[MODES.index(mode)]) for mode in modes]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,-.02), ncol=3, frameon=False)
        fig.suptitle(f"{args.label} · GPU policies · {config['host_threads']} host threads · real{config['precision_bits']}")
        if field:
            note = 'Medians; whiskers: sample min–max. One warmup process, 3 measured (7 near a gate).'
            if metric == 'complete-wall-relative': note += '\nStartup and teardown included; black: native, dashed: 5% regression gate.'
            else: note += '\nEntry time includes transfers and completion; excludes process startup and teardown.'
        else: note = 'Separate correctness runs; uploads + downloads, divided by executed calls. Zero means no CUDA transfer.\nTransfer panels use separate linear scales.'
        fig.text(.5,-.11,note+'\nHatching: automatic/hybrid policy selected native execution in validation.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.08,1,.96))
        for ext in ['png','svg','pdf']:
            fig.savefig(args.output/f'{metric}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    (args.output/'source.json').write_text(json.dumps({'report':str(args.report.resolve()),
        'sha256':hashlib.sha256(args.report.read_bytes()).hexdigest(),
        'plot_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'configuration':config,'values':saved},indent=2)+'\n')

if __name__ == '__main__': main()
