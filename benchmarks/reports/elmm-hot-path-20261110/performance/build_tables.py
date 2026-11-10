#!/usr/bin/env python3
"""Reproduce this report's extracts from retained measurements; does not benchmark."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REPORTS = ROOT / 'benchmarks/reports'
sources = {}


def read(rel):
    p = REPORTS / rel
    sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + '\n')


def csvwrite(name, rows):
    with (HERE / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


cohorts = []


def cohort(identity, rel, key, selected, description, commit=None):
    raw = read(rel)
    cases = raw
    for part in key.split('.'):
        cases = cases[part]
    norm = {}
    for case, value in cases.items():
        modes = value.get('modes', value.get('statistics'))
        norm[case] = {}
        for mode, stats in modes.items():
            samples = stats.get('samples', stats.get('samples_seconds'))
            median = statistics.median(samples)
            assert len(samples) == 7
            assert abs(median - stats.get('median', stats.get('median_seconds'))) < 1e-10
            q1, _, q3 = statistics.quantiles(samples, n=4, method='inclusive')
            norm[case][mode] = dict(samples_seconds=samples, median_seconds=median,
                                    min_seconds=min(samples), max_seconds=max(samples),
                                    q1_seconds=q1, q3_seconds=q3)
        native = norm[case]['native']
        for mode, stats in norm[case].items():
            stats['speedup_over_same_cohort_native'] = native['median_seconds'] / stats['median_seconds']
            stats['paired_native_speedups'] = [n / s for n, s in zip(native['samples_seconds'], stats['samples_seconds'])]
            stats['paired_wins_over_native'] = sum(n > s for n, s in zip(native['samples_seconds'], stats['samples_seconds']))
    cohorts.append(dict(id=identity, source=str((REPORTS / rel).relative_to(ROOT)),
                        selected_mode=selected, description=description, commit=commit,
                        warmups_per_mode_case=1, measured_rounds=7, cases=norm))


cohort('oct22', 'graph-redesign-parity-20261022/performance.json', 'cases', 'reworked',
       'Historical original cohort; CPU affinity 0..11, CUDA runtime 13.4.',
       '7a612e40b5790e2d112ea137c6a38a9ff0ef7d41')
cohort('empty-native', 'gpu-residency-speedup-20261108/summary.json', 'cases', 'candidate',
       'Empty native-call retention; channel regression preserved.',
       '2619b7f25745bef69cf9ab9e265a3cde278e7156')
cohort('copy-v3', 'gpu-residency-copy-v3-speedup-20261108/summary.json', 'cases', 'candidate',
       'Single-operation compact-copy calibration; no end-to-end gain.',
       '03c695d545c8c4b59220fe9fe46f1b04fd72f86b')
cohort('resident-four-controls', 'gpu-resident-four-controls-20261110/summary.json', 'summary', 'candidate',
       'Dated working-tree milestone: broad resident GPU path plus GPU pressure solver.')
cohort('solver-cache', 'gpu-local-solver-cache-20261110/summary.json', 'summary', 'candidate',
       'Dated working-tree milestone: persistent VkFFT compiled-binary cache.')
cohort('exact-division', 'gpu-exact-division-20261110/summary.json', 'performance', 'candidate',
       'Dated working-tree milestone: exact guarded division plus grouping/private-scalar fusion.')
cohort('host-preparation', 'gpu-host-preparation-20261110/summary.json', 'performance', 'candidate',
       'Dated working-tree milestone: host index preparation, valid second whole cohort only.')
cohort('async-output', 'gpu-async-output-20261110/summary.json', 'timing.cases', 'candidate',
       'Dated working-tree milestone: retained asynchronous-output implementation.')
cohort('latest-selected', 'gpu-kernel-screening-20261110/summary.json', 'coordinate_experiment.cases', 'gpu-cached',
       'Latest measurement of retained GPU binary, not the rejected coordinate-decoder candidate. '
       'Snapshot 620e630 includes source regeneration yielding identical binaries; no new timing of that commit.',
       '620e630e21735b8e90b496ed68621c444e26b9aa')

latest = read('gpu-kernel-screening-20261110/summary.json')
wrap = read('gpu-current-state-wrap-20261110/summary.json')
profiles = latest['selected_diagnostic_profiles']['cases']
profile_keys = ['kernel_count', 'kernel_activity_sum_seconds', 'kernel_union_seconds',
                'copy_union_seconds', 'physical_transfers_by_cuda_kind', 'kernel_window_seconds',
                'gpu_activity_within_kernel_window_seconds', 'gpu_idle_within_kernel_window_seconds',
                'cuda_api_union_seconds', 'cuda_api_overlap_gpu_seconds',
                'cuda_api_outside_gpu_inside_kernel_window_seconds']
compact_profiles = {case: {k: v[k] for k in profile_keys} for case, v in profiles.items()}
for case, values in profiles.items():
    compact_profiles[case]['stages'] = {k: {'seconds': v['seconds'], 'launches': v['launches']} for k, v in values['stages'].items()}
    assert abs(sum(s['seconds'] for s in values['stages'].values()) - values['kernel_activity_sum_seconds']) < 1e-8

# Preserve the only /tmp dependency in a small, durable extract. Subsequent runs
# use that extract. Hash the original analysis as well as the retained extract.
oldpath = HERE / 'be58-profile-extract.json'
if not oldpath.exists():
    oldsource = Path('/tmp/fortran-recovered-gap-profile-20261109-02/analysis.json')
    old = json.loads(oldsource.read_text())
    keys = ['case', 'mode', 'timing_sample', 'database', 'copies', 'gpu_kernel_union_seconds',
            'gpu_copy_union_seconds', 'gpu_all_union_seconds', 'host_api_union_seconds',
            'host_copy_api_union_seconds', 'host_api_and_gpu_intersection_seconds',
            'host_api_or_gpu_union_seconds']
    rows = [{k: r[k] for k in keys} | {'kernel_launches': sum(x['calls'] for x in r['gpu_kernels'])}
            for r in old['rows'] if r['mode'] == 'be58']
    dump(oldpath.name, dict(source=str(oldsource), source_sha256=hashlib.sha256(oldsource.read_bytes()).hexdigest(),
                           source_report='benchmarks/reports/gpu-residency-recovered-gap-20261109/README.md',
                           note='Separate traced full runs; CPU API and device spans overlap.', rows=rows))
old_profiles = json.loads(oldpath.read_text())
sources[str(oldpath.relative_to(ROOT))] = hashlib.sha256(oldpath.read_bytes()).hexdigest()

phase = read('gpu-phase-attribution-20261110/summary.json')
phase_rows = [r for r in phase['analysis']['host_runs']
              if not r['traced'] and '/fortran-phase-attribution-20261110-01/host/' in r['directory']]
assert len(phase_rows) == 4
data = dict(schema_version=1, timing_scope='Complete-process wall time including startup and output; separate cohorts.',
            no_new_benchmarks=True, cohorts=cohorts, selected_binary_identity=wrap['application'],
            latest_diagnostic_profiles=compact_profiles, be58_diagnostic_profiles=old_profiles,
            phase_diagnostics=dict(note='One observation each; disjoint host call phases, not exclusive device computation.', rows=phase_rows),
            unavailable=['Comparable exclusive CPU procedure times at every historical commit',
                         'Fresh four-control cohort at latest selected state',
                         'Additive wall-time breakdown combining CUDA API and device durations'])
dump('data.json', data)
dump('sources.json', dict(algorithm='sha256', sources=sources))

timing_rows = []
for c in cohorts:
    for case, modes in c['cases'].items():
        for mode, values in modes.items():
            for i, seconds in enumerate(values['samples_seconds'], 1):
                timing_rows.append(dict(cohort=c['id'], case=case, mode=mode, measured_round=i, seconds=seconds))
csvwrite('timing-samples.csv', timing_rows)
csvwrite('gpu-stage-times.csv', [dict(case=case, stage=stage, seconds=stats['seconds'], launches=stats['launches'])
                               for case, p in compact_profiles.items() for stage, stats in p['stages'].items()])
csvwrite('host-phase-times.csv', [dict(case=r['case'], mode=r['mode'], phase=p, seconds=s['seconds'], calls=s['calls'])
                                for r in phase_rows for p, s in r['phases'].items()])

def group(name):
    if name.startswith('subgrid::'): return 'SGS'
    if name.startswith('momentumadvection::') or name == 'dynamics::convection': return 'Convection and advection'
    if name.startswith('momentumdiffusion::'): return 'Diffusion'
    if name.startswith('filters::'): return 'Channel filter'
    if name.startswith('resident_wall::'): return 'Wall model'
    if name.startswith(('boundaries::', 'scalarboundaries::')): return 'Velocity and viscosity boundaries'
    if name == 'VkFFT_main': return 'Poisson FFT kernels'
    if name.startswith('void poisson_solver::'): return 'Poisson packing, denominator, normalization'
    if name in ['<unnamed>::plane_reference', '<unnamed>::subtract_reference']: return 'Pressure reference'
    if name.startswith('pressure::') or name in ['pressure_faces', '<unnamed>::pressure_faces', 'dynamics::pressuregrad']: return 'Pressure preparation, halos and correction'
    if name.startswith('arrayutilities::'): return 'RK array updates'
    if name.startswith('resident_time::') or name in ['<unnamed>::partial_max', '<unnamed>::final_max']: return 'Time control and reductions'
    raise ValueError(name)

groups = {}
for case, p in compact_profiles.items():
    groups[case] = {}
    for name, s in p['stages'].items():
        g = groups[case].setdefault(group(name), {'seconds': 0, 'launches': 0, 'source_stages': []})
        g['seconds'] += s['seconds']; g['launches'] += s['launches']; g['source_stages'].append(name)
    assert abs(sum(g['seconds'] for g in groups[case].values()) - p['kernel_activity_sum_seconds']) < 1e-8
dump('gpu-stage-groups.json', groups)
print('Extracted', len(timing_rows), 'measured samples across', len(cohorts), 'cohorts; profile sums verified.')
for c in cohorts:
    vals = []
    for case in ['channel-simple', 'channel']:
        m = c['cases'][case][c['selected_mode']]
        vals.append(f"{m['median_seconds']:.3f} s ({m['speedup_over_same_cohort_native']:.2f}×)")
    print(c['id'], ' | '.join(vals))
for g in groups['channel-simple']:
    print(g, *(f"{groups[c].get(g, {}).get('seconds', 0):.3f}" for c in ['channel-simple', 'channel']))
