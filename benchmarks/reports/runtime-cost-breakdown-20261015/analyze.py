"""Freeze this diagnostic from local Nsight captures; never replace changed evidence.

Run from the repository with .venv/bin/python PATH/analyze.py. Raw captures stay
under benchmarks/results/runtime-cost-breakdown-20261015 and are hash-bound here.
CPU sample counts are not wall-time estimates. Device times overlap host API time.
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RAW = ROOT / "benchmarks/results/runtime-cost-breakdown-20261015"
OBS = ROOT / "benchmarks/results/owner-measurement-checkpoint-20261015-retry"
BINARY = OBS / "resident/candidate/build/ELMM"
SOURCE = OBS / "resident/candidate/application/ELMM/src/momentum_diffusion.f90"
PREVIOUS = OUT.parent / "owner-measurement-checkpoint-20261015/report.json"
SYMBOL = "__momentumdiffusion_MOD_momentumdiffusion_nobranch_2ord"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(name, data):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, f"refusing to replace changed evidence: {path}"
    else:
        path.write_bytes(data)


def document(name, value):
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    freeze(name, gzip.compress(data, mtime=0) if name.endswith(".gz") else data)


def database(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def union_duration(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def profile(name, directory):
    receipt = json.loads((directory / "completion.json").read_text())
    assert receipt["completed"] and receipt["exit_code"] == 0
    assert receipt["completion"]["natural_end"] and receipt["completion"]["completed_steps"] == 109
    assert all(item["passed"] for item in receipt["vtk"].values())
    assert receipt["binary_sha256"] == sha(BINARY)
    for filename, digest in receipt["artifacts"].items():
        assert sha(directory / filename) == digest
    for filename in ("command.json", "completion.json", "stdout.log", "stderr.log", "profile.py"):
        freeze(f"evidence/{name}/{filename}", (directory / filename).read_bytes())
    observations = [json.loads(line.split(" ", 1)[1]) for line in
                    (directory / "stderr.log").read_text().splitlines()
                    if line.startswith("FORT_SCOPE_MEASUREMENT ")]
    assert len(observations) == 327 and all(o["status"] == "complete" for o in observations)
    assert len({(o["identity"], o["call_site"]) for o in observations}) == 1
    with database(directory / "profile.sqlite") as db:
        offset, = db.execute("SELECT systemClockNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()
        # Linux SYSTEM_CLOCK and this capture's systemClockNs share the monotonic
        # clock. Verify every API interval is contained in an observed call.
        windows = [(o["clock_start"] - offset, o["clock_finish"] - offset) for o in observations]
        assert all(o["clock_rate"] == 1_000_000_000 for o in observations)
        api = list(db.execute("SELECT r.start,r.end,r.globalTid,s.value FROM "
                              "CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId"))
        outside = sum(end - start - sum(max(0, min(end, b) - max(start, a))
                                       for a, b in windows) for start, end, _, _ in api)
        assert outside == 0
        threads = {tid for _, _, tid, _ in api}
        assert len(threads) == 1
        duration = union_duration([(s, e) for s, e, _, _ in api])
        assert duration == sum(e - s for s, e, _, _ in api)
        groups = collections.defaultdict(lambda: {"count": 0, "nanoseconds": 0})
        for start, end, _, symbol in api:
            groups[symbol]["count"] += 1
            groups[symbol]["nanoseconds"] += end - start
        copies = [dict(kind=k, count=n, bytes=b, nanoseconds=t) for k, n, b, t in db.execute(
            "SELECT e.name,count(*),sum(m.bytes),sum(m.end-m.start) "
            "FROM CUPTI_ACTIVITY_KIND_MEMCPY m JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=m.copyKind GROUP BY e.id")]
        kernels = [dict(symbol=s, count=n, nanoseconds=t) for s, n, t in db.execute(
            "SELECT s.value,count(*),sum(k.end-k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id=k.shortName GROUP BY k.shortName")]
        raw_api = "start_ns,end_ns,global_tid,api\n" + "".join(
            f"{s},{e},{tid},{symbol}\n" for s, e, tid, symbol in sorted(api))
        freeze(f"evidence/{name}/api-intervals.csv.gz", gzip.compress(raw_api.encode(), mtime=0))
    owner_ns = sum(o["clock_finish"] - o["clock_start"] for o in observations)
    return dict(observed_calls=len(observations), owner_nanoseconds=owner_ns,
                api_calls=len(api), api_nanoseconds=duration, api_outside_owner_nanoseconds=outside,
                api_threads=list(threads), other_owner_nanoseconds=owner_ns - duration,
                api_groups=dict(groups), device_copies=copies, device_kernels=kernels,
                complete_process_seconds=receipt["profile_process_seconds"],
                process_time_is_qualification=False, capture_hashes=receipt["artifacts"],
                observation_identity=observations[0]["identity"], call_site=observations[0]["call_site"])


def cpu_samples():
    with database(RAW / "cpu/profile.sqlite") as db:
        leaves = list(db.execute(
            "SELECT s.value,m.value,c.originalIP,count(*) FROM SAMPLING_CALLCHAINS c "
            "JOIN StringIds s ON c.symbol=s.id JOIN StringIds m ON c.module=m.id "
            "WHERE c.stackDepth=0 GROUP BY c.symbol,c.module,c.originalIP ORDER BY c.originalIP"))
    # Infer the PIE load base by intersecting all page-aligned candidate bases
    # from short named functions without compiler-generated child symbols.
    symbols = {}
    nm_text = subprocess.check_output(["nm", "-S", str(BINARY)], text=True)
    freeze("evidence/binary-symbols.txt.gz", gzip.compress(nm_text.encode(), mtime=0))
    for line in nm_text.splitlines():
        fields = line.split()
        if len(fields) == 4:
            try:
                symbols[fields[3]] = (int(fields[0], 16), int(fields[1], 16))
            except ValueError:
                pass
    candidates = []
    for symbol, module, ip, count in leaves:
        if module != str(BINARY) or symbol not in symbols:
            continue
        value, size = symbols[symbol]
        if size >= 4096 or any(k.startswith(symbol + ".") for k in symbols):
            continue
        low, high = ip - value - size + 1, ip - value
        bases = set(range((low + 4095) // 4096 * 4096, high // 4096 * 4096 + 1, 4096))
        if bases:
            candidates.append(bases)
    common = set.intersection(*candidates)
    assert len(common) == 1
    base = common.pop()
    selected = [(ip, count) for symbol, module, ip, count in leaves if symbol == SYMBOL and module == str(BINARY)]
    locations = subprocess.check_output(["addr2line", "-e", str(BINARY)], text=True,
        input="\n".join(hex(ip - base) for ip, _ in selected) + "\n").splitlines()
    assert len(selected) == len(locations)
    mapped, classification, source_cache = [], collections.Counter(), {}
    by_line = collections.Counter()
    for (ip, count), location in zip(selected, locations):
        match = re.fullmatch(r"(.+):(\d+)(?: \(discriminator \d+\))?", location)
        assert match, location
        path, number = Path(match[1]), int(match[2])
        if path not in source_cache:
            source_cache[path] = path.read_text().splitlines()
            freeze("evidence/sources/" + path.name + ".gz", gzip.compress(path.read_bytes(), mtime=0))
        text = source_cache[path][number - 1].strip()
        if path == SOURCE and re.search(r"fort_t[135]_\d+_", text):
            category = "Generated native-correction preparation/control"
        elif path == SOURCE and "fort_" in text:
            category = "Other generated control"
        else:
            category = "Original source / structural / unclassified"
        classification[category] += count
        by_line[(path.name, number, text)] += count
        mapped.append(dict(ip=hex(ip), binary_offset=hex(ip - base), count=count,
                           source=path.name, line=number, text=text, category=category))
    document("evidence/diffusion-sample-locations.json.gz", mapped)
    all_functions = collections.Counter()
    for symbol, module, _, count in leaves:
        all_functions[(symbol, module)] += count
    document("evidence/cpu-leaf-counts.json.gz", [dict(symbol=s, module=m, samples=n)
             for (s, m), n in all_functions.most_common()])
    return dict(total_leaf_samples=sum(n for _, _, _, n in leaves),
                diffusion_leaf_samples=sum(n for _, n in selected),
                classification=dict(classification), inferred_pie_base=hex(base),
                agreeing_base_constraints=len(candidates),
                classification_method="addr2line source text; fort_t1/3/5_* identifies correction preparation/control; no conversion to seconds",
                worker_text_bytes=symbols[SYMBOL + "._omp_fn.0"][1],
                hottest_lines=[dict(source=p, line=l, text=t, samples=n)
                               for (p, l, t), n in by_line.most_common(20)])


def plot(report):
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "runtime-cost-breakdown-20261015"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1.05, 1, 1.25]})
    c = report["cuda_profile"]
    ax = axes[0]
    api, rest = c["api_nanoseconds"] / 1e9, c["other_owner_nanoseconds"] / 1e9
    ax.barh([0], [api], color="#26828e", label="CUDA API calls")
    ax.barh([0], [rest], left=[api], color="#f3b461", label="Other elapsed time")
    ax.text(api / 2, 0, f"{api:.3f} s", ha="center", va="center", color="white")
    ax.text(api + rest / 2, 0, f"{rest:.3f} s", ha="center", va="center")
    ax.set(yticks=[], xlabel="Seconds in 327 measured calls", title="Observed host interval\nCUDA-only profile", ylim=(-0.8, 0.8))
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.35), frameon=False, fontsize=9)
    ax = axes[1]
    copy = {x["kind"]: x for x in c["device_copies"]}
    # Kind spelling is supplied by Nsight's enum table, not guessed from IDs.
    uploads = copy["CUDA_MEMCPY_KIND_HTOD"]
    downloads = copy["CUDA_MEMCPY_KIND_DTOH"]
    values = [uploads["nanoseconds"] / 1e9, downloads["nanoseconds"] / 1e9,
              sum(x["nanoseconds"] for x in c["device_kernels"]) / 1e9]
    ax.barh([2, 1, 0], values, color=["#4477aa", "#66ccee", "#228833"])
    ax.set(yticks=[2, 1, 0], yticklabels=["Upload", "Download", "Kernels"],
           xlabel="Sum of device durations (s)", title="Device work\nOverlaps host API time", xlim=(0, 1.65))
    for y, x in zip([2, 1, 0], values):
        ax.text(x + .03, y, f"{x:.3f}", va="center")
    ax = axes[2]
    counts = report["cpu_samples"]["classification"]
    inspector = counts["Generated native-correction preparation/control"]
    total = report["cpu_samples"]["diffusion_leaf_samples"]
    values = [inspector, total - inspector]
    ax.barh([1, 0], values, color=["#cc6677", "#aaaaaa"])
    ax.set(yticks=[1, 0], yticklabels=["Generated correction\npreparation/control", "Other diffusion lines"],
           xlabel="CPU samples at diffusion instructions", title="Separate CPU profile\nSample counts, not elapsed time", xlim=(0, 13700))
    for y, n in zip([1, 0], values):
        ax.text(n + 120, y, f"{n:,}", va="center")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Why isolated GPU diffusion remains slow", fontsize=17, weight="bold", y=.98)
    fig.text(.5, .045, "Unchanged channel-simple • 109 steps per capture • 4 threads • no performance qualification\n"
             "Device times overlap API time; CPU samples do not assign seconds to the host remainder.", ha="center", fontsize=10)
    fig.subplots_adjust(left=.04, right=.98, bottom=.29, top=.78, wspace=.9)
    import io
    for extension in ("svg", "png"):
        buffer = io.BytesIO()
        metadata = {"Date": None} if extension == "svg" else {}
        fig.savefig(buffer, format=extension, dpi=160, metadata=metadata)
        freeze("cost-breakdown." + extension, buffer.getvalue())
    plt.close(fig)


def main():
    report = dict(schema_version=1, kind="diagnostic cost breakdown", automatic_placement=False,
        performance_qualification=False, performance_improvement_demonstrated=False,
        compiler_build_commit="dd8ea33d55328d7ea7919ae328440f6a047a2cdf",
        profiling_repository_commit="91ed5907c7eed59689fb47f915a182bfe6fca810",
        pipeline_commit="b9418a9f1c3c242c4b77e948b29fa391b94aee60",
        binary_sha256=sha(BINARY), previous_report_sha256=sha(PREVIOUS),
        cuda_profile=profile("cuda", RAW), cpu_profile=profile("cpu", RAW / "cpu"),
        cpu_samples=cpu_samples(), calibration=None,
        limitations=["Separate profiling processes; times are not qualification samples.",
                     "Device durations overlap CPU API durations.",
                     "CPU sampling identifies hot source lines, not their wall-clock duration.",
                     "No channel GPU qualification or automatic resident selector is demonstrated."])
    for path in ("compiler/scopes/indirect_access.py", "compiler/scopes/joined_team.py",
                 "compiler/frontend/indirect_sections.py"):
        freeze("evidence/compiler/" + Path(path).name, (ROOT / path).read_bytes())
    report["analysis_script_sha256"] = sha(Path(__file__))
    report["evidence_sha256"] = {str(p.relative_to(OUT)): sha(p) for p in sorted((OUT / "evidence").rglob("*")) if p.is_file()}
    document("report.json", report)
    plot(report)
    print(json.dumps({"profiles_complete": True, "cpu_classification": report["cpu_samples"]["classification"],
                      "report": str(OUT / "report.json")}))


if __name__ == "__main__":
    main()
