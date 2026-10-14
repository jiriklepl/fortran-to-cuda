"""Freeze and render independent cold-call fixture evidence; never run a workload."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CALIBRATION = "complete-original-window-cold-v1"
PRODUCTION = "complete-original-window-cold-production-validation-v1"
INK = "#203047"
BLUE = "#1677a8"
GREY = "#64748b"
GREEN = "#21805a"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def samples(rows, alternate):
    """Validate rotated pairs, independent holdouts, and complete field hashes."""
    require(isinstance(rows, list) and len(rows) == 18, "seven measured pairs and two holdout pairs required")
    values = {"native": [], alternate: []}
    for position, row in enumerate(rows):
        round_, turn = divmod(position, 2)
        mode = ("native", alternate)[(round_ + turn) % 2]
        require(row.get("round") == round_ and row.get("mode") == mode
                and row.get("role") == ("fit" if round_ < 7 else "holdout"), "invalid rotated observation order")
        seconds = row.get("seconds")
        require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
                "invalid measured cold-call duration")
        require(row.get("agreement") is True and re.fullmatch(r"[0-9a-f]{64}", row.get("fields_sha256", "")),
                "complete float64 field agreement required")
        if turn:
            previous = rows[position - 1]
            require(row["fields_sha256"] == previous["fields_sha256"], "paired field hashes differ")
            require(row["observed_cpu_identity"] == previous["observed_cpu_identity"]
                    and row["observed_key"] == previous["observed_key"], "paired execution identities differ")
        values[mode].append(seconds)
    return values


def summarize(values):
    return {mode: {"median": statistics.median(data[:7]), "min": min(data), "max": max(data),
                   "fit": data[:7], "holdout": data[7:]} for mode, data in values.items()}


def conservative_advantage(values, alternate):
    return max(values[alternate]) < .95 * min(values["native"])


def semantic_fortran_options(raw):
    tokens, skip = [], False
    for token in shlex.split(raw):
        if skip:
            skip = False
        elif token in {"-I", "-J", "-o"}:
            skip = True
        elif token.startswith(("-openmp", "-offload", "-opt")) or not any(
                token.startswith(flag) and len(token) > len(flag) for flag in ("-I", "-J", "-o")):
            tokens.append(token)
    require(not skip, "incomplete Fortran location option")
    return "\x1f".join(tokens)


def observation_logs(root, size, rows, *, production):
    """Retain actual timer output and exclude runtime traces from timing data."""
    names = []
    for row in rows:
        stem = ("production-" if production else "") + f"{row['round']:02d}-{row['mode']}"
        output, errors = (f"size-{size}/{stem}.{suffix}" for suffix in ("stdout", "stderr"))
        stdout, stderr = (root / output).read_text(), (root / errors).read_text()
        timings = re.findall(r"^WINDOW_SECONDS\s+(\S+)$", stdout, re.MULTILINE)
        require(len(timings) == 1 and float(timings[0]) == row["seconds"], "raw timer output differs from observation")
        require("FORT_SCOPED " not in stderr, "runtime trace cannot be a timing sample")
        captured = [json.loads(line.removeprefix("FORT_EXACT_OBSERVATION ")) for line in stderr.splitlines()
                    if line.startswith("FORT_EXACT_OBSERVATION ")]
        if production or row["mode"] == "native":
            require(not captured, "production/native timing unexpectedly uses calibration selector")
        else:
            require(captured == [{"identity": row["identity"], "key": row["key"]}],
                    "calibration selector identity differs from raw stderr")
        names += [output, errors]
    return names


def ledger(root, size, label, report_ledger, *, resident):
    directory = root / f"size-{size}"
    stored = read(directory / f"{label}-ledger.json")
    require(stored == report_ledger, "diagnostic ledger differs from report: " + label)
    trace = directory / f"{label}.stderr"
    require(sha(trace) == stored["trace_sha256"], "diagnostic trace hash differs: " + label)
    text = trace.read_text()
    events = re.findall(r"^FORT_SCOPED (upload|download|launch) .*?bytes=(\d+)", text, re.MULTILINE)
    observed = {kind + "_bytes": sum(int(count) for event, count in events if event == kind)
                for kind in ("upload", "download")}
    observed.update({kind + ("_count" if kind == "launch" else "_trace_events"):
                     sum(event == kind for event, _ in events) for kind in ("upload", "download", "launch")})
    require(stored["observed"] == observed, "ledger is not reproduced by its complete trace: " + label)
    interior = (size - 2) ** 3
    predicted = {"upload_bytes": 8 * (size**3 + interior + 1),
                 "download_bytes": 8 * (size**3 + 2 * interior), "launch_count": 3}
    if not resident:
        predicted = dict.fromkeys(predicted, 0)
    require(stored["predicted"] == predicted and all(observed[k] == v for k, v in predicted.items()),
            "complete transfer prediction differs from trace: " + label)
    if resident:
        require(re.search(r"^FORT_SCOPED close\b", text, re.MULTILINE),
                "resident trace lacks completed owner close")
    return stored


def inspect(root, *, verify_builds=False):
    """Reconstruct plotting summaries from raw observations and separate traces."""
    root = Path(root)
    report = read(root / "report.json")
    require(isinstance(report, list) and report, "completed fixture report required")
    sizes = [row["size"] for row in report]
    require(len(sizes) == len(set(sizes)) and all(type(n) is int and n > 2 for n in sizes), "invalid fixture sizes")
    files = {"report.json", "original.f90", "calibration/build-receipt.json", "native/build-receipt.json"}
    validated = []
    for result in report:
        size = result["size"]
        directory = root / f"size-{size}"
        evidence = read(directory / "evidence.json")
        production = read(directory / "production-observations.json")
        require(evidence.get("protocol") == CALIBRATION and production.get("protocol") == PRODUCTION,
                "independent calibration and production-validation protocols required")
        fit_values = samples(evidence["observations"], "resident")
        production_values = samples(production["observations"], "automatic")
        identity = evidence["identity"]
        expected_cpu = {key: identity[key] for key in ("cpu_name", "fortran_version", "affinity", "environment")}
        expected_cpu.update(fortran_options=semantic_fortran_options(identity["fortran_options"]),
                            team=[identity["threads"], identity["dynamic"], identity["proc_bind"]])
        reference_fields = {row["round"]: row["fields_sha256"] for row in evidence["observations"]
                            if row["mode"] == "native"}
        for row in evidence["observations"]:
            require(row["identity"] == identity and row["key"] == evidence["key"],
                    "calibration observation identity differs from its contract")
        for row in [*evidence["observations"], *production["observations"]]:
            require(row["observed_cpu_identity"] == expected_cpu and row["observed_key"] == evidence["key"]
                    and row["fields_sha256"] == reference_fields[row["round"]],
                    "independent observation identity or fields differ from calibration reference")
        files.update(observation_logs(root, size, evidence["observations"], production=False))
        files.update(observation_logs(root, size, production["observations"], production=True))
        fit, final = summarize(fit_values), summarize(production_values)
        selection = "resident" if conservative_advantage(fit_values, "resident") else "native"
        decision = result["decision"]
        require(decision.get("available") is True and decision["selection"] == selection,
                "automatic selection differs from the conservative calibration rule")
        require(decision["evidence_sha256"] == object_sha(evidence), "calibration evidence digest differs")
        require(decision["native_seconds"] == fit["native"]["median"]
                and decision["resident_seconds"] == fit["resident"]["median"], "calibration medians differ")
        require(result["production_validation"] == production, "production observations differ from report")
        profitable = conservative_advantage(production_values, "automatic")
        require(production["native_seconds"] == final["native"]["median"]
                and production["automatic_seconds"] == final["automatic"]["median"]
                and production["automatic_profitable"] is profitable
                and production["predicted_selection"] == selection
                and production["model_miss"] is (selection == "resident" and not profitable),
                "production validation summary differs from its raw samples")
        require(result.get("automatic_execution_verified") is True, "automatic execution verification required")
        expected_fields = evidence["observations"][-1]["fields_sha256"]
        diagnostics = {}
        for label, field, resident in (
            ("resident-diagnostic", "resident_diagnostic", True),
            ("automatic", "automatic_diagnostic", selection == "resident"),
            ("automatic-thread-mismatch", "thread_mismatch_diagnostic", False),
        ):
            diagnostics[label] = ledger(root, size, label, result[field], resident=resident)
            require(diagnostics[label]["fields_sha256"] == expected_fields,
                    "diagnostic complete fields differ from independent reference")
            files.update(f"size-{size}/{label}{suffix}" for suffix in (".stderr", ".stdout", "-ledger.json"))
        for kind, relative in (("calibration_build", "calibration/build-receipt.json"),
                               ("native_build", "native/build-receipt.json"),
                               ("automatic_build", f"size-{size}/automatic/build-receipt.json")):
            receipt = read(root / relative)
            require(receipt == result[kind], "build receipt differs from report: " + kind)
            if verify_builds:
                require(sha(receipt["binary"]) == receipt["binary_sha256"], "compiled binary changed")
                for path, expected in receipt["objects"].items():
                    require(sha(path) == expected, "compiled object changed: " + path)
            files.add(relative)
        require(sha(root / "original.f90") in evidence["contract"]["sources"].values(), "original source changed")
        require(evidence["contract"]["build_contract"]["consumer"] == "independent exact_window_calibration producer only",
                "fixture-only consumer boundary required")
        files.update((f"size-{size}/evidence.json", f"size-{size}/production-observations.json"))
        validated.append({"size": size, "selection": selection, "calibration": fit, "production": final,
                          "production_profitable": profitable, "model_miss": production["model_miss"],
                          "diagnostics": diagnostics, "contract": evidence["contract"]})
    return validated, sorted(files)


def committed_identity(root, paths):
    root = Path(root).resolve(strict=True)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    tracked = set(subprocess.check_output(["git", "-C", str(root), "ls-tree", "-r", "--name-only", commit], text=True).splitlines())
    require(set(paths) <= tracked, "measurement implementation contains uncommitted files")
    # Runtime headers/templates also affect emitted workers; checking only the
    # Python inventory would mislabel an uncommitted runtime change as HEAD.
    changed = subprocess.check_output(["git", "-C", str(root), "diff", "--name-only", commit, "--", "compiler"], text=True)
    require(not changed, "measurement implementation differs from its recorded commit")
    for name, expected in paths.items():
        require(sha(root / name) == expected, "measured implementation differs from checkout: " + name)
    return commit


def freeze(artifacts, output, compiler_root, pipeline_root):
    artifacts = Path(artifacts).resolve(strict=True)
    output = Path(output).resolve()
    cases, files = inspect(artifacts, verify_builds=True)
    implementations = [{"compiler/" + name: value for name, value in case["contract"]["implementation"].items()}
                       for case in cases]
    require(all(item == implementations[0] for item in implementations), "cases use different compiler implementations")
    compiler_commit = committed_identity(compiler_root, implementations[0])
    pipeline_commit = subprocess.check_output(["git", "-C", str(pipeline_root), "rev-parse", "HEAD"], text=True).strip()
    hashes = {name: sha(artifacts / name) for name in files}
    output.mkdir(parents=True, exist_ok=False)
    for name in files:
        target = output / "raw" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(artifacts / name, target)
        require(sha(target) == hashes[name] == sha(artifacts / name), "evidence changed during curation")
    write(output / "snapshot.json", {
        "schema_version": 1, "kind": "independent_generic_cold_window_fixture", "source_artifacts": str(artifacts),
        "compatible_commits": {"compiler": compiler_commit, "pipeline": pipeline_commit},
        "pipeline_role": "application-consumer exclusion boundary; not used to build this fixture",
        "previous_source_preflight_commit": "d3c18bf", "raw_files_sha256": hashes,
        "renderer_sha256": sha(__file__), "application_qualification": False,
        "timing_scope": "one original cold subroutine invocation; excludes process startup and field-file output",
        "qualification_targets": None, "sgs_control_measured": False,
        "cases": [{key: value for key, value in case.items() if key != "contract"} for case in cases],
    })
    return output


def save(figure, output, name):
    import matplotlib.pyplot as plt
    svg = output / (name + ".svg")
    figure.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    figure.savefig(output / (name + ".png"), bbox_inches="tight", dpi=160, metadata={"Software": "matplotlib"})
    plt.close(figure)


def timing_plot(cases, output, *, relative, calibration=False):
    import matplotlib.pyplot as plt
    import numpy as np
    field, alternate = ("calibration", "resident") if calibration else ("production", "automatic")
    figure, axes = plt.subplots(1, len(cases), figsize=(6.4 * len(cases), 6.6), squeeze=False, sharey=relative)
    for axis, case in zip(axes[0], cases, strict=True):
        rows = case[field]
        baseline = rows["native"]["median"] if relative else 1
        modes = ["native", alternate]
        medians = [rows[mode]["median"] / baseline for mode in modes]
        errors = [[(rows[mode]["median"] - rows[mode]["min"]) / baseline for mode in modes],
                  [(rows[mode]["max"] - rows[mode]["median"]) / baseline for mode in modes]]
        axis.bar([0, 1], medians, color=[GREY, BLUE], width=.55, alpha=.8, yerr=errors, capsize=4)
        for position, mode in enumerate(modes):
            row = rows[mode]
            for role, marker, face in (("fit", "o", "white"), ("holdout", "D", "#f9c74f")):
                values = row[role]
                axis.scatter(position + np.linspace(-.17, .17, len(values)), np.array(values) / baseline,
                             marker=marker, color=face, edgecolor=INK, zorder=3, s=34,
                             label=("Seven paired samples" if role == "fit" else "Two independent holdouts")
                             if position == 0 else None)
        if calibration:
            axis.axhline(.95 * rows["native"]["min"] / baseline, color=GREEN, linestyle="--", linewidth=1.2,
                         label="95% of fastest calibration native call")
        elif relative:
            axis.axhline(1, color=GREY, linewidth=1)
        axis.set_xticks([0, 1], ["Pristine native", "Diagnostic resident" if calibration else "Production AUTO"])
        axis.set_title(f"{case['size']}³ producer interior\nAUTO selection: {case['selection']}", fontsize=13)
        axis.set_ylabel("Cold-call time / native median (seven runs)" if relative else "Cold-call time (seconds)")
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        axis.legend(loc="best", fontsize=8)
        axis.text(.5, -.15, f"Median: {rows['native']['median']:.6f} s native; {rows[alternate]['median']:.6f} s {alternate}",
                  transform=axis.transAxes, ha="center", fontsize=9)
    for axis in axes[0]:
        axis.set_ylim(bottom=0)
    title = "Diagnostic placement calibration" if calibration else "Independent production AUTO validation"
    figure.suptitle(title + " · generic fixture", fontsize=17, y=.99)
    figure.text(.5, .025, "Bars: medians of seven samples. Whiskers: all nine runs. Diamonds: independent holdouts.\n"
                "One cold original call per process; startup and field-file output excluded.\n"
                "No ELMM application result, SGS comparison, or application-target qualification.",
                ha="center", fontsize=10, linespacing=1.4)
    figure.subplots_adjust(bottom=.29, top=.84, left=.08, right=.98, wspace=.25)
    name = "calibration-costs" if calibration else "production-wall-" + ("relative" if relative else "absolute")
    save(figure, output, name)


def transfer_plot(cases, output):
    import matplotlib.pyplot as plt
    import numpy as np
    figure, axes = plt.subplots(1, len(cases), figsize=(6.4 * len(cases), 5.8), squeeze=False)
    for axis, case in zip(axes[0], cases, strict=True):
        ledger = case["diagnostics"]["resident-diagnostic"]
        automatic = case["diagnostics"]["automatic"]
        positions = np.arange(4)
        for offset, label, color in ((-.17, "Predicted", GREY), (.17, "Measured", BLUE)):
            field = "predicted" if label == "Predicted" else "observed"
            values = [entry[field][key] / 2**20 for entry in (ledger, automatic)
                      for key in ("upload_bytes", "download_bytes")]
            axis.bar(positions + offset, values, width=.32, label=label, color=color, alpha=.85)
        observed = ledger["observed"]
        auto = automatic["observed"]
        axis.set_title(f"{case['size']}³ · AUTO selects {case['selection']}\n"
                       f"Resident: {observed['launch_count']} launches, "
                       f"{observed['upload_trace_events']} upload / {observed['download_trace_events']} download events\n"
                       f"AUTO: {auto['launch_count']} launches, "
                       f"{auto['upload_trace_events']} upload / {auto['download_trace_events']} download events", fontsize=10)
        axis.set_xticks(positions, ["Resident\nupload", "Resident\ndownload", "AUTO\nupload", "AUTO\ndownload"])
        axis.set_ylabel("Total transferred MiB, including owner close")
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        axis.legend(fontsize=9)
    figure.suptitle("Predicted and observed complete transfer totals", fontsize=16, y=.98)
    figure.text(.5, .03, "Separate diagnostic resident and production AUTO traces; durations never enter performance plots.\n"
                "Upload/download trace events count CopyPlan/Box records, not individual CUDA copy API calls.\n"
                "Includes entry traffic, native-cell correction, final publication and cleanup.", ha="center", fontsize=10)
    figure.subplots_adjust(bottom=.25, top=.76, left=.08, right=.98, wspace=.3)
    save(figure, output, "transfer-totals")


def ownership_map(cases, output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    figure, axis = plt.subplots(figsize=(13, 10))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    axis.text(0, 1.02, "One original call: ownership and transfer edges", fontsize=19, weight="bold", color=INK)
    choices = "; ".join(f"n={case['size']}: {case['selection']}" for case in cases)
    axis.text(0, .99, "Resident execution map · original numerical order · no ELMM scheduling claim\n"
              "Production AUTO choices: " + choices, fontsize=10, color=INK, linespacing=1.4, va="top")
    rows = [
        ("1  GPU producer", "Upload a interior: 8 n³ bytes\na → b; b remains device-current", BLUE),
        ("2  Native reader", "audit = sum(a(:,1,1)); overlaps the uploaded a section\nCurrent host mirror: zero download; device a remains valid", GREY),
        ("3  Native correction", "Read b(2,2,2): download 8 bytes; then update that cell\nOnly the written cell loses device validity", GREY),
        ("4  GPU diffusion", "Reupload corrected b cell: 8 bytes; read retained a and b\nWrite d interior of (n−2)³ cells; a is reused after the native reader", BLUE),
        ("5  GPU accumulation", "Upload out interior: 8 (n−2)³ bytes\nout += 0.125 d; retain results until publication", BLUE),
        ("6  Owner close", "Publish remaining b plus d/out interiors\nDownload 8 [n³−1+2 (n−2)³] bytes; preserve original halos", GREEN),
    ]
    top = .93
    for index, (label, text, color) in enumerate(rows):
        bottom = top - .108
        patch = FancyBboxPatch((.02, bottom), .95, .096, boxstyle="round,pad=0.008", edgecolor=color,
                              facecolor="#eef6fa" if color == BLUE else "#edf5f0" if color == GREEN else "#f1f3f6")
        axis.add_patch(patch)
        axis.text(.045, bottom + .061, label, weight="bold", color=color, fontsize=11)
        axis.text(.30, bottom + .046, text, color=INK, fontsize=10.3, va="center", linespacing=1.4)
        if index < len(rows) - 1:
            axis.annotate("", xy=(.49, bottom - .023), xytext=(.49, bottom - .007),
                          arrowprops={"arrowstyle": "->", "color": GREY, "linewidth": 1.1})
        top = bottom - .026
    axis.text(.02, .105, "Predicted complete traffic: upload 8 [n³+(n−2)³+1]; download 8 [n³+2 (n−2)³] bytes.",
              fontsize=10.5, color=INK)
    totals = [f"n={case['size']}: {case['diagnostics']['resident-diagnostic']['observed']['upload_bytes']:,} B up / "
              f"{case['diagnostics']['resident-diagnostic']['observed']['download_bytes']:,} B down" for case in cases]
    axis.text(.02, .065, "Measured separate traces: " + "; ".join(totals), fontsize=10, color=INK)
    axis.text(.02, .02, "Trace-event counts are not CUDA copy-call counts. Full float64 array bytes, including halos, agree.",
              fontsize=10, color=INK)
    save(figure, output, "ownership-map")


def render(evidence, output):
    evidence, output = Path(evidence).resolve(strict=True), Path(output).resolve()
    snapshot = read(evidence / "snapshot.json")
    require(snapshot["kind"] == "independent_generic_cold_window_fixture", "fixture snapshot required")
    for name, expected in snapshot["raw_files_sha256"].items():
        require(sha(evidence / "raw" / name) == expected, "frozen evidence changed: " + name)
    cases, _files = inspect(evidence / "raw")
    require(snapshot["renderer_sha256"] == sha(__file__), "use the frozen renderer or create a new curated snapshot")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    output.mkdir(parents=True, exist_ok=False)
    with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": "exact-window-20261014",
                         "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False}):
        timing_plot(cases, output, relative=False)
        timing_plot(cases, output, relative=True)
        timing_plot(cases, output, relative=False, calibration=True)
        transfer_plot(cases, output)
        ownership_map(cases, output)
    write(output / "source.json", {"snapshot_sha256": sha(evidence / "snapshot.json"),
          "renderer_sha256": sha(__file__), "compatible_commits": snapshot["compatible_commits"],
          "plotting_versions": {"python": sys.version, "matplotlib": matplotlib.__version__, "numpy": np.__version__},
          "files_sha256": {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()},
          "rendering_only": True, "application_qualification": False})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    curate = subcommands.add_parser("freeze")
    curate.add_argument("--artifacts", required=True, type=Path)
    curate.add_argument("--compiler-root", required=True, type=Path)
    curate.add_argument("--pipeline-root", required=True, type=Path)
    curate.add_argument("--output", type=Path, default=HERE / "evidence")
    draw = subcommands.add_parser("render")
    draw.add_argument("--evidence", type=Path, default=HERE / "evidence")
    draw.add_argument("--output", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze(args.artifacts, args.output, args.compiler_root, args.pipeline_root))
    else:
        print(render(args.evidence, args.output))


if __name__ == "__main__":
    main()
