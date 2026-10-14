"""Render compiler scope preflight evidence without implying GPU execution.

Usage: python -m benchmarks.harness.plot_preflight INPUT --output NEW_DIRECTORY
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path

PHASES = ("native", "numerical", "cost", "ownership", "completion")
COLORS = {"available": "#3b986d", "unavailable": "#cc6358", "unknown": "#c3cbd5"}


def checked_document(path):
    """Keep diagnostic availability separate from execution authority."""
    raw = Path(path).read_bytes()
    document = json.loads(raw)
    if (
        document.get("schema_version") != 1
        or document.get("execution_authorized") is not False
        or not isinstance(document.get("operations"), list)
    ):
        raise ValueError(
            "version 1 scope preflight with no execution authority required"
        )
    ordinals = set()
    for operation in document["operations"]:
        ordinal = operation.get("ordinal")
        if type(ordinal) is not int or ordinal in ordinals:
            raise ValueError("unique integer operation ordinals required")
        ordinals.add(ordinal)
        for phase in PHASES:
            if not isinstance(operation.get(phase), dict) or not isinstance(
                operation[phase].get("status"), str
            ):
                raise TypeError("explicit phase status required: " + phase)
        transfers = operation.get("transfers", {})
        if transfers.get("status") == "unknown" and any(
            transfers.get(key) is not None for key in ("upload_bytes", "download_bytes")
        ):
            raise ValueError("unknown transfers cannot report numerical byte totals")
    return document, hashlib.sha256(raw).hexdigest()


def phase_state(operation, phase):
    status = operation[phase]["status"]
    return status if status in {"available", "unavailable"} else "unknown"


def render(source, output):
    document, source_hash = checked_document(source)
    output = Path(output)
    if output.exists():
        raise ValueError("visual evidence requires a new output directory")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/residency-preflight-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    groups = {}
    for operation in document["operations"]:
        groups.setdefault(operation["procedure"], []).append(operation)
    output.mkdir(parents=True)
    artifacts = []
    figures = []
    for group_index, (procedure, operations) in enumerate(groups.items(), 1):
        for page, start in enumerate(range(0, len(operations), 24), 1):
            selected = operations[start : start + 24]
            fig, ax = plt.subplots(figsize=(13, max(4.2, len(selected) * 0.4 + 2.4)))
            labels = []
            for row, operation in enumerate(selected):
                span = operation.get("span", [])
                location = str(span[0]) if span else "?"
                labels.append(
                    f"{operation['ordinal']:03d} · line {location} · {operation['kind']}"
                )
                for column, phase in enumerate(PHASES):
                    state = phase_state(operation, phase)
                    ax.add_patch(
                        Rectangle(
                            (column - 0.46, row - 0.4),
                            0.92,
                            0.8,
                            color=COLORS[state],
                            linewidth=0,
                        )
                    )
                    ax.text(
                        column,
                        row,
                        operation[phase]["status"].replace("_", " "),
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color="white" if state != "unknown" else "#253044",
                    )
            ax.set(xlim=(-0.5, len(PHASES) - 0.5), ylim=(len(selected) - 0.5, -0.5))
            ax.set_xticks(range(len(PHASES)), [name.capitalize() for name in PHASES])
            ax.xaxis.tick_top()
            ax.set_yticks(range(len(selected)), labels)
            ax.tick_params(length=0, pad=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.suptitle(procedure + " · scope preflight", fontsize=14, y=0.98)
            fig.legend(
                handles=[
                    Patch(color=COLORS[key], label=label)
                    for key, label in (
                        ("available", "Available proof/estimate"),
                        ("unavailable", "Unavailable"),
                        ("unknown", "Unknown / not requested / not applicable"),
                    )
                ],
                loc="lower center",
                bbox_to_anchor=(0.5, 0.08),
                ncol=3,
                frameon=False,
                fontsize=9,
            )
            fig.text(
                0.5,
                0.025,
                "Original order within this procedure; branches are not evaluated. "
                "Availability is not ownership or execution approval.\n"
                "No measured transfers or performance samples are implied. See HTML for guards, resources and reasons.",
                ha="center",
                fontsize=8,
            )
            fig.subplots_adjust(
                left=0.27,
                right=0.985,
                top=1 - 0.85 / fig.get_figheight(),
                bottom=1.2 / fig.get_figheight(),
            )
            stem = f"scope-{group_index:02d}-{page:02d}"
            for suffix in ("svg", "png"):
                name = stem + "." + suffix
                fig.savefig(output / name, dpi=160)
                artifacts.append(name)
            figures.append((procedure, stem + ".svg"))
            plt.close(fig)

    escape = lambda value: html.escape(str(value), quote=True)
    rows = []
    for operation in document["operations"]:
        details = []
        for phase in PHASES:
            reasons = operation[phase].get("reasons", [])
            details.append(
                "<li><b>"
                + escape(phase)
                + ": "
                + escape(operation[phase]["status"])
                + "</b> "
                + escape("; ".join(reasons))
                + "</li>"
            )
        resource_names = [item["resource"] for item in operation.get("resources", [])]
        rows.append(
            "<tr><td>"
            + escape(operation["ordinal"])
            + "</td><td>"
            + escape(operation["procedure"])
            + "<br>"
            + escape(operation.get("span", []))
            + "<br>"
            + escape(operation["kind"])
            + "</td><td>"
            + escape("; ".join(operation.get("guard", [])))
            + "<br>"
            + escape(", ".join(resource_names))
            + "</td><td><ul>"
            + "".join(details)
            + "</ul></td></tr>"
        )
    contents = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Scope preflight</title>
<style>body{font:15px system-ui;max-width:1400px;margin:2rem auto;padding:0 1rem;color:#253044}
table{border-collapse:collapse;width:100%}td,th{padding:.65rem;text-align:left;border-bottom:1px solid #ccd3dd;vertical-align:top}
img{width:100%;height:auto}ul{margin:0;padding-left:1.1rem}code{overflow-wrap:anywhere}</style>
<h1>Scope preflight</h1><p>Source diagnostics only. No owner is admitted and no application or GPU execution is established.
Operations retain source order within each procedure; procedure groups do not describe dynamic call order.
Transfer totals are unknown until a checked plan and runtime observations exist.</p>"""
    if document.get("truncated"):
        contents += "<p><strong>Analysis budget reached: this inventory is incomplete.</strong></p>"
    contents += "<p>Input SHA-256: <code>" + source_hash + "</code></p>"
    contents += "".join(
        "<h2>"
        + escape(name)
        + '</h2><img src="'
        + escape(path)
        + '" alt="'
        + escape(name + " preflight availability map")
        + '">'
        for name, path in figures
    )
    contents += (
        "<h2>Operation evidence</h2><table><thead><tr><th>Operation</th><th>Source</th>"
        "<th>Guards and resources</th><th>Availability and reasons</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></html>"
    )
    (output / "index.html").write_text(contents)
    artifacts.append("index.html")
    # Detect edits during plotting before publishing a successful receipt.
    if hashlib.sha256(Path(source).read_bytes()).hexdigest() != source_hash:
        raise ValueError("preflight source changed during rendering")
    receipt = {
        "schema_version": 1,
        "input_sha256": source_hash,
        "snapshot_identity": document.get("snapshot_identity"),
        "execution_authorized": False,
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifacts_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in artifacts
        },
    }
    (output / "source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(args.input, args.output)


if __name__ == "__main__":
    main()
