"""Assemble the report and an offline HTML copy (requires markdown-it-py)."""
from pathlib import Path
import html
import json
import os
import re

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent


def section(directory):
    path = ROOT / directory / "section.md"
    text = path.read_text()
    headings = re.findall(r"^(#{1,5}) ", text, flags=re.MULTILINE)
    offset = 2 - min(map(len, headings))
    text = re.sub(r"^(#{1,5}) ", lambda m: "#" * (len(m[1]) + offset) + " ", text, flags=re.MULTILINE)

    def link(match):
        target = match.group(2)
        if target.startswith(("http:", "https:", "#", "/", "mailto:")):
            return match.group(0)
        file, marker, fragment = target.partition("#")
        relative = os.path.relpath(path.parent / file, ROOT)
        return match.group(1) + "(" + relative + (marker + fragment if marker else "") + ")"

    return re.sub(r"(!?\[[^\]]*\])\(([^)]+)\)", link, text)


snapshot = json.loads((ROOT / "snapshot/revisions.json").read_text())
repos = snapshot["repositories"]
introduction = """# ELMM hot paths, GPU residency and performance

**Snapshot: 10 November 2026.** The retained resident GPU implementation runs the main field computation and pressure projection on the GPU. Fortran still controls the time loop and RK stages, performs selected diagnostics, and serializes output. The latest seven-round measurements are **3.78× native for channel-simple (109 steps)** and **3.70× native for channel (100 steps)**, using four native threads and complete-process timing.

This report shows the reached hot path, representative original/generated/calling code, data movement, and historical performance. Diagrams deliberately summarize procedures and control regions; they are not exhaustive call graphs or CPU/GPU concurrency timelines. The timing tables separate measurement cohorts and profiler activity from untraced wall time. No new performance campaign was run to produce this report.

Read this file on GitHub, or open the **[standalone HTML report](report.html)** for embedded scalable diagrams, tables and code samples. Editable Mermaid and Graphviz sources are included beside each diagram.

"""
identity = "## Committed snapshot and qualification\n\n"
identity += "| Repository | Implementation commit | Scope |\n|---|---|---|\n"
for key, label, scope in [
    ("compiler", "Compiler/runtime", "Shared graph backend, resident memory, reductions, snapshots and current issue fixes"),
    ("pipeline", "ELMM pipeline", "Resident RK/pressure/wall/output adapters; explicit local provider"),
    ("solver", "Poisson solver", "Stream-aware device ABI, persistent kernel cache and FD2/FD4 qualification"),
]:
    identity += f"| {label} | `{repos[key]['commit']}` | {scope} |\n"
identity += """
These are independent local Git repositories; commits have **not been pushed**. The overview itself is committed afterward. [Snapshot identities and binary hashes](snapshot/revisions.json) also pin the companion parents. [Pipeline](snapshot/pipeline.patch) and [solver](snapshot/solver.patch) binary-safe patches preserve their changes in this repository; apply them only to the recorded parent revisions. Generated build/install products are excluded from source control.

The last implementation wrap passed **316 CPU/CUDA checks and 126 focused checks**. All 130 ELMM numerical entries regenerated with zero changed CUDA sources; all four executable identities match the selected asynchronous-output build. Existing strict primary and approved natural channel2458 evidence belongs to those exact binaries. The broader audit and its repaired failures are described in the [wrap report](../gpu-current-state-wrap-20261110/README.md); this does not claim a fresh full-suite qualification of the final source snapshot.

One small preparation fix was necessary for committing: the pipeline now accepts local solver commits **descended from** the pinned upstream base, rather than requiring `HEAD` to equal that base. Source snapshot hashes and rejection of unrelated history remain enforced. **34 provider tests pass**, including actual descendant/unrelated Git histories, and a snapshot of the committed solver verifies all 49 selected source files. This changes preparation, not numerical kernels or the measured executable. A trailing blank line and the editor's workspace-relative solver path were also tidied.

Correctness uses the approved deterministic pressure reference and four native threads. Primary/operator checks retain `atol=1e-6`, `rtol=1e-5`. Natural channel2458 uses the approved independent 1% relative L2 limit per field/output component, with exact layout, coordinate, frame-inventory and time checks. The retained long run's worst L2 discrepancy is 0.154896%; its worst local pressure discrepancy is 1.467061% of peak, so the long run is **not** a strict pointwise pass. See [operator qualification](../gpu-operator-parity-20261110/README.md) and [solver residuals](../gpu-solver-residual-20261110/README.md).

The 20× research target is unmet. Complete graph/driver/host-memory accounting, remaining private legacy coupling, final release qualification, changed-identity calibration and a fresh comparison of all historical controls with the latest build remain deferred. No additional research run was started for this snapshot.
"""

markdown = introduction + "\n\n".join(section(d) for d in ("hot-path", "data-management", "performance")) + "\n\n" + identity
(ROOT / "README.md").write_text(markdown)

renderer = MarkdownIt("commonmark", {"html": True}).enable("table")
tokens = renderer.parse(markdown)
toc = []
counts = {}
for index, token in enumerate(tokens):
    if token.type != "heading_open":
        continue
    title = tokens[index + 1].content
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    counts[slug] = counts.get(slug, 0) + 1
    key = slug if counts[slug] == 1 else f"{slug}-{counts[slug]}"
    token.attrSet("id", key)
    if token.tag == "h2":
        toc.append(f'<a href="#{key}">{html.escape(title)}</a>')

body = renderer.renderer.render(tokens, renderer.options, {})

def embed_svg(match):
    target = html.unescape(match.group(1))
    if not target.endswith(".svg") or ":" in target:
        return match.group(0)
    svg = (ROOT / target).read_text()
    svg = svg[svg.index("<svg"):]
    svg = re.sub(r"\s(?:width|height)=\"[^\"]+\"", "", svg, count=2)
    return f'<div class="diagram">{svg}</div><p class="download"><a href="{html.escape(target)}">Open or save this diagram as SVG</a></p>'

body = re.sub(r'<img src="([^"]+)"[^>]*>', embed_svg, body)
style = """
:root { color-scheme: light; --ink:#172b43; --muted:#52677d; --line:#dce4ed; }
* { box-sizing:border-box; } body { margin:0; background:#f3f6fa; color:var(--ink); font:16px/1.65 system-ui,sans-serif; }
main { max-width:1440px; margin:32px auto; padding:44px 56px; background:white; box-shadow:0 8px 32px #183b5910; }
h1 { font-size:2.4rem; line-height:1.15; margin:0 0 28px; } h2 { margin:56px 0 20px; padding-top:16px; border-top:2px solid var(--line); font-size:1.65rem; } h3 { margin-top:32px; font-size:1.25rem; }
a { color:#0864a7; } p { max-width:110ch; } nav { display:flex; flex-wrap:wrap; gap:12px 24px; padding:18px 24px; border:1px solid var(--line); background:#f7fafc; } nav a { font-weight:600; }
table { display:block; width:100%; overflow-x:auto; border-collapse:collapse; font-size:0.9rem; margin:24px 0; } th,td { border:1px solid var(--line); padding:10px 13px; text-align:left; vertical-align:top; } th { background:#eaf1f8; } tbody tr:nth-child(even) { background:#f8fafc; }
pre { overflow:auto; padding:20px; background:#10273b; color:#e9f2fb; border-radius:8px; font-size:0.82rem; line-height:1.55; } code { font-family:ui-monospace,monospace; overflow-wrap:anywhere; } :not(pre)>code { font-size:0.9em; background:#edf2f7; padding:1px 4px; border-radius:3px; }
.diagram { margin:28px 0 0; border:1px solid var(--line); padding:16px; overflow:auto; } .diagram svg { display:block; width:100%; min-width:720px; height:auto; } .download { margin:4px 0 28px; font-size:0.8rem; color:var(--muted); }
blockquote { border-left:4px solid #e6ad45; margin-left:0; padding-left:20px; color:var(--muted); } footer { margin-top:40px; color:var(--muted); font-size:0.8rem; }
@media(max-width:800px) { main { margin:0; padding:24px 18px; } h1 { font-size:1.9rem; } }
@media print { body { background:white; } main { margin:0; padding:0; box-shadow:none; } nav,.download { display:none; } pre { white-space:pre-wrap; } .diagram svg { min-width:0; } h2,h3 { break-after:avoid; } table { font-size:8pt; } }
"""
page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
page += '<title>ELMM hot paths, GPU residency and performance — 10 November 2026</title>'
page += '<style>' + style + '</style><main><nav>' + "".join(toc) + '</nav>' + body
page += '<footer>Offline report. Diagrams, measurements and excerpts are retained in the adjacent source files. No external scripts, fonts or services are required.</footer></main></html>'
(ROOT / "report.html").write_text(page)
print("Assembled README.md and report.html")
