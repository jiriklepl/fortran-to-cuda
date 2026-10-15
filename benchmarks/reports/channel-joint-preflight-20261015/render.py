"""Render the archived read-only decision gates; no analysis or workloads."""
import json
from html import escape
from pathlib import Path

root = Path(__file__).resolve().parent
report = json.loads((root / "report.json").read_text())
parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="680" viewBox="0 0 1120 680">',
         '<rect width="1120" height="680" fill="#ffffff"/>',
         '<style>text{font-family:DejaVu Sans,sans-serif;fill:#24364b}.title{font-size:27px;font-weight:700}.sub{font-size:17px}.body{font-size:16px}.small{font-size:14px}.name{font-size:20px;font-weight:700}</style>',
         '<text x="36" y="46" class="title">Generic fixtures passed. The application interval remains unproved.</text>',
         '<text x="36" y="80" class="sub">Two capability clusters · both channel targets unmet · no new execution or timing claim</text>']
lines = [
    ['Discover one useful original interval;', 'prove future resources, stage definitions and team composition.'],
    ['Prove native payload separation, sections and completion;', 'retain original conditions and exact lifetime throughout the interval.'],
    ['Bind the same alternatives to workload, build and actual outcome;', 'emit an automatic selector before original work.'],
    ['Validate both cases: complete fields and all transfer traffic;', 'then fresh native / SGS direct / automatic complete-process comparisons.'],
]
for index, (gate, detail) in enumerate(zip(report['gates'], lines, strict=True)):
    y = 110 + index * 109
    parts += [f'<rect x="36" y="{y}" width="1048" height="91" rx="10" fill="#fff1df" stroke="#c9aa80"/>',
              f'<text x="54" y="{y + 29}" class="name">{index + 1}. {escape(gate["name"])}</text>',
              f'<text x="1070" y="{y + 29}" class="body" text-anchor="end">{escape(gate["status"].replace("_", " ").upper())}</text>',
              f'<text x="54" y="{y + 53}" class="body">{escape(detail[0])}</text>',
              f'<text x="54" y="{y + 75}" class="body">{escape(detail[1])}</text>']
parts += ['<rect x="36" y="559" width="1048" height="72" rx="10" fill="#eaf0f6"/>',
          '<text x="54" y="586" class="name">Stop alias-only patches and full-Convection retries.</text>',
          '<text x="54" y="614" class="body">Any next implementation must address admission, residency and pricing together.</text>',
          f'<text x="36" y="660" class="small">Read-only preflight · source {report["commits"]["parent"][:12]} · earlier evidence remains bound to its own receipts</text>',
          '</svg>']
(root / 'gate-map.svg').write_text('\n'.join(parts) + '\n')
