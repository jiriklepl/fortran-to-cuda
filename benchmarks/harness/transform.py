"""Apply a pinned tool recipe without editing the benchmark's original source."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
from pathlib import Path

from benchmarks.harness.paths import SOURCES

TOOLS = ("loki", "psyclone")


def run(tool, input_path: Path, case: str, output: Path, *, target="openacc", fuse=True, vector_length=None):
    """Share file handling and provenance across pure source-to-source recipes."""
    if tool not in TOOLS:
        raise ValueError(f"tool must be one of {TOOLS}")
    if vector_length is not None and (tool != "loki" or vector_length < 1):
        raise ValueError("--vector-length must be positive and is only supported by Loki")
    manifest_path = output.with_suffix(".json")
    if input_path.resolve() in (output.resolve(), manifest_path.resolve()):
        raise ValueError("Output must differ from input; original sources stay unchanged")
    if output.resolve() == manifest_path.resolve():
        raise ValueError("Output must differ from its .json provenance file")
    recipe = importlib.import_module(f"benchmarks.tools.{tool}")
    original = input_path.read_bytes()
    generated, manifest = recipe.transform(
        original.decode(), case, target=target, fuse=fuse, vector_length=vector_length or 128
    )
    manifest.update(
        {
            "case": case,
            "target": target,
            "input": str(input_path.resolve()),
            "output": str(output.resolve()),
            "source_sha256": hashlib.sha256(original).hexdigest(),
            "generated_sha256": hashlib.sha256(generated.encode()).hexdigest(),
            "python": platform.python_version(),
            "dependencies": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
            "original_source_edits": 0,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generated)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", choices=TOOLS, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--case", choices=SOURCES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=("cpu", "openacc"), default="openacc")
    parser.add_argument("--no-fuse", action="store_true", help="retain four loops")
    parser.add_argument("--vector-length", type=int, help="Loki OpenACC vector length (default: 128)")
    args = parser.parse_args()
    if args.vector_length is not None and (args.tool != "loki" or args.vector_length < 1):
        parser.error("--vector-length must be positive and is only supported by Loki")
    run(
        args.tool,
        args.input,
        args.case,
        args.output,
        target=args.target,
        fuse=not args.no_fuse,
        vector_length=args.vector_length,
    )


if __name__ == "__main__":
    main()
