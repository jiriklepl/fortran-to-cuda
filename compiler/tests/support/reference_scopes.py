"""Private subprocess adapter for historical source-proof regression fixtures.

Public command-line tests invoke ``python -m compiler`` directly. Older proof
fixtures use this adapter so their reference owners remain testable without
exposing removed execution strategies in the compiler interface.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from compiler._legacy.config import LegacyCompilerOptions, LegacyOffloadConfig
from compiler.driver.cli import _offload_config, _parse_args
from compiler.scopes.source import ScopeBuilder


def main():
    execution = "bounded"
    if "--scope-execution" in sys.argv:
        index = sys.argv.index("--scope-execution") + 1
        execution = sys.argv[index]
        if execution not in {"bounded", "reached"}:
            raise SystemExit("private reference adapter requires bounded or reached execution")
        sys.argv[index] = "graph"
    args = _parse_args()
    if not args.form_scopes:
        raise SystemExit("private reference adapter only accepts source-scope fixtures")
    options = LegacyCompilerOptions(
        opt_level=args.opt_level, schedule=args.schedule, tile_sizes=args.tile_sizes,
        fallback=args.fallback, indexing=args.indexing, gpu_policy=args.gpu_policy,
        memory_model=args.memory_model, scope_transfers=args.scope_transfers, scope_execution=execution)
    config = asdict(_offload_config(args))
    config["scope_execution"] = execution
    configuration = LegacyOffloadConfig(**config)
    parameters = {
        "facts": json.loads(Path(args.scope_facts).read_text()),
        "options": options, "config": configuration,
        "numerical_sources": json.loads(Path(args.numerical_sources).read_text()) if args.numerical_sources else None,
        "analysis_sources": json.loads(Path(args.analysis_sources).read_text()) if args.analysis_sources else None,
        "summary_cache": args.summary_cache,
    }
    if args.effect_contracts:
        parameters["contracts"] = json.loads(Path(args.effect_contracts).read_text())["procedures"]
    paths = [Path(args.input).resolve(), *args.source_file]
    if args.scope_preflight:
        from compiler.scopes.preflight import preflight_source_scopes

        report = preflight_source_scopes(paths, args.kernel, **parameters)
        print(json.dumps({"kernel": args.kernel, "supported": True, "outputs": [], "preflight": report.public()}))
        return
    builder = ScopeBuilder
    if configuration.collective:
        from compiler.scopes.collective import CollectiveScopeBuilder

        builder = CollectiveScopeBuilder
    outputs, manifest = builder(paths, args.kernel, **parameters).run()
    output = Path(args.output_dir)
    for name, content in outputs.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    print(json.dumps({"kernel": args.kernel, "supported": True, "outputs": list(outputs), "scopes": manifest}))


if __name__ == "__main__":
    main()
