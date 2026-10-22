"""Explicit historical compiler transport for frozen controls and proof fixtures.

This is not a public placement or source-execution option. Current application
generation must import ``local_adapter.transform_scopes`` instead.
"""

from .local_adapter import _transform_scopes_protocol


def transform_scopes(source_path, compiler_root, python, output_dir, *, entry,
                     source_files=(), capture_facts, gpu_options=None,
                     effect_contracts=None, numerical_sources=None,
                     analysis_sources=None, summary_cache=None):
    return _transform_scopes_protocol(
        source_path, compiler_root, python, output_dir, entry=entry,
        source_files=source_files, capture_facts=capture_facts, gpu_options=gpu_options,
        effect_contracts=effect_contracts, numerical_sources=numerical_sources,
        analysis_sources=analysis_sources, summary_cache=summary_cache,
        legacy_protocol=True)
