"""Source-only original-team preparation is shared with physical outlining."""

from copy import copy
from dataclasses import replace
from hashlib import sha256

import pytest
from fparser.two.utils import walk

from compiler._legacy.config import LegacyCompilerOptions as CompilerOptions
from compiler._legacy.config import LegacyOffloadConfig as OffloadConfig
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.numerical_preparation import NumericalPreparationProvider
from compiler.ir import CompilationError
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_joined_team_source import SOURCE
from compiler.tests.test_source_scopes import FACT


def fixture(directory, source=SOURCE):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "mixed.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: dict(FACT) for name in ("a", "b", "out")}}
    builder = ScopeBuilder([path], "joined_owner::step", facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="reached", host_threads=4))
    group, = [item.original_nodes for item in grouped_original_sequence(builder.entry.execution.content, {})
              if isinstance(item, NativeGroupCandidate)]
    loops = tuple(node for original in group for node in walk(original)
                  if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    provider = NumericalPreparationProvider(builder.analysis, options=builder.options, config=builder.config,
        environment_root=builder._numerical_environment_root)
    return path, builder, provider, group, loops


def test_source_only_mixed_team_keeps_native_correction_and_original_suffix(tmp_path, monkeypatch):
    _path, builder, provider, group, loops = fixture(tmp_path)

    def unexpected(*args, **kwargs):
        raise AssertionError("source preparation captured or emitted storage")

    monkeypatch.setattr(builder, "capture", unexpected)
    monkeypatch.setattr("compiler.scopes.region_dispatch.generate_sources", unexpected)
    with pytest.raises(CompilationError, match="array section requires bounded constant cardinality"):
        provider.prepare(builder.analysis, builder.entry.qualified, group)
    prepared = []
    for index, loop in enumerate(loops):
        proof = builder.analysis.worksharing_completion(builder.entry.qualified, group, (loop,))
        if index == 1:
            with pytest.raises(CompilationError, match="array section requires bounded constant cardinality"):
                provider.prepare_worksharing(builder.analysis, builder.entry.qualified, loop, proof)
        else:
            item = provider.prepare_worksharing(builder.analysis, builder.entry.qualified, loop, proof)
            assert item.validate(builder.analysis, builder.entry.qualified, (loop,)) is item
            assert item.completion_proof is proof
            assert item.offload.available
            assert not item.public()["execution_authorized"]
            assert not item.completion["native_effects_authority"]
            prepared.append(item)
    assert loops[1] in prepared[0].context.following
    assert loops[2] in prepared[0].context.following
    assert loops[0] in prepared[1].context.preceding
    assert not builder.inline.regions
    assert not builder.inline.generated
    assert not builder.outputs
    assert builder.inline.attempts == 0


def test_outlining_consumes_exact_shared_preparation_without_relowering(tmp_path, monkeypatch):
    _path, builder, provider, group, loops = fixture(tmp_path)
    proof = builder.analysis.worksharing_completion(builder.entry.qualified, group, (loops[0],))
    prepared = provider.prepare_worksharing(builder.analysis, builder.entry.qualified, loops[0], proof)

    def unexpected(*args, **kwargs):
        raise AssertionError("retained source preparation was lowered twice")

    monkeypatch.setattr(NumericalPreparationProvider, "prepare_worksharing", unexpected)
    monkeypatch.setattr("compiler.frontend.lower_source", unexpected)
    facade = builder.inline.prepare_worksharing(loops[0], proof, prepared=prepared)
    assert builder.inline.prepared_regions[facade.fort_inline_region] is prepared
    assert builder.inline.generated[facade.fort_inline_region].scoped["team"]["available"]
    assert builder.inline.attempts == 1
    call = builder.inline.call(facade)
    with pytest.raises(CompilationError, match="proven serial or original-team participation"):
        builder.inline.emit_call(call, {}, {}, [], query=False)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_any_unfinished_original_do_prevents_source_preparation(tmp_path, index):
    source = SOURCE.split("!$omp end do")
    source[index] += "!$omp end do nowait"
    changed = "".join(part if position == index else part + ("!$omp end do" if position < 3 else "")
                      for position, part in enumerate(source))
    _path, builder, _provider, group, loops = fixture(tmp_path, changed)
    for loop in (loops[0], loops[-1]):
        with pytest.raises(CompilationError, match="NOWAIT"):
            builder.analysis.worksharing_completion(builder.entry.qualified, group, (loop,))
    assert not builder.inline.regions


def test_foreign_copied_and_sibling_authority_fail_before_lowering(tmp_path, monkeypatch):
    _path, builder, provider, group, loops = fixture(tmp_path / "local")
    _other_path, other, _other_provider, other_group, other_loops = fixture(tmp_path / "foreign")
    proof = builder.analysis.worksharing_completion(builder.entry.qualified, group, (loops[0],))
    foreign = other.analysis.worksharing_completion(other.entry.qualified, other_group, (other_loops[0],))

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid proof reached numerical lowering")

    monkeypatch.setattr("compiler.frontend.lower_source", unexpected)
    for loop, token in ((loops[0], replace(proof)), (loops[0], proof.public()),
                        (loops[-1], proof), (loops[0], foreign)):
        with pytest.raises(CompilationError, match="original-team"):
            provider.prepare_worksharing(builder.analysis, builder.entry.qualified, loop, token)
    with pytest.raises(CompilationError, match="registered original policy authority"):
        copy(provider).prepare_worksharing(builder.analysis, builder.entry.qualified, loops[0], proof)
    assert not builder.inline.regions


def test_shared_preparation_rejects_copied_token_and_rechecks_source_at_publication(tmp_path):
    path, builder, provider, group, loops = fixture(tmp_path)
    proof = builder.analysis.worksharing_completion(builder.entry.qualified, group, (loops[0],))
    prepared = provider.prepare_worksharing(builder.analysis, builder.entry.qualified, loops[0], proof)
    with pytest.raises(CompilationError, match="registered exact original selection"):
        builder.inline.prepare_worksharing(loops[0], proof, prepared=copy(prepared))
    path.write_text(SOURCE.replace("b(i)=2*a(i)+real(i,8)", "b(i)=3*a(i)+real(i,8)"))
    # Preparations bind the immutable compilation session. Live source changes
    # are rechecked at publication, exactly as for complete original regions.
    assert prepared.validate(builder.analysis, builder.entry.qualified, (loops[0],)) is prepared
    with pytest.raises(CompilationError, match="changed"):
        builder.finish()
    assert not builder.inline.regions
    assert not builder.outputs
