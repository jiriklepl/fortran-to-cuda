"""Shared original ENTRY definitions have exact transactional ownership."""
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes import companion_emission
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_owner_local_call_groups_runtime import SOURCE
from compiler.tests.test_prepared_handoff_aliases import compile_published_fortran


def builder_for(tmp_path):
    calls = "call first(b,c,n)\ncall second(a,b,c,n)\ncall third(a,c,n)"
    source = SOURCE.replace(calls+"\nendif", calls+"\nelse\n"+calls+"\nendif")
    path = tmp_path/"original.f90"
    path.write_text(source)
    fact = {"storage": "stable", "initialized": "whole", "escapes": False,
            "allocation_changes": False, "lifetime": "invocation"}
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::"+name: dict(fact) for name in ("a", "b", "c")}}
    contracts = {"live_groups::run": {"native_ancestor": {"version": 2,
        "lifetime": "owner_local_call_groups", "device_roots": ["live_groups::"+name
            for name in ("first", "second", "third")]}}}
    return ScopeBuilder([path], "live_groups::run", facts=facts, contracts=contracts,
                        options=CompilerOptions(), config=OffloadConfig(policy="sections"))


@pytest.mark.parametrize("mutation", ["source_bytes", "control_signature", "missing_published_edits"])
def test_conflicting_shared_companion_rejects_only_the_later_candidate(tmp_path, monkeypatch, mutation):
    original = companion_emission.publish_original_companion
    reached = []

    def changed(member, edits):
        builder = member.owner.builder
        key = str(member.routine.scope.path), member.entry
        if member.routine.qualified == "live_groups::first" and key in builder._module_companion_publications:
            reached.append(mutation)
            if mutation == "source_bytes":
                edits = list(edits)
                path, first, last, text, prepend = edits[0]
                edits[0] = path, first, last, text+"\n! conflicting checked definition\n", prepend
            elif mutation == "control_signature":
                object.__setattr__(member, "context", member.context+"_different")
            else:
                previous = builder._module_companion_publications[key][1][0]
                path, first, last, text, _prepend = previous
                builder.edits[path].remove((first, last, text))
        return original(member, edits)

    monkeypatch.setattr(companion_emission, "publish_original_companion", changed)
    builder = builder_for(tmp_path)
    outputs, report = builder.run()
    assert reached == [mutation]
    assert report["scope_count"] == 1
    assert any("shared original module companion" in item["reason"] for item in report["boundaries"])
    assert len(builder._module_companion_publications) == 3
    for _signature, edits in builder._module_companion_publications.values():
        assert all((first, last, text) in builder.edits[path] for path, first, last, text, _prepend in edits)
    compile_published_fortran(tmp_path/"compiled", outputs)


def test_failed_companion_publication_does_not_reserve_a_definition_in_the_next_candidate(tmp_path, monkeypatch):
    original = companion_emission.publish_original_companion
    outcomes = []

    def fail_once(member, edits):
        published = original(member, edits)
        if member.routine.qualified == "live_groups::first":
            outcomes.append(published)
            if len(outcomes) == 1:
                raise CompilationError("injected checked publication failure")
        return published

    monkeypatch.setattr(companion_emission, "publish_original_companion", fail_once)
    builder = builder_for(tmp_path)
    outputs, report = builder.run()
    assert outcomes == [True, True]
    assert report["scope_count"] == 1
    assert any("injected checked publication failure" in item["reason"] for item in report["boundaries"])
    assert len(builder._module_companion_publications) == 3
    compile_published_fortran(tmp_path/"compiled", outputs)


def test_unchecked_member_cannot_publish_original_companion_source():
    with pytest.raises(CompilationError, match="checked module owner"):
        companion_emission.publish_original_companion(object(), ())


def test_different_original_procedure_cannot_claim_a_published_entry_symbol(tmp_path, monkeypatch):
    original = companion_emission.publish_original_companion

    def collide(member, edits):
        if member.routine.qualified == "live_groups::second":
            keys = member.owner.builder._module_companion_publications
            first_entry = next(entry for (_path, entry), (signature, _edits) in keys.items()
                               if '"procedure":"live_groups::first"' in signature)
            object.__setattr__(member, "entry", first_entry)
        return original(member, edits)

    monkeypatch.setattr(companion_emission, "publish_original_companion", collide)
    builder = builder_for(tmp_path)
    _outputs, report = builder.run()
    assert report["scope_count"] == 0
    assert report["boundaries"]
    assert all("conflicting source, signature or controls" in item["reason"] for item in report["boundaries"])
    assert not builder._module_companion_publications
    assert not report["source_edits"]
