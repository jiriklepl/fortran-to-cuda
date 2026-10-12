"""Scope authentication is local to one bounded original descriptor query."""

from collections import Counter
from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend import descriptor_windows as windows
from compiler.frontend.source_functions import _validate_resolution, _walk
from compiler.ir import CompilationError
from compiler.tests.test_descriptor_unit_resolution import fixture as unit_fixture
from compiler.tests.test_module_descriptor_window_proofs import ENTRY, ROOT, fixture, prove


def test_scope_memo_preserves_fresh_oracle_record_and_every_lookup_counter(tmp_path, monkeypatch):
    analysis, _, selected, _ = fixture(tmp_path)
    calls = Counter()
    original = windows._validate_scope

    def validate(owner, scope, **kwargs):
        assert owner is analysis  # The private joined view does not share this memo.
        calls[id(scope)] += 1
        return original(owner, scope, **kwargs)

    monkeypatch.setattr(windows, "_validate_scope", validate)
    proof = prove(analysis, selected)
    memo_calls = calls.copy()
    assert len(memo_calls) >= 5  # Caller, imports, and complete original children.
    assert set(memo_calls.values()) == {2}  # First use and fresh publication.

    def always_fresh(check, scope):
        windows._validate_scope(check.analysis, scope)
        check.resolution_scopes[id(scope)] = scope

    monkeypatch.setattr(windows._Check, "scope_authority", always_fresh)
    calls.clear()
    oracle = windows._derive(analysis, ENTRY, selected, (), (ROOT,))
    assert sum(calls.values()) > sum(memo_calls.values())
    public = proof.public()
    assert {key: value for key, value in public.items() if key != "proof_identity"} == windows._thaw(windows._freeze(oracle))
    resolution = public["resolution_validation"]
    assert resolution["units"] > 4
    assert resolution["work_entries"] == resolution["syntax_entries"] + resolution["lookup_transitions"]
    assert resolution["per_unit_lookup_limit"] == analysis.operation_limit
    assert resolution["aggregate_source_scan_limit"] == 65536
    assert not public["execution_authorized"]
    assert not public["entry_definition_authority"]


def mutate(analysis, paths, mutation):
    scope = analysis.routines["helper::work"].scope
    binding = scope.bindings["i"]
    if mutation == "ast":
        name = next(node for node in walk(analysis.routines["helper::work"].execution)
                    if type(node).__name__ == "Name" and str(node).lower() == "local")
        name.string = "changed"
    elif mutation == "file":
        paths[1].write_bytes(paths[1].read_bytes() + b"\n! changed after the final tick\n")
    elif mutation == "imports":
        analysis.modules["helper"].imports["local"] = ("storage", "changed")
    elif mutation == "binding":
        scope.bindings["i"] = copy(binding)
    elif mutation == "binding_kind":
        binding.kind = 8
    elif mutation == "kind_scope":
        analysis.modules["helper"].kinds.values["late_kind"] = 8
    elif mutation == "module_role":
        analysis.modules["helper"] = copy(analysis.modules["helper"])
    elif mutation == "routine_role":
        analysis.routines["helper::work"].scope = copy(scope)
    elif mutation == "owner":
        binding.declaring_scope = analysis.modules["helper"]
    elif mutation == "contract":
        analysis.contracts["unused"] = {"identity": "changed during query"}
    elif mutation == "capture":
        analysis._allocation_authorizations[ROOT] = {"identity": "changed during query"}
    elif mutation == "budget":
        analysis.depth_limit -= 1
    else:
        assert mutation == "execution"
        analysis.routines["helper::work"].execution = copy(analysis.routines["helper::work"].execution)


@pytest.mark.parametrize("mutation", [
    "ast", "file", "imports", "binding", "binding_kind", "kind_scope", "owner", "execution",
    "contract", "capture", "budget", "module_role", "routine_role",
])
def test_final_tick_after_scope_cache_hit_cannot_publish_changed_authority(tmp_path, monkeypatch, mutation):
    analysis, _, selected, paths = fixture(tmp_path)
    # Derive the unchanged exact demand without registering a proof. The last
    # real charged visit is identical on the next demand; no synthetic tick is
    # added to create the mutation opportunity.
    record = windows._derive(analysis, ENTRY, selected, (), (ROOT,))
    last_tick = record["source_traversal_entries"]
    original_scope, original_tick = windows._Check.scope_authority, windows._Check.tick
    cache_hits = 0
    changed = False

    def scope_authority(check, scope):
        nonlocal cache_hits
        if id(scope) in check.resolution_scopes:
            cache_hits += 1
        return original_scope(check, scope)

    def tick(check):
        nonlocal changed
        original_tick(check)
        if check.nodes == last_tick:
            assert cache_hits > 0
            mutate(analysis, paths, mutation)
            changed = True

    monkeypatch.setattr(windows._Check, "scope_authority", scope_authority)
    monkeypatch.setattr(windows._Check, "tick", tick)
    with pytest.raises(CompilationError, match="authority|source.*changed|input.*changed") as error:
        prove(analysis, selected)
    assert type(error.value) is CompilationError
    assert changed
    assert not getattr(analysis, "_descriptor_window_proofs", {})


@pytest.mark.parametrize("mutation", ["imports", "binding_kind", "file", "ast"])
def test_next_query_and_retained_proof_revalidate_authority(tmp_path, mutation):
    analysis, _, selected, paths = fixture(tmp_path)
    proof = prove(analysis, selected)
    mutate(analysis, paths, mutation)
    with pytest.raises(CompilationError):
        proof.validate(analysis, ENTRY, selected)
    with pytest.raises(CompilationError):
        prove(analysis, selected)
    assert list(analysis._descriptor_window_proofs.values()) == [proof]


def test_public_resolution_stays_fresh_after_a_local_scope_cache_hit(tmp_path):
    analysis, _, _, _ = fixture(tmp_path)
    routine = analysis.routines[ENTRY]
    check = windows._Check(analysis)
    check.resolution(routine, (), names=("slot",))
    check.resolution(routine, (), names=("slot",))
    analysis.modules["driver"].imports["slot"] = ("storage", "changed")
    with pytest.raises(CompilationError, match="original resolution authority changed"):
        _validate_resolution(analysis, routine.scope, (), names=("slot",))


def test_warmed_scope_memo_does_not_skip_syntax_or_lookup_budget_ticks(tmp_path):
    analysis, _, originals, _ = unit_fixture(tmp_path)
    routine = analysis.routines["application::advance"]
    check = windows._Check(analysis)
    check.resolution(routine, originals[0])
    assert id(routine.scope) in check.resolution_scopes
    before = check.resolution_work
    check.nodes = 65535
    with pytest.raises(CompilationError, match="original-source traversal budget exhausted"):
        check.resolution(routine, originals[0])
    assert check.nodes == 65537
    assert check.resolution_work == before + 2
    assert check.resolution_work == check.resolution_syntax + check.resolution_lookups


def test_private_projection_does_not_inherit_descriptor_query_authority(tmp_path):
    analysis, _, selected, _ = fixture(tmp_path)
    proof = prove(analysis, selected)
    private, _, _ = analysis._segment_projection(ENTRY, selected, descriptor_window=proof)
    with pytest.raises(CompilationError, match="original source-backed procedure authority"):
        windows.prove_descriptor_window(private, ENTRY, selected, module_roots=(ROOT,))
    assert proof.validate(analysis, ENTRY, selected) is proof


def test_unrelated_private_execution_does_not_reject_selected_original_closure(tmp_path):
    analysis, _, selected, _ = fixture(tmp_path)
    unrelated = analysis.routines["driver::unrelated"]
    unrelated.execution = copy(unrelated.execution)
    assert analysis._summary_authority()["role"] == "private_projection"
    proof = prove(analysis, selected)
    public = proof.public()
    assert public["resources"][0]["resource"] == ROOT
    assert set(public["source_call_closure"]) == {"helper::coordinate", "helper::work", "helper::forwarded"}
    assert proof.validate(analysis, ENTRY, selected) is proof


def test_cached_scope_mutation_remains_hard_when_later_expression_declines(tmp_path, monkeypatch):
    analysis, selected, _, _ = unit_fixture(tmp_path, "a(1)=a(1)+1.0_8\na(1)=absent(a(1))")
    entry = "application::advance"
    with pytest.raises(windows.DescriptorWindowUnsupported, match="unknown function/escape effects: absent"):
        windows.prove_descriptor_window(analysis, entry, selected, ("argument::a",))
    original_tick = windows._Check.resolution_tick
    original_scope = windows._Check.scope_authority
    scope = analysis.routines[entry].scope
    changed = False
    cache_hits = 0

    def scope_authority(check, current):
        nonlocal cache_hits
        if id(current) in check.resolution_scopes:
            cache_hits += 1
        return original_scope(check, current)

    def resolution_tick(check, role):
        nonlocal changed
        original_tick(check, role)
        if check.resolution_units == 2 and not changed:
            assert cache_hits > 0
            scope.bindings["a"] = copy(scope.bindings["a"])
            changed = True

    monkeypatch.setattr(windows._Check, "scope_authority", scope_authority)
    monkeypatch.setattr(windows._Check, "resolution_tick", resolution_tick)
    with pytest.raises(CompilationError, match="original resolution authority changed") as error:
        windows.prove_descriptor_window(analysis, entry, selected, ("argument::a",))
    assert type(error.value) is CompilationError
    assert changed
    assert not getattr(analysis, "_descriptor_window_proofs", {})


def test_default_resolution_keeps_per_unit_name_limit_after_another_unit(tmp_path):
    sources = ("module values\n" + "\n".join(
        f"real(8),parameter::v{index}=1.0_8" for index in range(260)) + "\nend module",)
    body = "a(1)=" + "+".join(f"v{index}" for index in range(260))
    analysis, _, originals, _ = unit_fixture(tmp_path, body, imports="use values", sources=sources)
    routine = analysis.routines["application::advance"]
    check = windows._Check(analysis)
    check.resolution(routine, (), names=("a",))
    with pytest.raises(CompilationError, match="resolution budget exhausted") as local:
        check.resolution(routine, originals[0])
    with pytest.raises(CompilationError, match="resolution budget exhausted") as public:
        _validate_resolution(analysis, routine.scope, _walk(originals[0]))
    assert str(local.value) == str(public.value)
