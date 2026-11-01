"""Private demand collection must precede one complete authoritative scan."""
import pytest

from compiler.frontend import descriptor_windows as windows
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

ENTRY = "batch::run"
ROOT = "argument::a"


def prepare(tmp_path, *, count=20, tail="", sealed=True, **limits):
    path = tmp_path / "batch.f90"
    path.write_text("module batch\nreal(8),allocatable::hidden(:)\ncontains\n"
                    "subroutine run(a)\nreal(8),allocatable,intent(inout)::a(:)\n"
                    + "\n".join(f"a(1)=value({i}.0_8)" for i in range(count))
                    + "\n" + tail + "\nend subroutine\n"
                    "real(8) function value(x)\nreal(8),intent(in)::x\nvalue=x+1.0_8\nend function\n"
                    "real(8) function unsafe(x)\nreal(8),intent(in)::x\n"
                    "deallocate(hidden)\nunsafe=x\nend function\nend module\n")
    source = SourceEffects([path], **limits)
    if sealed:
        source.seal_session()
    routine = source.routines[ENTRY]
    _graph, selected = source._selected_source(ENTRY, tuple(routine.execution.content))
    return path, source, selected


@pytest.mark.parametrize("sealed", [False, True])
def test_seventy_function_sites_need_three_scans_and_one_publication(tmp_path, monkeypatch, sealed):
    _path, source, selected = prepare(tmp_path, count=70, sealed=sealed)
    derive, publish = windows._derive, windows._publish_derivation
    scans, publications = [], []

    def count_scan(*args, **kwargs):
        scans.append(kwargs.get("_discover_functions", False))
        return derive(*args, **kwargs)

    def count_publication(*args, **kwargs):
        publications.append(args[3])
        return publish(*args, **kwargs)

    monkeypatch.setattr(windows, "_derive", count_scan)
    monkeypatch.setattr(windows, "_publish_derivation", count_publication)
    proof = windows.prove_descriptor_window(source, ENTRY, selected, (ROOT,))
    assert scans == [False, True, False]
    assert len(publications) == 1
    public = proof.public()
    assert len(public["native_storage_function_requirements"]) == 70
    assert public["execution_authorized"] is False
    scans.clear()
    assert proof.validate(source, ENTRY, selected) is proof
    assert scans == ([] if sealed else [False])
    assert proof.public() == public


def test_discovery_result_cannot_publish_or_register_a_window(tmp_path):
    _path, source, selected = prepare(tmp_path, count=3)
    with pytest.raises(windows._DescriptorFunctionsRequired) as error:
        windows._derive(source, ENTRY, selected, (ROOT,), _discover_functions=True)
    demands = error.value.demands
    assert len(demands) == 3
    assert [str(item.expression) for item in demands] == [f"value({i}.0_8)" for i in range(3)]
    assert not getattr(source, "_storage_mutation_effects", {})
    assert not getattr(source, "_descriptor_window_proofs", {})


@pytest.mark.parametrize("tail", ["a(1)=value(1.0_8)", "a(1)=unsafe(1.0_8)"])
def test_cleared_discovery_collector_still_requires_normal_validation(tmp_path, monkeypatch, tail):
    _path, source, selected = prepare(tmp_path, count=1, tail=tail)
    original = windows._Check.expression_children

    def clear(check, *args, **kwargs):
        yield from original(check, *args, **kwargs)
        if check.function_demands:
            check.function_demands.clear()

    monkeypatch.setattr(windows._Check, "expression_children", clear)
    if "unsafe" in tail:
        with pytest.raises(CompilationError, match="storage mutation/escape"):
            windows.prove_descriptor_window(source, ENTRY, selected, (ROOT,))
        assert not getattr(source, "_descriptor_window_proofs", {})
    else:
        proof = windows.prove_descriptor_window(source, ENTRY, selected, (ROOT,))
        assert len(proof.public()["native_storage_function_requirements"]) == 2


def test_empty_discovery_pass_still_cannot_publish_a_window(tmp_path):
    _path, source, selected = prepare(tmp_path, count=1)
    proof = windows.prove_descriptor_window(source, ENTRY, selected, (ROOT,))
    with pytest.raises(windows._DescriptorFunctionsRequired) as error:
        windows._derive(source, ENTRY, selected, (ROOT,), _discover_functions=True)
    assert not error.value.demands
    assert tuple(source._descriptor_window_proofs.values()) == (proof,)


@pytest.mark.parametrize("tail", ["a(1)=unsafe(1.0_8)", "call unknown(a)", "deallocate(a)"])
def test_batched_discovery_never_omits_an_unsafe_tail(tmp_path, tail):
    _path, source, selected = prepare(tmp_path, tail=tail)
    with pytest.raises(CompilationError):
        windows.prove_descriptor_window(source, ENTRY, selected, (ROOT,))
    assert not getattr(source, "_descriptor_window_proofs", {})


def test_explicit_request_cap_remains_a_hard_discovery_failure(tmp_path):
    _path, source, selected = prepare(tmp_path, count=4, procedures=3)
    with pytest.raises(CompilationError, match="request budget exhausted"):
        windows.prove_descriptor_window(source, ENTRY, selected, (ROOT,))
    assert not getattr(source, "_descriptor_window_proofs", {})
    assert len(source._storage_mutation_effects) == 3


def test_hard_error_after_a_pending_demand_is_not_deferred(tmp_path, monkeypatch):
    _path, source, selected = prepare(tmp_path, count=3)
    original = windows._Check.expression_children

    def fail(check, routine, node, **kwargs):
        if check.function_demands:
            raise CompilationError("hard source attachment failure")
        yield from original(check, routine, node, **kwargs)

    monkeypatch.setattr(windows._Check, "expression_children", fail)
    with pytest.raises(CompilationError, match="hard source attachment failure"):
        windows.prove_descriptor_window(source, ENTRY, selected, (ROOT,))
    # Only the first normal pass's independently validated projection exists.
    assert len(source._storage_mutation_effects) == 1
    assert not getattr(source, "_descriptor_window_proofs", {})


def test_nested_actual_functions_are_checked_after_outer_discovery(tmp_path):
    _path, source, selected = prepare(tmp_path, count=1, tail="a(1)=value(unsafe(1.0_8))")
    with pytest.raises(CompilationError, match="storage mutation/escape"):
        windows.prove_descriptor_window(source, ENTRY, selected, (ROOT,))
    assert not getattr(source, "_descriptor_window_proofs", {})


@pytest.mark.parametrize("mutation", ["live_source", "callee", "consumed_registry"])
def test_first_issuer_cannot_change_source_for_later_demands(tmp_path, monkeypatch, mutation):
    path, source, selected = prepare(tmp_path, count=3)
    original, issued = source.storage_mutation_effects, []

    def alter(*args, **kwargs):
        proof = original(*args, **kwargs)
        issued.append(proof)
        if len(issued) == 1:
            if mutation == "live_source":
                path.write_text(path.read_text() + "! source changed by first issuer\n")
            elif mutation == "callee":
                source.numerical_helpers["batch::value"].scope.bindings["x"].kind = 4
            else:
                source._original_call_associations = {}
        return proof

    monkeypatch.setattr(source, "storage_mutation_effects", alter)
    with pytest.raises(CompilationError):
        windows.prove_descriptor_window(source, ENTRY, selected, (ROOT,))
    assert len(issued) == 1
    assert not getattr(source, "_descriptor_window_proofs", {})
