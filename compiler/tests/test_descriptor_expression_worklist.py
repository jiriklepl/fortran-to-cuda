"""Native arithmetic uses charged source traversal, independent of call depth."""
from pathlib import Path

import pytest

from compiler.frontend import descriptor_windows as windows
from compiler.ir import CompilationError
from compiler.tests.test_descriptor_window_proofs import ENTRY, ROOT, fixture, prove

ORACLE = Path(__file__).with_name("fixtures") / "descriptor_expression_recursive.source"


def recursive_oracle():
    namespace = vars(windows).copy()
    exec(compile(ORACLE.read_text(), str(ORACLE), "exec"), namespace)
    return namespace["expression"]


@pytest.mark.parametrize("body", [
    "a(1)=a(1)+sin(a(2))", "a(1)=sqrt(abs(a(1)))", "a(1)=real(size(a),8)+a(2)",
    "v=a(1)\na(1)=norm2(v)",
])
def test_existing_accepted_proof_counts_records_and_identities_match_recursive_oracle(tmp_path, monkeypatch, body):
    _path, analysis, _graph, selected = fixture(tmp_path,
        child_decl="real(8),intent(inout)::a(:)\nreal(8)::v(3)", child=body)
    proof = prove(analysis, selected)
    before = proof.public()
    monkeypatch.setattr(windows._Check, "expression", recursive_oracle())
    old = windows._derive(analysis, ENTRY, selected, (ROOT,))
    assert {key: value for key, value in before.items() if key != "proof_identity"} == windows._thaw(windows._freeze(old))
    assert before["proof_identity"] == windows._hash(old)
    for key in ("source_operations", "source_traversal_entries", "resolution_validation"):
        assert before[key] == old[key]


def test_optional_scalar_presence_classification_is_unchanged_by_worklist(tmp_path, monkeypatch):
    from compiler.tests.test_descriptor_optional_scalars import fixture as optional_fixture

    _path, analysis, entry, selected = optional_fixture(tmp_path)
    public = windows.prove_descriptor_window(analysis, entry, selected, (ROOT,)).public()
    monkeypatch.setattr(windows._Check, "expression", recursive_oracle())
    old = windows._derive(analysis, entry, selected, (ROOT,))
    assert {k: v for k, v in public.items() if k != "proof_identity"} == windows._thaw(windows._freeze(old))


def test_original_wallmodel_depth9_arithmetic_is_scanned_without_depth_or_source_change(tmp_path, monkeypatch):
    expression = "ustar + (max(vel*0.4_8/(log(max((dist/z0)-psi,1e-5_8))),0._8)-ustar)/2"
    path, analysis, _graph, selected = fixture(tmp_path,
        child_decl="real(8),intent(inout)::a(:)\nreal(8)::ustar,vel,dist,z0,psi",
        child=f"ustar=a(1)\nvel=a(2)\ndist=a(3)\nz0=2\npsi=0\nustar={expression}\na(1)=ustar")
    before = path.read_bytes()
    assert analysis.depth_limit == 8
    original = windows._Check.expression
    monkeypatch.setattr(windows._Check, "expression", recursive_oracle())
    with pytest.raises(CompilationError, match="expression depth"):
        prove(analysis, selected)
    monkeypatch.setattr(windows._Check, "expression", original)
    proof = prove(analysis, selected)
    assert proof.validate(analysis, ENTRY, selected) is proof
    assert analysis.depth_limit == 8
    assert not proof.public()["execution_authorized"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("terms", [80, 128])
def test_huge_arithmetic_still_exhausts_existing_original_source_budget(tmp_path, terms):
    _path, analysis, _graph, selected = fixture(tmp_path,
        child_decl="real(8),intent(inout)::a(:)\nreal(8)::" + ",".join(f"v{i}" for i in range(terms)),
        child="a(1)=" + "+".join(f"v{i}" for i in range(terms)), operations=64)
    with pytest.raises(CompilationError, match="budget"):
        prove(analysis, selected)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


@pytest.mark.parametrize("tail", ["a(1)=absent(a(1))", "deallocate(a)"])
def test_long_arithmetic_cannot_hide_later_unknown_or_lifetime_effects(tmp_path, tail):
    expression = "sin(" * 12 + "a(1)" + ")" * 12
    _path, analysis, _graph, selected = fixture(tmp_path,
        child_decl="real(8),allocatable,intent(inout)::a(:)", child="a(1)=" + expression + "\n" + tail)
    with pytest.raises(CompilationError):
        prove(analysis, selected)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


def test_whole_original_call_closure_depth_is_still_bounded(tmp_path):
    helpers = "\n".join(f"subroutine helper{i}(a)\nreal(8),intent(inout)::a(:)\ncall helper{i+1}(a)\nend subroutine"
                         for i in range(10)) + "\nsubroutine helper10(a)\nreal(8),intent(inout)::a(:)\na(1)=1\nend subroutine"
    _path, analysis, _graph, selected = fixture(tmp_path, child="call helper0(a)", extra=helpers)
    with pytest.raises(CompilationError, match="call closure budget"):
        prove(analysis, selected)
    assert analysis.depth_limit == 8


def test_repeated_same_name_terms_still_charge_every_original_source_node(tmp_path):
    # Full original source may exceed the former project-wide scan window.
    # Every syntax/classifier visit remains charged against its exact reached
    # input inventory, without changing the independent per-unit lookup cap.
    statement = "a(1)=" + "+".join(["v"] * 60)
    _path, analysis, _graph, selected = fixture(tmp_path,
        child_decl="real(8),intent(inout)::a(:)\nreal(8)::v",
        child="v=a(1)\n" + "\n".join([statement] * 200))
    record = prove(analysis, selected).public()
    assert record["schema_version"] == 7
    assert record["source_operations"] == 202
    assert record["source_traversal_entries"] > 65536
    assert record["source_traversal_entries"] <= record["source_scan_accounting"]["limit"]
    assert record["resolution_validation"]["per_unit_lookup_limit"] == 256
    assert record["entry_definition_authority"] is False
    assert record["execution_authorized"] is False
    assert analysis.operation_limit == 256
