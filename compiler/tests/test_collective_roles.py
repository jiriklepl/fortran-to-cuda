"""Existing-team worksharing roles derive from original source, never names."""

from copy import deepcopy
from hashlib import sha256

import pytest

from compiler.frontend.source_effects import SourceEffects
from compiler.scopes.collective_roles import prove_existing_team_worksharing

LOOP = "do i=1,size(a)\na(i)=a(i)+scale\nend do"
WORKSHARING = "!$omp do\n" + LOOP + "\n!$omp end do"


def source(tmp_path, body=WORKSHARING, *, specification="", module_spec="", helpers="", extra_sources=()):
    path = tmp_path / "roles.f90"
    text = f"""module unrelated
{module_spec}
contains
subroutine adjust(a,scale)
real(8),intent(inout)::a(:)
real(8),intent(in)::scale
integer::i,j
{specification}
{body}
end subroutine
{helpers}
end module
"""
    path.write_text(text)
    analysis = SourceEffects([path, *extra_sources])
    return path, analysis


def test_original_plain_do_role_is_source_hash_bound_and_leaves_serial_proof_unchanged(tmp_path):
    path, analysis = source(tmp_path)
    before = deepcopy(analysis.summarize("unrelated::adjust"))
    proof = prove_existing_team_worksharing(analysis, "unrelated::adjust")
    assert proof == {"available": True, "reason": None, "source": str(path),
                     "source_sha256": sha256(path.read_bytes()).hexdigest(),
                     "kind": "existing_team_worksharing", "completion": "all_participants_before_effect_commit"}
    assert analysis.summarize("unrelated::adjust") == before
    assert before["native_completion"]["caller_contract"] == "serial_source_scope"
    assert not before["cloneable"]
    assert path.read_text().count("!$omp") == 2


def test_separate_direct_worksharing_loops_and_ordinary_nested_loops_are_supported(tmp_path):
    body = """!$omp do
do i=1,size(a)
do j=1,2
if(scale>0.d0) a(i)=a(i)+scale
end do
end do
!$omp end do
! ordinary comment
!$omp do
do i=1,size(a)
a(i)=2*a(i)
end do
!$omp end do"""
    _path, analysis = source(tmp_path, body)
    assert prove_existing_team_worksharing(analysis, "unrelated::adjust")["available"]


@pytest.mark.parametrize("body", [
    LOOP,
    "!$omp parallel do\n" + LOOP + "\n!$omp end parallel do",
    "!$omp do schedule(static)\n" + LOOP + "\n!$omp end do",
    "!$omp do\n" + LOOP + "\n!$omp end do nowait",
    "!$omp do\n" + LOOP,
    "!$omp task\na=a+scale\n!$omp end task",
    "!$omp target nowait\na=a+scale\n!$omp end target",
    WORKSHARING + "\n" + LOOP,
    LOOP + "\n" + WORKSHARING,
    "a=a+scale\n" + WORKSHARING,
    "if(scale>0.d0) then\n" + WORKSHARING + "\nend if",
    "j=2\n" + WORKSHARING,
    "!$omp do\ndo i=1,size(a)\n!$omp do\ndo j=1,2\na(i)=scale\nend do\n!$omp end do\nend do\n!$omp end do",
    "!$omp do\ndo i=1,size(a)\nif(scale<0.d0) return\na(i)=scale\nend do\n!$omp end do",
])
def test_unknown_clauses_teams_outside_effects_and_private_scalar_state_are_boundaries(tmp_path, body):
    _path, analysis = source(tmp_path, body)
    proof = prove_existing_team_worksharing(analysis, "unrelated::adjust")
    assert not proof["available"]
    assert proof["reason"]


def test_duplicate_rendered_header_cannot_hide_an_unmarked_array_writer(tmp_path):
    _path, analysis = source(tmp_path, LOOP + "\n" + WORKSHARING)
    assert not prove_existing_team_worksharing(analysis, "unrelated::adjust")["available"]


def test_activation_local_recurrence_resets_before_every_work_iteration(tmp_path):
    body = WORKSHARING.replace("a(i)=a(i)+scale", "x=a(i)\ndo j=1,8\nx=x*1.01d0+scale\nenddo\na(i)=x")
    _path, analysis = source(tmp_path, body, specification="real(8)::x")
    assert prove_existing_team_worksharing(analysis, "unrelated::adjust")["available"]


@pytest.mark.parametrize("statement", ["x=x+scale\na(i)=x", "a(i)=x\nx=scale",
                                        "if(scale>0) x=scale\na(i)=x"])
def test_iteration_carried_or_conditionally_defined_temporaries_are_boundaries(tmp_path, statement):
    _path, analysis = source(tmp_path, WORKSHARING.replace("a(i)=a(i)+scale", statement),
                             specification="real(8)::x")
    proof = prove_existing_team_worksharing(analysis, "unrelated::adjust")
    assert not proof["available"]
    assert "temporary" in proof["reason"]


@pytest.mark.parametrize("declaration", ["real(8),save::x", "real(8)::x=1.d0",
                                          "real(8),volatile::x", "real(8),asynchronous::x"])
def test_nonautomatic_scalar_storage_is_not_an_iteration_temporary(tmp_path, declaration):
    _path, analysis = source(tmp_path, WORKSHARING.replace("a(i)=a(i)+scale", "x=scale\na(i)=x"),
                             specification=declaration)
    assert not prove_existing_team_worksharing(analysis, "unrelated::adjust")["available"]


def test_scalar_output_even_inside_worksharing_is_not_a_shared_control(tmp_path):
    body = WORKSHARING.replace("a(i)=a(i)+scale", "scale=real(i,8)\na(i)=scale")
    path, analysis = source(tmp_path, body)
    path.write_text(path.read_text().replace("real(8),intent(in)::scale", "real(8),intent(inout)::scale"))
    analysis = SourceEffects([path])
    proof = prove_existing_team_worksharing(analysis, "unrelated::adjust")
    assert not proof["available"]
    assert "externally visible scalar" in proof["reason"]


def test_module_scalar_output_and_nonlocal_loop_iterator_are_boundaries(tmp_path):
    _path, analysis = source(tmp_path, WORKSHARING.replace("a(i)=a(i)+scale", "counter=counter+1\na(i)=scale"),
                             module_spec="integer::counter")
    assert not prove_existing_team_worksharing(analysis, "unrelated::adjust")["available"]
    path, _analysis = source(tmp_path, module_spec="integer::i")
    path.write_text(path.read_text().replace("integer::i,j", "integer::j"))
    analysis = SourceEffects([path])
    assert not prove_existing_team_worksharing(analysis, "unrelated::adjust")["available"]


def test_direct_source_and_opaque_calls_are_not_leaf_worksharing_roles(tmp_path):
    helpers = """subroutine helper(a)
real(8),intent(inout)::a(:)
a=a+1.d0
end subroutine"""
    _path, analysis = source(tmp_path, WORKSHARING.replace("a(i)=a(i)+scale", "call helper(a)"), helpers=helpers)
    proof = prove_existing_team_worksharing(analysis, "unrelated::adjust")
    assert not proof["available"]
    assert "direct leaf" in proof["reason"]
    path, _analysis = source(tmp_path, WORKSHARING.replace("a(i)=a(i)+scale", "call opaque(a)"),
                             module_spec="use external,only:opaque")
    contract = {"external::opaque": {"complete": True, "lifetime": "stable", "escapes": False,
                "ordering": "serial", "descriptor_changes": False, "identity": "opaque-v1",
                "effects": [{"kind": "read", "argument": 0, "section": "whole"}]}}
    analysis = SourceEffects([path], contracts=contract)
    proof = prove_existing_team_worksharing(analysis, "unrelated::adjust")
    assert not proof["available"]
    assert "direct leaf" in proof["reason"]


def test_threadprivate_ownership_and_persistent_locals_are_not_shared_roles(tmp_path):
    _path, analysis = source(tmp_path, module_spec="real(8)::hidden(4)\n!$omp threadprivate(hidden)")
    assert not prove_existing_team_worksharing(analysis, "unrelated::adjust")["available"]
    _path, analysis = source(tmp_path, specification="integer,save::calls=0")
    proof = prove_existing_team_worksharing(analysis, "unrelated::adjust")
    assert not proof["available"]
    assert "persistent local state" in proof["reason"]


def test_matching_optional_assertion_confirms_but_never_creates_a_role(tmp_path):
    path, analysis = source(tmp_path)
    assertion = {"source": str(path), "source_sha256": analysis.sources[str(path)],
                 "kind": "existing_team_worksharing", "completion": "all_participants_before_effect_commit"}
    assert prove_existing_team_worksharing(analysis, "unrelated::adjust", assertion=assertion)["available"]
    path, analysis = source(tmp_path, LOOP)
    assertion["source_sha256"] = analysis.sources[str(path)]
    assert not prove_existing_team_worksharing(analysis, "unrelated::adjust", assertion=assertion)["available"]


@pytest.mark.parametrize("field", ["source", "source_sha256", "kind", "completion"])
def test_contradictory_optional_assertions_remain_boundaries(tmp_path, field):
    path, analysis = source(tmp_path)
    assertion = {"source": str(path), "source_sha256": analysis.sources[str(path)],
                 "kind": "existing_team_worksharing", "completion": "all_participants_before_effect_commit"}
    assertion[field] = "contradictory"
    proof = prove_existing_team_worksharing(analysis, "unrelated::adjust", assertion=assertion)
    assert not proof["available"]
    assert "assertion disagrees" in proof["reason"]


def test_stale_original_source_cannot_produce_available_role(tmp_path):
    path, analysis = source(tmp_path)
    path.write_text(path.read_text() + "! source changed\n")
    proof = prove_existing_team_worksharing(analysis, "unrelated::adjust")
    assert not proof["available"]
    assert "source changed" in proof["reason"]


def test_missing_procedure_has_no_source_or_role_authority(tmp_path):
    _path, analysis = source(tmp_path)
    proof = prove_existing_team_worksharing(analysis, "unrelated::absent")
    assert not proof["available"]
    assert proof["source"] is None
