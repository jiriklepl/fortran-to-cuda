"""Original native OpenMP calls need synchronous completion before coherence."""

import pytest

from compiler.frontend.source_effects import SourceEffects


def analyze(tmp_path, body, *, helpers="", declarations="", module_spec="", contracts=None):
    path = tmp_path / "completion.f90"
    path.write_text(f"""module completion
{module_spec}
contains
subroutine step(a)
real(8),intent(inout)::a(:)
integer::i,j
{declarations}
{body}
end subroutine
{helpers}
end module
""")
    effects = SourceEffects([path], contracts=contracts)
    return effects, effects.summarize("completion::step")


LOOP = "do i=1,size(a)\na(i)=a(i)+1.d0\nend do"


@pytest.mark.parametrize("directive", ["do", "parallel do", "PARALLEL DO"])
def test_clause_free_matched_loops_have_serial_completion(tmp_path, directive):
    body = f"!$omp {directive}\n! ordinary comment\n{LOOP}\n! comment\n!$omp end {directive}"
    effects, summary = analyze(tmp_path, body)
    proof = summary["native_completion"]
    assert summary["complete"]
    assert proof["available"]
    assert proof["caller_contract"] == "serial_source_scope"
    assert proof["requires_serial_caller"] is True
    assert proof["has_openmp_in_closure"] is True
    assert "matched clause-free" in proof["reason"]
    assert not summary["cloneable"]
    assert len(summary["openmp_directives"]) == 2
    assert not effects.native_sections("completion::step").available
    assert (tmp_path / "completion.f90").read_text().count("!$omp ") == 2


def test_sequential_pairs_inside_original_guard_remain_supported(tmp_path):
    body = f"""if(size(a)>0) then
!$omp parallel do
{LOOP}
!$omp end parallel do
!$omp do
{LOOP}
!$omp end do
end if"""
    _, summary = analyze(tmp_path, body)
    assert summary["native_completion"]["available"]
    assert any(op.get("guard") and "SIZE(a) > 0" in op["guard"] for op in summary["operations"])


def test_no_directives_keeps_existing_cloneable_and_sections(tmp_path):
    effects, summary = analyze(tmp_path, "a=a+1.d0")
    assert summary["complete"]
    assert summary["cloneable"]
    assert effects.native_sections("completion::step").available
    assert summary["native_completion"] == {
        "available": True, "reason": "source closure has no OpenMP directives",
        "caller_contract": "serial_source_scope", "requires_serial_caller": True,
        "has_openmp_in_closure": False, "has_opaque_calls_in_closure": False,
    }


@pytest.mark.parametrize("body", [
    f"!$omp do schedule(static)\n{LOOP}\n!$omp end do",
    f"!$omp parallel do private(j)\n{LOOP}\n!$omp end parallel do",
    f"!$omp do\n{LOOP}\n!$omp end do nowait",
    f"!$omp do\n{LOOP}",
    f"{LOOP}\n!$omp end do",
    f"!$omp do\n{LOOP}\n!$omp end parallel do",
    f"!$omp parallel do simd\n{LOOP}\n!$omp end parallel do simd",
    f"!$omp simd\n{LOOP}\n!$omp end simd",
    "!$omp task\na=a+1.d0\n!$omp end task",
    "!$omp target nowait\na=a+1.d0\n!$omp end target",
    "!$omp cancel parallel\na=a+1.d0",
    "!$omp unknown\na=a+1.d0",
    "!$omp do\na=a+1.d0\n!$omp end do",
    f"!$omp do\na=a+1.d0\n{LOOP}\n!$omp end do",
    f"!$omp do\n{LOOP}\na=a+1.d0\n!$omp end do",
    f"!$omp do &\n!$omp& schedule(static)\n{LOOP}\n!$omp end do",
    "!$omp parallel do\ndo i=1,size(a)\n!$omp do\ndo j=1,1\na(i)=1.d0\nend do\n!$omp end do\nend do\n!$omp end parallel do",
])
def test_unknown_async_clauses_and_unmatched_directives_are_boundaries(tmp_path, body):
    _, summary = analyze(tmp_path, body)
    assert summary["complete"], summary["reasons"]
    assert not summary["native_completion"]["available"]
    assert summary["native_completion"]["reason"]
    assert not summary["cloneable"]


def test_directive_before_declaration_cannot_claim_later_loop(tmp_path):
    _, summary = analyze(tmp_path, LOOP + "\n!$omp end do", declarations="!$omp do\ninteger::extra")
    assert not summary["native_completion"]["available"]
    assert "associated" in summary["native_completion"]["reason"]


def test_nested_ordinary_loops_are_allowed(tmp_path):
    _, summary = analyze(tmp_path, """!$omp parallel do
do i=1,size(a)
do j=1,2
a(i)=a(i)+1.d0
end do
end do
!$omp end parallel do""")
    assert summary["native_completion"]["available"]


LEAF = """subroutine leaf(a)
real(8),intent(inout)::a(:)
a=a+1.d0
end subroutine"""


def test_source_backed_directive_free_helper_inside_loop_is_allowed(tmp_path):
    _, summary = analyze(tmp_path, "!$omp parallel do\ndo i=1,1\ncall leaf(a)\nend do\n!$omp end parallel do", helpers=LEAF)
    assert summary["native_completion"]["available"]


def test_transitive_async_helper_completion_is_unavailable(tmp_path):
    helpers = LEAF.replace("a=a+1.d0", "!$omp task\na=a+1.d0\n!$omp end task")
    _, summary = analyze(tmp_path, "call leaf(a)", helpers=helpers)
    assert summary["complete"]
    assert summary["cloneable"]
    assert not summary["native_completion"]["available"]
    assert "callee completion" in summary["native_completion"]["reason"]
    assert summary["native_completion"]["has_openmp_in_closure"]


def test_synchronous_helper_is_allowed_outside_openmp_loop(tmp_path):
    helpers = LEAF.replace("a=a+1.d0", "integer::i\n!$omp parallel do\n" + LOOP + "\n!$omp end parallel do")
    _, summary = analyze(tmp_path, "call leaf(a)", helpers=helpers)
    assert summary["native_completion"]["available"]
    assert summary["native_completion"]["has_openmp_in_closure"]
    assert summary["cloneable"]


def test_hidden_nested_team_through_directive_free_wrapper_is_rejected(tmp_path):
    helpers = LEAF.replace("a=a+1.d0", "integer::i\n!$omp parallel do\n" + LOOP + "\n!$omp end parallel do") + """
subroutine wrapper(a)
real(8),intent(inout)::a(:)
call leaf(a)
end subroutine"""
    _, summary = analyze(tmp_path, "!$omp parallel do\ndo i=1,1\ncall wrapper(a)\nend do\n!$omp end parallel do", helpers=helpers)
    assert summary["complete"]
    assert not summary["native_completion"]["available"]
    assert "calls a helper with OpenMP" in summary["native_completion"]["reason"]


def test_opaque_contract_inside_openmp_loop_has_no_nested_team_proof(tmp_path):
    contract = {"external::opaque": {"complete": True, "lifetime": "stable", "escapes": False,
                "ordering": "serial", "descriptor_changes": False, "identity": "opaque-v1",
                "effects": [{"kind": "read", "argument": 0, "section": "whole"}]}}
    _, summary = analyze(tmp_path, "!$omp parallel do\ndo i=1,1\ncall opaque(a)\nend do\n!$omp end parallel do",
                         module_spec="use external,only:opaque", contracts=contract)
    assert summary["complete"], summary["reasons"]
    assert not summary["native_completion"]["available"]
    assert "opaque native call" in summary["native_completion"]["reason"]


def test_opaque_call_hidden_by_source_wrapper_inside_loop_is_rejected(tmp_path):
    contract = {"external::opaque": {"complete": True, "lifetime": "stable", "escapes": False,
                "ordering": "serial", "descriptor_changes": False, "identity": "opaque-v1",
                "effects": [{"kind": "read", "argument": 0, "section": "whole"}]}}
    helpers = """subroutine wrapper(a)
real(8),intent(inout)::a(:)
call opaque(a)
end subroutine"""
    _, summary = analyze(tmp_path, "!$omp parallel do\ndo i=1,1\ncall wrapper(a)\nend do\n!$omp end parallel do",
                         helpers=helpers, module_spec="use external,only:opaque", contracts=contract)
    assert summary["complete"]
    assert not summary["native_completion"]["available"]
    assert "opaque nested-team behavior" in summary["native_completion"]["reason"]


@pytest.mark.parametrize("module_spec", ["real(8)::hidden(4)\n!$omp threadprivate(hidden)", "!$omp declare target"])
def test_module_openmp_ownership_requires_a_separate_capture_proof(tmp_path, module_spec):
    _, summary = analyze(tmp_path, LOOP, module_spec=module_spec)
    assert summary["complete"]
    assert not summary["native_completion"]["available"]
    assert "module OpenMP ownership" in summary["native_completion"]["reason"]


def test_unrelated_sibling_openmp_routine_does_not_restrict_this_closure(tmp_path):
    helpers = LEAF.replace("a=a+1.d0", "!$omp task\na=a+1.d0\n!$omp end task")
    _, summary = analyze(tmp_path, LOOP, helpers=helpers)
    assert summary["native_completion"]["available"]
    assert not summary["native_completion"]["has_openmp_in_closure"]


def test_use_associated_threadprivate_resource_is_not_shared_capture(tmp_path):
    owner = tmp_path / "state.f90"
    owner.write_text("module state\nreal(8)::hidden(4)\n!$omp threadprivate(hidden)\nend module\n")
    user = tmp_path / "user.f90"
    user.write_text("""module user
use state,only:hidden
contains
subroutine step(a)
real(8),intent(inout)::a(:)
a=hidden
end subroutine
end module""")
    summary = SourceEffects([owner, user]).summarize("user::step")
    assert summary["complete"]
    assert not summary["native_completion"]["available"]
    assert "module OpenMP ownership" in summary["native_completion"]["reason"]


def test_incomplete_effects_never_receive_completion_proof(tmp_path):
    _, summary = analyze(tmp_path, "call unknown(a)")
    assert not summary["complete"]
    assert not summary["native_completion"]["available"]
    assert "incomplete" in summary["native_completion"]["reason"]
