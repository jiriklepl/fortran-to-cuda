"""Reusable child regions retain their own descriptors and lexical authority."""

from compiler.tests.test_source_scopes import FACT, generate

import pytest


SOURCE = """module renamed_chain
implicit none
contains
subroutine leaf(p,q,s,m)
real(8),intent(in)::p(-5:)
real(8),intent(inout)::q(3:),s(-8:)
integer,intent(inout)::m
integer::i
do i=0,m-1
q(i+3)=2*p(i-5)+real(i,8)
enddo
if(m>2) then
q(3)=q(3)+7.d0
do i=0,m-1
s(i-8)=p(i-5)+q(i+3)
enddo
endif
m=m-1
end subroutine
subroutine left(a,b,c,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),c(:)
integer,intent(inout)::n
call leaf(a,b,c,n)
end subroutine
subroutine right(x,y,z,k)
real(8),intent(in)::x(:)
real(8),intent(inout)::y(:),z(:)
integer,intent(inout)::k
call leaf(x,y,z,k)
end subroutine
subroutine step(a,b,out,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),out(:)
integer,intent(inout)::n
call left(a,b,out,n)
call right(a,b,out,n)
end subroutine
end module
"""


def build(directory, source=SOURCE):
    return generate(directory, source, scope_execution="reached", facts={
        "schema_version": 1, "participation": "serial",
        "captures": {"argument::"+name: FACT for name in ("a", "b", "out")}})


def test_diamond_child_regions_share_artifacts_and_keep_original_bounds(tmp_path):
    original, output, manifest = build(tmp_path)
    assert original.read_text() == SOURCE
    assert manifest["scope_count"] == 1, manifest["boundaries"]
    child, = manifest["child_numerical_regions"]
    assert child["procedure"] == "renamed_chain::leaf"
    assert len(child["regions"]) == 2
    assert all(r["used"] for r in child["regions"])
    variants = next(p for p in manifest["implementation_variants"]["procedures"]
                    if p["procedure"] == child["procedure"])["variants"]
    assert {v["role"] for v in variants} == {"region_dispatcher", "call_worker"}
    assert len(variants) == 2
    assert len([s for s in manifest["build_sources"] if s["path"].startswith("regions/")]) == 1
    generated = (output/manifest["sources"][str(original)]["replacement"]).read_text()
    assert "lbound(p, 1, kind = 8)" in generated.lower()
    assert "lbound(q, 1, kind = 8)" in generated.lower()
    assert "lbound(s, 1, kind = 8)" in generated.lower()
    assert "fort_numerical_guard" in generated
    assert all(s["path"] in manifest["artifacts_sha256"] for s in manifest["build_sources"])


def test_rejected_outer_alias_does_not_leave_child_dispatcher_artifacts(tmp_path):
    source = SOURCE.replace("call left(a,b,out,n)", "call left(a,b,b,n)").replace(
        "call right(a,b,out,n)", "call right(a,b,b,n)")
    _, _, manifest = build(tmp_path, source)
    assert not manifest["source_edits"]
    assert not manifest["child_numerical_regions"]
    assert not manifest["build_sources"]
    assert any("alias" in item["reason"] for item in manifest["boundaries"])


def test_inactive_rejected_call_preserves_later_reused_child_artifacts(tmp_path):
    source = SOURCE.replace('call left(a,b,out,n)', 'if(n<0) then\ncall left(a,b,b,n)\nendif')
    _, _, manifest = build(tmp_path, source)
    owner, = manifest['scopes']
    assert owner['native_continuation'].startswith('original lexical scope')
    assert len(owner['boundaries']) == 1 and 'alias' in owner['boundaries'][0]['reason']
    assert {item['procedure'] for item in manifest['borrowed_source_coordinators']} == {
        'renamed_chain::right', 'renamed_chain::leaf'}
    child, = manifest['child_numerical_regions']
    assert all(region['used'] for region in child['regions'])
    assert len([item for item in manifest['build_sources'] if item['path'].startswith('regions/')]) == 1


def test_repeated_reached_calls_do_not_flatten_the_execution_budget(tmp_path):
    from compiler.frontend.source_effects import SourceEffects

    text = """module repeated_work
contains
subroutine update(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer,intent(in)::n
integer::i
do i=1,n
b(i)=b(i)+a(i)
b(i)=2*b(i)+a(i)
b(i)=3*b(i)-a(i)
enddo
end subroutine
subroutine chain(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer,intent(in)::n
""" + "call update(a,b,n)\n"*28 + """end subroutine
subroutine step(a,b,out,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),out(:)
integer,intent(in)::n
call chain(a,b,n)
out=b
end subroutine
end module
"""
    original, _, manifest = build(tmp_path, text)
    # The legacy flattened proof exceeds 256 operations. Each reusable child
    # and reached call list still fits the original local generation budgets.
    old = SourceEffects([original]).summarize('repeated_work::chain')
    assert not old['complete']
    assert any('budget' in reason for reason in old['reasons'])
    child = next(item for item in manifest['borrowed_source_coordinators']
                 if item['procedure'] == 'repeated_work::chain')
    assert child['effect_authority'] == 'proved reached operations and reusable child interfaces'
    assert manifest['scope_count'] == 1, manifest['boundaries']
    assert manifest['implementation_variants']['limits'] == {'per_procedure': 4, 'compilation': 128}


@pytest.mark.parametrize('combined', [False, True])
@pytest.mark.parametrize('leading', [False, True])
def test_guarded_child_loop_keeps_original_joined_native_fallback(tmp_path, combined, leading):
    opening = '!$omp parallel do private(i) schedule(runtime)' if combined else '!$omp parallel private(i)\n!$omp do'
    ending = '!$omp end parallel do' if combined else '!$omp end do\n!$omp end parallel'
    source = '''module guarded_chain
contains
subroutine leaf(a,b,n)
real(8),intent(in)::a(-5:)
real(8),intent(inout)::b(3:)
integer,intent(in)::n
integer::i
i=0
''' + opening + '''
do i=0,n-1
b(i+3)=a(i-5)+real(i,8)
enddo
''' + ending + '''
end subroutine
subroutine step(a,b,out,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),out(:)
integer,intent(in)::n
call leaf(a,b,n)
out=b
end subroutine
end module
'''
    if leading:
        source = source.replace('i=0\n!$omp', '!$omp')
    _, _, manifest = build(tmp_path, source)
    child, = manifest['borrowed_source_coordinators']
    assert child['procedure'] == 'guarded_chain::leaf'
    native, = [item for item in child['native_operations'] if item['completion'].get('retains_original_team_and_directives')]
    assert native['completion']['available']
    assert native['completion']['retains_original_team_and_directives']
    assert child['gpu_leaves'] == ['guarded_chain::leaf#region1']
