"""Only proved full-team callers acquire additive coherent source workers."""

from hashlib import sha256

from compiler.driver.options import CompilerOptions
from compiler.frontend.source_effects import SourceEffects
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import form_source_scopes

SOURCE = """module operators
implicit none
contains
subroutine produce(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer,intent(in)::n
integer::i
!$omp do
do i=1,n
b(i)=2*a(i)+real(i,8)
enddo
!$omp end do
end subroutine
subroutine consume(a,b,c,n)
real(8),intent(in)::a(:),b(:)
real(8),intent(inout)::c(:)
integer,intent(in)::n
integer::i
!$omp do
do i=1,n
c(i)=a(i)+b(i)
enddo
!$omp end do
end subroutine
subroutine step(a,b,c,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),c(:)
integer,intent(in)::n
call produce(a,b,n)
call consume(a,b,c,n)
end subroutine
end module
module callers
use operators,only:renamed_step=>step
implicit none
contains
subroutine qualified(a,b,c,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),c(:)
integer,intent(in)::n
!$omp parallel default(none) shared(a,b,c,n) num_threads(4)
call renamed_step(a,b,c,n)
!$omp end parallel
end subroutine
subroutine unknown(a,b,c,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),c(:)
integer,intent(in)::n
call renamed_step(a,b,c,n)
end subroutine
end module
"""


def generate(tmp_path, *, source=SOURCE, mode="sections", threads=4, facts_change=None, hardware_profile=None,
             scope_execution="bounded"):
    original = tmp_path / "source.f90"
    original.write_text(source)
    analysis = SourceEffects([original])
    lines = source.splitlines(keepends=True)
    first = next(i for i, line in enumerate(lines, 1) if line.startswith("call renamed_step"))
    stable = {"storage": "stable", "initialized": "whole", "allocation_changes": False, "escapes": False}
    facts = {"schema_version": 2, "sources": analysis.sources,
             "captures": {**{"argument::" + name: {**stable, "association": "shared_whole_storage",
                                                    "descriptor_uniform": True} for name in ("a", "b", "c")},
                          "argument::n": {**stable, "association": "shared_immutable_control"}},
             "participation": {"kind": "omp_full_team", "dispatch": "qualified_companion",
                               "entry": "operators::step", "host_threads": 4, "expected_omp_level": 1,
                               "call_sites": [{"source": str(original), "caller": "callers::qualified",
                                               "first_line": first, "last_line": first,
                                               "span_sha256": sha256(lines[first-1].encode()).hexdigest(),
                                               "team_first_line": first-1,
                                               "team_last_line": next(i for i,line in enumerate(lines,1)
                                                                      if i>first and line.startswith("!$omp end parallel")),
                                               "uniform_guard": "unconditional"}]}}
    if facts_change:
        facts_change(facts)
    outputs, report = form_source_scopes([original], "operators::step", facts=facts,
                                        options=CompilerOptions(gpu_policy=mode, memory_model="scoped"),
                                        config=OffloadConfig(mode, hardware_profile, threads, True,
                                                             scope_execution=scope_execution))
    replacement = outputs[report["sources"][str(original)]["replacement"]] if report["sources"] else source
    assert original.read_text() == source
    return original, outputs, report, replacement


def test_collective_entry_preparation_uses_effect_records_without_native_call_markers(tmp_path):
    from compiler.driver.pipeline import prepare_function
    from compiler.emission.cuda.scoped import generate_scoped
    from compiler.frontend import lower_file

    source = SOURCE.replace("integer::i\n!$omp do\ndo i=1,n\nb(i)",
                            "integer::i,limit\nlimit=n\n!$omp do\ndo i=1,limit\nb(i)")
    path = tmp_path / "preparation.f90"
    path.write_text(source)
    function, plan = prepare_function(lower_file(path, "operators::produce"),
                                      options=CompilerOptions(gpu_policy="sections"))
    emitted = generate_scoped(function, plan, OffloadConfig("sections", None, 4, True), "common_functions.cuh")
    assert ".record(FORT_SCOPE_PLAN_NATIVE," in emitted.cuda
    assert "fort_scope_plan_team_native_call_v1(" not in emitted.cuda
