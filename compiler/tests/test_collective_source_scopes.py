"""Only proved full-team callers acquire additive coherent source workers."""

import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
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


def generate(tmp_path, *, source=SOURCE, mode="sections", threads=4, facts_change=None):
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
                                               "team_first_line": first-1, "team_last_line": first+1,
                                               "uniform_guard": "unconditional"}]}}
    if facts_change:
        facts_change(facts)
    outputs, report = form_source_scopes([original], "operators::step", facts=facts,
                                        options=CompilerOptions(gpu_policy=mode, memory_model="scoped"),
                                        config=OffloadConfig(mode, None, threads, True))
    replacement = outputs[report["sources"][str(original)]["replacement"]] if report["sources"] else source
    assert original.read_text() == source
    return original, outputs, report, replacement


def test_original_entry_and_unqualified_callers_remain_unchanged(tmp_path):
    _, outputs, report, text = generate(tmp_path)
    assert report["scope_count"] == 1, report["boundaries"]
    assert SOURCE.split("subroutine step",1)[1].split("end subroutine",1)[0] in text
    assert SOURCE.split("subroutine unknown",1)[1].split("end subroutine",1)[0] in text
    scope, = report["scopes"]
    assert scope["participation"] == "qualified_full_team"
    assert scope["gpu_leaves"] == ["operators::consume", "operators::produce"]
    assert scope["definition_preflight"]["query_available"]
    assert not scope["estimate_available"]
    proof = report["participation"]
    assert proof["other_callers"] == "unchanged"
    assert proof["call_sites"][0]["caller"] == "callers::qualified"
    assert any(name.endswith("shared_entry.cu") for name in outputs)
    assert "run_team" in text


def test_team_and_allocation_checks_precede_owner_descriptors_and_queries(tmp_path):
    source = SOURCE.replace("subroutine qualified(a,b,c,n)\nreal(8),intent(in)::a(:)\n"
                            "real(8),intent(inout)::b(:),c(:)",
                            "subroutine qualified(a,b,c,n)\nreal(8),allocatable,intent(in)::a(:)\n"
                            "real(8),allocatable,intent(inout)::b(:),c(:)")
    _, _, report, text = generate(tmp_path, source=source)
    assert report["scope_count"] == 1, report["boundaries"]
    caller = text.split("subroutine qualified",1)[1].split("end subroutine",1)[0]
    assert caller.index("() /= 1") < caller.index("!$omp barrier")
    assert caller.index("allocated(a)") < caller.index("if (all(") < caller.index("lbound(a")
    owner = text.split("subroutine " + report["scopes"][0]["owner"],1)[1]
    assert owner.index("fort_state%contiguous") < owner.index("fort_scope_create")
    assert owner.index("fort_scope_plan_validate") < owner.index("fort_returned = fort_run_")
    assert "fort_state%addresses" in owner
    assert "fort_state%lowers" in owner
    assert "if (all(fort_state%extents" in owner


def test_automatic_team_execution_awaits_matching_offline_calibration(tmp_path):
    _, outputs, report, text = generate(tmp_path, mode="auto")
    assert report["scope_count"] == 0
    assert not outputs.get("scoped_runtime.cu")
    assert text == SOURCE
    assert report["participation"]["kind"] == "omp_full_team"
    assert "collective synchronization calibration" in report["boundaries"][0]["reason"]


@pytest.mark.parametrize("directive", ["do nowait", "parallel do", "do schedule(dynamic)"])
def test_unsupported_leaf_participation_is_successful_native(tmp_path, directive):
    source = SOURCE.replace("!$omp do", "!$omp " + directive)
    if directive == "parallel do":
        source = source.replace("!$omp end do", "!$omp end parallel do")
    _, _, report, text = generate(tmp_path, source=source)
    assert report["scope_count"] == 0
    assert text == source
    assert report["participation"]["schema_version"] == 2


def test_invalid_caller_assertion_is_rejected_without_source_edits(tmp_path):
    with pytest.raises(CompilationError, match="caller identity"):
        generate(tmp_path, facts_change=lambda facts: facts["participation"]["call_sites"][0].update(caller="wrong::caller"))


def test_native_assertion_does_not_authorize_a_different_source_role(tmp_path):
    def assertion(facts):
        source, digest = next(iter(facts["sources"].items()))
        facts["native_participation"] = {"operators::produce": {
            "source": source, "source_sha256": digest, "kind": "existing_team_worksharing",
            "completion": "asynchronous"}}
    with pytest.raises(CompilationError, match="differs from source proof"):
        generate(tmp_path, facts_change=assertion)


def test_an_uncalibrated_serial_profile_cannot_sneak_into_collective_costs(tmp_path):
    # Public source automatic availability is false even when numerical work
    # estimates are available; CPU fallback is an explicit successful result.
    _, _, report, _ = generate(tmp_path, mode="auto")
    assert not report["automatic_estimate_available"]
    assert all(item["available"] for item in report["collective_roles"])


@pytest.mark.parametrize("allocatable", [False, True])
def test_public_fortran_team_source_compiles_without_line_limit_extensions(tmp_path, allocatable):
    fortran = shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    source = SOURCE
    if allocatable:
        source = SOURCE.replace("subroutine qualified(a,b,c,n)\nreal(8),intent(in)::a(:)\n"
                                "real(8),intent(inout)::b(:),c(:)",
                                "subroutine qualified(a,b,c,n)\nreal(8),allocatable,intent(in)::a(:)\n"
                                "real(8),allocatable,intent(inout)::b(:),c(:)")
    _, outputs, report, _ = generate(tmp_path, source=source)
    for name, text in outputs.items():
        destination = tmp_path / "output" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
    sources = [tmp_path / "output" / item["path"] for item in report["build_sources"]
               if item["language"] == "fortran" and item["role"] == "common_runtime"]
    sources += [tmp_path / "output" / item["path"] for item in report["build_sources"]
                if item["language"] == "fortran" and item["role"] == "shared_entry"]
    sources += [tmp_path / "output" / item["path"] for item in report["build_sources"]
                if item["role"] == "original_source"]
    response = subprocess.run([fortran, "-c", "-fopenmp", "-fcheck=all,array-temps", *map(str,sources)],
                              cwd=tmp_path, capture_output=True, text=True, timeout=60, check=False)
    assert response.returncode == 0, response.stdout + response.stderr


@pytest.mark.parametrize("actual", ["allocated", "lbound", "all"])
def test_caller_intrinsic_capture_collisions_remain_unchanged_native(tmp_path, actual):
    block = SOURCE.split("subroutine qualified",1)[1].split("end subroutine",1)[0]
    renamed = block.replace("(a,", "(" + actual + ",", 1)
    renamed = renamed.replace("::a(:)", "::" + actual + "(:)")
    renamed = renamed.replace("shared(a,", "shared(" + actual + ",")
    renamed = renamed.replace("renamed_step(a,", "renamed_step(" + actual + ",")
    source = SOURCE.replace(block, renamed)
    _, _, report, text = generate(tmp_path, source=source)
    assert report["scope_count"] == 0
    assert "namespace conflicts" in report["boundaries"][0]["reason"]
    assert text == source


def test_unused_owning_allocatable_out_cannot_drop_its_deallocation(tmp_path):
    source = SOURCE.replace("subroutine step(a,b,c,n)\nreal(8),intent(in)::a(:)\n"
                            "real(8),intent(inout)::b(:),c(:)",
                            "subroutine step(a,b,c,n)\nreal(8),intent(in)::a(:)\n"
                            "real(8),allocatable,intent(out)::b(:)\nreal(8),intent(inout)::c(:)")
    source = source.replace("call produce(a,b,n)\ncall consume(a,b,c,n)",
                            "call produce(a,c,n)\ncall produce(a,c,n)")
    for caller in ("qualified", "unknown"):
        source = source.replace(f"subroutine {caller}(a,b,c,n)\nreal(8),intent(in)::a(:)\n"
                                "real(8),intent(inout)::b(:),c(:)",
                                f"subroutine {caller}(a,b,c,n)\nreal(8),intent(in)::a(:)\n"
                                "real(8),allocatable,intent(inout)::b(:)\nreal(8),intent(inout)::c(:)")
    _, outputs, report, text = generate(tmp_path, source=source)
    assert report["scope_count"] == 0
    assert report["participation"]["schema_version"] == 2
    assert "original descriptor and allocation semantics" in report["boundaries"][0]["reason"]
    assert text == source
    assert not report["source_edits"]
    assert list(outputs) == ["scope-manifest.json"]


@pytest.mark.parametrize("name", ["fort_run_0", "fort_plan_0"])
def test_original_leaf_name_cannot_bind_a_generated_interface(tmp_path, name):
    source = SOURCE.replace("produce", name)
    _, _, report, text = generate(tmp_path, source=source)
    assert report["scope_count"] == 0
    assert "helper namespace" in report["boundaries"][0]["reason"]
    assert text == source


@pytest.mark.parametrize("wildcard", [False, True])
@pytest.mark.parametrize("name", ["all", "c_loc", "omp_get_thread_num"])
def test_copied_use_cannot_shadow_owner_intrinsics(tmp_path, wildcard, name):
    extra = """module imported
implicit none
integer :: all
end module
"""
    extra = extra.replace(":: all", ":: " + name)
    association = "use imported" if wildcard else "use imported,only:" + name
    source = extra + SOURCE.replace("subroutine step(a,b,c,n)", "subroutine step(a,b,c,n)\n" + association)
    _, _, report, text = generate(tmp_path, source=source)
    assert report["scope_count"] == 0
    assert "helper namespace" in report["boundaries"][0]["reason"]
    assert text == source


@pytest.mark.parametrize("wildcard", [False, True])
def test_wildcard_reexport_cannot_shadow_a_generated_alias(tmp_path, wildcard):
    association = "use deeper" if wildcard else "use deeper,only:fort_run_0"
    extra = """module deeper
implicit none
integer :: fort_run_0
end module
module imported
""" + association + "\nimplicit none\nend module\n"
    source = extra + SOURCE.replace("subroutine step(a,b,c,n)", "subroutine step(a,b,c,n)\nuse imported")
    _, _, report, text = generate(tmp_path, source=source)
    assert report["scope_count"] == 0
    assert "helper namespace" in report["boundaries"][0]["reason"]
    assert text == source


def test_interoperable_logical_capture_keeps_its_native_reference(tmp_path):
    source = SOURCE.replace("produce(a,b,n)", "produce(a,b,n,flag)")
    source = source.replace("(a,b,c,n)", "(a,b,c,n,flag)")
    source = source.replace("integer,intent(in)::n", "integer,intent(in)::n\nlogical(1),intent(in)::flag")
    # CONSUME does not capture FLAG, so do not give its unused local an INTENT.
    source = source.replace("subroutine consume(a,b,c,n,flag)", "subroutine consume(a,b,c,n)")
    source = source.replace("call consume(a,b,c,n,flag)", "call consume(a,b,c,n)")
    source = source.replace("subroutine consume(a,b,c,n)\nreal(8),intent(in)::a(:),b(:)\n"
                            "real(8),intent(inout)::c(:)\ninteger,intent(in)::n\nlogical(1),intent(in)::flag",
                            "subroutine consume(a,b,c,n)\nreal(8),intent(in)::a(:),b(:)\n"
                            "real(8),intent(inout)::c(:)\ninteger,intent(in)::n")
    source = source.replace("shared(a,b,c,n)", "shared(a,b,c,n,flag)")
    source = source.replace("b(i)=2*a(i)+real(i,8)", "if(flag) b(i)=2*a(i)+real(i,8)")
    def facts_change(facts):
        facts["captures"]["argument::flag"] = {
            "storage": "stable", "allocation_changes": False, "escapes": False,
            "association": "shared_immutable_control"}
    _, _, report, text = generate(tmp_path, source=source, facts_change=facts_change)
    assert report["scope_count"] == 1, report["boundaries"]
    assert "logical(c_bool), intent(in) :: fort_capture_" in text
    assert report["scopes"][0]["gpu_leaves"] == ["operators::consume"]
    assert "call produce(" in text
