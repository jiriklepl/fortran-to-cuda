"""Empty native wrappers retain replicas without suppressing their original call."""

import shutil
import subprocess
from copy import deepcopy
from dataclasses import replace

import pytest

from compiler.ir import CompilationError
from compiler.scopes.conditional_return import _preserved_native_call
from compiler.scopes.lexical import LexicalBoundaryPatch
from compiler.scopes.native_empty import prove_empty_native_call
from compiler.scopes.native_guard import _association_proof
from compiler.tests.test_native_wrapper_eligibility import CALL, WRAPPER, fixture, setup, source


def formal_wrapper():
    return (WRAPPER.replace("apply(a,b,n,scale)", "apply(a,b,n,records,scale)")
            .replace("apply(a,b,n,fun)", "apply(a,b,n,src,fun)")
            .replace("procedure(scale)::fun", "type(record),allocatable,intent(in)::src(:)\nprocedure(scale)::fun")
            .replace("allocated(records)", "allocated(src)").replace("size(records)", "size(src)")
            .replace("records(i)", "src(i)"))


def issue(tmp_path, wrapper=WRAPPER):
    _path, builder, member, call, _routine = setup(tmp_path, source(wrapper))
    lifecycle = _preserved_native_call(member, (call,))
    assert lifecycle is not None
    return builder, member, lifecycle, prove_empty_native_call(member, lifecycle)


@pytest.mark.parametrize("wrapper", [WRAPPER, formal_wrapper()])
def test_source_proof_checks_allocated_and_empty_without_payload_credit(tmp_path, wrapper):
    _builder, member, lifetime, proof = issue(tmp_path, wrapper)
    record = proof.public()
    assert record["descriptors"] == [{"resource": "hierarchical::records", "name": "records", "import": None}]
    assert record["original_call_executes_once"]
    assert not record["payload_access_authorized"]
    assert not record["entry_definition_authority"]
    assert not record["allocation_state_assumed"]
    assert proof.validate(member, lifetime) is proof
    with pytest.raises(CompilationError, match="original issued"):
        replace(proof).validate(member, lifetime)


@pytest.mark.parametrize("formal_contiguous", [False, True])
def test_contiguous_actual_never_authorizes_a_contiguous_formal(tmp_path, formal_contiguous):
    before = "real(8),intent(inout)::a(:),b(:)"
    wrapper = WRAPPER.replace(before, "real(8),contiguous,intent(inout)::a(:),b(:)",
                              -1 if formal_contiguous else 1)
    _path, builder, _member, _call, routine = setup(tmp_path, source(wrapper))
    child = builder.analysis.routines[routine.qualified + "::apply"]
    call, = routine.execution.content
    with pytest.raises(CompilationError, match="actual association"):
        _association_proof(builder.analysis, routine, child, call)
    if formal_contiguous:
        with pytest.raises(CompilationError, match="actual association"):
            _association_proof(builder.analysis, routine, child, call, allow_contiguous_actual=True)
    else:
        assert _association_proof(builder.analysis, routine, child, call, allow_contiguous_actual=True)


@pytest.mark.parametrize(("before", "after"), [
    ("do i=1,size(records)", "do i=0,size(records)"),
    ("do i=1,size(records)", "do i=1,size(records),-1"),
    ("do i=1,size(records)", "do i=1,n"),
    ("integer::i", "integer,save::i"),
    ("integer::i", "integer::i=0"),
    ("if(allocated(records))then", "observed=1\nif(allocated(records))then"),
    ("endif\nend subroutine", "endif\nobserved=1\nend subroutine"),
    ("call apply(a,b,n,scale)", "observed=1\ncall apply(a,b,n,scale)"),
    ("if(allocated(records))then", "if(n>0)then"),
    ("integer::i", "integer::i\nreal(8)::scratch(n)"),
    ("real(8),intent(inout)::a(:),b(:)", "real(8),intent(out)::a(:),b(:)"),
])
def test_unproved_zero_work_or_observable_entry_keeps_existing_retirement(tmp_path, before, after):
    _path, _builder, member, call, _routine = setup(tmp_path, source(WRAPPER.replace(before, after)))
    lifetime = _preserved_native_call(member, (call,))
    if lifetime is not None:
        with pytest.raises(CompilationError):
            prove_empty_native_call(member, lifetime)


def emitted(tmp_path, text=None):
    text = (source(formal_wrapper()) if text is None else text).replace(
        CALL + "\ncall consumer(left,right,n)\ncall relocate(left,right)\nenddo",
        "call middle(n,change)\ncall relocate(left,right)\nenddo")
    path, builder = fixture(tmp_path, text)
    outputs, report = builder.run()
    assert report["scope_count"] == 1
    owner = builder._conditional_return_draft.owner
    selected = [(m, p) for m in owner.members.values() for _first, _last, p in m.patches
                if isinstance(p, LexicalBoundaryPatch) and p.empty_call is not None]
    assert len(selected) == 1
    member, patch = selected[0]
    patch.validate(member, require_source=True)
    assert patch.original_source.lower().count(CALL) == 1
    assert outputs[report["sources"][str(path)]["replacement"]].lower().count(CALL) == text.lower().count(CALL)
    return builder, report, member, patch


def imported_source(private=False):
    declaration = """type base
integer::xi
end type
type,extends(base)::record
real(8)::flux
end type
type(record),allocatable::records(:)"""
    wrapper = formal_wrapper()
    text = source(wrapper)
    assert text.count(declaration) == text.count(wrapper) == 1
    text = text.replace(declaration, "").replace(wrapper, "")
    text = text.replace("module hierarchical\n", "module hierarchical\nuse native_storage, only: native_wrapper\n", 1)
    access = "private\npublic::native_wrapper\n" if private else ""
    return "module native_storage\nimplicit none\n" + access + declaration + "\ncontains\n" + wrapper + "end module\n" + text


def test_existing_public_descriptor_can_be_imported_without_exposing_payload(tmp_path):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    builder, report, _member, patch = emitted(tmp_path, imported_source())
    descriptor, = patch.empty_call.public()["descriptors"]
    assert descriptor["resource"] == "native_storage::records"
    assert descriptor["import"] == {"module": "native_storage", "remote": "records"}
    assert descriptor["name"].startswith("fort_empty_meta_")
    validate_checked_execution_plans(report, capture_facts=builder.facts)


def test_private_descriptor_keeps_whole_call_retirement(tmp_path):
    text = imported_source(private=True).replace(
        CALL + "\ncall consumer(left,right,n)\ncall relocate(left,right)\nenddo",
        "call middle(n,change)\ncall relocate(left,right)\nenddo")
    _path, builder = fixture(tmp_path, text)
    _outputs, report = builder.run()
    assert report["scope_count"] == 1
    owner = builder._conditional_return_draft.owner
    patches = [p for member in owner.members.values() for _first, _last, p in member.patches
               if isinstance(p, LexicalBoundaryPatch) and CALL in p.original_source.lower()]
    assert len(patches) == 1
    assert patches[0].lifecycle is not None
    assert patches[0].empty_call is None


def test_normal_generation_and_strict_transport_keep_one_original_call(tmp_path):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    builder, report, _member, patch = emitted(tmp_path)
    returned = report["scopes"][0]["checked_execution_plan"]["storage_epoch"]["return_contract"]
    assert returned["native_call_preservation_policy_version"] == 3
    assert patch.completion is None
    assert patch.conditional_guard is None
    validate_checked_execution_plans(report, capture_facts=builder.facts)
    assert builder.finish()


@pytest.mark.native
@pytest.mark.parametrize("imported", [False, True])
def test_generated_descriptor_probe_compiles_with_original_module_abi(tmp_path, imported):
    compiler = shutil.which("gfortran-15") or shutil.which("gfortran")
    if compiler is None:
        pytest.skip("Fortran compiler unavailable")
    builder, report, _member, _patch = emitted(tmp_path, imported_source() if imported else None)
    for name, text in builder.outputs.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in report["build_sources"]:
            if item["role"] == role and item["language"] == "fortran":
                result = subprocess.run([compiler, "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", "-c", item["path"]],
                                        cwd=tmp_path, capture_output=True, text=True, timeout=30)
                assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(("before", "after"), [
    (" = .false.", " = .true."),
    ("size(records) /= 0", "size(records) == 0"),
    ("if (allocated(records)) then", "if (.true.) then"),
    ("if (size(records) /= 0)", "if (.false.)"),
])
def test_transport_rejects_changed_probe_control(tmp_path, before, after):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    _builder, report, member, patch = emitted(tmp_path)
    changed = deepcopy(report)
    edit, = [e for e in changed["source_edits"] if e["source"] == str(member.routine.scope.path)
             and (e["first_line"], e["last_line"]) == (patch.first, patch.last)]
    assert before in edit["replacement"]
    edit["replacement"] = edit["replacement"].replace(before, after, 1)
    with pytest.raises(ValueError, match="empty native CALL"):
        validate_checked_execution_plans(changed)
