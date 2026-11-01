"""A source-issued path reuses checked children and the ordinary lifetime engine."""
import json
import shutil
import subprocess
from copy import copy
from hashlib import sha256

import pytest
from fparser.two.utils import walk

from compiler.driver.options import CompilerOptions
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder

SOURCE = """module supported_scope
implicit none
contains
subroutine produce(a,b,n)
integer,intent(in)::n
real(8),intent(out)::a(:),b(:)
integer::i
do i=1,n
a(i)=real(i,8)
b(i)=2.0_8*real(i,8)
enddo
end subroutine
subroutine selected(a,b,n)
integer,intent(in)::n
real(8),intent(inout)::a(:)
real(8),intent(in)::b(:)
integer::i
do i=1,n
a(i)=a(i)+b(i)
enddo
end subroutine
subroutine finish(a,b,n)
integer,intent(in)::n
real(8),intent(in)::a(:)
real(8),intent(out)::b(:)
integer::i
do i=1,n
b(i)=2.0_8*a(i)
enddo
end subroutine
subroutine unavailable(a,b)
real(8),intent(inout)::a(:),b(:)
call unknown_private_escape(a)
b=a
end subroutine
subroutine run(n,mode)
integer,intent(in)::n,mode
integer,save::called=0
logical,save::eligible=.false.
real(8),save,allocatable::a(:),b(:)
if(called==0)then
allocate(a(n))
allocate(b(n))
a=0
eligible=.true.
endif
called=called+1
call produce(a,b,n)
if(mode==1)then
eligible=.false.
call unavailable(a,b)
else
call selected(a,b,n)
endif
call finish(a,b,n)
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE):
    """Return fresh source-bound input; useful for independent reader fixtures."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "supported_scope.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    routine = analysis.routines["supported_scope::run"]
    calls = [node for node in walk(routine.execution) if type(node).__name__ == "Call_Stmt"]

    def anchor(node):
        callee = str(node.items[0]).lower()
        return {"span": list(node.item.span), "sha256": sha256(str(node).encode()).hexdigest(),
                "callee": "supported_scope::"+callee}

    first = next(node for node in calls if str(node.items[0]).lower() == "produce")
    last = next(node for node in calls if str(node.items[0]).lower() == "finish")
    request = {"version": 1, "source": {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()},
        "first": anchor(first), "last": anchor(last),
        "eligibility": {"binding": "eligible", "resources": ["supported_scope::run::"+name for name in ("a", "b")],
                        "barred_calls": [anchor(node) for node in calls if str(node.items[0]).lower() == "unavailable"]}}
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): request["source"]["sha256"]},
        "captures": {root: {"storage": "stable", "initialized": "none", "escapes": False,
                            "allocation_changes": False, "lifetime": "selected_window"}
                     for root in request["eligibility"]["resources"]}}
    contracts = {"supported_scope::run": {"native_ancestor": {"version": 4,
        "lifetime": "supported_original_path", "path": request,
        "device_roots": ["supported_scope::"+name for name in ("produce", "selected", "finish")]}}}
    return path, facts, contracts


def builder_fixture(tmp_path, source=SOURCE):
    path, facts, contracts = fixture(tmp_path, source)
    builder = ScopeBuilder([path], "supported_scope::run", facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections"), contracts=contracts)
    return path, builder


def emit_supported_path_fixture(tmp_path, source=SOURCE):
    """Publish an actual compiler-issued sample with all generated artifact bytes."""
    path, builder = builder_fixture(tmp_path, source)
    outputs, report = builder.run()
    (tmp_path / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    for name, text in outputs.items():
        target = tmp_path / "generated" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return path, builder, outputs, report


def clock_source():
    return SOURCE.replace("integer,save::called=0", "integer,save::called=0\ninteger,save::rate").replace(
        "eligible=.true.\nendif", "if(mode>=0)call system_clock(count_rate=rate)\neligible=.true.\nendif")


def test_checked_prearm_clock_is_versioned_separately_and_binds_the_whole_original_predicate(tmp_path):
    path, _builder, _outputs, report = emit_supported_path_fixture(tmp_path, clock_source())
    assert report["scope_count"] == 1, report["boundaries"]
    scope, = report["scopes"]
    epoch = scope["supported_path_epoch"]
    assert epoch == scope["checked_execution_plan"]["storage_epoch"]["supported_path_epoch"]
    assert epoch["version"] == 2
    metadata, = epoch["gate_metadata_edges"]
    assert len(epoch["outside_interval_edges"]) == 1
    assert metadata["node_id"] not in {row["node_id"] for row in epoch["outside_interval_edges"]}
    assert metadata["source_span"][1] < epoch["eligibility"]["arm_source_span"][0]
    original = path.read_text().splitlines()[metadata["source_span"][0]-1]
    assert "if(mode>=0)call system_clock" in original
    node = _builder.analysis.structure("supported_scope::run").source_nodes(metadata["node_id"])[0]
    assert metadata["original_source_sha256"] == sha256(str(node).encode()).hexdigest()


def test_source_bound_conditional_path_emits_one_owner_and_exact_persistent_child_controls(tmp_path):
    path, _builder, outputs, report = emit_supported_path_fixture(tmp_path)
    assert report["scope_count"] == 1, report["boundaries"]
    scope, = report["scopes"]
    assert scope["graph_root_kind"] == "supported original source path"
    plan = scope["checked_execution_plan"]
    assert plan["schema_version"] == 5
    epoch = plan["storage_epoch"]["supported_path_epoch"]
    assert epoch == scope["supported_path_epoch"]
    assert "selected_position_epoch" not in plan["storage_epoch"]
    assert "return_contract" not in plan["storage_epoch"]
    assert epoch["resources"] == ["supported_scope::run::a", "supported_scope::run::b"]
    barrier, = epoch["eligibility"]["barred_calls"]
    assert barrier["procedure"] == "supported_scope::unavailable"
    assert barrier["exposure_resources"] == []
    assert barrier["exposure_descriptor_identity"] is None
    assert all(item["control_version"] == 6 for item in scope["module_coordinators"])
    assert all(item["execution_authorized"] is False for item in [epoch])
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert " .and. eligible" in text
    run = text.lower().split("subroutine run(n,mode)", 1)[1]
    assert run.index("called=called+1") < run.index("fort_scope_serial_caller()")
    assert run.index("eligible=.false.") < run.index("call unavailable(a,b)")
    assert "unknown_private_escape" in text


@pytest.mark.parametrize("mutation", ["source", "first_span", "call_sha", "callee", "extra_key", "resource"])
def test_typed_path_request_cannot_derive_authority_from_changed_public_anchors(tmp_path, mutation):
    path, facts, contracts = fixture(tmp_path)
    request = contracts["supported_scope::run"]["native_ancestor"]["path"]
    if mutation == "source":
        request["source"]["sha256"] = "0"*64
    elif mutation == "first_span":
        request["first"]["span"][0] += 1
    elif mutation == "call_sha":
        request["first"]["sha256"] = "0"*64
    elif mutation == "callee":
        request["first"]["callee"] = "supported_scope::finish"
    elif mutation == "extra_key":
        request["guessed_authority"] = True
    else:
        request["eligibility"]["resources"][0] = "supported_scope::run::missing"
    with pytest.raises(CompilationError):
        ScopeBuilder([path], "supported_scope::run", facts=facts, options=CompilerOptions(),
                     config=OffloadConfig(policy="sections"), contracts=contracts)


@pytest.mark.parametrize("mutation", ["reverse", "duplicate"])
def test_barred_request_anchors_require_the_original_distinct_source_order(tmp_path, mutation):
    source = SOURCE.replace("else\ncall selected(a,b,n)",
        "else if(mode==2)then\neligible=.false.\ncall unavailable(a,b)\nelse\ncall selected(a,b,n)")
    path, facts, contracts = fixture(tmp_path, source)
    request = contracts["supported_scope::run"]["native_ancestor"]["path"]
    barriers = request["eligibility"]["barred_calls"]
    assert len(barriers) == 2
    request["eligibility"]["barred_calls"] = list(reversed(barriers)) if mutation == "reverse" else [barriers[0], barriers[0]]
    with pytest.raises(CompilationError, match="source order"):
        ScopeBuilder([path], "supported_scope::run", facts=facts, options=CompilerOptions(),
                     config=OffloadConfig(policy="sections"), contracts=contracts)


@pytest.mark.parametrize("mutation", ["owner", "selector", "managed", "draft"])
def test_final_epoch_and_children_retain_the_same_private_issuer_object(tmp_path, mutation):
    from compiler.frontend.supported_paths import requested_supported_path
    from compiler.scopes.supported_paths import OwnerLocalSupportedPath

    _path, builder = builder_fixture(tmp_path)
    selection = builder._native_ancestor_selection.public()
    path, gate = requested_supported_path(builder.analysis, builder.entry, selection["path_request"],
        tuple(row["procedure"] for row in selection["device_roots"]))
    owner = OwnerLocalSupportedPath(builder, path, gate)
    original = owner.position_epoch
    assert owner.run(owner.closed_selection)
    assert owner.position_epoch is original
    assert original.validate_plan(owner.execution_plan) is original
    assert all(member.builder._graph_storage_epoch is original for member in owner.members.values() if member is not owner)
    if mutation == "owner":
        object.__setattr__(owner, "position_epoch", copy(original))
    elif mutation == "selector":
        builder._native_ancestor_selection = copy(builder._native_ancestor_selection)
    elif mutation == "managed":
        object.__setattr__(original, "_managed", ())
    else:
        object.__setattr__(original, "_managed", None)
    with pytest.raises(CompilationError):
        original.validate_plan(owner.execution_plan)


@pytest.mark.native
@pytest.mark.parametrize("clock", [False, True])
def test_original_supported_interval_and_control6_entry_abis_compile(tmp_path, clock):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if fortran is None:
        pytest.skip("Fortran compiler unavailable")
    _path, _builder, _outputs, report = emit_supported_path_fixture(tmp_path, clock_source() if clock else SOURCE)
    assert report["scope_count"] == 1, report["boundaries"]
    generated = tmp_path / "generated"
    for index, role in enumerate(("common_runtime", "shared_entry", "original_source")):
        for item in report["build_sources"]:
            if item["role"] != role or item["language"] != "fortran":
                continue
            result = subprocess.run([fortran, "-fopenmp", "-ffp-contract=off", "-J", str(generated),
                "-I", str(generated), "-c", str(generated / item["path"]),
                "-o", str(generated / f"{index}-{sha256(item['path'].encode()).hexdigest()[:12]}.o")],
                capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stderr
