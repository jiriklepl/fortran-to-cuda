"""Local checked groups retire before returning to unrelated native source."""
import json
import shutil
import subprocess
from copy import copy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder

SOURCE = """module local_groups
contains
subroutine first(b,n)
integer,intent(in)::n
real(8),intent(out)::b(:)
integer::i
do i=1,n
b(i)=real(i,8)
enddo
end subroutine
subroutine second(a,b,n)
integer,intent(in)::n
real(8),intent(out)::a(:)
real(8),intent(in)::b(:)
integer::i
do i=1,n
a(i)=2.0_8*b(i)
enddo
end subroutine
subroutine third(a,b,n)
integer,intent(in)::n
real(8),intent(inout)::a(:)
real(8),intent(in)::b(:)
integer::i
do i=1,n
a(i)=a(i)+b(i)
enddo
end subroutine
subroutine run(a,b,n,flag)
real(8),allocatable,intent(inout)::a(:),b(:)
integer,intent(in)::n
logical,intent(in)::flag
call unknown_prefix()
if(flag)then
call first(b,n)
call second(a,b,n)
call third(a,b,n)
endif
call unknown_suffix()
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE, *, lifetime="selected_window", unused=False):
    if unused:
        source = source.replace("subroutine run(a,b,n,flag)", "subroutine run(a,b,n,flag,unused)")
        source = source.replace("real(8),allocatable,intent(inout)::a(:),b(:)",
                                "real(8),allocatable,intent(inout)::a(:),b(:),unused(:)")
    path = tmp_path / "local_groups.f90"
    path.write_text(source)
    fact = {"storage": "stable", "initialized": "none", "escapes": False,
            "allocation_changes": False, "lifetime": lifetime}
    facts = {"schema_version": 1, "participation": "serial",
        "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
        "captures": {"argument::a": dict(fact), "argument::b": dict(fact)}}
    if unused:
        facts["captures"]["argument::unused"] = dict(fact)
    contracts = {"local_groups::run": {"native_ancestor": {"version": 2,
        "lifetime": "owner_local_call_groups", "device_roots": ["local_groups::"+name for name in ["first", "second", "third"]]}}}
    builder = ScopeBuilder([path], "local_groups::run", facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections"), contracts=contracts)
    return path, builder


def publish(tmp_path, outputs, report):
    (tmp_path / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, source in outputs.items():
        destination = tmp_path / "generated" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source)


@pytest.mark.parametrize("lifetime", ["selected_window", "invocation"])
def test_three_branch_ending_children_share_only_owner_local_epoch(tmp_path, lifetime):
    path, builder = fixture(tmp_path, lifetime=lifetime)
    outputs, report = builder.run()
    publish(tmp_path, outputs, report)
    assert report["scope_count"] == 1, report["boundaries"]
    scope, = report["scopes"]
    plan = scope["checked_execution_plan"]
    assert plan["schema_version"] == 4
    epoch = plan["storage_epoch"]["selected_position_epoch"]
    assert epoch["version"] == 2
    assert epoch == scope["owner_local_call_group"]
    assert epoch["sequence_exit"]["exit_anchor"]["parser_kind"] == "End_If_Stmt"
    assert epoch["selection_identity"] == report["native_ancestor_selection"]["identity"]
    assert "return_contract" not in plan["storage_epoch"]
    assert len(epoch["calls"]) == 3
    assert all(len(call["source_span"]) == 2 for call in epoch["calls"])
    replacement = outputs[report["sources"][str(path)]["replacement"]]
    lower = replacement.lower().split("subroutine run(a,b,n,flag)", 1)[1]
    assert lower.index("call unknown_prefix()") < lower.index("if(flag)then")
    assert lower.index("fort_scope_serial_caller()") > lower.index("if(flag)then")
    assert lower.index("fort_scope_resident_close_v1") < lower.index("call unknown_suffix()")
    assert "fort_child_retention_" not in replacement
    assert all(member["control_version"] == 4 for member in scope["module_coordinators"])


def test_unused_capture_is_only_a_descriptor_guard_without_registration(tmp_path):
    path, builder = fixture(tmp_path, unused=True)
    outputs, report = builder.run()
    publish(tmp_path, outputs, report)
    assert report["scope_count"] == 1, report["boundaries"]
    scope, = report["scopes"]
    epoch = scope["checked_execution_plan"]["storage_epoch"]["selected_position_epoch"]
    assert epoch["resources"] == ["argument::a", "argument::b"]
    assert epoch["descriptor_guard_resources"] == ["argument::unused"]
    assert "unused" not in json.dumps(scope["checked_execution_plan"]["storage_epoch"]["resources"])
    root = outputs[report["sources"][str(path)]["replacement"]].lower().split(
        "subroutine run(a,b,n,flag,unused)", 1)[1]
    assert "allocated(unused)" not in root
    assert "lbound(unused" not in root
    assert "c_loc(unused" not in root


def test_original_address_escape_attributes_cannot_be_discharged_by_a_fresh_group(tmp_path):
    source = SOURCE.replace("module local_groups\n", "module local_groups\nuse iso_c_binding\n"
                            "type(c_ptr),save::escaped\n")
    source = source.replace("real(8),allocatable,intent(inout)::a(:),b(:)",
                            "real(8),allocatable,target,intent(inout)::a(:),b(:)")
    source = source.replace("call unknown_prefix()", "escaped=c_loc(a)\ncall unknown_prefix()")
    _path, builder = fixture(tmp_path, source)
    _outputs, report = builder.run()
    assert report["scope_count"] == 0
    assert not report["source_edits"]


@pytest.mark.parametrize("mutation", ["owner", "selector", "exit", "managed", "draft_inventory"])
def test_sealed_local_epoch_rejects_changed_consumer_exit_or_final_inventory(tmp_path, mutation):
    from compiler.scopes.call_groups import OwnerLocalCallGroup, _candidates

    _path, builder = fixture(tmp_path)
    owner = OwnerLocalCallGroup(builder, next(_candidates(builder)))
    assert owner.run(owner.closed_selection)
    epoch = owner.position_epoch
    assert epoch.validate_plan(owner.execution_plan) is epoch
    if mutation == "owner":
        object.__setattr__(owner, "position_epoch", copy(epoch))
    elif mutation == "selector":
        builder._native_ancestor_selection = copy(builder._native_ancestor_selection)
    elif mutation == "exit":
        object.__setattr__(owner, "local_exit", copy(owner.local_exit))
    elif mutation == "managed":
        object.__setattr__(epoch, "_managed", ())
    else:
        object.__setattr__(epoch, "_managed", None)
    with pytest.raises(CompilationError):
        epoch.validate_plan(owner.execution_plan)


@pytest.mark.native
def test_published_local_group_and_child_entry_abis_compile_with_real_runtime_module(tmp_path):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    path, builder = fixture(tmp_path, unused=True)
    outputs, report = builder.run()
    publish(tmp_path, outputs, report)
    directory = tmp_path / "generated"
    sources = [item["path"] for role in ("common_runtime", "shared_entry", "original_source")
               for item in report["build_sources"] if item["role"] == role and item["language"] == "fortran"]
    for index, name in enumerate(sources):
        result = subprocess.run([fortran, "-fopenmp", "-ffp-contract=off", "-J", str(directory),
                                 "-I", str(directory), "-c", str(directory / name),
                                 "-o", str(directory / f"abi-{index}.o")],
                                capture_output=True, text=True, timeout=60)
        (directory / f"abi-{index}.stdout").write_text(result.stdout)
        (directory / f"abi-{index}.stderr").write_text(result.stderr)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("source", [SOURCE.replace("call second(a,b,n)", "call second(a,a,n)"),
    SOURCE.replace("call second(a,b,n)\n", "call second(a,b,n);call third(a,b,n)\n").replace("call third(a,b,n)\nendif", "endif"),
    SOURCE.replace("call first(b,n)", "call unknown_effect();call first(b,n)")])
def test_alias_or_shared_edit_lines_do_not_publish_a_partial_group(tmp_path, source):
    _path, builder = fixture(tmp_path, source)
    _outputs, report = builder.run()
    assert report["scope_count"] == 0
    assert not report["source_edits"]
