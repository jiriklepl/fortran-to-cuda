"""Checked private plans survive native work; every field registration retires."""
import json
import shutil
import subprocess
from copy import copy

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.provider_lifetime import PROVIDER_RETENTION_DUTIES, provider_only_lifetime
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_external_device_tasks import PROVIDER_INTERFACE, digest
from compiler.tests.test_external_device_tasks_v2 import OPAQUE_SOURCE, TYPES
from compiler.tests.test_external_entry_reads import qualified, rewrite

SOURCE = OPAQUE_SOURCE.replace("do i=1,n\nb(i)=2.0_8*a(i)\nenddo\n", "").replace(
    "do i=1,n\nb(i)=a(i)+1.0_8\nenddo\n", "")
PARENT = """module native_parent
use external_fixture, only: advance
implicit none
contains
subroutine owner(a,b,n)
real(8),intent(inout)::a(:),b(:)
integer,intent(in)::n
call middle(a,b,n)
b=a+10.0_8
if(n>0)call middle(a,b,n)
end subroutine
subroutine middle(a,b,n)
real(8),intent(inout)::a(:),b(:)
integer,intent(in)::n
call advance(a,b,n)
end subroutine
end module
"""
REQUEST = {"version": 1, "call_path": ["native_parent::owner", "native_parent::middle", "external_fixture::advance"]}


def qualify_retained_provider(tmp_path, task, manifest):
    from compiler.frontend.external_entry_reads import provider_read_binding
    from compiler.tests.test_external_device_tasks import requalify
    provider = json.loads(manifest.read_text())
    evidence = tmp_path / "provider-state-evidence.json"
    evidence.write_text(json.dumps({"test_only": True, "claim": "recording provider retains private state only"}))
    prerequisite = {"version": 1, "kind": "reviewed_external_provider_retention_prerequisite",
        "provider_binding": provider_read_binding(provider), "duties": dict(PROVIDER_RETENTION_DUTIES),
        "evidence": [{"path": str(evidence), "sha256": digest(evidence)}]}
    original = tmp_path / "provider-state-prerequisite.json"
    original.write_text(json.dumps(prerequisite))
    state = {**prerequisite, "kind": "external_provider_retained_plan_state",
             "prerequisite": {"path": str(original), "sha256": digest(original)}}
    path = tmp_path / "provider-state.json"
    path.write_text(json.dumps(state))
    provider["retained_state_qualification"] = {"path": str(path), "sha256": digest(path)}
    manifest.write_text(json.dumps(provider))
    requalify(task, manifest)


def fixture(tmp_path, *, request=REQUEST, source=PARENT, entry="external_fixture::advance"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path, _, _, _, task, manifest, qualification, record = qualified(tmp_path, SOURCE)
    parent = tmp_path / "parent.f90"
    parent.write_text(source)
    paths = [path, parent]
    analysis = SourceEffects(paths)
    record["source_inputs"] = analysis.inputs.identity()
    rewrite(task, manifest, qualification, record)
    if request is not None:
        qualify_retained_provider(tmp_path, task, manifest)
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(p): digest(p) for p in paths},
             "captures": {"argument::" + name: {"storage": "stable", "escapes": False,
                 "allocation_changes": False, "initialized": "none"} for name in ("a", "b")}}
    if request is not None:
        facts["provider_only_lifetime"] = json.loads(json.dumps(request))
    builder = ScopeBuilder(paths, entry, facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="graph"),
        contracts={"external_fixture::advance": {"device_tasks": [task]}})
    return paths, builder


def test_checked_parent_keeps_only_explicit_plan_state_and_publishes_every_return(tmp_path):
    paths, builder = fixture(tmp_path)
    outputs, report = builder.run()
    record = report["provider_only_lifetime"]
    assert record["owner"] == "native_parent::owner"
    assert [len(edge["calls"]) for edge in record["edges"]] == [2, 1]
    assert all(record[name] is False for name in (
        "field_retention_authorized", "incoming_definition_authority", "alias_authority",
        "native_effect_authority", "execution_authority"))
    leaf = outputs[report["sources"][str(paths[0])]["replacement"]]
    parent = outputs[report["sources"][str(paths[1])]["replacement"]]
    helper = outputs[record["helper"]["path"]]
    assert "type(fort_provider_lifetime), target :: fort_provider_owner" in parent
    assert "call fort_provider_reset(fort_provider_owner)" in parent
    assert "call fort_provider_retire(fort_provider_owner)" in parent
    assert "b=a+10.0_8" in parent
    assert "if (n > 0) then" in parent
    assert "if (fort_provider_checked) then" in parent
    assert "CALL advance(a, b, n)" in parent
    assert "entry " + record["checked_leaf_entry"] in leaf
    assert "fort_provider_borrow(fort_provider_life" in leaf
    assert "fort_provider_life%generation, fort_actual_layout" in leaf
    assert "fort_external_create(fort_external_domain,fort_external_config,fort_provider_life%provider)" in "".join(leaf.split()).replace("&", "")
    assert helper.index("status = fort_scope_wait(context)") < helper.index("status = fort_scope_unregister(context,fields(i))")
    assert helper.index("status = fort_scope_unregister(context,fields(i))") < helper.index("status = fort_scope_close(context)")
    assert "life%generation = life%generation + 1" in helper
    assert "SAVE" not in helper.upper()
    assert "host" not in helper.lower()
    assert "buffer" not in helper.lower()


def test_absent_selection_uses_the_existing_standalone_entry_and_record(tmp_path):
    paths, builder = fixture(tmp_path, request=None)
    outputs, report = builder.run()
    assert "provider_only_lifetime" not in report
    assert not any(path.startswith("provider_lifetimes/") for path in outputs)
    text = outputs[report["sources"][str(paths[0])]["replacement"]]
    assert "fort_provider_life%" not in text
    assert str(paths[1]) not in report["sources"]


@pytest.mark.parametrize("change", ["unknown", "cycle", "entry", "procedure_limit", "depth_limit", "return", "label", "missing", "private"])
def test_source_selection_rejects_unproved_call_and_normal_exit_ownership(tmp_path, change):
    request = json.loads(json.dumps(REQUEST))
    source = PARENT
    if change == "unknown":
        request["call_path"][1] = "native_parent::missing"
    elif change == "cycle":
        request["call_path"].insert(1, request["call_path"][0])
    elif change == "entry":
        request["call_path"][-1] = "native_parent::middle"
    elif change == "return":
        source = source.replace("b=a+10.0_8", "if(n<0)return\nb=a+10.0_8")
    elif change == "label":
        source = source.replace("b=a+10.0_8", "123 b=a+10.0_8")
    elif change == "missing":
        source = source.replace("call middle(a,b,n)", "b=b", 1).replace("if(n>0)call middle(a,b,n)", "b=b")
    elif change == "private":
        source = source.replace("integer,intent(in)::n", "integer,intent(in)::n\ninteger::fort_provider_owner", 1)
    if change in {"procedure_limit", "depth_limit"}:
        paths, builder = fixture(tmp_path)
        analysis = builder.analysis
        setattr(analysis, change, 1)
        with pytest.raises(CompilationError, match="bounded"):
            provider_only_lifetime(analysis, builder.entry.qualified, request)
    else:
        with pytest.raises(CompilationError):
            fixture(tmp_path, request=request, source=source)


def test_issued_source_path_rejects_foreign_copy_and_late_source_or_request_change(tmp_path):
    paths, builder = fixture(tmp_path)
    proof = builder.provider_only_lifetime
    with pytest.raises(CompilationError, match="exact issued"):
        copy(proof).validate()
    paths[1].write_text(paths[1].read_text() + "! changed original\n")
    with pytest.raises(CompilationError):
        proof.validate()


def compile_parent(tmp_path, *, replace_storage=False, program=False, internal=False):
    compiler = shutil.which("gfortran-15") or shutil.which("gfortran")
    assert compiler, "maintained native fixture requires its recorded Fortran compiler"
    source = PARENT
    if replace_storage:
        source = source.replace("real(8),intent(inout)::a(:),b(:)", "real(8),allocatable,intent(inout)::a(:),b(:)", 1)
        source = source.replace("call middle(a,b,n)\nb=a+10.0_8", "block\nreal(8)::previous(n)\ncall middle(a,b,n)\nprevious=a\ndeallocate(a,b)\nallocate(a(n),b(n))\na=-100.0_8\nb=previous+10.0_8\nend block")
    if internal:
        source = source.replace("end subroutine\nsubroutine middle",
            "contains\nsubroutine unchanged_internal\ninteger::x\nx=1\nend subroutine\nend subroutine\nsubroutine middle", 1)
    if program:
        source += PROGRAM
    paths, builder = fixture(tmp_path, source=source, request=PROGRAM_REQUEST if program else REQUEST)
    outputs, report = builder.run()
    build = tmp_path / "build"
    build.mkdir()
    for name, text in outputs.items():
        destination = build / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
    types, interface, query = [build / name for name in ("types.f90", "provider.f90", "query.f90")]
    types.write_text(TYPES)
    interface.write_text(PROVIDER_INTERFACE)
    query.write_text("""module external_metadata
use iso_c_binding
use external_types
use external_provider
implicit none
contains
integer(c_int) function metadata_query(config,d) result(status)
type(provider_config),intent(inout)::config
type(state_type),intent(in)::d
config%coefficient=d%scratch
config%extent=8
status=0
end function
end module
""")
    runtime = [build / item["path"] for item in report["build_sources"]
               if item["language"] == "fortran" and item["role"] == "common_runtime"]
    compile_paths = [types, *runtime, interface, query, build / report["provider_only_lifetime"]["helper"]["path"],
                     *[build / report["sources"][str(path)]["replacement"] for path in paths]]
    for index, path in enumerate(compile_paths):
        command = [compiler, "-std=f2018", "-ffree-line-length-none", "-fopenmp", "-fcheck=all", "-J", str(build), "-I", str(build),
                   "-c", str(path), "-o", str(build / f"source-{index}.o")]
        result = subprocess.run(command, cwd=build, capture_output=True, text=True, timeout=30)
        (build / f"source-{index}.command.json").write_text(json.dumps(command))
        (build / f"source-{index}.stderr").write_text(result.stderr)
        assert result.returncode == 0, result.stderr
    return build, [build / f"source-{i}.o" for i in range(1, len(compile_paths))], report


@pytest.mark.native
def test_generated_provider_only_parent_and_standalone_abis_compile_with_gnu(tmp_path):
    compile_parent(tmp_path)


def test_adjacent_original_statement_on_a_shared_call_line_is_never_dropped(tmp_path):
    source = PARENT.replace("call middle(a,b,n)\nb=a+10.0_8", "call middle(a,b,n); b=a+10.0_8")
    with pytest.raises(CompilationError, match="semicolon/co-spanning"):
        fixture(tmp_path, source=source)
    assert (tmp_path / "parent.f90").read_text() == source


@pytest.mark.parametrize(("prefix", "suffix"), [("pure ", ""), ("elemental ", ""), ("", " bind(C)")])
def test_unsupported_entry_characteristics_reject_before_emission(tmp_path, prefix, suffix):
    source = f"""module characteristics
contains
pure subroutine leaf(x)
real(8),intent(inout)::x
x=x+1
end subroutine
{prefix}subroutine owner(x){suffix}
real(8),intent(inout)::x
call leaf(x)
end subroutine
end module
"""
    path = tmp_path / "characteristics.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    analysis.seal_session()
    with pytest.raises(CompilationError, match="PURE, ELEMENTAL and BIND"):
        provider_only_lifetime(analysis, "characteristics::leaf",
            {"version": 1, "call_path": ["characteristics::owner", "characteristics::leaf"]})


def test_absent_mode_preserves_the_exact_external_patch_dataclass_inventory():
    from dataclasses import fields

    from compiler.scopes.device_tasks import ExternalDevicePatch

    assert [part.name for part in fields(ExternalDevicePatch)] == ["proof", "node", "registration", "first", "last"]


def test_retained_source_proof_rejects_a_same_record_registry_copy(tmp_path):
    _, builder = fixture(tmp_path)
    proof = builder.provider_only_lifetime
    _, node, call = proof._calls[0]
    analysis = builder.analysis
    analysis._original_call_associations[id(analysis), id(node)] = copy(call)
    with pytest.raises(CompilationError, match="call spine changed"):
        proof.validate()


def test_fresh_factory_never_renders_a_forged_cached_actual_argument(tmp_path):
    _, builder = fixture(tmp_path)
    proof = builder.provider_only_lifetime
    _, node, call = proof._calls[0]
    changed = copy(call)
    argument = copy(call.original_arguments[0])
    object.__setattr__(argument, "actual", copy(argument.actual))
    object.__setattr__(changed, "original_arguments", (argument, *call.original_arguments[1:]))
    analysis = builder.analysis
    analysis._original_call_associations[id(analysis), id(node)] = changed
    with pytest.raises(CompilationError, match="cached CALL arguments"):
        provider_only_lifetime(analysis, builder.entry.qualified, REQUEST)


@pytest.mark.parametrize("reuse", [False, True])
def test_final_context_callback_cannot_publish_changed_live_original(tmp_path, monkeypatch, reuse):
    import sys

    from compiler.frontend.authority_context import SourceAuthorityContext

    paths, builder = fixture(tmp_path)
    original = SourceAuthorityContext.require
    count = 0
    def changed(context, analysis):
        nonlocal count
        result = original(context, analysis)
        caller = sys._getframe(1)
        if caller.f_code.co_name == "_derive" and caller.f_code.co_filename.endswith("frontend/provider_lifetime.py"):
            count += 1
            if count == 2:
                paths[1].write_text(paths[1].read_text() + "! late live original mutation\n")
        return result
    monkeypatch.setattr(SourceAuthorityContext, "require", changed)
    query = builder.provider_only_lifetime.validate if reuse else lambda: provider_only_lifetime(
        builder.analysis, builder.entry.qualified, REQUEST)
    with pytest.raises(CompilationError):
        query()
    assert count >= 2


PROGRAM = '''program main_owner
use iso_c_binding
use native_parent
implicit none
interface
 integer(c_int) function main_provider_check(stage) bind(C)
 import c_int
 integer(c_int),value::stage
 end function
end interface
real(8),allocatable::a(:),b(:)
real(8)::original(8)
integer::stage,i
allocate(a(8),b(8))
original=[(real(i,8),i=1,8)]
do stage=1,2
 b=original+stage
 call owner(a,b,8)
 if(any(a/=3.0_8*(3.0_8*(original+stage)+10.0_8)))error stop 'main publication'
 if(any(b/=3.0_8*(original+stage)+10.0_8))error stop 'main native mutation'
 call checkpoint(stage)
 if(stage==1)cycle
enddo
print *, 'MAIN_PROVIDER_FIELDS_OK'
contains
subroutine checkpoint(stage)
integer,intent(in)::stage
if(main_provider_check(stage)/=0)error stop 'plan must survive timestep and fields must retire'
end subroutine
end program main_owner
'''
PROGRAM_REQUEST = {"version": 1, "call_path": ["$program::main_owner", *REQUEST["call_path"]]}


def test_explicit_program_owner_keeps_internal_procedures_and_loop_exits_original(tmp_path):
    paths, builder = fixture(tmp_path, request=PROGRAM_REQUEST, source=PARENT + PROGRAM)
    program = builder.analysis.routines["$program::main_owner"]
    assert program.source_kind == "program"
    assert "save" in program.scope.bindings["a"].attributes
    outputs, report = builder.run()
    record = report["provider_only_lifetime"]
    assert record["members"][0]["source_kind"] == "program"
    text = outputs[report["sources"][str(paths[1])]["replacement"]]
    original_start = text.lower().index("program main_owner")
    owner = text[original_start:]
    assert owner.index("call fort_provider_reset(fort_provider_owner)") < owner.index("allocate(a(8),b(8))")
    assert owner.index("call fort_provider_retire(fort_provider_owner)") < owner.index("contains")
    assert "if(stage==1)cycle" in owner
    assert PROGRAM[PROGRAM.index("contains"):] in owner
    assert "$program::" not in owner
    assert record["field_retention_authorized"] is False


def test_program_metadata_is_opt_in_and_never_an_executable_entry(tmp_path):
    paths, _ = fixture(tmp_path, request=None, source=PARENT + PROGRAM)
    assert not any(name.startswith("$program::") for name in SourceEffects(paths).routines)
    with pytest.raises(CompilationError, match="cannot be selected as an executable"):
        ScopeBuilder(paths, "$program::main_owner", facts={"schema_version": 1, "participation": "serial",
            "captures": {}, "sources": {str(path): digest(path) for path in paths},
            "provider_only_lifetime": PROGRAM_REQUEST}, options=CompilerOptions(),
            config=OffloadConfig(policy="sections", scope_execution="graph"))


@pytest.mark.parametrize("change", ["stop", "goto", "semicolon", "unnamed"])
def test_program_owner_requires_exact_named_source_and_normal_teardown(tmp_path, change):
    source = PARENT + PROGRAM
    if change == "stop":
        source = source.replace("call checkpoint(stage)", "if(stage==1)stop\n call checkpoint(stage)")
    elif change == "goto":
        source = source.replace("call checkpoint(stage)", "if(stage==1)goto 100\n100 call checkpoint(stage)")
    elif change == "semicolon":
        source = source.replace("call owner(a,b,8)", "call owner(a,b,8); b=b+1")
    else:
        source = source.replace("program main_owner\n", "", 1).replace("end program main_owner", "end")
    with pytest.raises(CompilationError):
        fixture(tmp_path, request=PROGRAM_REQUEST, source=source)


def test_program_and_internal_name_collisions_do_not_expand_executable_entry_candidates(tmp_path):
    program = PROGRAM.replace("program main_owner", "program advance")
    program = program.replace("end program advance", "subroutine advance\nend subroutine\nend program advance")
    request = {"version": 1, "call_path": ["$program::advance", *REQUEST["call_path"]]}
    _, builder = fixture(tmp_path, request=request, source=PARENT + program, entry="advance")
    assert builder.entry.qualified == "external_fixture::advance"
    assert "$program::advance::advance" in builder.analysis.routines
    with pytest.raises(CompilationError, match="cannot be selected as an executable"):
        fixture(tmp_path / "explicit", request=request, source=PARENT + program, entry="$program::advance::advance")


@pytest.mark.parametrize("directive", ["parallel", "task", "parallel do", "single"])
def test_selected_call_spine_never_shares_plan_slot_through_unproved_openmp(tmp_path, directive):
    source = PARENT.replace("call middle(a,b,n)", "!$omp " + directive + "\ncall middle(a,b,n)\n!$omp end " + directive, 1)
    with pytest.raises(CompilationError, match="rejects OpenMP"):
        fixture(tmp_path, source=source)


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("field", ["operation_limit", "depth_limit", "contracts"])
def test_final_publication_cannot_invalidate_owned_policy_context(tmp_path, monkeypatch, reuse, field):
    _, builder = fixture(tmp_path)
    analysis = builder.analysis
    original = SourceEffects.verify_publication
    def changed(current):
        result = original(current)
        if current is analysis:
            setattr(current, field, {} if field == "contracts" else getattr(current, field) + 1)
        return result
    monkeypatch.setattr(SourceEffects, "verify_publication", changed)
    query = builder.provider_only_lifetime.validate if reuse else lambda: provider_only_lifetime(
        analysis, builder.entry.qualified, REQUEST)
    with pytest.raises(CompilationError, match="source or namespace changed"):
        query()


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("which", ["routine", "argument"])
def test_final_callbacks_cannot_replace_consumed_typed_routine_or_call_fields(tmp_path, monkeypatch, reuse, which):
    from compiler.frontend.call_bindings import ResolvedOriginalCall
    _, builder = fixture(tmp_path)
    analysis = builder.analysis
    original_publication = SourceEffects.verify_publication
    original_public = ResolvedOriginalCall.public
    fired = []
    if which == "routine":
        target = builder.provider_only_lifetime._routines[0]
        def publication(current):
            result = original_publication(current)
            if current is analysis and not fired:
                fired.append(True)
                object.__setattr__(target, "arguments", ("forged_argument",))
            return result
        monkeypatch.setattr(SourceEffects, "verify_publication", publication)
    else:
        def public(token):
            result = original_public(token)
            if not fired:
                fired.append(True)
                object.__setattr__(token.original_arguments[0], "keyword", "forged_keyword")
            return result
        monkeypatch.setattr(ResolvedOriginalCall, "public", public)
    query = builder.provider_only_lifetime.validate if reuse else lambda: provider_only_lifetime(
        analysis, builder.entry.qualified, REQUEST)
    with pytest.raises(CompilationError, match="source or namespace changed"):
        query()
    assert fired


def test_call_path_request_cannot_retain_unqualified_provider_state(tmp_path):
    paths, builder = fixture(tmp_path, request=None)
    facts = {**builder.facts, "provider_only_lifetime": REQUEST}
    guarded = ScopeBuilder(paths, builder.entry.qualified, facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="graph"), contracts=builder.analysis.contracts)
    with pytest.raises(CompilationError, match="explicit retained-state provider qualification"):
        guarded.run()


@pytest.mark.native
def test_module_owner_retirement_precedes_its_original_internal_procedures(tmp_path):
    build, _, report = compile_parent(tmp_path, internal=True)
    original_parent = tmp_path / "parent.f90"
    output = (build / report["sources"][str(original_parent)]["replacement"]).read_text()
    owner = output[output.lower().index("subroutine owner"):]
    assert owner.index("call fort_provider_retire(fort_provider_owner)") < owner.index("contains")
    assert "subroutine unchanged_internal\ninteger::x\nx=1\nend subroutine" in owner


@pytest.mark.parametrize("reuse", [False, True])
def test_delegated_source_check_cannot_rebaseline_initial_policy(tmp_path, monkeypatch, reuse):
    import compiler.frontend.provider_lifetime as issuing
    _, builder = fixture(tmp_path)
    original = issuing._original
    fired = []
    def changed(analysis, name):
        result = original(analysis, name)
        if not fired:
            fired.append(True)
            analysis.operation_limit += 1
        return result
    monkeypatch.setattr(issuing, "_original", changed)
    query = builder.provider_only_lifetime.validate if reuse else lambda: provider_only_lifetime(
        builder.analysis, builder.entry.qualified, REQUEST)
    with pytest.raises(CompilationError, match="source or namespace changed|association inspection budgets changed"):
        query()
    assert fired
