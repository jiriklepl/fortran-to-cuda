"""Source ownership survives reached guards and coherent native operations."""

from __future__ import annotations

import re

import pytest

from compiler.tests.test_source_scopes import FACT, PROGRAM, generate

TREE = PROGRAM.replace("integer,intent(in)::n\ncall producer(a,b,n)",
                       "integer,intent(inout)::n\nlogical::flag\ncall producer(a,b,n)")
TREE = TREE.replace("subroutine step(a,b,out,n)\nreal(8),intent(in)::a(:)\n"
                    "real(8),intent(out)::b(:),out(:)",
                    "subroutine step(a,b,out,n)\nreal(8),intent(in)::a(-2:)\n"
                    "real(8),intent(inout)::b(-2:),out(-2:)")
TREE = TREE.replace("call transform(b)\ncall consumer(a,b,out,n)\nend subroutine",
                    "flag=sum(b)>0\nif(flag) then\n"
                    "  if(b(-2)>0) b(-2)=b(-2)+7\n"
                    "else if(n<0) then\n  call consumer(a,b,out,n)\n"
                    "else\n  call transform(b)\nendif\nn=n-1\n"
                    "call consumer(a,b,out,n)\nend subroutine")


def generate_tree(directory, source=TREE, *, mode="sections", profile=None):
    facts = {"schema_version": 1, "participation": "serial",
             "captures": {"argument::a": FACT, "argument::b": FACT, "argument::out": FACT}}
    original, output, manifest = generate(directory, source, mode=mode, facts=facts, profile=profile)
    text = (output / manifest["sources"][str(original)]["replacement"]).read_text()
    return original, output, manifest, text


def owner_text(manifest, text):
    owner, = manifest["scopes"]
    name = owner["owner"]
    return owner, text[text.index("subroutine " + name):text.index("end subroutine " + name)]


def test_structured_owner_queries_only_reached_segments_and_preserves_bounds(tmp_path):
    original, _, manifest, text = generate_tree(tmp_path)
    assert original.read_text() == TREE
    scope, owner = owner_text(manifest, text)
    assert scope["ownership"]["retained_resources"] == ["argument::a", "argument::b", "argument::out"]
    assert scope["ownership"]["close_count"] == 1
    assert scope["ownership"]["planning_mode"] == "continuation"
    assert len(scope["planning_segments"]) == 4
    assert "fort_scope_plan_reset_mode(fort_context, FORT_SCOPE_PLAN_CONTINUE)" in owner
    assert "gpu_units == 0" not in owner
    assert "fort_scope_plan_reset(fort_context)" not in owner
    captures = {entry["resource"]: entry["name"] for entry in scope["parameters"]}
    assert captures["argument::b"] + "_view(- 2)" in owner
    assert captures["argument::b"] + "_lower(1):" in owner
    caller = text[text.index("subroutine step"):text.index("end subroutine", text.index("subroutine step"))]
    kind_import = re.search(r"use iso_c_binding, only: (\w+) => c_int64_t", caller)
    assert kind_import is not None
    kind_alias = kind_import.group(1)
    assert re.search(rf"lbound\(\s*b\s*,\s*1\s*,\s*kind\s*=\s*{re.escape(kind_alias)}\s*\)", caller)
    assert "fort_branch =" in owner
    assert owner.index("fort_branch =") < owner.index("if (fort_branch) then")
    assert all(segment["position"] == "when reached after preceding source operations"
               for segment in scope["planning_segments"])
    assert all(operation["planned_effects"] for operation in scope["native_operations"])
    assert any(operation["sections"]["available"] for operation in scope["native_operations"]
               if operation["kind"] == "native source")
    # Descriptor evaluation is behind serial participation even for ordinary
    # formal captures; the original body remains available before execution.
    step = text[text.index("subroutine step"):text.index("end subroutine", text.index("subroutine step"))]
    assert step.index("fort_scope_serial_") < step.index("lbound(")


def test_missing_calibration_keeps_coherent_native_segment_not_owner_replay(tmp_path):
    _, _, manifest, text = generate_tree(tmp_path, mode="auto")
    scope, owner = owner_text(manifest, text)
    assert not scope["estimate_available"]
    assert "fort_choose(fort_context, fort_decision)" in owner
    assert "gpu_units == 0" not in owner
    assert "fort_scope_wait(fort_context)" in owner
    assert owner.rfind("fort_scope_close(fort_context)") > owner.rfind("fort_scope_wait(fort_context)")


def test_native_scalar_change_splits_future_queries(tmp_path):
    change = "subroutine change(n)\ninteger,intent(inout)::n\nn=n-1\nend subroutine\n"
    source = TREE.replace("n=n-1\ncall consumer", "call change(n)\ncall consumer").replace("end module", change + "end module")
    _, _, manifest, text = generate_tree(tmp_path, source)
    scope, owner = owner_text(manifest, text)
    consumer = [segment for segment in scope["planning_segments"] if segment["calls"] == ["original::consumer"]]
    assert len(consumer) == 2
    assert all(segment["query_available"] for segment in consumer)
    last_change = owner.rfind("call change(")
    assert last_change >= 0
    assert owner.index("fort_scope_query_", last_change) > last_change


def test_call_only_scalar_mutation_uses_continuations_without_changing_legacy_scopes(tmp_path):
    change = "subroutine change(n)\ninteger,intent(inout)::n\nn=n-1\nend subroutine\n"
    source = PROGRAM.replace("integer,intent(in)::n\ncall producer(a,b,n)",
                             "integer,intent(inout)::n\ncall producer(a,b,n)").replace(
        "call transform(b)", "call change(n)").replace("end module", change + "end module")
    _, _, manifest, text = generate_tree(tmp_path, source)
    scope, owner = owner_text(manifest, text)
    assert scope["ownership"]["planning_mode"] == "continuation"
    assert [segment["calls"] for segment in scope["planning_segments"]] == [
        ["original::producer"], ["original::change"], ["original::consumer"]]
    assert "gpu_units == 0" not in owner


def test_partial_definition_condition_has_exact_point_read_and_native_overwrite(tmp_path):
    source = TREE.replace("flag=sum(b)>0", "flag=.true.")
    facts = {"schema_version": 1, "participation": "serial",
             "captures": {"argument::a": FACT, "argument::b": {**FACT, "initialized": "none"}, "argument::out": FACT}}
    original, output, manifest = generate(tmp_path, source, facts=facts)
    scope, = manifest["scopes"]
    operation, = [operation for operation in scope["native_operations"]
                  if operation["kind"] == "condition read" and operation["resources"] == ["argument::b"]]
    assert operation["sections"]["available"]
    rectangle, = operation["sections"]["resources"][0]["reads"]
    assert rectangle["axes"] == [{"kind": "point", "lower": {"kind": "literal", "value": -2},
                                  "upper": {"kind": "literal", "value": -2}}]
    text = (output / manifest["sources"][str(original)]["replacement"]).read_text()
    _, owner = owner_text(manifest, text)
    assert "%read_count =" in owner
    assert "%overwrite_count =" in owner


def test_unknown_condition_point_on_partial_definition_ends_owner_safely(tmp_path):
    source = TREE.replace("flag=sum(b)>0", "flag=.true.").replace("b(-2)>0", "b(n-3)>0")
    _, _, manifest = generate(tmp_path, source, facts={
        "schema_version": 1, "participation": "serial",
        "captures": {"argument::a": FACT, "argument::b": {**FACT, "initialized": "none"}, "argument::out": FACT}})
    assert not any(scope.get("ownership") for scope in manifest["scopes"])
    assert any("unknown native footprint requires complete definitions" in boundary["reason"]
               for boundary in manifest["boundaries"])


def test_inline_parameter_capture_is_read_only(tmp_path):
    source = TREE.replace("logical::flag", "logical::flag\ninteger,parameter::fixed=2").replace("n=n-1", "n=n-fixed")
    _, _, manifest, text = generate_tree(tmp_path, source)
    scope, owner = owner_text(manifest, text)
    parameter, = [item["name"] for item in scope["parameters"] if item["resource"] == "original::step::fixed"]
    assert "integer(c_int), intent(in) :: " + parameter in owner


def test_hidden_mutable_scalar_retains_original_module_binding(tmp_path):
    source = TREE.replace("implicit none", "implicit none\ninteger,save::limit=0,next_limit=0", 1).replace(
        "n=n-1\ncall consumer(a,b,out,n)", "next_limit=n-1\ncall set_limit()\nif(limit>0) then\ncall consumer(a,b,out,limit)\nendif").replace(
        "end module", "subroutine set_limit()\nlimit=next_limit\nend subroutine\nend module")
    _, _, manifest, text = generate_tree(tmp_path, source)
    scope, owner = owner_text(manifest, text)
    assert not any(item["resource"] == "original::limit" for item in scope["parameters"])
    assert {entry["resource"] for entry in scope["original_scalar_bindings"]} == {"original::limit", "original::next_limit"}
    assert "SAVE" not in owner
    assert owner.rfind("call set_limit(") < owner.rfind("fort_branch = limit")


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("target", [False, True])
def test_hidden_array_native_alias_requires_original_target(tmp_path, structured, target):
    source = PROGRAM.replace("implicit none", "implicit none\nreal(8)" + (",target" if target else "") + "::field(32)", 1)
    before, entry = source.split("subroutine step", 1)
    entry = entry.replace("call producer(a,b,n)", "call producer(a,field,n)").replace(
        "call transform(b)", "call touch_field()").replace("call consumer(a,b,out,n)", "call consumer(a,field,out,n)")
    if structured:
        entry = entry.replace("call consumer(a,field,out,n)", "if(n>1) then\ncall consumer(a,field,out,n)\nendif")
    source = before + "subroutine step" + entry.replace("end module", "subroutine touch_field()\n"
                                                       "field(1)=field(1)+1\nend subroutine\nend module")
    _, _, manifest = generate(tmp_path, source, facts={"schema_version": 1, "participation": "serial",
                                                    "captures": {"argument::a": FACT, "argument::out": FACT, "original::field": FACT}})
    if target:
        assert manifest["scope_count"] == 1, manifest["boundaries"]
    else:
        assert manifest["scope_count"] == 0
        assert any("native host/use array alias requires original TARGET" in boundary["reason"]
                   for boundary in manifest["boundaries"])


def test_original_allocation_predicate_is_guarded_before_association(tmp_path):
    source = TREE.replace("real(8),intent(out)::b(:)", "real(8),intent(inout)::b(:)")
    source = source.replace("real(8),intent(in)::a(-2:)", "real(8),allocatable,intent(in)::a(:)").replace(
        "real(8),intent(inout)::b(-2:),out(-2:)", "real(8),allocatable,intent(inout)::b(:),out(:)")
    source = source.replace("if(flag) then", "if(allocated(b).and.flag) then")
    _, _, manifest, text = generate_tree(tmp_path, source)
    scope, owner = owner_text(manifest, text)
    assert scope["allocation_preflight"]["resources"] == ["argument::a", "argument::b", "argument::out"]
    assert "ALLOCATED(fort_capture_" not in owner
    assert "fort_branch = .TRUE. .AND." in owner
    guard = text[text.index("subroutine step"):text.index("end subroutine", text.index("subroutine step"))]
    kind_import = re.search(r"use iso_c_binding, only: (\w+) => c_int64_t", guard)
    assert kind_import is not None
    bound = re.search(rf"lbound\(\s*b\s*,\s*1\s*,\s*kind\s*=\s*{re.escape(kind_import.group(1))}\s*\)", guard)
    assert bound is not None
    assert guard.index("allocated(b)") < bound.start()


@pytest.mark.parametrize("statement", ["deallocate(b)", "return", "exit", "call unknown(b)"])
def test_unknown_or_lifetime_operations_end_structured_ownership(tmp_path, statement):
    source = TREE.replace("n=n-1", statement)
    if statement.startswith("deallocate"):
        source = source.replace("real(8),intent(inout)::b(-2:),out(-2:)",
                                "real(8),allocatable,intent(inout)::b(:)\nreal(8),intent(inout)::out(-2:)")
    _, _, manifest = generate(tmp_path, source, facts={"schema_version": 1, "participation": "serial",
                                                    "captures": {"argument::a": FACT, "argument::b": FACT, "argument::out": FACT}})
    assert not any(scope.get("ownership") and scope["last_line"] > source.splitlines().index(statement) + 1
                   and scope["first_line"] < source.splitlines().index(statement) + 1 for scope in manifest["scopes"])


def test_condition_rewrite_uses_binding_nodes_and_keeps_literals(tmp_path):
    source = TREE.replace("if(flag) then", "if(flag.and.b(-2)>-99) then ! b is deliberately mentioned\n")
    _, _, manifest, text = generate_tree(tmp_path, source)
    scope, owner = owner_text(manifest, text)
    assert scope["ownership"]
    assert re.search(r"fort_branch = .*fort_capture_.*_view\(- 2\)", owner)


JOINED = TREE.replace("real(8),intent(out)::b(:)", "real(8),intent(inout)::b(:)").replace("logical::flag", "logical::flag\ninteger::i").replace(
    "n=n-1\ncall consumer", "!$omp parallel private(i) shared(b,n)\n!$omp do\n"
    "do i=-2,n-3\nb(i)=b(i)+1\nenddo\n!$omp end do nowait\n"
    "!$omp end parallel\nn=n-1\ncall consumer")


def test_joined_native_team_retains_and_renames_original_private_variables(tmp_path, monkeypatch):
    # Exercise the native operation independently of optional inline numerical
    # admission; a legal joined region remains inside its owner either way.
    from compiler.ir import CompilationError
    from compiler.driver.options import CompilerOptions
    from compiler.offload.config import OffloadConfig
    from compiler.scopes.source import ScopeBuilder
    from compiler.scopes import region_dispatch
    from hashlib import sha256

    def native_region(*args, **kwargs):
        raise CompilationError("inline numerical admission unavailable")

    monkeypatch.setattr(region_dispatch, "extract_region", native_region)
    original = tmp_path / "original.f90"
    original.write_text(JOINED)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(original): sha256(original.read_bytes()).hexdigest()},
             "captures": {"argument::a": FACT, "argument::b": FACT, "argument::out": FACT}}
    outputs, manifest = ScopeBuilder([original], "original::step", facts=facts,
                                     options=CompilerOptions(), config=OffloadConfig(policy="sections")).run()
    text = outputs[manifest["sources"][str(original)]["replacement"]]
    scope, owner = owner_text(manifest, text)
    operation, = [item for item in scope["native_operations"] if item["kind"] == "joined native OpenMP"]
    assert operation["completion"]["available"]
    captures = {entry["resource"]: entry["name"] for entry in scope["parameters"]}
    iterator = captures["original::step::i"]
    assert "!$omp parallel private(" + iterator + ") shared(" in owner
    assert "DO " + iterator + " = - 2" in owner
    assert "!$omp end do nowait" in owner
    assert "!$omp end parallel" in owner


@pytest.mark.parametrize("replacement", ["!$omp end parallel nowait", "!$omp task", "!$omp parallel firstprivate(i)"])
def test_unproved_native_team_completion_remains_an_explained_boundary(tmp_path, replacement):
    directive = "!$omp parallel private(i) shared(b,n)" if "parallel firstprivate" in replacement else "!$omp end parallel"
    _, _, manifest = generate(tmp_path, JOINED.replace(directive, replacement), facts={
        "schema_version": 1, "participation": "serial",
        "captures": {"argument::a": FACT, "argument::b": FACT, "argument::out": FACT}})
    assert any("parallel" in boundary["reason"].lower() or "openmp" in boundary["reason"].lower()
               for boundary in manifest["boundaries"])
