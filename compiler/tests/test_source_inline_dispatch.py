"""Saved allocations remain owned by the original procedure around borrowed loops."""

from hashlib import sha256

from compiler.driver.options import CompilerOptions
from compiler.frontend import lower_source
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

PROGRAM = """module saved_work
implicit none
contains
subroutine step(a,out,n)
real(8),intent(in)::a(-2:)
real(8),intent(inout)::out(-2:)
integer,intent(in)::n
real(8),allocatable,save::scratch(:)
integer,save::calls=0
integer::i
if(allocated(scratch)) then
 if(size(scratch)/=n) deallocate(scratch)
endif
if(.not.allocated(scratch)) allocate(scratch(-2:n-3))
calls=calls+1
do i=-2,n-3
scratch(i)=2*a(i)+real(i,8)
enddo
do i=-2,n-3
out(i)=scratch(i)+a(i)
enddo
end subroutine
end module
"""


def build(tmp_path, source=PROGRAM, *, policy="sections", local_fact=True):
    path = tmp_path / "saved.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::a": FACT, "argument::out": FACT,
                          "saved_work::step::scratch": {**FACT, "initialized": "none"}}}
    if not local_fact:
        del facts["captures"]["saved_work::step::scratch"]
    compiler = ScopeBuilder([path], "saved_work::step", facts=facts, options=CompilerOptions(),
                            config=OffloadConfig(policy=policy))
    outputs, report = compiler.run()
    return outputs, report


def test_inline_saved_storage_uses_one_dispatcher_and_one_owner(tmp_path):
    outputs, report = build(tmp_path)
    assert report["scope_count"] == 1, report["boundaries"]
    inline = report["inline_numerical_regions"]
    assert inline["dispatcher"]
    assert [region["region_id"] for region in inline["regions"]] == [1, 2]
    assert all(region["used"] for region in inline["regions"])
    scope, = report["scopes"]
    assert len(scope["gpu_leaves"]) == 2
    assert scope["allocation_preflight"]["resources"] == ["saved_work::step::scratch"]
    assert scope["allocation_preflight"]["bounds_guard"]["resources"] == [
        "argument::a", "argument::out", "saved_work::step::scratch"]
    variants, = report["implementation_variants"]["procedures"]
    assert {variant["role"] for variant in variants["variants"]} == {"coordinator", "region_dispatcher"}
    generated = outputs[next(path for path in outputs if path.startswith("sources/"))]
    assert generated.count("allocatable,save::scratch") == 1
    assert generated.count("integer,save::calls=0") == 1
    assert "integer(c_int), intent(inout)" in generated
    dispatch, = [text for path, text in outputs.items() if path.startswith("regions/") and path.endswith(".f90")]
    assert "select case (region)" in dispatch
    assert "case (1)" in dispatch
    assert "case (2)" in dispatch
    assert all(source["path"] in outputs for source in report["build_sources"])


def test_missing_local_allocation_facts_preserves_the_original_loop(tmp_path):
    outputs, report = build(tmp_path, local_fact=False)
    assert report["scope_count"] == 0
    assert any("missing stable storage/definition facts" in item["reason"] for item in report["boundaries"])
    assert report["source_edits"] == []
    assert set(outputs) == {"scope-manifest.json"}


def test_identical_inline_computation_reuses_its_entry_with_distinct_region_ids(tmp_path):
    source = PROGRAM.replace("out(i)=scratch(i)+a(i)", "scratch(i)=2*a(i)+real(i,8)")
    _, report = build(tmp_path, source)
    assert report["scope_count"] == 1, report["boundaries"]
    first, second = report["inline_numerical_regions"]["regions"]
    assert first["region_id"] != second["region_id"]
    assert first["shared_computation"] == second["shared_computation"]
    entries = {source["path"] for source in report["build_sources"] if source["path"].endswith("/shared_entry.cu")}
    assert len(entries) == 1


def test_compiler_owned_source_lowering_has_a_deterministic_identity():
    source = """module prepared
contains
subroutine f(a)
real(8),intent(inout)::a(:)
integer::i
do i=1,size(a)
a(i)=2*a(i)
enddo
end subroutine
end module
"""
    first = lower_source(source, "prepared::f", source_name="original.f90#inline:identity")
    second = lower_source(source, "prepared::f", source_name="original.f90#inline:identity")
    assert first == second
    assert first.source == "original.f90#inline:identity"


def test_reached_inline_segments_keep_native_partial_writes_and_mutable_controls(tmp_path):
    source = PROGRAM.replace("integer::i", "integer::i,extent").replace(
        "do i=-2,n-3\nout(i)=scratch(i)+a(i)\nenddo",
        "out(-2)=out(-2)+1\nif(n>0) then\nextent=n\n"
        "do i=-2,extent-3\nout(i)=scratch(i)+a(i)\nenddo\nendif")
    outputs, report = build(tmp_path, source)
    assert report["scope_count"] == 1, report["boundaries"]
    scope, = report["scopes"]
    assert len(scope["planning_segments"]) == 2
    assert any(item["kind"] == "branch" for item in scope["structured_tree"]["nodes"])
    assert scope["definition_preflight"]["position"] == "when segment is reached"
    generated = outputs[next(path for path in outputs if path.startswith("sources/"))]
    assert "fort_scope_plan_reset_mode" in generated
    assert "call " + report["inline_numerical_regions"]["dispatcher"].split("::")[0] + "()" not in generated
    assert "fort_inline_query(" in generated
    assert "0_c_int" in generated  # Inactive dispatcher controls do not read extent.
    assert "DO i = - 2" in generated  # Native fallback expands every private facade.


def test_rejected_inline_attempts_are_bounded_before_lowering(tmp_path, monkeypatch):
    from compiler.ir import CompilationError
    from compiler.scopes import region_dispatch
    source = PROGRAM[:PROGRAM.index("do i=-2,n-3")] + (
        "do i=-2,n-3\nscratch(i)=2*a(i)+real(i,8)\nenddo\n" * 40) + "end subroutine\nend module\n"
    attempts = []

    def decline(*args, **kwargs):
        attempts.append(args[1])
        raise CompilationError("unsupported computation in admitted region")
    monkeypatch.setattr(region_dispatch, "lower_source", decline)
    outputs, report = build(tmp_path, source)
    assert len(attempts) == 32
    assert report["scope_count"] == 0
    assert any("bounded inline region/operation budget exhausted" in item["reason"] for item in report["boundaries"])
    assert set(outputs) == {"scope-manifest.json"}


def test_inline_operation_budget_rejects_before_lowering(tmp_path, monkeypatch):
    from compiler.scopes import region_dispatch
    source = PROGRAM[:PROGRAM.index("do i=-2,n-3")] + "do i=-2,n-3\n" + (
        "scratch(i)=2*a(i)+real(i,8)\n" * 256) + "enddo\nend subroutine\nend module\n"
    attempts = []
    monkeypatch.setattr(region_dispatch, "lower_source", lambda *args, **kwargs: attempts.append(args))
    _, report = build(tmp_path, source)
    assert not attempts
    assert report["scope_count"] == 0
    assert any("bounded inline region/operation budget exhausted" in item["reason"] for item in report["boundaries"])


def test_inline_artifacts_are_identical_in_independent_source_directories(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    first, _ = build(left)
    second, _ = build(right)
    def computational(outputs, directory):
        return {path: text.replace(str(directory), "<original-source-directory>") for path, text in outputs.items()
                if path.startswith(("entries/", "regions/"))}
    assert computational(first, left) == computational(second, right)


def test_nested_region_cannot_hide_an_iterator_read_after_the_outer_branch(tmp_path):
    source = PROGRAM[:PROGRAM.index("do i=-2,n-3")] + (
        "if(n>0) then\ndo i=-2,n-3\nscratch(i)=2*a(i)\nenddo\nendif\ncalls=calls+i\nend subroutine\nend module\n")
    _, report = build(tmp_path, source)
    assert report["scope_count"] == 0
    assert any("loop-written scalar is live after" in item["reason"] for item in report["boundaries"])
