"""Saved allocations remain owned by the original procedure around borrowed loops."""

from hashlib import sha256

from compiler.offload.config import OffloadConfig
from compiler.driver.options import CompilerOptions
from compiler.frontend import lower_source
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
