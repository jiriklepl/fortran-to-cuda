"""Low-level eligibility forwarding is distinct from source path admission."""
from hashlib import sha256
from types import SimpleNamespace

from fparser.common.readfortran import FortranStringReader
from fparser.two.parser import ParserFactory
from fparser.two.utils import walk

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.graph import GraphOwner
from compiler.scopes.lexical import LexicalOwner
from compiler.scopes.source import ScopeBuilder


def test_every_rearm_requires_eligibility_even_when_return_retention_allows_it():
    owner = SimpleNamespace(builder=SimpleNamespace(config=SimpleNamespace(scope_execution="graph", policy="sections")),
                            enabled="enabled", retention_allowed="retained", eligibility_allowed="eligible")
    owner.owner = owner
    text = "\n".join(LexicalOwner.rearm_epoch(owner))
    assert "enabled = fort_scope_serial_caller() /= 0 .and. retained .and. eligible" in text


def test_ordinary_rearm_output_remains_unchanged_without_eligibility_control():
    owner = SimpleNamespace(builder=SimpleNamespace(config=SimpleNamespace(scope_execution="graph", policy="sections")),
                            enabled="enabled")
    owner.owner = owner
    assert LexicalOwner.rearm_epoch(owner) == ["block", "use fort_scoped_memory, only: fort_scope_serial_caller",
                                              "enabled = fort_scope_serial_caller() /= 0", "end block"]


SOURCE = """module controls
implicit none
contains
subroutine first(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
a(i)=a(i)+1.0_8
enddo
end subroutine
subroutine second(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
call first(a,n)
end subroutine
subroutine run(a,n,eligible)
real(8),intent(inout)::a(:)
integer,intent(in)::n
logical,intent(inout)::eligible
call second(a,n)
end subroutine
end module
"""


def test_checked_entry_forwards_the_same_control_through_every_child(tmp_path):
    path = tmp_path / "source.f90"
    path.write_text(SOURCE)
    builder = ScopeBuilder([path], "controls::run", options=CompilerOptions(),
        config=OffloadConfig(policy="sections"), facts={"schema_version": 1, "participation": "serial",
        "sources": {str(path): sha256(path.read_bytes()).hexdigest()}, "captures": {
        "argument::a": {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}}})
    # This exercises only the private emitter seam. There is deliberately no
    # public request/issuer admission of the synthetic ordinary formal gate.
    owner = GraphOwner(builder)
    owner.eligibility_allowed = "eligible"
    assert owner.run(tuple(builder.entry.execution.content))
    outputs, report = builder.finish()
    text = outputs[report["sources"][str(path)]["replacement"]]
    parsed = ParserFactory().create(std="f2008")(FortranStringReader(text))
    forwarded = {str(entity.items[0]).lower() for node in walk(parsed)
        if type(node).__name__ == "Type_Declaration_Stmt" and "intent(inout)" in str(node).lower().replace(" ", "")
        for entity in node.items[2].items}
    assert all(hasattr(member, "eligibility_allowed") for member in owner.members.values())
    for name, member in owner.members.items():
        if name == "controls::run":
            continue
        public = member.public()
        assert public["control_version"] == 6
        assert public["eligibility_control"]["argument"] == member.eligibility_allowed
        assert f" .and. {member.eligibility_allowed}" in text
        assert member.eligibility_allowed in forwarded
    child = owner.members["controls::second"]
    leaf = owner.members["controls::first"]
    assert f"{child.eligibility_allowed} = eligible" in text
    assert f"{leaf.eligibility_allowed} = {child.eligibility_allowed}" in text
