"""Only exact original descriptor exchanges may retain physical allocations."""
from copy import copy
from dataclasses import replace
from hashlib import sha256

import pytest
from fparser.two.utils import walk

from compiler.driver.options import CompilerOptions
from compiler.frontend.allocation_exchange import allocation_exchange
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_source_owner import FACT

SOURCE = """module fields
implicit none
contains
subroutine relocate(left,right)
real(8),allocatable,intent(inout)::left(:),right(:)
real(8),allocatable::temporary(:)
call move_alloc(left,temporary)
call move_alloc(right,left)
call move_alloc(temporary,right)
end subroutine
subroutine advance(a,b,n)
real(8),allocatable,intent(inout)::a(:),b(:)
integer,intent(in)::n
integer::i
do i=1,n
b(i)=2*a(i)
enddo
call relocate(a,b)
do i=1,n
a(i)=a(i)+b(i)
enddo
end subroutine
end module
"""


def source_fixture(tmp_path, source=SOURCE):
    path = tmp_path / "source.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    routine = analysis.routines["fields::advance"]
    node, = [item for item in walk(routine.execution) if type(item).__name__ == "Call_Stmt"]
    return path, analysis, routine, node


def test_exchange_follows_exact_original_associations_without_payload_authority(tmp_path):
    _path, analysis, routine, node = source_fixture(tmp_path)
    proof = allocation_exchange(analysis, routine, node)
    assert proof.validate(analysis, routine, node) is proof
    assert tuple(binding.root for binding in proof.bindings) == ("argument::a", "argument::b")
    public = proof.public()
    assert public["payload_reads"] is public["payload_writes"] is public["initialization_authorized"] is False
    public["initialization_authorized"] = True
    assert proof.public()["initialization_authorized"] is False
    with pytest.raises(CompilationError, match="exact compiler-issued"):
        copy(proof).validate(analysis, routine, node)
    with pytest.raises(CompilationError, match="exact compiler-issued"):
        proof.validate(analysis, routine, copy(node))
    with pytest.raises(CompilationError, match="exact compiler-issued"):
        replace(proof, _issuer=object()).validate(analysis, routine, node)


@pytest.mark.parametrize(("old", "new"), [
    ("call move_alloc(right,left)", "call move_alloc(left,right)"),
    ("allocatable::temporary", "allocatable,save::temporary"),
    ("call relocate(a,b)", "call relocate(a,a)"),
])
def test_unsafe_exchange_cannot_receive_a_lifetime_transition(tmp_path, old, new):
    _path, analysis, routine, node = source_fixture(tmp_path, SOURCE.replace(old, new))
    with pytest.raises(CompilationError):
        allocation_exchange(analysis, routine, node)


@pytest.mark.parametrize("extra", ["write(*,*) 'effect'", "deallocate(left)", "if(size(left)>0)left=1"])
def test_additional_source_work_cannot_be_hidden_inside_exchange(tmp_path, extra):
    source = SOURCE.replace("call move_alloc(left,temporary)", extra + "\ncall move_alloc(left,temporary)")
    _path, analysis, routine, node = source_fixture(tmp_path, source)
    assert allocation_exchange(analysis, routine, node) is None


def test_graph_retains_exchange_as_a_lifetime_task_with_postmutation_checks(tmp_path):
    path, _, _, _ = source_fixture(tmp_path)
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::a": dict(FACT), "argument::b": dict(FACT)}}
    outputs, report = ScopeBuilder([path], "fields::advance", facts=facts, options=CompilerOptions(),
                                  config=OffloadConfig(policy="sections", scope_execution="graph")).run()
    owner, = report["scopes"]
    assert not owner["boundaries"]
    member, = owner["checked_execution_plan"]["storage_epoch"]["members"]
    transfer, = member["storage_transfers"]
    assert transfer["resources"] == ["argument::a", "argument::b"]
    procedure, = owner["graph_execution"]["procedures"]
    task, = [item for item in procedure["task_graph"]["tasks"] if ":storage:" in item["id"]]
    assert task["lifetime_resources"] == ["argument::a", "argument::b"]
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert text.count("call relocate(a,b)") == 1
    assert text.index("fort_scope_layout_get(") < text.index("call relocate(a,b)")
    assert "fort_swap_registered_lowers == lbound(" in text
    assert "fort_swap_registered_extents == shape(" in text
    assert "c_associated(fort_swap_registered%host, fort_swap_original_pointer)" in text
    assert text.index("call relocate(a,b)") < text.index("fort_scope_identity_swap_v1(")
    assert "error stop 'allocation exchange coherence failed after mutation'" in text
