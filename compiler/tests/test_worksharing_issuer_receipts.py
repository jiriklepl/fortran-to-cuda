"""Transactions retain exact issued baselines while isolating registry writes."""
import json
from copy import copy

import pytest

from compiler.frontend.descriptor_worksharing import (
    _IssuedReceipt,
    original_worksharing_storage,
)
from compiler.ir import CompilationError
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.storage_epoch import require_original_worksharing_publication
from compiler.tests.test_owner_local_call_groups import SOURCE, fixture, publish

NATIVE = """subroutine native_filter(b,n)
integer,intent(in)::n
real(8),intent(inout)::b(:)
integer::i
!$omp do
do i=1,n
b(i)=real(i,8)
enddo
!$omp end do nowait
end subroutine
subroutine private_scratch(b,n)
integer,intent(in)::n
real(8),intent(in)::b(:)
real(8),allocatable,save::scratch(:)
scratch=b
end subroutine
"""


def builder_at(tmp_path):
    source = SOURCE.replace("subroutine second(a,b,n)", NATIVE+"subroutine second(a,b,n)")
    source = source.replace("integer::i\ndo i=1,n\nb(i)=real(i,8)",
        "integer::i\nif(n<0)call native_filter(b,n)\nif(n<0)call private_scratch(b,n)\ndo i=1,n\nb(i)=real(i,8)")
    calls = "call first(b,n)\ncall second(a,b,n)\ncall third(a,b,n)"
    source = source.replace("endif\ncall unknown_suffix()", "endif\nif(.not.flag)then\n"+calls+"\nendif\ncall unknown_suffix()")
    return fixture(tmp_path, source)


def test_two_completed_owners_keep_orphan_and_private_constraints_through_publication(tmp_path):
    _path, builder = builder_at(tmp_path)
    outputs, report = builder.run()
    publish(tmp_path, outputs, report)
    assert report["scope_count"] == 2, report["boundaries"]
    epochs = [scope["checked_execution_plan"]["storage_epoch"] for scope in report["scopes"]]
    assert all(epoch["version"] == 3 for epoch in epochs)
    assert all(epoch["original_worksharing_storage_obligations"] for epoch in epochs)
    assert all(epoch["private_allocation_obligations"] for epoch in epochs)
    assert len({epoch["identity"] for epoch in epochs}) == 2
    assert len(builder._original_worksharing_publications) == 2
    assert getattr(builder, "_graph_storage_epoch", None) is None
    require_original_worksharing_publication(builder)
    (tmp_path/"epochs.json").write_text(json.dumps(epochs, indent=2)+"\n")


def test_transaction_borrows_original_receipt_but_rolls_back_registry_writes(tmp_path):
    _path, builder = builder_at(tmp_path)
    analysis = builder.analysis
    routine = analysis.routines["local_groups::native_filter"]
    proof = original_worksharing_storage(analysis, routine)
    public = proof.public()
    registry = analysis._original_worksharing_storage
    issued = analysis._original_worksharing_storage_receipts
    receipt = issued[proof.identity]
    assert type(receipt) is _IssuedReceipt
    with GenerationTransaction(builder):
        assert analysis._original_worksharing_storage is not registry
        assert analysis._original_worksharing_storage_receipts is not issued
        assert analysis._original_worksharing_storage_receipts[proof.identity] is receipt
        assert original_worksharing_storage(analysis, routine) is proof
        assert proof.validate(analysis, routine) is proof
        analysis._original_worksharing_storage[proof.identity] = copy(proof)
        with pytest.raises(CompilationError, match="foreign authority"):
            original_worksharing_storage(analysis, routine)
    assert analysis._original_worksharing_storage is registry
    assert analysis._original_worksharing_storage_receipts is issued
    assert issued[proof.identity] is receipt
    assert proof.validate(analysis, routine) is proof
    assert proof.public() == public


def test_issued_baseline_is_readonly_and_plain_tuple_cannot_replace_it(tmp_path):
    _path, builder = builder_at(tmp_path)
    analysis = builder.analysis
    routine = analysis.routines["local_groups::native_filter"]
    proof = original_worksharing_storage(analysis, routine)
    issued = analysis._original_worksharing_storage_receipts
    receipt = issued[proof.identity]
    with pytest.raises(TypeError):
        receipt[0] = copy(proof)
    with pytest.raises(AttributeError):
        receipt.extra = "mutable"
    issued[proof.identity] = tuple(receipt)
    with pytest.raises(CompilationError, match="foreign authority"):
        original_worksharing_storage(analysis, routine)
    with pytest.raises(CompilationError, match="compiler-issued authority"):
        proof.checkpoint()
    issued[proof.identity] = receipt
    assert proof.validate(analysis, routine) is proof
