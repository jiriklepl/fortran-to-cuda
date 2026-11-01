"""Only independently added TARGET attributes may share a declaration edit."""

import pytest

from compiler.ir import CompilationError
from compiler.scopes.compose import _apply
from compiler.scopes.declaration_union import union_public_entries, union_targets

ORIGINAL = "integer(kind=sint),allocatable,dimension(:,:,:)::u,v,w,p\n"
U = ("integer(kind=sint),allocatable,dimension(:,:,:),target::u\n"
     "integer(kind=sint),allocatable,dimension(:,:,:)::v,w,p\n")
V = ("integer(kind=sint),allocatable,dimension(:,:,:)::u\n"
     "integer(kind=sint),allocatable,dimension(:,:,:),target::v\n"
     "integer(kind=sint),allocatable,dimension(:,:,:)::w,p\n")
W = ("integer(kind=sint),allocatable,dimension(:,:,:)::u,v\n"
     "integer(kind=sint),allocatable,dimension(:,:,:),target::w\n"
     "integer(kind=sint),allocatable,dimension(:,:,:)::p\n")


def test_union_preserves_order_shapes_attributes_and_untargeted_entities():
    expected = ("INTEGER(KIND = sint), ALLOCATABLE, DIMENSION(:, :, :), TARGET :: u, v, w\n"
                "INTEGER(KIND = sint), ALLOCATABLE, DIMENSION(:, :, :) :: p\n")
    assert union_targets(ORIGINAL, [U, V, W]) == expected
    assert union_targets(ORIGINAL, [W, U, V]) == expected
    assert _apply("module fields\n" + ORIGINAL + "end module\n", [
        {"first_line": 2, "last_line": 2, "replacement": item} for item in (U, V, W)
    ]) == "module fields\n" + expected + "end module\n"


@pytest.mark.parametrize("bad", [
    V.replace("kind=sint", "kind=8"),
    V.replace("::w,p", "::w"),
    V.replace("::w,p", "::w,p,p"),
    V.replace("::w,p", "::w,p,q"),
    V.replace("::w,p", "::w,p=3"),
    V.replace("::w,p", "::w(2),p"),
    V.replace("allocatable", "pointer"),
    V.replace("dimension(:,:,:)", "dimension(:,:)"),
    V.replace("target::v", "volatile,target::v"),
    V + "call execute()\n",
    "not a declaration\n",
])
def test_non_target_changes_are_rejected_before_composition(bad):
    assert union_targets(ORIGINAL, [U, bad]) is None
    with pytest.raises(CompilationError, match="conflicting original source edits"):
        _apply(ORIGINAL, [{"first_line": 1, "last_line": 1, "replacement": item} for item in (U, bad)])


def test_existing_targets_cannot_be_removed_and_initializers_are_retained():
    assert union_targets("integer,target::a=1,b=2\n", ["integer::a=1,b=2\n"]) is None
    assert union_targets("integer::a=1,b=2\n", ["integer,target::b=2\ninteger::a=1\n"]) == (
        "INTEGER :: a = 1\nINTEGER, TARGET :: b = 2\n")


def test_whole_multideclaration_and_executable_spans_remain_conflicts():
    assert union_targets("integer::a\ninteger::b\n", ["integer,target::a,b\n"]) is None
    assert union_targets("a=b+1\n", ["a=b+2\n"]) is None


def test_partially_shared_public_entry_insertions_export_each_symbol_once():
    first, second = "fort_scope_entry_012345abcdef", "fort_scope_entry_abcdef012345"
    values = [f"public :: {first}\npublic :: {second}\n", f"public :: {second}\n"]
    expected = f"public :: {first}\npublic :: {second}\n"
    assert union_public_entries(values) == expected
    assert union_public_entries([f"PUBLIC :: {first.upper()}, {second.upper()}\n", values[1]]) == expected.upper().replace("PUBLIC", "public")
    assert _apply("module declarations\ncontains\nend module\n", [
        {"first_line": 2, "last_line": 1, "replacement": value} for value in values
    ]) == "module declarations\n" + expected + "contains\nend module\n"


@pytest.mark.parametrize("other", [
    "private :: fort_scope_entry_012345abcdef\n", "public\n", "public :: original_symbol\n",
    "public :: operator(+)\n", "integer :: fort_scope_entry_012345abcdef\n", "call run()\n",
    "public :: fort_scope_entry_012345abcdef\ncontains\nsubroutine other()\nend subroutine\n",
])
def test_public_export_union_cannot_normalize_other_source_changes(other):
    assert union_public_entries(["public :: fort_scope_entry_abcdef012345\n", other]) is None
