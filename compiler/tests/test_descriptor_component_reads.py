"""Native numeric reads preserve descriptors without lending payload layout."""
from copy import copy

import pytest

from compiler.frontend.descriptor_windows import _Check, prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

ENTRY = "steps::advance"


def fixture(tmp_path, *, body="a(1)=sum(current%tempfl)+sum(current%tempflsgs)",
            root_attributes="", field_attributes="allocatable", tail="", sealed=False,
            declarations=""):
    path = tmp_path / "component.f90"
    root_attributes = "," + root_attributes if root_attributes else ""
    field_attributes = "," + field_attributes if field_attributes else ""
    path.write_text(f"""module profiles
implicit none
integer,parameter::knd=8
type Profile
real(knd){field_attributes}::tempfl(:),tempflsgs(:)
end type
type(Profile){root_attributes}::current_profiles
{declarations}
end module
module steps
use profiles,only:current=>current_profiles
implicit none
contains
subroutine child(a,n)
integer,intent(in)::n
real(8),intent(inout)::a(:)
{body}
{tail}
end subroutine
subroutine advance(a,n)
integer,intent(in)::n
real(8),allocatable,intent(inout)::a(:)
call child(a,n)
end subroutine
end module
""")
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    graph = analysis.structure(ENTRY)
    routine = analysis.routines[ENTRY]
    call = routine.execution.content[0]
    selected = (graph.node_id(call),)
    return path, analysis, selected


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("body", [
    "a(1)=sum(current%tempfl)+sum(current%tempflsgs)",
    "if(allocated(current%tempfl)) a(1)=sum(current%tempfl)",
    "a(1)=current%tempfl(n-1)",
])
def test_complete_original_component_reads_need_no_payload_capture_or_allocation_fact(tmp_path, sealed, body):
    path, analysis, selected = fixture(tmp_path, body=body, sealed=sealed)
    original = path.read_bytes()
    proof = prove_descriptor_window(analysis, ENTRY, selected, ("argument::a",))
    assert proof.validate(analysis, ENTRY, selected) is proof
    public = proof.public()
    assert public["schema_version"] == 17
    assert public["resources"][0]["stable"]
    assert not public["entry_definition_authority"]
    assert not public["execution_authorized"]
    assert not public["gpu_legality_established"]
    reads = public["native_numeric_component_reads"]
    assert reads
    assert all(row["procedure"] == "steps::child" for row in reads)
    assert all(row["object_resource"] == "profiles::current_profiles" for row in reads)
    assert all(row["kind"] == "original_numeric_component_read" for row in reads)
    for row in reads:
        for key in ("allocation_or_definition_inferred", "descriptor_mutation_authorized", "payload_access_authorized",
                    "layout_authorized", "capture_authorized", "execution_authorized", "motion_authorized", "alias_authority"):
            assert row[key] is False
    assert path.read_bytes() == original
    assert not analysis._allocation_authorizations


@pytest.mark.parametrize("body", [
    "current%tempfl=a", "current%tempfl(1)=a(1)",
    "call unknown(current%tempfl)", "a(1)=unknown(current%tempfl)",
    "a(1)=c_loc(current%tempfl)", "read(*,*)current%tempfl",
])
def test_component_writes_or_address_passing_do_not_borrow_read_authority(tmp_path, body):
    _path, analysis, selected = fixture(tmp_path, body=body)
    with pytest.raises(CompilationError):
        prove_descriptor_window(analysis, ENTRY, selected, ("argument::a",))


@pytest.mark.parametrize("root_attributes", ["pointer", "target", "allocatable", "volatile", "asynchronous"])
def test_unproved_parent_association_or_concurrency_remains_rejected(tmp_path, root_attributes):
    _path, analysis, selected = fixture(tmp_path, root_attributes=root_attributes)
    with pytest.raises(CompilationError):
        prove_descriptor_window(analysis, ENTRY, selected, ("argument::a",))


@pytest.mark.parametrize("field_attributes", ["pointer"])
def test_unproved_component_association_or_concurrency_remains_rejected(tmp_path, field_attributes):
    _path, analysis, selected = fixture(tmp_path, field_attributes=field_attributes)
    with pytest.raises(CompilationError):
        prove_descriptor_window(analysis, ENTRY, selected, ("argument::a",))


@pytest.mark.parametrize("tail", ["call hidden(a)", "deallocate(current%tempfl)", "a(1)=unknown(1)"])
def test_later_unsafe_original_work_is_still_scanned(tmp_path, tail):
    _path, analysis, selected = fixture(tmp_path, tail=tail)
    with pytest.raises(CompilationError):
        prove_descriptor_window(analysis, ENTRY, selected, ("argument::a",))


def test_late_equal_parser_replacement_does_not_reuse_a_read_receipt(tmp_path, monkeypatch):
    _path, analysis, selected = fixture(tmp_path)
    original = _Check.validate_numeric_component_reads
    changed = False

    def replace_node(check):
        nonlocal changed
        if check.numeric_component_reads and not changed:
            changed = True
            receipt = next(iter(check.numeric_component_reads.values()))
            node = receipt[1]
            children = node.items
            node.items = (copy(children[0]), *children[1:])
        return original(check)

    monkeypatch.setattr(_Check, "validate_numeric_component_reads", replace_node)
    with pytest.raises(CompilationError):
        prove_descriptor_window(analysis, ENTRY, selected, ("argument::a",))
    assert changed


def test_real_source_changes_after_read_scanning_reject_before_publication(tmp_path, monkeypatch):
    path, analysis, selected = fixture(tmp_path)
    original = _Check.validate_numeric_component_reads

    def change_source(check):
        path.write_text(path.read_text() + "! late source change\n")
        return original(check)

    monkeypatch.setattr(_Check, "validate_numeric_component_reads", change_source)
    with pytest.raises(CompilationError):
        prove_descriptor_window(analysis, ENTRY, selected, ("argument::a",))


def test_unknown_component_subscript_keeps_its_original_call_effect_frontier(tmp_path):
    _path, analysis, selected = fixture(tmp_path, body="a(1)=current%tempfl(unknown(n))")
    with pytest.raises(CompilationError):
        prove_descriptor_window(analysis, ENTRY, selected, ("argument::a",))


def test_hard_component_source_or_budget_errors_are_never_fallback_signals(tmp_path, monkeypatch):
    import compiler.frontend.descriptor_windows as windows

    _path, analysis, selected = fixture(tmp_path)

    def hard_error(*_args, **_kwargs):
        raise CompilationError("original component source budget marker")

    monkeypatch.setattr(windows, "component_access", hard_error)
    with pytest.raises(CompilationError, match="original component source budget marker"):
        prove_descriptor_window(analysis, ENTRY, selected, ("argument::a",))


def test_final_authority_callback_cannot_replace_a_consumed_read_child(tmp_path, monkeypatch):
    _path, analysis, selected = fixture(tmp_path)
    original = _Check.validate_numeric_component_reads
    authority = analysis._summary_authority
    armed = None

    def retain(check):
        nonlocal armed
        original(check)
        if check.numeric_component_reads:
            armed = next(iter(check.numeric_component_reads.values()))[1]

    def late_change(*args, **kwargs):
        public = authority(*args, **kwargs)
        if armed is not None:
            armed.items = (copy(armed.items[0]), *armed.items[1:])
        return public

    monkeypatch.setattr(_Check, "validate_numeric_component_reads", retain)
    monkeypatch.setattr(analysis, "_summary_authority", late_change)
    with pytest.raises(CompilationError):
        prove_descriptor_window(analysis, ENTRY, selected, ("argument::a",))


@pytest.mark.parametrize("sealed", [False, True])
def test_first_helper_callback_cannot_reseed_the_original_canonical_root(tmp_path, monkeypatch, sealed):
    import compiler.frontend.storage_component_reads as service

    _path, analysis, selected = fixture(tmp_path, sealed=sealed)
    root = analysis.modules["profiles"].bindings["current_profiles"]
    original = service.numeric_component_read
    mutated = False

    def wrapped(*args, tick, **kwargs):
        def callback():
            nonlocal mutated
            if not mutated:
                object.__setattr__(root, "root", "forged::profiles")
                mutated = True
            tick()

        return original(*args, tick=callback, **kwargs)

    monkeypatch.setattr(service, "numeric_component_read", wrapped)
    with pytest.raises(CompilationError):
        prove_descriptor_window(analysis, ENTRY, selected, ("argument::a",))
    assert mutated
