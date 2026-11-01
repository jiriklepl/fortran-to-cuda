"""Owned record regions constrain associations without admitting payload work."""
from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.storage_record_regions import prove_owned_record_region
from compiler.ir import CompilationError

SOURCE = """module regions
implicit none
type cell
real(8)::value
integer::index
logical::active
end type
type(cell),allocatable,target::records(:)
real(8),allocatable::field(:)
real(8)::hidden
contains
subroutine advance()
call touch(records,field)
end subroutine
subroutine touch(points,a)
type(cell),target,intent(inout)::points(:)
real(8),intent(inout)::a(:)
type(cell),pointer::p
integer::i
!$omp parallel do private(i,p)
do i=1,size(points)
p=>points(i)
p%value=p%value+a(i)
a(i)=p%value
end do
!$omp end parallel do
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE, sealed=False):
    path = tmp_path / "regions.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    caller = analysis.routines["regions::advance"]
    call, = [node for node in walk(caller.execution) if type(node).__name__ == "Call_Stmt"]
    numeric = (analysis.modules["regions"].bindings["field"],)
    if sealed:
        analysis.seal_session()
    return path, analysis, caller, call, numeric


def prove(data, **kwargs):
    _path, analysis, caller, call, numeric = data
    return prove_owned_record_region(analysis, caller.qualified, call, numeric, **kwargs)


@pytest.mark.parametrize("sealed", [False, True])
def test_original_owned_record_pointer_is_scoped_and_has_no_payload_grant(tmp_path, sealed):
    data = fixture(tmp_path, sealed=sealed)
    proof = prove(data)
    public = proof.public()
    assert public["record_owner"] == "regions::records"
    assert public["record_formal"] == "argument::points"
    assert public["disjoint_numeric_allocations"] == ["regions::field"]
    assert public["original_association"] == "p => points(i)"
    assert public["original_directives"] == ["parallel do private(i,p)", "end parallel do"]
    assert public["original_field_occurrences"] == ["p % value", "p % value", "p % value"]
    assert public["complete_original_body_and_callees_must_be_scanned"]
    for name in ["payload_access_authorized", "native_execution_authorized", "numerical_completion_authorized",
                 "entry_definition_authority", "allocation_presence_proved"]:
        assert public[name] is False
    assert proof.validate() is proof
    assert prove(data) is proof
    assert not getattr(data[1], "_descriptor_team_proofs", {})
    assert not getattr(data[1], "_descriptor_window_proofs", {})


@pytest.mark.parametrize(("before", "after"), [
    ("real(8)::value", "real(8),allocatable::value(:)"),
    ("real(8)::value", "real(8),pointer::value"),
    ("real(8)::value", "real(8)::value(2)"),
    ("type(cell),allocatable,target::records(:)", "type(cell),pointer::records(:)"),
    ("type(cell),allocatable,target::records(:)", "type(cell),allocatable::records(:)"),
    ("real(8),allocatable::field(:)", "real(8),allocatable,target::field(:)"),
    ("real(8),allocatable::field(:)", "real(8),pointer::field(:)"),
    ("type(cell),target,intent(inout)::points(:)", "type(cell),intent(inout)::points(:)"),
    ("type(cell),pointer::p", "type(cell),pointer,save::p"),
    ("private(i,p)", "private(i)"),
    ("private(i,p)", "private(p)"),
    ("parallel do private(i,p)", "parallel do private(i,p) nowait"),
    ("end parallel do", "end parallel do nowait"),
    ("p=>points(i)", "p=>points(1)"),
    ("p=>points(i)", "p=>records(i)"),
    ("p=>points(i)\np%value=p%value+a(i)", "p%value=p%value+a(i)\np=>points(i)"),
    ("a(i)=p%value", "a(i)=p%value\np=>points(i)"),
    ("a(i)=p%value", "a(i)=p%value\ncall unknown(p)"),
    ("a(i)=p%value", "a(i)=p%value\nnullify(p)"),
    ("a(i)=p%value", "a(i)=p%value\nprint *,p"),
])
def test_descriptor_bearing_fields_escapes_and_unproved_lifetimes_reject(tmp_path, before, after):
    data = fixture(tmp_path, SOURCE.replace(before, after))
    with pytest.raises(CompilationError):
        prove(data)
    assert not getattr(data[1], "_owned_record_regions", {})


def test_hidden_numeric_payload_remains_original_and_requires_complete_consumer_scan(tmp_path):
    source = SOURCE.replace("a(i)=p%value", "a(i)=p%value\nhidden=p%value")
    proof = prove(fixture(tmp_path, source))
    assert proof.public()["complete_original_body_and_callees_must_be_scanned"]
    assert "hidden = p % value" in str(proof._analysis.routines["regions::touch"].execution)


def test_exact_original_contained_callee_keeps_its_host_and_source_identity(tmp_path):
    source = SOURCE.replace("end subroutine\nsubroutine touch", "contains\nsubroutine touch", 1)
    source = source.replace("end subroutine\nend module", "end subroutine\nend subroutine\nend module")
    proof = prove(fixture(tmp_path, source, sealed=True))
    assert proof.public()["callee"] != "regions::touch"
    assert proof.public()["caller"] == "regions::advance"
    assert proof.public()["original_host_execution_required"]


@pytest.mark.parametrize("inside_team", [False, True])
def test_unknown_original_call_cannot_be_omitted_by_effect_consumer(tmp_path, inside_team):
    from compiler.frontend.descriptor_record_calls import original_record_call

    source = (SOURCE.replace("a(i)=p%value", "a(i)=p%value\ncall unknown(a)") if inside_team else
              SOURCE.replace("!$omp end parallel do", "!$omp end parallel do\ncall unknown(field)"))
    data = fixture(tmp_path, source)
    proof = prove(data)
    assert proof.public()["complete_original_body_and_callees_must_be_scanned"]
    _path, analysis, caller, call, numeric = data
    with pytest.raises(CompilationError):
        original_record_call(analysis, caller, call, numeric)


def test_numeric_dummy_without_original_owned_allocation_cannot_supply_separation(tmp_path):
    data = fixture(tmp_path)
    numeric = data[1].routines["regions::touch"].scope.bindings["a"]
    with pytest.raises(CompilationError, match="allocatable ownership"):
        prove_owned_record_region(data[1], data[2].qualified, data[3], (numeric,))


def test_source_storage_association_declaration_does_not_supply_separation(tmp_path):
    source = SOURCE.replace("real(8)::hidden", "real(8)::hidden,other\nequivalence(hidden,other)")
    with pytest.raises(CompilationError):
        prove(fixture(tmp_path, source))


def test_visible_implicit_assignment_cannot_hide_unscanned_source_call(tmp_path):
    source = SOURCE.replace("contains\nsubroutine advance", """interface assignment(=)
module procedure assign
end interface
contains
subroutine assign(left,right)
real(8),intent(out)::left
logical,intent(in)::right
left=1
end subroutine
subroutine advance""", 1)
    with pytest.raises(CompilationError, match="defined assignment"):
        prove(fixture(tmp_path, source))


def test_copied_call_and_foreign_analysis_cannot_issue_or_consume(tmp_path):
    data = fixture(tmp_path)
    path, analysis, caller, call, numeric = data
    with pytest.raises(CompilationError, match="exact original caller"):
        prove_owned_record_region(analysis, caller.qualified, copy(call), numeric)
    proof = prove(data)
    with pytest.raises(CompilationError, match="exact registered"):
        proof.validate(SourceEffects([path]))
    with pytest.raises(CompilationError, match="exact registered"):
        copy(proof).validate()


@pytest.mark.parametrize("sealed", [False, True])
def test_late_consumed_binding_and_original_source_changes_reject(tmp_path, sealed):
    data = fixture(tmp_path, sealed=sealed)
    binding = data[1].modules["regions"].bindings["records"]
    fired = False

    def changed():
        nonlocal fired
        if not fired:
            fired = True
            object.__setattr__(binding, "rank", 2)

    expected = "attachment changed" if sealed else "original resolution authority changed"
    with pytest.raises(CompilationError, match=expected):
        prove(data, tick=changed)
    assert fired
    assert not getattr(data[1], "_owned_record_regions", {})


def test_live_source_change_never_publishes_association_receipt(tmp_path):
    data = fixture(tmp_path, sealed=True)
    fired = False

    def changed():
        nonlocal fired
        if not fired:
            fired = True
            data[0].write_text(data[0].read_text() + "! changed original\n")

    with pytest.raises(CompilationError, match="source changed"):
        prove(data, tick=changed)
    assert not getattr(data[1], "_owned_record_regions", {})


def test_cached_issuance_retains_existing_token_before_derivation_callbacks(tmp_path):
    data = fixture(tmp_path, sealed=True)
    proof = prove(data)

    def changed():
        object.__setattr__(proof, "_record", "{}")

    with pytest.raises(CompilationError, match="retained authority changed"):
        prove(data, tick=changed)


@pytest.mark.parametrize("fault", ["map", "token"])
def test_standalone_validation_rechecks_its_issuer_after_last_callback(tmp_path, monkeypatch, fault):
    data = fixture(tmp_path, sealed=True)
    analysis = data[1]
    proof = prove(data)
    publication = analysis.verify_publication

    def changed():
        publication()
        if fault == "map":
            analysis._owned_record_regions = dict(analysis._owned_record_regions)
        else:
            analysis._owned_record_regions[proof.identity] = copy(proof)

    monkeypatch.setattr(analysis, "verify_publication", changed)
    with pytest.raises(CompilationError, match="consuming attachment changed"):
        proof.validate()


@pytest.mark.parametrize("sealed", [False, True])
def test_final_type_namespace_replacement_does_not_publish(tmp_path, monkeypatch, sealed):
    data = fixture(tmp_path, sealed=sealed)
    analysis = data[1]
    owner = analysis.modules["regions"]
    publication = analysis.verify_publication

    def changed():
        publication()
        object.__setattr__(owner, "component_types", dict(owner.component_types))

    monkeypatch.setattr(analysis, "verify_publication", changed)
    with pytest.raises(CompilationError, match="attachment changed"):
        prove(data)
    assert not getattr(analysis, "_owned_record_regions", {})


@pytest.mark.parametrize("effect", ["a(i)=unknown(p%value)", "a(i)=unknown(a(i))",
                                    "a(i)=c_loc(p%value)", "a(i)=p%value\ndeallocate(field)"])
def test_association_evidence_cannot_replace_the_complete_effect_consumer(tmp_path, effect):
    from compiler.frontend.descriptor_record_calls import original_record_call

    data = fixture(tmp_path, SOURCE.replace("a(i)=p%value", effect))
    _path, analysis, caller, call, numeric = data
    region = prove(data)
    assert region.public()["complete_original_body_and_callees_must_be_scanned"]
    assert not region.public()["native_execution_authorized"]
    # The association-only issuer no longer runs a duplicate effects engine.
    # The actual integrated consumer still rejects every original effect.
    with pytest.raises(CompilationError):
        original_record_call(analysis, caller, call, numeric)
