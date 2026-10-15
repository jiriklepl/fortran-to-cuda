"""Native coherence needs complete demanded effects and storage separation."""

from copy import copy

import pytest

from compiler.frontend.native_demand import prove_native_demand
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

SOURCE = """module fields
implicit none
type profile
real(8),allocatable::flux(:)
integer::visits
end type
type(profile)::wall
real(8)::old(8),star(8),out(8)
integer::n=8
contains
subroutine advance()
integer::i
call correction()
end subroutine
subroutine correction()
wall%flux(1)=wall%flux(1)+old(1)
wall%visits=wall%visits+1
star(2)=star(2)+wall%flux(1)
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE, *, seal=True):
    path = tmp_path / "source.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    if seal:
        analysis.seal_session()
    routine = analysis.routines["fields::advance"]
    selected = tuple(routine.execution.content)
    demanded = tuple(analysis.modules["fields"].bindings[name] for name in ("old", "star", "out", "n"))
    demanded += (routine.scope.bindings["i"],)
    return path, analysis, selected, demanded


def test_effects_keep_host_read_and_partial_write_without_overwrite(tmp_path, monkeypatch):
    _path, analysis, selected, demanded = fixture(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("native demanded effects must not invoke complete effect summaries")

    monkeypatch.setattr(analysis, "summarize", forbidden)
    proof = prove_native_demand(analysis, "fields::advance", selected, demanded)
    assert proof.validate(analysis, "fields::advance", selected, demanded) is proof
    assert {binding.root: actions for binding, actions in proof.effects} == {
        "fields::old": {"read"}, "fields::star": {"read", "write"}}
    public = proof.public()
    assert public["storage_separation"]
    assert public["native_completion"]["available"]
    assert not public["capture_authorized"]
    assert not public["placement_authorized"]
    public["effects"].clear()
    assert proof.public()["effects"]
    assert prove_native_demand(analysis, "fields::advance", selected, demanded) is proof


@pytest.mark.parametrize("change", ["copy", "selection", "demand", "registry", "source"])
def test_stale_or_foreign_authority_rejected(tmp_path, change):
    path, analysis, selected, demanded = fixture(tmp_path)
    proof = prove_native_demand(analysis, "fields::advance", selected, demanded)
    if change == "copy":
        proof = copy(proof)
    elif change == "selection":
        selected = (copy(selected[0]),)
    elif change == "demand":
        demanded = (*demanded[:-1], copy(demanded[-1]))
    elif change == "registry":
        analysis._native_demand_effects.clear()
    else:
        path.write_text(SOURCE + "! live source changed\n")
        with pytest.raises(CompilationError, match="source changed"):
            analysis.verify_publication()
        return
    with pytest.raises(CompilationError):
        proof.validate(analysis, "fields::advance", selected, demanded)


def test_unknown_dummy_alias_cannot_borrow_distinct_module_name_rule(tmp_path):
    source = SOURCE.replace("subroutine advance()", "subroutine advance(other)\nreal(8)::other(8)")
    _path, analysis, selected, demanded = fixture(tmp_path, source)
    demanded += (analysis.routines["fields::advance"].scope.bindings["other"],)
    with pytest.raises(CompilationError, match="separation.*owned storage"):
        prove_native_demand(analysis, "fields::advance", selected, demanded)
    assert not getattr(analysis, "_native_demand_effects", {})


@pytest.mark.parametrize(("before", "after", "message"), [
    ("real(8),allocatable::flux(:)", "real(8),pointer::flux(:)", "reachability incomplete"),
    ("wall%visits=wall%visits+1", "n=n+1", "scalar writes"),
    ("star(2)=star(2)+wall%flux(1)", "call unknown(star)", "reachability incomplete"),
    ("star(2)=star(2)+wall%flux(1)", "allocate(wall%flux(8))", "reachability incomplete"),
    ("integer::n=8", "integer::n=8\nreal(8)::alias(8)\nequivalence(old,alias)", "reachability incomplete"),
])
def test_unresolved_effects_never_issue_native_execution_authority(tmp_path, before, after, message):
    _path, analysis, selected, demanded = fixture(tmp_path, SOURCE.replace(before, after))
    with pytest.raises(CompilationError, match=message):
        prove_native_demand(analysis, "fields::advance", selected, demanded)
    assert not getattr(analysis, "_native_demand_effects", {})


def test_demanded_intent_out_definition_kill_rejected(tmp_path):
    source = SOURCE.replace("call correction()", "call clear(star)").replace(
        "subroutine correction()", "subroutine clear(a)\nreal(8),intent(out)::a(:)\na(2)=1\nend subroutine\nsubroutine correction()")
    _path, analysis, selected, demanded = fixture(tmp_path, source)
    with pytest.raises(CompilationError, match="entry or lifetime change"):
        prove_native_demand(analysis, "fields::advance", selected, demanded)


def test_entry_kill_remains_visible_through_nested_actual_mapping(tmp_path):
    source = SOURCE.replace("call correction()", "call forward(star)").replace(
        "subroutine correction()", """subroutine forward(a)
real(8),intent(inout)::a(:)
call clear(a)
end subroutine
subroutine clear(a)
real(8),intent(out)::a(:)
a(2)=1
end subroutine
subroutine correction()""")
    _path, analysis, selected, demanded = fixture(tmp_path, source)
    with pytest.raises(CompilationError, match="entry or lifetime change"):
        prove_native_demand(analysis, "fields::advance", selected, demanded)


def test_unsealed_changed_owned_declaration_invalidates_token(tmp_path):
    _path, analysis, selected, demanded = fixture(tmp_path, seal=False)
    proof = prove_native_demand(analysis, "fields::advance", selected, demanded)
    analysis.modules["fields"].bindings["old"].attributes |= {"pointer"}
    with pytest.raises(CompilationError):
        proof.validate(analysis, "fields::advance", selected, demanded)


@pytest.mark.parametrize("association", ["common /shared/ future", "equivalence(future,other)",
                                        "!$omp threadprivate(future)",
                                        "integer(8)::address\npointer(address,future)"])
def test_future_only_demand_scope_cannot_hide_storage_association(tmp_path, association):
    source = "module future_fields\nreal(8)::future(8),other(8)\n" + association + "\nend module\n" + SOURCE
    _path, analysis, selected, demanded = fixture(tmp_path, source)
    demanded += (analysis.modules["future_fields"].bindings["future"],)
    with pytest.raises(CompilationError, match="separation.*owned storage"):
        prove_native_demand(analysis, "fields::advance", selected, demanded)
