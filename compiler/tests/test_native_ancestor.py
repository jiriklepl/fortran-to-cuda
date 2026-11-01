"""Native ancestors delegate exact selected children through one checked graph."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder

SOURCE = """module ancestor_fixture
implicit none
real(8)::observed
integer::opaque_visits=0
contains
subroutine advance(a,b,n,flag)
real(8),intent(inout)::a(:),b(:)
integer,intent(in)::n
logical,intent(in)::flag
call middle(a,b,n,flag)
end subroutine
subroutine middle(a,b,n,flag)
real(8),intent(inout)::a(:),b(:)
integer,intent(in)::n
logical,intent(in)::flag
integer::i
call producer(a,b,n)
observed=0.0_8
do i=1,n
observed=observed+b(i)
enddo
if(flag) call opaque(b,n)
call consumer(a,b,n)
end subroutine
subroutine producer(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer,intent(in)::n
integer::i
do i=1,n
b(i)=2.0_8*a(i)
enddo
end subroutine
subroutine consumer(a,b,n)
real(8),intent(inout)::a(:)
real(8),intent(in)::b(:)
integer,intent(in)::n
integer::i
do i=1,n
a(i)=b(i)+1.0_8
enddo
end subroutine
subroutine opaque(x,n)
integer,intent(in)::n
real(8),intent(inout)::x(*)
integer::i
write(*,*) 'REACHED_SYNCHRONOUS_WRITER'
opaque_visits=opaque_visits+1
do i=1,n
x(i)=x(i)+3.0_8
enddo
end subroutine
end module
"""
ENTRY = "ancestor_fixture::advance"
ROOTS = ["ancestor_fixture::producer", "ancestor_fixture::consumer"]


def fixture(tmp_path, *, source=SOURCE, request=None, contracts=True):
    path = tmp_path / "ancestor.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: {"storage": "stable", "initialized": "whole",
                 "escapes": False, "allocation_changes": False} for name in ("a", "b")}}
    return path, ScopeBuilder([path], ENTRY, facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections"),
        contracts={ENTRY: {"native_ancestor": request or {"version": 1, "device_roots": ROOTS}}} if contracts else None)


def test_selected_child_subtrees_use_normal_preparation_and_ancestors_never_outline(tmp_path, monkeypatch):
    from compiler.scopes.region_dispatch import InlineRegions

    calls = []
    original = InlineRegions.prepare

    def tracked(self, nodes, **keywords):
        calls.append(self.builder.entry.qualified)
        assert self.builder.entry.qualified in ROOTS
        return original(self, nodes, **keywords)

    monkeypatch.setattr(InlineRegions, "prepare", tracked)
    _path, builder = fixture(tmp_path)
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report["graph_owner_attempt"]
    assert set(calls) == set(ROOTS)
    selection = report["native_ancestor_selection"]
    assert selection["kind"] == "original_native_ancestor_selection"
    for flag in ("payload_access_authorized", "descriptor_stability_authorized", "entry_definition_authority",
                 "execution_authorized", "placement_authorized", "alias_authority"):
        assert selection[flag] is False
    scope, = report["scopes"]
    procedures = {row["procedure"] for row in scope["graph_execution"]["procedures"]}
    assert procedures == {ENTRY, "ancestor_fixture::middle", *ROOTS}
    changed = "\n".join(text for name, text in outputs.items() if name.startswith("sources/"))
    assert "if (flag) then" in changed.lower()
    assert "fort_scope_resident_close_v1" in changed
    assert "call opaque(b, n)" in changed.lower() or "call opaque(b,n)" in changed.lower()
    assert scope["checked_execution_plan"]["entry"] == ENTRY


def test_no_selected_work_cannot_fall_back_to_outlining_the_native_root(tmp_path, monkeypatch):
    source = SOURCE.replace("call middle(a,b,n,flag)", "a(:)=a(:)+1.0_8")
    # A supported original numerical assignment must not cause the independent
    # numerical-roots fallback to broaden the selected descendant inventory.
    _path, builder = fixture(tmp_path, source=source)
    from compiler.scopes.region_dispatch import InlineRegions

    monkeypatch.setattr(InlineRegions, "prepare", lambda *_args, **_kwargs: pytest.fail("unselected outline"))
    outputs, report = builder.run()
    assert not report["scope_count"]
    assert set(outputs) == {"scope-manifest.json"}
    assert not report["source_edits"]


def test_default_discovery_remains_unselected_without_new_contract(tmp_path):
    _path, builder = fixture(tmp_path, contracts=False)
    assert builder._native_ancestor_selection is None
    _outputs, report = builder.run()
    assert "native_ancestor_selection" not in report


@pytest.mark.parametrize("selection_request", [
    {"version": True, "device_roots": ROOTS},
    {"version": 1, "device_roots": []},
    {"version": 1, "device_roots": [*ROOTS, ROOTS[0]]},
    {"version": 1, "device_roots": [ENTRY]},
    {"version": 1, "device_roots": ["missing::worker"]},
    {"version": 1, "device_roots": ["ANCESTOR_FIXTURE::producer"]},
    {"version": 1, "device_roots": ROOTS, "lifetime": "stable"},
])
def test_selection_request_cannot_supply_source_or_lifetime_authority(tmp_path, selection_request):
    with pytest.raises(CompilationError):
        fixture(tmp_path, request=selection_request)


@pytest.mark.parametrize("mutation", ["registry", "contract", "source"])
def test_registered_selection_revalidates_original_request_and_source(tmp_path, mutation):
    path, builder = fixture(tmp_path)
    proof = builder._native_ancestor_selection
    if mutation == "registry":
        builder.analysis._native_ancestor_selections[proof.identity] = replace(proof)
    elif mutation == "contract":
        contracts = deepcopy(builder.analysis.contracts)
        contracts[ENTRY]["native_ancestor"]["device_roots"].pop()
        builder.analysis.contracts = contracts
    else:
        path.write_text(path.read_text().replace("b(i)=2.0_8*a(i)", "b(i)=3.0_8*a(i)"))
    with pytest.raises(CompilationError):
        proof.validate()


def test_independently_selected_root_does_not_make_ancestor_native_writes_defined(tmp_path):
    _path, builder = fixture(tmp_path)
    proof = builder._native_ancestor_selection
    assert proof.public()["entry_definition_authority"] is False
    assert proof.public()["descriptor_stability_authorized"] is False
    assert all("original_body_sha256" in member for member in proof.public()["device_roots"])
    assert not hasattr(proof, "descriptor_unchanged")
