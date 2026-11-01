"""Case headers are structural; every original alternative retains its effects."""
from copy import copy

import pytest

from compiler.frontend import resource_reachability
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError


SOURCE = """module work
implicit none
integer,parameter::lo=2,hi=4
real(8)::a(8),input(4),later(8),observed
contains
subroutine owner()
call helper(a,input)
end subroutine
subroutine helper(value,values)
real(8),intent(inout)::value(:)
real(8),intent(in)::values(:)
BODY
end subroutine
integer function choose(values)
real(8),intent(in)::values(:)
observed=values(1)
choose=int(values(1))
end function
end module
"""

BODY = """select case(choose(values))
case(1)
value(1)=values(2)
case(lo:hi)
value(2)=values(3)
case(:0,7:)
later(1)=values(4)
case default
select case(int(values(2)))
case(1,2)
later(2)=value(3)
case default
value(4)=later(3)
end select
end select"""


def setup(tmp_path, body=BODY, *, seal=True, source=SOURCE):
    path = tmp_path / "source.f90"
    path.write_text(source.replace("BODY", body))
    analysis = SourceEffects([path])
    if seal:
        analysis.seal_session()
    routine = analysis.routines["work::owner"]
    originals = tuple(routine.execution.content)
    demands = tuple(analysis.modules["work"].bindings[name]
                    for name in ("a", "input", "later", "observed"))
    proof = resource_reachability.prove_resource_reachability(
        analysis, routine.qualified, originals, demands)
    return path, analysis, originals, demands, proof


@pytest.mark.parametrize("seal", [False, True])
def test_all_case_alternatives_selector_calls_and_range_bounds_remain_visible(tmp_path, seal):
    _path, analysis, originals, demands, proof = setup(tmp_path, seal=seal)
    assert not proof.barriers, proof.public()["barriers"]
    assert proof.validate(analysis, "work::owner", originals, demands) is proof
    assert {call.callee for call in proof.calls} == {"work::helper", "work::choose"}
    actions = {(access.storage.resource, access.kind) for access in proof.accesses}
    assert {("work::a", "read"), ("work::a", "write"),
            ("work::input", "read"), ("work::later", "read"),
            ("work::later", "write"), ("work::observed", "write")} <= actions
    # Named constants remain authenticated syntax, without payload accesses.
    assert not {("work::lo", "read"), ("work::hi", "read")} & actions
    assert not proof.public()["execution_authorized"]
    assert not proof.native_graph.public()["alias_authority"]


@pytest.mark.parametrize("body", [
    BODY.replace("value(1)=values(2)", "call missing(value)"),
    BODY.replace("choose(values)", "unknown(values)"),
    BODY.replace("value(4)=later(3)", "allocate(value(4))"),
    BODY.replace("value(4)=later(3)", "stop"),
])
def test_case_wrappers_do_not_hide_unknown_calls_or_lifetime_effects(tmp_path, body):
    _path, _analysis, _originals, _demands, proof = setup(tmp_path, body)
    assert proof.barriers


@pytest.mark.parametrize("mutation", ["copy", "registry", "source", "policy"])
def test_case_graph_retains_source_registry_and_policy_authentication(tmp_path, monkeypatch, mutation):
    path, analysis, originals, demands, proof = setup(tmp_path)
    assert not proof.barriers, proof.public()["barriers"]
    candidate = proof
    if mutation == "copy":
        candidate = copy(proof)
    elif mutation == "registry":
        analysis = SourceEffects([path])
    elif mutation == "source":
        path.write_text(path.read_text().replace("later(2)=value(3)", "later(2)=value(4)"))
    else:
        monkeypatch.setattr(resource_reachability, "VERSION", resource_reachability.VERSION + 1)
    with pytest.raises(CompilationError):
        if mutation == "source":
            analysis.verify_publication()
        candidate.validate(analysis, "work::owner", originals, demands)
