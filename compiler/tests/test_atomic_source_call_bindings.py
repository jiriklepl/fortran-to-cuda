"""Repeated atomic helpers retain checked per-call allocation projections."""

import json
from dataclasses import replace
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

TEAM = """!$omp parallel private(i)
if(flag)then
!$omp sections
!$omp section
do i=n+1,n+3
a(i)=p(i-n)
enddo
!$omp section
do i=-2,0
a(i)=p(i+n)
enddo
!$omp end sections nowait
endif
!$omp end parallel
"""
SOURCE = (
    """module atomic_calls
contains
subroutine helper(a,p,n,flag)
real(8),intent(inout)::a(-2:)
real(8),intent(in)::p(-2:)
integer,intent(in)::n
logical,intent(in)::flag
integer::i
"""
    + TEAM
    + """end subroutine
subroutine relay(a,p,n,flag)
real(8),intent(inout)::a(-2:)
real(8),intent(in)::p(-2:)
integer,intent(in)::n
logical,intent(in)::flag
call helper(a,p,n,flag)
end subroutine
subroutine step(a,b,p,q,n,flag)
real(8),intent(inout)::a(-2:),b(-2:)
real(8),intent(in)::p(-2:),q(-2:)
integer,intent(in)::n
logical,intent(in)::flag
call relay(a,p,n,flag)
call relay(b,q,n,flag)
end subroutine
end module
"""
)


def atomic_builder(tmp_path, source=SOURCE, names=("a", "b", "p", "q", "n", "flag"), extra=None):
    path = tmp_path / "atomic.f90"
    path.write_text(source)
    return ScopeBuilder(
        [path],
        "atomic_calls::step",
        options=CompilerOptions(opt_level=0),
        config=OffloadConfig(policy="sections", scope_execution="graph", host_threads=4),
        facts={
            "schema_version": 1,
            "participation": "serial",
            "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
            "captures": {**{"argument::" + name: dict(FACT) for name in names}, **(extra or {})},
        },
    )


def test_repeated_atomic_helper_projects_all_fields_and_exact_call_edges(tmp_path):
    builder = atomic_builder(tmp_path)
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report["graph_owner_attempt"]
    (scope,) = report["scopes"]
    (task,) = scope["source_device_tasks"]
    assert task["kind"] == "original_atomic_conditional_joined_team"
    projection = task["execution_binding"]["call_projection"]
    assert projection["field_roots"] == {
        "argument::a": ["argument::a", "argument::b"],
        "argument::p": ["argument::p", "argument::q"],
    }
    assert len(projection["calls"]) == 3
    assert projection["conservative_effects"] == {
        "reads": ["argument::a", "argument::b", "argument::p", "argument::q"],
        "metadata_reads": ["argument::a", "argument::b", "argument::p", "argument::q"],
        "may_writes": ["argument::a", "argument::b"],
        "must_writes": [],
        "overwrites": [],
    }
    assert projection["definition_authorized"] is projection["lifetime_authorized"] is False
    assert projection["alias_authorized"] is projection["value_specialization_authorized"] is False
    assert "fort_storage_identity_" in "\n".join(v for k, v in outputs.items() if k.endswith(".f90"))
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    validate_checked_execution_plans(json.loads(outputs["scope-manifest.json"]), capture_facts=builder.facts)


@pytest.mark.parametrize("form", ["section", "do", "point_predicate", "keywords"])
def test_atomic_projection_keeps_existing_team_and_predicate_authority(tmp_path, form):
    text = SOURCE
    names = ("a", "b", "p", "q", "n", "flag")
    if form == "do":
        text = text.replace("!$omp section\ndo i=-2,0\na(i)=p(i+n)\nenddo\n", "")
        text = text.replace("!$omp sections\n", "").replace("!$omp section\n", "!$omp do\n")
        text = text.replace("enddo\n", "enddo\n!$omp end do nowait\n").replace("!$omp end sections nowait\n", "")
    elif form == "point_predicate":
        text = text.replace("module atomic_calls\n", "module atomic_calls\ninteger::mask(2)\n")
        text = text.replace("if(flag)then", "if(mask(1)==1)then")
    elif form == "keywords":
        text = text.replace("call relay(b,q,n,flag)", "call relay(p=q,flag=flag,a=b,n=n)")
    builder = atomic_builder(
        tmp_path, text, names, extra={"atomic_calls::mask": dict(FACT)} if form == "point_predicate" else None
    )
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report["graph_owner_attempt"]
    (task,) = report["scopes"][0]["source_device_tasks"]
    assert task["version"] == 4
    assert task["source_proof"]["version"] == (2 if form == "do" else 3)
    if form == "point_predicate":
        projection = task["execution_binding"]["call_projection"]
        assert projection["field_roots"]["atomic_calls::mask"] == ["atomic_calls::mask"]
        assert "atomic_calls::mask" not in projection["conservative_effects"]["may_writes"]
        assert task["source_proof"]["predicates"][0]["host_point_reads"]
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    validate_checked_execution_plans(json.loads(outputs["scope-manifest.json"]), capture_facts=builder.facts)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_receipt",
        "foreign_receipt",
        "changed_source",
        "changed_handle",
        "changed_identity",
        "changed_resolved",
        "team_registry",
        "parent_registry",
        "unknown_namespace",
    ],
)
def test_atomic_projection_revalidates_receipts_before_sealing(tmp_path, monkeypatch, fault):
    from compiler.scopes import execution_plan
    from compiler.scopes.source_task_bindings import call_projection

    original = execution_plan._seal_execution
    checked = []

    def seal(owner):
        if checked:
            return original(owner)
        checked.append(True)
        member = owner.members["atomic_calls::helper"]
        relay = owner.members["atomic_calls::relay"]
        (task,) = member.source_device_tasks
        expected = call_projection(member, task)
        assert expected is not None
        receipts = owner.source_companion_calls
        receipt = receipts[-1]
        if fault == "missing_receipt":
            old = receipts.pop()

            def restore():
                return receipts.append(old)
        elif fault in {"foreign_receipt", "changed_source"}:
            receipts[-1] = (
                replace(receipt, callee=member)
                if fault == "foreign_receipt"
                else replace(receipt, source=receipt.source + "! changed")
            )

            def restore():
                return receipts.__setitem__(-1, receipt)
        elif fault == "changed_handle":
            values = relay.call_sites[-1]["handle_bindings"]
            key = next(iter(values))
            old = values[key]
            values[key] += 999

            def restore():
                return values.__setitem__(key, old)
        elif fault == "changed_identity":
            values = relay.resource_slot_roots
            key = next(iter(values))
            old = values[key]
            values[key] = "foreign::array"

            def restore():
                return values.__setitem__(key, old)
        elif fault == "changed_resolved":
            site = relay.call_sites[-1]
            key = (site["caller"], (site["first_line"], site["last_line"]))
            values = owner.builder.resolved_calls
            old = values[key]
            values[key] = {}

            def restore():
                return values.__setitem__(key, old)
        elif fault in {"team_registry", "parent_registry"}:
            proof = task.proof if fault == "team_registry" else task.proof.parent
            if fault == "team_registry":
                values = member.builder.analysis._atomic_worksharing_teams
            else:
                values = member.builder.analysis._joined_completions
            key = proof.identity
            old = values.pop(key)

            def restore():
                return values.__setitem__(key, old)
        else:
            old = relay.resource_slot_namespace
            relay.resource_slot_namespace = owner

            def restore():
                return setattr(relay, "resource_slot_namespace", old)

        try:
            with pytest.raises(CompilationError):
                call_projection(member, task)
        finally:
            restore()
        assert call_projection(member, task) == expected
        return original(owner)

    monkeypatch.setattr(execution_plan, "_seal_execution", seal)
    _outputs, report = atomic_builder(tmp_path).run()
    assert report["scope_count"] == 1, report["graph_owner_attempt"]
    assert checked
