"""One source inventory precedes cross-procedure ownership and placement."""

import shutil
import subprocess
from copy import copy
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.interval_draft import prepare_interval_draft
from compiler.scopes.lexical import LexicalOwner
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

SOURCE = """module interval_fields
implicit none
real(8)::old(-1:16),viscosity(-1:16),star(-1:16),out(-1:16)
integer::visits=0
contains
subroutine advance(n,correcting)
integer,intent(in)::n
logical,intent(in)::correcting
call producer(old,viscosity,n)
call momentum(old,viscosity,star,out,n,correcting)
end subroutine
subroutine producer(a,v,n)
real(8),intent(in)::a(-1:)
real(8),intent(inout)::v(-1:)
integer,intent(in)::n
integer::i
!$omp parallel do private(i) schedule(runtime)
do i=0,n-1
v(i)=2.d0*a(i)
enddo
!$omp end parallel do
end subroutine
subroutine momentum(a,v,s,o,n,correcting)
real(8),intent(in)::a(-1:)
real(8),intent(inout)::v(-1:),s(-1:),o(-1:)
integer,intent(in)::n
logical,intent(in)::correcting
integer::i
visits=visits+1
if(correcting) then
v(0)=v(0)+a(0)
endif
!$omp parallel do private(i) schedule(runtime)
do i=0,n-1
s(i)=v(i)+a(i)
enddo
!$omp end parallel do
!$omp parallel do private(i) schedule(runtime)
do i=0,n-1
o(i)=o(i)+s(i)
enddo
!$omp end parallel do
end subroutine
end module
"""


def generate(directory, *, source=SOURCE, config=None):
    path = directory / "interval.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"interval_fields::" + name: dict(FACT)
                          for name in ("old", "viscosity", "star", "out")}}
    config = config or OffloadConfig(policy="sections", scope_execution="reached")
    builder = ScopeBuilder([path], "interval_fields::advance", facts=facts,
                           options=CompilerOptions(gpu_policy=config.policy), config=config)
    outputs, report = builder.run()
    return path, builder, outputs, report


def draft_only(directory, source):
    path = directory / "interval.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"interval_fields::" + name: dict(FACT)
                          for name in ("old", "viscosity", "star", "out")}}
    builder = ScopeBuilder([path], "interval_fields::advance", facts=facts,
        options=CompilerOptions(gpu_policy="sections"),
        config=OffloadConfig(policy="sections", scope_execution="reached"))
    owner = LexicalOwner(builder)
    return prepare_interval_draft(owner, tuple(builder.entry.execution.content))


def test_cross_procedure_interval_keeps_one_inventory_and_original_correction(tmp_path):
    path, builder, outputs, report = generate(tmp_path)
    assert report["scope_count"] == 1, report.get("lexical_owner_attempt")
    scope, = report["scopes"]
    assert not scope["boundaries"], scope["boundaries"]
    plan = scope["checked_execution_plan"]
    assert plan["numerical_invocations"] == 3
    assert len(plan["procedures"]) == 3
    assert {row["resource"] for row in plan["resources"]} == {
        "interval_fields::old", "interval_fields::viscosity", "interval_fields::star", "interval_fields::out"}
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert "v(0)=v(0)+a(0)" in text
    assert "fort_scope_host_begin" in text
    assert "fort_scope_host_end" in text
    assert text.count("visits=visits+1") == 1
    drafts = tuple(builder._interval_drafts.values())
    assert len(drafts) == 1
    draft, = drafts
    assert draft.validate() is draft
    assert draft.demand.validate_plan(draft._owner.execution_plan)
    assert len(draft.numerical_anchors) == 3
    entry = draft.cost_entry()
    assert [binding.name for binding in entry["integers"]] == ["n"]
    assert entry["integers"][0] is builder.entry.scope.bindings["n"]
    assert all(binding.declaring_scope is builder.analysis.modules["interval_fields"] for binding in entry["arrays"])


def test_draft_and_native_context_require_exact_registered_authority(tmp_path):
    _path, builder, _outputs, _report = generate(tmp_path)
    draft, = builder._interval_drafts.values()
    with pytest.raises(CompilationError, match="exact unchanged issued"):
        copy(draft).validate()
    member = next(owner for owner in draft._owner.execution_plan.members
                  if owner.routine.qualified.endswith("::momentum"))
    context = draft.native_context(member)
    original = next(node for node in member.routine.execution.content if type(node).__name__ == "If_Construct")
    assert context.validate(builder.analysis, member.routine.qualified, (original,)) is context
    with pytest.raises(CompilationError, match="exact issued"):
        copy(context).validate(builder.analysis, member.routine.qualified, (original,))


@pytest.mark.parametrize("adopt", [False, True])
def test_interval_transaction_preserves_exact_draft_and_context_authority(tmp_path, adopt):
    _path, builder, _outputs, _report = generate(tmp_path)
    draft, = builder._interval_drafts.values()
    with GenerationTransaction(builder) as transaction:
        candidate, = transaction.builder._interval_drafts.values()
        assert candidate is not draft
        plan = candidate._owner.execution_plan
        assert plan.interval is candidate
        assert plan.validate() is plan
        for context in transaction.builder._interval_native_contexts.values():
            assert context._draft is candidate
            assert context.validate(transaction.builder.analysis, context._member.procedure,
                                    context._member.originals) is context
        if adopt:
            transaction.adopt()
    final, = builder._interval_drafts.values()
    assert (final is draft) is (not adopt)
    assert final._owner.execution_plan.validate()


def test_interval_identity_preserves_native_semantics_without_private_proof_tokens(tmp_path):
    plans, manifests = [], []
    for source in (SOURCE, SOURCE, SOURCE.replace("v(0)=v(0)+a(0)", "v(0)=v(0)-a(0)")):
        _path, builder, _outputs, _report = generate(tmp_path, source=source)
        draft, = builder._interval_drafts.values()
        manifests.append(draft.completion_manifest())
        plans.append(draft._owner.execution_plan)
    proofs = [tuple(unit.scope.native[0].demand_effects.identity
                    for member in plan.members for unit in plan.units(member)
                    if unit.scope.native and unit.scope.native[0].demand_effects is not None)
              for plan in plans]
    assert proofs[0]
    assert proofs[0] != proofs[1]
    assert plans[0].execution_identity == plans[1].execution_identity
    # Original native arithmetic remains part of the portable source contract.
    assert plans[0].execution_identity != plans[2].execution_identity
    assert manifests[0] == manifests[1]


def test_declaration_free_original_owner_and_children_compile(tmp_path):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if fortran is None:
        pytest.skip("native Fortran compiler required")
    source = """module interval_fields
implicit none
real(8)::old(-1:16),viscosity(-1:16),star(-1:16),out(-1:16)
integer::visits=0
contains
subroutine advance()
call producer()
call momentum()
end subroutine
subroutine producer()
call produce_values()
end subroutine
subroutine produce_values()
integer::i
!$omp parallel do private(i) schedule(runtime)
do i=0,15
viscosity(i)=2.d0*old(i)
enddo
!$omp end parallel do
end subroutine
subroutine momentum()
integer::i
visits=visits+1
!$omp parallel do private(i) schedule(runtime)
do i=0,15
star(i)=viscosity(i)+old(i)
enddo
!$omp end parallel do
!$omp parallel do private(i) schedule(runtime)
do i=0,15
out(i)=out(i)+star(i)
enddo
!$omp end parallel do
end subroutine
end module
"""
    _path, _builder, outputs, report = generate(tmp_path, source=source)
    scope, = report["scopes"]
    assert not scope["boundaries"]
    assert len(scope["checked_execution_plan"]["procedures"]) == 4
    build = tmp_path / "build"
    build.mkdir()
    for name, text in outputs.items():
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    ordinal = 0
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in report["build_sources"]:
            if item["role"] != role or item["language"] != "fortran":
                continue
            result = subprocess.run([fortran, "-O0", "-fopenmp", "-ffp-contract=off", "-I", str(build),
                "-J", str(build), "-c", str(build / item["path"]), "-o", str(build / f"source-{ordinal}.o")],
                capture_output=True, text=True, check=False, timeout=30)
            assert result.returncode == 0, result.stdout + result.stderr
            ordinal += 1


def test_diamond_and_repeated_calls_share_members_but_keep_original_edges(tmp_path):
    wrappers = """subroutine left(a,v,n)
real(8),intent(in)::a(-1:)
real(8),intent(inout)::v(-1:)
integer,intent(in)::n
call producer(a,v,n)
end subroutine
subroutine right(a,v,n)
real(8),intent(in)::a(-1:)
real(8),intent(inout)::v(-1:)
integer,intent(in)::n
call producer(a,v,n)
call producer(a,v,n-1)
end subroutine
"""
    source = SOURCE.replace("call producer(old,viscosity,n)",
                            "call left(old,viscosity,n)\ncall right(old,viscosity,n)")
    source = source.replace("subroutine producer(a,v,n)", wrappers + "subroutine producer(a,v,n)")
    draft = draft_only(tmp_path, source)
    producer, = [member for member in draft.members if member.procedure.endswith("::producer")]
    incoming = [edge for edge in draft.edges if edge.callee == producer.ordinal]
    assert len(draft.members) == 5
    assert len(draft.edges) == 6
    assert len(incoming) == 3
    assert len({id(edge.original) for edge in incoming}) == 3
    assert len({edge.caller for edge in incoming}) == 2
    assert len(draft.numerical_anchors) == 3
    assert draft.public()["work"]["call_instances"] == 6
    root_edges = [edge for edge in draft.edges if edge.caller == 0]
    assert [str(edge.original.items[0]).lower() for edge in root_edges] == ["left", "right", "momentum"]
    assert all(draft.has_numerical_child(edge.original) for edge in root_edges)
    assert draft.validate() is draft
    with pytest.raises(FrozenInstanceError):
        incoming[0].callee = 0


def test_distinct_actual_members_and_same_procedure_retarget_rejection(tmp_path):
    source = SOURCE.replace("call producer(old,viscosity,n)",
        "call producer(old,viscosity,n)\ncall producer(old,star,n)")
    draft = draft_only(tmp_path, source)
    producers = [member for member in draft.members if member.procedure.endswith("::producer")]
    assert len(producers) == 2
    targets = [{binding.root for _local, binding in member.mappings} for member in producers]
    assert targets == [{"interval_fields::old", "interval_fields::viscosity"},
                       {"interval_fields::old", "interval_fields::star"}]
    original_edges = draft.edges
    first = next(edge for edge in original_edges if edge.callee == producers[0].ordinal)
    corrupted = replace(first, callee=producers[1].ordinal)
    # Even bypassing the frozen dataclass cannot retarget a valid source CALL
    # to a same-procedure member with a different canonical actual allocation.
    object.__setattr__(draft, "edges", tuple(corrupted if edge is first else edge for edge in original_edges))
    try:
        with pytest.raises(CompilationError, match="canonical mapping changed"):
            draft.validate()
    finally:
        object.__setattr__(draft, "edges", original_edges)
    assert draft.validate() is draft
    object.__setattr__(draft, "edges", original_edges[:-1])
    try:
        with pytest.raises(CompilationError, match="call edge inventory changed"):
            draft.validate()
    finally:
        object.__setattr__(draft, "edges", original_edges)


def test_dag_memoization_does_not_hide_recursion(tmp_path):
    source = SOURCE.replace("integer::i\n!$omp", "integer::i\ncall producer(a,v,n)\n!$omp", 1)
    with pytest.raises(CompilationError, match="recursive or over-depth"):
        draft_only(tmp_path, source)
