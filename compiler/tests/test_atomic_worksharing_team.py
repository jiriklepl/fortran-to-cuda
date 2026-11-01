"""Conditional NOWAIT work is indivisible through its original parent join."""
from copy import copy
from dataclasses import replace

import pytest

from compiler.frontend.atomic_worksharing_team import atomic_worksharing_alternatives, prove_atomic_worksharing_team
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.worksharing_completion import ordered_worksharing_inventory
from compiler.ir import CompilationError

ENTRY = "atomic_fixture::step"


def unit(target, value):
    return f"!$omp do\ndo i=1,n\n{target}(i)={value}\nenddo\n!$omp end do nowait\n"


BODY = ("if(mode==4)then\n" + unit("a", "a(i)+p(i-1)") + unit("b", "b(i)+p(i+1)")
        + "else\n" + unit("a", "a(i)+p(i)") + "if(flag)then\n"
        + unit("b", "b(i)+p(i)") + "else\n" + unit("b", "b(i)-p(i)") + "endif\nendif\n")


def fixture(tmp_path, body=BODY, *, declarations="", opening="parallel private(i)"):
    path = tmp_path / "atomic.f90"
    path.write_text("module atomic_fixture\ncontains\nsubroutine step(a,b,p,n,mode,flag)\n"
                    "real(8),intent(inout)::a(:),b(:)\nreal(8),intent(in)::p(0:)\n"
                    "integer,intent(in)::n,mode\nlogical,intent(in)::flag\ninteger::i,j\n"
                    + declarations + "\n!$omp " + opening + "\n" + body
                    + "!$omp end parallel\nend subroutine\nend module\n")
    analysis = SourceEffects([path])
    routine = analysis.routines[ENTRY]
    groups = [item.original_nodes for item in grouped_original_sequence(routine.execution.content, {})
              if isinstance(item, NativeGroupCandidate)]
    nodes, = groups
    return path, analysis, routine, nodes


def test_complete_conditional_nowait_team_has_parent_only_authority(tmp_path):
    _path, analysis, routine, nodes = fixture(tmp_path)
    proof = prove_atomic_worksharing_team(analysis, routine, nodes)
    public = proof.public()
    assert public["alternatives"] == [[0, 1], [2, 3], [2, 4]]
    assert len(public["predicates"]) == 2
    assert all(unit["nowait"] for unit in public["units"])
    assert public["parent_completion"]["join"] == "explicit original parallel end"
    assert "parent PARALLEL join only" in public["completion"]
    assert all(public[name] is False for name in ("execution_authorized", "gpu_legality_established",
               "definitions_authorized", "capture_authorized", "alias_authorized", "motion_authorized"))
    assert proof.validate(analysis, routine, nodes) is proof
    assert prove_atomic_worksharing_team(analysis, routine, nodes) is proof
    with pytest.raises(CompilationError, match="NOWAIT remains native"):
        ordered_worksharing_inventory(analysis, ENTRY, nodes)


def test_empty_original_branch_is_retained_without_speculative_unit(tmp_path):
    _, analysis, routine, nodes = fixture(tmp_path, "if(flag)then\n" + unit("a", "p(i)") + "endif\n")
    proof = prove_atomic_worksharing_team(analysis, routine, nodes)
    assert proof.public()["alternatives"] == [[0], []]


@pytest.mark.parametrize(("body", "declarations", "opening"), [
    (unit("a", "p(i)") + unit("b", "a(i)"), "", "parallel private(i)"),
    (unit("a", "p(i)") + unit("a", "p(i)+1.d0"), "", "parallel private(i)"),
    (unit("a", "p(i)") + unit("b", "p(i)").replace("b(i)=p(i)", "if(a(i)>0) b(i)=p(i)"), "", "parallel private(i)"),
    (unit("a", "p(i)").replace("a(i)=p(i)", "x=p(i)\na(i)=x"), "real(8)::x", "parallel private(i,x)"),
    (unit("a", "p(i)"), "real(8)::x", "parallel private(i,x)"),
    ("if(p(1)>0)then\n" + unit("a", "p(i)") + "endif\n", "", "parallel private(i)"),
    ("if(sin(1.d0)>0)then\n" + unit("a", "p(i)") + "endif\n", "", "parallel private(i)"),
    ("if(mode>4.d0)then\n" + unit("a", "p(i)") + "endif\n", "", "parallel private(i)"),
    (unit("a", "p(i)") + "call opaque(a)\n", "", "parallel private(i)"),
    (unit("a", "p(i)").replace("a(i)=p(i)", "a=p"), "", "parallel private(i)"),
    (unit("a", "p(i)").replace("a(i)=p(i)", "allocate(a(8))"), "", "parallel private(i)"),
])
def test_private_carries_dependencies_and_unproved_control_reject(tmp_path, body, declarations, opening):
    _, analysis, routine, nodes = fixture(tmp_path, body, declarations=declarations, opening=opening)
    with pytest.raises(CompilationError):
        prove_atomic_worksharing_team(analysis, routine, nodes)


@pytest.mark.parametrize("fault", ["copy", "registry", "record", "unit", "binding", "predicate", "namespace"])
def test_exact_source_token_state_cannot_be_substituted(tmp_path, fault):
    _, analysis, routine, nodes = fixture(tmp_path)
    proof = prove_atomic_worksharing_team(analysis, routine, nodes)
    if fault == "copy":
        proof = copy(proof)
    elif fault == "registry":
        analysis._atomic_worksharing_teams[proof.identity] = copy(proof)
    elif fault == "record":
        object.__setattr__(proof, "_record", "{}")
    elif fault == "unit":
        object.__setattr__(proof, "units", (replace(proof.units[0], loop=copy(proof.units[0].loop)), *proof.units[1:]))
    elif fault == "binding":
        object.__setattr__(proof.bindings[0], "kind", 4)
    elif fault == "predicate":
        object.__setattr__(proof, "predicates", ((copy(proof.predicates[0][0]), proof.predicates[0][1]),
                                               *proof.predicates[1:]))
    else:
        analysis.modules = dict(analysis.modules)
        analysis.modules.clear()
    with pytest.raises(CompilationError):
        proof.validate(analysis, routine, nodes)


def test_foreign_projection_and_source_change_cannot_reuse_team(tmp_path):
    path, analysis, routine, nodes = fixture(tmp_path)
    proof = prove_atomic_worksharing_team(analysis, routine, nodes)
    with pytest.raises(CompilationError):
        proof.validate(analysis.private_projection(), routine, nodes)
    path.write_text(path.read_text().replace("mode==4", "mode==2"))
    with pytest.raises(CompilationError):
        proof.validate()


def test_alternatives_use_normal_numerical_lowering_and_exact_physical_guards(tmp_path):
    from compiler.driver.options import CompilerOptions
    from compiler.frontend.numerical_effects import prove_numerical_effects
    from compiler.frontend.numerical_preparation import NumericalPreparationProvider
    from compiler.offload.config import OffloadConfig

    _, analysis, routine, nodes = fixture(tmp_path)
    proof = prove_atomic_worksharing_team(analysis, routine, nodes)
    provider = NumericalPreparationProvider(analysis, options=CompilerOptions(opt_level=0),
        config=OffloadConfig(policy="sections", host_threads=4), environment_root=ENTRY)
    alternatives = atomic_worksharing_alternatives(proof)
    for alternative in alternatives:
        prepared = provider.prepare_atomic_alternative(alternative)
        assert prepared.completion_proof is alternative
        assert prepared.completion["individual_unit_completion_authorized"] is False
        assert len(prepared.plan.regions) == 2
        assert prepared.offload.available
        assert "!$omp" not in prepared.extraction.source
        assert "if (" not in prepared.extraction.source.lower()
        assert prepared.validate(analysis, ENTRY, nodes) is prepared
        effect = prove_numerical_effects(analysis, ENTRY, nodes, prepared)
        assert len(effect.steps) == 2
        assert {item.resource for item in effect.sections.resources} == {"argument::a", "argument::b", "argument::p"}
        assert effect.public()["execution_authorized"] is False
    with pytest.raises(CompilationError, match="registered exact"):
        replace(alternatives[0], loops=alternatives[1].loops).validate(analysis, ENTRY, nodes)


def test_original_real_parameters_use_typed_readonly_abi_without_host_initializers(tmp_path):
    from compiler.driver.options import CompilerOptions
    from compiler.frontend.numerical_preparation import NumericalPreparationProvider
    from compiler.ir import ParallelRegion
    from compiler.offload.config import OffloadConfig

    body = BODY.replace("a(i)+p(i-1)", "a(i)+c1*p(i-1)-c3*p(i)")
    _, analysis, routine, nodes = fixture(tmp_path, body,
        declarations="real(8),parameter::c1=9._8/8,c3=1._8/(8*3)")
    team = prove_atomic_worksharing_team(analysis, routine, nodes)
    policy = NumericalPreparationProvider(analysis, options=CompilerOptions(opt_level=0),
        config=OffloadConfig(policy="sections", host_threads=4), environment_root=ENTRY)
    prepared = policy.prepare_atomic_alternative(atomic_worksharing_alternatives(team)[0])
    assert all(isinstance(step, ParallelRegion) for step in prepared.plan.steps)
    assert prepared.extraction.immutable_scalar_captures == (ENTRY + "::c1", ENTRY + "::c3")
    assert not prepared.extraction.immutable_constants
    assert "parameter ::" not in prepared.extraction.source
    assert {p.resource for p in prepared.extraction.parameters} >= {ENTRY + "::c1", ENTRY + "::c3"}
