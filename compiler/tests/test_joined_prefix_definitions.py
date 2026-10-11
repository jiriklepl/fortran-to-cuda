"""Whole joined native definitions seed exact sections in one resident chain."""

import re
from copy import copy, deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.call_bindings import SourceDependency
from compiler.frontend.native_definitions import (
    NativeDefinitionUnsupported,
    _covers,
    invalidate_rectangles,
    prove_native_definitions,
    rectangle_covers,
)
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.native_sections import (
    NativeAxis,
    NativeBound,
    NativeIteration,
    NativeRectangle,
    NativeResourceSections,
)
from compiler.frontend.reached_windows import plan_reached_window, prove_reached_window
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.module_owner import ModuleOwner
from compiler.scopes.reached_owner import ReachedWindowOwner
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

ENTRY = "coast_owner::advance"
ROOTS = ("argument::a", "argument::seed", "argument::delta", "argument::out")
OWNER_SOURCE = 'module coast_owner\nuse coast_numerics,only:produce=>form_delta,consume=>apply_delta\nimplicit none\ninteger::prefix_visits=0,tail_visits=0\ncontains\nsubroutine advance(a,seed,delta,out,n,m,rounds)\ninteger,intent(in)::n,m,rounds\nreal(8),intent(in)::a(-3:,-2:)\nreal(8),intent(out)::seed(-3:,-2:),delta(-3:,-2:),out(-3:,-2:)\ninteger::i,j,stage\nprefix_visits=prefix_visits+1\n!$omp parallel private(i,j)\n!$omp do\ndo j=-1,m-2\ndo i=-2,n-3\nseed(i,j)=a(i,j)+real(i+3*j,8)\nenddo\nenddo\n!$omp end do\n!$omp end parallel\ndo stage=1,rounds\ncall produce(seed,delta,n,m)\ncall consume(seed,delta,out,n,m)\nenddo\n! Original native halo restoration occurs after the selected window.\ndo j=-2,m-1\nseed(-3,j)=a(-3,j)\nseed(n-2,j)=a(n-2,j)\ndelta(-3,j)=a(-3,j)+1.0_8\ndelta(n-2,j)=a(n-2,j)+1.0_8\nout(-3,j)=a(-3,j)+2.0_8\nout(n-2,j)=a(n-2,j)+2.0_8\nenddo\ndo i=-2,n-3\nseed(i,-2)=a(i,-2)\nseed(i,m-1)=a(i,m-1)\ndelta(i,-2)=a(i,-2)+1.0_8\ndelta(i,m-1)=a(i,m-1)+1.0_8\nout(i,-2)=a(i,-2)+2.0_8\nout(i,m-1)=a(i,m-1)+2.0_8\nenddo\ntail_visits=tail_visits+1\nend subroutine\nend module\n'
CHILD_SOURCE = 'module coast_numerics\nimplicit none\ncontains\nsubroutine form_delta(seed,delta,n,m)\ninteger,intent(in)::n,m\nreal(8),intent(in)::seed(-3:,-2:)\nreal(8),intent(out)::delta(-3:,-2:)\ninteger::i,j\ndo j=-1,m-2\ndo i=-2,n-3\ndelta(i,j)=2.0_8*seed(i,j)+real(i-2*j,8)\nenddo\nenddo\nend subroutine\nsubroutine apply_delta(seed,delta,out,n,m)\ninteger,intent(in)::n,m\nreal(8),intent(in)::seed(-3:,-2:),delta(-3:,-2:)\nreal(8),intent(out)::out(-3:,-2:)\ninteger::i,j\ndo j=-1,m-2\ndo i=-2,n-3\nout(i,j)=delta(i,j)+0.5_8*seed(i,j)\nenddo\nenddo\nend subroutine\nend module\n'


def joined_prefix_fixture(directory, *, precision=8, device_budget=None, controls="out",
                          owner_source=OWNER_SOURCE, child_source=CHILD_SOURCE, captures=None,
                          roots=ROOTS, expected_calls=2):
    directory.mkdir(parents=True, exist_ok=True)
    if controls not in {"out", "sentinel"}:
        raise ValueError("unknown joined-prefix fixture control")
    if controls == "sentinel":
        owner_source = owner_source.split("! Original native halo restoration", 1)[0] + (
            "tail_visits=tail_visits+1\nend subroutine\nend module\n")
        owner_source = owner_source.replace("intent(out)", "intent(inout)")
        child_source = child_source.replace("intent(out)", "intent(inout)")
    paths = directory / "coast_numerics.f90", directory / "coast_owner.f90"
    for path, source in zip(paths, (child_source, owner_source), strict=True):
        path.write_text(source.replace("real(8)", f"real({precision})")
                        .replace("_8", f"_{precision}").replace(",8)", f",{precision})"))
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths},
             "captures": {roots[0]: deepcopy(FACT), **{root: {**deepcopy(FACT), "initialized": "none",
                          "lifetime": "selected_window"} for root in roots[1:]}}}
    for root, values in (captures or {}).items():
        facts["captures"][root].update(values)
    if device_budget is not None:
        facts["device_budget_bytes"] = device_budget
    builder = ScopeBuilder(paths, ENTRY, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="reached"))
    outer = next(node for node in builder.entry.execution.content
                 if type(node).__name__ == "Block_Nonlabel_Do_Construct"
                 and "stage" in str(node.content[0]).lower())
    originals = tuple(outer.content[1:-1])
    assert sum(type(node).__name__ == "Call_Stmt" for node in originals) == expected_calls
    proof = plan_reached_window(builder.analysis, ENTRY, originals, roots, facts)
    return paths, builder, proof, originals


def emit_joined_prefix_fixture(directory, **options):
    paths, builder, proof, originals = joined_prefix_fixture(directory, **options)
    owner = ReachedWindowOwner(builder, proof, originals)
    assert owner.run()
    outputs, report = owner.builder.finish()
    return paths, outputs, report, owner


def native_fixture(directory, body, *, operations=256):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "joined.f90"
    path.write_text("module renamed_native\nimplicit none\ncontains\nsubroutine touch(a,seed,delta,n,m,flag)\n"
                    "real(8),intent(in)::a(-3:,-2:)\n"
                    "real(8),intent(out)::seed(-3:,-2:),delta(-3:,-2:)\n"
                    "integer,intent(inout)::n,m\nlogical,intent(in)::flag\ninteger::i,j\n"
                    + body + "\nend subroutine\nend module\n")
    entry = "renamed_native::touch"
    analysis = SourceEffects([path], operations=operations)
    groups = [item for item in grouped_original_sequence(analysis.routines[entry].execution.content, {})
              if isinstance(item, NativeGroupCandidate)]
    assert len(groups) == 1
    selection = groups[0].original_nodes
    completion = analysis.joined_completion(entry, selection)
    return path, analysis, entry, selection, completion


def group_body(*assignments, nowait=False):
    units = []
    for assignment in assignments:
        units.append("!$omp do\ndo j=-1,m-2\ndo i=-2,n-3\n" + assignment
                     + "\nenddo\nenddo\n!$omp end do" + (" nowait" if nowait else ""))
    return "!$omp parallel private(i,j)\n" + "\n".join(units) + "\n!$omp end parallel"


def native_proof(directory, body, **options):
    path, analysis, entry, selection, completion = native_fixture(directory, body, **options)
    proof = prove_native_definitions(analysis, entry, selection, completion=completion)
    return path, analysis, entry, selection, completion, proof


def whole(rank=2):
    return (NativeRectangle(tuple(NativeAxis() for _ in range(rank)), "independently admitted entry"),)


@pytest.mark.parametrize("nowait", [False, True])
def test_joined_units_publish_only_exact_postjoin_interiors(tmp_path, nowait):
    _path, _analysis, _entry, _selection, completion, proof = native_proof(tmp_path,
        group_body("seed(i,j)=a(i,j)+1", "delta(i,j)=a(i,j)+2", nowait=nowait))
    result = proof.transfer({"argument::a": whole()})
    assert result.reads_covered
    assert len(result.definitions["argument::seed"]) == len(result.definitions["argument::delta"]) == 1
    assert result.definitions["argument::seed"][0].iterations
    assert proof.public()["schema_version"] == 3
    assert proof.public()["transfer_mode"] == "whole_joined_entry_snapshot"
    assert proof.public()["completion_identity"] == completion.identity
    assert proof.public()["execution_authorized"] is False
    assert proof.public()["whole_definition_inferred"] is False


@pytest.mark.parametrize("nowait", [False, True])
def test_joined_reads_require_entry_coverage_even_after_a_writer(tmp_path, nowait):
    _path, _analysis, _entry, _selection, _completion, proof = native_proof(tmp_path,
        group_body("seed(i,j)=a(i,j)", "delta(i,j)=seed(i,j)", nowait=nowait))
    result = proof.transfer({"argument::a": whole()})
    assert not result.reads_covered
    assert {read.resource for read in result.uncovered_reads} == {"argument::seed"}
    assert proof.transfer({"argument::a": whole(), "argument::seed": whole()}).reads_covered


def test_ordinary_definition_api_cannot_borrow_joined_authority(tmp_path):
    _path, analysis, entry, selection, _completion = native_fixture(tmp_path, group_body("seed(i,j)=1"))
    with pytest.raises(CompilationError):
        prove_native_definitions(analysis, entry, selection)
    assert not getattr(analysis, "_native_definition_proofs", {})


def test_joined_tokens_reject_foreign_copied_partial_reordered_and_stale_source(tmp_path):
    path, analysis, entry, selection, completion, proof = native_proof(tmp_path, group_body("seed(i,j)=1"))
    for token in (replace(completion), object(), completion.public()):
        with pytest.raises(CompilationError):
            prove_native_definitions(analysis, entry, selection, completion=token)
    with pytest.raises(CompilationError):
        prove_native_definitions(analysis, entry, selection[1:], completion=completion)
    with pytest.raises(CompilationError):
        prove_native_definitions(analysis, entry, tuple(reversed(selection)), completion=completion)
    with pytest.raises(CompilationError):
        proof.validate(copy(analysis), entry, selection)
    path.write_text(path.read_text().replace("seed(i,j)=1", "seed(i,j)=2"))
    with pytest.raises(CompilationError):
        proof.transfer()


@pytest.mark.parametrize("change", ["conditional", "multiple", "scalar"])
def test_unproved_joined_mustwrites_are_typed_support_boundaries(tmp_path, change):
    body = group_body("seed(i,j)=1")
    if change == "conditional":
        body = body.replace("!$omp do", "if(flag)then\n!$omp do").replace("!$omp end parallel", "endif\n!$omp end parallel")
    elif change == "multiple":
        body = body.replace("seed(i,j)=1", "seed(i,j)=1\ndelta(i,j)=2")
    else:
        body = body.replace("seed(i,j)=1", "n=n+1")
    _path, analysis, entry, selection, completion = native_fixture(tmp_path, body)
    with pytest.raises(NativeDefinitionUnsupported):
        prove_native_definitions(analysis, entry, selection, completion=completion)


def test_joined_parent_work_failure_stays_hard(tmp_path):
    _path, analysis, entry, selection, completion = native_fixture(tmp_path, group_body("seed(i,j)=1"))
    def fail(_count):
        raise CompilationError("parent shared source budget exhausted")
    with pytest.raises(CompilationError, match="parent shared") as failure:
        prove_native_definitions(analysis, entry, selection, completion=completion, consume=fail)
    assert not isinstance(failure.value, NativeDefinitionUnsupported)
    assert not getattr(analysis, "_native_definition_proofs", {})


def test_cold_and_reused_joined_proofs_charge_identical_work(tmp_path):
    _path, analysis, entry, selection, completion = native_fixture(tmp_path, group_body("seed(i,j)=a(i,j)"))
    cold, reused = [], []
    proof = prove_native_definitions(analysis, entry, selection, completion=completion, consume=cold.append)
    assert prove_native_definitions(analysis, entry, selection, completion=completion, consume=reused.append) is proof
    assert cold == reused
    assert sum(cold) == proof.public()["source_work"]


def test_omitted_original_middle_unit_cannot_form_definition_authority(tmp_path):
    _path, analysis, entry, selection, _completion = native_fixture(tmp_path,
        group_body("seed(i,j)=1", "delta(i,j)=2", "seed(i,j)=3"))
    loops = [index for index, node in enumerate(selection) if type(node).__name__ == "Block_Nonlabel_Do_Construct"]
    assert len(loops) == 3
    # Select only the opening, last whole DO and final join. The existing
    # completion constructor may decline first; either way no transfer token
    # may certify an unchanged source interval with omitted original work.
    partial = (selection[0], *selection[loops[-1] - 1:])
    with pytest.raises(CompilationError):
        prove_native_definitions(analysis, entry, partial, completion=analysis.joined_completion(entry, partial))
    assert not getattr(analysis, "_native_definition_proofs", {})


def test_typed_activation_sections_do_not_turn_prefix_interior_into_whole(tmp_path):
    _paths, builder, proof, originals = joined_prefix_fixture(tmp_path)
    public = proof.public()
    assert public["schema_version"] == 9
    assert public["prefix_eligibility"]["activation_path_possible"]
    assert not public["prefix_eligibility"]["disabled_path_possible"]
    sections = proof.activation_sections("argument::seed")
    assert isinstance(sections, NativeResourceSections)
    assert not sections.reads
    assert not sections.writes
    assert len(sections.overwrites) == 1
    assert sections.lower_bounds == (-3, -2)
    assert proof.activation_sections("argument::a") is None
    assert proof.activation_sections("argument::delta") is None
    assert proof.activation_sections("argument::out") is None
    assert [state["initialized"] for state in public["states"]] == ["whole", "sections", "none", "none"]
    assert proof.validate(builder.analysis, ENTRY, originals) is proof
    with pytest.raises(CompilationError):
        proof.activation_sections("foreign::seed")
    with pytest.raises(CompilationError, match="OpenMP"):
        prove_reached_window(builder.analysis, ENTRY, originals, ROOTS, builder.facts)


def test_selected_window_whole_input_does_not_prove_earlier_joined_reads(tmp_path):
    _paths, _builder, proof, _originals = joined_prefix_fixture(tmp_path,
        captures={"argument::a": {"lifetime": "selected_window"}})
    public = proof.public()
    assert not public["prefix_eligibility"]["activation_path_possible"]
    point, = public["prefix_eligibility"]["disable_points"]
    assert point["original_source_kind"] == "complete_original_joined_group"
    assert "selected-window whole" in point["reason"]
    assert len(point["selected_node_ids"]) > 1


@pytest.mark.parametrize("conditional", [False, True])
def test_activation_bounds_may_not_be_relabelled_before_lazy_registration(tmp_path, conditional):
    change = "if(n>1)then\nn=n-1\nendif" if conditional else "n=n-1"
    source = OWNER_SOURCE.replace("integer,intent(in)::n,m,rounds", "integer,intent(inout)::n\ninteger,intent(in)::m,rounds")
    source = source.replace("call produce(seed,delta,n,m)", change + "\ncall produce(seed,delta,n,m)")
    with pytest.raises(CompilationError, match="partial activation bound changes"):
        joined_prefix_fixture(tmp_path, owner_source=source)


def test_single_line_active_close_retains_whole_original_if_span(tmp_path):
    source = OWNER_SOURCE.replace("call consume(seed,delta,out,n,m)",
                                  "if(n>0)call opaque_child(delta)\ncall consume(seed,delta,out,n,m)")
    _paths, builder, proof, originals = joined_prefix_fixture(tmp_path, owner_source=source)
    original = next(node for node in originals if type(node).__name__ == "If_Stmt")
    point, = proof.public()["planned_closures"]["close_points"]
    graph = builder.analysis.structure(ENTRY)
    identity = graph.node_id(original, role="statement")
    assert point["selected_node_ids"] == [identity]
    assert point["original_source_kind"] == "If_Stmt"
    assert point["span"] == list(graph.nodes[identity].span)
    assert len(point["span"]) == 2
    assert "whole-construct" in point["reason"]


def test_pure_original_child_cannot_borrow_impure_runtime_companion(tmp_path):
    source = CHILD_SOURCE.replace("subroutine form_delta", "pure subroutine form_delta")
    _paths, builder, proof, originals = joined_prefix_fixture(tmp_path, child_source=source)
    parent = ReachedWindowOwner(builder, proof, originals)
    child = builder.analysis.routines["coast_numerics::form_delta"]
    with pytest.raises(CompilationError, match="ordinary impure subroutine entry"):
        ModuleOwner(parent, child, {})


def joined_child_source():
    body = "do j=-1,m-2\ndo i=-2,n-3\ndelta(i,j)=2.0_8*seed(i,j)+real(i-2*j,8)\nenddo\nenddo"
    original_group = ("!$omp parallel private(i,j)\n!$omp do\ndo j=-1,m-2\ndo i=-2,n-3\n"
                      "delta(i,j)=seed(i,j)\ndelta(i,j)=delta(i,j)+1.0_8\n"
                      "enddo\nenddo\n!$omp end do\n!$omp end parallel")
    return CHILD_SOURCE.replace(body, "if(m==13)then\n" + original_group + "\nelse\n" + body + "\nendif")


def joined_consumer_source():
    body = "do j=-1,m-2\ndo i=-2,n-3\nout(i,j)=delta(i,j)+0.5_8*seed(i,j)\nenddo\nenddo"
    original_group = ("!$omp parallel private(i,j)\n!$omp do\ndo j=-1,m-2\ndo i=-2,n-3\n"
                      "out(i,j)=delta(i,j)\nout(i,j)=out(i,j)+0.5_8*seed(i,j)\n"
                      "enddo\nenddo\n!$omp end do\n!$omp end parallel")
    return CHILD_SOURCE.replace(body, "if(m==13)then\n" + original_group + "\nelse\n" + body + "\nendif")


def test_mapped_consumer_closes_before_native_team_after_gpu_producer(tmp_path):
    _paths, outputs, report, owner = emit_joined_prefix_fixture(tmp_path, child_source=joined_consumer_source())
    child = owner.members["coast_numerics::apply_delta"]
    point, = child.reached_record()["planned_closures"]["close_points"]
    boundary, = child.boundaries
    assert point["selected_node_ids"] == boundary["original_node_ids"]
    assert len(point["selected_node_ids"]) > 1
    assert len(owner.owner.leaves) == 2
    generated = outputs[report["sources"][str(_paths[0])]["replacement"]]
    native = generated.index("out(i,j)=delta(i,j)")
    assert generated.rfind("fort_scope_close(", 0, native) >= 0
    assert generated.count("out(i,j)=out(i,j)+0.5_8*seed(i,j)") == 1


def test_mapped_child_support_boundary_closes_one_whole_original_team(tmp_path):
    _paths, _outputs, _report, owner = emit_joined_prefix_fixture(tmp_path, child_source=joined_child_source())
    child = owner.members["coast_numerics::form_delta"]
    point, = child.reached_record()["planned_closures"]["close_points"]
    boundary, = child.boundaries
    assert point["original_source_kind"] == "complete_original_joined_group"
    assert point["completion_identity"]
    assert len(point["selected_node_ids"]) > 1
    assert point["selected_node_ids"] == boundary["original_node_ids"]
    assert point["span"] == [boundary["first_line"], boundary["last_line"]]
    assert len(owner.owner.leaves) == 2
    child.boundaries = [{**boundary, "original_node_ids": boundary["original_node_ids"][:1]},
                        {**boundary, "original_node_ids": boundary["original_node_ids"][1:]}]
    with pytest.raises(CompilationError, match="staged closes"):
        child.validate_mapped_plan()


def test_unsupported_prefix_group_disables_before_whole_opening_only_on_its_arm(tmp_path):
    start = OWNER_SOURCE.index("!$omp parallel")
    end = OWNER_SOURCE.index("!$omp end parallel") + len("!$omp end parallel")
    native = OWNER_SOURCE[start:end]
    unsupported = native.replace("!$omp do", "if(n>0)then\n!$omp do").replace(
        "!$omp end parallel", "endif\n!$omp end parallel")
    source = OWNER_SOURCE[:start] + "if(n>1)then\n" + unsupported + "\nelse\n" + native + "\nendif" + OWNER_SOURCE[end:]
    _paths, outputs, report, owner = emit_joined_prefix_fixture(tmp_path, owner_source=source)
    prefix = owner.window_proof.public()["prefix_eligibility"]
    assert prefix["activation_path_possible"]
    assert prefix["disabled_path_possible"]
    point, = prefix["disable_points"]
    assert point["original_source_kind"] == "complete_original_joined_group"
    assert len(point["selected_node_ids"]) > 1
    generated = next(text for name, text in outputs.items() if name == report["sources"][str(_paths[1])]["replacement"])
    disabled = generated.index(f"{owner.prefix_eligible} = .false.")
    assert generated.index("if(n>1)then") < disabled < generated.index("!$omp parallel")


@pytest.mark.parametrize("controls", ["out", "sentinel"])
def test_emitted_joined_prefix_seeds_two_mapped_gpu_children(tmp_path, controls):
    paths, outputs, report, owner = emit_joined_prefix_fixture(tmp_path, controls=controls)
    original = paths[1].read_text()
    generated = outputs[report["sources"][str(paths[1])]["replacement"]]
    scope, = report["scopes"]
    assert len(scope["gpu_leaves"]) == 2
    assert len(owner.members) == 3
    assert original.count("!$omp parallel private(i,j)") == generated.count("!$omp parallel private(i,j)") == 1
    assert generated.count("prefix_visits=prefix_visits+1") == 1
    assert generated.count("tail_visits=tail_visits+1") == 1
    assert "fort_scope_register_sections(" in generated
    assert generated.index("!$omp end parallel") < generated.index(f"{owner.spent} = .true.")
    assert not scope["boundaries"]
    assert "intent(out)" in original if controls == "out" else "intent(out)" not in original
    assert "seed(-3,j)" in original if controls == "out" else "seed(-3,j)" not in original


def test_hidden_renamed_partial_module_is_registered_before_nested_borrow(tmp_path):
    owner_source = OWNER_SOURCE.replace("consume=>apply_delta", "consume=>apply_delta, &\n"
                                       "seed_alias=>cache,relay=>bridge")
    owner_source = owner_source.replace("advance(a,seed,delta,out,n,m,rounds)",
                                       "advance(a,delta,out,n,m,rounds)")
    owner_source = owner_source.replace("::seed(-3:,-2:),delta", "::delta")
    owner_source = owner_source.replace("seed(", "seed_alias(")
    owner_source = owner_source.replace("call produce(seed,delta,n,m)\ncall consume(seed,delta,out,n,m)",
                                       "call relay(delta,out,n,m)")
    child_source = CHILD_SOURCE.replace("implicit none\ncontains", "implicit none\n"
                                       "real(8),allocatable::cache(:,:)\ncontains\n"
                                       "subroutine bridge(delta,out,n,m)\n"
                                       "real(8),intent(out)::delta(-3:,-2:),out(-3:,-2:)\n"
                                       "integer,intent(in)::n,m\n"
                                       "call form_delta(delta,n,m)\ncall apply_delta(delta,out,n,m)\nend subroutine")
    child_source = child_source.replace("form_delta(seed,delta,n,m)", "form_delta(delta,n,m)")
    child_source = child_source.replace("apply_delta(seed,delta,out,n,m)", "apply_delta(delta,out,n,m)")
    child_source = child_source.replace("real(8),intent(in)::seed(-3:,-2:),delta(-3:,-2:)",
                                       "real(8),intent(in)::delta(-3:,-2:)")
    child_source = child_source.replace("real(8),intent(in)::seed(-3:,-2:)\n", "")
    child_source = child_source.replace("seed(i,j)", "cache(i,j)")
    roots = ("argument::a", "argument::delta", "argument::out", "coast_numerics::cache")
    paths, outputs, report, owner = emit_joined_prefix_fixture(tmp_path, owner_source=owner_source,
        child_source=child_source, roots=roots, expected_calls=1)
    bridge = owner.members["coast_numerics::bridge"]
    text = outputs[report["sources"][str(paths[1])]["replacement"]]
    assert owner.builder.capture(owner.builder.analysis._binding(owner.routine.scope, "seed_alias"))["source_sections"]
    assert text.index("fort_scope_register_sections(") < text.index("call " + bridge.entry)
    assert "shape(seed_alias, kind=c_size_t)" in text
    for name in ("bridge", "form_delta", "apply_delta"):
        child = owner.members["coast_numerics::" + name]
        binding = child.builder.analysis._binding(child.routine.scope, "cache")
        fact = child.builder.capture(binding)
        assert fact["requires_registered_handle"]
        assert fact["initialized"] == "none"
        assert "source_sections" not in fact
        assert "sections" not in fact
    assert len(owner.owner.leaves) == 2


def test_nested_unused_out_kills_partial_activation_without_reseeding(tmp_path):
    owner_source = OWNER_SOURCE.replace("call produce(seed,delta,n,m)",
                                       "call produce(a,delta,seed,n,m)\ncall produce(a,delta,seed,n,m)")
    child_source = """module coast_numerics
contains
subroutine form_delta(input,delta,unused,n,m)
real(8),intent(in)::input(-3:,-2:)
real(8),intent(out)::delta(-3:,-2:),unused(-3:,-2:)
integer,intent(in)::n,m
call leaf(input,delta,unused,n,m)
end subroutine
subroutine leaf(input,delta,unused,n,m)
real(8),intent(in)::input(-3:,-2:)
real(8),intent(out)::delta(-3:,-2:),unused(-3:,-2:)
integer,intent(in)::n,m
integer::i,j
do j=-1,m-2
do i=-2,n-3
delta(i,j)=2.0_8*input(i,j)
enddo
enddo
end subroutine
""" + "subroutine apply_delta" + CHILD_SOURCE.split("subroutine apply_delta", 1)[1]
    paths, outputs, report, owner = emit_joined_prefix_fixture(tmp_path, owner_source=owner_source,
        child_source=child_source, expected_calls=3)
    assert not owner.window_proof.public()["planned_closures"]["close_points"]
    consumer = owner.members["coast_numerics::apply_delta"]
    point, = consumer.reached_record()["planned_closures"]["close_points"]
    assert point["original_source_kind"] == "Block_Nonlabel_Do_Construct"
    assert "definition" in point["reason"]
    states = {item["resource"]: item for item in owner.window_proof.public()["states"]}
    assert states["argument::seed"]["active_path_coverage_on_return"] == "none"
    assert states["argument::seed"]["source_sections_on_activation"]
    for name in ("form_delta", "leaf"):
        child = owner.members["coast_numerics::" + name]
        fact = child.builder.capture(child.routine.scope.bindings["unused"])
        assert fact["initialized"] == "none"
        assert not {"source_sections", "sections", "requires_registered_handle"} & fact.keys()
    child_text = outputs[report["sources"][str(paths[0])]["replacement"]]
    assert "fort_scope_forget_definition(" in child_text
    assert consumer.boundaries[0]["original_node_ids"] == point["selected_node_ids"]
    assert child_text.count("out(i,j)=delta(i,j)+0.5_8*seed(i,j)") == 1


def shifted_child_source(source=CHILD_SOURCE):
    producer, consumer = source.split("subroutine apply_delta", 1)
    names = {"seed": "field", "delta": "increment", "n": "nx", "m": "ny"}
    producer = re.sub(r"\b(seed|delta|n|m)\b", lambda match: names[match[0]], producer)
    producer = producer.replace("(-3:,-2:)", "(-7:,5:)")
    producer = producer.replace("j=-1,ny-2", "j=6,ny+5").replace("i=-2,nx-3", "i=-6,nx-7")
    consumer = "subroutine apply_delta" + consumer
    names["seed"] = "history"
    consumer = re.sub(r"\b(seed|delta|n|m)\b", lambda match: names[match[0]], consumer)
    consumer = consumer.replace("(-3:,-2:)", "(-11:,9:)")
    consumer = consumer.replace("j=-1,ny-2", "j=10,ny+9").replace("i=-2,nx-3", "i=-10,nx-11")
    return producer + consumer


def test_shifted_child_formals_keep_root_partial_registration_coordinates(tmp_path):
    paths, outputs, report, owner = emit_joined_prefix_fixture(tmp_path, child_source=shifted_child_source())
    root_text = outputs[report["sources"][str(paths[1])]["replacement"]]
    child = owner.members["coast_numerics::form_delta"]
    assert "lbound(seed, kind=c_int64_t)" in root_text
    assert root_text.index("fort_scope_register_sections(") < root_text.index("call " + child.entry)
    fact = child.builder.capture(child.routine.scope.bindings["field"])
    assert fact["requires_registered_handle"]
    assert "source_sections" not in fact
    assert "sections" not in fact
    state = owner.window_proof.capture_state("argument::seed")
    assert state["source_sections_on_activation"]["overwrites"][0]["axes"][0]["lower"]["value"] == -2
    assert len(owner.owner.leaves) == 2


def test_constant_translation_keeps_original_nonempty_guard_and_scalar_identity():
    def literal(value):
        return NativeBound("literal", value)
    scalar = NativeBound("scalar", resource="argument::n",
                         dependencies=(SourceDependency("argument::n", "scalar_read", object()),))
    def add(base, amount):
        return NativeBound("binary", operator="+", children=(base, literal(amount)))
    # Both unit-stride loops have exactly n trips; axes retain their original
    # checked expressions instead of being rewritten by the comparison.
    definition = NativeRectangle((NativeAxis(literal(-2), add(add(scalar, -7), 4)),), "original write",
                                 (NativeIteration(literal(-6), add(scalar, -7)),))
    read = NativeRectangle((NativeAxis(literal(-2), add(scalar, -3)),), "original read",
                           (NativeIteration(literal(-2), add(scalar, -3)),))
    assert rectangle_covers(definition, read)
    assert definition.axes[0].upper.children[0].children[1].value == -7
    assert invalidate_rectangles((definition,), ("argument::n",)) == ()
    assert invalidate_rectangles((definition,), ("other_module::n",)) == (definition,)
    foreign = replace(read, axes=(replace(read.axes[0], upper=NativeBound("binary", operator="+", children=(
        NativeBound("scalar", resource="other_module::n"), literal(-3)))),))
    assert not rectangle_covers(definition, foreign)


def test_shifted_point_write_does_not_credit_different_or_absent_empty_guard():
    scalar = NativeBound("scalar", resource="argument::n")
    def literal(value):
        return NativeBound("literal", value)
    upper = NativeBound("binary", operator="-", children=(scalar, literal(1)))
    point = NativeAxis(literal(0), literal(0), point=True)
    definition = NativeRectangle((point,), "conditional point write", (NativeIteration(literal(0), upper),))
    # At n=0, the write is absent but the one-trip reader is reached.
    read = NativeRectangle((point,), "point read", (NativeIteration(literal(0), scalar),))
    assert not rectangle_covers(definition, read)
    assert not rectangle_covers(definition, replace(read, iterations=()))
    assert not rectangle_covers(definition, replace(read, iterations=(NativeIteration(literal(0), upper, -1),)))


@pytest.mark.parametrize("change", ["outside", "multiple_bases", "multiply", "overflow", "depth", "work"])
def test_unsupported_shift_comparisons_never_expand_coverage(change):
    def literal(value):
        return NativeBound("literal", value)
    scalar = NativeBound("scalar", resource="argument::n")
    shifted = NativeBound("binary", operator="+", children=(scalar, literal(0)))
    expected = scalar
    if change == "outside":
        expected = NativeBound("binary", operator="+", children=(scalar, literal(1)))
    elif change == "multiple_bases":
        shifted = NativeBound("binary", operator="+", children=(scalar, NativeBound("scalar", resource="argument::m")))
    elif change == "multiply":
        shifted = NativeBound("binary", operator="*", children=(scalar, literal(1)))
    elif change == "overflow":
        shifted = NativeBound("binary", operator="+", children=(scalar,
            NativeBound("binary", operator="+", children=(literal(2**31 - 1), literal(1)))))
    elif change == "depth":
        for _index in range(9):
            shifted = NativeBound("parenthesis", children=(shifted,))
    else:
        constant = literal(1)
        for _index in range(7):
            constant = NativeBound("binary", operator="+", children=(constant, constant))
        shifted = NativeBound("binary", operator="+", children=(scalar, constant))
        expected = NativeBound("binary", operator="+", children=(scalar, literal(128)))
    definition = NativeRectangle((NativeAxis(literal(0), shifted),), "original translated write")
    read = NativeRectangle((NativeAxis(literal(0), expected),), "original read")
    assert not rectangle_covers(definition, read)


def test_translation_comparison_never_swallows_shared_budget_failure():
    scalar = NativeBound("scalar", resource="argument::n")
    shifted = NativeBound("binary", operator="+", children=(scalar, NativeBound("literal", 0)))
    definition = NativeRectangle((NativeAxis(NativeBound("literal", 0), shifted),), "original write")
    read = replace(definition, axes=(replace(definition.axes[0], upper=scalar),))
    def exhausted(_count):
        raise CompilationError("shared original work budget exhausted")
    with pytest.raises(CompilationError, match="shared original work"):
        _covers(definition, read, consume=exhausted)
