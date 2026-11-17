"""Whole joined native definitions seed exact sections in one resident chain."""

import re
from copy import copy, deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.frontend.call_bindings import SourceDependency
from compiler.frontend.native_definitions import (
    NativeDefinitionUnsupported,
    _covers,
    invalidate_rectangles,
    prove_native_definitions)
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.native_sections import (
    NativeAxis,
    NativeBound,
    NativeIteration,
    NativeRectangle,
    NativeResourceSections,
)
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.scopes.module_owner import ModuleOwner
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

ENTRY = "coast_owner::advance"
ROOTS = ("argument::a", "argument::seed", "argument::delta", "argument::out")
OWNER_SOURCE = 'module coast_owner\nuse coast_numerics,only:produce=>form_delta,consume=>apply_delta\nimplicit none\ninteger::prefix_visits=0,tail_visits=0\ncontains\nsubroutine advance(a,seed,delta,out,n,m,rounds)\ninteger,intent(in)::n,m,rounds\nreal(8),intent(in)::a(-3:,-2:)\nreal(8),intent(out)::seed(-3:,-2:),delta(-3:,-2:),out(-3:,-2:)\ninteger::i,j,stage\nprefix_visits=prefix_visits+1\n!$omp parallel private(i,j)\n!$omp do\ndo j=-1,m-2\ndo i=-2,n-3\nseed(i,j)=a(i,j)+real(i+3*j,8)\nenddo\nenddo\n!$omp end do\n!$omp end parallel\ndo stage=1,rounds\ncall produce(seed,delta,n,m)\ncall consume(seed,delta,out,n,m)\nenddo\n! Original native halo restoration occurs after the selected window.\ndo j=-2,m-1\nseed(-3,j)=a(-3,j)\nseed(n-2,j)=a(n-2,j)\ndelta(-3,j)=a(-3,j)+1.0_8\ndelta(n-2,j)=a(n-2,j)+1.0_8\nout(-3,j)=a(-3,j)+2.0_8\nout(n-2,j)=a(n-2,j)+2.0_8\nenddo\ndo i=-2,n-3\nseed(i,-2)=a(i,-2)\nseed(i,m-1)=a(i,m-1)\ndelta(i,-2)=a(i,-2)+1.0_8\ndelta(i,m-1)=a(i,m-1)+1.0_8\nout(i,-2)=a(i,-2)+2.0_8\nout(i,m-1)=a(i,m-1)+2.0_8\nenddo\ntail_visits=tail_visits+1\nend subroutine\nend module\n'
CHILD_SOURCE = 'module coast_numerics\nimplicit none\ncontains\nsubroutine form_delta(seed,delta,n,m)\ninteger,intent(in)::n,m\nreal(8),intent(in)::seed(-3:,-2:)\nreal(8),intent(out)::delta(-3:,-2:)\ninteger::i,j\ndo j=-1,m-2\ndo i=-2,n-3\ndelta(i,j)=2.0_8*seed(i,j)+real(i-2*j,8)\nenddo\nenddo\nend subroutine\nsubroutine apply_delta(seed,delta,out,n,m)\ninteger,intent(in)::n,m\nreal(8),intent(in)::seed(-3:,-2:),delta(-3:,-2:)\nreal(8),intent(out)::out(-3:,-2:)\ninteger::i,j\ndo j=-1,m-2\ndo i=-2,n-3\nout(i,j)=delta(i,j)+0.5_8*seed(i,j)\nenddo\nenddo\nend subroutine\nend module\n'


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


def test_translation_comparison_never_swallows_shared_budget_failure():
    scalar = NativeBound("scalar", resource="argument::n")
    shifted = NativeBound("binary", operator="+", children=(scalar, NativeBound("literal", 0)))
    definition = NativeRectangle((NativeAxis(NativeBound("literal", 0), shifted),), "original write")
    read = replace(definition, axes=(replace(definition.axes[0], upper=scalar),))
    def exhausted(_count):
        raise CompilationError("shared original work budget exhausted")
    with pytest.raises(CompilationError, match="shared original work"):
        _covers(definition, read, consume=exhausted)
