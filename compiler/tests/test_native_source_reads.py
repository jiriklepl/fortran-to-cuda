"""Native metadata reads use the same reached, partial-definition contract."""

from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.reached_read_requirements import native_read_sections, prove_native_read_requirements
from compiler.ir import CompilationError
from compiler.scopes.closed_selection import ClosedIntervalOwner
from compiler.scopes.source_reads import validate_native_source_reads
from compiler.tests.test_interval_draft import SOURCE
from compiler.tests.test_original_interval_owner import builder, call_pair
from compiler.tests.test_selection_activation import PREFIX
from compiler.tests.test_source_scopes import FACT

METADATA_SOURCE = SOURCE.replace("integer::visits=0", "integer::visits=0\ninteger::tiles(3)=[-7,0,-9]")
METADATA_SOURCE = METADATA_SOURCE.replace("integer::i\n!$omp", "integer::i,limit\nlimit=tiles(2)\n!$omp", 1)
METADATA_SOURCE = METADATA_SOURCE.replace("do i=0,n-1", "do i=0,limit-1", 1)


def emit_native_metadata_fixture(path, *, protected=False):
    source = METADATA_SOURCE.replace("integer::tiles", "integer,protected::tiles") if protected else METADATA_SOURCE
    selected = builder(path, source=source, prefix=PREFIX + "tiles(2)=n\n")
    for name in ("out", "tiles"):
        selected.facts["captures"]["interval_fields::" + name] = {**FACT, "initialized": "none"}
    site = call_pair(selected)
    owner = ClosedIntervalOwner(selected, site.originals, site.prefix, site.controls)
    assert owner.run(site.originals)
    outputs, report = owner.builder.finish()
    return selected, owner, outputs, report


@pytest.mark.parametrize("protected", [False, True])
def test_native_metadata_read_keeps_unknown_entry_and_exact_sections(tmp_path, protected):
    selected, owner, outputs, report = emit_native_metadata_fixture(tmp_path / "native.f90", protected=protected)
    native = [(member, operation) for member in owner.execution_plan.members for unit in member.units
              for operation in unit.scope.native if operation.source_reads is not None]
    member, operation = next((member, operation) for member, operation in native
                              if "tiles" in str(operation.nodes[0]))
    assert operation.kind == ("native source" if protected else "original demand-relative native operation")
    sections = validate_native_source_reads(member.builder, operation)
    resource, = sections.resources
    assert resource.resource == "interval_fields::tiles"
    read, = resource.reads
    assert read.axes[0].lower.value == read.axes[0].upper.value == 2
    assert not resource.writes
    assert not resource.overwrites
    assert selected.facts["captures"][resource.resource]["initialized"] == "none"
    assert owner.emission.selection_prefix.capture_state(resource.resource)["initialized"] == "none"
    assert owner.execution_plan.validate() is owner.execution_plan
    source = outputs[report["sources"][str(tmp_path / "native.f90")]["replacement"]]
    assert source.count("tiles(2)=n") == 1
    assert "fort_native_source_read" in source
    assert "fort_scope_admit_source_reads_v1" in source


@pytest.mark.parametrize("replacement", [
    "if(n>0) limit=tiles(2)",
    "limit=merge(tiles(1),tiles(2),n>0)",
    "limit=sum(tiles,mask=tiles>0)",
])
def test_conditional_native_unions_do_not_grant_must_read_coverage(tmp_path, replacement):
    selected = builder(tmp_path / "source.f90", source=METADATA_SOURCE.replace("limit=tiles(2)", replacement))
    routine = selected.analysis.routines["interval_fields::producer"]
    node = next(node for node in routine.execution.content if "tiles" in str(node))
    assert prove_native_read_requirements(selected.analysis, routine.qualified, (node,)) is None


@pytest.mark.parametrize("mutation", ["copy", "selection", "analysis"])
def test_native_read_authority_cannot_be_borrowed(tmp_path, mutation):
    selected = builder(tmp_path / "source.f90", source=METADATA_SOURCE)
    analysis = selected.analysis
    routine = analysis.routines["interval_fields::producer"]
    selection = (next(node for node in walk(routine.execution)
                      if type(node).__name__ == "Assignment_Stmt"),)
    proof = prove_native_read_requirements(analysis, routine.qualified, selection)
    assert proof is not None
    assert native_read_sections(analysis, routine.qualified, selection, proof).resources
    if mutation == "copy":
        proof = copy(proof)
    elif mutation == "selection":
        selection = (copy(selection[0]),)
    else:
        analysis = analysis.private_projection()
        analysis._native_definition_proofs = {}
    with pytest.raises(CompilationError):
        native_read_sections(analysis, routine.qualified, selection, proof)
