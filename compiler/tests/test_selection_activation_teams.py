"""Multiple reached tiled units share one original team's guard scratch safely."""

import re
import shutil
import subprocess

import pytest

from compiler.scopes.closed_selection import ClosedIntervalOwner
from compiler.scopes.source_reads import validate_source_reads
from compiler.tests.test_interval_draft import SOURCE
from compiler.tests.test_joined_team_source import teams, write_fortran
from compiler.tests.test_original_interval_owner import builder, call_pair
from compiler.tests.test_selection_activation import emit_activation_fixture


def tiled(statement):
    return "!$omp do\n" + "\n".join([
        "do bk=1,n,tile", "do bj=1,n,tile", "do bi=1,n,tile",
        "do k=bk,min(bk+tile-1,n)", "do j=bj,min(bj+tile-1,n)", "do i=bi,min(bi+tile-1,n)",
        statement, *["enddo"] * 6, "!$omp end do"]) + "\n"


def tiled_source():
    source = SOURCE.replace("(-1:16)", "(-1:16,-1:16,-1:16)").replace("(-1:)", "(-1:,-1:,-1:)")
    source = source.replace("integer::visits=0", "integer::visits=0,tile=2")
    source = source.replace("integer::i\n", "integer::i,j,k,bi,bj,bk\n")
    source = source.replace("parallel do private(i)", "parallel do private(i,j,k)")
    source = source.replace("do i=0,n-1\nv(i)=2.d0*a(i)\nenddo", """do k=1,n
do j=1,n
do i=1,n
v(i,j,k)=2.d0*a(i,j,k)
enddo
enddo
enddo""")
    start = source.index("visits=visits+1")
    end = source.index("end subroutine", start)
    return source[:start] + """visits=visits+1
!$omp parallel private(i,j,k,bi,bj,bk)
""" + tiled("s(i,j,k)=v(i,j,k)+a(i,j,k)") + """!$omp do
do i=1,n
s(i,1,1)=s(i,1,1)+sum(a)
enddo
!$omp end do
""" + tiled("o(i,j,k)=o(i,j,k)+s(i,j,k)") + "!$omp end parallel\n" + source[end:]


@pytest.fixture(scope="module")
def emitted(tmp_path_factory):
    directory = tmp_path_factory.mktemp("activation_tiled_teams")
    path = directory / "teams.f90"
    selected = builder(path, source=tiled_source(), prefix="call opaque()\n")
    site = call_pair(selected)
    owner = ClosedIntervalOwner(selected, site.originals, site.prefix, site.controls)
    assert owner.run(site.originals)
    outputs, report = owner.builder.finish()
    text = outputs[report["sources"][str(path)]["replacement"]]
    return directory, owner, outputs, report, text


def test_multiple_tiled_original_units_retain_distinct_reached_read_authority(emitted):
    _directory, owner, _outputs, report, text = emitted
    (team, _operations), = teams(report)
    assert len(team["numerical_units"]) == len(team["source_reads"]) == 2
    for proof in team["source_reads"].values():
        assert len(proof["preconditions"]) == 6
        assert {item["role"] for item in proof["preconditions"]} == {"positive_tile_step", "tile_integer_range"}
    assert owner.emission.selection_prefix.public()["states"]
    assert all(state["initialized"] == "none" for state in owner.emission.selection_prefix.public()["states"])
    assert text.count("call opaque()") == 1
    assert text.count("s(i,1,1)=s(i,1,1)+sum(a)") == 1
    assert text.count("s(i,j,k)=v(i,j,k)+a(i,j,k)") == 1
    assert text.count("o(i,j,k)=o(i,j,k)+s(i,j,k)") == 1
    # All three typed tile guards need temporaries in each numerical DO.
    # They execute serially in the same master; their shared declarations
    # must occur once in that original team's enclosing specification.
    names = re.findall(r"^integer\(c_int64_t\) :: (fort_precondition_\d+)$", text, re.M)
    assert names
    assert len(names) == len(set(names))
    kinds = set()
    for member in owner.execution_plan.members:
        for unit in owner.execution_plan.units(member):
            requirements = getattr(unit.scope, "source_reads", None)
            if requirements is None:
                continue
            kinds.add(type(unit.scope).__name__)
            for call in unit.scope.calls:
                proof = requirements.get(call.procedure) if hasattr(requirements, "get") else requirements
                assert validate_source_reads(member.builder, call, proof) is proof
                originals = member.builder.inline_for(call.procedure).original_selection(call.node)
                assert len(originals) == len(proof.prepared._selection)
                assert all(left is right for left, right in zip(originals, proof.prepared._selection, strict=True))
    assert kinds == {"JoinedTeam", "CombinedNumericalTeam"}


def test_multiple_tiled_original_units_compile_with_shared_guard_scratch(emitted):
    directory, _owner, outputs, report, _text = emitted
    compiler = shutil.which("gfortran-15") or shutil.which("gfortran")
    if compiler is None:
        pytest.skip("native Fortran compiler unavailable")
    build = directory / "syntax"
    paths = write_fortran(outputs, report, build)
    result = subprocess.run([compiler, "-cpp", "-fopenmp", "-ffree-line-length-none", "-fsyntax-only",
                             *map(str, paths)], cwd=build, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("route", ["lexical", "joined"])
def test_mutable_preparation_registry_cannot_reassociate_original_reads(emitted, tmp_path, route):
    from compiler.ir import CompilationError

    if route == "lexical":
        _selected, owner, _outputs, _report = emit_activation_fixture(tmp_path / "original.f90")
        expected_kind = "CombinedNumericalTeam"
    else:
        _directory, owner, _outputs, _report, _text = emitted
        expected_kind = "JoinedTeam"
    member = next(item for item in owner.execution_plan.members
                  if item.routine.qualified == "interval_fields::momentum")
    pairs = []
    for unit in owner.execution_plan.units(member):
        if type(unit.scope).__name__ != expected_kind:
            continue
        requirements = unit.scope.source_reads
        pairs.extend((call, requirements.get(call.procedure) if hasattr(requirements, "get") else requirements)
                     for call in unit.scope.calls)
    (first, first_proof), (_second, second_proof) = pairs
    assert first_proof.procedure == second_proof.procedure
    inline = member.builder.inline_for(first.procedure)
    original = inline.prepared_regions[first.procedure]
    try:
        inline.prepared_regions[first.procedure] = second_proof.prepared
        with pytest.raises(CompilationError, match="original.*(selection|source|position)"):
            validate_source_reads(member.builder, first, second_proof)
    finally:
        inline.prepared_regions[first.procedure] = original


def test_combined_original_team_facade_retains_exact_read_selection(tmp_path):
    from compiler.frontend.reached_read_requirements import prove_reached_read_requirements
    from compiler.tests.support.numerical_regions import combined_units, emit_combined_team_fixture

    _paths, _outputs, _report, owner = emit_combined_team_fixture(tmp_path)
    units = combined_units(owner)
    assert units
    for unit in units:
        scope = unit.scope
        call, = scope.calls
        prepared = scope.prepared
        proof = prove_reached_read_requirements(scope.builder.analysis, prepared.procedure,
                                                prepared._selection, prepared)
        originals = scope.builder.inline_for(call.procedure).original_selection(call.node)
        assert len(originals) == len(prepared._selection)
        assert all(left is right for left, right in zip(originals, prepared._selection, strict=True))
        assert validate_source_reads(scope.builder, call, proof) is proof
