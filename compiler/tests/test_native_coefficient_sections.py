"""Original joined CPU teams retain exact read-only coefficient-table bounds."""
# ruff: noqa: F811 - imported pytest fixtures.

from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.source_effects import SourceEffects
from compiler.offload.config import OffloadConfig
from compiler.scopes.access import build_native_access
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_scoped_native_unit_handoff import native_unit  # noqa: F401
from compiler.tests.test_scoped_planning_entries import calibration
from compiler.tests.test_scoped_planning_runtime import planning  # noqa: F401
from compiler.tests.test_scoped_runtime import runtime  # noqa: F401


def source(*, lower=1, upper=3, team=True):
    begin = "!$omp parallel private(i,j,k)\n" if team else ""
    end = "!$omp end parallel\n" if team else ""
    work = []
    for output in ("b", "c", "d"):
        work.append(
            ("!$omp do\n" if team else "")
            + "do k=1,n\ndo j=1,n\ndo i=1,n\n"
            + f"{output}(i,j,k)=a(i,j,k)*coefficient(stage)*dt\n"
            + "enddo\nenddo\nenddo\n"
            + ("!$omp end do nowait\n" if team else "")
        )
    return (
        """module coefficient_teams
implicit none
contains
subroutine advance(a,b,c,d,coefficient,n,stage,dt)
real(8),intent(in)::a(-2:,-2:,-2:),dt
real(8),intent(inout)::b(-2:,-2:,-2:),c(-2:,-2:,-2:),d(-2:,-2:,-2:)
integer,intent(in)::n,stage
real(8),intent(in)::coefficient("""
        + f"{lower}:{upper})\ninteger::i,j,k\n"
        + begin
        + "".join(work)
        + end
        + "end subroutine\nend module\n"
    )


def emit(directory, *, profile=None, lower=1):
    directory.mkdir(parents=True, exist_ok=True)
    original = directory / "original.f90"
    original.write_text(source(lower=lower, upper=lower + 2))
    builder = ScopeBuilder(
        [original],
        "coefficient_teams::advance",
        facts={
            "schema_version": 1,
            "participation": "serial",
            "sources": {str(original): sha256(original.read_bytes()).hexdigest()},
            "captures": {"argument::" + name: dict(FACT) for name in ("a", "b", "c", "d", "coefficient")},
        },
        options=CompilerOptions(opt_level=0),
        config=OffloadConfig("auto", calibration() if profile is None else profile),
    )
    outputs, manifest = builder.run()
    return original, outputs, manifest


@pytest.mark.parametrize("lower", [1, -1])
def test_original_3d_team_has_independent_native_coefficient_sections(tmp_path, lower):
    original, outputs, manifest = emit(tmp_path, lower=lower)
    segments = manifest["scopes"][0]["planning_segments"]
    (team,) = [item["operations"]["combined_numerical_team"] for item in segments]
    alternative = team["original_cpu_alternative"]
    assert alternative["available"], alternative["reason"]
    (coefficient,) = [r for r in alternative["sections"]["resources"] if r["resource"] == "argument::coefficient"]
    assert coefficient["guarded_explicit_dummy_extents"] == [3]
    assert coefficient["logical_lower_bounds"] == [lower]
    assert coefficient["reads"]
    assert not coefficient["writes"]
    assert not coefficient["overwrites"]
    text = outputs[manifest["sources"][str(original)]["replacement"]]
    assert "fort_team_native_begin" in text
    assert "/= [3_c_size_t]" in text
    assert "fort_team_run = .not. fort_team_native" in text
    assert "if (fort_team_native) then" in text


def test_exact_extent_guard_precedes_any_native_acquisition(tmp_path):
    path = tmp_path / "leaf.f90"
    path.write_text(source(team=False))
    analysis = SourceEffects([path])
    sections = analysis.native_sections("coefficient_teams::advance")
    assert sections.available, sections.reason
    (coefficient,) = [r for r in sections.resources if r.resource == "argument::coefficient"]
    code = build_native_access(
        coefficient, "handle", "fort_cf", parameters={"argument::stage": "stage", "argument::n": "n"}
    )
    prepare = "\n".join(code.prepare)
    assert "any(fort_cf_extents /= [3_c_size_t])" in prepare
    assert prepare.index("/= [3_c_size_t]") < prepare.index("fort_cf_bound1")
    assert "fort_scope_host_begin" not in prepare


@pytest.mark.parametrize(
    "declaration",
    [
        "real(8),intent(inout)::coefficient(1:3)",
        "real(8),intent(in)::coefficient(1:n)",
        "real(8),intent(in)::coefficient(*)",
        "real(8),intent(in)::coefficient(1:3,1:2)",
    ],
)
def test_unproved_explicit_shape_does_not_gain_native_physical_authority(tmp_path, declaration):
    text = source(team=False).replace("real(8),intent(in)::coefficient(1:3)", declaration)
    if "1:3,1:2" in declaration:
        text = text.replace("coefficient(stage)", "coefficient(stage,1)")
    path = tmp_path / "decline.f90"
    path.write_text(text)
    sections = SourceEffects([path]).native_sections("coefficient_teams::advance")
    assert not sections.available
    assert "assumed-shape array dummies" in sections.reason or "constant original bounds" in sections.reason


@pytest.mark.native
@pytest.mark.parametrize(("lower", "extent"), [(1, 3), (-1, 3), (1, 2), (1, 4)])
def test_compiled_extent_guard_does_not_consume_the_cpu_choice(tmp_path, native_unit, lower, extent):
    import ctypes as c
    import shutil
    import subprocess
    from pathlib import Path

    from compiler.tests.test_scoped_continuation_runtime import consume
    from compiler.tests.test_scoped_native_unit_handoff import add, begin_plan, bindings, choose
    from compiler.tests.test_scoped_runtime import TOKEN, Scope, access

    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    original = tmp_path / "original.f90"
    original.write_text(source(lower=lower, upper=lower + 2, team=False))
    sections = SourceEffects([original]).native_sections("coefficient_teams::advance")
    (coefficient,) = [r for r in sections.resources if r.resource == "argument::coefficient"]
    code = build_native_access(
        coefficient,
        "handle",
        "fort_cf",
        context="context",
        status="result",
        parameters={"argument::stage": "stage", "argument::n": "n"},
    )
    text = "\n".join(
        [
            "subroutine prepare(context,handle,n,stage,result) bind(C)",
            "use iso_c_binding",
            "use fort_scoped_memory",
            "implicit none",
            "integer(c_int64_t),value::context,handle",
            "integer(c_int),value::n,stage",
            "integer(c_int),intent(out)::result",
            *code.specification,
            "result=0",
            *code.prepare,
            "end subroutine",
        ]
    )
    checked = tmp_path / "checked.f90"
    checked.write_text(text)
    target = tmp_path / "guard.so"
    built = subprocess.run(
        [
            fortran,
            "-shared",
            "-fPIC",
            "-std=f2018",
            "-fcheck=all",
            str(Path(__file__).resolve().parents[1] / "runtime/scoped_memory.f90"),
            str(checked),
            native_unit._name,
            "-o",
            str(target),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    guard = c.CDLL(str(target))
    guard.prepare.argtypes = [TOKEN, TOKEN, c.c_int, c.c_int, c.POINTER(c.c_int)]
    scope = Scope(native_unit)
    values = (c.c_double * extent)(*range(extent))
    status, handle = scope.register(values, [extent], lower=[lower])
    scope.check(status)
    effect = bindings((handle, access(read=[([1], [2])])))
    begin_plan(scope)
    add(scope, 17, effect)
    choose(scope)
    result = c.c_int(-1)
    guard.prepare(scope.handle, handle, 1, lower + 1, c.byref(result))
    assert result.value == (0 if extent == 3 else 5)
    # Preparation, including rejected descriptor mapping, never consumes or
    # acquires the selected numerical worker. Its original exact token remains.
    consume(scope, 17, effect, 0)
    assert not scope.stats().uploads
    assert not scope.stats().allocations
    scope.close()


def tiled_source(*, multiple=False):
    body = """!$omp parallel private(i,base)
!$omp do
do base=1,n,tile
do i=base,min(base+tile-1,n)
b(i)=2.0_8*a(i)
enddo
enddo
!$omp end do
"""
    if multiple:
        body += """!$omp do
do i=1,n-1
b(i)=b(i)+1.0_8
enddo
!$omp end do
"""
    return (
        """module original_tiles
contains
subroutine advance(a,b,n,tile)
real(8),intent(in)::a(-2:)
real(8),intent(inout)::b(-2:)
integer,intent(in)::n,tile
integer::i,base
"""
        + body
        + "!$omp end parallel\nend subroutine\nend module\n"
    )


def tiled_unit(directory, *, multiple=False):
    from compiler.scopes.closed_selection import candidates
    from compiler.scopes.combined_team import CombinedNumericalTeam

    original = directory / "tiles.f90"
    original.write_text(tiled_source(multiple=multiple))
    builder = ScopeBuilder(
        [original],
        "original_tiles::advance",
        facts={
            "schema_version": 1,
            "participation": "serial",
            "sources": {str(original): sha256(original.read_bytes()).hexdigest()},
            "captures": {"argument::" + name: dict(FACT) for name in ("a", "b")},
        },
        options=CompilerOptions(opt_level=0),
        config=OffloadConfig("auto", calibration()),
    )
    ((nodes, _prefix, _controls),) = candidates(builder)
    return CombinedNumericalTeam(builder, nodes)


def test_single_worker_reuses_prepared_physical_sections_without_faking_native_analysis(tmp_path):
    unit = tiled_unit(tmp_path)
    assert unit.cpu_native is not None, unit.cpu_native_decline
    assert not unit.cpu_native.coherence_sections.available
    assert "constant stride" in unit.cpu_native.coherence_sections.reason
    assert len(unit.prepared.offload.units) == len(unit.effect.steps) == 1
    assert unit.cpu_native_sections is unit.effect.steps[0].sections
    alternative = unit.public()["combined_numerical_team"]["original_cpu_alternative"]
    assert alternative["physical_section_authority"] == "registered prepared single numerical worker; no aggregation"
    assert alternative["numerical_effect_identity"] == unit.effect.identity
    unit._validate()


def test_single_worker_section_replacement_loses_issued_authority(tmp_path):
    from dataclasses import replace

    from compiler.ir import CompilationError

    unit = tiled_unit(tmp_path)
    unit.cpu_native_sections = replace(unit.cpu_native_sections, resources=())
    with pytest.raises(CompilationError, match="single-worker physical section authority"):
        unit._validate()


def test_multiple_workers_do_not_gain_an_aggregate_from_numerical_section_union(tmp_path):
    unit = tiled_unit(tmp_path, multiple=True)
    assert len(unit.prepared.offload.units) > 1
    assert unit.cpu_native is None
    assert "constant stride" in unit.cpu_native_decline
