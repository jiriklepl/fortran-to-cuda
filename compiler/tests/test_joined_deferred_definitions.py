"""Joined numerical alternatives validate their own reached input coverage."""

from compiler.scopes.definitions import summarize
from compiler.scopes.joined_team import JoinedTeam
from compiler.tests.test_joined_team_source import teams
from compiler.tests.test_ordinary_numerical_regions import ordinary_numerical_fixture

TILED = """!$omp do schedule(static)
do block=-3,n,tile
do i=block,min(block+tile-1,n)
b(i)=b(i)+a(i)*coefficient(0)
enddo
enddo
!$omp end do
"""
CORRECTION = """!$omp do schedule(static)
do i=-3,n
b(i)=b(i)+sum(coefficient)
enddo
!$omp end do
"""
SOURCE = """module quartz_numerics
use,intrinsic::ieee_exceptions,only:ieee_set_halting_mode,ieee_invalid
implicit none
integer::bridge_visits=0,middle_visits=0
real(8)::coefficient(-1:0)=[2.0_8,3.0_8]
contains
subroutine bridge(a,b,c,n,tile)
real(8),intent(in)::a(-4:)
real(8),intent(inout)::b(-4:),c(-4:)
integer,intent(in)::n,tile
integer::i,block
bridge_visits=bridge_visits+1
!$omp parallel
call ieee_set_halting_mode(ieee_invalid,.false.)
!$omp end parallel
!$omp parallel private(i,block)
!$omp do schedule(static)
do i=-3,n
b(i)=b(i)+a(i)*coefficient(-1)
enddo
!$omp end do
""" + TILED + CORRECTION + """!$omp do schedule(static)
do i=-3,n
c(i)=2.0_8*b(i)+real(i+1,8)
enddo
!$omp end do
!$omp end parallel
if(n>=-3) then
call touch_face(b)
endif
end subroutine
subroutine touch_face(b)
real(8),intent(inout)::b(-4:)
middle_visits=middle_visits+1
b(-3)=b(-3)+0.5_8
end subroutine
end module
"""


def partial_joined_fixture(directory, *, precision=8, device_budget=None, source=SOURCE):
    # Physical sections are half open: dummy a(-4:8), active a(-3:7).
    return ordinary_numerical_fixture(directory, precision=precision, device_budget=device_budget,
        child_source=source, captures={"argument::a": {
            "initialized": "sections", "sections": [{"lower": [1], "upper": [12]}],
            "expected_descriptor": {"lower_bounds": [-4], "extents": [13]}}})


def emit_partial_joined_fixture(directory, **options):
    paths, builder = partial_joined_fixture(directory, **options)
    outputs, report = builder.run()
    return paths, outputs, report, builder


def test_tiled_gpu_alternative_does_not_inherit_coarse_native_whole_reads(tmp_path):
    paths, outputs, report, builder = emit_partial_joined_fixture(tmp_path)
    owner, = report["scopes"]
    (team, operations), = teams(report)
    assert len(team["numerical_units"]) == 3
    assert not owner["boundaries"]
    segment, = [item for item in owner["planning_segments"] if item["operations"].get("joined_team")]
    definitions = segment["ordered_definitions"]
    assert "argument::a" not in definitions["required_whole"]
    assert "argument::a" not in definitions["whole_on_return"]
    assert "argument::c" not in definitions["whole_on_return"]
    tiled_line = SOURCE.splitlines().index("do block=-3,n,tile") + 1
    fallback, = [item for item in operations["native_operations"]
                 if item["first_line"] <= tiled_line <= item["last_line"]]
    assert not fallback["sections"]["available"]
    assert "stride" in fallback["sections"]["reason"]
    assert team["numerical_fallback_definitions"]["additional_definition_authority"] is False
    assert builder.facts["captures"]["argument::a"]["initialized"] == "sections"
    assert builder.facts["captures"]["argument::c"]["initialized"] == "none"
    text = outputs[report["sources"][str(paths[1])]["replacement"]]
    for statement in ("b(i)=b(i)+a(i)*coefficient(-1)", "b(i)=b(i)+a(i)*coefficient(0)",
                      "b(i)=b(i)+sum(coefficient)", "c(i)=2.0_8*b(i)+real(i+1,8)"):
        assert text.count(statement) == 1
    assert "FORT_SCOPE_READ_ALL" in text
    assert "fort_scope_host_begin" in text
    assert owner["reopen_after_boundary"] is False


def test_unlowered_native_tiled_read_keeps_its_conservative_entry_obligation(tmp_path):
    native = CORRECTION.replace("do i=-3,n\n", "do block=-3,n,tile\ndo i=block,min(block+tile-1,n)\n")
    native = native.replace("sum(coefficient)", "sum(a)").replace("!$omp end do", "enddo\n!$omp end do")
    _paths, _outputs, report, _builder = emit_partial_joined_fixture(
        tmp_path, source=SOURCE.replace(CORRECTION, native))
    owner, = report["scopes"]
    segment, = [item for item in owner["planning_segments"] if item["operations"].get("joined_team")]
    assert "argument::a" in segment["ordered_definitions"]["required_whole"]


def test_deferred_fallback_does_not_restore_killed_or_partial_output_definitions(tmp_path):
    _paths, builder = partial_joined_fixture(tmp_path)
    nodes = builder.inline.prepare(builder.entry.execution.content)
    group, = [item for item in builder.inline.grouped_nodes(nodes)
              if isinstance(item, tuple) and any("coefficient" in str(node).lower() for node in item)]
    scope = JoinedTeam(builder, group, native_host_reads=True)
    arrays, scalars, written = builder.owner_inputs(scope.calls, reached_definitions=True)
    scope.inputs(arrays, scalars, written)
    scope.prepare(arrays)
    proof = summarize(scope, entry_undefined=("argument::a", "argument::c"))
    assert "argument::a" not in proof.whole_on_return
    assert "argument::c" not in proof.whole_on_return
    assert {"argument::a", "argument::c"} <= proof.changes
