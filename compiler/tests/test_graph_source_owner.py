"""Whole-caller graph ownership beyond source-window and outline limits."""

import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder

FACT = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
SOURCE = """module graph_fixture
implicit none
contains
subroutine advance(a,b,x,n)
real(8),intent(inout)::a(:),b(:)
integer,intent(inout)::x(:)
integer,intent(in)::n
integer::i
do i=1,n
b(i)=2.0_8*a(i)
enddo
do i=2,n
x(i)=x(i-1)+1
enddo
do i=1,n
a(i)=b(i)+1.0_8
enddo
end subroutine
end module
"""

PRIVATE_SCRATCH_SOURCE = SOURCE.replace(
    "integer::i", "integer::i\ninteger,parameter::mask(4)=[-1,9,9,-1]\nreal(8)::scratch(n)"
).replace("do i=2,n\nx(i)=x(i-1)+1\nenddo", """!$omp parallel private(i,scratch)
!$omp do
do i=1,n
scratch(i)=b(i)+mask(1)
a(i)=scratch(i)+b(i)
enddo
!$omp end do
!$omp end parallel""")

NATIVE_HELPER_SOURCE = SOURCE.replace(
    "implicit none", "implicit none\ninteger,allocatable::types(:),other_types(:)"
).replace("contains\n", """contains
subroutine smooth(field,kinds,n)
real(8),intent(inout)::field(:)
integer,intent(in)::kinds(:),n
integer,parameter::weights(2)=[1,1]
real(8)::pencil(n)
integer::i,j
!$omp parallel do private(i,j,pencil) schedule(runtime)
do j=1,1
pencil=field
do i=2,n-1
if(kinds(i)==0) field(i)=0.5_8*(weights(1)*pencil(i-1)+weights(2)*pencil(i+1))
enddo
enddo
!$omp end parallel do
end subroutine
""").replace("do i=2,n\nx(i)=x(i-1)+1\nenddo", "call smooth(a,types,n)\ncall smooth(b,other_types,n)")

EPOCH_SOURCE = SOURCE.replace("do i=1,n\nb(i)", "call opaque(a,n)\ndo i=1,n\nb(i)", 1).replace(
    "do i=2,n\nx(i)=x(i-1)+1\nenddo", "call opaque(b,n)")
EPOCH_SOURCE = EPOCH_SOURCE.replace("integer::i", """integer::i
interface
subroutine opaque(v,n)
real(8),intent(inout)::v(*)
integer,intent(in)::n
end subroutine
end interface""")


def emit(directory, source=SOURCE, *, policy="sections", captures=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "graph.f90"
    path.write_text(source)
    facts = {
        "schema_version": 1,
        "participation": "serial",
        "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
        "captures": {"argument::" + name: dict(FACT) for name in ("a", "b", "x")},
    }
    facts["captures"].update(captures or {})
    builder = ScopeBuilder(
        [path],
        "graph_fixture::advance",
        facts=facts,
        options=CompilerOptions(),
        config=OffloadConfig(policy=policy, scope_execution="graph"),
    )
    outputs, report = builder.run()
    return path, outputs, report


def walk_graph(graph):
    for task in graph["tasks"]:
        yield task
        for child in task.get("branch", ()):
            yield from walk_graph(child)
        if "loop" in task:
            yield from walk_graph(task["loop"])


def walk_schedule(schedule):
    for placement in schedule["placements"]:
        yield placement
        for child in placement["children"]:
            yield from walk_schedule(child)


def test_graph_owner_keeps_device_state_across_known_native_only_recurrence(tmp_path, monkeypatch):
    from compiler.scopes import interval_draft, window_discovery

    def forbidden(*_args, **_kwargs):
        pytest.fail("graph execution used window enumeration or eager interval closure collection")

    monkeypatch.setattr(window_discovery, "discover_reached_window", forbidden)
    monkeypatch.setattr(interval_draft, "prepare_interval_draft", forbidden)
    path, outputs, report = emit(tmp_path)
    (owner,) = report["scopes"]
    assert not owner["boundaries"]
    assert len(owner["planning_segments"]) == 3
    assert len(owner["gpu_leaves"]) == 2
    graph_record = owner["graph_execution"]
    assert graph_record["source_windows"] is False
    (procedure,) = graph_record["procedures"]
    tasks = list(walk_graph(procedure["task_graph"]))
    assert [task["implementations"] for task in tasks] == [["cpu", "gpu"], ["cpu"], ["cpu", "gpu"]]
    placements = list(walk_schedule(procedure["schedule"]))
    assert [placement["target"] for placement in placements] == ["cpu", "gpu", "gpu"]
    assert placements[0]["transfers"] == []
    assert placements[2]["transfers"] == []
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert "fort_configure" in text  # Direct copies still enable graph planning/virtual definitions.
    assert text.count("x(i)=x(i-1)+1") == 1
    assert text.index("fort_reached_1: block") < text.index("fort_reached_0: block")
    assert text.index("fort_reached_0: block") < text.index("fort_reached_2: block")
    (motion,) = graph_record["source_motion"]
    assert motion["scheduled_tasks"] == [
        "graph_fixture::advance:unit:1",
        "graph_fixture::advance:unit:0",
        "graph_fixture::advance:unit:2",
    ]
    assert graph_record["source_order_executor"] is False
    assert len({segment["segment_id"] for segment in owner["planning_segments"]}) == 3
    assert owner["ownership"]["lifetime"] == "one original invocation"
    assert owner["reopen_after_boundary"] is True


def test_original_private_scratch_and_parameter_arrays_need_no_managed_capture(tmp_path):
    path, outputs, report = emit(tmp_path, PRIVATE_SCRATCH_SOURCE)
    owner, = report["scopes"]
    assert not owner["boundaries"]
    assert len(owner["planning_segments"]) == 3
    native = [item for scope in owner["planning_segments"] for item in scope["operations"]["native_operations"]]
    # The original team stays native because its per-thread array has an
    # automatic bound. Neither the scratch array nor its immutable coefficient
    # array becomes a managed allocation or acquires caller capture facts.
    team = next(item for item in native if item["source_constants"])
    assert set(team["source_constants"]) == {"graph_fixture::advance::mask"}
    assert "graph_fixture::advance::scratch" in team["private_resources"]
    assert set(team["managed_resources"]) == {"argument::a", "argument::b"}
    assert all(not proof["device_capture"] for proof in team["source_constants"].values())
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert text.count("scratch(i)=b(i)+mask(1)") == 1
    assert "!$omp parallel private(i,scratch)" in text


def native_helper_source(nested):
    if not nested:
        return NATIVE_HELPER_SOURCE
    return NATIVE_HELPER_SOURCE.replace("subroutine advance", """subroutine wrapped(field,kinds,n)
real(8),intent(inout)::field(:)
integer,intent(in)::kinds(:),n
call smooth(field,kinds,n)
end subroutine
subroutine advance""").replace("call smooth(a,types,n)", "call wrapped(a,types,n)").replace(
        "call smooth(b,other_types,n)", "call wrapped(b,other_types,n)")


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested-forwarding"])
def test_native_helper_borrows_each_readonly_module_actual_without_gpu_capture_facts(tmp_path, nested):
    path, outputs, report = emit(tmp_path, native_helper_source(nested))
    owner, = report["scopes"]
    assert not owner["boundaries"]
    helper = next(item for item in owner["module_coordinators"] if item["procedure"].endswith("::smooth"))
    assert not helper["boundaries"]
    caller = next(item for item in owner["module_coordinators"] if item["procedure"].endswith(
        "::wrapped" if nested else "::smooth"))
    assert not caller["boundaries"]
    assert len(caller["calls"]) == 2
    first, second = caller["calls"]
    assert first["handle_bindings"] != second["handle_bindings"]
    operations = [operation for unit in helper["planning_segments"]
                  for operation in unit["operations"]["native_operations"]]
    assert any(operation["host_only_native_reads"] for operation in operations)
    assert any(operation["source_constants"] for operation in operations)
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert "shape(types, kind=c_size_t)" in text
    assert "shape(other_types, kind=c_size_t)" in text
    assert text.count("pencil=field") == 2  # Active native branch and original fallback retain the same snapshot.
    assert "FORT_SCOPE_BYTES" in text


def test_native_helper_reservation_cannot_become_mutable_managed_storage(tmp_path):
    source = NATIVE_HELPER_SOURCE.replace("call smooth(b,other_types,n)", "call smooth(b,other_types,n)\ntypes(1)=7")
    _path, _outputs, report = emit(tmp_path, source)
    owner, = report["scopes"]
    boundary, = owner["boundaries"]
    assert "read-only complete effects" in boundary["reason"] or "managed write" in boundary["reason"]
    assert boundary["reopen"] is True


def test_graph_owner_removes_aggregate_32_region_and_256_operation_limits(tmp_path):
    prefix = SOURCE[: SOURCE.index("do i=1,n")]
    suffix = "end subroutine\nend module\n"
    # Each candidate is small; the entire chain exceeds both historical caps.
    unit = "do i=1,n\nb(i)=a(i)+1\na(i)=b(i)+1\nb(i)=a(i)+1\na(i)=b(i)+1\nb(i)=a(i)+1\na(i)=b(i)+1\nenddo\n"
    _path, _outputs, report = emit(tmp_path, prefix + unit * 40 + suffix)
    (owner,) = report["scopes"]
    assert len(owner["planning_segments"]) == 40
    assert len(owner["graph_execution"]["procedures"][0]["admitted_unit_tasks"]) == 40
    assert report["inline_numerical_regions"]["limits"] == {
        "regions": None,
        "cumulative_operations": None,
        "operations_per_candidate": 256,
    }
    assert not any("budget exhausted" in boundary["reason"] for boundary in report["boundaries"])
    placements = list(walk_schedule(owner["graph_execution"]["procedures"][0]["schedule"]))
    assert len(placements) == 40
    assert all(not placement["transfers"] for placement in placements[1:])


def test_branches_keep_native_guards_and_untaken_units_in_their_child_graphs(tmp_path):
    source = SOURCE.replace(
        "do i=2,n\nx(i)=x(i-1)+1\nenddo", "if(n>2) then\ndo i=2,n\nx(i)=x(i-1)+1\nenddo\nelse\nx(1)=7\nendif"
    )
    path, outputs, report = emit(tmp_path, source)
    (owner,) = report["scopes"]
    (procedure,) = owner["graph_execution"]["procedures"]
    (branch,) = [task for task in procedure["task_graph"]["tasks"] if "branch" in task]
    assert len(branch["branch"]) == 2
    assert any(task["may_writes"] == ["argument::x"] for task in walk_graph(branch["branch"][0]))
    assert any(task["may_writes"] == ["argument::x"] for task in walk_graph(branch["branch"][1]))
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert text.count("if(n>2) then") == 1
    assert text.count("x(1)=7") == 1
    assert text.index("if(n>2) then") < text.index("x(1)=7") < text.index("endif", text.index("x(1)=7"))


def test_unknown_reached_call_closes_epoch_and_reopens_without_entry_definition_facts(tmp_path):
    source = SOURCE.replace("do i=2,n\nx(i)=x(i-1)+1\nenddo", "call unknown(x)")
    path, outputs, report = emit(tmp_path, source)
    (owner,) = report["scopes"]
    assert len(owner["boundaries"]) == 1
    assert owner["reopen_after_boundary"] is True
    assert len(owner["planning_segments"]) == 2
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert text.count("call unknown(x)") == 1
    before = text[: text.index("call unknown(x)")]
    assert "fort_scope_close(" in before
    assert "= .false." in before
    assert text.count("a(i)=b(i)+1.0_8") == 1
    assert "fort_scope_admit_source_reads_v1" in text[text.index("call unknown(x)"):]
    assert "fort_scope_buffer_valid_v1" in text


@pytest.mark.parametrize("guarded", [False, True], ids=["straight-line", "original-branch"])
def test_opaque_gpu_opaque_gpu_uses_reached_read_admission_in_fresh_epochs(tmp_path, guarded):
    source = EPOCH_SOURCE
    if guarded:
        source = source.replace("call opaque(a,n)", "if(n>0) then\ncall opaque(a,n)").replace(
            "end subroutine\nend module", "endif\nend subroutine\nend module")
    path, outputs, report = emit(tmp_path, source)
    owner, = report["scopes"]
    assert len(owner["boundaries"]) == 2
    assert all(boundary["reopen"] for boundary in owner["boundaries"])
    assert len(owner["gpu_leaves"]) == 2
    assert len(owner["planning_segments"]) == 2
    assert owner["graph_execution"]["epochs"]["reopened_coverage"].startswith("none;")
    text = outputs[report["sources"][str(path)]["replacement"]]
    initial = next(line.split("::", 1)[1].strip() for line in text.splitlines()
                   if line.startswith("logical :: fort_entry_epoch_"))
    assert text.count(initial + " = .true.") == 1
    assert initial + " = .false." in text[:text.index("call opaque(a,n)")]
    assert text.count("call opaque(a,n)") == text.count("call opaque(b,n)") == 1
    for unit in text.split("fort_reached_")[1:]:
        if "if (" + initial + ") then" in unit:
            assert "0_c_int, fort_buffer_" in unit
    assert text.count("fort_scope_admit_source_reads_v1") == 2
    if guarded:
        assert text.index("if(n>0) then") < text.index("call opaque(a,n)")


def test_true_native_consumer_retains_source_order_and_required_transfers(tmp_path):
    source = SOURCE.replace("do i=2,n\nx(i)=x(i-1)+1\nenddo", "do i=2,n\nb(i)=b(i-1)+a(i)\nenddo")
    path, outputs, report = emit(tmp_path, source)
    (owner,) = report["scopes"]
    assert owner["graph_execution"]["source_motion"] == []
    (procedure,) = owner["graph_execution"]["procedures"]
    placements = list(walk_schedule(procedure["schedule"]))
    assert [placement["target"] for placement in placements] == ["gpu", "cpu", "gpu"]
    assert [(copy["resource"], copy["target"]) for copy in placements[1]["transfers"]] == [("argument::b", "cpu")]
    assert [(copy["resource"], copy["target"]) for copy in placements[2]["transfers"]] == [("argument::b", "gpu")]
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert text.index("fort_reached_0: block") < text.index("fort_reached_1: block")
    assert text.index("fort_reached_1: block") < text.index("fort_reached_2: block")


def test_original_scalar_tasks_are_scheduled_with_their_actual_definition_dependencies(tmp_path):
    source = SOURCE.replace("integer::i", "integer::i,scratch").replace("do i=2,n\nx(i)=x(i-1)+1\nenddo", "scratch=7")
    path, outputs, report = emit(tmp_path / "movable", source)
    (owner,) = report["scopes"]
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert text.index("scratch=7") < text.index("fort_reached_0: block")
    assert owner["graph_execution"]["source_motion"]
    source = source.replace("integer,intent(in)::n", "integer,intent(inout)::n").replace("scratch=7", "n=n-1")
    path, outputs, report = emit(tmp_path / "dependent", source)
    (owner,) = report["scopes"]
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert text.index("fort_reached_0: block") < text.index("n=n-1") < text.index("fort_reached_1: block")
    assert owner["graph_execution"]["source_motion"] == []


def test_statement_labels_retain_native_source_before_any_graph_instrumentation(tmp_path):
    source = SOURCE.replace("do i=1,n", "10 do i=1,n", 1)
    path, outputs, report = emit(tmp_path, source)
    assert report["scopes"] == []
    assert list(outputs) == ["scope-manifest.json"]
    assert path.read_text() == source
    assert "labels remain native" in report["boundaries"][0]["reason"]


def test_counted_caller_retains_zero_trip_control_and_shares_context_with_borrowed_child(tmp_path):
    source = """module graph_fixture
implicit none
contains
subroutine leaf(a,b,n)
real(8),intent(inout)::a(:),b(:)
integer,intent(in)::n
integer::i
do i=1,n
b(i)=a(i)+1
enddo
end subroutine
subroutine advance(a,b,x,n)
real(8),intent(inout)::a(:),b(:)
integer,intent(inout)::x(:)
integer,intent(in)::n
integer::trip
do trip=1,n
call leaf(a,b,n)
enddo
end subroutine
end module
"""
    path, outputs, report = emit(tmp_path, source)
    (owner,) = report["scopes"]
    assert not owner["boundaries"]
    assert len(owner["counted_controls"]) == 1
    root, child = owner["graph_execution"]["procedures"]
    (loop,) = [task for task in root["task_graph"]["tasks"] if "loop" in task]
    assert loop["loop_control"] == {"iterator": "trip", "lower": "1", "upper": "n", "step": "1"}
    assert child["admitted_unit_tasks"]
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert text.count("do trip=1,n") == 1
    assert owner["counted_controls"][0]["reopen_after_boundary"] is False


def reusable_helper_source(nested):
    middle = ("subroutine middle(field,n)\nreal(8),intent(inout)::field(:)\ninteger,intent(in)::n\n"
              "call leaf(field,n)\nend subroutine\n") if nested else ""
    target = "middle" if nested else "leaf"
    source = """module graph_fixture
implicit none
contains
subroutine leaf(field,n)
real(8),intent(inout)::field(:)
integer,intent(in)::n
integer::i
do i=1,n
field(i)=field(i)+1
enddo
end subroutine
""" + middle + f"""subroutine advance(a,b,x,n)
real(8),intent(inout)::a(:),b(:)
integer,intent(inout)::x(:)
integer,intent(in)::n
call {target}(a,n)
call {target}(b,n)
end subroutine
end module
"""
    return source


@pytest.mark.parametrize("nested", [False, True], ids=["direct-helper", "nested-forwarding"])
def test_reusable_helper_binds_each_call_to_its_original_actual_handle(tmp_path, nested):
    path, outputs, report = emit(tmp_path, reusable_helper_source(nested))
    target = "middle" if nested else "leaf"
    owner, = report["scopes"]
    assert not owner["boundaries"]
    child = next(item for item in owner["module_coordinators"] if item["procedure"] == "graph_fixture::" + target)
    assert child["parameterized_handles"] is True
    assert len(child["calls"]) == 2
    first, second = child["calls"]
    assert set(first["handle_bindings"]) == set(second["handle_bindings"])
    assert first["handle_bindings"] != second["handle_bindings"]
    assert set(first["resource_mappings"].values()) == {"argument::a"}
    assert set(second["resource_mappings"].values()) == {"argument::b"}
    text = outputs[report["sources"][str(path)]["replacement"]]
    for slot, number in second["handle_bindings"].items():
        assert f"fort_buffer_{slot} = fort_buffer_{number}" in text
    assert text.lower().count("entry " + child["entry"]) == 1
    assert len(owner["graph_execution"]["procedures"]) == (3 if nested else 2)


def test_checked_helper_program_is_serialized_once_for_all_procedure_identities(tmp_path, monkeypatch):
    from compiler.scopes import execution_plan

    original_digest = execution_plan._digest
    program_roots = []

    def digest(value):
        if (isinstance(value, dict) and set(value) == {"type", "fields"}
                and {"routine", "units", "owner"} <= value["fields"].keys()):
            program_roots.append(value["type"])
        return original_digest(value)

    monkeypatch.setattr(execution_plan, "_digest", digest)
    _path, _outputs, report = emit(tmp_path, reusable_helper_source(True))
    owner, = report["scopes"]
    assert len(owner["graph_execution"]["procedures"]) == 3
    assert len(program_roots) == 1
    identities = [procedure["emission_identity"] for procedure in owner["checked_execution_plan"]["procedures"]]
    assert len(set(identities)) == 3


def test_exception_observer_scan_reuses_only_the_exact_immutable_source_session(tmp_path, monkeypatch):
    from copy import copy

    from compiler.frontend.source_effects import SourceEffects
    from compiler.scopes import regions

    path = tmp_path / "observer.f90"
    path.write_text(SOURCE)
    analysis = SourceEffects([path])
    analysis.seal_session()
    original_walk, scans = regions.walk, []

    def walk(*args, **kwargs):
        scans.append(args[0])
        return original_walk(*args, **kwargs)

    monkeypatch.setattr(regions, "walk", walk)
    assert regions._exception_observers(analysis) == ()
    first = len(scans)
    assert first > 0
    assert regions._exception_observers(analysis) == ()
    assert len(scans) == first
    projection = analysis.private_projection(routines=True)
    projection.routines["graph_fixture::advance"] = copy(analysis.routines["graph_fixture::advance"])
    assert regions._exception_observers(projection) == ()
    assert len(scans) > first


@pytest.mark.parametrize("reads_incoming", [False, True], ids=["normal-out", "diagnosed-body-inout"])
def test_graph_body_read_preserves_incoming_definitions_without_inventing_them(tmp_path, reads_incoming):
    source = reusable_helper_source(False).replace("intent(inout)::field", "intent(out)::field")
    if not reads_incoming:
        source = source.replace("field(i)=field(i)+1", "field(i)=1")
    path, outputs, report = emit(tmp_path, source)
    owner, = report["scopes"]
    assert not owner["boundaries"]
    child, = owner["module_coordinators"]
    diagnostics = child["body_intent_diagnostics"]
    assert diagnostics
    text = outputs[report["sources"][str(path)]["replacement"]]
    body = text[text.index("subroutine leaf("):text.index("end subroutine", text.index("subroutine leaf("))]
    assert "fort_scope_forget_definition" not in body
    assert all(item["entry_definitions_inferred"] is False for item in diagnostics)
    assert all(item["declared_intent"] == "out" for item in diagnostics)
    assert all(item["graph_body_intent"] == "body_effects" for item in diagnostics)
    assert all(item["strict_fortran_undefinedness_extension"] is True for item in diagnostics)


@pytest.mark.native
@pytest.mark.parametrize("source", [SOURCE, reusable_helper_source(False), reusable_helper_source(True), PRIVATE_SCRATCH_SOURCE,
                                    PRIVATE_SCRATCH_SOURCE.replace("parallel private(i,scratch)",
                                        "parallel default(none) shared(a,b,n) private(i,scratch)"),
                                    reusable_helper_source(False).replace("do i=1,n", "!$omp parallel do default(none) "
                                        "shared(field,n) private(i)\ndo i=1,n").replace(
                                            "enddo\nend subroutine", "enddo\n!$omp end parallel do\nend subroutine"),
                                    NATIVE_HELPER_SOURCE, native_helper_source(True), EPOCH_SOURCE],
                         ids=["mixed-source-motion", "direct-helper-mapping", "nested-helper-forwarding", "native-private-scratch",
                              "native-private-default-none", "helper-team-default-none", "native-formal-reservations",
                              "nested-native-formal-reservations", "fresh-epochs"])
def test_graph_generated_fortran_compiles_with_native_bounds_checks(tmp_path, source):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if fortran is None:
        pytest.skip("Fortran compiler unavailable")
    _path, outputs, report = emit(tmp_path / "inputs", source)
    assert report["scope_count"] == 1
    build = tmp_path / "build"
    build.mkdir()
    for name, text in outputs.items():
        artifact = build / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(text)
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, artifact in enumerate(report["build_sources"]):
            if artifact["language"] != "fortran" or artifact["role"] != role:
                continue
            command = [
                fortran,
                "-std=f2018",
                "-fopenmp",
                "-fcheck=all,array-temps",
                "-c",
                str(build / artifact["path"]),
                "-J",
                str(build),
                "-I",
                str(build),
                "-o",
                str(build / f"source-{index}.o"),
            ]
            result = subprocess.run(command, cwd=build, capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stdout + result.stderr
