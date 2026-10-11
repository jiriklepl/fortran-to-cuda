"""A one-shot window publishes before allocation movement and loop backedges."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from hashlib import sha256

import pytest

from compiler.numerical_contract import require_numerical_build_contract
from compiler.tests.test_mapped_child_reached_plans import (
    CHILD_SOURCE,
    PARENT_SOURCE,
    PARTIAL_CHILD,
    emit_mapped_child_window,
)
from compiler.tests.test_module_reached_windows import MODULE_BOUNDARY_SOURCE, module_fixture
from compiler.tests.test_planned_reached_closures import PLANNED_SOURCE, emit_planned_window
from compiler.tests.test_reached_partial_definitions import PARTIAL_BOUNDARY_SOURCE, partial_fixture
from compiler.tests.test_reached_window_discovery import discovery_fixture
from compiler.tests.test_reached_window_owner import SOURCE, emit_window
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_scopes import run

DRIVER = """program verify
use renamed_stage,only:step,prefix_visits,tail_visits,prefix_total
implicit none
real(8),allocatable::a(:),b(:)
integer,parameter::sizes(3)=[0,8,4096],counts(3)=[0,1,3]
integer::shape,stages,iteration,n,i,unit
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=1,size(sizes)
 n=sizes(shape)
 do stages=1,size(counts)
  allocate(a(-7:n+2),b(-7:n+2))
  do iteration=1,2
   do i=lbound(a,1),ubound(a,1)
    a(i)=real(i+16,8)*0.125_8+real(iteration,8)
   enddo
   b=-77.0_8
   call step(a,b,n,counts(stages))
   write(unit)prefix_visits,tail_visits,prefix_total,lbound(a),ubound(a),a,b
  enddo
  deallocate(a,b)
 enddo
enddo
close(unit)
end program
"""

PLANNED_DRIVER = """program verify
use renamed_stage,only:step,prefix_visits,tail_visits,prefix_total,bad_visits,good_visits,condition_visits
implicit none
real(8),allocatable::a(:),b(:)
integer,parameter::sizes(3)=[0,8,4096],counts(3)=[0,1,3]
integer::shape,stages,iteration,n,i,unit,path
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do path=0,1
 do shape=1,size(sizes)
  n=sizes(shape)
  do stages=1,size(counts)
   allocate(a(-7:n+2),b(-7:n+2))
   do iteration=1,2
    do i=lbound(a,1),ubound(a,1)
     a(i)=real(i+16,8)*0.125_8+real(iteration,8)
    enddo
    b=-77.0_8
    call step(a,b,n,counts(stages),path==1)
    write(unit)prefix_visits,tail_visits,prefix_total,bad_visits,good_visits,condition_visits
    write(unit)lbound(a),ubound(a),lbound(b),ubound(b),a,b
   enddo
   deallocate(a,b)
  enddo
 enddo
enddo
close(unit)
end program
"""

MAPPED_DRIVER = """program verify
use outer_fields,only:advance,prefix_visits,tail_visits
use inner_fields,only:child_visits,condition_visits,bad_visits,good_visits,workspace
implicit none
real(8),allocatable::a(:),b(:)
integer,parameter::sizes(3)=[0,8,4096],lowers(3)=[-13,-7,4],counts(3)=[0,1,3]
integer::shape,stages,iteration,n,i,unit,path
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do path=0,1
 do shape=1,size(sizes)
  n=sizes(shape)
  do stages=1,size(counts)
   allocate(a(lowers(shape):lowers(shape)+n-1),b(lowers(shape):lowers(shape)+n-1))
   do iteration=1,2
    do i=lbound(a,1),ubound(a,1)
     a(i)=real(i+16,8)*0.125_8+real(iteration,8)
    enddo
    b=-77.0_8
    call advance(a,b,n,counts(stages),path==1)
    write(unit)prefix_visits,tail_visits,child_visits,condition_visits,bad_visits,good_visits
    write(unit)lbound(a),ubound(a),lbound(b),ubound(b),a,b
    write(unit)allocated(workspace)
    if(allocated(workspace)) write(unit)lbound(workspace),ubound(workspace),workspace
   enddo
   deallocate(a,b)
  enddo
 enddo
enddo
close(unit)
end program
"""


@pytest.fixture(scope="module", params=(
    *((name, fallback, 8) for name, fallback in (
        ("explicit", False), ("explicit", True), ("discovered", False), ("discovered", True),
        ("module_discovered", False), ("module_discovered", True), ("module_native_boundary", False),
        ("explicit_live_iterator", False))),
    *((name, fallback, precision) for precision in (4, 8) for name, fallback in (
        ("module_partial", False), ("module_partial", True), ("module_partial_boundary", False))),
    *(("planned", fallback, precision) for precision in (4, 8) for fallback in (False, True)),
    *(("mapped", fallback, precision) for precision in (4, 8) for fallback in (False, True)),
    *(("mapped_partial", False, precision) for precision in (4, 8)),
    *(("mapped_joined", False, precision) for precision in (4, 8)),
    *(pytest.param(("planned_single", fallback, precision),
                   id=f"planned_single-real{precision}-{'resource' if fallback else 'normal'}")
      for precision in (4, 8) for fallback in (False, True)),
))
def window_binaries(tmp_path_factory, request):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA, C++ and Fortran toolchain required")
    directory = tmp_path_factory.mktemp("one_shot_window")
    architecture = actual_profile(directory, nvcc, host)["hardware"]["compute_capability"].replace(".", "")
    emission, resource_fallback, precision = request.param
    if emission == "mapped_partial":
        parent_source = PARENT_SOURCE.replace(
            "do i=lbound(a,1),ubound(a,1)\nb(i)=a(i)+2.0_8\nenddo",
            "b(lbound(b,1)+1:lbound(b,1)+4)=a(lbound(a,1)+1:lbound(a,1)+4)+2.0_8")
        # INTENT(OUT) made the other elements undefined. Restore them in the
        # excluded native tail before any complete-array consumer or backedge.
        parent_source = parent_source.replace("call move_alloc(a,temporary)",
            "a(lbound(a,1))=-123.0_8\na(lbound(a,1)+5:ubound(a,1))=-124.0_8\ncall move_alloc(a,temporary)")
        declarations = CHILD_SOURCE.split("contains", 1)[0].removeprefix("module inner_fields\n")
        child_source = PARTIAL_CHILD.replace("module inner_fields\n", "module inner_fields\n" + declarations).replace(
            "y(-2:1)=", "child_visits=child_visits+1\ncondition_visits=condition_visits+1\ny(-2:1)=")
        original, outputs, report, _owner = emit_mapped_child_window(
            directory, precision=precision, parent_source=parent_source, child_source=child_source)
        assert len(report["scopes"][0]["gpu_leaves"]) == 3
        assert not report["scopes"][0]["reached_window"]["planned_closures"]["closed_path_possible"]
    elif emission in {"mapped", "mapped_joined"}:
        # Keep the native fixture valid for empty allocations; all changes to
        # workspace storage still execute in their original child owner.
        child_source = CHILD_SOURCE.replace("x(-3)=x(-3)+10.0_8", "if(size(x)>0) then\nx(-3)=x(-3)+10.0_8\nendif").replace(
            "x(-3)=x(-3)+3.0_8", "if(n>0) then\nx(-3)=x(-3)+3.0_8\nendif")
        if emission == "mapped_joined":
            child_source = child_source.replace("do i=-3,n-4\nx(i)=x(i)+real(i,8)\nenddo",
                "!$omp parallel private(i)\n!$omp do\ndo i=-3,n-4\nx(i)=x(i)+real(i,8)\nenddo\n"
                "!$omp end do\n!$omp end parallel")
        original, outputs, report, _owner = emit_mapped_child_window(
            directory, precision=precision, device_budget=0 if resource_fallback else None,
            child_source=child_source)
        assert len(report["scopes"][0]["reached_window"]["planned_closures"]["mapped_child_plans"]) == 1
        assert len(report["scopes"][0]["gpu_leaves"]) == 4
    elif emission in {"planned", "planned_single"}:
        source = PLANNED_SOURCE
        if emission == "planned_single":
            source = source.replace("if(bad) then\ncall replace_storage(b)\nelse\ncall face_update(b)\nendif",
                                    "if(bad) call replace_storage(b)")
        if precision == 4:
            source = source.replace("real(8)", "real(4)").replace("_8", "_4").replace(",8)", ",4)")
        original, outputs, report, _owner = emit_planned_window(
            directory, source, device_budget=0 if resource_fallback else None)
        assert report["scopes"][0]["reached_window"]["planned_closures"]["closed_path_possible"]
        if emission == "planned_single":
            point, = report["scopes"][0]["reached_window"]["planned_closures"]["close_points"]
            assert point["conditional_action"]["action_source_kind"] == "Call_Stmt"
            assert len(report["scopes"][0]["gpu_leaves"]) == 2
    elif emission in {"module_partial", "module_partial_boundary"}:
        options = {"source": PARTIAL_BOUNDARY_SOURCE} if emission == "module_partial_boundary" else {}
        original, builder = partial_fixture(directory, device_budget=0 if resource_fallback else None,
                                            precision=precision, **options)
        outputs, report = builder.run()
        assert report["reached_window_discovery"]["selected"]
    elif emission in {"module_discovered", "module_native_boundary"}:
        options = {"source": MODULE_BOUNDARY_SOURCE} if emission == "module_native_boundary" else {}
        original, builder = module_fixture(directory, device_budget=0 if resource_fallback else None, **options)
        outputs, report = builder.run()
        assert report["reached_window_discovery"]["selected"]
        assert builder.analysis.stable_module_allocatables == frozenset()
    elif emission == "discovered":
        original, builder = discovery_fixture(directory, device_budget=0 if resource_fallback else None)
        outputs, report = builder.run()
        assert report["reached_window_discovery"]["selected"]
    elif emission == "explicit_live_iterator":
        source = SOURCE.replace("tail_visits=tail_visits+1", "tail_visits=tail_visits+i")
        original, outputs, report, _owner = emit_window(directory, source)
        assert len(report["scopes"][0]["gpu_leaves"]) == 1
    else:
        original, outputs, report, _owner = emit_window(directory, device_budget=0 if resource_fallback else None)
    for name, text in outputs.items():
        artifact = directory / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(text)
    objects, compile_commands, build_commands = [], [], []
    flags = [fortran, "-O3", "-cpp", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps"]
    originals = original if isinstance(original, tuple) else (original,)
    original_order = {report["sources"][str(path)]["replacement"]: index
                      for index, path in enumerate(originals)}
    for role in ("common_runtime", "shared_entry", "original_source"):
        sources = [item for item in report["build_sources"] if item["role"] == role]
        if role == "original_source":
            sources.sort(key=lambda item: original_order[item["path"]])
        for item in sources:
            artifact = directory / item["path"]
            target = artifact.with_suffix(".o")
            if item["language"] == "cuda":
                contract = item["numerical_contract"]
                require_numerical_build_contract(contract)
                compiler = [nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_" + architecture,
                            "-Xcompiler=-fopenmp", *contract["required_cuda_options"],
                            *("-Xcompiler=" + option for option in contract["required_host_options"])]
            else:
                compiler = flags
            command = [*compiler, "-I", str(directory), "-c", str(artifact), "-o", str(target)]
            run(command, cwd=directory)
            build_commands.append({"role": role, "argv": command})
            compile_commands.append({"path": item["path"], "language": item["language"], "argv": command,
                                     "numerical_contract": item.get("numerical_contract")})
            objects.append(str(target))
    driver = directory / "driver.f90"
    driver_text = MAPPED_DRIVER if emission.startswith("mapped") else (
        PLANNED_DRIVER if emission in {"planned", "planned_single"} else DRIVER)
    if emission == "mapped_partial":
        driver_text = driver_text.replace("sizes(3)=[0,8,4096]", "sizes(3)=[8,16,4096]")
    if emission.startswith("module_"):
        driver_text = driver_text.replace("tail_visits,prefix_total", "tail_visits,prefix_total,a,b").replace(
            "real(8),allocatable::a(:),b(:)\n", "").replace("call step(a,b,n,counts(stages))",
                                                           "call step(n,counts(stages))")
    if precision == 4:
        driver_text = driver_text.replace("real(8)", "real(4)").replace("_8", "_4").replace(",8)", ",4)")
    driver.write_text(driver_text)
    candidate, native = directory / "candidate", directory / "native"
    cuda_root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    cuda_lib = os.path.join(cuda_root, "lib64")
    command = [*flags, str(driver), *objects, "-L" + cuda_lib, "-Wl,-rpath," + cuda_lib,
               "-lcudart", "-lstdc++", "-o", str(candidate)]
    run(command, cwd=directory)
    build_commands.append({"role": "candidate-link", "argv": command})
    command = [*flags, *(str(path) for path in originals), str(driver), "-o", str(native)]
    run(command, cwd=directory)
    build_commands.append({"role": "native-compile-link", "argv": command})
    native_env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"}
    run([str(native)], cwd=directory, env=native_env)
    build_commands.append({"role": "native-reference", "argv": [str(native)],
                           "environment": {"OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"}})
    expected = (directory / "fields.bin").read_bytes()
    (directory / "native-fields.bin").write_bytes(expected)
    (directory / "compile-commands.json").write_text(json.dumps(compile_commands, indent=2) + "\n")
    (directory / "build-commands.json").write_text(json.dumps(build_commands, indent=2) + "\n")
    (directory / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return directory, candidate, expected, report, resource_fallback, emission, precision


def _single_if_context_transfers(trace, report, precision, *, native_only):
    """Bind every event to a live owner and price each reached path exactly."""
    bindings = report["scopes"][0]["resource_bindings"]["resources"]
    identities = {item["resource"]: item["registration_identity"] for item in bindings}
    a, b = identities["argument::a"], identities["argument::b"]
    records, closed = {}, set()
    for line in trace.splitlines():
        if not line.startswith(tuple("FORT_SCOPED " + event + " " for event in
                                     ("create", "launch", "upload", "download", "close"))):
            continue
        parts = line.split()
        event = parts[1]
        fields = dict(part.split("=", 1) for part in parts[2:])
        assert fields["provenance_version"] == "1"
        context = fields["context"]
        if event == "create":
            assert context not in records
            records[context] = {"launches": 0, "upload": Counter(), "download": Counter()}
            continue
        assert context in records
        assert context not in closed
        if event == "close":
            closed.add(context)
        elif event == "launch":
            records[context]["launches"] += 1
        else:
            assert fields["generation"] == "1"
            assert "buffer_handle" in fields
            identity, size = int(fields["buffer"]), int(fields["bytes"])
            assert identity in {a, b}
            assert size > 0
            records[context][event][identity] += size
    assert set(records) == closed
    observed = Counter((item["launches"], tuple(sorted(item["upload"].items())),
                        tuple(sorted(item["download"].items()))) for item in records.values()
                       if item["launches"] or item["upload"] or item["download"])
    expected = Counter()
    if not native_only:
        # Each path/size has two nonzero stage counts and two calls. The window
        # is spent after its first traversal; later stages stay original native.
        for n in (8, 4096):
            size = n * precision
            incoming = ((a, size),)
            expected[(2, incoming, tuple(sorted(((a, size), (b, size)))))] = 4
            expected[(1, incoming, ((b, size),))] = 4
    assert observed == expected
    return {context: {"launches": item["launches"], "upload": dict(item["upload"]),
                      "download": dict(item["download"])} for context, item in records.items()}


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
def test_one_shot_window_matches_complete_native_arrays_and_all_original_visits(window_binaries, disabled):
    directory, candidate, expected, report, resource_fallback, emission, precision = window_binaries
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    result = run([str(candidate)], cwd=directory, env=env)
    label = "window-disabled" if disabled else "window-enabled"
    (directory / (label + ".stderr")).write_text(result.stderr)
    actual = (directory / "fields.bin").read_bytes()
    assert actual == expected
    assert "array temporary" not in result.stderr.lower()
    # Two kernels run only on the first traversal of each nonempty window.
    # Extra original stages still execute once, after mandatory publication and
    # MOVE_ALLOC. Zero-trip enclosing loops never activate the window at all.
    launches = result.stderr.count("FORT_SCOPED launch ")
    # Planned closures execute both original paths: two kernels on the good
    # path, one before publishing/replacing storage on the bad path. The native
    # suffix must never register or launch, including subsequent backedges.
    numerical_launches = {"mapped_partial": 72, "mapped": 48, "mapped_joined": 48, "planned": 24, "planned_single": 24,
                          "explicit_live_iterator": 8}.get(emission, 16)
    assert launches == (0 if disabled or resource_fallback else numerical_launches)
    contexts = result.stderr.count("FORT_SCOPED create ")
    # At most one context may be opened at each first reached traversal. A
    # failed pre-execution resource guard cannot retry on later backedges.
    assert contexts <= (24 if emission in {"planned", "planned_single"} or emission.startswith("mapped") else 12)
    context_launches = Counter()
    for line in result.stderr.splitlines():
        if line.startswith("FORT_SCOPED launch "):
            context = next(part.removeprefix("context=") for part in line.split() if part.startswith("context="))
            context_launches[context] += 1
    if emission in {"mapped", "mapped_joined"} and not disabled and not resource_fallback:
        # The good path's caller producer, two child kernels and caller consumer
        # must use one owner, rather than four individually published contexts.
        # The bad path stops after the caller/child producers, then continues
        # native once. Both paths run eight nonempty first traversals.
        assert Counter(context_launches.values()) == {4: 8, 2: 8}
    if emission == "mapped_partial" and not disabled:
        assert Counter(context_launches.values()) == {3: 24}
    context_transfers = None
    if emission == "planned_single":
        if not disabled and not resource_fallback:
            assert Counter(context_launches.values()) == {2: 8, 1: 8}
        context_transfers = _single_if_context_transfers(result.stderr, report, precision,
                                                       native_only=disabled or resource_fallback)
    receipt = {"schema_version": 1, "disabled": disabled, "fields_bytes": len(actual),
               "native_fields_sha256": sha256(expected).hexdigest(),
               "candidate_fields_sha256": sha256(actual).hexdigest(),
               "launches": launches, "contexts_created": contexts, "resource_fallback": resource_fallback,
               "launches_per_context": dict(context_launches),
               "complete_arrays_and_halos_equal": True,
               "original_prefix_and_tail_counters_equal": True,
               "conditional_native_replacement_and_branch_counters_equal": emission in {
                   "planned", "planned_single", "mapped", "mapped_joined"},
               "single_line_conditional_context_transfers": context_transfers,
               "mapped_original_caller_bounds_and_child_index_as_data_equal": emission in {"mapped", "mapped_joined"},
               "mapped_partial_output_and_restored_native_halo_equal": emission == "mapped_partial",
               "mapped_original_joined_openmp_and_native_entry_equal": emission == "mapped_joined",
               "reached_window": report["scopes"][0]["reached_window"],
               "compiler_owned_discovery": report["scopes"][0]["reached_window"]["automatic_discovery"],
               "emission": emission,
               "precision_bytes": precision,
               "runtime_trace_sha256": sha256(result.stderr.encode()).hexdigest(),
               "performance_sample": False}
    (directory / (label + ".json")).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
