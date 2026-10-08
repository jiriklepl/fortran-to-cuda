"""Shared numerical queries expose checked inputs without executing source work."""

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission.common.resources import read_scoped_runtime
from compiler.emission.cuda.scoped import generate_scoped
from compiler.frontend import lower_file
from compiler.offload.config import OffloadConfig
from compiler.offload.profile import SCOPED_COST_NAMES
from compiler.tests.test_offload_profile import profile
from compiler.tests.test_structured_offload import SOURCE


def emit(tmp_path, source=SOURCE, calibration=None):
    path = tmp_path / "source.f90"
    path.write_text(source)
    function, plan = prepare_function(lower_file(path, "advance"), options=CompilerOptions(gpu_policy="auto"))
    _, runtime = read_scoped_runtime()
    return generate_scoped(function, plan, OffloadConfig("auto", calibration, 4), "common_functions.cuh",
                           runtime_id=runtime["runtime_id"])


def query_text(emitted):
    return emitted.cuda.split('extern "C" int ' + emitted.report["planning"]["entry"], 1)[1].split(
        'extern "C" int ' + emitted.report["planning"]["selector"], 1)[0]


def calibration():
    value = profile()
    value["toolchain"]["nvcc_version"] = "Cuda compilation tools, release 13.4, V13.4.92"
    value["toolchain"]["host_cxx_version"] = "g++ (Debian 14.4.0) 14.4.0"
    value["scoped"] = {"schema_version": 1, "runtime_id": read_scoped_runtime()[1]["runtime_id"],
                       "max_allocation_bytes": 256 * 1024 * 1024,
                       "costs": {name: 1e-6 for name in SCOPED_COST_NAMES}}
    return value


def test_query_publishes_control_inputs_and_does_not_execute_numerical_setup(tmp_path):
    emitted = emit(tmp_path)
    planning = emitted.report["planning"]
    assert planning["available"]
    assert not planning["source_effects"]
    assert planning["scalar_inputs"] == ["n"]
    assert planning["argument_order"] == ["context", "a", "b", "configuration", "n"]
    assert planning["payload_arrays"] == ["configuration"]
    assert planning["layout_arrays"] == ["a", "b", "configuration"]
    text = query_text(emitted)
    assert "query_load(" in text
    assert "configuration, {" in text
    assert "coefficient =" not in text
    assert "fort_scope_forget_definition(" not in text
    assert ".begin(" not in text
    assert "cuda" not in text.lower()
    assert "FORT_SCOPE_PLAN_WORKER" in text
    assert "fort_scope_plan_host_current(" in text
    assert text.index("fort_scope_plan_host_current(") < text.index("query_load(")
    ids = [unit["id"] for unit in planning["units"]]
    assert len(ids) == len(set(ids)) == 2
    assert ids == [unit["id"] for unit in emit(tmp_path).report["planning"]["units"]]
    for identifier in ids:
        assert f"fort_access.decision({identifier}ULL" in emitted.cuda


def test_query_records_host_metadata_effects_at_host_conditions_and_guarded_bounds(tmp_path):
    source = SOURCE.replace("if(limit>0)", "if(configuration(1)>0)").replace(
        "do i=1,n", "do i=1,min(n,configuration(2))")
    emitted = emit(tmp_path, source)
    assert emitted.report["planning"]["available"]
    query = query_text(emitted)
    # Two original HostBlocks and two array-reading conditions each execute
    # one AccessBatch host operation. Bound prefetches are separate operations
    # inside their checked, per-axis active guards.
    assert query.count("record(FORT_SCOPE_PLAN_NATIVE") == 4
    assert "fort_host.add(" in query
    assert "fort_metadata.add(" in query
    assert "fort_host.begin(" not in query
    helpers = emitted.cuda.split("static offload::Data plan_unit_0", 1)[1].split(
        'extern "C" int ' + emitted.report["planning"]["entry"], 1)[0]
    assert helpers.count("record(FORT_SCOPE_PLAN_NATIVE") == 2
    assert helpers.index("if (active) {") < helpers.index("record(FORT_SCOPE_PLAN_NATIVE")
    assert "fort_query_status = fort_metadata.record" in helpers


def test_source_out_is_a_planned_event_not_a_definition_change(tmp_path):
    source = '''module source_case
contains
subroutine advance(a,n)
real(8),intent(out)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
 a(i)=real(i,8)
enddo
end subroutine
end module
'''
    emitted = emit(tmp_path, source)
    text = query_text(emitted)
    assert "FORT_SCOPE_PLAN_FORGET" in text
    assert "fort_definition{}" in text
    assert "fort_scope_forget_definition(" not in text
    assert text.index("FORT_SCOPE_PLAN_FORGET") < text.index("FORT_SCOPE_PLAN_WORKER")
    assert "d.valid && unit.units[0].iterations" in text


@pytest.mark.parametrize(("replacement", "reason"), [
    (("limit=configuration(1)", "limit=int(scale)"), "numerical scalar setup"),
    (("a(i+1)=b(i+1)*coefficient+real(i,8)", "if(b(i+1)>0) a(i+1)=b(i+1)"), "unknown or conditional"),
])
def test_unavailable_query_explains_native_choice_without_executing_source(tmp_path, replacement, reason):
    emitted = emit(tmp_path, SOURCE.replace(*replacement))
    assert not emitted.report["planning"]["available"]
    assert reason in emitted.report["planning"]["reason"]
    assert "FORT_SCOPE_BOUNDARY" in query_text(emitted)
    assert "FORT_SCOPE_PLAN_WORKER" not in query_text(emitted)


def test_missing_or_old_runtime_calibration_cannot_select_gpu(tmp_path):
    emitted = emit(tmp_path)
    assert not emitted.report["automatic_estimate_available"]
    assert "missing" in emitted.report["automatic_reason"]
    value = calibration()
    value["scoped"]["runtime_id"] = "0" * 64
    emitted = emit(tmp_path, calibration=value)
    assert not emitted.report["automatic_estimate_available"]
    assert "runtime_id mismatch" in emitted.report["automatic_reason"]
    assert "costs.valid = 1" not in emitted.cuda


def test_selector_previews_complete_plan_before_hardware_compatibility(tmp_path):
    emitted = emit(tmp_path, calibration=calibration())
    assert emitted.report["automatic_estimate_available"]
    text = emitted.cuda.split('extern "C" int ' + emitted.report["planning"]["selector"], 1)[1]
    assert text.index("&costs, -1, &preview") < text.index("offload::compatible")
    assert "!preview.available || (scoped_host_compatible(profile) && (!preview.gpu_units ||" in text
    assert text.index("fort_scope_device_get") < text.index("&costs, -1, &preview")
    assert "cudaGetDevice(&fort_current_device) == cudaSuccess" in text
    assert text.index("fort_current_device == fort_scope_device") < text.index("offload::compatible")
    assert "&costs, compatible ? 1 : 0, decision" in text
    for name in SCOPED_COST_NAMES:
        assert f"costs.{name} =" in text
    assert "use fort_scoped_memory, only: fort_scope_plan_decision" in emitted.fortran
    assert "public :: run, plan, choose" in emitted.fortran
    host_check = emitted.cuda.split("static bool scoped_host_compatible", 1)[1].split("}", 1)[0]
    assert "cudaGet" not in host_check
    assert "cudaRuntime" not in host_check
    assert "cpu_name == profile.cpu_name" in emitted.cuda
    assert "host == profile.host_compiler && cuda == profile.cuda_compiler" in emitted.cuda


BEHAVIOR_SOURCE = '''module planned_numerical_case
contains
subroutine advance(a,configuration,n,flag,scale)
real(8),intent(inout)::a(:)
integer,intent(in)::configuration(:),n
logical,intent(in)::flag
real(8),intent(in)::scale
integer::i,limit
real(8)::coefficient
if(flag) then
 limit=configuration(1)
 coefficient=scale*2.0_8
 do i=1,min(n,configuration(1))
  a(i)=a(i)*coefficient
 enddo
endif
do i=1,n
 a(i)=a(i)+1
enddo
end subroutine
end module
'''

GUARDED_BOUNDS_SOURCE = '''module planned_bounds_case
contains
subroutine advance(a,n,m)
real(8),intent(inout)::a(:,:)
integer,intent(in)::n,m
integer::i,j
do j=1,n
 do i=1,m
  a(i,j)=a(i,j)+1
 enddo
enddo
end subroutine
end module
'''

OUT_SOURCE = '''module planned_out_case
contains
subroutine advance(a,n)
real(8),intent(out)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
 a(i)=real(i,8)
enddo
end subroutine
end module
'''


def _command(command, directory):
    import subprocess
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def planning_library(tmp_path_factory):
    """Compile real emitted CUDA; use the deterministic CPU runtime backend."""
    import ctypes as c
    import shutil

    from compiler.emission.common.resources import read_common_header
    from compiler.tests.test_scoped_runtime import TOKEN, Access, Layout, Stats

    nvcc, host, fortran = shutil.which("nvcc"), shutil.which("g++-14"), shutil.which("gfortran-15")
    if not nvcc or not host or not fortran:
        pytest.skip("CUDA/Fortran toolchain required for shared query ABI")
    directory = tmp_path_factory.mktemp("numerical_planning")
    sources, _ = read_scoped_runtime()
    for name, text in sources.items():
        (directory / name).write_text(text)
    (directory / "common_functions.cuh").write_text(read_common_header())
    _command([fortran, "-std=f2018", "-c", "fort_scoped_memory.f90", "-o", "memory.o"], directory)
    objects = []
    reports = {}
    for name, source in {"guarded": BEHAVIOR_SOURCE, "bounds": GUARDED_BOUNDS_SOURCE,
                         "out": OUT_SOURCE, "overflow": OUT_SOURCE.replace("i=1,n", "i=1,n+1"),
                         "incompatible": OUT_SOURCE}.items():
        subdirectory = directory / name
        subdirectory.mkdir()
        emitted = emit(subdirectory, source, calibration=calibration() if name == "incompatible" else None)
        reports[name] = emitted.report
        assert emitted.report["planning"]["available"]
        (directory / f"{name}.cu").write_text(emitted.cuda)
        (directory / f"{name}.f90").write_text(emitted.fortran)
        _command([fortran, "-std=f2018", "-c", f"{name}.f90", "-o", f"{name}-interface.o"], directory)
        _command([nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fPIC,-fopenmp",
                  "-c", f"{name}.cu", "-o", f"{name}.o"], directory)
        objects.append(f"{name}.o")
    # Device storage here is a host allocation. The shared coherence/decision
    # API is unchanged, so stale-metadata checks need no actual GPU workload.
    _command([host, "-O2", "-std=c++17", "-fPIC", "-fopenmp", "-DFORT_SCOPE_CPU_TEST", "-x", "c++",
              "-c", "scoped_runtime.cu", "-o", "runtime.o"], directory)
    library = directory / "queries.so"
    _command([nvcc, "-ccbin", host, "-Xcompiler=-fopenmp", "-shared", *objects, "runtime.o", "-o", str(library)],
             directory)
    lib = c.CDLL(str(library))
    lib.fort_scope_error.restype = c.c_char_p
    for name, signature in {
        "create": [c.c_int, c.POINTER(TOKEN)], "register": [TOKEN, TOKEN, TOKEN, c.POINTER(Layout), c.c_int, c.POINTER(TOKEN)],
        "plan_reset": [TOKEN], "wait": [TOKEN], "close": [TOKEN], "stats_get": [TOKEN, c.POINTER(Stats)],
        "host_begin": [TOKEN, TOKEN, c.POINTER(Access)], "host_end": [TOKEN, TOKEN],
        "device_begin": [TOKEN, TOKEN, c.POINTER(Access), c.POINTER(c.c_void_p)], "device_end": [TOKEN, TOKEN],
    }.items():
        getattr(lib, "fort_scope_" + name).argtypes = signature
    for report in reports.values():
        planning = report["planning"]
        scalar_types = {"integer": c.c_int, "logical": c.c_bool, "real": c.c_double}
        parameters = [TOKEN, *[TOKEN for _ in report["array_parameters"]],
                      *[c.POINTER(scalar_types[s["dtype"]]) for s in report["scalar_parameters"]
                        if s["name"] in planning["scalar_inputs"]]]
        getattr(lib, planning["entry"]).argtypes = parameters
        getattr(lib, report["entry"]).argtypes = [TOKEN, c.c_int, *[TOKEN for _ in report["array_parameters"]],
            *[c.POINTER(scalar_types[s["dtype"]]) for s in report["scalar_parameters"]]]
        getattr(lib, planning["selector"]).argtypes = [TOKEN, c.c_void_p]
    return lib, reports, directory


def _integer_registration(scope, values, identity=2):
    import ctypes as c

    from compiler.tests.test_scoped_runtime import SIZE, TOKEN, Layout
    dimensions = (SIZE * 1)(len(values))
    lower = (c.c_int64 * 1)(-2)
    layout = Layout(1, 3, 4, c.cast(values, c.c_void_p), dimensions, lower, 1)
    scope.references.extend([values, dimensions, lower, layout])
    buffer = TOKEN()
    scope.check(scope.lib.fort_scope_register(scope.handle, identity, 1, c.byref(layout), 1, c.byref(buffer)))
    return buffer


def _decision_storage():
    import ctypes as c
    # Public planning decision is 4*u32, 9*u64, 2*double.
    return (c.c_uint64 * 13)()


@pytest.mark.native
def test_query_only_records_out_without_mutating_definition_or_values(planning_library):
    import ctypes as c

    from compiler.tests.test_scoped_runtime import Scope, access
    lib, reports, _ = planning_library
    scope = Scope(lib)
    values = (c.c_double * 8)(*range(8))
    status, buffer = scope.register(values, [8])
    scope.check(status)
    n = c.c_int(8)
    before = bytes(scope.stats())
    scope.check(lib.fort_scope_plan_reset(scope.handle))
    scope.check(getattr(lib, reports["out"]["planning"]["entry"])(scope.handle, buffer, c.byref(n)))
    assert list(values) == list(range(8))
    assert bytes(scope.stats()) == before
    scope.cpu_begin(buffer, access(flags=1))
    scope.cpu_end(buffer)
    scope.close()


@pytest.mark.native
def test_query_rejects_stale_integer_metadata_without_reading_or_transferring(planning_library):
    import ctypes as c

    from compiler.tests.test_scoped_runtime import Scope, access
    lib, reports, _ = planning_library
    scope = Scope(lib)
    values = (c.c_double * 8)(*range(8))
    status, buffer = scope.register(values, [8])
    scope.check(status)
    metadata = (c.c_int * 2)(8, 0)
    bounds = _integer_registration(scope, metadata)
    pointer = c.c_void_p()
    effects = access(flags=6)
    scope.check(lib.fort_scope_device_begin(scope.handle, bounds, c.byref(effects), c.byref(pointer)))
    c.cast(pointer, c.POINTER(c.c_int))[0] = 4
    scope.check(lib.fort_scope_device_end(scope.handle, bounds))
    scope.check(lib.fort_scope_wait(scope.handle))
    before = bytes(scope.stats())
    n, flag = c.c_int(8), c.c_bool(True)
    scope.check(lib.fort_scope_plan_reset(scope.handle))
    result = getattr(lib, reports["guarded"]["planning"]["entry"])(scope.handle, buffer, bounds,
              c.byref(n), c.byref(flag))
    assert result == 5, lib.fort_scope_error().decode()
    assert metadata[0] == 8
    assert list(values) == list(range(8))
    assert bytes(scope.stats()) == before
    scope.close()


@pytest.mark.native
def test_invalid_checked_query_does_not_execute_out_or_change_values(planning_library):
    import ctypes as c

    from compiler.tests.test_scoped_runtime import Scope, access
    lib, reports, _ = planning_library
    scope = Scope(lib)
    values = (c.c_double * 8)(*range(8))
    status, buffer = scope.register(values, [8])
    scope.check(status)
    n = c.c_int(2**31 - 1)
    before = bytes(scope.stats())
    scope.check(lib.fort_scope_plan_reset(scope.handle))
    result = getattr(lib, reports["overflow"]["planning"]["entry"])(scope.handle, buffer, c.byref(n))
    assert result == 5, lib.fort_scope_error().decode()
    assert list(values) == list(range(8))
    assert bytes(scope.stats()) == before
    scope.cpu_begin(buffer, access(flags=1))
    scope.cpu_end(buffer)
    scope.close()


@pytest.mark.native
def test_missing_calibration_selector_and_scheduled_entry_continue_native(planning_library, monkeypatch, capfd):
    import ctypes as c

    from compiler.tests.test_scoped_runtime import Scope
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    lib, reports, _ = planning_library
    scope = Scope(lib)
    values = (c.c_double * 8)(*range(8))
    status, buffer = scope.register(values, [8])
    scope.check(status)
    metadata = (c.c_int * 2)(8, 0)
    bounds = _integer_registration(scope, metadata)
    n, flag, scale = c.c_int(8), c.c_bool(False), c.c_double(2)
    arguments = [scope.handle, buffer, bounds, c.byref(n), c.byref(flag), c.byref(scale)]
    scope.check(lib.fort_scope_plan_reset(scope.handle))
    public = reports["guarded"]
    scope.check(getattr(lib, public["planning"]["entry"])(*arguments[:-1]))
    decision = _decision_storage()
    scope.check(getattr(lib, public["planning"]["selector"])(scope.handle, c.byref(decision)))
    assert "reason=missing_or_incompatible_calibration" in capfd.readouterr().err
    # A missing profile must not require CUDA initialization or invalidate the
    # numerical API's successful automatic native mode.
    scope.check(getattr(lib, public["entry"])(scope.handle, 2, *arguments[1:]))
    scope.close()
    assert list(values) == [i+1 for i in range(8)]


@pytest.mark.native
def test_native_preview_rejects_mismatched_host_calibration_without_cuda(planning_library):
    import ctypes as c

    from compiler.tests.test_scoped_runtime import Scope
    lib, reports, _ = planning_library
    scope = Scope(lib)
    values = (c.c_double * 8)(*range(8))
    status, buffer = scope.register(values, [8])
    scope.check(status)
    n = c.c_int(8)
    scope.check(lib.fort_scope_plan_reset(scope.handle))
    public = reports["incompatible"]
    assert public["automatic_estimate_available"]
    scope.check(getattr(lib, public["planning"]["entry"])(scope.handle, buffer, c.byref(n)))
    decision = _decision_storage()
    scope.check(getattr(lib, public["planning"]["selector"])(scope.handle, c.byref(decision)))
    fields = c.cast(decision, c.POINTER(c.c_uint32))
    assert fields[0] == 0  # available: Test CPU/profile does not match this host.
    assert fields[1] == 0  # gpu_units: a valid native preview needs no CUDA probe.
    assert scope.stats().launches == 0
    assert scope.stats().allocations == 0
    scope.close()


@pytest.mark.native
def test_native_worker_and_query_use_same_direct_integer_array_bound(planning_library):
    import ctypes as c

    from compiler.tests.test_scoped_runtime import Scope
    lib, reports, _ = planning_library
    scope = Scope(lib)
    values = (c.c_double * 8)(*range(8))
    status, buffer = scope.register(values, [8])
    scope.check(status)
    metadata = (c.c_int * 2)(4, 0)
    bounds = _integer_registration(scope, metadata)
    n, flag, scale = c.c_int(8), c.c_bool(True), c.c_double(2)
    scope.check(lib.fort_scope_plan_reset(scope.handle))
    public = reports["guarded"]
    scope.check(getattr(lib, public["planning"]["entry"])(scope.handle, buffer, bounds, c.byref(n), c.byref(flag)))
    scope.check(getattr(lib, public["entry"])(scope.handle, 0, buffer, bounds, c.byref(n), c.byref(flag), c.byref(scale)))
    scope.close()
    assert list(values) == [i*4+1 if i<4 else i+1 for i in range(8)]


@pytest.mark.native
def test_query_preserves_false_if_and_empty_outer_bounds_under_optimization(planning_library):
    import os
    import subprocess
    import sys
    _, reports, directory = planning_library
    # A subprocess makes any speculative protected load an explicit failure.
    source = f'''import ctypes as c, mmap
from compiler.tests.test_scoped_runtime import Scope, SIZE, TOKEN, Access, Layout, Stats
lib=c.CDLL({str(directory / "queries.so")!r})
lib.fort_scope_error.restype=c.c_char_p
lib.fort_scope_create.argtypes=[c.c_int,c.POINTER(TOKEN)]
lib.fort_scope_register.argtypes=[TOKEN,TOKEN,TOKEN,c.POINTER(Layout),c.c_int,c.POINTER(TOKEN)]
lib.fort_scope_close.argtypes=[TOKEN]
lib.fort_scope_stats_get.argtypes=[TOKEN,c.POINTER(Stats)]
lib.fort_scope_plan_reset.argtypes=[TOKEN]
libc=c.CDLL(None)
libc.mmap.argtypes=[c.c_void_p,c.c_size_t,c.c_int,c.c_int,c.c_int,c.c_long]
libc.mmap.restype=c.c_void_p
pointer=libc.mmap(None,4096,0,0x22,-1,0)
assert pointer not in (None,c.c_void_p(-1).value)
scope=Scope(lib)
values=(c.c_double*8)(*range(8))
status,buffer=scope.register(values,[8]); scope.check(status)
metadata=(c.c_int*2)(8,0); shape=(SIZE*1)(2); lower=(c.c_int64*1)(0)
layout=Layout(1,3,4,c.cast(metadata,c.c_void_p),shape,lower,1); bounds=TOKEN()
scope.check(lib.fort_scope_register(scope.handle,2,1,c.byref(layout),1,c.byref(bounds)))
query=getattr(lib,{reports["guarded"]["planning"]["entry"]!r})
query.argtypes=[TOKEN,TOKEN,TOKEN,c.POINTER(c.c_int),c.POINTER(c.c_bool)]
n=c.c_int(8); flag=c.c_bool(False)
scope.check(lib.fort_scope_plan_reset(scope.handle))
scope.check(query(scope.handle,buffer,bounds,c.byref(n),c.byref(flag)))
run=getattr(lib,{reports["guarded"]["entry"]!r})
run.argtypes=[TOKEN,c.c_int,TOKEN,TOKEN,c.POINTER(c.c_int),c.POINTER(c.c_bool),c.POINTER(c.c_double)]
scope.check(run(scope.handle,2,buffer,bounds,c.byref(n),c.byref(flag),c.cast(pointer,c.POINTER(c.c_double))))
scope.close()
assert list(values)==[i+1 for i in range(8)]
for i in range(8): values[i]=i
scope=Scope(lib)
status,buffer=scope.register(values,[4,2]); scope.check(status)
query=getattr(lib,{reports["bounds"]["planning"]["entry"]!r})
query.argtypes=[TOKEN,TOKEN,c.POINTER(c.c_int),c.POINTER(c.c_int)]
n=c.c_int(0)
scope.check(lib.fort_scope_plan_reset(scope.handle))
scope.check(query(scope.handle,buffer,c.byref(n),c.cast(pointer,c.POINTER(c.c_int))))
assert list(values)==list(range(8))
assert scope.stats().launches==0
scope.close()
'''
    script = directory / "protected-query.py"
    script.write_text(source)
    root = str(__import__("pathlib").Path(__file__).resolve().parents[2])
    result = subprocess.run([sys.executable, str(script)], cwd=directory,
                            env={**os.environ, "PYTHONPATH": root}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
