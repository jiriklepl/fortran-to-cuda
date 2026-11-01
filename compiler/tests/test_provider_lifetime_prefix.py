"""Native prefixes share the explicit provider domain before external registration."""
import json
import os
import shutil
import subprocess

import pytest

from compiler.tests import test_provider_only_lifetime as source_fixture
from compiler.tests.test_provider_only_lifetime_runtime import DRIVER, MAIN_CONTROLS, PROVIDER

# A real source SYSTEM_CLOCK unit precedes every external CALL. The saved scalar
# counter observes native-once independently of the provider's own call count.
PREFIX = source_fixture.SOURCE.replace("implicit none\ncontains", "implicit none\ninteger,save::prefix_calls=0\ncontains", 1)
PREFIX = PREFIX.replace("integer::i\n", "integer::i,clock_tick\n", 1)
PREFIX = PREFIX.replace("call original_transform(cache,b,a)",
    "prefix_calls=prefix_calls+1\ncall system_clock(count=clock_tick)\ncall original_transform(cache,b,a)")


def test_native_prefix_uses_parent_creation_generation_latch_and_query_retirement(tmp_path, monkeypatch):
    monkeypatch.setattr(source_fixture, "SOURCE", PREFIX)
    paths, builder = source_fixture.fixture(tmp_path)
    outputs, report = builder.run()
    leaf = outputs[report["sources"][str(paths[0])]["replacement"]]
    assert leaf.index("fort_provider_borrow(fort_provider_life,fort_context") < leaf.index("call system_clock(count=clock_tick)")
    assert "fort_scope_resident_create_v1(0_c_int, fort_context)" not in leaf
    assert " .and. fort_provider_life%enabled" in leaf
    assert "fort_scope_plan_abandon_remaining_v1" in leaf
    assert "unretired query schedule at source boundary" in leaf
    assert "provider-only publication has a foreign domain" in outputs[report["provider_only_lifetime"]["helper"]["path"]]
    assert PREFIX.count("prefix_calls=prefix_calls+1") == leaf.count("prefix_calls=prefix_calls+1") == 1


@pytest.fixture(scope="module", params=[False, True], ids=["same_storage", "changed_storage"])
def prefix_binary(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp("provider_native_prefix")
    previous = source_fixture.SOURCE
    source_fixture.SOURCE = PREFIX
    try:
        build, objects, _ = source_fixture.compile_parent(directory, replace_storage=request.param)
    finally:
        source_fixture.SOURCE = previous
    cpp = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    provider, driver = build / "prefix_provider.cpp", build / "prefix_driver.f90"
    provider.write_text(PROVIDER)
    driver.write_text(DRIVER.replace("use native_parent", "use native_parent\nuse external_fixture, only: prefix_calls")
        .replace("status=provider_check(1)", "if(prefix_calls/=2)error stop 'native prefix repeated or omitted'\nstatus=provider_check(1)")
        .replace("status=provider_check(2)", "if(prefix_calls/=3)error stop 'public native prefix repeated or omitted'\nstatus=provider_check(2)"))
    commands = [
        [cpp, "-std=c++17", "-fopenmp", "-pthread", "-DFORT_SCOPE_CPU_TEST", "-x", "c++", "-I", str(build),
         "-c", str(build / "scoped_runtime.cu"), "-o", str(build / "runtime.o")],
        [cpp, "-std=c++17", "-fopenmp", "-pthread", "-I", str(build), "-c", str(provider), "-o", str(build / "provider.o")],
        [fortran, "-std=f2018", "-fopenmp", "-fcheck=all", "-I", str(build), str(driver),
         *map(str, objects), str(build / "runtime.o"), str(build / "provider.o"), "-lstdc++", "-pthread", "-o", str(build / "prefix")]]
    for index, command in enumerate(commands):
        result = subprocess.run(command, cwd=build, capture_output=True, text=True, timeout=90)
        (build / f"prefix-{index}.command.json").write_text(json.dumps(command))
        (build / f"prefix-{index}.stdout").write_text(result.stdout)
        (build / f"prefix-{index}.stderr").write_text(result.stderr)
        assert result.returncode == 0, result.stderr
    return build


@pytest.mark.native
@pytest.mark.parametrize("mode", ["enabled", "preaccept_reject", "accepted_fatal"])
def test_native_prefix_then_external_keeps_exact_plan_fields_and_failure_disposition(prefix_binary, mode):
    build = prefix_binary
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FORT_", "ELMM_", "CUDA_"))}
    if mode == "preaccept_reject":
        env["FORT_PROVIDER_REJECT"] = "1"
    elif mode == "accepted_fatal":
        env["FORT_PROVIDER_FATAL"] = "1"
    result = subprocess.run([str(build / "prefix")], cwd=build, env=env, capture_output=True, text=True, timeout=15)
    (build / f"prefix-{mode}.stdout").write_text(result.stdout)
    (build / f"prefix-{mode}.stderr").write_text(result.stderr)
    (build / f"prefix-{mode}.completion.json").write_text(json.dumps({"returncode": result.returncode}))
    if mode == "accepted_fatal":
        assert result.returncode != 0
        assert "external device work failed; replay prohibited" in result.stderr
        assert "PROVIDER_ONLY_FIELDS_OK" not in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "PROVIDER_ONLY_FIELDS_OK" in result.stdout


# A genuine original array actual reaches the opaque-state constructor before
# the clock/external task. This constructor remains an ordinary native
# boundary; it grants no array coverage. The coefficient stays exactly three.
METADATA_PREFIX = PREFIX.replace(
    "integer,save::prefix_calls=0", "integer,save::prefix_calls=0\ninteger,target,save::prefix_metadata(3)=[1,1,1]"
).replace(
    "function make_state() result(value)",
    "function make_state(seed) result(value)\ninteger,intent(in)::seed(:)"
).replace(
    "value%scratch=3.0_8", "value%scratch=real(sum(seed),8)"
).replace("cache=make_state()", "cache=make_state(prefix_metadata)")


def test_original_metadata_constructor_stays_native_before_borrowed_clock(tmp_path, monkeypatch):
    monkeypatch.setattr(source_fixture, "SOURCE", METADATA_PREFIX)
    paths, builder = source_fixture.fixture(tmp_path)
    outputs, report = builder.run()
    leaf = outputs[report["sources"][str(paths[0])]["replacement"]]
    before_constructor = leaf[:leaf.index("cache=make_state(prefix_metadata)")]
    before_clock = leaf[:leaf.index("call system_clock(count=clock_tick)")]
    assert "call fort_provider_publish(" in before_constructor
    assert "call fort_provider_retire(fort_provider_life)" not in before_constructor
    assert "fort_provider_borrow(fort_provider_life,fort_context" in before_clock
    assert "fort_provider_life%generation" in leaf
    assert "fort_scope_resident_create_v1(0_c_int, fort_context)" not in leaf
    assert leaf.count("cache=make_state(prefix_metadata)") == 1


@pytest.fixture(scope="module", params=[False, True], ids=["clock", "metadata_then_clock"])
def prefix_main_binary(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp("provider_program_prefix")
    previous_source, previous_program = source_fixture.SOURCE, source_fixture.PROGRAM
    source_fixture.SOURCE = METADATA_PREFIX if request.param else PREFIX
    source_fixture.PROGRAM = previous_program.replace(
        "use native_parent", "use native_parent\nuse external_fixture, only: prefix_calls"
    ).replace(
        "if(main_provider_check(stage)/=0)",
        "if(prefix_calls/=2*stage)error stop 'PROGRAM native prefix repeated or omitted'\n"
        "if(main_provider_check(stage)/=0)"
    )
    try:
        build, objects, report = source_fixture.compile_parent(directory, program=True)
    finally:
        source_fixture.SOURCE, source_fixture.PROGRAM = previous_source, previous_program
    cpp = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    assert cpp, "maintained native-prefix fixture requires its recorded C++ compiler"
    assert fortran, "maintained native-prefix fixture requires its recorded Fortran compiler"
    code = PROVIDER.replace('#include <new>', '#include <new>\n#include <cstdio>')
    code = code.replace(
        "static int creates=0,executes=0,destroys=0;",
        "static int creates=0,executes=0,destroys=0;\n"
        "static Provider *main_active=nullptr;\nstatic void verify_main_retirement();"
    ).replace(
        " ++creates; *handle=nullptr;",
        " if(creates==0)std::atexit(verify_main_retirement);\n ++creates; *handle=nullptr;"
    ).replace(
        "*handle=p;return FORT_SCOPE_OK;", "*handle=p;main_active=p;return FORT_SCOPE_OK;"
    ).replace("++destroys;delete p;", "++destroys;main_active=nullptr;delete p;")
    provider = build / "prefix_main_provider.cpp"
    provider.write_text(code + MAIN_CONTROLS)
    commands = [
        [cpp, "-std=c++17", "-fopenmp", "-pthread", "-DFORT_SCOPE_CPU_TEST", "-x", "c++", "-I", str(build),
         "-c", str(build / "scoped_runtime.cu"), "-o", str(build / "runtime.o")],
        [cpp, "-std=c++17", "-fopenmp", "-pthread", "-I", str(build), "-c", str(provider),
         "-o", str(build / "provider.o")],
        [fortran, "-std=f2018", "-fopenmp", "-fcheck=all", "-I", str(build),
         *map(str, objects), str(build / "runtime.o"), str(build / "provider.o"),
         "-lstdc++", "-pthread", "-o", str(build / "prefix_main")]
    ]
    for index, command in enumerate(commands):
        result = subprocess.run(command, cwd=build, capture_output=True, text=True, timeout=90)
        (build / f"prefix-main-{index}.command.json").write_text(json.dumps(command))
        (build / f"prefix-main-{index}.stdout").write_text(result.stdout)
        (build / f"prefix-main-{index}.stderr").write_text(result.stderr)
        assert result.returncode == 0, result.stderr
    return build, report


@pytest.mark.native
@pytest.mark.parametrize("mode", ["enabled", "preaccept_reject", "accepted_fatal"])
def test_program_native_prefix_preserves_plan_native_once_and_sticky_rejection(prefix_main_binary, mode):
    build, _ = prefix_main_binary
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FORT_", "ELMM_", "CUDA_"))}
    env["FORT_RUNTIME_TRACE"] = "1"
    if mode == "preaccept_reject":
        env["FORT_PROVIDER_REJECT"] = "1"
    elif mode == "accepted_fatal":
        env["FORT_PROVIDER_FATAL"] = "1"
    result = subprocess.run([str(build / "prefix_main")], cwd=build, env=env,
                            capture_output=True, text=True, timeout=15)
    (build / f"prefix-main-{mode}.stdout").write_text(result.stdout)
    (build / f"prefix-main-{mode}.stderr").write_text(result.stderr)
    (build / f"prefix-main-{mode}.completion.json").write_text(json.dumps({"returncode": result.returncode}))
    if mode == "accepted_fatal":
        assert result.returncode != 0
        assert "external device work failed; replay prohibited" in result.stderr
        assert "MAIN_PROVIDER_FIELDS_OK" not in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "MAIN_PROVIDER_FIELDS_OK" in result.stdout
        assert "MAIN_PLAN_RETIRED" in result.stdout
        # The PROGRAM checkpoint independently checks native prefix count,
        # one retained plan, zero field allocations and fresh generations 1..4.
        assert result.stderr.count('"event":"domain_create"') == 1
        assert result.stderr.count('"event":"domain_retire"') == 1
        assert result.stderr.count('"event":"domain_attach"') == (1 if mode == "preaccept_reject" else 4)
        assert result.stderr.count('"event":"domain_detach"') == (1 if mode == "preaccept_reject" else 4)


@pytest.mark.native
def test_stale_context_publication_rejects_before_any_domain_or_field_use(prefix_binary):
    build = prefix_binary
    helper, = (build / "provider_lifetimes").glob("*.f90")
    module = helper.stem
    source = build / "stale_publication.f90"
    source.write_text(f"""program stale_publication
use iso_c_binding
use fort_scoped_memory
use {module}
implicit none
type(fort_provider_lifetime)::life
integer(c_int64_t)::context,fields(0)
integer(c_int)::status
call fort_provider_reset(life)
status=fort_provider_borrow(life,context,1048576_c_size_t)
if(status/=FORT_SCOPE_OK.or.context==0)error stop 'fixture borrow failed'
status=fort_scope_close(context)
if(status/=FORT_SCOPE_OK)error stop 'fixture detach failed'
! The borrowed context is now genuinely stale; its owner domain still exists.
call fort_provider_publish(life,context,fields)
error stop 'stale context accepted'
end program
""")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    objects = [p for p in build.glob("source-*.o") if p.name != "source-0.o"]
    command = [fortran, "-std=f2018", "-fopenmp", "-fcheck=all", "-I", str(build), str(source),
               *map(str, objects), str(build / "runtime.o"), str(build / "provider.o"),
               "-lstdc++", "-pthread", "-o", str(build / "stale_publication")]
    result = subprocess.run(command, cwd=build, capture_output=True, text=True, timeout=30)
    (build / "stale_publication.command.json").write_text(json.dumps(command))
    (build / "stale_publication.compile.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FORT_", "ELMM_", "CUDA_"))}
    result = subprocess.run([str(build / "stale_publication")], cwd=build, env=env, capture_output=True, text=True, timeout=15)
    (build / "stale_publication.stdout").write_text(result.stdout)
    (build / "stale_publication.stderr").write_text(result.stderr)
    assert result.returncode != 0
    assert "provider-only publication has a foreign domain" in result.stderr
    assert "stale context accepted" not in result.stderr
