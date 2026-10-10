"""Native compute calibration is checked in the actual original source file."""

import shutil
import subprocess
from copy import copy, deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from fparser.common.readfortran import FortranStringReader
from fparser.two.parser import ParserFactory

from compiler.frontend.source_effects import SourceEffects
from compiler.offload.collective_calibration import normalize_fortran_options
from compiler.scopes.compute_identity import compute_identity_guard
from compiler.scopes.source import ScopeBuilder

FORTRAN = {"compiler_version": "GCC version 15.3.0", "semantic_options": "-O3\x1f-fopenmp"}


def artifact(identity=FORTRAN, *, units=1, runtime_schedule=None):
    return SimpleNamespace(scoped={"planning": {"units": [
        {"compute_model": None if identity is None else {"fortran": deepcopy(identity),
            **({"runtime_schedule": deepcopy(runtime_schedule)} if runtime_schedule is not None else {})}}
        for _ in range(units)]}})


def builder(tmp_path, *, policy="auto", caller_extra="", module_extra=None, module_use=None):
    sources = []
    for module, routine, extra in (("driver_unit", "advance", caller_extra),
                                   ("producer_unit", "produce", ""), ("leaf_unit", "leaf", "")):
        path = tmp_path / (module + ".f90")
        path.write_text(f"""module {module}
{(module_use or {}).get(module, "")}
implicit none
{(module_extra or {}).get(module, "")}
contains
subroutine {routine}(a)
real(8),intent(inout)::a(:)
{extra}
a=a+1.0_8
end subroutine
end module
""")
        sources.append(path)
    analysis = SourceEffects(sources)
    result = SimpleNamespace(analysis=analysis, config=SimpleNamespace(policy=policy),
        entry=analysis.routines["driver_unit::advance"], generated={}, edits={}, regions={})
    result.inline_for = result.regions.get

    def add_edit(path, first, last, replacement):
        result.edits.setdefault(path, []).append((first, last, replacement))

    result.add_edit = add_edit
    result.append_procedure = lambda module, name, code: ScopeBuilder.append_procedure(result, module, name, code)
    return result


def helpers(value, module):
    return [text for _, _, text in value.edits.get(value.analysis.modules[module].path, ())
            if text.startswith("logical function fort_compute_identity_")]


def test_cross_file_guard_runs_intrinsics_in_child_original_object(tmp_path):
    value = builder(tmp_path)
    value.generated["producer_unit::produce"] = artifact()
    result = compute_identity_guard(value, ["producer_unit::produce"])
    assert result.available
    assert result.required
    requirement, = result.requirements
    assert requirement.module == "producer_unit"
    assert requirement.source == str(tmp_path / "producer_unit.f90")
    assert requirement.source_sha256 == sha256((tmp_path / "producer_unit.f90").read_bytes()).hexdigest()
    assert requirement.procedures == ("producer_unit::produce",)
    assert result.imports == ("use producer_unit, only: " + requirement.helper,)
    assert result.expression == requirement.helper + "()"
    assert not helpers(value, "driver_unit")
    helper, = helpers(value, "producer_unit")
    assert "compiler_version" in helper
    assert "compiler_options" in helper
    assert "fort_scope_team_fortran_compatible_v1" in helper
    assert "GCC version 15.3.0" in helper
    assert "'-O3'" in helper
    assert "'-fopenmp'" in helper
    assert "achar(31)" in helper
    assert "pure function" not in helper
    assert result.public()["available"]
    assert result.public()["requirements"][0] == requirement.public()


def test_helper_alone_is_exported_from_original_private_module(tmp_path):
    value = builder(tmp_path, module_extra={"producer_unit": "private\npublic :: produce"})
    value.generated["producer_unit::produce"] = artifact()
    result = compute_identity_guard(value, ["producer_unit::produce"])
    assert result.available
    assert result.required
    requirement, = result.requirements
    module = value.analysis.modules["producer_unit"]
    exports = [text for _, _, text in value.edits[module.path] if text.startswith("public ::")]
    assert exports == ["public :: " + requirement.helper + "\n"]
    assert not module.default_public


@pytest.mark.parametrize("module", ["driver_unit", "producer_unit"])
def test_imported_helper_name_conflict_rejects_before_source_edits(tmp_path, module):
    source = str(tmp_path / "producer_unit.f90")
    name = "fort_compute_identity_" + sha256((source + "\0producer_unit").encode()).hexdigest()[:12]
    value = builder(tmp_path, module_use={module: "use external_metadata, only: " + name + " => other"})
    value.generated["producer_unit::produce"] = artifact()
    result = compute_identity_guard(value, ["producer_unit::produce"])
    assert not result.available
    assert "conflicts" in result.reason
    assert not value.edits


def test_synthetic_caller_cannot_receive_original_source_imports(tmp_path):
    value = builder(tmp_path)
    value.generated["producer_unit::produce"] = artifact()
    result = compute_identity_guard(value, ["producer_unit::produce"], caller=copy(value.entry))
    assert not result.available
    assert "caller lacks original" in result.reason
    assert not value.edits


def test_same_module_helpers_reuse_actual_edits_and_survive_checkpoint_rollback(tmp_path):
    value = builder(tmp_path)
    value.generated["producer_unit::produce"] = artifact(units=3)
    first = compute_identity_guard(value, ["producer_unit::produce"], caller="producer_unit::produce")
    assert not first.imports
    assert len(helpers(value, "producer_unit")) == 1
    before = deepcopy(value.edits)
    second = compute_identity_guard(value, ["producer_unit::produce"])
    assert first.requirements == second.requirements
    assert value.edits == before
    value.edits = {}  # SourceBuilder checkpoint restore removes the helper too.
    third = compute_identity_guard(value, ["producer_unit::produce"])
    assert third.available
    assert len(helpers(value, "producer_unit")) == 1


def test_runtime_schedule_helper_has_separate_identity_and_rechecks_each_call(tmp_path):
    value = builder(tmp_path)
    value.generated["producer_unit::produce"] = artifact()
    static = compute_identity_guard(value, ["producer_unit::produce"])
    before = deepcopy(value.edits)
    value.generated["producer_unit::produce"] = artifact(runtime_schedule={"kind": "static", "chunk": 0})
    runtime = compute_identity_guard(value, ["producer_unit::produce"])
    assert runtime.available and runtime.required
    assert runtime.requirements[0].helper != static.requirements[0].helper
    assert runtime.requirements[0].runtime_schedule == {"kind": "static", "chunk": 0}
    code = next(text for text in helpers(value, "producer_unit") if "call fort_get_schedule" in text)
    assert "save" not in code.lower()
    assert "fort_chunk == 0" in code
    assert "iand(fort_schedule,huge(fort_schedule)) == fort_static" in code
    assert len(helpers(value, "producer_unit")) == 2
    repeat = compute_identity_guard(value, ["producer_unit::produce"])
    assert repeat.requirements == runtime.requirements
    assert len(helpers(value, "producer_unit")) == 2
    assert all(edit in value.edits[path] for path, edits in before.items() for edit in edits)


@pytest.mark.parametrize("schedule", [{"kind": "dynamic", "chunk": 0}, {"kind": "static", "chunk": 1},
                                      {"kind": "static", "chunk": False}, {}])
def test_unsupported_runtime_schedule_contract_adds_no_original_edits(tmp_path, schedule):
    value = builder(tmp_path)
    value.generated["producer_unit::produce"] = artifact(runtime_schedule=schedule)
    guard = compute_identity_guard(value, ["producer_unit::produce"])
    assert not guard.available
    assert "runtime schedule" in guard.reason
    assert not value.edits


def test_original_runtime_guard_observes_icv_changes_without_creating_a_team(tmp_path):
    from compiler.scopes.compute_identity import _helper

    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    cpp = shutil.which("g++")
    if not fortran or not cpp:
        pytest.skip("native Fortran/C++ toolchain unavailable")
    runtime = Path(__file__).resolve().parents[1] / "runtime"

    def run(arguments):
        completed = subprocess.run(arguments, cwd=tmp_path, text=True, capture_output=True, timeout=30)
        assert completed.returncode == 0, completed.stdout + completed.stderr

    # Existing tests authenticate compiler intrinsics against a real C++
    # compatibility implementation. This sentinel isolates the additional ICV
    # condition, including changes between calls within the same native thread.
    (tmp_path / "compatibility.cpp").write_text(
        'extern "C" int fort_scope_team_fortran_compatible_v1(const char*,const char*,const char*,const char*) { return 1; }\n')
    code = _helper("check_schedule", "GCC test", "-O3\x1f-fopenmp", runtime_schedule=True)
    (tmp_path / "check.f90").write_text("module schedule_checks\ncontains\n" + code + "end module\n" + """
program check
use omp_lib
use schedule_checks
implicit none
call omp_set_schedule(omp_sched_static,0)
if (.not. check_schedule()) error stop 1
call omp_set_schedule(omp_sched_static,1)
if (check_schedule()) error stop 2
call omp_set_schedule(omp_sched_dynamic,1)
if (check_schedule()) error stop 3
call omp_set_schedule(omp_sched_guided,1)
if (check_schedule()) error stop 4
call omp_set_schedule(ior(omp_sched_static,not(huge(omp_sched_static))),0)
if (.not. check_schedule()) error stop 5
call omp_set_schedule(omp_sched_static,0)
if (.not. check_schedule()) error stop 6
if (omp_in_parallel()) error stop 7
end program
""")
    run([cpp, "-std=c++17", "-c", "compatibility.cpp", "-o", "compatibility.o"])
    run([fortran, "-fopenmp", "-c", str(runtime / "scoped_team_observer.f90"), "-o", "observer.o"])
    run([fortran, "-std=f2018", "-fopenmp", "check.f90", "observer.o", "compatibility.o", "-o", "check"])
    run([str(tmp_path / "check")])


def test_original_caller_flags_cannot_substitute_for_native_child_flags(tmp_path):
    value = builder(tmp_path)
    value.generated["driver_unit::advance"] = artifact({**FORTRAN, "semantic_options": "-O0\x1f-fopenmp"})
    value.generated["producer_unit::produce"] = artifact(FORTRAN)
    result = compute_identity_guard(value, ["driver_unit::advance", "producer_unit::produce"])
    assert result.available
    assert len(result.requirements) == 2
    caller_helper, = helpers(value, "driver_unit")
    child_helper, = helpers(value, "producer_unit")
    assert "'-O0'" in caller_helper
    assert "'-O3'" not in caller_helper
    assert "'-O3'" in child_helper
    assert "'-O0'" not in child_helper
    assert result.expression.count("()") == 2
    assert " .and. &\n" in result.expression


def test_inline_region_uses_registered_original_lexical_authority(tmp_path):
    value = builder(tmp_path)
    original = value.analysis.routines["producer_unit::produce"]
    regional = "producer_unit::produce#region7"
    value.regions[regional] = SimpleNamespace(builder=SimpleNamespace(entry=original),
        generated={regional: artifact()})
    result = compute_identity_guard(value, [regional])
    assert result.available
    requirement, = result.requirements
    assert requirement.module == "producer_unit"
    assert requirement.procedures == (regional,)
    assert len(helpers(value, "producer_unit")) == 1
    assert not helpers(value, "driver_unit")


@pytest.mark.parametrize("policy", ["sections", "auto"])
def test_forced_or_static_all_native_adds_no_helpers_or_source_edits(tmp_path, policy):
    value = builder(tmp_path, policy=policy)
    value.generated["producer_unit::produce"] = artifact(FORTRAN if policy != "auto" else None)
    value.generated["leaf_unit::leaf"] = None
    result = compute_identity_guard(value, ["producer_unit::produce", "leaf_unit::leaf"])
    assert result.available
    assert not result.required
    assert result.expression == ".true."
    assert not result.imports
    assert not result.requirements
    assert not value.edits


def test_missing_prepared_artifact_or_source_authority_is_unavailable_before_edits(tmp_path):
    value = builder(tmp_path)
    value.generated["producer_unit::produce"] = artifact()
    result = compute_identity_guard(value, ["producer_unit::produce", "missing::entry"])
    assert not result.available
    assert result.expression == ".false."
    assert not value.edits
    value.generated["missing::entry"] = artifact()
    result = compute_identity_guard(value, ["producer_unit::produce", "missing::entry"])
    assert not result.available
    assert "authority" in result.reason
    assert not value.edits


def test_changed_original_source_rejects_even_after_helper_reuse(tmp_path):
    value = builder(tmp_path)
    value.generated["producer_unit::produce"] = artifact()
    assert compute_identity_guard(value, ["producer_unit::produce"]).available
    before = deepcopy(value.edits)
    path = tmp_path / "producer_unit.f90"
    path.write_text(path.read_text().replace("a=a+1.0_8", "a=a+2.0_8"))
    result = compute_identity_guard(value, ["producer_unit::produce"])
    assert not result.available
    assert "changed" in result.reason
    assert value.edits == before


def test_conflicting_identity_in_original_object_adds_no_partial_edits(tmp_path):
    value = builder(tmp_path)
    original = value.analysis.routines["producer_unit::produce"]
    first, second = "producer_unit::produce#region1", "producer_unit::produce#region2"
    for procedure, options in ((first, "-O3\x1f-fopenmp"), (second, "-O2\x1f-fopenmp")):
        value.regions[procedure] = SimpleNamespace(builder=SimpleNamespace(entry=original),
            generated={procedure: artifact({**FORTRAN, "semantic_options": options})})
    result = compute_identity_guard(value, [first, second])
    assert not result.available
    assert "conflicting" in result.reason
    assert not value.edits


def test_baked_literals_are_bounded_and_achar_is_local_despite_caller_binding(tmp_path):
    value = builder(tmp_path, caller_extra="real(8)::achar")
    identity = {"compiler_version": "GCC 'quoted' version " * 8,
                "semantic_options": "-O3\x1f-DVALUE='name'\x1f-fopenmp"}
    value.generated["producer_unit::produce"] = artifact(identity)
    result = compute_identity_guard(value, ["producer_unit::produce"])
    assert result.available
    assert "achar" not in result.expression
    helper, = helpers(value, "producer_unit")
    assert "intrinsic :: achar" in helper
    assert "''quoted''" in helper
    assert "''name''" in helper
    assert all(len(line) <= 132 for line in helper.splitlines())
    parse = ParserFactory().create(std="f2008")
    parse(FortranStringReader("module validate_helper\ncontains\n" + helper + "end module\n"))


@pytest.mark.parametrize("identity", [
    {}, {**FORTRAN, "compiler_version": "GCC\nversion"},
    {**FORTRAN, "semantic_options": "-DVALUE=\x00bad"},
    {**FORTRAN, "semantic_options": "x" * (193 * 48)},
    {"compiler_version": "v" * (97 * 48), "semantic_options": "x" * (97 * 48)},
])
def test_unsupported_identity_is_unavailable_without_source_mutation(tmp_path, identity):
    value = builder(tmp_path)
    value.generated["producer_unit::produce"] = artifact(identity)
    result = compute_identity_guard(value, ["producer_unit::produce"])
    assert not result.available
    assert result.reason
    assert not value.edits


def test_unproved_original_ast_cannot_issue_identity_helper(tmp_path):
    value = builder(tmp_path)
    value.generated["producer_unit::produce"] = artifact()
    original = value.analysis.routines["producer_unit::produce"]
    original.execution = copy(original.execution)
    result = compute_identity_guard(value, ["producer_unit::produce"])
    assert not result.available
    assert "authority" in result.reason
    assert not value.edits


@pytest.mark.native
def test_original_child_identity_uses_real_object_flags_and_c_abi(tmp_path):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    cpp = shutil.which("g++")
    if not fortran or not cpp:
        pytest.skip("native Fortran/C++ toolchain unavailable")
    runtime = Path(__file__).resolve().parents[1] / "runtime"

    def run(arguments):
        result = subprocess.run(arguments, cwd=tmp_path, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    flags = ["-std=f2018", "-O3", "-fopenmp", "-I", str(tmp_path), "-J", str(tmp_path)]
    native_flags = ["-O0" if flag == "-O3" else flag for flag in flags]
    probe = tmp_path / "probe.f90"
    probe.write_text("""program probe
use, intrinsic :: iso_fortran_env, only: compiler_version, compiler_options
implicit none
print '(a)', compiler_version()
print '(a)', compiler_options()
end program
""")
    run([fortran, *flags, "-c", str(probe), "-o", str(tmp_path / "probe.o")])
    run([fortran, str(tmp_path / "probe.o"), "-fopenmp", "-o", str(tmp_path / "probe")])
    version, options = run([str(tmp_path / "probe")]).splitlines()
    expected = {"compiler_version": version, "semantic_options": normalize_fortran_options(options)}
    assert "-O3" in expected["semantic_options"].split("\x1f")

    value = builder(tmp_path, module_extra={"producer_unit": "private\npublic :: produce"})
    value.generated["producer_unit::produce"] = artifact(expected)
    guard = compute_identity_guard(value, ["producer_unit::produce"])
    assert guard.available
    requirement, = guard.requirements
    child = Path(requirement.source)
    lines = child.read_text().splitlines(keepends=True)
    for first, last, replacement in sorted(value.edits[child], reverse=True):
        lines[first - 1:last] = [replacement]
    child.write_text("".join(lines))
    # The outer source object deliberately has different semantics. Only the
    # child's intrinsics may authenticate the calibrated child counterfactual.
    caller = tmp_path / "driver_unit.f90"
    original = caller.read_text()
    original = original.replace("subroutine advance(a)\n", "subroutine advance(a)\n" + "\n".join(guard.imports) + "\n")
    original = original.replace("a=a+1.0_8", "if (" + guard.expression + ") then\n"
        "a=a+1.0_8\nelse\na=a-1.0_8\nend if")
    caller.write_text(original)
    main = tmp_path / "check.f90"
    main.write_text("""program check
use driver_unit, only: advance
implicit none
real(8) :: value(1)
value=0.0_8
call advance(value)
print '(i0)', int(value(1))
end program
""")
    implementation = tmp_path / "compatibility.cpp"
    implementation.write_text('#define FORT_SCOPE_TEAM_OBSERVER_IMPLEMENTATION\n#include "scoped_team_observer.hpp"\n')
    run([cpp, "-std=c++17", "-O0", "-I", str(runtime), "-c", str(implementation),
         "-o", str(tmp_path / "compatibility.o")])
    run([fortran, *native_flags, "-c", str(runtime / "scoped_team_observer.f90"),
         "-o", str(tmp_path / "observer.o")])
    child_object = tmp_path / "producer.o"
    run([fortran, *flags, "-c", str(child), "-o", str(child_object)])
    run([fortran, *native_flags, "-c", str(caller), "-o", str(tmp_path / "driver.o")])
    run([fortran, *native_flags, "-c", str(main), "-o", str(tmp_path / "check.o")])

    def link_and_check(name):
        binary = tmp_path / name
        run([fortran, str(tmp_path / "check.o"), str(tmp_path / "driver.o"), str(child_object),
             str(tmp_path / "observer.o"), str(tmp_path / "compatibility.o"), "-fopenmp", "-lstdc++", "-o", str(binary)])
        return run([str(binary)]).strip()

    assert link_and_check("matching_child") == "1"
    # Keep the original caller/main/C++ objects unchanged, then replace only
    # the child object. The helper must now reject its actual -O0 compilation.
    run([fortran, *native_flags, "-c", str(child), "-o", str(child_object)])
    assert link_and_check("mismatched_child") == "-1"
