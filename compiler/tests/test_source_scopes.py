"""Historical source owners retain native procedures and source guards."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FACT = {"storage":"stable","initialized":"whole","escapes":False,"allocation_changes":False}

PROGRAM = """module original
implicit none
contains
subroutine producer(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(out)::b(:)
integer,intent(in)::n
integer::i
do i=1,n
b(i)=2*a(i)+real(i,8)
enddo
end subroutine
subroutine transform(b)
real(8),intent(inout)::b(:)
b=3*b
end subroutine
subroutine consumer(a,b,out,n)
real(8),intent(in)::a(:),b(:)
real(8),intent(out)::out(:)
integer,intent(in)::n
integer::i
do i=1,n
out(i)=a(i)+b(i)
enddo
end subroutine
subroutine step(a,b,out,n)
real(8),intent(in)::a(:)
real(8),intent(out)::b(:),out(:)
integer,intent(in)::n
call producer(a,b,n)
call transform(b)
call consumer(a,b,out,n)
end subroutine
end module
"""

DRIVER = """program caller
use original,only:step
implicit none
real(8),allocatable::a(:),b(:),output(:)
integer::n,shape,repeat,i
do shape=1,2
 n=8+8*shape
 allocate(a(-2:n-3),b(-2:n-3),output(-2:n-3))
 do repeat=1,2
  do i=-2,n-3
   a(i)=real(i+3,8)*0.25d0+repeat
  enddo
  b=-99
  output=-99
  call step(a,b,output,n)
  do i=-2,n-3
   if (b(i)/=6*a(i)+3*real(i+3,8)) error stop 'native intermediate disagreement'
   if (output(i)/=7*a(i)+3*real(i+3,8)) error stop 'complete field disagreement'
  enddo
 enddo
 deallocate(a,b,output)
enddo
print *, 'FIELDS_OK'
end program
"""


def run(command, *, cwd, env=None, timeout=120):
    result = subprocess.run(command,cwd=cwd,env=env,capture_output=True,text=True,timeout=timeout)
    assert result.returncode == 0, result.stdout+result.stderr
    return result


def generate(directory, source=PROGRAM, *, mode="sections", entry="step", facts=None, checkout=None, profile=None,
             scope_execution="bounded"):
    directory.mkdir(parents=True,exist_ok=True)
    if isinstance(profile, dict):
        profile_path = directory / "synthetic-profile.json"
        profile_path.write_text(json.dumps(profile))
        profile = profile_path
    original = directory/"original.f90"
    original.write_text(source)
    captures = {"argument::a":FACT,
                "argument::b":{**FACT,"initialized":"none"},
                "argument::out":{**FACT,"initialized":"none"}}
    facts_file = directory/"captures.json"
    document={"schema_version":1,"participation":"serial","captures":captures} if facts is None else facts
    document={"sources":{str(original):sha256(original.read_bytes()).hexdigest()},**document}
    facts_file.write_text(json.dumps(document))
    output = directory/"output"
    module = ("compiler.tests.support.reference_scopes"
              if scope_execution != "graph" and (Path(checkout or ROOT) / "compiler" / "_legacy" / "config.py").is_file()
              else "compiler")
    response = run([sys.executable,"-m",module,"--form-scopes","--scope-facts",str(facts_file),
                    "--input",str(original),"--kernel",entry,"--memory-model","scoped","--gpu-policy",mode,
                    *(["--scope-execution", scope_execution] if scope_execution != "bounded" else []),
                    *( ["--calibration-profile", str(profile)] if profile else []),
                    "--json","--output-dir",str(output)],cwd=checkout or ROOT,
                   env={**os.environ,"PYTHONPATH":str(checkout or ROOT)})
    report = json.loads(response.stdout)
    assert report["supported"]
    manifest = json.loads((output/"scope-manifest.json").read_text())
    assert manifest == report["scopes"]
    return original,output,manifest


def assert_uncalibrated_native(manifest):
    assert manifest["scope_count"] == 0
    assert not manifest["scopes"]
    assert not manifest["source_edits"]
    assert not manifest["sources"]
    assert not manifest["build_sources"]
    assert manifest["runtime"] is None
    assert not manifest["automatic_estimate_available"]
    assert any("no calibration profile or exact-window evidence supplied; "
               "automatic policy retains unchanged native source" in item["reason"]
               for item in manifest["boundaries"])


@pytest.mark.parametrize("statement", ["hidden = 3*b", "allocate(hidden(8))", "deallocate(hidden)"])
def test_module_capture_assertion_cannot_override_source_allocation_effects(tmp_path, statement):
    source = PROGRAM.replace("implicit none\ncontains", "implicit none\nreal(8),allocatable::hidden(:)\ncontains")
    source = source.replace("b=3*b", statement + "\nb=3*b+hidden")
    facts = {"schema_version":1,"participation":"serial","captures":{
        "argument::a":FACT,"argument::b":{**FACT,"initialized":"none"},
        "argument::out":{**FACT,"initialized":"none"},"original::hidden":FACT}}
    _, _, manifest = generate(tmp_path, source, facts=facts)
    assert not manifest["automatic_scope_available"]
    assert manifest["boundaries"]


@pytest.mark.parametrize("shadowed", [False, True])
def test_allocatable_owner_and_shadowing_guards_compile_as_fortran(tmp_path, shadowed):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    source = ALLOCATABLE_PROGRAM
    if shadowed:
        source = source.replace("end module", """logical function allocated(value)
real(8),intent(in)::value(:)
allocated=.false.
end function
end module""")
    _, output, manifest = generate(tmp_path, source)
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in manifest["build_sources"]:
            if item["role"] == role and item["language"] == "fortran":
                run([fortran,"-std=f2018","-fopenmp","-fcheck=all","-c",str(output/item["path"]),
                     "-o",str((output/item["path"]).with_suffix(".o"))],cwd=output)


def test_collective_scope_is_rejected_without_source_changes(tmp_path):
    directory=tmp_path/"collective"
    directory.mkdir()
    original=directory/"original.f90"
    original.write_text(PROGRAM)
    facts=directory/"facts.json"
    facts.write_text(json.dumps({"schema_version":1,"participation":"collective","captures":{}}))
    result=subprocess.run([sys.executable,"-m","compiler","--form-scopes","--scope-facts",str(facts),
                           "--input",str(original),"--kernel","step","--memory-model","scoped","--gpu-policy","sections",
                           "--json","--output-dir",str(directory/"output")],cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert result.returncode
    assert "serial" in json.loads(result.stdout)["reason"]
    assert not (directory/"output").exists()


def test_automatic_unknown_cost_selects_original_native_span(tmp_path):
    original,output,manifest = generate(tmp_path,mode="auto")
    assert_uncalibrated_native(manifest)
    assert original.read_text() == PROGRAM
    assert [path.name for path in output.iterdir()] == ["scope-manifest.json"]


MIRROR_PROGRAM=PROGRAM.replace("subroutine transform(b)\nreal(8),intent(inout)::b(:)\nb=3*b",
                               "subroutine transform(b,c,total)\nreal(8),intent(in)::b(:),c(:)\n"
                               "real(8),intent(out)::total\ntotal=sum(b)+sum(c)")
MIRROR_PROGRAM=MIRROR_PROGRAM.replace("call producer(a,b,n)\ncall transform(b)",
                                     "call producer(a,b,n)\ncall transform(b,b,total)")
MIRROR_PROGRAM=MIRROR_PROGRAM.replace("subroutine step(a,b,out,n)\n", "subroutine step(a,b,out,n)\nreal(8)::total\n")
MIRROR_DRIVER=DRIVER.replace("6*a(i)+3*real(i+3,8)","2*a(i)+real(i+3,8)")
MIRROR_DRIVER=MIRROR_DRIVER.replace("7*a(i)+3*real(i+3,8)","3*a(i)+real(i+3,8)")

ELEMENT_PROGRAM=PROGRAM.split("subroutine step", 1)[0].replace(
    "implicit none", "implicit none\nreal(8)::field(16)", 1) + """subroutine inspect_scalar(value,total)
real(8),intent(in)::value
real(8),intent(out)::total
total=value
end subroutine
subroutine step(a,out,total)
real(8),intent(in)::a(:)
real(8),intent(out)::out(:),total
call producer(a,field,16)
call transform(field)
call inspect_scalar(field(1),total)
call consumer(a,field,out,16)
call transform(out)
end subroutine
end module
"""
ELEMENT_DRIVER="""program caller
use original,only:step,field
real(8)::a(16),output(16),total
integer::repeat,i
do repeat=1,2
 do i=1,16
  a(i)=real(i,8)*0.25d0+repeat
 enddo
 field=-99
 output=-99
 total=-99
 call step(a,output,total)
 if (total/=6*a(1)+3) error stop 'scalar actual observed stale GPU field'
 do i=1,16
  if (field(i)/=6*a(i)+3*real(i,8)) error stop 'module field disagreement'
  if (output(i)/=21*a(i)+9*real(i,8)) error stop 'post-boundary field disagreement'
 enddo
enddo
print *, 'FIELDS_OK'
end program
"""

WRAPPER_PROGRAM=PROGRAM.replace("subroutine step(a,b,out,n)","subroutine wrapper(a,b,out,n)")
WRAPPER_PROGRAM=WRAPPER_PROGRAM.replace("end module", """subroutine step(a,b,out,n)
real(8),intent(in)::a(:)
real(8),intent(out)::b(:),out(:)
integer,intent(in)::n
call wrapper(a,b,out,n)
call wrapper(a,b,out,n)
end subroutine
end module""")

CONTIGUOUS_WRAPPER_PROGRAM = WRAPPER_PROGRAM.replace("real(8),intent", "real(8),contiguous,intent")


def _synthetic_native_fallback_profile(directory, supplied_profile, flags):
    """Bind test-only observations to the supplied identity and original flags."""
    from compiler.offload.collective_calibration import normalize_fortran_options
    from compiler.offload.numerical_calibration import profile_from_compute_measurements
    from compiler.tests.test_numerical_calibration_v2 import observations_v2

    base = json.loads(supplied_profile.read_text())
    assert base["precision_bits"] == 64, "this fixture uses original REAL(8) source"
    probe = directory / "fortran-identity.f90"
    probe.write_text("""program identity
use iso_fortran_env,only:compiler_version,compiler_options
print '(a)',compiler_version()
print '(a)',compiler_options()
end program
""")
    object_file, binary = directory / "fortran-identity.o", directory / "fortran-identity"
    commands = []
    # Original modules are compiled with -c. A combined compile/link probe
    # can report driver-injected link options absent from those source objects.
    for role, command in (
        ("identity-compile", [*flags, "-c", str(probe), "-o", str(object_file)]),
        ("identity-link", [*flags, str(object_file), "-o", str(binary)]),
        ("identity-run", [str(binary)]),
    ):
        response = run(command, cwd=directory)
        commands.append({"role": role, "argv": command, "returncode": response.returncode,
                         "stdout": response.stdout, "stderr": response.stderr})
    version, options = commands[-1]["stdout"].strip().splitlines()
    observations = observations_v2()
    identity = observations[0]
    identity.update({name: base["hardware"][name]
                     for name in ("gpu_name", "gpu_uuid", "compute_capability")})
    identity.update({name: base["toolchain"][name]
                     for name in ("cuda_runtime_version", "driver_version")})
    identity.update(precision_bits=base["precision_bits"], cpu_threads=base["cpu_threads"],
                    cpu_affinity=sorted(os.sched_getaffinity(0)),
                    fortran={"compiler_version": version, "compiler_options": options,
                             "semantic_options": normalize_fortran_options(options)})
    assert len(identity["cpu_affinity"]) == base["cpu_threads"], "matching offline CPU placement required"
    # This fixture tests the all-native branch, independently of measured
    # host management costs. The shared synthetic observations model GPU
    # compute at 0.05 times serial native cost; scale only those synthetic
    # GPU durations to make its primitive and memory costs 100 times native.
    # Keep the fit/holdout grid, repetitions and genuine base costs intact.
    gpu_duration_scale = 2000
    for observation in observations[1:]:
        if observation["backend"] == "gpu":
            for sample in observation["samples"]:
                sample["elapsed_seconds"] *= gpu_duration_scale
                sample["wall_seconds"] *= gpu_duration_scale
    synthetic_ranking = {"version": 1, "purpose": "all_native_correctness",
                         "gpu_duration_scale": gpu_duration_scale,
                         "gpu_to_native_serial_compute_cost": 100}
    # AUTO graph placement uses the generic hardware-rate path instead of
    # the numerical-v2 rates. Give that path the same explicit test ranking.
    synthetic_rate_overrides = {
        "gpu_flops_per_second": base["rates"]["cpu_flops_per_second"] / 100,
        "gpu_memory_bytes_per_second": base["rates"]["cpu_memory_bytes_per_second"] / 100,
    }
    synthetic_base = {**base, "rates": {**base["rates"], **synthetic_rate_overrides}}
    synthetic_ranking["hardware_rate_overrides"] = synthetic_rate_overrides
    calibration = profile_from_compute_measurements(synthetic_base, observations,
        calibration={"synthetic_correctness_only": True,
                     "synthetic_ranking": synthetic_ranking})
    assert {key: value for key, value in calibration.items() if key != "numerical"} == {
        key: value for key, value in synthetic_base.items() if key != "numerical"}
    profile = directory / "synthetic-native-fallback-profile.json"
    profile.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")
    (directory / "synthetic-native-fallback-profile-evidence.json").write_text(json.dumps({
        "schema_version": 2, "synthetic_calibration_for_correctness_only": True,
        "synthetic_ranking": synthetic_ranking,
        "performance_samples": 0, "base_profile": {
            "path": str(supplied_profile), "sha256": sha256(supplied_profile.read_bytes()).hexdigest()},
        "profile": {"path": str(profile), "sha256": sha256(profile.read_bytes()).hexdigest()},
        "compile_mode": "separate_object", "fortran_flags": flags,
        "fortran": identity["fortran"], "cpu_affinity": identity["cpu_affinity"], "commands": commands,
        "artifacts_sha256": {path.name: sha256(path.read_bytes()).hexdigest()
                             for path in (probe, object_file, binary)},
    }, indent=2, sort_keys=True) + "\n")
    return profile


def compile_contiguous_native_fallback(directory, *, checkout, profile, source=None, shared_objects=None):
    """Small public-interface fixture; reusable CUDA objects need byte proof."""
    directory.mkdir(parents=True, exist_ok=True)
    original = source or directory / "original.f90"
    if source is None:
        original.write_text(CONTIGUOUS_WRAPPER_PROGRAM)
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    host = shutil.which("g++-14") or shutil.which("g++")
    nvcc = shutil.which("nvcc")
    assert fortran, "CUDA/Fortran fixture requires installed Fortran toolchain"
    assert host, "CUDA/Fortran fixture requires installed C++ toolchain"
    assert nvcc, "CUDA/Fortran fixture requires installed CUDA toolchain"
    flags = [fortran, "-std=f2018", "-fopenmp", "-fcheck=all,array-temps"]
    profile = _synthetic_native_fallback_profile(directory, profile, flags)
    captures = {"schema_version": 1, "participation": "serial",
                "sources": {str(original): sha256(original.read_bytes()).hexdigest()},
                "captures": {"argument::a": FACT, "argument::b": {**FACT, "initialized": "none"},
                             "argument::out": {**FACT, "initialized": "none"}}}
    facts = directory / "captures.json"
    facts.write_text(json.dumps(captures))
    output = directory / "output"
    response = run([sys.executable, "-m", "compiler", "--input", str(original), "--kernel", "step",
                    "--form-scopes", "--scope-facts", str(facts), "--memory-model", "scoped", "--gpu-policy", "auto",
                    "--calibration-profile", str(profile), "--json", "--output-dir", str(output)], cwd=checkout,
                   env={**os.environ, "PYTHONPATH": str(checkout), "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "4"})
    manifest = json.loads(response.stdout)["scopes"]
    assert manifest["automatic_estimate_available"], manifest["boundaries"]
    objects, reusable = [], {}
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in manifest["build_sources"]:
            if item["role"] != role:
                continue
            target = output / (item["path"].replace("/", "_") + ".o")
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = manifest["artifacts_sha256"][item["path"]]
            old = (shared_objects or {}).get(item["path"]) if item["language"] == "cuda" else None
            if old:
                assert old["sha256"] == digest, "CUDA object reuse requires byte-identical source: " + item["path"]
                target = Path(old["object"])
            else:
                command = ([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp", "-I", str(output)]
                           if item["language"] == "cuda" else flags)
                run([*command, "-c", str(output / item["path"]), "-o", str(target)], cwd=output)
            objects.append(str(target))
            if item["language"] == "cuda":
                reusable[item["path"]] = {"sha256": digest, "object": str(target)}
    driver = output / "caller.f90"
    # Keep the generic shared driver unchanged. This calibrated fixture needs
    # dimensions within the existing v2 compute and physical working-set ranges.
    driver.write_text(DRIVER.replace("n=8+8*shape", "n=131072*shape"))
    target = output / "caller"
    run([*flags, str(driver), *objects,
         "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++", "-o", str(target)], cwd=output)
    native = output / "native"
    run([*flags, str(original), str(driver),
         "-o", str(native)], cwd=output)
    reference = run([str(native)], cwd=output, env={**os.environ, "OMP_NUM_THREADS": "4"})
    assert "FIELDS_OK" in reference.stdout
    assert "array temporary" not in reference.stderr.lower()
    return target, output, manifest, reusable


@pytest.fixture(scope="module")
def contiguous_native_fallback(tmp_path_factory):
    profile = os.environ.get("FORT_TEST_SCOPED_PROFILE")
    if not profile:
        pytest.skip("matching explicit offline scoped calibration profile required")
    if not all(shutil.which(tool) for tool in ("nvcc", "gfortran-15", "g++-14")):
        pytest.skip("CUDA/Fortran toolchain unavailable")
    return compile_contiguous_native_fallback(tmp_path_factory.mktemp("contiguous_native_fallback"),
                                              checkout=ROOT, profile=Path(profile).resolve())


@pytest.mark.native
def test_calibrated_all_native_wrappers_do_not_create_array_temporaries(contiguous_native_fallback):
    target, output, _manifest, _reusable = contiguous_native_fallback
    result = run([str(target)], cwd=output,
                 env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"})
    assert "FIELDS_OK" in result.stdout
    assert "array temporary" not in result.stderr.lower()
    decisions = [json.loads(line.removeprefix("FORT_SCOPED evidence ")) for line in result.stderr.splitlines()
                 if line.startswith("FORT_SCOPED evidence ")]
    decisions = [entry for entry in decisions if entry["event"] == "decision"]
    assert decisions
    assert all(entry["available"] == 1 for entry in decisions)
    assert all(entry["gpu_units"] == 0 for entry in decisions)
    assert "FORT_SCOPED launch" not in result.stderr
    assert "FORT_SCOPED upload" not in result.stderr
    assert "FORT_SCOPED download" not in result.stderr

_before_step, _step = PROGRAM.split("subroutine step(", 1)
ALLOCATABLE_PROGRAM = _before_step + "subroutine step(" + _step.replace(
    "real(8),intent(in)::a(:)", "real(8),allocatable,intent(in)::a(:)", 1).replace(
    "real(8),intent(out)::b(:),out(:)", "real(8),allocatable,intent(inout)::b(:),out(:)", 1).replace(
    "call producer(a,b,n)\ncall transform(b)\ncall consumer(a,b,out,n)",
    "if (allocated(a) .and. allocated(b) .and. allocated(out)) then\n"
    "call producer(a,b,n)\ncall transform(b)\ncall consumer(a,b,out,n)\nendif")

ALLOCATABLE_DRIVER = DRIVER.replace("do shape=1,2", "n=1\ncall step(a,b,output,n)\ndo shape=0,2").replace(
    "n=8+8*shape", "n=0\n if(shape>0) n=8+8*shape")

PARTIAL_PROGRAM="""module original
contains
subroutine producer(a,b)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer::i
do i=3,6
b(i)=2*a(i)
enddo
end subroutine
subroutine consumer(a,b,out)
real(8),intent(in)::a(:),b(:)
real(8),intent(inout)::out(:)
integer::i
do i=3,6
out(i)=a(i)+b(i)
enddo
end subroutine
subroutine step(a,b,out)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),out(:)
call producer(a,b)
call consumer(a,b,out)
end subroutine
end module
"""
PARTIAL_DRIVER="""program caller
use original,only:step
real(8)::a(-2:5),b(-2:5),output(-2:5)
a=4
b=-99
output=-99
call step(a,b,output)
if (any(b(0:3)/=8) .or. any(output(0:3)/=12)) error stop 'partial field disagreement'
if (any(b(-2:-1)/=-99) .or. any(b(4:5)/=-99)) error stop 'input halo overwritten'
if (any(output(-2:-1)/=-99) .or. any(output(4:5)/=-99)) error stop 'output halo overwritten'
print *, 'FIELDS_OK'
end program
"""

TEAM_DRIVER="""program caller
use original,only:step
real(8)::a(16),b(16),output(16)
!$omp parallel private(a,b,output)
a=4
call step(a,b,output,16)
if(any(b/=[(24.d0+3*i,i=1,16)])) error stop 'team intermediate disagreement'
if(any(output/=[(28.d0+3*i,i=1,16)])) error stop 'team output disagreement'
!$omp end parallel
print *, 'FIELDS_OK'
end program
"""

ALLOCATABLE_TEAM_DRIVER = TEAM_DRIVER.replace("real(8)::a(16),b(16),output(16)",
                                             "real(8),allocatable::a(:),b(:),output(:)").replace(
    "a=4\ncall step", "allocate(a(-2:13),b(-2:13),output(-2:13))\na=4\ncall step").replace(
    "!$omp end parallel", "deallocate(a,b,output)\n!$omp end parallel")


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    nvcc=shutil.which("nvcc")
    host=shutil.which("g++-14") or shutil.which("g++")
    fortran=shutil.which("gfortran-15") or shutil.which("gfortran")
    if not nvcc or not host or not fortran:
        pytest.skip("CUDA/Fortran toolchain unavailable")
    directory=tmp_path_factory.mktemp("automatic_source_scope")
    checkout=directory/"independent-compiler"
    shutil.copytree(ROOT/"compiler",checkout/"compiler",ignore=shutil.ignore_patterns("__pycache__",".*cache"))
    targets={}
    runtime_objects = {}
    partial_facts={"schema_version":1,"participation":"serial","captures":{
        "argument::a":{**FACT,"initialized":"sections","sections":[{"lower":[2],"upper":[6]}]},
        "argument::b":{**FACT,"initialized":"none"},"argument::out":{**FACT,"initialized":"none"}}}
    budget_facts={"schema_version":1,"participation":"serial","device_budget_bytes":400,
                  "captures":{"argument::a":FACT,"argument::b":{**FACT,"initialized":"none"},
                              "argument::out":{**FACT,"initialized":"none"}}}
    contiguous = PROGRAM.replace("real(8),intent", "real(8),contiguous,intent")
    element_facts={"schema_version":1,"participation":"serial","captures":{
        "argument::a":FACT,"original::field":FACT,"argument::out":{**FACT,"initialized":"none"}}}
    cases=[("sections","sections",PROGRAM,DRIVER,None), ("auto","auto",PROGRAM,DRIVER,None),
           ("allocated_sections","sections",ALLOCATABLE_PROGRAM,ALLOCATABLE_DRIVER,None),
           ("allocated_auto","auto",ALLOCATABLE_PROGRAM,ALLOCATABLE_DRIVER,None),
           ("contiguous","sections",contiguous,DRIVER,None),
           ("budget","sections",PROGRAM,DRIVER,budget_facts),
           ("mirror","sections",MIRROR_PROGRAM,MIRROR_DRIVER,None),
           ("wrapper","sections",WRAPPER_PROGRAM,DRIVER,None),
           ("partial","sections",PARTIAL_PROGRAM,PARTIAL_DRIVER,partial_facts),
           ("indexed","sections",ELEMENT_PROGRAM,ELEMENT_DRIVER,element_facts)]
    for label,mode,program,driver_source,facts in cases:
        case=directory/label
        checks = "-fcheck=all,array-temps" if label.startswith("allocated") else "-fcheck=all"
        profile = os.environ.get("FORT_TEST_SCOPED_PROFILE") if label == "allocated_auto" else None
        original,output,manifest=generate(case,program,mode=mode,facts=facts,checkout=checkout,profile=profile)
        if mode == "auto" and profile is None:
            assert_uncalibrated_native(manifest)
            driver = output / "caller.f90"
            driver.write_text(driver_source)
            target = output / "caller"
            run([fortran,"-std=f2018","-fopenmp",checks,str(original),str(driver),"-o",str(target)],cwd=output)
            targets[label] = target, output
            continue
        assert manifest["scope_count"] == (2 if label == "indexed" else 1)
        objects=[]
        # Build common interface first, then generated interfaces, then the
        # original modules with approved replacements. No IR/CUDA introspection.
        sources=manifest["build_sources"]
        for item in [s for s in sources if s["role"]=="common_runtime"]:
            target=output/Path(item["path"]).with_suffix(".o").name
            runtime_key = (manifest["runtime"]["runtime_id"], item["path"])
            if item["language"] == "cuda" and runtime_key in runtime_objects:
                # Runtime identity binds all published headers and sources;
                # this suite uses identical CUDA compiler flags for every case.
                assert sha256((output/item["path"]).read_bytes()).hexdigest() == manifest["artifacts_sha256"][item["path"]]
                objects.append(runtime_objects[runtime_key])
                continue
            command=([nvcc,"-std=c++17","-ccbin",host,"-arch=sm_86","-Xcompiler=-fopenmp","-c"]
                     if item["language"]=="cuda" else [fortran,"-std=f2018","-c"])
            run([*command,str(output/item["path"]),"-o",str(target)],cwd=output)
            objects.append(str(target))
            if item["language"] == "cuda":
                runtime_objects[runtime_key] = str(target)
        for item in [s for s in sources if s["role"]=="shared_entry"]:
            target=(output/item["path"]).with_suffix(".o")
            if item["language"]=="cuda":
                command=[nvcc,"-std=c++17","-ccbin",host,"-arch=sm_86","-Xcompiler=-fopenmp","-I",str(output),"-c"]
            else:
                command=[fortran,"-std=f2018","-c"]
            run([*command,str(output/item["path"]),"-o",str(target)],cwd=output)
            objects.append(str(target))
        for item in [s for s in sources if s["role"]=="original_source"]:
            target=(output/item["path"]).with_suffix(".o")
            run([fortran,"-std=f2018","-fopenmp",checks,"-c",str(output/item["path"]),
                 "-o",str(target)],cwd=output)
            objects.append(str(target))
        driver=output/"caller.f90"
        driver.write_text(driver_source)
        target=output/"caller"
        run([fortran,"-std=f2018","-fopenmp",checks,str(driver),*objects,
             "-L/usr/local/cuda/lib64","-Wl,-rpath,/usr/local/cuda/lib64","-lcudart","-lstdc++",
             "-o",str(target)],cwd=output)
        # Check the original native program too, with all shape/call variations.
        native=output/"native"
        run([fortran,"-std=f2018","-fopenmp",checks,str(original),str(driver),"-o",str(native)],cwd=output)
        run([str(native)],cwd=output,env={**os.environ,"OMP_NUM_THREADS":"4"})
        targets[label]=(target,output)
        if label in {"sections", "allocated_sections"}:
            driver.write_text(ALLOCATABLE_TEAM_DRIVER if label == "allocated_sections" else TEAM_DRIVER)
            team=output/"team-caller"
            run([fortran,"-std=f2018","-fopenmp",checks,str(driver),*objects,
                 "-L/usr/local/cuda/lib64","-Wl,-rpath,/usr/local/cuda/lib64","-lcudart","-lstdc++",
                 "-o",str(team)],cwd=output)
            targets["allocated_team" if label == "allocated_sections" else "team"]=(team,output)
    return targets
