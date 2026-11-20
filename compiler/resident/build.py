"""Build a prepared resident application and its native control with CMake.

The generated project compiles the application twice: once from the original
sources, and once from the prepared sources together with the compiled
entries and the residency runtime. Both use the compiler's numerical build
options, so their arithmetic differs only by where it runs.
"""
import contextlib
import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from compiler.numerical_contract import CUDA_OPTIONS, HOST_OPTIONS
from compiler.runtime import DEFAULT_DOMAIN_SOURCE, RESIDENT_SOURCES

FAST_CUDA_OPTIONS = ("--fmad=true", "--ftz=false", "--prec-div=false", "--prec-sqrt=false")
HIDDEN_OPTIONS = ("CFLAGS", "CXXFLAGS", "CUDAFLAGS", "FFLAGS", "FCFLAGS", "F90FLAGS", "LDFLAGS",
                  "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")
ENTRY_SOURCES = ("shared_interface.f90", "shared_entry.cu")


@dataclass(frozen=True)
class ObjectLibrary:
    """Sources compiled together into one set of objects and Fortran modules."""

    name: str
    sources: tuple[Path, ...]
    link: tuple[str, ...] = ()


@dataclass(frozen=True)
class Executable:
    """A main program linked against one object library."""

    name: str
    source: Path
    library: str


def cmake_path(path):
    value = str(Path(path).resolve())
    if any(character in value for character in ('"', ";", "\n", "$")):
        raise ValueError("Unsupported CMake path: " + value)
    return '"' + value + '"'


def cmake_lists(name, libraries, executables, *, include_directories=(), definitions=(),
                fortran_options=("-O3", "-fopenmp", "-ffree-line-length-none", "-g"),
                c_options=("-O3", "-fopenmp", "-g"), prologue=(), source_options=None,
                strict_fp=True):
    """The text of a CMake project for the given object libraries and executables.

    ``prologue`` lines come before the targets, for dependencies the
    application brings (``find_library``, ``add_subdirectory``, interface
    targets named in a library's ``link``). ``source_options`` maps a source to
    compile options that apply to that file only. Fortran sources are
    preprocessed; when ``strict_fp`` is true (default), every language gets the
    compiler's strict numerical options.
    """
    host_options = HOST_OPTIONS if strict_fp else ()
    cuda_options = CUDA_OPTIONS if strict_fp else FAST_CUDA_OPTIONS
    host_cmake = (";" + ";".join(host_options)) if host_options else ""
    host_xcompiler = ("," + ",".join(host_options)) if host_options else ""
    host_space = (" " + " ".join(host_options)) if host_options else ""
    lines = ["cmake_minimum_required(VERSION 3.24)", f"project({name} LANGUAGES C CXX Fortran CUDA)",
             "set(CMAKE_EXPORT_COMPILE_COMMANDS ON)", "find_package(CUDAToolkit REQUIRED)",
             "find_package(OpenMP REQUIRED)", *prologue]
    includes = " ".join(cmake_path(path) for path in include_directories)
    # A program that is its application's only source has no library beside it.
    empty = {library.name for library in libraries if not library.sources}
    for library in libraries:
        if library.name in empty:
            continue
        target = library.name + "_objects"
        modules = "${CMAKE_BINARY_DIR}/" + library.name + "-modules"
        lines += [f"add_library({target} OBJECT\n  " + "\n  ".join(cmake_path(path) for path in library.sources) + "\n)",
                  f'file(MAKE_DIRECTORY "{modules}")',
                  f'set_target_properties({target} PROPERTIES Fortran_MODULE_DIRECTORY "{modules}")',
                  f'target_include_directories({target} PUBLIC "{modules}" {includes})']
        if definitions:
            lines.append(f"target_compile_definitions({target} PRIVATE {' '.join(definitions)})")
        lines += [f"target_compile_options({target} PRIVATE "
                  f'"$<$<COMPILE_LANGUAGE:Fortran>:-cpp;{";".join(fortran_options)}{host_cmake}>" '
                  f'"$<$<COMPILE_LANGUAGE:C,CXX>:{";".join(c_options)}{host_cmake}>" '
                  f'"$<$<COMPILE_LANGUAGE:CUDA>:-O3;-std=c++17;{";".join(cuda_options)};'
                  f'-Xcompiler=-fPIC,-fopenmp{host_xcompiler}>" )',
                  f"target_link_libraries({target} PUBLIC OpenMP::OpenMP_Fortran {' '.join(library.link)})"]
    for source, options in (source_options or {}).items():
        lines.append(f"set_source_files_properties({cmake_path(source)} PROPERTIES COMPILE_OPTIONS "
                     + ";".join(options) + ")")
    for executable in executables:
        lines += [f"add_executable({executable.name} {cmake_path(executable.source)})",
                  f"target_link_libraries({executable.name} PRIVATE "
                  + ("OpenMP::OpenMP_Fortran)" if executable.library in empty else f"{executable.library}_objects)"),
                  f"target_compile_options({executable.name} PRIVATE -cpp {' '.join(fortran_options)}{host_space})"]
        if definitions:
            lines.append(f"target_compile_definitions({executable.name} PRIVATE {' '.join(definitions)})")
        lines.append(f"set_target_properties({executable.name} PROPERTIES LINKER_LANGUAGE Fortran)")
    return "\n".join(lines) + "\n"


def _hashes(root):
    return {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest() for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".o", ".mod", ".py"}}


def snapshot_directory(source, target):
    """Copy a source directory into the build, or check an earlier copy is still what the source holds."""
    source, target = Path(source), Path(target)
    if target.exists():
        if _hashes(source) != _hashes(target):
            raise ValueError("Build dependency changed; prepare a fresh build: " + str(source))
    else:
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.o", "*.mod", "*.py"))
    return target


def run_logged(argv, directory, label, *, timeout):
    """Run one build step in its own process group, with its output in ``<label>.log``."""
    directory = Path(directory)
    record = {"argv": [str(item) for item in argv], "cwd": str(directory), "status": "running"}
    receipt, log = directory / (label + "-command.json"), directory / (label + ".log")
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    start, process = time.monotonic(), None
    try:
        with log.open("w") as stream:
            process = subprocess.Popen(record["argv"], cwd=directory, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            returncode = process.wait(timeout=timeout)
        record.update(returncode=returncode, status="passed" if returncode == 0 else "failed")
        if returncode:
            raise RuntimeError(f"{label} failed; inspect {log}")
    except BaseException as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        if process is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        record["seconds"] = time.monotonic() - start
        receipt.write_text(json.dumps(record, indent=2) + "\n")


def configure_and_compile(directory, *, fortran, cxx, nvcc, architecture, jobs=2, configure_timeout=240,
                          compile_timeout=2400):
    """Run CMake on ``directory/CMakeLists.txt`` and build into ``directory/build``."""
    if jobs < 1:
        raise ValueError("Build jobs must be positive")
    for key in HIDDEN_OPTIONS:
        if os.environ.get(key):
            raise ValueError("Hidden compiler options are not supported: " + key)
    directory = Path(directory).resolve()
    output = directory / "build"
    fortran, cxx, nvcc = (Path(tool).resolve() for tool in (fortran, cxx, nvcc))
    run_logged(["cmake", "-S", directory, "-B", output, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_Fortran_COMPILER={fortran}", f"-DCMAKE_CXX_COMPILER={cxx}",
                f"-DCMAKE_CUDA_COMPILER={nvcc}", f"-DCMAKE_CUDA_HOST_COMPILER={cxx}",
                f"-DCMAKE_CUDA_ARCHITECTURES={architecture}"], directory, "configure", timeout=configure_timeout)
    run_logged(["cmake", "--build", output, "--parallel", str(jobs)], directory, "compile", timeout=compile_timeout)
    return output


def resident_sources(prepared, runtime, *, programs=()):
    """The sources of the resident object library: prepared sources, runtime and compiled entries."""
    programs = {str(Path(path).resolve()) for path in programs}
    sources = [Path(item["resident"]) for item in prepared["sources"]
               if item["original"] is None or str(Path(item["original"]).resolve()) not in programs]
    sources += [Path(runtime) / name for name in RESIDENT_SOURCES]
    if prepared["domain"]["default"]:
        sources.append(Path(runtime) / DEFAULT_DOMAIN_SOURCE)
    for entry in prepared["entries"]:
        sources += [Path(entry) / name for name in ENTRY_SOURCES]
    return sources


def build(prepared, directory, *, name, programs, fortran, cxx, nvcc, architecture, jobs=2, native=True,
          fortran_options=("-O3", "-fopenmp", "-ffree-line-length-none", "-g"), strict_fp=True):
    """Build the prepared application into ``directory/build``.

    ``programs`` maps an executable name to the original source of its main
    program, which must be one of the project's sources. Each program is built
    resident under that name and, with ``native``, from the original sources as
    ``<name>-native``. Returns the paths of the executables by name.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    runtime = snapshot_directory(Path(__file__).resolve().parents[1] / "runtime", directory / "runtime")
    known = {str(Path(item["original"]).resolve()): Path(item["resident"]) for item in prepared["sources"]
             if item["original"] is not None}
    mains = {target: str(Path(source).resolve()) for target, source in programs.items()}
    missing = [source for source in mains.values() if source not in known]
    if missing:
        raise ValueError("Program sources are not project sources: " + ", ".join(missing))
    libraries = [ObjectLibrary("resident", tuple(resident_sources(prepared, runtime, programs=mains.values())),
                               link=("CUDA::cudart",))]
    executables = [Executable(target, known[source], "resident") for target, source in mains.items()]
    if native:
        libraries.append(ObjectLibrary("native", tuple(Path(path) for path in known if path not in mains.values())))
        executables += [Executable(target + "-native", Path(source), "native") for target, source in mains.items()]
    include = sorted({Path(path).parent for path in known}) + [runtime]
    (directory / "CMakeLists.txt").write_text(cmake_lists(
        name, libraries, executables, include_directories=include,
        definitions=tuple(prepared["defines"]), fortran_options=fortran_options, strict_fp=strict_fp))
    output = configure_and_compile(directory, fortran=fortran, cxx=cxx, nvcc=nvcc, architecture=architecture, jobs=jobs)
    return {executable.name: output / executable.name for executable in executables}
