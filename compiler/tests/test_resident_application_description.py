"""An application described in TOML: its edits, the checks of a description, and the examples under benchmarks/."""
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from compiler.resident import build as resident_build
from compiler.resident.__main__ import main
from compiler.resident.application import Edit, apply_edit, load, prepare, snapshot_inputs
from compiler.resident.build import compiler_path, configure_and_compile
from compiler.tests.resident_application_support import fortran_compiler

ROOT = Path(__file__).resolve().parents[2]

SOURCE = """module m
contains
  subroutine first(a)
    real :: a
    call helper(a)
    a = a + 1
  end subroutine first

  subroutine second(a)
    real :: a
    a = a + 1
  end subroutine second
end module m
"""


def edited(**fields):
    return apply_edit(SOURCE, Edit(file="m.f90", **fields))


def test_an_edit_inserts_before_or_after_an_anchor_found_once_in_its_procedure():
    # "a = a + 1" occurs in both procedures: confined to one, it is found once.
    text = edited(procedure="second", at="    a = a + 1\n", action="after", text="    a = 2 * a\n")
    assert "    a = a + 1\n    a = 2 * a\n  end subroutine second" in text
    with pytest.raises(ValueError, match="expected one"):
        edited(at="    a = a + 1\n", action="after", text="x")


def test_a_replacement_keeps_what_it_replaces_where_it_says_original():
    # A CALL statement is replaced from the start of its line.
    text = edited(procedure="first", call="helper", action="replace", text="    if (on) then\n{original}\n    end if")
    assert "    if (on) then\n    call helper(a)\n    end if\n" in text
    text = edited(procedure="first", between=["call helper", "a = a + 1"], action="replace", text="! gone")
    assert "    ! gone\n  end subroutine first" in text


def test_an_edit_can_add_a_procedure_before_another_or_at_the_end_of_a_module():
    text = edited(procedure="second", action="before", text="  subroutine third\n  end subroutine third\n\n")
    assert text.index("subroutine third") < text.index("subroutine second(a)")
    text = edited(module_end="m", action="before", text="\n  ! last\n")
    assert text.endswith("  end subroutine second\n\n  ! last\nend module m\n")


def test_a_description_is_checked_where_it_is_read(tmp_path):
    def write(text):
        path = tmp_path / "app.toml"
        path.write_text('name = "app"\nsources = ["a.f90"]\nprogram = "a.f90"\n' + text)
        return path
    assert load(write("[variants.default]\n")).sources == ("a.f90",)
    with pytest.raises(ValueError, match="Unknown keys in selection"):
        load(write('[[selection]]\nlabel = "x"\nprocedures = []\nfloor = 3\n'))
    with pytest.raises(ValueError, match="locates no place"):
        load(write('[[edit]]\nfile = "a.f90"\naction = "after"\ntext = ""\n'))
    with pytest.raises(ValueError, match="not described"):
        load(write('[build]\ninclude = ["solver:include"]\n'))
    with pytest.raises(ValueError, match="a digest, not a revision"):
        load(write('[inputs.suite]\nsha256 = "0"\nrevision = "0"\n'))


def test_an_archive_input_is_checked_against_its_digest_and_unpacked_without_its_top_directory(tmp_path):
    tree = tmp_path / "tree"
    for name, text in (("suite-1.0/src/a.f90", "! kept\n"), ("suite-1.0/src/notes.txt", "suffix not copied\n"),
                       ("suite-1.0/doc/b.f90", "not in the copy list\n")):
        (tree / name).parent.mkdir(parents=True, exist_ok=True)
        (tree / name).write_text(text)
    archive = tmp_path / "suite.tar.gz"
    with tarfile.open(archive, "w:gz") as members:
        members.add(tree / "suite-1.0", arcname="suite-1.0")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()

    def snapshot(given, sha256=digest):
        path = tmp_path / "app.toml"
        path.write_text('name = "app"\nsources = ["suite:src/a.f90"]\nprogram = "suite:src/a.f90"\n'
                        f'[inputs.suite]\nsha256 = "{sha256}"\ncopy = ["src"]\nsuffixes = [".f90"]\n')
        output = tmp_path / "snapshot"
        shutil.rmtree(output, ignore_errors=True)
        return snapshot_inputs(load(path), {"suite": given}, output, ["suite"])

    paths, records = snapshot(archive)
    assert sorted(path.relative_to(paths["suite"]).as_posix() for path in paths["suite"].rglob("*") if path.is_file()) \
        == ["src/a.f90"]
    assert records["suite"][0]["original"]["path"] == f"{archive}:src/a.f90"
    with pytest.raises(ValueError, match="must be the archive with sha256"):
        snapshot(archive, sha256="0" * 64)
    with pytest.raises(ValueError, match="must be the archive"):
        snapshot(tree)
    # A member that would land outside the directory it is unpacked into is refused.
    unsafe = tmp_path / "unsafe.tar"
    with tarfile.open(unsafe, "w") as members:
        members.add(tree / "suite-1.0/src/a.f90", arcname="../a.f90")
    with pytest.raises(ValueError, match="unpacks safely"):
        snapshot(unsafe, sha256=hashlib.sha256(unsafe.read_bytes()).hexdigest())


@pytest.mark.skipif(fortran_compiler() is None, reason="GNU Fortran is required for preprocessing")
def test_a_procedure_inside_a_main_program_is_selected_by_itself_with_its_header_from_an_include_directory(
        tmp_path, capsys):
    (tmp_path / "include").mkdir()
    (tmp_path / "include/sizes.h").write_text("#define N 64\n")
    (tmp_path / "main.F90").write_text("""#include <sizes.h>
program scale
  implicit none
  real(8), allocatable :: a(:)
  allocate(a(N))
  a = 1
  call twice(N, a)
  print *, sum(a)
contains
  subroutine twice(n, a)
    integer :: n
    real(8) :: a(n)
    integer :: i
    do i = 1, n
      a(i) = 2 * a(i)
    end do
  end subroutine twice
end program scale
""")

    def describe(procedure, include='include = ["include"]\n'):
        path = tmp_path / "scale.toml"
        path.write_text(f'name = "scale"\nsources = ["main.F90"]\nprogram = "main.F90"\n[variants.default]\n'
                        f'[[selection]]\nlabel = "all"\nprocedures = ["{procedure}"]\nminimum_nests = 1\n'
                        f'[build]\n{include}')
        return path

    prepare(load(describe("$program::scale::twice")), {}, tmp_path / "prepared", variant="default",
            fortran=fortran_compiler())
    with pytest.raises(RuntimeError, match="Preprocessing failed"):
        prepare(load(describe("$program::scale::twice", include="")), {}, tmp_path / "unfound", variant="default",
                fortran=fortran_compiler())
    # The driver reports a compiler error in one line.
    with pytest.raises(SystemExit):
        main([str(describe("$program::scale::thrice")), "--output", str(tmp_path / "missing"),
              "--fortran", fortran_compiler(), "--prepare-only"])
    assert capsys.readouterr().err == "error: Resident procedure is unavailable: $program::scale::thrice\n"


def test_a_compiler_is_a_path_or_a_name_looked_up_on_path_as_a_shell_would(tmp_path, monkeypatch):
    tools, work = tmp_path / "tools", tmp_path / "work"
    for directory in (tools, work / "bin"):
        directory.mkdir(parents=True)
    for tool in (tools / "fc", tools / "cxx", tools / "nvcc", work / "bin" / "cxx", work / "cxx"):
        tool.write_text("#!/bin/sh\n")
        tool.chmod(0o755)
    (tools / "wrapper").symlink_to(tools / "cxx")
    monkeypatch.setenv("PATH", f"{tools}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.chdir(work)
    # A bare name is found on PATH, even where the working directory has a file of that name.
    assert compiler_path("cxx") == tools / "cxx"
    # A name with a directory is a path, relative to the working directory.
    assert compiler_path("bin/cxx") == Path.cwd() / "bin/cxx"
    assert compiler_path("./cxx") == Path.cwd() / "cxx"
    # A symbolic link stays the name it is called by, as a wrapper such as ccache needs.
    assert compiler_path("wrapper") == tools / "wrapper"
    with pytest.raises(FileNotFoundError, match="No such compiler: missing"):
        compiler_path("missing")
    # CMake, which runs in the build directory, is given each compiler's path.
    commands = []
    monkeypatch.setattr(resident_build, "run_logged", lambda argv, *rest, **options: commands.append(argv))
    configure_and_compile(tmp_path / "project", fortran="fc", cxx="wrapper", nvcc="nvcc", architecture="86")
    assert {f"-DCMAKE_Fortran_COMPILER={tools / 'fc'}", f"-DCMAKE_CXX_COMPILER={tools / 'wrapper'}",
            f"-DCMAKE_CUDA_COMPILER={tools / 'nvcc'}", f"-DCMAKE_CUDA_HOST_COMPILER={tools / 'wrapper'}"} <= set(commands[0])


@pytest.mark.skipif(fortran_compiler() is None, reason="GNU Fortran is required for preprocessing")
def test_the_driver_refuses_a_compiler_it_cannot_find_before_it_prepares(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main([str(ROOT / "benchmarks/cdu/cdu.toml"), "--output", str(tmp_path / "cdu"), "--fortran", fortran_compiler(),
              "--cxx", "no-such-cxx", "--nvcc", "no-such-nvcc", "--architecture", "86"])
    assert capsys.readouterr().err == "error: No such compiler: no-such-cxx\n"
    assert not (tmp_path / "cdu").exists()


def test_an_input_is_a_checkout_of_its_revision_or_of_a_descendant(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()

    def git(*arguments):
        return subprocess.run(["git", "-C", str(checkout), "-c", "user.name=Test", "-c", "user.email=test@example.com",
                               *arguments], capture_output=True, text=True, check=True).stdout.strip()

    git("init", "-q")
    (checkout / "a.f90").write_text("! pinned\n")
    git("add", "a.f90")
    git("commit", "-qm", "pinned")
    pinned = git("rev-parse", "HEAD")

    def check(clean, revision=pinned):
        path = tmp_path / f"app-{clean}.toml"
        path.write_text(f'name = "app"\nsources = ["in:a.f90"]\nprogram = "in:a.f90"\n[variants.default]\n'
                        f'[inputs.in]\nrevision = "{revision}"\nclean = {str(clean).lower()}\n')
        return snapshot_inputs(load(path), {"in": checkout}, tmp_path / "snapshot", ["in"])

    check(clean=True)
    (checkout / "a.f90").write_text("! a local change\n")
    git("commit", "-qam", "local")
    check(clean=False)
    with pytest.raises(ValueError, match="clean checkout"):
        check(clean=True)
    with pytest.raises(ValueError, match="based on"):
        check(clean=False, revision="0" * 40)


@pytest.mark.parametrize("description", sorted((ROOT / "benchmarks").glob("*/*.toml")), ids=lambda path: path.parent.name)
def test_every_example_description_reads(description):
    application = load(description)
    assert application.program in application.sources
    assert application.variants


@pytest.mark.skipif(fortran_compiler() is None, reason="GNU Fortran is required for preprocessing")
def test_the_kernel_example_prepares_its_entries_and_keeps_the_fields_on_the_device(tmp_path):
    application = load(ROOT / "benchmarks/cdu/cdu.toml")
    prepared = prepare(application, {}, tmp_path / "build", variant="default", fortran=fortran_compiler())
    summary, = json.loads((tmp_path / "build/resident/prepared.json").read_text())["selections"]
    assert summary["nests"] >= 4
    assert not summary["rejected"]
    main = next(Path(row["resident"]) for row in prepared["sources"] if row["original"].endswith("main.f90"))
    # The edits' directives became the domain's calls around the warm-up and the timed loop.
    text = main.read_text().lower()
    assert "begin_epoch" in text
    assert "end_epoch" in text
    assert "!$fort" not in text
