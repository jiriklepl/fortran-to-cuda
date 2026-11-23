"""An application described in TOML: its edits, the checks of a description, and the examples under benchmarks/."""
import json
import subprocess
from pathlib import Path

import pytest

from compiler.resident.application import Edit, apply_edit, load, prepare, snapshot_inputs
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
