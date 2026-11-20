"""A loop nest whose DO variables are read after it is still an entry; the host recomputes the variables."""
from hashlib import sha256

from compiler.scopes.resident_entries import export_resident_entries
from compiler.scopes.resident_splice import ResidentDomain, splice_resident_entries

HEAD = """module halo
contains
subroutine faces(a, n, m, last)
real(8), intent(inout) :: a(0:,0:,0:)
integer, intent(in) :: n, m
integer, intent(out) :: last
integer :: i, j, k
real(8) :: t
"""
TAIL = """
end subroutine
end module
"""


def export(tmp_path, body):
    source = tmp_path / "halo.f90"
    source.write_text(HEAD + body + TAIL)
    return source, export_resident_entries([source], ["halo::faces"], tmp_path / "entries")


def test_the_loops_are_kept_without_their_statements_for_the_variables_read_later(tmp_path):
    # ELMM's Bound_Phi has such a read: a later nest uses a variable that only an earlier nest sets.
    _source, result = export(tmp_path, """do k = 1, m
  do j = 1, n
    a(0,j,k) = a(n,j,k)
  end do
end do
last = j + k
""")
    assert not result["rejected"]
    entry, = result["entries"]
    assert entry["final_iterators"] == ["do k = 1, m", "  do j = 1, n", "  end do", "end do"]


def test_only_the_loops_that_hold_such_a_variable_are_kept(tmp_path):
    _source, result = export(tmp_path, """do k = 1, m
  do j = 1, n
    do i = 1, 2
      a(i,j,k) = a(n+i,j,k)
    end do
  end do
end do
last = k
""")
    entry, = result["entries"]
    assert entry["final_iterators"] == ["do k = 1, m", "end do"]


def test_a_nest_whose_variables_are_not_read_later_keeps_nothing(tmp_path):
    _source, result = export(tmp_path, """do k = 1, m
  do j = 1, n
    a(0,j,k) = a(n,j,k)
  end do
end do
last = n
""")
    entry, = result["entries"]
    assert entry["final_iterators"] == []


def test_a_value_the_statements_compute_cannot_be_recomputed(tmp_path):
    for number, body in enumerate((
            # The last iteration's value of an ordinary scalar.
            "do k = 1, m\n  t = a(1,1,k)\n  a(0,1,k) = t\nend do\nlast = int(t)\n",
            # A loop that runs only when the statements say so.
            "do k = 1, m\n  if (a(1,1,k) > 0) then\n    do j = 1, n\n      a(0,j,k) = a(n,j,k)\n    end do\n  end if\n"
            "end do\nlast = j\n")):
        directory = tmp_path / str(number)
        directory.mkdir()
        _source, result = export(directory, body)
        assert not result["entries"]
        rejected, = result["rejected"]
        assert rejected["reason"] == "inline loop-written scalar is live after the region"


def test_the_host_runs_the_kept_loops_after_the_entry(tmp_path):
    source, result = export(tmp_path, """do k = 1, m
  do j = 1, n
    a(0,j,k) = a(n,j,k)
  end do
end do
last = j + k
""")
    region, = result["entries"]
    package = {"source_inputs": {str(source): sha256(source.read_bytes()).hexdigest()}, "rejected": [],
               "entries": [{"region": region,
                            "entry": {"fortran_module": "fort_halo",
                                      "argument_order": ["context", "mode", *(item["name"] for item in region["parameters"])]}}]}
    splice_resident_entries([package], tmp_path / "spliced", domain=ResidentDomain("state", "flow", "require"))
    text = (tmp_path / "spliced/halo.f90").read_text()
    active, native = text.split("      else\n", 1)
    # After the call, in the branch that did not run the statements.
    assert active.split("call require(fort_device_run(", 1)[1].split("))\n", 1)[1] == (
        "        do k = 1, m\n          do j = 1, n\n          end do\n        end do\n")
    assert "a(0,j,k) = a(n,j,k)" in native
    assert "last = j + k" in native.split("end block", 1)[1]
