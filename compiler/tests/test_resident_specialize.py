"""A procedure gets a second body with the branches that assumed conditions decide folded away."""
import pytest

from compiler.resident.specialize import integer_parameters, specialize

PARAMETERS = """module kinds
  integer, parameter :: WALL = 1, PERIODIC = 3, &  ! boundary kinds
                        INLET = 4
  integer, parameter :: west = 1, east = 2
  integer, parameter :: shifted = WALL + 1
  real, parameter :: half = 0.5
end module
module other
  integer, parameter :: INLET = 4, east = 7
end module
"""

SOURCE = """module faces
  use kinds
  implicit none
  integer :: bc(2), mode
contains
  subroutine apply(u, n)
    real, intent(inout) :: u(0:)
    integer, intent(in) :: n
    integer :: i
    ! West.
    if (bc(west)==INLET .and. mode/=2) then
      u(0) = 1
    else if (bc(west)==WALL .or. (bc(west)==INLET .and. mode==2)) then
      u(0) = -u(1)
    else if (bc(west)==PERIODIC) then
      do i = 0, 0
        u(i) = u(n)
      end do
    else
      u(0) = u(1)
    end if
    if (bc(east)==PERIODIC) then
      do i = n+1, n+1
        u(i) = u(1)
      end do
    end if
    if (mode==2) then
      u(1) = 0
    else if (bc(west)/=PERIODIC) then
      u(1) = 2
    else
      u(1) = 3
    end if
    if (.not. bc(west)==PERIODIC) u(2) = 0
  end subroutine
end module
"""


def test_named_integer_constants_are_collected_unless_their_value_is_ambiguous():
    constants = integer_parameters([PARAMETERS])
    assert constants == {"wall": 1, "periodic": 3, "inlet": 4, "west": 1}
    # east has two values, shifted is not a literal, half is not an integer.


def test_decided_branches_are_folded_in_a_copy_of_the_body():
    constants = integer_parameters([PARAMETERS])
    result = specialize(SOURCE, "apply", ("bc(west) == PERIODIC", "bc(east) == PERIODIC"), constants)
    head, rest = result.split("    if ((bc(west) == PERIODIC) .and. &\n        (bc(east) == PERIODIC)) then\n")
    folded, original = rest.split("    else\n    if (bc(west)==INLET .and. mode/=2) then\n", 1)
    # Declarations and the comment before the first statement come first, unchanged; the original body follows.
    first = "    if (bc(west)==INLET .and. mode/=2) then\n"
    assert head == SOURCE.split(first)[0]
    assert original == SOURCE.split(first)[1].split("  end subroutine")[0] + "    end if\n  end subroutine\nend module\n"
    # The two decided chains leave their periodic nests next to each other.
    assert folded.startswith("      do i = 0, 0\n        u(i) = u(n)\n      end do\n"
                             "      do i = n+1, n+1\n        u(i) = u(1)\n      end do\n")
    # An undecided branch stays, a decided false one goes, and a decided true one becomes the ELSE.
    assert "    if (mode==2) then\n      u(1) = 0\n    else\n      u(1) = 3\n    end if\n" in folded
    assert "u(1) = 2" not in folded
    # A one-line IF is left as it is.
    assert folded.rstrip().endswith("if (.not. bc(west)==PERIODIC) u(2) = 0")


def test_an_else_if_that_becomes_the_first_branch_is_an_if():
    constants = integer_parameters([PARAMETERS])
    result = specialize(SOURCE, "apply", ("bc(west) == WALL",), constants)
    folded = result.split("    if ((bc(west) == WALL)) then\n")[1].split(
        "    else\n    if (bc(west)==INLET .and. mode/=2) then\n")[0]
    # The inlet branch needs another kind and goes; the wall branch after it is decided and replaces the chain.
    assert folded.startswith("      u(0) = -u(1)\n    if (bc(east)==PERIODIC) then\n")
    # In the third chain the first branch is undecided, so the decided one after it becomes its ELSE.
    assert "    if (mode==2) then\n      u(1) = 0\n    else\n      u(1) = 2\n    end if\n" in folded
    # With the first branch dropped and the second undecided, an ELSE IF would have to open the construct.
    result = specialize(SOURCE.replace("mode/=2) then", "mode==9) then"), "apply", ("mode == 2",),
                        integer_parameters([PARAMETERS]))
    folded = result.split("    if ((mode == 2)) then\n")[1].split("    else\n    if (bc(west)==INLET")[0]
    assert folded.startswith("    if (bc(west)==WALL .or. (bc(west)==INLET .and. mode==2)) then\n")


def test_an_order_comparison_of_known_values_is_decided_too():
    # ELMM tells its ordinary boundary kinds from the parallel ones by a range.
    source = SOURCE.replace("    else if (bc(west)==PERIODIC) then\n", "    else if (bc(west) < INLET .or. bc(west) >= 9) then\n")
    constants = integer_parameters([PARAMETERS])
    folded = specialize(source, "apply", ("bc(west) == PERIODIC",), constants).split("    else\n    if (bc(west)==INLET")[0]
    assert "      do i = 0, 0\n        u(i) = u(n)\n      end do\n    if (bc(east)==PERIODIC) then\n" in folded
    folded = specialize(source.replace("< INLET", "> INLET"), "apply", ("bc(west) == PERIODIC",),
                        constants).split("    else\n    if (bc(west)==INLET")[0]
    assert "      u(0) = u(1)\n    if (bc(east)==PERIODIC) then\n" in folded
    assert "u(i) = u(n)" not in folded


@pytest.mark.parametrize(("conditions", "message"), [
    (("mode == 2 .or. mode == 3",), "has the form <expression> == <integer constant>"),
    (("bc(west) == east",), "no known integer value"),
    (("bc(west) == WALL", "bc(west) == PERIODIC"), "contradict each other"),
    (("n == 5",), "decide no branch of apply"),
])
def test_conditions_that_cannot_be_used_are_refused(conditions, message):
    with pytest.raises(ValueError, match=message):
        specialize(SOURCE, "apply", conditions, integer_parameters([PARAMETERS]))


def test_a_body_that_cannot_be_written_twice_is_refused():
    named = SOURCE.replace("      do i = 0, 0\n", "      halo: do i = 0, 0\n").replace(
        "        u(i) = u(n)\n      end do\n", "        u(i) = u(n)\n      end do halo\n")
    with pytest.raises(ValueError, match="cannot be written twice"):
        specialize(named, "apply", ("bc(west) == PERIODIC",), integer_parameters([PARAMETERS]))
    with pytest.raises(ValueError, match="Expected one procedure missing"):
        specialize(SOURCE, "missing", ("bc(west) == PERIODIC",), integer_parameters([PARAMETERS]))


@pytest.mark.parametrize("statement", ["bc(west) = WALL", "call reset(bc)", "mode = 2"])
def test_a_body_that_may_change_what_a_condition_tests_is_refused(statement):
    changing = SOURCE.replace("      u(0) = 1\n", "      u(0) = 1\n      " + statement + "\n")
    conditions = ("mode == 1",) if statement.startswith("mode") else ("bc(west) == PERIODIC",)
    with pytest.raises(ValueError, match="may change what an assumed condition tests"):
        specialize(changing, "apply", conditions, integer_parameters([PARAMETERS]))
