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
  integer :: kind(2), mode
contains
  subroutine apply(u, n)
    real, intent(inout) :: u(0:)
    integer, intent(in) :: n
    integer :: i
    ! West.
    if (kind(west)==INLET .and. mode/=2) then
      u(0) = 1
    else if (kind(west)==WALL .or. (kind(west)==INLET .and. mode==2)) then
      u(0) = -u(1)
    else if (kind(west)==PERIODIC) then
      do i = 0, 0
        u(i) = u(n)
      end do
    else
      u(0) = u(1)
    end if
    if (kind(east)==PERIODIC) then
      do i = n+1, n+1
        u(i) = u(1)
      end do
    end if
    if (mode==2) then
      u(1) = 0
    else if (kind(west)/=PERIODIC) then
      u(1) = 2
    else
      u(1) = 3
    end if
    if (.not. kind(west)==PERIODIC) u(2) = 0
  end subroutine
end module
"""


def test_named_integer_constants_are_collected_unless_their_value_is_ambiguous():
    constants = integer_parameters([PARAMETERS])
    assert constants == {"wall": 1, "periodic": 3, "inlet": 4, "west": 1}
    # east has two values, shifted is not a literal, half is not an integer.


def test_decided_branches_are_folded_in_a_copy_of_the_body():
    constants = integer_parameters([PARAMETERS])
    result = specialize(SOURCE, "apply", ("kind(west) == PERIODIC", "kind(east) == PERIODIC"), constants)
    head, rest = result.split("    if ((kind(west) == PERIODIC) .and. &\n        (kind(east) == PERIODIC)) then\n")
    folded, original = rest.split("    else\n    if (kind(west)==INLET .and. mode/=2) then\n", 1)
    # Declarations and the comment before the first statement come first, unchanged; the original body follows.
    first = "    if (kind(west)==INLET .and. mode/=2) then\n"
    assert head == SOURCE.split(first)[0]
    assert original == SOURCE.split(first)[1].split("  end subroutine")[0] + "    end if\n  end subroutine\nend module\n"
    # The two decided chains leave their periodic nests next to each other.
    assert folded.startswith("      do i = 0, 0\n        u(i) = u(n)\n      end do\n"
                             "      do i = n+1, n+1\n        u(i) = u(1)\n      end do\n")
    # An undecided branch stays, a decided false one goes, and a decided true one becomes the ELSE.
    assert "    if (mode==2) then\n      u(1) = 0\n    else\n      u(1) = 3\n    end if\n" in folded
    assert "u(1) = 2" not in folded
    # A one-line IF is left as it is.
    assert folded.rstrip().endswith("if (.not. kind(west)==PERIODIC) u(2) = 0")


def test_an_else_if_that_becomes_the_first_branch_is_an_if():
    constants = integer_parameters([PARAMETERS])
    result = specialize(SOURCE, "apply", ("kind(west) == WALL",), constants)
    folded = result.split("    if ((kind(west) == WALL)) then\n")[1].split(
        "    else\n    if (kind(west)==INLET .and. mode/=2) then\n")[0]
    # The inlet branch needs another kind and goes; the wall branch after it is decided and replaces the chain.
    assert folded.startswith("      u(0) = -u(1)\n    if (kind(east)==PERIODIC) then\n")
    # In the third chain the first branch is undecided, so the decided one after it becomes its ELSE.
    assert "    if (mode==2) then\n      u(1) = 0\n    else\n      u(1) = 2\n    end if\n" in folded
    # With the first branch dropped and the second undecided, an ELSE IF would have to open the construct.
    result = specialize(SOURCE.replace("mode/=2) then", "mode==9) then"), "apply", ("mode == 2",),
                        integer_parameters([PARAMETERS]))
    folded = result.split("    if ((mode == 2)) then\n")[1].split("    else\n    if (kind(west)==INLET")[0]
    assert folded.startswith("    if (kind(west)==WALL .or. (kind(west)==INLET .and. mode==2)) then\n")


@pytest.mark.parametrize(("conditions", "message"), [
    (("mode == 2 .or. mode == 3",), "has the form <expression> == <integer constant>"),
    (("kind(west) == east",), "no known integer value"),
    (("kind(west) == WALL", "kind(west) == PERIODIC"), "contradict each other"),
    (("n == 5",), "decide no branch of apply"),
])
def test_conditions_that_cannot_be_used_are_refused(conditions, message):
    with pytest.raises(ValueError, match=message):
        specialize(SOURCE, "apply", conditions, integer_parameters([PARAMETERS]))


def test_a_body_that_cannot_be_written_twice_is_refused():
    named = SOURCE.replace("      do i = 0, 0\n", "      halo: do i = 0, 0\n").replace(
        "        u(i) = u(n)\n      end do\n", "        u(i) = u(n)\n      end do halo\n")
    with pytest.raises(ValueError, match="cannot be written twice"):
        specialize(named, "apply", ("kind(west) == PERIODIC",), integer_parameters([PARAMETERS]))
    with pytest.raises(ValueError, match="Expected one procedure missing"):
        specialize(SOURCE, "missing", ("kind(west) == PERIODIC",), integer_parameters([PARAMETERS]))
