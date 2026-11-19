"""Residency directives become calls on the application's domain; malformed ones are refused."""
import pytest

from compiler.resident.directives import expand_directives
from compiler.scopes.resident_splice import ResidentDomain

DOMAIN = ResidentDomain("my_state", "flow", "require", prefix="app")


def expand(line):
    text, records = expand_directives("  x = 1\n" + line + "\n  y = 2\n", DOMAIN, origin="unit")
    assert text.startswith("  x = 1\n")
    assert text.endswith("  y = 2\n")
    return text[len("  x = 1\n"):-len("  y = 2\n")], records


def test_text_without_directives_is_unchanged():
    source = "  !$omp parallel do\n  ! $fort is not a directive\n  a = '!$fort publish'\n"
    assert expand_directives(source, DOMAIN) == (source, [])


def test_enter_starts_an_epoch_once_under_its_condition():
    block, records = expand("    !$FORT enter budget(2*1024**3) if(supported() .and. (n > 1)) device(gpu)")
    assert block == ("    block\n"
                     "      use, intrinsic :: iso_c_binding\n"
                     "      use my_state, only: flow\n"
                     "      if (.not. flow%active) then\n"
                     "        if (supported() .and. (n > 1)) call flow%begin_epoch(int(gpu, c_int), "
                     "int(2*1024**3, c_size_t))\n"
                     "      endif\n"
                     "    end block\n")
    assert records == [{"line": 2, "directive": "enter", "arrays": [],
                        "clauses": {"budget": "2*1024**3", "if": "supported() .and. (n > 1)", "device": "gpu"}}]
    unconditional, _ = expand("!$fort enter budget(1024)")
    assert "  if (.not. flow%active) then\n    call flow%begin_epoch(int(0, c_int), int(1024, c_size_t))\n" in unconditional


def test_leave_close_and_publish_act_on_the_whole_domain():
    assert "if (flow%active) call flow%end_epoch()" in expand("!$fort leave")[0]
    # Closing is unconditional: it is also valid when the domain was never entered.
    assert "  call flow%close()\n" in expand("!$fort close")[0]
    assert "if (flow%active) call flow%publish()" in expand("!$fort publish")[0]
    assert "if (flow%active .and. (count > 0)) call flow%publish()" in expand("!$fort publish if(count > 0)")[0]


def test_observe_binds_the_named_arrays_and_keeps_their_device_copies():
    block, records = expand("  !$fort observe(U, state%pressure) if(mode == 2)")
    assert block == ("  block\n"
                     "    use, intrinsic :: iso_c_binding\n"
                     "    use my_state, only: flow\n"
                     "    integer(c_int64_t) :: app_observed(2)\n"
                     "    if (flow%active .and. (mode == 2)) then\n"
                     "      app_observed(1) = flow%bind(U, lbound(U), .true., origin='unit::u')\n"
                     "      app_observed(2) = flow%bind(state%pressure, lbound(state%pressure), .true., "
                     "origin='unit::state%pressure')\n"
                     "      call flow%observe(app_observed)\n"
                     "    endif\n"
                     "  end block\n")
    assert records[0]["arrays"] == ["U", "state%pressure"]


def test_observe_brings_one_field_of_an_array_of_derived_type_back():
    block, records = expand("!$fort observe(points(:)%seen, u) if(ready)")
    assert block == ("block\n"
                     "  use, intrinsic :: iso_c_binding\n"
                     "  use my_state, only: flow\n"
                     "  integer(c_int64_t) :: app_observed(1)\n"
                     "  if (flow%active .and. (ready)) then\n"
                     "    app_observed(1) = flow%bind(u, lbound(u), .true., origin='unit::u')\n"
                     "    call flow%observe(app_observed)\n"
                     "    if (size(points) > 0) call flow%observe_field(c_loc(points(lbound(points,1))%seen))\n"
                     "  endif\n"
                     "end block\n")
    assert records[0]["arrays"] == ["points(:)%seen", "u"]
    only_field, _ = expand("!$fort observe(points(:)%seen)")
    assert "app_observed" not in only_field
    with pytest.raises(ValueError, match="takes array names, not expressions"):
        expand_directives("!$fort release(points(:)%seen)\n", DOMAIN)


def test_release_retires_temporaries_without_bringing_them_back():
    block, _ = expand("!$fort release(tmp, work)")
    assert "if (flow%active) then\n    call flow%release_real(tmp)\n    call flow%release_real(work)\n  endif" in block


@pytest.mark.parametrize(("line", "message"), [
    ("!$fort", "Empty residency directive on line 2"),
    ("!$fort sync", "Unknown residency directive on line 2: sync"),
    ("!$fort enter", "needs budget"),
    ("!$fort enter budget(1) when(x)", "Unsupported clause"),
    ("!$fort enter budget(1) if(a", "Unbalanced parentheses"),
    ("!$fort publish if", "Unsupported clause"),
    ("!$fort leave(a)", "takes no arguments"),
    ("!$fort observe", "needs a list of arrays"),
    ("!$fort observe(a(1:n))", "takes array names, not expressions"),
    ("!$fort release(a+b)", "takes array names, not expressions"),
    ("!$fort publish, if(x)", "Malformed residency directive"),
])
def test_malformed_directives_are_refused_with_their_line(line, message):
    with pytest.raises(ValueError, match=message):
        expand_directives("  x = 1\n" + line + "\n", DOMAIN)
