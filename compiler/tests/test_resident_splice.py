"""Compiled resident entries are called from the source spans they replace."""
import json
from hashlib import sha256

import pytest

from compiler.scopes.resident_splice import ResidentDomain, addressable_arrays, splice_resident_entries

DOMAIN = ResidentDomain("flow_state", "flow", "require")

SOURCE = """subroutine update(u, mask, n)
  u=1
  mask=0
  n=n+1
end subroutine
"""


def package(tmp_path, *, rejected=()):
    source = tmp_path / "example.f90"
    source.write_text(SOURCE)
    region = {
        "source": str(source), "first_line": 2, "last_line": 2, "procedure": "example::update",
        "resources": [
            {"resource": "argument::u", "name": "u", "type": "real", "kind": 8},
            {"resource": "argument::mask", "name": "mask", "type": "integer", "kind": 4},
            {"resource": "argument::n", "name": "n", "type": "integer", "kind": 4},
        ],
        "parameters": [
            {"resource": "argument::u", "name": "u", "rank": 3, "lower_bound_dimension": None},
            {"resource": "argument::u", "name": "u_lb_2", "rank": 0, "lower_bound_dimension": 2},
            {"resource": "argument::mask", "name": "mask", "rank": 3, "lower_bound_dimension": None},
            {"resource": "argument::n", "name": "n", "rank": 0, "lower_bound_dimension": None},
        ],
        "runtime_guards": ["n > 2"],
    }
    entry = {"fortran_module": "fort_example", "argument_order": ["context", "mode", "u", "u_lb_2", "mask", "n"]}
    return source, {"source_inputs": {str(source): sha256(source.read_bytes()).hexdigest()},
                    "entries": [{"region": region, "entry": entry}], "rejected": list(rejected)}


def test_a_compiled_span_becomes_a_guarded_entry_call_with_the_original_as_fallback(tmp_path):
    source, compiled = package(tmp_path)
    result = splice_resident_entries([compiled], tmp_path / "generated", domain=DOMAIN)
    assert json.loads((tmp_path / "generated/source-edits.json").read_text()) == result
    record, = result["sources"]
    assert record["original"]["path"] == str(source)
    assert record["region_count"] == 1
    generated = (tmp_path / "generated/example.f90").read_text()
    lines = generated.splitlines()
    # Everything outside the span is untouched.
    assert lines[0] == "subroutine update(u, mask, n)"
    assert lines[-3:] == ["  mask=0", "  n=n+1", "end subroutine"]
    assert "      use flow_state, only: flow, require" in lines
    assert "      use fort_example, only: fort_device_run => run" in lines
    assert "      integer(c_int64_t) :: fort_handle_u,fort_handle_mask" in lines
    assert "        if (.not. (n > 2)) error stop 'resident numerical geometry changed'" in lines
    assert "fort_handle_u=flow%bind_real64(u,lbound(u),.true., &\n          origin='example::update::u',tolerant=.true.)" in generated
    assert "fort_handle_mask=flow%bind_integer32(mask,lbound(mask),.true., &\n          origin='example::update::mask',tolerant=.true.)" in generated
    # Arrays pass as handles, lower bounds as inquiries and scalars by name, in the entry's order.
    call = generated.split("call require(fort_device_run( &\n", 1)[1].split("))", 1)[0]
    assert [argument.strip() for argument in call.split(", &\n")] == [
        "flow%context", "FORT_SCOPE_GPU", "fort_handle_u", "lbound(u,2)", "fort_handle_mask", "n"]
    # The original statement runs where the domain is not active, and after a publication where an array
    # could not be kept as it is bound here.
    active = lines.index("      fort_resident = flow%active")
    assert lines[lines.index("      else", active) + 1:][:2] == ["        if (flow%active) call flow%publish()", "  u=1"]


def test_a_rejected_span_publishes_first_unless_the_caller_supplies_its_text(tmp_path):
    source = str(tmp_path / "example.f90")
    rejected = [{"procedure": "example::update", "source": source, "first_line": line, "last_line": line,
                 "reason": "unsupported"} for line in (3, 4)]
    _source, compiled = package(tmp_path, rejected=rejected)
    seen = []

    def native(region, original):
        seen.append((region["first_line"], original))
        return "  ! kept by the caller\n" + original if region["first_line"] == 4 else None

    result = splice_resident_entries([compiled], tmp_path / "generated", domain=DOMAIN, native=native)
    assert result["sources"][0]["region_count"] == 3
    assert sorted(seen) == [(3, "  mask=0\n"), (4, "  n=n+1\n")]
    generated = (tmp_path / "generated/example.f90").read_text()
    assert ("    block\n      use flow_state, only: flow\n      if (flow%active) call flow%publish()\n"
            "  mask=0\n    end block\n") in generated
    assert "  ! kept by the caller\n  n=n+1\nend subroutine\n" in generated
    assert generated.count("publish()") == 2


def test_changed_sources_overlaps_and_unknown_array_types_are_refused(tmp_path):
    source, compiled = package(tmp_path)
    overlapping = {**compiled, "rejected": [{"procedure": "example::update", "source": str(source),
                                              "first_line": 2, "last_line": 3, "reason": "unsupported"}]}
    with pytest.raises(ValueError, match="Overlapping"):
        splice_resident_entries([overlapping], tmp_path / "overlap", domain=DOMAIN)
    compiled["entries"][0]["region"]["resources"][0]["kind"] = 16
    with pytest.raises(ValueError, match="Unsupported resident array ABI"):
        splice_resident_entries([compiled], tmp_path / "abi", domain=DOMAIN)
    compiled["entries"][0]["region"]["resources"][0]["kind"] = 8
    source.write_text(SOURCE + "! edited\n")
    with pytest.raises(ValueError, match="source changed"):
        splice_resident_entries([compiled], tmp_path / "changed", domain=DOMAIN)


def test_bindable_arrays_get_target_and_everything_else_is_left_alone():
    text = """module fields
  real(8), allocatable :: u(:,:,:)
  real(8), dimension(4) :: weights
  integer :: count
  real(8), parameter :: table(2) = [1.0d0, 2.0d0]
  real(8), pointer :: view(:)
  type box
    real(8) :: corners(8)
  end type
contains
  subroutine step(a, n)
    real(8), intent(inout) :: a(:)
    integer, intent(in) :: n
    real(8) :: local(3)
    real(8), allocatable :: work(:)
  end subroutine
end module
"""
    changed = addressable_arrays(text)
    for declared in ("REAL(KIND = 8), ALLOCATABLE, TARGET :: u(:, :, :)", "REAL(KIND = 8), DIMENSION(4), TARGET :: weights",
                     "REAL(KIND = 8), INTENT(INOUT), TARGET :: a(:)", "REAL(KIND = 8), ALLOCATABLE, TARGET :: work(:)"):
        assert declared in changed, declared
    for untouched in ("  integer :: count\n", "  real(8), parameter :: table(2) = [1.0d0, 2.0d0]\n",
                      "  real(8), pointer :: view(:)\n", "    real(8) :: corners(8)\n",
                      "    integer, intent(in) :: n\n", "    real(8) :: local(3)\n"):
        assert untouched in changed, untouched
    assert addressable_arrays(changed) == changed


def test_an_array_the_domain_already_holds_is_passed_by_its_handle(tmp_path):
    from compiler.scopes.resident_splice import entry_call_block

    region = {"procedure": "example::update", "runtime_guards": [],
              "resources": [{"resource": "r_u", "name": "u", "type": "real", "kind": 8},
                            {"resource": "r_w", "name": "Work", "type": "real", "kind": 8}],
              "parameters": [{"name": "u", "resource": "r_u", "rank": 1, "lower_bound_dimension": None},
                             {"name": "work", "resource": "r_w", "rank": 1, "lower_bound_dimension": None}]}
    entry = {"fortran_module": "fort_shared_x", "argument_order": ["context", "mode", "u", "work"]}
    domain = ResidentDomain("state", "flow", "require")
    block = entry_call_block(region, entry, domain, {("example::update", "work"): "flow%scratch"})
    assert "bind_real64(u,lbound(u)" in block
    assert "bind_real64(Work" not in block
    assert "      integer(c_int64_t) :: fort_handle_u\n" in block
    assert "          fort_handle_u, &\n          flow%scratch))" in block
    # With every array held, nothing is bound and no handle variable is declared.
    held = {("example::update", "work"): "flow%scratch", ("example::update", "u"): "flow%field"}
    assert "integer(c_int64_t)" not in entry_call_block(region, entry, domain, held)
