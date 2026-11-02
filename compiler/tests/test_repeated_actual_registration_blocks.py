"""Cloned registration markers retain per-occurrence Fortran construct names."""
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from compiler.emission.common.resources import read_scoped_runtime
from compiler.frontend.source_effects import SourceEffects
from compiler.scopes.call_registration import OriginalActual, finish_original_registrations, register_original_actuals
from compiler.scopes.source import _name

SOURCE = """module registration_fixture
contains
subroutine original_helper(a,b,n)
integer,intent(in)::n
integer,intent(inout)::a(-2:)
real(8),intent(inout)::b(-4:,2:)
end subroutine
subroutine owner(a,b,n)
integer,intent(in)::n
integer,intent(inout)::a(-2:)
real(8),intent(inout)::b(-4:,2:)
call original_helper(a,b,n)
end subroutine
end module
"""


def rendered(tmp_path, monkeypatch, repeats):
    from compiler.scopes import provenance

    path = tmp_path / "original.f90"
    path.write_text(SOURCE)
    analysis = SourceEffects([path])
    routine = analysis.routines["registration_fixture::owner"]
    monkeypatch.setattr(provenance, "position", lambda *_args, **_kwargs: [])
    builder = SimpleNamespace(entry=routine, config=SimpleNamespace(scope_execution="graph"),
        provider_only_lifetime=None, device_budget=4096,
        capture_many=lambda bindings: [{"storage": "stable", "initialized": "none", "escapes": False,
                                      "allocation_changes": False,
                                      "requires_registered_handle": binding.name == "a"} for binding in bindings],
        owner_transfer_setup=lambda *_args, **_kwargs: [])

    class Owner:
        patches = []
        context = "fort_context"
        enabled = "fort_enabled"
        epoch_pristine = "fort_epoch_pristine"
        leaves = ()

        def __init__(self):
            self.owner, self.routine, self.builder = self, routine, builder
            self.resources, self.scalar_element_resources = {}, {}

        def declaration(self, _binding):
            pass

        def add_resource(self, binding):
            self.resources.setdefault(binding.root, (binding, len(self.resources) + 1))

        def registration_identity(self, number):
            return f"{number}_c_int64_t"

        def creation_trace_owner(self):
            return 1

        def close(self, **_kwargs):
            return ["fort_enabled = .false.", "fort_context = 0_c_int64_t"]

    owner = Owner()
    first, = register_original_actuals(owner, (OriginalActual(routine.scope.bindings["a"], "a"),))
    second, = register_original_actuals(owner, (OriginalActual(routine.scope.bindings["b"], "b"),))
    # Atomic predicate and native/fallback rendering can clone an admitted
    # deferred marker; these are occurrences of the SAME original registration.
    arms = []
    for index in range(repeats):
        arms += ["if (first) then" if index == 0 else "else if (second) then",
                 first, "if (second) then", second, "endif", "call original_helper(a,b,n)"]
    arms += ["endif"]
    text = "\n".join(arms)
    result = finish_original_registrations(owner, text)
    assert first not in result
    assert second not in result
    return result, (first, second)


@pytest.mark.parametrize("repeats", [1, 2, 4])
def test_each_cloned_marker_has_matching_unique_construct_and_exit_targets(tmp_path, monkeypatch, repeats):
    text, markers = rendered(tmp_path, monkeypatch, repeats)
    names = re.findall(r"^(fort_actuals_\w+): block$", text, re.M)
    endings = re.findall(r"^end block (fort_actuals_\w+)$", text, re.M)
    exits = re.findall(r"^exit (fort_actuals_\w+)$", text, re.M)
    assert len(names) == 2 * repeats
    assert len(set(names)) == len(names)
    assert endings == names
    assert exits
    assert set(exits) <= set(names)
    assert text.count("call original_helper(a,b,n)") == repeats
    if repeats == 1:
        assert names == [_name("fort_actuals_", marker[1:]) for marker in markers]
    reference, _markers = rendered(tmp_path, monkeypatch, 1)
    for name in names:
        body = text.split(name + ": block\n", 1)[1].split("end block " + name, 1)[0]
        base = next(_name("fort_actuals_", marker[1:]) for marker in markers
                    if name.startswith(_name("fort_actuals_", marker[1:])))
        original_body = reference.split(base + ": block\n", 1)[1].split("end block " + base, 1)[0]
        assert body.replace("exit " + name, "exit " + base) == original_body
    assert rendered(tmp_path, monkeypatch, repeats)[0] == text


@pytest.mark.native
@pytest.mark.parametrize("repeats", [1, 4])
def test_cloned_registration_control_arms_compile_with_original_actual_ranks(tmp_path, monkeypatch, repeats):
    text, _markers = rendered(tmp_path, monkeypatch, repeats)
    compiler = shutil.which("gfortran-15")
    assert compiler is not None
    runtime, _manifest = read_scoped_runtime()
    (tmp_path / "fort_scoped_memory.f90").write_text(runtime["fort_scoped_memory.f90"])
    source = """module registration_fixture
use iso_c_binding
implicit none
contains
subroutine original_helper(a,b,n)
integer,intent(in)::n
integer(c_int),intent(inout)::a(-2:)
real(c_double),intent(inout)::b(-4:,2:)
end subroutine
subroutine owner(a,b,n,first,second)
integer,intent(in)::n
logical,intent(in)::first,second
integer(c_int),target,intent(inout)::a(-2:)
real(c_double),target,intent(inout)::b(-4:,2:)
logical::fort_enabled,fort_epoch_pristine
integer(c_int64_t)::fort_context,fort_buffer_1,fort_buffer_2
fort_enabled=.true.
fort_epoch_pristine=.true.
fort_context=0
fort_buffer_1=0
fort_buffer_2=0
""" + text + "\nend subroutine\nend module\n"
    (tmp_path / "generated.f90").write_text(source)
    for name in ("fort_scoped_memory.f90", "generated.f90"):
        result = subprocess.run([compiler, "-fopenmp", "-ffree-line-length-none", "-fcheck=all",
            "-J", str(tmp_path), "-I", str(tmp_path), "-c", str(tmp_path / name),
            "-o", str(tmp_path / (name + ".o"))], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
