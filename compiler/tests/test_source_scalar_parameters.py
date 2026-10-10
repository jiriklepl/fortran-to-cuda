"""Unsupported immutable scalar initializers keep their native Fortran value."""

from __future__ import annotations

import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.source_effects import SourceEffects, _children, _kind
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.regions import extract_region
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT


def fixture(tmp_path, precision=8, *, helper_local=False, visible=True, integer=False):
    constants = tmp_path / "constants.f90"
    constant_type = "integer" if integer else f"real({precision})"
    initializer = f"int(acos(-1._{precision}))" if integer else f"acos(-1._{precision})"
    constants.write_text(f"""module renamed_values
implicit none
{constant_type},parameter::angle={initializer}
end module
""")
    helpers = tmp_path / "helpers.f90"
    helpers.write_text(f"""module renamed_helpers
{'use renamed_values,only:helper_angle=>angle' if not helper_local else ''}
implicit none
contains
pure real({precision}) function weight(x)
real({precision}),intent(in)::x
{constant_type + ',parameter::helper_angle=' + initializer if helper_local else ''}
weight=x*helper_angle
end function
end module
""")
    source = tmp_path / "caller.f90"
    source.write_text(f"""module renamed_owner
use renamed_helpers,only:evaluate=>weight
{'use renamed_values,only:original_value=>angle' if visible else ''}
implicit none
contains
subroutine advance(a,out,n)
real({precision}),intent(in)::a(-2:)
real({precision}),intent(inout)::out(-2:)
integer,intent(in)::n
integer::i
do i=1,n
out(i)=local_value(a(i))
enddo
contains
pure real({precision}) function local_value(x)
real({precision}),intent(in)::x
local_value=evaluate(x)
end function
end subroutine
end module
""")
    paths = constants, helpers, source
    analysis = SourceEffects(paths)
    routine = analysis.routines["renamed_owner::advance"]
    loop = next(node for node in _children(routine.execution) if _kind(node) == "Block_Nonlabel_Do_Construct")
    region = extract_region(analysis, routine, loop)
    return analysis, routine, region, paths


@pytest.mark.parametrize("precision", [4, 8])
@pytest.mark.parametrize("integer", [False, True])
def test_native_scalar_parameter_is_captured_through_renamed_lexical_helper(tmp_path, precision, integer):
    _, _, region, _ = fixture(tmp_path, precision, integer=integer)
    root = "renamed_values::angle"
    captured = next(binding for binding in region.bindings if binding.root == root)
    parameter = next(item for item in region.parameters if item.resource == root)
    assert captured.name == "angle"  # The original binding retains its authority.
    assert captured.rank == 0
    assert "parameter" in captured.attributes
    assert parameter.name.startswith("fort_region_parameter_")
    assert f"intent(in) :: {parameter.name}" in region.source
    assert "acos" not in region.source.lower()
    assert "original_value" not in region.source.lower()
    assert "helper_angle" not in region.source.lower()
    assert region.immutable_scalar_captures == (root,)
    assert region.public()["immutable_scalar_captures"] == [root]
    assert not region.immutable_constants


@pytest.mark.parametrize("helper_local", [False, True])
def test_inaccessible_scalar_parameter_retains_native_boundary(tmp_path, helper_local):
    with pytest.raises(CompilationError, match="scalar PARAMETER is unavailable in the original owner"):
        fixture(tmp_path, helper_local=helper_local, visible=helper_local)


def test_imported_scalar_initializer_changes_region_identity(tmp_path):
    _, _, original, paths = fixture(tmp_path)
    paths[0].write_text(paths[0].read_text().replace("acos(-1._8)", "acos(0._8)"))
    analysis = SourceEffects(paths)
    routine = analysis.routines["renamed_owner::advance"]
    loop = next(node for node in _children(routine.execution) if _kind(node) == "Block_Nonlabel_Do_Construct")
    changed = extract_region(analysis, routine, loop)
    assert changed.source_identity != original.source_identity
    assert changed.immutable_scalar_captures == original.immutable_scalar_captures


@pytest.mark.parametrize("precision", [4, 8])
def test_reached_scope_transports_original_visible_alias_without_recomputing_initializer(tmp_path, precision):
    _, _, _, paths = fixture(tmp_path, precision)
    facts = {"schema_version": 1, "participation": "serial",
             "captures": {"argument::a": FACT, "argument::out": FACT},
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths}}
    outputs, manifest = ScopeBuilder(paths, "renamed_owner::advance", facts=facts,
        options=CompilerOptions(), config=OffloadConfig("sections", scope_execution="reached")).run()
    assert manifest["scope_count"] == 1, manifest["boundaries"]
    region, = manifest["inline_numerical_regions"]["regions"]
    assert region["used"]
    assert region["immutable_scalar_captures"] == ["renamed_values::angle"]
    replacement = outputs[manifest["sources"][str(paths[2])]["replacement"]]
    assert "original_value" in replacement
    assert "acos" not in replacement.lower()
    numerical_sources = [text for path, text in outputs.items() if path.startswith("regions/") and path.endswith(".source")]
    assert len(numerical_sources) == 1
    assert "acos" not in numerical_sources[0].lower()


def test_constant_array_cannot_depend_on_a_scalar_capture(tmp_path):
    _, _, _, paths = fixture(tmp_path)
    paths[0].write_text(paths[0].read_text().replace("end module", "real(8),parameter::weights(2)=[angle,angle]\nend module"))
    paths[2].write_text(paths[2].read_text().replace("original_value=>angle", "original_value=>angle,weights")
                        .replace("out(i)=local_value(a(i))", "out(i)=weights(1)*a(i)"))
    analysis = SourceEffects(paths)
    routine = analysis.routines["renamed_owner::advance"]
    loop = next(node for node in _children(routine.execution) if _kind(node) == "Block_Nonlabel_Do_Construct")
    with pytest.raises(CompilationError, match="proved numeric conversions"):
        extract_region(analysis, routine, loop)


@pytest.mark.parametrize("precision", [4, 8])
def test_distinct_same_spelling_parameters_get_distinct_capture_dummies(tmp_path, precision):
    left = tmp_path / "left.f90"
    right = tmp_path / "right.f90"
    for path, module, expression in [(left, "renamed_left", "-1"), (right, "renamed_right", "0")]:
        path.write_text(f"module {module}\nimplicit none\nreal({precision}),parameter::angle=acos({expression}._{precision})\nend module\n")
    caller = tmp_path / "caller.f90"
    caller.write_text(f"""module renamed_owner
use renamed_left,only:left_value=>angle
use renamed_right,only:right_value=>angle
implicit none
contains
subroutine advance(a,out,n)
real({precision}),intent(in)::a(:)
real({precision}),intent(inout)::out(:)
integer,intent(in)::n
integer::i
do i=1,n
out(i)=left_value*a(i)+right_value
enddo
end subroutine
end module
""")
    analysis = SourceEffects([left, right, caller])
    routine = analysis.routines["renamed_owner::advance"]
    loop = next(node for node in _children(routine.execution) if _kind(node) == "Block_Nonlabel_Do_Construct")
    region = extract_region(analysis, routine, loop)
    parameters = [item for item in region.parameters if item.resource in region.immutable_scalar_captures]
    assert len(parameters) == len({item.name for item in parameters}) == 2
    assert all(item.name.startswith("fort_region_parameter_") for item in parameters)
    assert "acos" not in region.source.lower()
    paths = left, right, caller
    facts = {"schema_version": 1, "participation": "serial",
             "captures": {"argument::a": FACT, "argument::out": FACT},
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths}}
    builder = ScopeBuilder(paths, routine.qualified, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig("sections", scope_execution="reached"))
    assert builder.visible(builder.entry, "renamed_left::angle") == "left_value"
    assert builder.visible(builder.entry, "renamed_right::angle") == "right_value"
    outputs, manifest = builder.run()
    assert manifest["scope_count"] == 1, manifest["boundaries"]
    replacement = "".join(outputs[manifest["sources"][str(caller)]["replacement"]].replace("&", "").split())
    assert ",left_value,right_value," in replacement


@pytest.mark.native
@pytest.mark.parametrize("precision", [4, 8])
@pytest.mark.parametrize("integer", [False, True])
def test_original_compile_time_parameter_value_matches_outlined_complete_fields(tmp_path, precision, integer):
    compiler = shutil.which("gfortran-15") or shutil.which("gfortran")
    if compiler is None:
        pytest.skip("requires native Fortran")
    _, _, region, paths = fixture(tmp_path, precision, integer=integer)
    normalized = region.write(tmp_path / "normalized")
    module, procedure = region.entry.split("::")
    actuals = {"argument::a": "a", "argument::out": "expected", "argument::n": "n",
               "renamed_values::angle": "original_value"}
    arguments = ["-2" if parameter.lower_bound_dimension else actuals[parameter.resource]
                 for parameter in region.parameters]
    bits = "int32" if precision == 4 else "int64"
    driver = tmp_path / "driver.f90"
    driver.write_text(f"""program verify
use iso_fortran_env,only:{bits}
use renamed_owner,only:advance
use renamed_values,only:original_value=>angle
use {module},only:{procedure}
implicit none
integer,parameter::n=17
real({precision})::a(-2:n+1),out(-2:n+1),expected(-2:n+1)
integer::i
do i=-2,n+1
a(i)=real(3*i-7,{precision})/16._{precision}
enddo
out=-117._{precision}
expected=out
call advance(a,out,n)
call {procedure}({','.join(arguments)})
if(any(transfer(out,[0_{bits}],size(out))/=transfer(expected,[0_{bits}],size(expected)))) &
 error stop 'compile-time scalar value or complete fields differ'
print *,'FIELDS_BITWISE_OK'
end program
""")
    command = [compiler, "-O3", "-fopenmp", "-ffp-contract=off", "-fcheck=all", *(str(path) for path in paths),
               str(normalized), str(driver), "-o", str(tmp_path / "verify")]
    compiled = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    executed = subprocess.run([str(tmp_path / "verify")], cwd=tmp_path, capture_output=True,
                              text=True, timeout=10, check=False)
    assert executed.returncode == 0, executed.stdout + executed.stderr
    assert "FIELDS_BITWISE_OK" in executed.stdout
