"""Type-only scalar actual proofs preserve original evaluation and storage."""

import pytest
from fparser.two.utils import walk

from compiler.frontend.call_bindings import resolve_source_call
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError


def resolve(tmp_path, expression, *, dtype="integer", kind=4, intent="in", declaration="", helpers=""):
    path = tmp_path / "scalar_calls.f90"
    path.write_text(f"""module renamed_scalar_calls
implicit none
contains
subroutine leaf(value)
{dtype}({kind}),intent({intent})::value
end subroutine
subroutine driver(k,a)
{dtype}({kind}),intent(in)::k,a(:)
{declaration}
call leaf({expression})
end subroutine
{helpers}
end module
""")
    analysis = SourceEffects([path])
    routine = analysis.routines["renamed_scalar_calls::driver"]
    call, = [node for node in walk(routine.execution) if type(node).__name__ == "Call_Stmt"]
    return analysis, resolve_source_call(analysis, routine.scope, call)


@pytest.mark.parametrize("expression,dtype,kind", [
    ("k-6", "integer", 4),
    ("-(k-6)", "integer", 4),
    ("+(k*2)/3", "integer", 4),
    ("k-6_8", "integer", 8),
    ("-(k+2.e0)/3.e0", "real", 4),
    ("(k-6.d0)*2.d0", "real", 8),
])
def test_same_kind_scalar_arithmetic_is_proved_without_rewriting_original_actual(tmp_path, expression, dtype, kind):
    analysis, call = resolve(tmp_path, expression, dtype=dtype, kind=kind)
    actual, = call.actuals
    mapping, = call.mappings
    assert mapping.binding is None and mapping.section is None
    assert mapping.resource is None
    assert call.render_original_arguments() == (str(actual),)
    summary = analysis.summarize("renamed_scalar_calls::driver")
    assert summary["complete"], summary["reasons"]
    assert any(item["kind"] == "read" and item["resource"] == "argument::k"
               for item in summary["operations"])


@pytest.mark.parametrize("expression,options,reason", [
    ("k-6", {"kind": 8}, "type, kind or rank mismatch"),
    ("k+1.d0", {"dtype": "real"}, "type, kind or rank mismatch"),
    ("a+1", {}, "type, kind or rank mismatch"),
    ("k-6", {"intent": "inout"}, "writable or descriptor actual requires original storage"),
    ("unknown(k)", {}, "type, kind or rank mismatch"),
])
def test_unproved_types_arrays_calls_and_writable_temporaries_stay_boundaries(tmp_path, expression, options, reason):
    with pytest.raises(CompilationError, match=reason):
        resolve(tmp_path, expression, **options)
