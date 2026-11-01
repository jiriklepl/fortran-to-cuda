"""Source associations retain the original scalar arithmetic evaluation."""
import shutil
import subprocess

import pytest

from compiler.frontend.call_bindings import resolve_original_call, resolve_source_call
from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.source_scalar_types import SOURCE_SCALAR_OPERATOR_VERSION
from compiler.ir import CompilationError

CASES = [
    ("1/r64**2", "real", 8),
    ("r32**i64", "real", 4),
    ("r64**i32", "real", 8),
    ("i64**r32", "real", 4),
    ("r32**r64", "real", 8),
    ("i32**i64", "integer", 8),
    ("i64**i32", "integer", 8),
    ("i32**(-2)", "integer", 4),
    ("r64**(-2)", "real", 8),
    ("1_i64_kind/(i32**2)", "integer", 8),
    ("r32+i64", "real", 4),
    ("r64-i32", "real", 8),
    ("i32*i64", "integer", 8),
    ("r32/(i64+1_i64_kind)", "real", 4),
]


def fixture(tmp_path, expression, dtype, width, *, declarations="", intent="in", tail=""):
    path = tmp_path / "scalar_operators.f90"
    path.write_text(f"""module scalar_operators
use iso_fortran_env,only:i64_kind=>int64
contains
subroutine leaf(value)
{dtype}({width}),intent({intent})::value
continue
end subroutine
subroutine advance(a)
real(8),allocatable,intent(inout)::a(:)
real(4)::r32
real(8)::r64
integer::i32
integer(i64_kind)::i64
{declarations}
call leaf({expression})
{tail}
a(1)=a(1)+1
deallocate(a)
end subroutine
end module
""")
    analysis = SourceEffects([path])
    routine = analysis.routines["scalar_operators::advance"]
    call = next(node for node in routine.execution.content if type(node).__name__ == "Call_Stmt")
    return path, analysis, routine, call


@pytest.mark.parametrize(("expression", "dtype", "width"), CASES)
def test_original_and_effect_call_associations_agree_without_evaluating_or_rewriting(tmp_path, expression, dtype, width):
    path, analysis, routine, call = fixture(tmp_path, expression, dtype, width)
    before = path.read_bytes()
    original = resolve_original_call(analysis, routine.scope, call)
    mapped = resolve_source_call(analysis, routine.scope, call)
    actual = original.associations[0].actual_type
    assert (actual.dtype, actual.kind, actual.rank) == (dtype, width, 0)
    assert analysis._signature(routine.scope, original.actuals[0]) == (dtype, width, 0)
    assert original.actuals[0] is mapped.actuals[0] is call.items[1].items[0]
    assert original.public()["execution_authority"] is original.public()["capture_authority"] is False
    graph = analysis.structure(routine.qualified)
    selected = tuple(graph.nodes[graph.root_id].children[:-1])
    proof = prove_descriptor_window(analysis, routine.qualified, selected, ("argument::a",))
    assert proof.public()["entry_definition_authority"] is proof.public()["execution_authorized"] is False
    assert analysis._summary_authority()["source_scalar_operator_version"] == SOURCE_SCALAR_OPERATOR_VERSION == 1
    assert path.read_bytes() == before


@pytest.mark.parametrize(("expression", "declarations"), [
    ("a**2", ""),
    ("r16**2", "real(16)::r16"),
    ("unknown_kind**2", "real(unknown)::unknown_kind"),
    (".true.**2", ""),
    ("r64>r32", ""),
    ("'text'// 'value'", ""),
    ("mystery()**2", "external mystery\nreal(8)::mystery"),
    ("item**2", "type private_t\ninteger::value\nend type\ntype(private_t)::item"),
])
def test_array_derived_unknown_and_unsupported_operators_do_not_gain_signatures(tmp_path, expression, declarations):
    _path, analysis, routine, call = fixture(tmp_path, expression, "real", 8, declarations=declarations)
    assert analysis._signature(routine.scope, call.items[1].items[0]) is None
    with pytest.raises(CompilationError):
        resolve_original_call(analysis, routine.scope, call)
    with pytest.raises(CompilationError):
        resolve_source_call(analysis, routine.scope, call)


def test_type_association_does_not_create_writable_actual_storage(tmp_path):
    _path, analysis, routine, call = fixture(tmp_path, "r64**2", "real", 8, intent="out")
    with pytest.raises(CompilationError):
        resolve_original_call(analysis, routine.scope, call)


def test_operator_typing_does_not_hide_later_original_allocation_mutation(tmp_path):
    _path, analysis, routine, _call = fixture(tmp_path, "1/r64**2", "real", 8, tail="deallocate(a)")
    graph = analysis.structure(routine.qualified)
    selected = tuple(graph.nodes[graph.root_id].children[:-1])
    with pytest.raises(CompilationError):
        prove_descriptor_window(analysis, routine.qualified, selected, ("argument::a",))


def test_native_fortran_confirms_result_kinds_of_the_complete_supported_matrix(tmp_path):
    compiler = shutil.which("gfortran")
    if compiler is None:
        pytest.skip("native Fortran compiler is unavailable")
    source, binary = tmp_path / "native_scalar_types.f90", tmp_path / "native_scalar_types"
    source.write_text("program native_scalar_types\nuse iso_fortran_env,only:i64_kind=>int64\n"
                      "real(4)::r32=2\nreal(8)::r64=2\ninteger::i32=2\ninteger(i64_kind)::i64=2\n" +
                      "\n".join(f"if(kind({expression})/={width}) stop {index+1}"
                                for index, (expression, _dtype, width) in enumerate(CASES)) +
                      "\nend program\n")
    subprocess.run([compiler, str(source), "-o", str(binary)], check=True, capture_output=True, timeout=30)
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)
