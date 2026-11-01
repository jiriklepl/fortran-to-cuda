"""Standard integer kind exports use the existing supported target ABI model."""
import shutil
import subprocess

import pytest

from compiler.frontend import lower_file
from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.immutable_graph import ImmutableGraphMutation
from compiler.frontend.lowering import INTRINSIC_KIND_VERSION
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError


def fixture(tmp_path, imports, selector, *, specification="", prefix=""):
    path = tmp_path / "integer_kinds.f90"
    path.write_text(f"""{prefix}
module integer_kinds
{imports}
{specification}
contains
subroutine work(a)
real(8),intent(inout)::a(:)
integer({selector})::first,last
first=1
last=2
write(*,*) real(last-first),1_{selector}
a(1)=a(1)+1
end subroutine
subroutine advance(a)
real(8),allocatable,intent(inout)::a(:)
call work(a)
deallocate(a)
end subroutine
end module
""")
    analysis = SourceEffects([path])
    routine = analysis.routines["integer_kinds::advance"]
    graph = analysis.structure(routine.qualified)
    selected = tuple(graph.nodes[graph.root_id].children[:-1])
    return path, analysis, routine, selected


@pytest.mark.parametrize(("imports", "selector", "expected"), [
    ("use iso_fortran_env", "int32", 4),
    ("use iso_fortran_env", "int64", 8),
    ("use,intrinsic::iso_fortran_env,only:int32", "int32", 4),
    ("use,intrinsic::iso_fortran_env,only:clock_kind=>int64", "clock_kind", 8),
    ("use iso_fortran_env,clock_kind=>int64", "clock_kind", 8),
])
def test_known_integer_declarations_and_literals_keep_original_output_evaluation(tmp_path, imports, selector, expected):
    path, analysis, routine, selected = fixture(tmp_path, imports, selector)
    before = path.read_bytes()
    assert analysis.routines["integer_kinds::work"].scope.bindings["last"].kind == expected
    record = prove_descriptor_window(analysis, routine.qualified, selected, ("argument::a",)).public()
    output, = record["native_intrinsic_output_requirements"]
    assert output["version"] == 1
    assert [(item["type"], item["kind"], item["rank"]) for item in output["outputs"]] == [
        ("real", 4, 0), ("integer", expected, 0)]
    assert output["execution_authorized"] is output["entry_definition_authority"] is False
    assert analysis._summary_authority()["intrinsic_kind_version"] == INTRINSIC_KIND_VERSION == 3
    assert path.read_bytes() == before


def test_reexported_standard_integer_kind_is_resolved_from_original_module(tmp_path):
    prefix = "module original_kinds\nuse iso_fortran_env,only:wide=>int64\nend module"
    _path, analysis, routine, selected = fixture(tmp_path, "use original_kinds,only:clock_kind=>wide", "clock_kind", prefix=prefix)
    assert analysis.routines["integer_kinds::work"].scope.bindings["last"].kind == 8
    prove_descriptor_window(analysis, routine.qualified, selected, ("argument::a",))


@pytest.mark.parametrize("conflict", ["private", "unknown", "distinct"])
def test_private_unknown_and_distinct_reexports_cannot_supply_a_kind(tmp_path, conflict):
    prefix = "module original_kinds\nuse iso_fortran_env,only:wide=>int64\n"
    prefix += "private::wide\n" if conflict == "private" else ""
    prefix += "end module\n"
    imports = "use original_kinds,only:clock_kind=>wide"
    if conflict == "unknown":
        imports += "\nuse unknown_kinds"
    elif conflict == "distinct":
        prefix += "module distinct_kinds\ninteger,parameter::clock_kind=4\nend module\n"
        imports += "\nuse distinct_kinds"
    _path, analysis, routine, selected = fixture(tmp_path, imports, "clock_kind", prefix=prefix)
    assert analysis.routines["integer_kinds::work"].scope.bindings["last"].kind is None
    with pytest.raises(CompilationError):
        prove_descriptor_window(analysis, routine.qualified, selected, ("argument::a",))


def test_multiple_reexports_of_the_same_intrinsic_entity_remain_unambiguous(tmp_path):
    prefix = ("module first_kinds\nuse iso_fortran_env,only:int64\nend module\n"
              "module second_kinds\nuse iso_fortran_env,only:int64\nend module\n")
    _path, analysis, routine, selected = fixture(tmp_path, "use first_kinds\nuse second_kinds", "int64", prefix=prefix)
    assert analysis.routines["integer_kinds::work"].scope.bindings["last"].kind == 8
    prove_descriptor_window(analysis, routine.qualified, selected, ("argument::a",))


@pytest.mark.parametrize("conflict", ["unknown", "distinct"])
def test_direct_intrinsic_import_cannot_keep_its_provisional_width_after_full_resolution(tmp_path, conflict):
    imports = "use iso_fortran_env,only:int64"
    prefix = ""
    if conflict == "unknown":
        imports += "\nuse unknown_kinds"
    else:
        imports += "\nuse distinct_kinds"
        prefix = "module distinct_kinds\ninteger,parameter::int64=4\nend module\n"
    _path, analysis, routine, selected = fixture(tmp_path, imports, "int64", prefix=prefix)
    assert analysis.routines["integer_kinds::work"].scope.bindings["last"].kind is None
    with pytest.raises(CompilationError):
        prove_descriptor_window(analysis, routine.qualified, selected, ("argument::a",))


@pytest.mark.parametrize(("child_import", "expected"), [("use unknown_kinds", None),
                                                         ("use child_kinds", 4), ("", 8)])
def test_contained_scope_resolves_use_candidates_before_host_kind(tmp_path, child_import, expected):
    prefix = "module child_kinds\ninteger,parameter::int64=4\nend module\n"
    path, _analysis, _routine, _selected = fixture(tmp_path, "use iso_fortran_env,only:int64", "int64", prefix=prefix)
    path.write_text(path.read_text().replace("subroutine work(a)\n", "subroutine work(a)\n"+child_import+"\n"))
    analysis = SourceEffects([path])
    assert analysis.routines["integer_kinds::work"].scope.bindings["last"].kind == expected


@pytest.mark.parametrize(("imports", "selector", "specification"), [
    ("use iso_fortran_env,only:clock_kind=>int64", "int64", ""),
    ("use iso_fortran_env,only:int32", "int64", ""),
    ("use,non_intrinsic::iso_fortran_env", "int64", ""),
    ("", "int64", "integer::int64"),
    ("", "unresolved", "integer,parameter::unresolved=unknown"),
])
def test_unimported_shadowed_or_unresolved_kind_cannot_gain_output_authority(tmp_path, imports, selector, specification):
    _path, analysis, routine, selected = fixture(tmp_path, imports, selector, specification=specification)
    assert analysis.routines["integer_kinds::work"].scope.bindings["last"].kind is None
    with pytest.raises(CompilationError):
        prove_descriptor_window(analysis, routine.qualified, selected, ("argument::a",))


def test_original_local_parameter_keeps_its_own_value_instead_of_standard_spelling(tmp_path):
    _path, analysis, routine, selected = fixture(tmp_path, "", "int64", specification="integer,parameter::int64=4")
    assert analysis.routines["integer_kinds::work"].scope.bindings["last"].kind == 4
    prove_descriptor_window(analysis, routine.qualified, selected, ("argument::a",))


@pytest.mark.parametrize("sealed", [False, True])
def test_cached_output_cannot_keep_a_mutated_imported_kind_environment(tmp_path, sealed):
    _path, analysis, routine, selected = fixture(tmp_path, "use iso_fortran_env,only:int64", "int64")
    if sealed:
        analysis.seal_session()
    proof = prove_descriptor_window(analysis, routine.qualified, selected, ("argument::a",))
    if sealed:
        with pytest.raises(ImmutableGraphMutation):
            analysis.modules["integer_kinds"].kinds.values["int64"] = 4
        assert proof.validate(analysis, routine.qualified, selected) is proof
        return
    analysis.modules["integer_kinds"].kinds.values["int64"] = 4
    with pytest.raises(CompilationError):
        proof.validate(analysis, routine.qualified, selected)


def test_numeric_lowering_still_rejects_nondefault_integer_width(tmp_path):
    path = tmp_path / "wide_numeric.f90"
    path.write_text("module wide_numeric\nuse iso_fortran_env,only:int64\ncontains\n"
                    "subroutine advance(a)\ninteger(int64),intent(inout)::a(:)\n"
                    "a(1)=a(1)+1_int64\nend subroutine\nend module\n")
    with pytest.raises(CompilationError, match="only default INTEGER"):
        lower_file(path, "advance")


def test_native_target_confirms_the_supported_intrinsic_kind_abi(tmp_path):
    compiler = shutil.which("gfortran")
    if compiler is None:
        pytest.skip("native Fortran compiler is unavailable")
    source, binary = tmp_path / "target_kinds.f90", tmp_path / "target_kinds"
    source.write_text("program target_kinds\nuse,intrinsic::iso_fortran_env,only:int32,int64\n"
                      "if(int32/=4.or.int64/=8) stop 1\n"
                      "if(storage_size(0_int32)/=32.or.storage_size(0_int64)/=64) stop 2\n"
                      "end program\n")
    subprocess.run([compiler, str(source), "-o", str(binary)], check=True, capture_output=True, timeout=30)
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)
