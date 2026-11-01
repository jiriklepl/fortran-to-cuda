"""Original intrinsic output preserves lifetimes, without replacing execution."""
from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.call_bindings import OriginalCallBudgetExceeded
from compiler.frontend.descriptor_output import original_intrinsic_output
from compiler.frontend.descriptor_windows import _Check, prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError


def fixture(tmp_path, statement="write(*,*) 'flux',flux,sum(flux)", *, spec="", imports="", tail="", root=False):
    path = tmp_path / "outputs.f90"
    body = f"{statement}\n{tail}\na(1)=a(1)+1"
    path.write_text(f"""module outputs
{imports}
contains
subroutine work(a)
real(8),intent(inout)::a(:)
real(8)::flux(6)
integer::i
{spec}
flux=1
{body if not root else 'a(1)=a(1)+1'}
end subroutine
subroutine advance(a)
real(8),allocatable,intent(inout)::a(:)
{('real(8)::flux(6)' + chr(10) + spec) if root else ''}
{body if root else 'call work(a)'}
deallocate(a)
end subroutine
end module
""")
    analysis = SourceEffects([path])
    routine = analysis.routines["outputs::advance"]
    graph = analysis.structure(routine.qualified)
    selected = tuple(graph.nodes[graph.root_id].children[:-1])
    return path, analysis, routine, selected


def issue(data):
    _path, analysis, routine, selected = data
    return prove_descriptor_window(analysis, routine.qualified, selected, ("argument::a",))


@pytest.mark.parametrize(("statement", "imports"), [
    ("write(*,*) 'flux',flux,sum(flux)", ""),
    ("print *, 'flux',flux(2),sum(flux),1+2,1.0_8/.5_8,.true.", ""),
    ("write(unit=output_unit,fmt='(a,6es24.16)') 'flux',flux", "use iso_fortran_env,only:output_unit"),
    ("write(diagnostic,'(a,es24.16)') 'value',real(i),dble(i),int(1.0_8)",
     "use,intrinsic::iso_fortran_env,only:diagnostic=>error_unit"),
    ("write(*,*)", ""),
])
def test_complete_original_outputs_keep_order_and_all_authority_separate(tmp_path, statement, imports):
    data = fixture(tmp_path, statement, imports=imports)
    before = data[0].read_bytes()
    record = issue(data).public()
    assert record["schema_version"] == 8
    output, = record["native_intrinsic_output_requirements"]
    assert output["original_statement"].lower().startswith(("write", "print"))
    assert output["requires_original_position_and_guards"] is True
    assert output["requires_original_io_execution"] is True
    for flag in ("coherence_authorized", "payload_access_authorized", "entry_definition_authority",
                 "execution_authorized", "motion_authorized"):
        assert output[flag] is False
    assert record["execution_authorized"] is record["entry_definition_authority"] is False
    assert data[0].read_bytes() == before
    assert data[1].summarize("outputs::work")["complete"] is False


@pytest.mark.parametrize(("statement", "spec", "imports"), [
    ("write(6,*) flux", "", ""),
    ("write(buffer,*) flux", "character(80)::buffer", ""),
    ("write(*,*,iostat=i) flux", "", ""),
    ("write(*,*,err=100) flux\n100 continue", "", ""),
    ("write(*,*,asynchronous='yes') flux", "", ""),
    ("write(*,100) flux\n100 format(6f10.3)", "", ""),
    ("write(*,*) (flux(i),i=1,6)", "", ""),
    ("write(*,*) mystery(flux(1))", "external mystery\nreal(8)::mystery", ""),
    ("write(*,*) item", "type private_t\ninteger::x\nend type\ntype(private_t)::item", ""),
    ("write(output_unit,*) flux", "integer::output_unit", ""),
    ("write(output_unit,*) flux", "", "use unknown_module"),
    ("write(*,*) a", "", ""),
    ("write(*,*) sum(flux)", "external sum\nreal(8)::sum", ""),
])
def test_unsafe_or_unknown_io_cannot_lend_lifetime_authority(tmp_path, statement, spec, imports):
    with pytest.raises(CompilationError):
        issue(fixture(tmp_path, statement, spec=spec, imports=imports))


def test_safe_output_does_not_hide_later_original_descriptor_replacement(tmp_path):
    data = fixture(tmp_path, tail="call release(a)")
    path = data[0]
    path.write_text(path.read_text().replace("subroutine advance(a)",
        "subroutine release(a)\nreal(8),allocatable,intent(inout)::a(:)\ndeallocate(a)\nend subroutine\nsubroutine advance(a)"))
    # Reparse the exact new source instead of mutating a registered proof.
    analysis = SourceEffects([path])
    routine = analysis.routines["outputs::advance"]
    graph = analysis.structure(routine.qualified)
    selected = tuple(graph.nodes[graph.root_id].children[:-1])
    with pytest.raises(CompilationError):
        prove_descriptor_window(analysis, routine.qualified, selected, ("argument::a",))


def test_original_statement_identity_and_shared_hard_budget_failures_are_not_soft_declines(tmp_path, monkeypatch):
    data = fixture(tmp_path, "write(output_unit,*) flux", imports="use iso_fortran_env,only:output_unit")
    _path, analysis, _routine, _selected = data
    routine = analysis.routines["outputs::work"]
    node = next(item for item in walk(routine.execution) if type(item).__name__ == "Write_Stmt")
    check = _Check(analysis)
    with pytest.raises(CompilationError):
        original_intrinsic_output(check, routine, copy(node))
    monkeypatch.setattr(check, "resolution_tick", lambda *_: (_ for _ in ()).throw(OriginalCallBudgetExceeded("hard marker")))
    with pytest.raises(OriginalCallBudgetExceeded, match="hard marker"):
        original_intrinsic_output(check, routine, node)


def test_ordinary_no_output_proof_retains_historical_schema_and_fieldset(tmp_path):
    record = issue(fixture(tmp_path, "continue")).public()
    assert record["schema_version"] == 6
    assert "native_intrinsic_output_requirements" not in record


@pytest.mark.parametrize("declaration", ["character(8)", "character(kind=1,len=8)"])
def test_original_fixed_character_scalar_output_is_readonly_and_keeps_its_source_binding(tmp_path, declaration):
    record = issue(fixture(tmp_path, "print *, message", spec=declaration + "::message", root=True)).public()
    output, = record["native_intrinsic_output_requirements"]
    assert output["outputs"][0]["type"] == "character"
    assert output["outputs"][0]["rank"] == 0
    assert output["payload_access_authorized"] is False


@pytest.mark.parametrize("declaration", ["character(kind=4,len=8)", "character(8),pointer", "character(:),allocatable"])
def test_character_kind_and_replaceable_storage_are_not_inferred_from_length(tmp_path, declaration):
    with pytest.raises(CompilationError):
        issue(fixture(tmp_path, "print *, message", spec=declaration + "::message", root=True))
