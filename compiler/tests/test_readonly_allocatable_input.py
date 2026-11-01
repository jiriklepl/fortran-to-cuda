"""Read-only allocation association is independent of payload initialization."""
from copy import copy
from dataclasses import replace

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.frontend.readonly_allocatable_input import prove_readonly_allocatable_input
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

SOURCE = """module readonly_input
implicit none
contains
subroutine work(q,a,n)
real(8),allocatable,intent(in)::q(:)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
call unrelated_effects(a)
if(allocated(q))then
!$omp parallel do private(i)
do i=1,n
a(i)=a(i)+q(lbound(q,1)+i-1)
enddo
!$omp end parallel do
endif
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE, *, sealed=False):
    path = tmp_path / "source.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine = analysis.routines["readonly_input::work"]
    return path, analysis, routine, routine.scope.bindings["q"]


def issue(parts):
    _path, analysis, routine, binding = parts
    return prove_readonly_allocatable_input(analysis, routine, binding)


@pytest.mark.parametrize("sealed", [False, True])
def test_whole_original_read_projection_has_no_definition_or_presence_credit(tmp_path, sealed):
    parts = fixture(tmp_path, sealed=sealed)
    path, analysis, routine, binding = parts
    before = path.read_bytes()
    proof = issue(parts)
    assert proof.validate(analysis, routine, binding) is proof
    assert issue(parts) is proof
    public = proof.public()
    assert public["original_read_occurrences"] == 3
    assert public["original_binding"]["intent"] == "in"
    assert public["unrelated_native_effects_require_ordinary_boundaries"] is True
    assert public["requires_reached_allocation_presence_and_alias_guards"] is True
    for key in ("allocation_presence_authority", "payload_definition_authority", "payload_access_authorized",
                "execution_authorized", "capture_authorized", "cross_frontier_retention_authorized"):
        assert public[key] is False
    public["payload_definition_authority"] = True
    assert proof.public()["payload_definition_authority"] is False
    assert analysis._allocation_authorizations == {}
    assert path.read_bytes() == before


@pytest.mark.parametrize("declaration", [
    "real(8),allocatable,intent(inout)::q(:)",
    "real(8),allocatable,intent(out)::q(:)",
    "real(8),intent(in)::q(:)",
    "real(8),allocatable,target,intent(in)::q(:)",
    "real(8),allocatable,optional,intent(in)::q(:)",
    "real(8),allocatable,volatile,intent(in)::q(:)",
    "real(8),allocatable,asynchronous,intent(in)::q(:)",
    "complex(8),allocatable,intent(in)::q(:)",
])
def test_unsupported_descriptor_declarations_reject(tmp_path, declaration):
    with pytest.raises(CompilationError, match="independent numeric ALLOCATABLE INTENT"):
        issue(fixture(tmp_path, SOURCE.replace("real(8),allocatable,intent(in)::q(:)", declaration)))


@pytest.mark.parametrize("statement", [
    "q=0.0_8", "q(1)=0.0_8", "deallocate(q)", "allocate(q(4))", "read(*,*)q",
    "call unknown(q)", "call move_alloc(q,other)", "i=loc(q)",
    "associate(alias=>q)\ni=size(alias)\nend associate", "read(*,*,iostat=q(1)) i",
])
def test_any_direct_write_address_or_alias_escape_rejects_even_in_tail(tmp_path, statement):
    source = SOURCE.replace("integer::i", "integer::i\nreal(8),allocatable::other(:)")
    source = source.replace("endif\nend subroutine", "endif\n" + statement + "\nend subroutine")
    with pytest.raises(CompilationError):
        issue(fixture(tmp_path, source))


@pytest.mark.parametrize("specification", ["namelist /state/ q", "common /state/ q", "equivalence(q,a)",
                                            "real(8)::scratch(size(q))"])
def test_specification_alias_or_observer_rejects(tmp_path, specification):
    source = SOURCE.replace("integer::i", "integer::i\n" + specification)
    with pytest.raises(CompilationError, match="declaration storage|specification use"):
        issue(fixture(tmp_path, source))


@pytest.mark.parametrize(("opening", "closing"), [
    ("parallel do private(q,i)", "end parallel do"),
    ("parallel do firstprivate(q) private(i)", "end parallel do"),
    ("task shared(q)", "end task"),
    ("target teams distribute parallel do", "end target teams distribute parallel do"),
    ("parallel do private(i)", "end do nowait"),
])
def test_unproved_concurrency_cannot_export_or_replace_dummy(tmp_path, opening, closing):
    source = SOURCE.replace("parallel do private(i)", opening).replace("end parallel do", closing)
    with pytest.raises(CompilationError, match="allocatable input"):
        issue(fixture(tmp_path, source))


def test_complete_parent_join_preserves_nowait_without_per_unit_authority(tmp_path):
    source = SOURCE.replace("!$omp parallel do private(i)", "!$omp parallel private(i)\n!$omp do")
    source = source.replace("!$omp end parallel do", "!$omp end do nowait\n!$omp end parallel")
    assert issue(fixture(tmp_path, source)).public()["original_directives"][-2:] == ["end do nowait", "end parallel"]


def test_unrelated_local_scalar_reduction_does_not_change_dummy_association(tmp_path):
    source = SOURCE.replace("integer::i", "integer::i\nreal(8)::s")
    source = source.replace("parallel do private(i)", "parallel do private(i) reduction(max:s)")
    assert "reduction(max:s)" in issue(fixture(tmp_path, source)).public()["original_directives"][0]


@pytest.mark.parametrize("name", ["q", "a", "missing"])
def test_reduction_must_not_replace_dummy_or_unknown_storage(tmp_path, name):
    source = SOURCE.replace("parallel do private(i)", f"parallel do private(i) reduction(max:{name})")
    with pytest.raises(CompilationError, match="independent scalar reduction storage"):
        issue(fixture(tmp_path, source))


def test_contained_host_association_cannot_observe_receiver(tmp_path):
    source = SOURCE.replace("end subroutine\nend module", "contains\nsubroutine observe()\nprint *,q\n"
                            "end subroutine\nend subroutine\nend module")
    with pytest.raises(CompilationError, match="contained host association"):
        issue(fixture(tmp_path, source))


@pytest.mark.parametrize("mutation", ["copy", "registry", "foreign", "binding", "namespace", "source", "occurrence"])
def test_issued_object_and_every_original_authority_revalidate(tmp_path, mutation):
    parts = fixture(tmp_path)
    path, analysis, routine, binding = parts
    proof = issue(parts)
    if mutation == "copy":
        proof = copy(proof)
    elif mutation == "registry":
        analysis._readonly_allocatable_inputs.clear()
    elif mutation == "foreign":
        analysis = SourceEffects([path])
        routine = analysis.routines[routine.qualified]
        binding = routine.scope.bindings["q"]
    elif mutation == "binding":
        routine.scope.bindings["q"] = replace(binding)
    elif mutation == "namespace":
        routine.scope.imports["new_alias"] = ("elsewhere", "q")
    elif mutation == "source":
        path.write_text(path.read_text().replace("a(i)=", "a(i) = 2*"))
    elif mutation == "occurrence":
        parent = next(node for node in walk(routine.execution) if type(node).__name__ == "Part_Ref"
                      and str(node.items[0]).lower() == "q")
        parent.items = (F.Name("q"), parent.items[1])
    with pytest.raises(CompilationError, match="original|compiler-issued|authority|source|scope"):
        proof.validate(analysis, routine, binding)


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("mutation", ["live_source", "input_metadata", "registry", "occurrence"])
def test_publication_rechecks_live_inputs_and_consumed_occurrences(tmp_path, monkeypatch, sealed, existing, mutation):
    from compiler.frontend import readonly_allocatable_input as service

    parts = fixture(tmp_path, sealed=sealed)
    path, analysis, routine, _binding = parts
    if existing:
        issue(parts)
    encode = service._encoded
    fired = False

    def alter(value):
        nonlocal fired
        result = encode(value)
        if not fired:
            fired = True
            if mutation == "live_source":
                path.write_text(path.read_text() + "! changed after projection\n")
            elif mutation == "input_metadata":
                inputs = copy(analysis.inputs)
                inputs.configuration = {"changed_after_projection": True}
                analysis.inputs = inputs
            elif mutation == "registry":
                analysis._readonly_allocatable_inputs = {}
            else:
                parent = next(node for node in walk(routine.execution) if type(node).__name__ == "Part_Ref"
                              and str(node.items[0]).lower() == "q")
                parent.items = (F.Name("q"), parent.items[1])
        return result

    monkeypatch.setattr(service, "_encoded", alter)
    with pytest.raises(CompilationError):
        issue(parts)
    assert fired
