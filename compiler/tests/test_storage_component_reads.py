"""Numeric component reads carry storage effects, never payload authority."""
from copy import copy

import pytest
from fparser.two.Fortran2003 import Data_Ref
from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.storage_component_reads import numeric_component_read
from compiler.ir import CompilationError


def fixture(tmp_path, *, body="out=current%tempfl+current%tempflsgs", root_attributes="", field_attributes="allocatable",
            header="type Profiles", field_type="real(knd)", field_shape="(:)", private="", extra="",
            root_type="type(Profiles)", root_shape="", module_declarations=""):
    path = tmp_path / "profiles.f90"
    root_attributes = "," + root_attributes if root_attributes else ""
    field_attributes = "," + field_attributes if field_attributes else ""
    path.write_text(f"""module declared
implicit none
integer,parameter::knd=8
{header}
{private}
{field_type}{field_attributes}::tempfl{field_shape},tempflsgs{field_shape}
contains
procedure::unused_method
end type
{root_type}{root_attributes}::current_profiles{root_shape}
{module_declarations}
contains
subroutine unused_method(self)
class(Profiles)::self
end subroutine
end module
module user
use declared,only:current=>current_profiles
implicit none
contains
subroutine advance(n,out)
integer,intent(in)::n
real(8)::out(0:n)
{body}
end subroutine
{extra}
end module
""")
    analysis = SourceEffects([path])
    routine = analysis.routines["user::advance"]
    node = next((node for node in walk(routine.execution) if type(node).__name__ == "Data_Ref"), None)
    if node is None:
        node = next(node for node in walk(routine.execution) if type(node).__name__ == "Function_Reference")
    return path, analysis, routine, node


def read(analysis, routine, node, *, callback=None):
    indices, scopes, ticks = [], {}, []

    def index(node):
        indices.append(node)
        if callback is not None:
            callback(node)

    result = numeric_component_read(analysis, routine, node, visit_index=index,
                                    visit_scope=lambda scope: scopes.setdefault(id(scope), scope),
                                    tick=lambda: ticks.append(1))
    return result, indices, scopes, len(ticks)


@pytest.mark.parametrize("body", [
    "out=current%tempfl+current%tempflsgs",
    "if(allocated(current%tempfl)) out=current%tempfl",
    "out=sqrt(current%tempfl**2+current%tempflsgs**2)",
])
def test_original_tempfl_shaped_reads_do_not_require_component_layout_or_allocation_facts(tmp_path, body):
    _path, analysis, routine, node = fixture(tmp_path, body=body)
    result, indices, scopes, ticks = read(analysis, routine, node)
    public = result.public()
    assert result.resource == "declared::current_profiles%tempfl"
    assert result.root_binding is analysis.modules["declared"].bindings["current_profiles"]
    assert public["terminal"] == {"type": "real", "kind": 8, "rank": 1}
    assert public["fields"][0]["declaring_module"] == "declared"
    assert not indices
    assert scopes
    assert ticks
    for key in ("allocation_or_definition_inferred", "descriptor_mutation_authorized", "payload_access_authorized",
                "layout_authorized", "capture_authorized", "execution_authorized", "motion_authorized", "alias_authority"):
        assert public[key] is False
    assert not getattr(analysis, "_native_functions", {})
    assert not getattr(analysis, "_allocation_authorizations", {})


def test_original_subscripts_are_passed_to_the_same_scanner_callback(tmp_path):
    _path, analysis, routine, node = fixture(tmp_path, body="out=current%tempfl(n-1)")
    result, indices, _scopes, _ticks = read(analysis, routine, node)
    assert tuple(indices) == result.indices
    assert len(indices) == 1
    assert indices[0] is node.items[-1].items[1].items[0]
    assert result.public()["terminal"]["rank"] == 0


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("phase", ["tick", "index", "scope"])
def test_callback_cannot_reseed_the_canonical_original_object_resource(tmp_path, sealed, phase):
    _path, analysis, routine, node = fixture(tmp_path, body="out=current%tempfl(n-1)")
    if sealed:
        analysis.seal_session()
        routine = analysis.routines[routine.qualified]
    root = analysis.modules["declared"].bindings["current_profiles"]
    mutated = False

    def callback(selected):
        nonlocal mutated
        if selected == phase and not mutated:
            object.__setattr__(root, "root", "forged::profiles")
            mutated = True

    with pytest.raises(CompilationError):
        numeric_component_read(analysis, routine, node,
            visit_index=lambda _index: callback("index"),
            visit_scope=lambda _scope: callback("scope"), tick=lambda: callback("tick"))
    assert mutated


@pytest.mark.parametrize("sealed", [False, True])
def test_existing_wrong_canonical_resource_cannot_become_the_read_baseline(tmp_path, sealed):
    _path, analysis, routine, node = fixture(tmp_path)
    if sealed:
        analysis.seal_session()
        routine = analysis.routines[routine.qualified]
    root = analysis.modules["declared"].bindings["current_profiles"]
    object.__setattr__(root, "root", "forged::profiles")
    with pytest.raises(CompilationError):
        read(analysis, routine, node)


@pytest.mark.parametrize("body", [
    "current%tempfl=out", "current%tempfl(:)=out", "read(*,*)current%tempfl",
    "call unknown(current%tempfl)", "out=unknown(current%tempfl)",
    "out=c_loc(current%tempfl)", "out=current%unused_method()",
    "out=.custom.current%tempfl",
])
def test_writes_address_passing_methods_and_defined_operators_are_not_read_capabilities(tmp_path, body):
    _path, analysis, routine, node = fixture(tmp_path, body=body)
    with pytest.raises(CompilationError):
        read(analysis, routine, node)


@pytest.mark.parametrize("attributes", ["pointer", "target", "volatile", "asynchronous", "allocatable"])
def test_unproved_parent_association_and_concurrency_reject(tmp_path, attributes):
    _path, analysis, routine, node = fixture(tmp_path, root_attributes=attributes)
    with pytest.raises(CompilationError):
        read(analysis, routine, node)


def test_pointer_component_cannot_borrow_allocatable_read_capability(tmp_path):
    _path, analysis, routine, node = fixture(tmp_path, field_attributes="pointer")
    with pytest.raises(CompilationError):
        read(analysis, routine, node)


@pytest.mark.parametrize("body", [
    "write(*,*,iostat=current%tempfl) n",
    "open(newunit=current%tempfl,status='scratch')",
    "close(6,iostat=current%tempfl)",
    "flush(6,iostat=current%tempfl)",
    "inquire(unit=6,number=current%tempfl)",
])
def test_original_io_output_designator_is_not_a_numeric_component_read(tmp_path, body):
    _path, analysis, routine, node = fixture(tmp_path, field_type="integer(4)", field_shape="", body=body)
    with pytest.raises(CompilationError, match="I/O"):
        read(analysis, routine, node)


def test_original_write_data_list_component_is_a_read_without_io_execution_authority(tmp_path):
    _path, analysis, routine, node = fixture(tmp_path, body="write(*,*)current%tempfl")
    result, _indices, _scopes, _ticks = read(analysis, routine, node)
    assert result.resource == "declared::current_profiles%tempfl"
    assert not result.public()["execution_authorized"]


def test_original_iomsg_designator_rejects_before_nonnumeric_terminal_classification(tmp_path):
    _path, analysis, routine, node = fixture(tmp_path, field_type="character(64)", field_shape="",
                                            body="write(*,*,iomsg=current%tempfl,iostat=n) n")
    with pytest.raises(CompilationError, match="I/O output"):
        read(analysis, routine, node)


@pytest.mark.parametrize("declaration", ["common /alias/ current_profiles", "namelist /alias/ current_profiles",
                                        "!$omp threadprivate(current_profiles)"])
def test_hidden_declaring_storage_alias_or_thread_state_rejects(tmp_path, declaration):
    _path, analysis, routine, node = fixture(tmp_path, module_declarations=declaration)
    with pytest.raises(CompilationError, match="alias/concurrency"):
        read(analysis, routine, node)


def test_array_parent_with_bound_procedures_requires_separate_original_element_authority(tmp_path):
    _path, analysis, routine, node = fixture(tmp_path, root_shape="(2)", body="out=current(1)%tempfl")
    with pytest.raises(CompilationError, match="without extension, finalization or bound procedures"):
        read(analysis, routine, node)


def test_polymorphic_parent_has_no_declared_numeric_leaf_authority(tmp_path):
    _path, analysis, routine, node = fixture(tmp_path, root_type="class(Profiles)", root_attributes="allocatable")
    with pytest.raises(CompilationError):
        read(analysis, routine, node)


def test_coindexed_source_cannot_enter_local_component_read_service(tmp_path):
    # The current parser rejects this source form before component admission.
    with pytest.raises(CompilationError, match="invalid Fortran syntax"):
        fixture(tmp_path, root_shape="[*]", body="out=current[1]%tempfl")


def test_fixed_nested_scalar_object_still_uses_original_declaring_type_and_numeric_leaf(tmp_path):
    _path, analysis, routine, node = fixture(tmp_path,
        header="type Inner\nreal(knd)::value\nend type\ntype Profiles", field_type="type(Inner)",
        field_attributes="", field_shape="", body="out=current%tempfl%value")
    result, _indices, _scopes, _ticks = read(analysis, routine, node)
    assert result.resource == "declared::current_profiles%tempfl%value"
    assert [field["declaring_type"] for field in result.public()["fields"]] == ["profiles", "inner"]


def test_dynamic_intermediate_object_cannot_borrow_terminal_allocatable_read_permission(tmp_path):
    _path, analysis, routine, node = fixture(tmp_path,
        header="type Inner\nreal(knd)::value\nend type\ntype Profiles", field_type="type(Inner)",
        field_shape="", body="out=current%tempfl%value")
    with pytest.raises(CompilationError, match="intermediate"):
        read(analysis, routine, node)


def test_nonnumeric_terminal_requires_separate_type_authority(tmp_path):
    _path, analysis, routine, node = fixture(tmp_path, field_type="character(4)")
    with pytest.raises(CompilationError):
        read(analysis, routine, node)


def test_private_field_rejects_imported_object_only_access(tmp_path):
    _path, analysis, routine, node = fixture(tmp_path, private="private")
    with pytest.raises(CompilationError, match="inaccessible|private"):
        read(analysis, routine, node)


@pytest.mark.parametrize("mutation", ["source", "binding", "schema", "field", "imports", "module"])
def test_stale_and_equal_looking_declaration_resolution_authority_rejects(tmp_path, mutation):
    path, analysis, routine, node = fixture(tmp_path)
    read(analysis, routine, node)
    owner = analysis.modules["declared"]
    if mutation == "source":
        path.write_text(path.read_text() + "! changed\n")
    elif mutation == "binding":
        owner.bindings["current_profiles"] = copy(owner.bindings["current_profiles"])
    elif mutation == "schema":
        owner.component_types["profiles"] = copy(owner.component_types["profiles"])
    elif mutation == "field":
        schema = owner.component_types["profiles"]
        dtype, attributes, entity = schema.members["tempfl"]
        schema.members["tempfl"] = (dtype, attributes, copy(entity))
    elif mutation == "imports":
        routine.scope.parent.imports["current"] = ("foreign", "current_profiles")
    else:
        analysis.modules["declared"] = copy(owner)
    with pytest.raises(CompilationError):
        read(analysis, routine, node)


def test_lookalike_expression_has_no_original_read_occurrence(tmp_path):
    _path, analysis, routine, node = fixture(tmp_path)
    with pytest.raises(CompilationError, match="exact original"):
        read(analysis, routine, Data_Ref(str(node)))


def test_callback_cannot_mutate_namespace_between_type_check_and_publication(tmp_path):
    _path, analysis, routine, node = fixture(tmp_path, body="out=current%tempfl(n)")

    def mutate(_node):
        routine.scope.parent.imports["current"] = ("foreign", "current_profiles")

    with pytest.raises(CompilationError):
        read(analysis, routine, node, callback=mutate)


@pytest.mark.parametrize("mutation", ["write_swap", "equal_parent", "equal_read_sibling"])
def test_callback_cannot_substitute_equal_text_original_read_parent_edges(tmp_path, mutation):
    body = ("current%tempfl(n)=current%tempfl(n)" if mutation == "write_swap" else
            "out=current%tempfl(n)+current%tempfl(n)" if mutation == "equal_read_sibling" else
            "out=current%tempfl(n)")
    _path, analysis, routine, _node = fixture(tmp_path, body=body)
    assignment = next(node for node in walk(routine.execution) if type(node).__name__ == "Assignment_Stmt")
    nodes = [node for node in walk(assignment.items[2]) if type(node).__name__ == "Data_Ref"]
    node = nodes[0]
    before = str(routine.execution)

    def mutate(_index):
        if mutation == "write_swap":
            left, equals, right = assignment.items
            assignment.items = right, equals, left
        elif mutation == "equal_parent":
            ordinal = next(i for i, value in enumerate(routine.execution.content) if value is assignment)
            routine.execution.content[ordinal] = copy(assignment)
        else:
            left, plus, right = assignment.items[2].items
            assignment.items[2].items = right, plus, left
        assert str(routine.execution) == before

    with pytest.raises(CompilationError, match="writes|parent/branch|original"):
        read(analysis, routine, node, callback=mutate)


def test_shadowed_intrinsic_cannot_borrow_read_argument_semantics(tmp_path):
    _path, analysis, routine, node = fixture(tmp_path, body="out=sqrt(current%tempfl)", extra="""function sqrt(x) result(y)
real(8)::x(:),y(size(x))
y=x
end function""")
    with pytest.raises(CompilationError, match="procedure/address"):
        read(analysis, routine, node)
