"""Primitive scalar broadcast preserves descriptors, never invents allocation."""
from copy import copy

import pytest

from compiler.frontend.descriptor_scalar_fill import scalar_fill_descriptor_effects
from compiler.frontend.descriptor_windows import _Check, prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

ENTRY = "fills::advance"


def fixture(tmp_path, *, dtype="real", kind=8, rank=3, rhs="value", suffix="",
            attributes="allocatable", value_decl="real(8)::value", selected=None, module=False,
            module_before="", module_spec="", routine_spec="", module_procedures=""):
    shape = ",".join(":" for _ in range(rank))
    declaration = f"{dtype}({kind}),{attributes}::a({shape})" if attributes else f"{dtype}({kind})::a({shape})"
    path = tmp_path / "fills.f90"
    path.write_text(f"""{module_before}module fills
{module_spec}
{declaration if module else ''}
{value_decl}
contains
subroutine advance({'' if module else 'a'})
{routine_spec}
{'' if module else declaration}
{selected or 'a='+rhs}
{suffix}
end subroutine
{module_procedures}
end module
""")
    analysis = SourceEffects([path], operations=2048)
    routine = analysis.routines[ENTRY]
    originals = tuple(node for node in routine.execution.content if type(node).__name__ != "Comment")
    graph = analysis.structure(ENTRY)
    selection = tuple(graph.node_id(node) for node in originals)
    roots, modules = ((), ("fills::a",)) if module else (("argument::a",), ())
    return path, analysis, routine, selection, roots, modules


def prove(analysis, selection, roots, modules):
    return prove_descriptor_window(analysis, ENTRY, selection, roots, module_roots=modules)


@pytest.mark.parametrize("dtype", ["real", "integer"])
@pytest.mark.parametrize("kind", [4, 8])
@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize("sealed", [False, True])
def test_primitive_allocated_array_scalar_broadcast_is_descriptor_only(tmp_path, dtype, kind, rank, sealed):
    path, analysis, _routine, selection, roots, modules = fixture(tmp_path, dtype=dtype, kind=kind, rank=rank)
    if sealed:
        analysis.seal_session()
    original = path.read_bytes()
    public = prove(analysis, selection, roots, modules).public()
    assert public["schema_version"] == 24
    requirement, = public["native_intrinsic_descriptor_requirements"]
    assert requirement["target"] == {"resource": "argument::a", "type": dtype, "kind": kind, "rank": rank}
    assert requirement["rhs"] == {"type": "real", "kind": 8, "rank": 0}
    assert requirement["requires_original_allocated_array"]
    assert requirement["storage_writes"]
    for flag in ("allocation_presence_proved", "allocation_or_escape", "entry_definition_authority",
                 "execution_authorized", "capture_authorized", "gpu_legality_established",
                 "numerical_lowering_authorized", "motion_authorized"):
        assert requirement[flag] is False
    assert public["resources"][0]["requires_original_allocation_presence_and_alias_guards"]
    assert not public["entry_definition_authority"]
    assert path.read_bytes() == original


@pytest.mark.parametrize("rhs", ["2", "2_8", "2.0", "2.0_8", "-2.0_8", "(value)",
                                  "2.0_8*value+1", "value**(-2)"])
def test_existing_scalar_type_rules_keep_original_expression_and_conversion(tmp_path, rhs):
    _path, analysis, routine, selection, roots, modules = fixture(tmp_path, rhs=rhs)
    proof = prove(analysis, selection, roots, modules)
    record, = proof.public()["native_intrinsic_descriptor_requirements"]
    assert record["source_expression"] == str(routine.execution.content[0])
    assert record["rhs"]["rank"] == 0
    assert proof.validate(analysis, ENTRY, selection) is proof


def test_original_module_array_needs_allocation_guard_and_keeps_native_summary(tmp_path):
    _path, analysis, _routine, selection, roots, modules = fixture(tmp_path, module=True)
    before = analysis.summarize(ENTRY)
    public = prove(analysis, selection, roots, modules).public()
    assert public["resources"][0]["resource_kind"] == "original_module_allocatable"
    assert public["resources"][0]["execution_requires_runtime_guard"]
    assert public["resources"][0]["requires_original_runtime_bounds"]
    assert analysis.summarize(ENTRY) == before
    assert not before["complete"]
    assert any("storage" in reason for reason in before["reasons"])


@pytest.mark.parametrize(("rhs", "value_decl"), [
    ("value", "real(8)::value(2,2,2)"), ("[1.0_8,2.0_8]", "real(8)::value"),
    ("value", "real(16)::value"), ("value", "logical::value"),
    ("value", "complex(8)::value"), ("value", "character(2)::value"),
    ("missing()", "real(8)::value"), ("value", "real(8),pointer::value"),
])
def test_array_rhs_and_unproved_scalar_effects_cannot_borrow_broadcast(tmp_path, rhs, value_decl):
    _path, analysis, _routine, selection, roots, modules = fixture(tmp_path, rhs=rhs, value_decl=value_decl)
    with pytest.raises(CompilationError):
        prove(analysis, selection, roots, modules)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


@pytest.mark.parametrize("suffix", ["call unknown(a)", "deallocate(a)", "a=value_array"])
def test_every_later_original_effect_remains_checked(tmp_path, suffix):
    _path, analysis, _routine, selection, roots, modules = fixture(
        tmp_path, suffix=suffix, value_decl="real(8)::value,value_array(2,2,2)")
    with pytest.raises(CompilationError):
        prove(analysis, selection, roots, modules)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


def test_copied_statement_and_routine_cannot_claim_original_occurrence(tmp_path):
    _path, analysis, routine, _selection, _roots, _modules = fixture(tmp_path)
    statement = routine.execution.content[0]
    for owner, node in ((copy(routine), statement), (routine, copy(statement))):
        with pytest.raises(CompilationError, match="exact original"):
            scalar_fill_descriptor_effects(analysis, owner, node)


def test_old_nonallocatable_element_write_does_not_use_new_schema(tmp_path):
    _path, analysis, _routine, selection, roots, modules = fixture(tmp_path, attributes="", selected="a(1,1,1)=value")
    public = prove(analysis, selection, roots, modules).public()
    assert public["schema_version"] == 10
    assert "native_intrinsic_descriptor_requirements" not in public


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("fault", ["kind", "root", "namespace", "record", "syntax"])
def test_scalar_fill_consumption_reuses_strong_publication_fences(tmp_path, monkeypatch, sealed, fault):
    _path, analysis, routine, selection, roots, modules = fixture(tmp_path)
    if sealed:
        analysis.seal_session()
    binding = analysis.modules["fills"].bindings["value"]
    publish = _Check.publish

    def changed(check, *args):
        if fault in {"kind", "root"}:
            object.__setattr__(binding, fault, 4 if fault == "kind" else "forged::value")
        elif fault == "namespace":
            object.__setattr__(routine.scope, "bindings", dict(routine.scope.bindings))
        elif fault == "record":
            next(iter(check.intrinsics.values()))["allocation_presence_proved"] = True
        else:
            statement = routine.execution.content[0]
            object.__setattr__(statement, "items", (*statement.items[:2], copy(statement.items[2])))
        return publish(check, *args)

    monkeypatch.setattr(_Check, "publish", changed)
    with pytest.raises(CompilationError):
        prove(analysis, selection, roots, modules)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


@pytest.mark.parametrize("sealed", [False, True])
def test_rhs_type_callback_cannot_mutate_its_earlier_consumed_declaration(tmp_path, monkeypatch, sealed):
    _path, analysis, _routine, selection, roots, modules = fixture(tmp_path)
    if sealed:
        analysis.seal_session()
    signature = analysis._signature
    binding = analysis.modules["fills"].bindings["value"]

    def changed(scope, node, *args, **kwargs):
        result = signature(scope, node, *args, **kwargs)
        if type(node).__name__ == "Name":
            object.__setattr__(binding, "kind", 16)
        return result

    monkeypatch.setattr(analysis, "_signature", changed)
    with pytest.raises(CompilationError):
        prove(analysis, selection, roots, modules)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


@pytest.mark.parametrize("sealed", [False, True])
def test_source_mutation_during_rhs_typing_is_hard_source_failure(tmp_path, monkeypatch, sealed):
    path, analysis, _routine, selection, roots, modules = fixture(tmp_path)
    if sealed:
        analysis.seal_session()
    signature = analysis._signature

    def changed(scope, node, *args, **kwargs):
        result = signature(scope, node, *args, **kwargs)
        path.write_text(path.read_text() + "! changed during original scalar RHS typing\n")
        return result

    monkeypatch.setattr(analysis, "_signature", changed)
    with pytest.raises(CompilationError, match="source changed"):
        prove(analysis, selection, roots, modules)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


GENERIC = """interface assignment(=)
module procedure assign
end interface"""
ASSIGN = """subroutine assign(left,right)
real(8),allocatable,intent(inout)::left(:,:,:)
logical,intent(in)::right
deallocate(left)
end subroutine"""
ASSIGNMENT_MODULE = "module assignments\n" + GENERIC + "\ncontains\n" + ASSIGN + "\nend module\n"


@pytest.mark.parametrize("route", ["local", "host", "use", "reexport"])
@pytest.mark.parametrize("sealed", [False, True])
def test_visible_defined_assignment_rejects_helper_and_complete_consumer(tmp_path, route, sealed):
    options = {"module_procedures": ASSIGN}
    if route == "local":
        options["routine_spec"] = GENERIC
    elif route == "host":
        options["module_spec"] = GENERIC
    else:
        options = {"module_before": ASSIGNMENT_MODULE,
                   "module_spec": "use assignments,only:assignment(=)"}
        if route == "reexport":
            options["module_before"] += "module wrapper\nuse assignments\nend module\n"
            options["module_spec"] = "use wrapper"
    _path, analysis, routine, selection, roots, modules = fixture(tmp_path, **options)
    if sealed:
        analysis.seal_session()
    statement = next(node for node in routine.execution.content if type(node).__name__ == "Assignment_Stmt")
    with pytest.raises(CompilationError, match="visible defined assignment"):
        scalar_fill_descriptor_effects(analysis, routine, statement)
    with pytest.raises(CompilationError):
        prove(analysis, selection, roots, modules)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


@pytest.mark.parametrize("route", ["wildcard", "only", "reexport"])
def test_unknown_assignment_namespace_is_hard_rejection_for_helper_and_consumer(tmp_path, route):
    options = {"module_spec": "use unknown_assignments"}
    if route == "only":
        options["module_spec"] = "use unknown_assignments,only:assignment(=)"
    elif route == "reexport":
        options = {"module_before": "module wrapper\nuse unknown_assignments\nend module\n",
                   "module_spec": "use wrapper"}
    _path, analysis, routine, selection, roots, modules = fixture(tmp_path, **options)
    statement = next(node for node in routine.execution.content if type(node).__name__ == "Assignment_Stmt")
    with pytest.raises(CompilationError, match="closed-name namespace is unavailable"):
        scalar_fill_descriptor_effects(analysis, routine, statement)
    with pytest.raises(CompilationError, match="closed-name namespace is unavailable"):
        prove(analysis, selection, roots, modules)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


def test_private_nonexported_generic_does_not_poison_intrinsic_assignment(tmp_path):
    module = ASSIGNMENT_MODULE.replace("module assignments\n", "module assignments\nprivate\n")
    _path, analysis, _routine, selection, roots, modules = fixture(
        tmp_path, module_before=module, module_spec="use assignments")
    assert prove(analysis, selection, roots, modules).public()["schema_version"] == 24


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("fault", ["replace_map", "insert_defined"])
def test_final_imported_assignment_namespace_is_retained_before_publication(tmp_path, monkeypatch, sealed, fault):
    _path, analysis, _routine, selection, roots, modules = fixture(
        tmp_path, module_before="module imported\ninteger::marker\nend module\n", module_spec="use imported")
    if sealed:
        analysis.seal_session()
    imported = analysis.modules["imported"]
    validate, authority = _Check.validate_native_intrinsics, analysis._summary_authority
    armed = False

    def complete(check, *args, **kwargs):
        nonlocal armed
        result = validate(check, *args, **kwargs)
        if kwargs.get("rederive", True):
            armed = True
        return result

    def late(*args, **kwargs):
        result = authority(*args, **kwargs)
        if armed:
            if fault == "replace_map":
                object.__setattr__(imported, "generics", dict(imported.generics))
            else:
                dict.__setitem__(imported.generics, "assignment(=)", [("imported", "assign")])
        return result

    monkeypatch.setattr(_Check, "validate_native_intrinsics", complete)
    monkeypatch.setattr(analysis, "_summary_authority", late)
    with pytest.raises(CompilationError):
        prove(analysis, selection, roots, modules)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


def block_local_consumer(tmp_path, sealed):
    from fparser.two.utils import walk

    from compiler.frontend.descriptor_windows import _hash

    path = tmp_path / "block_local.f90"
    path.write_text("""module imported
integer::marker
end module
module fills
contains
subroutine advance()
block
use imported
real(8),allocatable::a(:,:,:)
real(8)::value
a=value
end block
end subroutine
end module
""")
    analysis = SourceEffects([path], operations=2048)
    routine = analysis.routines[ENTRY]
    assignment, = [node for node in walk(routine.execution) if type(node).__name__ == "Assignment_Stmt"]
    scope = analysis.source_scope_for(assignment, routine.scope)
    assert scope is not routine.scope
    assert all(analysis._binding(scope, name).declaring_scope is scope for name in ("a", "value"))
    if sealed:
        analysis.seal_session()
    authority = _hash(analysis._summary_authority())
    check = _Check(analysis)
    actuals = check.native_intrinsic(routine, assignment, scalar_fill_descriptor_effects,
                                    names=("assignment(=)",))
    assert actuals == (assignment.items[0], assignment.items[2])
    record, = check.intrinsics.values()
    assert record["intrinsic"] == "scalar_array_assignment"
    assert record["target"]["resource"] == scope.bindings["a"].root
    assert len(check.intrinsic_tokens) == 1
    assert not check.private_allocations
    return analysis, check, authority


@pytest.mark.parametrize("sealed", [False, True])
def test_block_local_assignment_namespace_positive_consumes_scalar_requirement(tmp_path, sealed):
    analysis, check, authority = block_local_consumer(tmp_path, sealed)
    check.publish(ENTRY, authority)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("fault", ["replace_map", "insert_defined"])
def test_final_block_local_assignment_namespace_mutation_rejects(tmp_path, monkeypatch, sealed, fault):
    analysis, check, authority = block_local_consumer(tmp_path, sealed)
    imported = analysis.modules["imported"]
    validate, summary = _Check.validate_native_intrinsics, analysis._summary_authority
    armed = fired = False

    def complete(current, *args, **kwargs):
        nonlocal armed
        result = validate(current, *args, **kwargs)
        if kwargs.get("rederive", True):
            armed = True
        return result

    def late(*args, **kwargs):
        nonlocal fired
        result = summary(*args, **kwargs)
        if armed:
            fired = True
            if fault == "replace_map":
                object.__setattr__(imported, "generics", dict(imported.generics))
            else:
                dict.__setitem__(imported.generics, "assignment(=)", [("imported", "assign")])
        return result

    monkeypatch.setattr(_Check, "validate_native_intrinsics", complete)
    monkeypatch.setattr(analysis, "_summary_authority", late)
    with pytest.raises(CompilationError, match="namespace attachment changed"):
        check.publish(ENTRY, authority)
    assert fired
    assert not getattr(analysis, "_descriptor_window_proofs", {})
