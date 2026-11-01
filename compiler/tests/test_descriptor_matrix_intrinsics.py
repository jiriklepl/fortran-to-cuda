"""Native matrix reads grant descriptor effects without numerical admission."""
from copy import copy
from pathlib import Path

import pytest
from fparser.two.utils import walk

from compiler.frontend.descriptor_intrinsics import matrix_descriptor_effects
from compiler.ir import CompilationError
from compiler.ir.intrinsics import INTRINSICS
from compiler.tests.test_descriptor_window_proofs import ENTRY, fixture, prove


def original(tmp_path, *, expression="matmul(x,y)", left="real(8)::x(3,3)",
             right="real(8)::y(3,3)", extra="", prefix="", suffix=""):
    _path, analysis, _graph, selected = fixture(
        tmp_path,
        child_decl=f"real(8),intent(inout)::a(:)\n{left}\n{right}\nreal(8)::z(3,3)\n{prefix}",
        child=f"z={expression}\na(1)=a(1)+1.0_8\n{suffix}", extra=extra)
    routine = analysis.routines["intervals::read_modify"]
    nodes = [node for node in walk(routine.execution)
             if type(node).__name__ == "Intrinsic_Function_Reference"
             and str(node.items[0]).lower() in {"matmul", "transpose"}]
    return analysis, routine, nodes, selected


@pytest.mark.parametrize("kind", [4, 8])
@pytest.mark.parametrize(("left", "right", "rank"), [("3,3", "3,3", 2), ("3,3", "3", 1), ("3", "3,3", 1)])
def test_matching_real_kinds_and_legal_matrix_ranks(tmp_path, kind, left, right, rank):
    analysis, routine, nodes, _selected = original(tmp_path, left=f"real({kind})::x({left})",
                                                  right=f"real({kind})::y({right})")
    before = analysis.summarize(routine.qualified)
    actuals, record = matrix_descriptor_effects(analysis, routine, nodes[0])
    assert [str(node) for node in actuals] == ["x", "y"]
    assert record["result"] == {"type": "real", "kind": kind, "rank": rank}
    assert [item["effect"] for item in record["arguments"]] == ["read", "read"]
    assert record["descriptor_evidence_only"]
    assert record["requires_original_execution"]
    for flag in ("storage_writes", "allocation_or_escape", "execution_authorized", "capture_authorized",
                 "entry_definition_authority", "gpu_legality_established", "numerical_lowering_authorized", "motion_authorized"):
        assert record[flag] is False
    assert "matmul" not in INTRINSICS
    assert "transpose" not in INTRINSICS
    assert analysis.summarize(routine.qualified) == before
    assert not before["complete"]


@pytest.mark.parametrize("expression", ["transpose(x)", "transpose(matrix=x)",
                                       "matmul(transpose(x),y)", "matmul(transpose(transpose(x)),y)",
                                       "matmul((x),y)"])
def test_exact_native_transpose_and_parentheses_retain_type_and_rank(tmp_path, expression):
    analysis, routine, nodes, selected = original(tmp_path, expression=expression)
    _actuals, record = matrix_descriptor_effects(analysis, routine, nodes[0])
    assert record["result"] == {"type": "real", "kind": 8, "rank": 2}
    public = prove(analysis, selected).public()
    assert public["schema_version"] == 18
    records = public["native_intrinsic_descriptor_requirements"]
    assert len(records) == len(nodes)
    assert {item["source_expression"] for item in records} == {str(node) for node in nodes}
    assert not public["execution_authorized"]
    assert not public["entry_definition_authority"]


def test_reordered_keywords_retain_original_actual_order(tmp_path):
    analysis, routine, nodes, selected = original(tmp_path, expression="matmul(matrix_b=y,matrix_a=x)")
    actuals, record = matrix_descriptor_effects(analysis, routine, nodes[0])
    assert [str(node) for node in actuals] == ["y", "x"]
    assert [item["formal"] for item in record["arguments"]] == ["matrix_b", "matrix_a"]
    assert prove(analysis, selected).validate(analysis, ENTRY, selected)


@pytest.mark.parametrize(("expression", "left", "right"), [
    ("matmul(x,y)", "integer::x(3,3)", "integer::y(3,3)"),
    ("matmul(x,y)", "logical::x(3,3)", "logical::y(3,3)"),
    ("matmul(x,y)", "real(4)::x(3,3)", "real(8)::y(3,3)"),
    ("matmul(x,y)", "real(8)::x(3)", "real(8)::y(3)"),
    ("matmul(x,y)", "real(8)::x", "real(8)::y(3,3)"),
    ("matmul(x,y)", "real(8)::x(3,3,3)", "real(8)::y(3,3)"),
    ("transpose(x)", "real(8)::x(3)", "real(8)::y(3,3)"),
    ("matmul(x(:,1),y)", "real(8)::x(3,3)", "real(8)::y(3,3)"),
    ("matmul(x+y,y)", "real(8)::x(3,3)", "real(8)::y(3,3)"),
    ("matmul(absent(),y)", "real(8)::x(3,3)", "real(8)::y(3,3)"),
    ("matmul(x)", "real(8)::x(3,3)", "real(8)::y(3,3)"),
    ("matmul(x,y,x)", "real(8)::x(3,3)", "real(8)::y(3,3)"),
    ("matmul(x,matrix_a=y)", "real(8)::x(3,3)", "real(8)::y(3,3)"),
    ("matmul(matrix_a=x,y)", "real(8)::x(3,3)", "real(8)::y(3,3)"),
    ("transpose(matrix_b=x)", "real(8)::x(3,3)", "real(8)::y(3,3)"),
])
def test_unproved_type_rank_expression_and_actual_association_reject(tmp_path, expression, left, right):
    def classify():
        analysis, routine, nodes, selected = original(tmp_path, expression=expression, left=left, right=right)
        matrix_descriptor_effects(analysis, routine, nodes[0])
        prove(analysis, selected)

    with pytest.raises(CompilationError):
        classify()


@pytest.mark.parametrize("name", ["matmul", "transpose"])
def test_external_and_unknown_export_names_cannot_borrow_intrinsic_identity(tmp_path, name):
    expression = "matmul(x,y)" if name == "matmul" else "transpose(x)"
    analysis, routine, nodes, selected = original(tmp_path, expression=expression, prefix=f"external::{name}")
    with pytest.raises(CompilationError, match="shadowed"):
        matrix_descriptor_effects(analysis, routine, nodes[0])
    with pytest.raises(CompilationError):
        prove(analysis, selected)


def test_unknown_wildcard_rejects_intrinsic_identity(tmp_path):
    analysis, routine, nodes, _selected = original(tmp_path, prefix="use missing")
    with pytest.raises(CompilationError):
        matrix_descriptor_effects(analysis, routine, nodes[0])


@pytest.mark.parametrize("name", ["matmul", "transpose"])
def test_source_override_including_nested_transpose_rejects_intrinsic_identity(tmp_path, name):
    parameters = "x,y" if name == "matmul" else "x"
    override = f"function {name}({parameters}) result(z)\nreal(8)::x(3,3),z(3,3)"
    if name == "matmul":
        override += "\nreal(8)::y(3,3)"
    override += "\nz=x\nend function"
    analysis, routine, nodes, selected = original(tmp_path, expression="matmul(transpose(x),y)", extra=override)
    with pytest.raises(CompilationError, match="resolution"):
        matrix_descriptor_effects(analysis, routine, nodes[0])
    with pytest.raises(CompilationError):
        prove(analysis, selected)


def test_detached_expression_and_copied_routine_reject(tmp_path):
    analysis, routine, nodes, _selected = original(tmp_path)
    with pytest.raises(CompilationError, match="exact original"):
        matrix_descriptor_effects(analysis, routine, copy(nodes[0]))
    with pytest.raises(CompilationError):
        matrix_descriptor_effects(analysis, copy(routine), nodes[0])


@pytest.mark.parametrize("suffix", ["call absent(a)", "deallocate(a)"])
def test_full_consumer_still_rejects_later_escape_or_descriptor_mutation(tmp_path, suffix):
    _path, analysis, _graph, selected = fixture(tmp_path,
        child_decl="real(8),allocatable,intent(inout)::a(:)\nreal(8)::x(3,3),y(3,3),z(3,3)",
        child=f"z=matmul(transpose(x),y)\n{suffix}")
    with pytest.raises(CompilationError):
        prove(analysis, selected)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


def test_old_window_without_matrix_reads_keeps_its_schema(tmp_path):
    _path, analysis, _graph, selected = fixture(tmp_path)
    assert prove(analysis, selected).public()["schema_version"] == 6


def consumed_fixture(tmp_path, intrinsic):
    if intrinsic == "norm2":
        from compiler.tests.test_descriptor_intrinsics import original as norm2_original

        analysis, routine, node, selected = norm2_original(tmp_path)
        return analysis, routine, node, selected, routine.scope.bindings["v"]
    analysis, routine, nodes, selected = original(tmp_path, expression="matmul(transpose(x),y)")
    return analysis, routine, nodes[0], selected, routine.scope.bindings["x"]


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("intrinsic", ["matrix", "norm2"])
@pytest.mark.parametrize("fault", ["kind", "root", "namespace", "copied_binding", "syntax", "record"])
def test_late_consumed_declaration_and_publication_mutation_rejects(tmp_path, monkeypatch, sealed, intrinsic, fault):
    from compiler.frontend.descriptor_windows import _Check

    analysis, routine, node, selected, binding = consumed_fixture(tmp_path, intrinsic)
    if sealed:
        analysis.seal_session()
    publish = _Check.publish

    def changed(check, *args):
        if fault in {"kind", "root"}:
            field = fault
            saved = getattr(binding, field)
            object.__setattr__(binding, field, 16 if fault == "kind" else "forged::operand")
            restore = object.__setattr__, (binding, field, saved)
        elif fault == "namespace":
            saved = routine.scope.bindings
            object.__setattr__(routine.scope, "bindings", dict(saved))
            restore = object.__setattr__, (routine.scope, "bindings", saved)
        elif fault == "copied_binding":
            namespace, name = routine.scope.bindings, binding.name
            dict.__setitem__(namespace, name, copy(binding))
            restore = dict.__setitem__, (namespace, name, binding)
        elif fault == "syntax":
            saved = node.items
            object.__setattr__(node, "items", (saved[0], copy(saved[1])))
            restore = object.__setattr__, (node, "items", saved)
        else:
            item = next(iter(check.intrinsics.values()))["arguments"][0]
            saved = item["kind"]
            item["kind"] = 4
            restore = item.__setitem__, ("kind", saved)
        try:
            return publish(check, *args)
        finally:
            restore[0](*restore[1])

    monkeypatch.setattr(_Check, "publish", changed)
    with pytest.raises(CompilationError):
        prove(analysis, selected)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("intrinsic", ["matrix", "norm2"])
def test_early_valid_kind_reseed_cannot_replace_canonical_declaration(tmp_path, monkeypatch, sealed, intrinsic):
    from compiler.frontend.descriptor_windows import _Check

    analysis, routine, _node, selected, binding = consumed_fixture(tmp_path, intrinsic)
    if sealed:
        analysis.seal_session()
    classify = _Check.native_intrinsic

    def reseed(check, *args):
        saved = binding.kind
        object.__setattr__(binding, "kind", 4)
        try:
            return classify(check, *args)
        finally:
            object.__setattr__(binding, "kind", saved)

    monkeypatch.setattr(_Check, "native_intrinsic", reseed)
    with pytest.raises(CompilationError):
        prove(analysis, selected)


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("intrinsic", ["matrix", "norm2"])
def test_last_authority_callback_cannot_mutate_an_earlier_consumed_binding(tmp_path, monkeypatch, sealed, intrinsic):
    from compiler.frontend.descriptor_windows import _Check

    analysis, routine, _node, selected, binding = consumed_fixture(tmp_path, intrinsic)
    if sealed:
        analysis.seal_session()
    validate = _Check.validate_native_intrinsics
    authority = analysis._summary_authority
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
            object.__setattr__(binding, "kind", 16)
        return result

    monkeypatch.setattr(_Check, "validate_native_intrinsics", complete)
    monkeypatch.setattr(analysis, "_summary_authority", late)
    with pytest.raises(CompilationError):
        prove(analysis, selected)


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("intrinsic", ["matrix", "norm2"])
def test_signature_callback_cannot_change_its_consumed_operand(tmp_path, monkeypatch, sealed, intrinsic):
    analysis, _routine, _node, selected, binding = consumed_fixture(tmp_path, intrinsic)
    if sealed:
        analysis.seal_session()
    signature = analysis._signature

    def changed(scope, node, *args, **kwargs):
        result = signature(scope, node, *args, **kwargs)
        if type(node).__name__ == "Name" and analysis._binding(scope, node) is binding:
            object.__setattr__(binding, "kind", 16)
        return result

    monkeypatch.setattr(analysis, "_signature", changed)
    with pytest.raises(CompilationError):
        prove(analysis, selected)


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("intrinsic", ["matrix", "norm2"])
def test_unchanged_sealed_and_unsealed_consumers_accept(tmp_path, sealed, intrinsic):
    analysis, _routine, _node, selected, _binding = consumed_fixture(tmp_path, intrinsic)
    if sealed:
        analysis.seal_session()
    assert prove(analysis, selected).validate(analysis, ENTRY, selected)


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("intrinsic", ["matrix", "norm2"])
@pytest.mark.parametrize("fault", ["externals", "imports", "parent", "kind_values", "parser", "binding_insert",
                                  "module_registry", "routine_registry", "authority_registry", "session", "file"])
def test_final_callback_cannot_change_intrinsic_resolution_namespaces(tmp_path, monkeypatch, sealed, intrinsic, fault):
    from compiler.frontend.descriptor_windows import _Check

    analysis, routine, _node, selected, binding = consumed_fixture(tmp_path, intrinsic)
    if sealed:
        analysis.seal_session()
    validate, authority = _Check.validate_native_intrinsics, analysis._summary_authority
    spelling = "matmul" if intrinsic == "matrix" else "norm2"
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
            scope = routine.scope
            if fault == "externals":
                object.__setattr__(scope, "externals", frozenset((*scope.externals, spelling)))
            elif fault == "imports":
                dict.__setitem__(scope.imports, spelling, ("unavailable", spelling))
            elif fault == "parent":
                object.__setattr__(scope, "parent", None)
            elif fault == "kind_values":
                object.__setattr__(scope.kinds, "values", dict(scope.kinds.values))
            elif fault == "parser":
                original = scope.parent.node
                children = original.content
                object.__setattr__(original, "content", (copy(children[0]), *children[1:]))
            elif fault == "binding_insert":
                dict.__setitem__(scope.bindings, spelling, copy(binding))
            elif fault == "module_registry":
                analysis.modules = dict(analysis.modules)
            elif fault == "routine_registry":
                dict.__setitem__(analysis.routines, routine.qualified, copy(routine))
            elif fault == "authority_registry":
                analysis._source_function_scopes = dict(analysis._source_function_scopes)
            elif fault == "file":
                path = Path(scope.path)
                path.write_bytes(path.read_bytes() + b"\n! changed after final authority callback\n")
            else:
                analysis._source_session = None if sealed else object()
        return result

    monkeypatch.setattr(_Check, "validate_native_intrinsics", complete)
    monkeypatch.setattr(analysis, "_summary_authority", late)
    with pytest.raises(CompilationError):
        prove(analysis, selected)
