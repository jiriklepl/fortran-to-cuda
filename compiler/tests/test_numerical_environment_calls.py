"""Environment scans retain original CALLs without walking arithmetic trees."""

from copy import copy

import pytest
from fparser.common.readfortran import FortranStringReader
from fparser.two import Fortran2003 as F
from fparser.two.parser import ParserFactory
from fparser.two.utils import walk

from compiler._legacy.config import LegacyCompilerOptions as CompilerOptions
from compiler._legacy.config import LegacyOffloadConfig as OffloadConfig
from compiler.frontend import numerical_preparation
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.numerical_preparation import NumericalPreparationProvider, NumericalPreparationUnsupported
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

ENTRY = "environment_route::evolve"
OPTIONS = CompilerOptions(gpu_policy="sections", memory_model="scoped", scope_execution="reached")
CONFIG = OffloadConfig(policy="sections", scope_execution="reached")
STRUCTURAL_SOURCE = """module nested_calls
type receiver
contains
procedure,nopass::ping=>bound_ping
end type
contains
subroutine traverse(flag,n,x,object)
logical,intent(in)::flag
integer,intent(in)::n
real,intent(inout)::x
type(receiver),intent(in)::object
integer::i,j
call notice(1)
if(flag) call notice(2)
if(flag)then
do i=1,n
associate(value=>x)
select case(i)
case(1)
call notice(3)
case default
block
call notice(4)
end block
end select
end associate
enddo
endif
do 100 i=1,n
100 call notice(5)
do 110 i=1,n
110 if(flag) call notice(6)
do 120 i=1,n
do 120 j=1,n
120 call notice(7)
call object%ping()
call notice(8)
end subroutine
subroutine notice(value)
integer,intent(in)::value
end subroutine
subroutine bound_ping()
end subroutine
end module
"""


def execution(source):
    tree = ParserFactory().create(std="f2008")(FortranStringReader(source, ignore_comments=False))
    return walk(tree, F.Execution_Part)[0]


def environment_fixture(directory, *, ieee=True):
    path = directory / "environment.f90"
    operation = "call set_mode(ieee_invalid,.false.)" if ieee else "continue"
    path.write_text("""module environment_exports
use,intrinsic::ieee_exceptions,only:exposed=>ieee_set_halting_mode,ieee_invalid
integer,parameter::marker=1
end module
module environment_route
use environment_exports,only:set_mode=>exposed,ieee_invalid
contains
subroutine evolve(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(out)::b(:)
integer,intent(in)::n
integer::i
if(.false.) call update_mode()
!$omp parallel do private(i)
do i=1,n
b(i)=a(i)+1.0_8
enddo
!$omp end parallel do
end subroutine
subroutine update_mode()
""" + operation + """
end subroutine
end module
""")
    analysis = SourceEffects([path])
    groups = grouped_original_sequence(analysis.routines[ENTRY].execution.content, {})
    selection = next(group.original_nodes for group in groups if isinstance(group, NativeGroupCandidate))
    provider = NumericalPreparationProvider(analysis, options=OPTIONS, config=CONFIG, environment_root=ENTRY)
    return path, analysis, provider, selection


def test_structural_iterator_retains_exact_call_identity_and_original_order():
    body = execution(STRUCTURAL_SOURCE)
    expected = walk(body, F.Call_Stmt)
    budget = numerical_preparation._Budget()
    actual = tuple(numerical_preparation._source_calls(body, budget))
    assert len(actual) == len(expected) == 9
    assert all(left is right for left, right in zip(actual, expected, strict=True))
    assert [str(call.items[0]).lower().replace(" ", "") for call in actual] == ["notice"] * 7 + ["object%ping", "notice"]
    assert str(actual[4].item.label) == "100"
    inline, labeled_inline = walk(body, F.If_Stmt)
    assert inline.items[1] is actual[1]
    assert labeled_inline.items[1] is actual[5]
    kinds = {type(node).__name__ for node in walk(body)}
    assert {"Associate_Construct", "Case_Construct", "Block_Construct"} <= kinds
    assert budget.visits < len(walk(body))


def test_call_free_expression_size_does_not_spend_environment_syntax_budget(monkeypatch):
    expression = "+".join("x" for _ in range(96))
    source = "subroutine arithmetic(x)\nreal::x\n" + ("x=" + expression + "\n") * 16 + "end subroutine\n"
    body = execution(source)
    budget = numerical_preparation._Budget()
    assert tuple(numerical_preparation._source_calls(body, budget)) == ()
    assert budget.visits == len(body.content) + 1
    assert len(walk(body)) > 20 * budget.visits
    monkeypatch.setattr(numerical_preparation, "_SCAN_LIMIT", budget.visits)
    assert tuple(numerical_preparation._source_calls(body, numerical_preparation._Budget())) == ()
    with pytest.raises(CompilationError, match="source-scan budget exhausted"):
        tuple(numerical_preparation._walk(body, numerical_preparation._Budget()))


@pytest.mark.parametrize("limit", ["local", "parent"])
def test_structural_visits_keep_local_and_parent_exhaustion_hard(monkeypatch, limit):
    body = execution(STRUCTURAL_SOURCE)
    charged = []

    def consume(amount):
        charged.append(amount)
        if limit == "parent" and len(charged) == 4:
            raise CompilationError("parent environment budget exhausted")

    if limit == "local":
        monkeypatch.setattr(numerical_preparation, "_SCAN_LIMIT", 3)
    budget = numerical_preparation._Budget(consume)
    with pytest.raises(CompilationError, match="budget exhausted") as failure:
        tuple(numerical_preparation._source_calls(body, budget))
    assert not isinstance(failure.value, NumericalPreparationUnsupported)
    assert charged == [1] * 4
    assert budget.visits == 4


def test_inactive_original_calls_keep_the_same_registered_environment_proof(tmp_path, monkeypatch):
    _path, analysis, provider, selection = environment_fixture(tmp_path)
    prepared = provider.prepare(analysis, ENTRY, selection)
    assert prepared.requires_original_team_environment_preflight
    assert prepared.function.requires_numerical_environment
    assert prepared.extraction.numerical_environment_required

    def full_walk_calls(node, budget):
        return (item for item in numerical_preparation._walk(node, budget)
                if type(item).__name__ == "Call_Stmt")

    # Compare with the previous expression-walking route on identical original
    # authority; neither route folds away the source's inactive IF action.
    with monkeypatch.context() as previous:
        previous.setattr(numerical_preparation, "_source_calls", full_walk_calls)
        assert provider.prepare(analysis, ENTRY, selection) is prepared
        assert prepared.validate(analysis, ENTRY, selection) is prepared


def test_environment_queries_check_each_module_twice_without_rechecking_overlapping_callees(tmp_path, monkeypatch):
    _path, analysis, _provider, _selection = environment_fixture(tmp_path, ieee=False)
    modules = tuple(scope.node for scope in analysis.modules.values())
    snapshot_type = type(analysis._source_syntax)
    original_layout = snapshot_type._layout
    original_verify = analysis.inputs.verify
    original_require = analysis._require_original_routine
    layouts, verifications, routines = [], [], []

    def verify():
        verifications.append(True)
        original_verify()

    def require(procedure, *, _syntax):
        routines.append(procedure)
        return original_require(procedure, _syntax=_syntax)

    def layout(snapshot, node):
        if snapshot is analysis._source_syntax:
            layouts.append(node)
        return original_layout(snapshot, node)

    monkeypatch.setattr(snapshot_type, "_layout", layout)
    monkeypatch.setattr(analysis.inputs, "verify", verify)
    monkeypatch.setattr(analysis, "_require_original_routine", require)
    for demands in (1, 2):
        assert not numerical_preparation._environment_required(analysis, ENTRY, numerical_preparation._Budget())
        assert len(verifications) == 2 * demands
        assert routines.count(ENTRY) == 2 * demands
        assert routines.count("environment_route::update_mode") == 2 * demands
        assert len(layouts) == 2 * demands * len(modules)
        assert all(sum(node is module for node in layouts) == 2 * demands for module in modules)


def test_public_original_requirement_keeps_fresh_global_and_exact_routine_checks(tmp_path, monkeypatch):
    _path, analysis, _provider, _selection = environment_fixture(tmp_path, ieee=False)
    original_verify = analysis.inputs.verify
    verifications = []

    def verify():
        verifications.append(True)
        original_verify()

    monkeypatch.setattr(analysis.inputs, "verify", verify)
    for procedure in (ENTRY, "environment_route::update_mode"):
        assert analysis._require_original(procedure) is analysis.routines[procedure]
    assert len(verifications) == 2

    routine = analysis.routines["environment_route::update_mode"]
    routine.execution = copy(routine.execution)
    with pytest.raises(CompilationError, match="original source-backed procedure authority"):
        analysis._require_original(routine.qualified)
    assert len(verifications) == 3


@pytest.mark.parametrize("route", ["later_callee", "early_ieee", "procedure_limit", "operation_limit"])
def test_budget_callback_syntax_mutation_cannot_publish_any_environment_result(tmp_path, monkeypatch, route):
    _path, analysis, _provider, _selection = environment_fixture(tmp_path)
    if route in {"procedure_limit", "operation_limit"}:
        setattr(analysis, route, 1)
    call, = walk(analysis.routines["environment_route::update_mode"].execution, F.Call_Stmt)
    assignment, = walk(analysis.routines[ENTRY].execution, F.Assignment_Stmt)
    original_require = analysis._require_original_routine
    root_checked, changed, charged = False, False, []

    def require(procedure, *, _syntax):
        nonlocal root_checked
        routine = original_require(procedure, _syntax=_syntax)
        if procedure == ENTRY:
            root_checked = True
        return routine

    def consume(amount):
        nonlocal changed
        charged.append(amount)
        if root_checked and not changed:
            changed = True
            if route == "later_callee":
                # The module was checked before this later callee is visited.
                # Its changed call target would otherwise turn True into False.
                call.items = (F.Name("different_mode"), call.items[1])
            else:
                # The IEEE and limit paths return True early, but still must
                # reject a changed original body before publishing that result.
                assignment.items = (*assignment.items[:2], F.Int_Literal_Constant("0"))

    monkeypatch.setattr(analysis, "_require_original_routine", require)
    with pytest.raises(CompilationError, match="original.*authority") as failure:
        numerical_preparation._environment_required(analysis, ENTRY, numerical_preparation._Budget(consume))
    assert not isinstance(failure.value, NumericalPreparationUnsupported)
    assert changed
    assert charged
    assert all(amount == 1 for amount in charged)


@pytest.mark.parametrize("change", ["source", "module", "provenance"])
def test_mid_query_global_authority_mutation_cannot_publish_early_true(tmp_path, monkeypatch, change):
    path, analysis, _provider, _selection = environment_fixture(tmp_path)
    original_require = analysis._require_original_routine
    root_checked, changed = False, False

    def require(procedure, *, _syntax):
        nonlocal root_checked
        routine = original_require(procedure, _syntax=_syntax)
        if procedure == ENTRY:
            root_checked = True
        return routine

    def consume(_amount):
        nonlocal changed
        if not root_checked or changed:
            return
        changed = True
        if change == "source":
            path.write_text(path.read_text() + "! changed during environment query\n")
        elif change == "module":
            declaration = next(node for node in walk(analysis.modules["environment_exports"].node,
                                                     F.Type_Declaration_Stmt) if "marker" in str(node).lower())
            literal, = walk(declaration, F.Int_Literal_Constant)
            literal.items = ("2", literal.items[1])
        else:
            call, = walk(analysis.routines["environment_route::update_mode"].execution, F.Call_Stmt)
            call.item.fort_original_span = (1, 1)

    monkeypatch.setattr(analysis, "_require_original_routine", require)
    with pytest.raises(CompilationError) as failure:
        numerical_preparation._environment_required(analysis, ENTRY, numerical_preparation._Budget(consume))
    assert not isinstance(failure.value, NumericalPreparationUnsupported)
    assert changed


@pytest.mark.parametrize("change", ["reexport", "binding"])
def test_mid_query_consulted_namespace_mutation_cannot_publish_early_true(tmp_path, monkeypatch, change):
    _path, analysis, _provider, _selection = environment_fixture(tmp_path)
    original_require = analysis._require_original_routine
    root_checked, changed = False, False

    def require(procedure, *, _syntax):
        nonlocal root_checked
        routine = original_require(procedure, _syntax=_syntax)
        if procedure == ENTRY:
            root_checked = True
        return routine

    def consume(_amount):
        nonlocal changed
        if root_checked and not changed:
            changed = True
            scope = analysis.modules["environment_exports"]
            if change == "reexport":
                # Both names require the same True result, but changed
                # namespace authority still cannot publish that result.
                scope.intrinsic_procedures["exposed"] = "$intrinsic::ieee_exceptions::ieee_get_halting_mode"
            else:
                scope.bindings["marker"] = copy(scope.bindings["marker"])

    monkeypatch.setattr(analysis, "_require_original_routine", require)
    with pytest.raises(CompilationError, match="original resolution authority changed") as failure:
        numerical_preparation._environment_required(analysis, ENTRY, numerical_preparation._Budget(consume))
    assert not isinstance(failure.value, NumericalPreparationUnsupported)
    assert changed


@pytest.mark.parametrize("change", ["module", "callee", "imports", "binding"])
def test_each_environment_query_rechecks_original_syntax_from_a_fresh_checkpoint(tmp_path, change):
    _path, analysis, _provider, _selection = environment_fixture(tmp_path, ieee=False)
    assert not numerical_preparation._environment_required(analysis, ENTRY, numerical_preparation._Budget())
    if change == "module":
        statement, = walk(analysis.modules["environment_exports"].node, F.Module_Stmt)
        statement.items = (*statement.items[:-1], F.Name("changed_exports"))
    elif change == "callee":
        statement, = walk(analysis.routines[ENTRY].execution, F.Assignment_Stmt)
        statement.items = (*statement.items[:2], F.Int_Literal_Constant("0"))
    elif change == "imports":
        analysis.modules["environment_route"].imports["set_mode"] = ("environment_exports", "different_export")
    else:
        scope = analysis.modules["environment_exports"]
        scope.bindings["marker"] = copy(scope.bindings["marker"])
    with pytest.raises(CompilationError, match="original.*authority") as failure:
        numerical_preparation._environment_required(analysis, ENTRY, numerical_preparation._Budget())
    assert not isinstance(failure.value, NumericalPreparationUnsupported)


@pytest.mark.parametrize("change", ["source", "call", "reexport"])
def test_next_environment_demand_rejects_fresh_original_authority_mutation(tmp_path, change):
    path, analysis, provider, selection = environment_fixture(tmp_path)
    prepared = provider.prepare(analysis, ENTRY, selection)
    if change == "source":
        path.write_text(path.read_text() + "! changed source\n")
    elif change == "call":
        call, = walk(analysis.routines["environment_route::update_mode"].execution, F.Call_Stmt)
        call.items = (F.Name("different_mode"), call.items[1])
    else:
        analysis.modules["environment_exports"].intrinsic_procedures["exposed"] = (
            "$intrinsic::ieee_exceptions::ieee_get_halting_mode")
    with pytest.raises(CompilationError) as failure:
        prepared.validate(analysis, ENTRY, selection)
    assert not isinstance(failure.value, NumericalPreparationUnsupported)


@pytest.mark.parametrize("limit", ["procedure_limit", "operation_limit"])
def test_original_procedure_and_call_limits_remain_conservative(tmp_path, limit):
    _path, analysis, _provider, _selection = environment_fixture(tmp_path, ieee=False)
    assert not numerical_preparation._environment_required(analysis, ENTRY, numerical_preparation._Budget())
    setattr(analysis, limit, 1)
    if limit == "operation_limit":
        # Both calls are original and counted, even when one is intrinsic.
        path = tmp_path / "environment.f90"
        text = path.read_text().replace("continue\nend subroutine", "call random_seed()\nend subroutine")
        path.write_text(text)
        analysis = SourceEffects([path], operations=1)
    assert numerical_preparation._environment_required(analysis, ENTRY, numerical_preparation._Budget())
