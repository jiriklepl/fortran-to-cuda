"""Original span indexing preserves lexical liveness and source coordinates."""

from hashlib import sha256
from types import SimpleNamespace

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import Base, walk

from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.scopes.region_context import OriginalRegionContext, original_region_context

ENTRY = "renamed_context::advance"
SOURCE = """module renamed_context
contains
subroutine advance(a,b,n,flag)
integer,intent(in)::n
logical,intent(in)::flag
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer::phase,i,counter,tail
real(8)::temporary
counter=0
temporary=0.0_8
do phase=1,3
counter=counter+int(temporary)
if(flag)then
temporary=real(phase,8)
! original context marker
!$omp parallel do private(i)
do i=1,n
b(i)=a(i)+ &
 real(counter,8)
enddo
!$omp end parallel do
counter=counter+i
endif
temporary=temporary+1.0_8
enddo
tail=counter+i
end subroutine
end module
"""


def fixture(directory, source=SOURCE):
    path = directory / "renamed_context.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    return analysis.routines[ENTRY]


def selected_loop(routine, iterator="i"):
    for loop in walk(routine.execution, F.Block_Nonlabel_Do_Construct):
        header = next(node for node in loop.content if isinstance(node, F.Nonlabel_Do_Stmt))
        if str(header).lower().replace(" ", "").startswith("do" + iterator + "="):
            return loop
    raise AssertionError("original selected loop is missing")


def assignment(routine, target):
    return next(node for node in walk(routine.execution, F.Assignment_Stmt)
                if str(node.items[0]).lower() == target)


def legacy_context(routine, sources):
    """The previous full-walk algorithm, kept only as an identity oracle."""
    visits = 0

    def prepared_span(node, *, optional=False):
        nonlocal visits
        spans = []
        for item in walk(node):
            visits += 1
            if getattr(item, "item", None) is not None:
                spans.append(item.item.span)
        if not spans or any(span is None for span in spans):
            if optional:
                return None
            raise CompilationError("source context has no prepared ordering span")
        return min(span[0] for span in spans), max(span[1] for span in spans)

    spans = [prepared_span(node) for node in sources]
    if not spans:
        raise CompilationError("original numerical context requires a nonempty selection")
    first, last = min(span[0] for span in spans), max(span[1] for span in spans)
    before, after, cyclic = [], [], []

    def visit(node):
        span = prepared_span(node, optional=True)
        if span is None:
            return
        low, high = span
        if high < first:
            before.append(node)
        elif low > last:
            after.append(node)
        elif type(node).__name__ in {"If_Construct", "Block_Nonlabel_Do_Construct"}:
            for child in node.content:
                visit(child)
            if type(node).__name__ == "Block_Nonlabel_Do_Construct":
                for child in node.content:
                    low, high = prepared_span(child)
                    if high < first and type(child).__name__ not in {"Nonlabel_Do_Stmt", "Comment"}:
                        cyclic.append(child)

    for node in routine.execution.content:
        visit(node)
    return OriginalRegionContext(tuple(before), tuple(after), tuple(cyclic)), visits


def assert_same_context(actual, expected):
    for field in ("preceding", "following", "cyclic_following"):
        left, right = getattr(actual, field), getattr(expected, field)
        assert len(left) == len(right)
        assert all(a is b for a, b in zip(left, right, strict=True))


def test_nested_original_context_preserves_exact_paths_comments_and_continuations(tmp_path):
    routine = fixture(tmp_path)
    loop = selected_loop(routine)
    expected, _visits = legacy_context(routine, (loop,))
    actual = original_region_context(routine, (loop,))
    assert_same_context(actual, expected)
    prefix = next(node for node in walk(routine.execution, F.Assignment_Stmt)
                  if str(node).lower().replace(" ", "") == "counter=counter+int(temporary)")
    tail = assignment(routine, "tail")
    assert any(node is prefix for node in actual.preceding)
    assert any(node is prefix for node in actual.cyclic_following)
    assert any(node is tail for node in actual.following)
    comments = [node for node in walk(routine.execution) if type(node).__name__ == "Comment"]
    marker = next(node for node in comments if "original context marker" in str(node).lower())
    opening = next(node for node in comments if str(node).strip().lower().startswith("!$omp parallel do"))
    ending = next(node for node in comments if str(node).strip().lower().startswith("!$omp end parallel do"))
    assert any(node is marker for node in actual.preceding)
    assert any(node is opening for node in actual.preceding)
    assert any(node is ending for node in actual.following)
    continued = assignment(routine, "b(i)")
    assert continued.item.span[1] > continued.item.span[0]
    assert actual.public() == expected.public()


def test_included_unmapped_suffix_uses_prepared_order_and_remains_live(tmp_path):
    original = tmp_path / "original.f90"
    prepared = tmp_path / "prepared.f90"
    included = tmp_path / "observe.inc"
    original.write_text(SOURCE.replace("tail=counter+i", '#include "observe.inc"'))
    included.write_text("tail=counter+ &\n i\n")
    lines, mapping = [], []
    for number, line in enumerate(original.read_text().splitlines(), 1):
        if line.startswith("#include"):
            lines.extend(included.read_text().splitlines())
            mapping.extend((None, None))
        else:
            lines.append(line)
            mapping.append(number)
    prepared.write_text("\n".join(lines) + "\n")
    document = {"schema_version": 1, "preserves_source_order": True, "configuration": {"defines": []},
                "source_inputs": {str(original): sha256(original.read_bytes()).hexdigest()},
                "dependencies": {str(included): sha256(included.read_bytes()).hexdigest()},
                "entries": [{"source": str(original), "path": str(prepared),
                             "sha256": sha256(prepared.read_bytes()).hexdigest(), "line_map": mapping}]}
    routine = SourceEffects([original], analysis_sources=document).routines[ENTRY]
    loop, tail = selected_loop(routine), assignment(routine, "tail")
    assert tail.item.fort_original_span is None
    assert tail.item.span[1] > tail.item.span[0]
    actual = original_region_context(routine, (loop,))
    expected, _visits = legacy_context(routine, (loop,))
    assert_same_context(actual, expected)
    assert any(node is tail for node in actual.following)


@pytest.mark.parametrize("layout", ["itemless", "own_span_nested", "own_span_none"])
def test_unfamiliar_wrapper_matches_full_walk_for_descendant_spans_and_poison(tmp_path, layout):
    routine = fixture(tmp_path)
    loop, tail = selected_loop(routine), assignment(routine, "tail")

    class OriginalWrapper(Base):
        """Exercise an unfamiliar parser wrapper without changing its child."""

    wrapper = object.__new__(OriginalWrapper)
    wrapper.item = None if layout == "itemless" else selected_loop(routine, "phase").content[-1].item
    if layout == "own_span_none":
        wrapper.items = None
    elif layout == "own_span_nested":
        wrapper.items = (tail.items[0], ([tail],))
    else:
        wrapper.items = (tail.items[0], tail)
    assert getattr(tail.items[0], "item", None) is None
    routine.execution.content[-1] = wrapper
    actual = original_region_context(routine, (loop,))
    expected, _visits = legacy_context(routine, (loop,))
    assert_same_context(actual, expected)
    assert any(node is wrapper for node in actual.following)
    # An absent item contributes no span. An existing item with a None span
    # poisons the aggregate instead of borrowing a sibling's valid bounds.
    descendant = ([tail],) if layout == "own_span_nested" else tail
    wrapper.items = (tail.items[0], descendant, loop.content[-1])
    tail.item.span = None
    actual = original_region_context(routine, (loop,))
    expected, _visits = legacy_context(routine, (loop,))
    assert_same_context(actual, expected)
    assert not any(node is wrapper for node in actual.following)
    with pytest.raises(CompilationError, match="prepared ordering span"):
        original_region_context(routine, (wrapper,))
    with pytest.raises(CompilationError, match="prepared ordering span"):
        original_region_context(routine, (tail.items[0],))


def test_new_context_demand_rechecks_changed_original_span(tmp_path):
    routine = fixture(tmp_path)
    loop, tail = selected_loop(routine), assignment(routine, "tail")
    before = original_region_context(routine, (loop,))
    assert any(node is tail for node in before.following)
    # This helper orders spans; its callers separately reject altered original
    # provenance. A demand-local memo must not hide even a direct span change.
    tail.item.span = (1, 1)
    after = original_region_context(routine, (loop,))
    expected, _visits = legacy_context(routine, (loop,))
    assert_same_context(after, expected)
    assert any(node is tail for node in after.preceding)
    assert not any(node is tail for node in after.following)


def test_context_parent_work_budget_remains_hard(tmp_path):
    routine = fixture(tmp_path)
    charged = []

    def consume(amount):
        charged.append(amount)
        if len(charged) == 7:
            raise CompilationError("parent context work budget exhausted")

    with pytest.raises(CompilationError, match="parent context work budget exhausted"):
        original_region_context(routine, (selected_loop(routine),), consume=consume)
    assert charged == [1] * 7


def test_repeated_original_statement_lookups_keep_local_work_bounded(tmp_path):
    routine = fixture(tmp_path)
    statement = assignment(routine, "tail")
    # A small synthetic container repeats one real original statement. Even
    # cached span lookups are work; repetition cannot evade the existing cap.
    repeated = SimpleNamespace(execution=SimpleNamespace(content=[statement] * 65537))
    charged = []
    with pytest.raises(CompilationError, match="context source-scan budget exhausted"):
        original_region_context(repeated, (statement,), consume=charged.append)
    assert charged == [1] * 65537


def test_large_rhs_is_not_rewalked_for_each_enclosing_context_query(tmp_path):
    expression = "+".join("a(i)" for _ in range(96))
    source = SOURCE.replace("integer::phase,i,counter,tail", "integer::phase,i,counter,tail,p,q,r")
    source = source.replace("do phase=1,3", "do phase=1,3\ndo p=1,2\ndo q=1,2\ndo r=1,2")
    source = source.replace("b(i)=a(i)+ &\n real(counter,8)", "b(i)=" + expression)
    source = source.replace("tail=counter+i", "enddo\nenddo\nenddo\ntail=counter+i")
    routine = fixture(tmp_path, source)
    loop = selected_loop(routine)
    expected, old_visits = legacy_context(routine, (loop,))
    charged = []
    actual = original_region_context(routine, (loop,), consume=charged.append)
    assert_same_context(actual, expected)
    # The implementation may still visit every expression descendant once.
    # This checks reuse across overlapping ancestors, rather than pruning RHSs.
    assert len(charged) < old_visits
    assert all(amount == 1 for amount in charged)


def test_empty_original_selection_still_fails(tmp_path):
    routine = fixture(tmp_path)
    with pytest.raises(CompilationError, match="nonempty selection"):
        original_region_context(routine, ())
