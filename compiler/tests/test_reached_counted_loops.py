"""Original serial loops retain one owner and their reached escape semantics."""

from copy import copy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.source_effects import _children, _kind
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.counted_loops import reached_counted_body
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_lexical_source_owner import SOURCE, emit
from compiler.tests.test_source_scopes import FACT

COUNTED_SOURCE = (SOURCE.replace("integer::i", "integer::i,stage")
    .replace("visits=visits+1", "visits=visits+1\ndo stage=1,3")
    .replace("if(escape) then", "if(escape.and.stage==2) then")
    .replace("end subroutine\nend module", "enddo\nend subroutine\nend module"))


def builder_for(directory, source=COUNTED_SOURCE):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "counted.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::"+name: FACT for name in ("a", "b", "out")}}
    return ScopeBuilder([path], "local_owner::step", facts=facts,
                        options=CompilerOptions(),
                        config=OffloadConfig(policy="sections", scope_execution="reached"))


def original_loop(builder):
    return next(node for node in _children(builder.entry.execution)
                if _kind(node) == "Block_Nonlabel_Do_Construct")


def test_reached_counted_body_preserves_header_and_one_invocation_owner(tmp_path):
    path, outputs, report = emit(tmp_path, COUNTED_SOURCE)
    owner, = report["scopes"]
    control, = owner["counted_controls"]
    assert control["header"] == "DO stage = 1, 3"
    assert not control["reopen_after_boundary"]
    assert not control["numerical_authority"]
    assert len(owner["gpu_leaves"]) == len(owner["planning_segments"]) == 2
    text = outputs[report["sources"][str(path)]["replacement"]]
    start = text.index("do stage=1,3")
    assert text.count("do stage=1,3") == 1
    assert "fort_buffer_1 = 0" in text[:start]
    assert "fort_buffer_1 = 0" not in text[start:]
    assert text.count("call opaque(b,n)") == 1
    escape = text.index("if(escape.and.stage==2) then")
    call = text.index("call opaque(b,n)")
    assert "fort_scope_close(" in text[escape:call]
    assert "= .false." in text[escape:call]
    assert "= .true." not in text[escape:call]
    assert text.count("visits=visits+1") == 1


@pytest.mark.parametrize("action", ["exit", "cycle", "return"])
def test_reached_loop_control_closes_before_original_action_once(tmp_path, action):
    source = COUNTED_SOURCE.replace("call opaque(b,n)", action)
    # A loop with no calls can be retained as one complete native operation.
    source = source.replace("visits=visits+1\ndo stage", "visits=visits+1\ncall manage()\ndo stage")
    source = source.replace("do stage=1,3", "do stage=1,3\ncall manage()")
    source = source.replace("end subroutine\nend module", "contains\nsubroutine manage()\nend subroutine\nend subroutine\nend module")
    path, outputs, report = emit(tmp_path, source)
    text = outputs[report["sources"][str(path)]["replacement"]]
    branch = text.index("if(escape.and.stage==2) then")
    position = text.index("\n"+action+"\n", branch)
    assert "fort_scope_close(" in text[branch:position]
    assert text.count("\n"+action+"\n") == 1
    assert report["scopes"][0]["reopen_after_boundary"] is False


@pytest.mark.parametrize("header", ["do while(escape)", "do stage=b(-2),3"])
def test_unproved_control_does_not_authorize_body_traversal(tmp_path, header):
    builder = builder_for(tmp_path, COUNTED_SOURCE.replace("do stage=1,3", header))
    with pytest.raises(CompilationError, match="counted DO|scalar/descriptor-only"):
        reached_counted_body(builder, original_loop(builder))


def test_counted_body_requires_original_registered_node_and_unchanged_source(tmp_path):
    builder = builder_for(tmp_path)
    node = original_loop(builder)
    assert reached_counted_body(builder, node).nodes
    with pytest.raises(CompilationError, match="original node or approved"):
        reached_counted_body(builder, copy(node))
    builder.entry.scope.path.write_text(COUNTED_SOURCE+"! changed source\n")
    with pytest.raises(CompilationError, match="changed|identity|hash"):
        reached_counted_body(builder, node)


@pytest.mark.parametrize("directive", ["!$omp simd", "!$omp single", "!$acc parallel loop", "!$ call outside()"])
def test_unjoined_directives_cannot_be_cut_into_reached_body(tmp_path, directive):
    source = COUNTED_SOURCE.replace("do i=-2,n-3", directive+"\ndo i=-2,n-3", 1)
    builder = builder_for(tmp_path, source)
    with pytest.raises(CompilationError, match="OpenMP|directive"):
        reached_counted_body(builder, original_loop(builder))


def test_backedge_private_value_read_prevents_outlining_away_its_definition(tmp_path):
    source = COUNTED_SOURCE.replace("integer::i,stage", "integer::i,stage\nreal(8)::t")
    source = source.replace("do stage=1,3", "t=0.d0\ndo stage=1,3\nvisits=visits+int(t)")
    source = source.replace("b(i)=2*a(i)+real(i,8)", "t=a(i)+1\nb(i)=t")
    _, _, report = emit(tmp_path, source)
    owner, = report["scopes"]
    # The later out loop can still be GPU work. The earlier loop must keep
    # its original t, which the prefix consumes at the next backedge.
    assert len(owner["gpu_leaves"]) == 1
    assert any("live after" in boundary["reason"] for boundary in report["boundaries"])


@pytest.mark.parametrize("escape", ["cycle", "if(escape) cycle"])
def test_suffix_definition_cannot_hide_next_iteration_read_across_cycle(tmp_path, escape):
    source = COUNTED_SOURCE.replace("integer::i,stage", "integer::i,stage\nreal(8)::t")
    source = source.replace("do stage=1,3", "t=0.d0\ndo stage=1,3\nvisits=visits+int(t)")
    source = source.replace("b(i)=2*a(i)+real(i,8)", "t=a(i)+1\nb(i)=t")
    source = source.replace("if(escape.and.stage==2) then", escape+"\nt=0.d0\nif(escape.and.stage==2) then")
    _, _, report = emit(tmp_path, source)
    assert len(report["scopes"][0]["gpu_leaves"]) == 1
    assert any("live after" in boundary["reason"] for boundary in report["boundaries"])


def test_coordinator_shell_does_not_consume_descendant_generation_budget(tmp_path):
    builder = builder_for(tmp_path)
    builder.inline.prepare(_children(builder.entry.execution))
    assert builder.inline.attempts == 2
    assert builder.inline.operations == 4


def test_pure_helper_loop_keeps_transitive_numerical_extraction(tmp_path):
    source = SOURCE.replace("if(escape) then\ncall opaque(b,n)\nendif\n", "")
    source = source.replace("integer::i", "integer::i\nreal(8)::t")
    source = source.replace("b(i)=2*a(i)+real(i,8)", "call calculate(a(i),t,i)\nb(i)=t")
    source = source.replace("end subroutine\nend module", """end subroutine
pure subroutine calculate(x,y,i)
real(8),intent(in)::x
real(8),intent(out)::y
integer,intent(in)::i
y=2*x+real(i,8)
end subroutine
end module""")
    builder = builder_for(tmp_path, source)
    prepared = builder.inline.prepare(_children(builder.entry.execution))
    assert len(builder.inline.regions) == 2
    first = next(iter(builder.inline.regions.values()))
    assert first.numerical_helpers == ("local_owner::calculate",)
    assert all(_kind(node) != "Block_Nonlabel_Do_Construct" for node in prepared)
