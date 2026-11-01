"""Native continuation liveness distinguishes absent and supplied optionals."""
import pytest
from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.scopes.regions import _helper_actuals, _upward_reads


def analysis(tmp_path, call):
    path = tmp_path / "optional.f90"
    path.write_text("""module optional_liveness
integer::observed
contains
subroutine step(i)
integer::i
""" + call + """
end subroutine
subroutine continuation(value)
integer,optional,intent(in)::value
if(present(value)) observed=value
end subroutine
end module
""")
    source = SourceEffects([path])
    routine = source.routines["optional_liveness::step"]
    statement, = [item for item in walk(routine.execution) if type(item).__name__ == "Call_Stmt"]
    return source, routine, statement


def test_absent_present_controlled_optional_has_no_caller_value(tmp_path):
    source, routine, statement = analysis(tmp_path, "call continuation()")
    helper, actuals = _helper_actuals(source, routine, statement, allow_omitted_optional=True)
    assert helper.qualified == "optional_liveness::continuation"
    assert actuals == (None,)
    assert "argument::i" not in _upward_reads(source, routine, (statement,))
    with pytest.raises(CompilationError, match="requires all original arguments"):
        _helper_actuals(source, routine, statement)


def test_supplied_optional_still_observes_caller_value(tmp_path):
    source, routine, statement = analysis(tmp_path, "call continuation(i)")
    assert "argument::i" in _upward_reads(source, routine, (statement,))


def test_omitted_required_native_argument_still_rejects(tmp_path):
    source, routine, statement = analysis(tmp_path, "call continuation()")
    binding = source.routines["optional_liveness::continuation"].scope.bindings["value"]
    object.__setattr__(binding, "attributes", binding.attributes - {"optional"})
    with pytest.raises(CompilationError, match="requires all original arguments"):
        _helper_actuals(source, routine, statement, allow_omitted_optional=True)


def test_absent_optional_does_not_erase_host_associated_observation(tmp_path):
    path = tmp_path / "host.f90"
    path.write_text("""module host_liveness
integer::observed
contains
subroutine step(i)
integer::i
call continuation()
contains
subroutine continuation(value)
integer,optional,intent(in)::value
observed=i
if(present(value))observed=value
end subroutine
end subroutine
end module
""")
    source = SourceEffects([path])
    routine = source.routines["host_liveness::step"]
    statement, = [item for item in walk(routine.execution) if type(item).__name__ == "Call_Stmt"]
    assert routine.scope.bindings["i"].root in _upward_reads(source, routine, (statement,))
