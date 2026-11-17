"""Unsupported original native calls disable only reached dormant paths."""

from copy import copy, deepcopy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.frontend.native_prefix import (
    prove_native_prefix,
    reject_unsupported_native_prefix_call,
)
from compiler.frontend.preowner_transfers import PreownerUnsupported
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

ENTRY = "observer_user::advance"
ROOTS = ("argument::a", "argument::b", "argument::c")
IO_BODY = "write(*,*) n"
TEAM_BODY = """!$omp parallel private(local)
local=n
!$omp end parallel"""
NATIVE = """module native_observers
implicit none
contains
subroutine notify(n)
integer,intent(in)::n
integer::local
{body}
end subroutine
subroutine quiet(n)
integer,intent(in)::n
integer::local
local=n
end subroutine
end module
"""
BRIDGE = """module observer_bridge
use native_observers,only:exported=>notify
implicit none
private
public::exported
end module
"""
SOURCE = """module observer_user
use observer_bridge,only:local_hook=>exported
implicit none
contains
subroutine advance(a,b,c,n,enabled)
real(8),intent(in)::a(-2:)
real(8),intent(inout)::b(-2:),c(-2:)
integer,intent(in)::n
logical,intent(in)::enabled
integer::i
if(enabled) then
call local_hook(n)
endif
do i=-2,n-3
b(i)=a(i)+real(i+4,8)
enddo
do i=-2,n-3
c(i)=2.0_8*b(i)
enddo
end subroutine
end module
"""


def call_fixture(directory, *, body=IO_BODY, source=SOURCE, native=None, builder=False, call_procedure=ENTRY):
    directory.mkdir(parents=True, exist_ok=True)
    paths = tuple(directory / name for name in ("observers.f90", "bridge.f90", "worker.f90"))
    for path, text in zip(paths, (native or NATIVE.format(body=body), BRIDGE, source), strict=True):
        path.write_text(text)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths},
             "captures": {root: deepcopy(FACT) for root in ROOTS}}
    owner = (ScopeBuilder(paths, ENTRY, facts=facts, options=CompilerOptions(),
                          config=OffloadConfig(policy="sections", scope_execution="reached")) if builder else None)
    analysis = owner.analysis if owner is not None else SourceEffects(paths)
    routine = analysis.routines[ENTRY]
    call_routine = analysis.routines[call_procedure]
    branch = next((node for node in call_routine.execution.content if type(node).__name__ == "If_Construct"), None)
    call = next(node for node in (branch.content if branch is not None else call_routine.execution.content)
                if type(node).__name__ == "Call_Stmt")
    selection = tuple(node for node in routine.execution.content
                      if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    assert len(selection) == 2
    return paths, analysis, facts, call, selection, owner


def test_supported_original_call_still_uses_the_existing_effect_proof(tmp_path):
    _paths, analysis, _facts, call, _selection, _owner = call_fixture(tmp_path, body="local=n")
    assert reject_unsupported_native_prefix_call(analysis, ENTRY, call, consume=lambda _count: None) is None
    assert not getattr(analysis, "_native_prefix_proofs", {})
    complete = prove_native_prefix(analysis, "native_observers::notify")
    assert complete.validate(analysis, "native_observers::notify") is complete
    assert complete.summary["complete"]


def component_fixture(directory, *, dtype="real(8)"):
    initial = ".true." if dtype == "logical" else "1.0"
    declarations = f"""module original_settings
implicit none
type tuning
{dtype}::gain={initial},alternate={initial}
real(8),allocatable::unrelated(:)
end type
type(tuning)::retained
end module
"""
    native = declarations + NATIVE.format(body=IO_BODY).replace(
        "integer,intent(in)::n", f"{dtype},intent(in)::n").replace("integer::local", f"{dtype}::local")
    source = SOURCE.replace("implicit none\ncontains", """use original_settings,only:settings=>retained
implicit none
contains""", 1).replace("call local_hook(n)", "call local_hook(settings%gain)")
    return call_fixture(directory, native=native, source=source)


def test_copied_original_call_has_no_classification_authority(tmp_path):
    _paths, analysis, _facts, call, _selection, _owner = call_fixture(tmp_path)
    with pytest.raises(CompilationError) as error:
        reject_unsupported_native_prefix_call(analysis, ENTRY, copy(call), consume=lambda _count: None)
    assert not isinstance(error.value, PreownerUnsupported)


def test_shared_scan_budget_failure_remains_hard(tmp_path):
    _paths, analysis, _facts, call, _selection, _owner = call_fixture(tmp_path)

    def exhausted(_count):
        raise CompilationError("sentinel exhausted aggregate coordinate/source scan budget")

    with pytest.raises(CompilationError, match="sentinel exhausted aggregate") as error:
        reject_unsupported_native_prefix_call(analysis, ENTRY, call, consume=exhausted)
    assert not isinstance(error.value, PreownerUnsupported)
    assert not getattr(analysis, "_native_prefix_proofs", {})


@pytest.mark.parametrize("limit", ["operation_limit", "procedure_limit", "depth_limit"])
def test_local_call_graph_budget_failures_remain_hard(tmp_path, limit):
    native = NATIVE.format(body="call intermediate(n)").replace("subroutine quiet(n)", """subroutine intermediate(n)
integer,intent(in)::n
call report_value(n)
end subroutine
subroutine report_value(n)
integer,intent(in)::n
write(*,*) n
end subroutine
subroutine quiet(n)""")
    _paths, analysis, _facts, call, _selection, _owner = call_fixture(tmp_path, native=native)
    setattr(analysis, limit, 1)
    with pytest.raises(CompilationError, match="budget|bounded|depth|limit") as error:
        reject_unsupported_native_prefix_call(analysis, ENTRY, call, consume=lambda _count: None)
    assert not isinstance(error.value, PreownerUnsupported)
    assert not getattr(analysis, "_native_prefix_proofs", {})
