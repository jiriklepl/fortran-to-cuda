"""Ordinary lexical owners consume source-issued numerical pairs, without a plan."""

from copy import copy, deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.ir import CompilationError
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.source import ScopeBuilder
from compiler.tests.support.numerical_regions import CHILD, FACT, MATH_SOURCE, OWNER_SOURCE, ROOTS

ORDINARY_CHILD_SOURCE = """module quartz_numerics
use,intrinsic::ieee_exceptions,only:ieee_set_halting_mode,ieee_invalid
implicit none
integer::bridge_visits=0,middle_visits=0
real(8)::coefficient(-1:0)=[2.0_8,3.0_8]
contains
subroutine bridge(a,b,c,n,tile)
real(8),intent(in)::a(-4:)
real(8),intent(inout)::b(-4:),c(-4:)
integer,intent(in)::n,tile
integer::i
bridge_visits=bridge_visits+1
i=0
!$omp parallel
call ieee_set_halting_mode(ieee_invalid,.false.)
!$omp end parallel
!$omp parallel private(i)
!$omp do schedule(static)
do i=-3,n
b(i)=b(i)+a(i)*coefficient(-1)
enddo
!$omp end do
!$omp do schedule(static)
do i=-2,n
b(i)=b(i)+a(i)*coefficient(0)
enddo
!$omp end do nowait
!$omp end parallel
if(n>=-3) then
call touch_face(b)
endif
!$omp parallel private(i)
!$omp do schedule(static)
do i=-3,n
c(i)=2.0_8*b(i)+real(i+1,8)
enddo
!$omp end do
!$omp end parallel
end subroutine
subroutine touch_face(b)
real(8),intent(inout)::b(-4:)
middle_visits=middle_visits+1
b(-3)=b(-3)+0.5_8
end subroutine
end module
"""


def ordinary_numerical_fixture(directory, *, precision=8, device_budget=None,
                               child_source=ORDINARY_CHILD_SOURCE, captures=None):
    """Keep the public driver ABI, with an ordinary owner at original bridge."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = tuple(directory / name for name in ("quartz_math.f90", "quartz_numerics.f90", "quartz_owner.f90"))
    for path, source in zip(paths, (MATH_SOURCE, child_source, OWNER_SOURCE), strict=True):
        path.write_text(source.replace("real(8)", f"real({precision})")
                        .replace("_8", f"_{precision}").replace(",8)", f",{precision})"))
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths},
             "captures": {ROOTS[0]: deepcopy(FACT), ROOTS[1]: deepcopy(FACT),
                          ROOTS[2]: {**deepcopy(FACT), "initialized": "none"},
                          "quartz_numerics::coefficient": deepcopy(FACT)}}
    for root, values in (captures or {}).items():
        facts["captures"].setdefault(root, deepcopy(FACT)).update(values)
    if device_budget is not None:
        facts["device_budget_bytes"] = device_budget
    builder = ScopeBuilder(paths, CHILD, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="reached"))
    return paths, builder


def emit_ordinary_numerical_fixture(directory, **options):
    paths, builder = ordinary_numerical_fixture(directory, **options)
    outputs, report = builder.run()
    return paths, outputs, report, builder


def numerical_groups(builder):
    return tuple(group.original_nodes for group in grouped_original_sequence(builder.entry.execution.content, {})
                 if isinstance(group, NativeGroupCandidate)
                 and any(type(node).__name__ == "Block_Nonlabel_Do_Construct" for node in group.original_nodes))
