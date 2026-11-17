"""Whole separate numerical teams retain their source worksharing sequence."""

from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.numerical_effects import prove_numerical_effects
from compiler.frontend.numerical_preparation import NumericalPreparationProvider
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.support.numerical_regions import EmittedNumericalText
from compiler.tests.test_ordinary_numerical_regions import emit_ordinary_numerical_fixture

ENTRY = "bounded_team::step"
BODY = """!$omp parallel private(i)
!$omp do schedule(static)
do i=-3,n
b(i)=a(i)+real(i,8)
enddo
!$omp end do nowait
!$omp do schedule(static)
do i=-3,n
c(i)=b(i)+a(i)
enddo
!$omp end do nowait
!$omp end parallel
"""
SOURCE = """module bounded_team
implicit none
contains
subroutine step(a,b,c,n)
real(8),intent(in)::a(-4:)
real(8),intent(inout)::b(-4:),c(-4:)
integer,intent(in)::n
integer::i
real(8)::s
""" + BODY + "end subroutine\nend module\n"


def group_fixture(directory, source=SOURCE, *, opt_level=1):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "bounded_team.f90"
    path.write_text(source)
    fact = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: dict(fact) for name in ("a", "b", "c")}}
    builder = ScopeBuilder([path], ENTRY, facts=facts, options=CompilerOptions(opt_level=opt_level),
                           config=OffloadConfig(policy="sections", scope_execution="reached"))
    group, = (item for item in grouped_original_sequence(builder.entry.execution.content, {})
              if isinstance(item, NativeGroupCandidate))
    return builder, group.original_nodes


def physical_effects(builder, selection):
    provider = NumericalPreparationProvider(builder.analysis, options=builder.options, config=builder.config,
                                            environment_root=builder._numerical_environment_root)
    prepared = provider.prepare(builder.analysis, ENTRY, selection)
    return prepared, prove_numerical_effects(builder.analysis, ENTRY, selection, prepared)


@pytest.fixture(scope="module", params=[4, 8])
def separate_emission(tmp_path_factory, request):
    paths, outputs, report, builder = emit_ordinary_numerical_fixture(
        tmp_path_factory.mktemp("separate_team"), precision=request.param)
    assert builder.validate_reached_window() is None
    return EmittedNumericalText.capture(paths, outputs, report)
