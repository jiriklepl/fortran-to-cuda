"""Immutable arithmetic actuals retain Fortran evaluation and mutable boundaries."""
import pytest
from fparser.two import Fortran2003 as F

from compiler.frontend.constant_actual import constant_scalar_actual
from compiler.frontend.source_effects import SourceEffects
from compiler.tests.test_graph_source_owner import emit, reusable_helper_source


@pytest.mark.parametrize(("expression", "expected"), [
    ("-1.0_8/divisor", True), ("(gain+1.0_8)/(divisor*2)", True), ("gain**2", True),
    ("mutable/divisor", False), ("1.0_8/size(field)", False), ("sqrt(gain)", False),
    ("coefficients(index)", False),
])
def test_only_original_scalar_parameters_and_arithmetic_are_constant(tmp_path, expression, expected):
    path = tmp_path / "original.f90"
    path.write_text("""module constant_source
implicit none
integer,parameter::divisor=256
real(8),parameter::gain=2.0_8,coefficients(2)=[1.0_8,2.0_8]
real(8)::mutable
contains
subroutine caller(field,index)
real(8)::field(:)
integer::index
field=0
end subroutine
end module
""")
    analysis = SourceEffects([path])
    scope = analysis.routines["constant_source::caller"].scope
    assert constant_scalar_actual(analysis, scope, F.Expr(expression)) is expected


def test_constant_fraction_is_admitted_without_moving_its_original_call(tmp_path):
    source = reusable_helper_source(False).replace("subroutine leaf(field,n)", "subroutine leaf(field,n,factor)")
    source = source.replace("integer::i", "real(8),intent(in)::factor\ninteger::i")
    source = source.replace("field(i)=field(i)+1", "field(i)=field(i)*factor")
    source = source.replace("call leaf(a,n)", "call leaf(a,n,-1.0_8/256)")
    source = source.replace("call leaf(b,n)", "call leaf(b,n,-1.0_8/256)")
    path, outputs, report = emit(tmp_path, source)
    owner, = report["scopes"]
    assert not owner["boundaries"]
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert "-1.0_8 / 256" in text or "-1.0_8/256" in text
    assert len(owner["module_coordinators"][0]["calls"]) == 2
