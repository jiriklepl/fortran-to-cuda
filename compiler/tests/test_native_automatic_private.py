"""Native automatic PRIVATE arrays retain entry-time allocation descriptors."""

import os
import re
import shutil
from pathlib import Path

import pytest

from compiler.frontend.source_effects import SourceEffects, _children, _kind
from compiler.ir import CompilationError
from compiler.numerical_contract import cuda_compile_options
from compiler.scopes.native_atomic import summarize
from compiler.scopes.regions import extract_region
from compiler.scopes.segments import grouped_nodes
from compiler.tests.test_source_scopes import FACT, PROGRAM, generate, run


def source_text():
    source = PROGRAM.replace("implicit none\ncontains", "implicit none\ninteger::extent=0,visits=0\ncontains")
    source = source.replace("integer,intent(in)::n\ncall producer(a,b,n)",
                            "integer,intent(in)::n\ninteger::i\nreal(8)::scratch(-2:extent)\n"
                            "extent=extent+7\nvisits=visits+1\ncall producer(a,b,n)")
    return source.replace("call transform(b)\ncall consumer", """!$omp parallel do private(i,scratch) shared(a,b,n)
do i=1,n
scratch=a(i)
b(i)=b(i)+sum(scratch)+real(lbound(scratch,1)+ubound(scratch,1),8)
enddo
!$omp end parallel do
call consumer""")


def generated(tmp_path, *, scope_execution="reached"):
    return generate(tmp_path, source_text(), facts={"schema_version": 1, "participation": "serial",
        "captures": {"argument::" + name: FACT for name in ("a", "b", "out")}},
        scope_execution=scope_execution)


@pytest.mark.parametrize("declaration", ["real(8),save::scratch(3)", "real(8),allocatable::scratch(:)",
    "real(8),pointer::scratch(:)", "real(8),target::scratch(3)", "real(8)::scratch(3)=1"])
def test_private_storage_with_other_lifetime_or_association_is_rejected(tmp_path, declaration):
    source = source_text().replace("real(8)::scratch(-2:extent)", declaration)
    path = tmp_path / "source.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    routine = analysis.routines["original::step"]
    group, = [node for node in grouped_nodes(_children(routine.execution)) if isinstance(node, tuple)]
    with pytest.raises(CompilationError, match="PRIVATE"):
        analysis.joined_completion(routine.qualified, group)


def test_native_authority_does_not_allow_dynamic_private_numerical_lowering(tmp_path):
    path = tmp_path / "source.f90"
    path.write_text(source_text())
    analysis = SourceEffects([path])
    routine = analysis.routines["original::step"]
    group, = [node for node in grouped_nodes(_children(routine.execution)) if isinstance(node, tuple)]
    assert analysis.joined_completion(routine.qualified, group).public()["original_private_arrays"]
    with pytest.raises(CompilationError, match="constant explicit bounds"):
        extract_region(analysis, routine, group)


@pytest.mark.parametrize("name", ["CD4divU", "CD4divV", "CD4divW", "Filter3ord"])
def test_primary_source_bodies_retain_native_automatic_private_arrays(tmp_path, name):
    # Optional real-source probe; no application procedure names are recognized
    # by the compiler. The generic fixtures establish the reusable contract.
    filename = "filters.f90" if name == "Filter3ord" else "advection.f90"
    upstream = Path(os.environ.get("FORT_TEST_ELMM_SOURCE", str(
        Path(__file__).resolve().parents[2] / "elmm-pipeline/.cache/repos/elmm/src"))) / filename
    if not upstream.exists():
        pytest.skip("upstream source checkout unavailable")
    source = upstream.read_text()
    body = re.search(r"(?im)^  subroutine " + name + r"\(.*?^  end subroutine " + name + r"\s*$", source, re.S).group()
    path = tmp_path / "original.f90"
    path.write_text("module source_probe\nimplicit none\ninteger,parameter::knd=8\n"
        "integer::tilenx(8),tileny(8),tilenz(8),Unx,Uny,Unz,Vnx,Vny,Vnz,Wnx,Wny,Wnz\n"
        "real(8)::dxmin,dymin,dzmin\ninteger,allocatable::Utype(:,:,:),Vtype(:,:,:),Wtype(:,:,:)\n"
        "contains\n" + body + "\nend module\n")
    analysis = SourceEffects([path])
    analysis.authorize_stable_module_allocatables({"source_probe::" + name for name in ("utype", "vtype", "wtype")})
    routine = analysis.routines["source_probe::" + name.lower()]
    groups = []

    def collect(nodes):
        for node in grouped_nodes(nodes):
            if isinstance(node, tuple):
                groups.append(node)
            elif _kind(node) == "If_Construct":
                collect(_children(node))

    collect(_children(routine.execution))
    assert len(groups) == (3 if name == "Filter3ord" else 1)
    for group in groups:
        proof = analysis.joined_completion(routine.qualified, group)
        assert len(proof.public()["original_private_arrays"]) == (1 if name == "Filter3ord" else 2)
        assert all(not item["numerical_authority"] for item in proof.public()["original_private_arrays"])
        effects = summarize(analysis, routine.qualified, group, proof)
        assert effects["complete"], effects["reasons"]
        assert max(effects["native_atomic"]["unit_operation_counts"]) <= analysis.operation_limit
        arrays = {item["resource"] for item in effects["operations"]
                  if item.get("rank") and item["resource"] not in proof.private_roots}
        expected = ({"argument::u", "argument::utype"} if name == "Filter3ord" else
                    {"argument::u", "argument::v", "argument::w", "argument::" + name[-1].lower() + "2",
                     "source_probe::" + name[-1].lower() + "type"})
        assert arrays == expected
