"""Exact complete private line patterns, with no implied kernel support."""

from copy import copy
from dataclasses import replace

import pytest
from fparser.two.utils import walk

from compiler.frontend.line_snapshot import prove_line_snapshot
from compiler.frontend.source_effects import SourceEffects, _children, _kind
from compiler.ir import CompilationError
from compiler.scopes.segments import directive, grouped_nodes


def source(axis=1, kind=8):
    transverse, line, neighbor, reference, point, shifted = {
        1: (("k", "j"), "i", "ii", "a(:,j,k)", "a(i,j,k)", "mask(ii,j,k)"),
        2: (("k", "i"), "j", "jj", "a(i,:,k)", "a(i,j,k)", "mask(i,jj,k)"),
        3: (("j", "i"), "k", "kk", "a(i,j,:)", "a(i,j,k)", "mask(i,j,kk)"),
    }[axis]
    x, y = transverse
    return f"""module lines
contains
subroutine advance(a,mask,nx,ny,nz)
integer,intent(in)::nx,ny,nz
real({kind}),intent(inout)::a(-2:,-2:,-2:)
integer,intent(in)::mask(-2:,-2:,-2:)
real({kind})::tmp(-2:ubound(a,{axis})),s,q
real({kind}),parameter::w(-3:3)=[-1._{kind}/32,0._{kind},9._{kind}/32,1._{kind}/2,9._{kind}/32,0._{kind},-1._{kind}/32]
integer::i,j,k,ii,jj,kk
!$omp parallel do private(i,j,k,ii,jj,kk,tmp,s,q) shared(a,mask,nx,ny,nz) schedule(runtime)
do {x}=1,n{dict(i="x", j="y", k="z")[x]}
 do {y}=1,n{dict(i="x", j="y", k="z")[y]}
  tmp={reference}
  do {line}=1,n{dict(i="x", j="y", k="z")[line]}
   if(mask(i,j,k)==0)then
    s=0
    q=0
    do {neighbor}={line}-3,{line}+3
     if({shifted}<=0)then
      s=s+w({neighbor}-{line})*tmp({neighbor})
      q=q+w({neighbor}-{line})
     endif
    enddo
    if(abs(q)>0.74){point}=s/q
   endif
  enddo
 enddo
enddo
!$omp end parallel do
end subroutine
end module
"""


def selections(routine):
    result = []

    def visit(nodes):
        for node in grouped_nodes(nodes):
            if isinstance(node, tuple):
                if any((directive(x) or "").startswith("parallel do") for x in node):
                    result.append(node)
            elif _kind(node) == "If_Construct":
                visit(_children(node))

    visit(_children(routine.execution))
    return result


def fixture(tmp_path, text=None):
    path = tmp_path / "source.f90"
    path.write_text(source() if text is None else text)
    analysis = SourceEffects([path])
    routine = analysis.routines["lines::advance"]
    (nodes,) = selections(routine)
    return path, analysis, routine, nodes


@pytest.mark.parametrize("axis", [1, 2, 3])
@pytest.mark.parametrize("kind", [4, 8])
def test_three_original_line_axes_preserve_source_private_storage_and_order(tmp_path, axis, kind):
    path, a, r, nodes = fixture(tmp_path, source(axis, kind))
    before = path.read_bytes()
    p = prove_line_snapshot(a, r, nodes)
    record = p.public()
    assert p.validate() is p
    assert prove_line_snapshot(a, r, nodes) is p
    assert record["line_axis"] == axis
    assert len(record["transverse_domain"]) == 2
    assert record["snapshot"]["scratch"]["rank"] == 1
    assert record["snapshot"]["scratch"]["kind"] == kind
    assert record["scratch_reads"][0]["interval"]["lower"] == {"constant": -2, "terms": {}}
    assert record["scratch_reads"][0]["interval"]["upper"] == {
        "constant": 3,
        "terms": {["nx", "ny", "nz"][axis - 1]: 1},
    }
    assert [(b.name, w) for b, w in p.fields] == [("a", True), ("mask", False)]
    assert len(p.transverse_loops) == 2
    assert len(p.constant_accesses) == 2
    assert all(len(loops) == 4 for _, _, loops in p.constant_accesses)
    for access in record["constant_array_accesses"]:
        assert access["affine_index_interval"] == {
            "lower": {"constant": -3, "terms": {}},
            "upper": {"constant": 3, "terms": {}},
        }
        assert access["integer_range_and_lookup_execution_authorized"] is False
    assert p._producer is next(
        x for x in walk(r.execution) if _kind(x) == "Assignment_Stmt" and str(x.items[0]).lower() == "tmp"
    )
    assert all(
        record[k] is False
        for k in [
            "definitions_authorized",
            "placement_authorized",
            "execution_authorized",
            "arithmetic_reassociation_authorized",
            "cooperative_mapping_authorized",
        ]
    )
    assert "axes" not in record
    assert "scratch_axis_order" not in record
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "change",
    [
        ("tmp=a(:,j,k)", "if(j>1)tmp=a(:,j,k)"),
        ("tmp=a(:,j,k)", "tmp=a(:,j,k)\n tmp=a(:,j,k)"),
        ("tmp=a(:,j,k)", "tmp(1)=a(1,j,k)"),
        ("tmp=a(:,j,k)", "tmp=a(:,j+1,k)"),
        ("tmp=a(:,j,k)", "tmp=a(1:nx,j,k)"),
        ("tmp=a(:,j,k)", "call opaque(tmp)"),
        ("tmp=a(:,j,k)", "tmp=mask(:,j,k)"),
        ("tmp=a(:,j,k)", "tmp=a(:,j,k)\n call opaque(a)"),
        ("a(i,j,k)=s/q", "a(i,j+1,k)=s/q"),
        ("mask(ii,j,k)", "mask(ii,j+1,k)"),
        ("s=s+w(ii-i)*tmp(ii)", "tmp(ii)=s+w(ii-i)*tmp(ii)"),
        ("s=0\n    q=0", "q=0"),
        ("s=0", "if(i>1)s=0"),
        ("s=0", "if(i>1)then\n     s=0\n    else\n    endif"),
        ("q=q+w(ii-i)", "s=s+unknown(q)"),
        ("q=q+w(ii-i)", "print *,q"),
        ("do ii=i-3,i+3", "do i=i-3,i+3"),
        ("do j=1,ny", "do j=1,k"),
        ("do j=1,ny", "do j=1,ny,2"),
        ("private(i,j,k,ii,jj,kk,tmp,s,q)", "private(i,j,k,ii,jj,kk,s,q)"),
        ("private(i,j,k,ii,jj,kk,tmp,s,q)", "private(i,j,k,ii,jj,kk,tmp,q)"),
        ("real(8)::tmp", "real(8),save::tmp"),
        ("real(8)::tmp", "real(8),target::tmp"),
        ("real(8),intent(inout)::a", "real(8),target,intent(inout)::a"),
        ("!$omp end parallel do", "!$omp end parallel do nowait"),
        ("schedule(runtime)", "schedule(runtime) ordered"),
    ],
)
def test_unsafe_or_incomplete_original_patterns_reject_without_edits(tmp_path, change):
    text = source()
    assert change[0] in text
    path = tmp_path / "source.f90"
    changed = text.replace(*change)
    def derive_changed():
        _, a, r, nodes = fixture(tmp_path, changed)
        return prove_line_snapshot(a, r, nodes)

    with pytest.raises(CompilationError):
        derive_changed()
    assert path.read_text() == changed


def test_exact_source_and_registered_identity_cannot_be_forged(tmp_path):
    path, a, r, nodes = fixture(tmp_path)
    p = prove_line_snapshot(a, r, nodes)
    for forged in [copy(p), replace(p), replace(p, source_binding=p.scratch_binding)]:
        with pytest.raises(CompilationError):
            forged.validate()
    other = SourceEffects([path])
    (foreign,) = selections(other.routines[r.qualified])
    for selected in [nodes[1:], nodes[:-1], foreign]:
        with pytest.raises(CompilationError):
            prove_line_snapshot(a, r, selected)
    record = p.public()
    record["line_axis"] = 9
    assert p.public()["line_axis"] == 1
    path.write_text(path.read_text() + "!changed\n")
    with pytest.raises(CompilationError):
        p.validate()


def test_forged_identity_registry_and_discarded_completion_never_reissue(tmp_path):
    _, a, r, nodes = fixture(tmp_path)
    p = prove_line_snapshot(a, r, nodes)
    forged = replace(p, identity="0" * 64)
    a._line_snapshot_proofs[forged.identity] = forged
    with pytest.raises(CompilationError):
        forged.validate()
    a._joined_completions.clear()
    with pytest.raises(CompilationError):
        p.validate()
    assert not a._joined_completions


def test_substitute_constant_occurrence_or_loop_ancestry_rejects(tmp_path):
    _, a, r, nodes = fixture(tmp_path)
    p = prove_line_snapshot(a, r, nodes)
    node, binding, loops = p.constant_accesses[0]
    for accesses in [((node, binding, loops[:-1]),), ((node, p.source_binding, loops),), p.constant_accesses[::-1]]:
        forged = replace(p, constant_accesses=accesses)
        a._line_snapshot_proofs[p.identity] = forged
        with pytest.raises(CompilationError):
            forged.validate()
    a._line_snapshot_proofs[p.identity] = p
    assert p.validate() is p
