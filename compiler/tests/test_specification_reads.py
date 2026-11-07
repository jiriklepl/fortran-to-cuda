"""Entry bounds are independent of whole-body effects and execution authority."""
from copy import copy
from dataclasses import replace

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.specification_reads import prove_specification_reads
from compiler.ir import CompilationError

SOURCE = """module entry_reads
implicit none
integer,protected::tile(4)=[2,4,6,8]
contains
subroutine child(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer,parameter::narr=2
real(8)::scratch(-n:tile(narr)+2),other(tile(n))
call unknown_body(a,scratch,other)
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE, *, sealed=False):
    path = tmp_path / "entry.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine = analysis.routines["entry_reads::child"]
    return path, analysis, routine


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("protected", [False, True])
def test_original_bounds_reuse_storage_effects_without_body_or_lifetime_authority(tmp_path, sealed, protected):
    source = SOURCE if protected else SOURCE.replace(",protected", "")
    path, analysis, routine = fixture(tmp_path, source, sealed=sealed)
    before = path.read_bytes()
    proof = prove_specification_reads(analysis, routine)
    assert proof.validate(analysis, routine) is proof
    assert prove_specification_reads(analysis, routine) is proof
    record = proof.public()
    assert record["array_resources"] == ["entry_reads::tile"]
    assert record["phase"] == "before original procedure entry and specification evaluation"
    assert {(item.storage.resource, item.kind) for item in proof.accesses} == {
        ("argument::n", "read"), ("entry_reads::tile", "read")}
    assert [item.source for item in proof.accesses if item.storage.rank] == ["tile(narr)", "tile(n)"]
    assert all(record[key] is False for key in ("definition_authority", "alias_authority", "lifetime_authority",
        "body_completion_authority", "payload_access_authorized", "execution_authorized"))
    record["execution_authorized"] = True
    assert not proof.public()["execution_authorized"]
    assert path.read_bytes() == before
    assert analysis._allocation_authorizations == {}


@pytest.mark.parametrize("bound", ["source_function(n)", "size(a)", "obj%length", "tile(:)",
                                   "tile(tile)", "unknown", "tile(n,n)", "a(n)"])
def test_calls_components_unknown_or_nonscalar_bounds_decline(tmp_path, bound):
    source = SOURCE.replace("tile(narr)+2", bound)
    if bound == "obj%length":
        source = source.replace("contains", "type shape_type\ninteger::length\nend type\n"
                                "type(shape_type)::obj\ncontains", 1)
    _, analysis, routine = fixture(tmp_path, source)
    with pytest.raises(CompilationError, match="specification reads"):
        prove_specification_reads(analysis, routine)


@pytest.mark.parametrize("attribute", ["pointer", "target", "volatile", "asynchronous"])
def test_payload_binding_does_not_imply_independent_entry_storage(tmp_path, attribute):
    source = SOURCE.replace("integer,protected::tile(4)=[2,4,6,8]", f"integer,{attribute}::tile(4)")
    _, analysis, routine = fixture(tmp_path, source)
    with pytest.raises(CompilationError, match="specification reads"):
        prove_specification_reads(analysis, routine)


@pytest.mark.parametrize("declaration", ["common /alias/ n", "namelist /alias/ n", "equivalence(n,narr)"])
def test_unproved_specification_aliases_decline(tmp_path, declaration):
    _, analysis, routine = fixture(tmp_path, SOURCE.replace("call unknown_body", declaration + "\ncall unknown_body"))
    with pytest.raises(CompilationError, match="specification reads"):
        prove_specification_reads(analysis, routine)


@pytest.mark.parametrize("mutation", ["copy", "registry", "binding", "namespace", "source", "occurrence", "limits"])
def test_retained_authority_rejects_original_replacement(tmp_path, mutation):
    path, analysis, routine = fixture(tmp_path)
    proof = prove_specification_reads(analysis, routine)
    if mutation == "copy":
        proof = copy(proof)
    elif mutation == "registry":
        analysis._specification_reads.clear()
    elif mutation == "binding":
        routine.scope.bindings["n"] = replace(routine.scope.bindings["n"])
    elif mutation == "namespace":
        routine.scope.imports["tile"] = ("elsewhere", "tile")
    elif mutation == "source":
        path.write_text(path.read_text() + "! changed input\n")
    elif mutation == "limits":
        analysis.operation_limit += 1
    else:
        parent = next(node for node in walk(routine.scope.node) if type(node).__name__ == "Part_Ref"
                      and str(node.items[0]).lower() == "tile")
        parent.items = (F.Name("tile"), parent.items[1])
    with pytest.raises(CompilationError):
        proof.validate(analysis, routine)


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("mutation", ["source", "binding", "occurrence", "registry"])
def test_publication_callbacks_cannot_change_entry_authority(tmp_path, monkeypatch, sealed, existing, mutation):
    from compiler.frontend import specification_reads as service

    path, analysis, routine = fixture(tmp_path, sealed=sealed)
    if existing:
        prove_specification_reads(analysis, routine)
    encode = service._encoded
    changed = False

    def mutate(record):
        nonlocal changed
        encoded = encode(record)
        if not changed:
            changed = True
            if mutation == "source":
                path.write_text(path.read_text() + "! changed while publishing\n")
            elif mutation == "binding":
                object.__setattr__(routine.scope, "bindings", {
                    **routine.scope.bindings, "n": replace(routine.scope.bindings["n"])})
            elif mutation == "registry":
                analysis._specification_reads = {}
            else:
                # Equivalent spelling must not replace the consumed original.
                parent = next(node for node in walk(routine.scope.node) if type(node).__name__ == "Part_Ref"
                              and str(node.items[0]).lower() == "tile")
                parent.items = (F.Name("tile"), parent.items[1])
        return encoded

    monkeypatch.setattr(service, "_encoded", mutate)
    with pytest.raises(CompilationError):
        prove_specification_reads(analysis, routine)
