"""A once-armed restriction cannot create storage or payload authority."""
from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.allocation_epoch_eligibility import prove_allocation_gate
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

SOURCE = """module gate_fixture
implicit none
contains
subroutine advance(n,mode)
integer,intent(in)::n,mode
integer,save::called=0
real(8),save,allocatable::a(:),b(:)
logical,save::eligible=.false.
if(called==0)then
allocate(a(n))
allocate(b(n))
a=0
eligible=.true.
endif
called=called+1
if(eligible)then
 a(1)=a(1)+1.0_8
endif
if(mode==1)then
 eligible=.false.
 call unavailable(a,b)
endif
if(eligible) a(1)=a(1)+2.0_8
end subroutine
subroutine unavailable(a,b)
real(8),intent(inout)::a(:),b(:)
b=a
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE, *, sealed=False):
    path = tmp_path / "gate.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine = analysis.routines["gate_fixture::advance"]
    barriers = tuple(node for node in walk(routine.execution) if type(node).__name__ == "Call_Stmt"
                     and str(node.items[0]).lower() == "unavailable")
    return path, analysis, routine, routine.scope.bindings["eligible"], tuple(
        routine.scope.bindings[name] for name in ("a", "b")), barriers


def issue(parts):
    return prove_allocation_gate(*parts[1:])


@pytest.mark.parametrize("sealed", [False, True])
def test_exact_once_armed_gate_remains_false_after_excluded_edge_and_context_recreation(tmp_path, sealed):
    parts = fixture(tmp_path, sealed=sealed)
    proof = issue(parts)
    assert proof.validate() is proof
    assert issue(parts) is proof
    public = proof.public()
    assert public["flag"]["resource"] == "gate_fixture::advance::eligible"
    assert [row["procedure"] for row in public["barriers"]] == ["gate_fixture::unavailable"]
    for field in ("lifetime_authorized", "allocation_presence_authority", "payload_definition_authority",
                  "payload_access_authorized", "execution_authorized", "cross_frontier_retention_authorized"):
        assert public[field] is False
    assert "every new context" in public["rule"]
    public["payload_definition_authority"] = True
    assert proof.public()["payload_definition_authority"] is False


@pytest.mark.parametrize(("before", "after"), [
    ("logical,save::eligible=.false.", "logical,save::eligible=.true."),
    ("logical,save::eligible=.false.", "logical,save,target::eligible=.false."),
    ("logical,save::eligible=.false.", "logical(8),save::eligible=.false."),
    ("logical,save::eligible=.false.", "logical::eligible"),
    ("integer,save::called=0", "integer,save::called=1"),
    ("called=called+1", "called=called-1"),
    ("called=called+1", "called=called+1\ncalled=0"),
    ("if(eligible)then", "call observe(eligible)\nif(eligible)then"),
    ("eligible=.false.\n call unavailable", "eligible=.false.\n a=1\n call unavailable"),
    ("eligible=.false.\n call unavailable", "eligible=.true.\n call unavailable"),
    ("a=0\neligible=.true.", "call unavailable(a,b)\neligible=.true."),
    ("allocate(a(n))", "allocate(a(n),stat=mode)"),
    ("real(8),save,allocatable::a(:),b(:)", "real(8),save,target,allocatable::a(:),b(:)"),
    ("logical,save::eligible=.false.", "logical,save::eligible=.false.\nnamelist /state/ eligible"),
    ("if(mode==1)then", "eligible=.true.\nif(mode==1)then"),
    ("if(mode==1)then", "if(f(called)==1)then"),
    ("allocate(b(n))", "allocate(b(f(a)))"),
    ("a=0\neligible=.true.", "a=f(a)\neligible=.true."),
    ("eligible=.true.\nendif", "eligible=.true.\ncall unavailable(a,b)\nendif"),
])
def test_rearming_escape_alias_observer_and_incomplete_barrier_shapes_reject(tmp_path, before, after):
    with pytest.raises(CompilationError):
        issue(fixture(tmp_path, SOURCE.replace(before, after)))


def clock_source(call="if(mode>0)call system_clock(count_rate=rate)"):
    return SOURCE.replace("integer,save::called=0", "integer,save::called=0\ninteger,save::rate").replace(
        "eligible=.true.\nendif", call+"\neligible=.true.\nendif")


def test_prearm_clock_has_exact_independent_intrinsic_and_descriptor_authority(tmp_path):
    proof = issue(fixture(tmp_path, clock_source(), sealed=True))
    record = proof.public()
    assert record["version"] == 2
    node, clock, descriptor = proof._prearm[0]
    assert record["prearm_metadata"][0]["original_node"]["source"] == str(node)
    assert "IF (MODE > 0)" in str(node).upper()
    assert record["prearm_metadata"][0]["clock_identity"] == clock.identity
    assert record["prearm_metadata"][0]["descriptor_identity"] == descriptor.identity
    assert proof.validate() is proof
    assert record["payload_definition_authority"] is False


@pytest.mark.parametrize("call", ["call unavailable(a,b)", "deallocate(a)", "call unknown_export(a)",
    "if(any(a>0))call system_clock(count_rate=rate)", "call system_clock(count_rate=a)"])
def test_only_original_scalar_clock_metadata_can_intervene_before_final_arming(tmp_path, call):
    with pytest.raises(CompilationError):
        issue(fixture(tmp_path, clock_source(call)))


@pytest.mark.parametrize("kind", ["content", "foreign"])
def test_gate_registry_cannot_retag_content_or_lend_original_analysis_authority(tmp_path, kind):
    parts = fixture(tmp_path)
    proof = issue(parts)
    if kind == "content":
        object.__setattr__(proof, "identity", "0"*64)
        parts[1]._allocation_gates[proof.identity] = proof
        with pytest.raises(CompilationError):
            proof.validate()
    else:
        with pytest.raises(CompilationError):
            prove_allocation_gate(parts[1].private_projection(), *parts[2:])


@pytest.mark.parametrize("mutation", ["proof", "registry", "namespace", "source", "binding", "barrier", "guard_parent"])
def test_registered_source_token_cannot_retag_changed_authority(tmp_path, mutation):
    parts = fixture(tmp_path)
    path, analysis, routine, flag, resources, barriers = parts
    proof = issue(parts)
    if mutation == "proof":
        proof = copy(proof)
    elif mutation == "registry":
        analysis._allocation_gates.clear()
    elif mutation == "namespace":
        routine.scope.imports["other"] = ("hidden", "x")
    elif mutation == "source":
        path.write_text(path.read_text() + "! changed\n")
    elif mutation == "binding":
        routine.scope.bindings[flag.name] = copy(flag)
    elif mutation == "barrier":
        original = barriers[0]
        parent = next(node for node in walk(routine.execution) if hasattr(node, "content")
                      and any(child is original for child in node.content))
        parent.content[parent.content.index(original)] = copy(original)
    else:
        original = next(node for node in routine.execution.content if type(node).__name__ == "If_Construct")
        routine.execution.content[routine.execution.content.index(original)] = copy(original)
    with pytest.raises(CompilationError):
        proof.validate()


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("mutation", ["source", "input_metadata", "registry"])
def test_publication_fence_rejects_live_input_or_registry_mutation(tmp_path, monkeypatch, sealed, mutation):
    from compiler.frontend import allocation_epoch_eligibility as service

    parts = fixture(tmp_path, sealed=sealed)
    path, analysis, *_rest = parts
    encode = service._encoded
    fired = False

    def change(record):
        nonlocal fired
        encoded = encode(record)
        if not fired:
            fired = True
            if mutation == "source":
                path.write_text(path.read_text() + "! publication mutation\n")
            elif mutation == "registry":
                analysis._allocation_gates = {}
            else:
                inputs = copy(analysis.inputs)
                inputs.configuration = {"different": True}
                analysis.inputs = inputs
        return encoded

    monkeypatch.setattr(service, "_encoded", change)
    with pytest.raises(CompilationError):
        issue(parts)
    assert fired


def test_serialized_metadata_cannot_populate_the_source_registry(tmp_path):
    parts = fixture(tmp_path)
    analysis = parts[1]
    record = issue(parts).public()
    analysis._allocation_gates = {record["identity"]: record}
    with pytest.raises(CompilationError, match="public metadata"):
        issue(parts)
