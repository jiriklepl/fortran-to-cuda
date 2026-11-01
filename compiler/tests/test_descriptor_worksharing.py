"""Structure-only orphan DO receipts never supply native completion."""
from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend import descriptor_worksharing as service
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

ENTRY = "worksharing::run"
SOURCE = """module worksharing
contains
subroutine run(a,b,n)
integer,intent(in)::n
real(8),intent(in)::a(:,:,:)
real(8),intent(inout)::b(:,:,:)
integer::i,j,k
!$omp do collapse(3)
do k=1,n
do j=1,n
do i=1,n
b(i,j,k)=a(i,j,k)
enddo
enddo
enddo
!$omp end do nowait
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE, sealed=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "source.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    return path, analysis, analysis.routines[ENTRY]


def issue(parts):
    return service.original_worksharing_storage(parts[1], parts[2])


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("nowait", [False, True])
def test_original_worksharing_is_structure_only_without_a_parent_team(tmp_path, sealed, nowait):
    source = SOURCE if nowait else SOURCE.replace("end do nowait", "end do")
    parts = fixture(tmp_path, source, sealed)
    before = parts[0].read_bytes()
    proof = issue(parts)
    public = proof.public()
    import json
    (tmp_path / "public.json").write_text(json.dumps(public, indent=2)+"\n")
    assert public["version"] == 1
    assert len(public["regions"]) == 1
    assert public["regions"][0]["collapse"] == 3
    assert public["regions"][0]["retained_nowait"] is nowait
    assert public["requires_complete_original_callee_and_body_rescan"]
    assert all(public[name] is False for name in ("descriptor_preservation_authorized",
        "storage_lifetime_authority", "caller_participation_authority", "work_completion_authority",
        "execution_authorized", "payload_access_authorized", "entry_definition_authority",
        "alias_authority", "internal_cuts_authorized", "gpu_legality_established"))
    assert issue(parts) is proof
    assert proof.validate(parts[1], parts[2]) is proof
    assert not getattr(parts[1], "_descriptor_team_proofs", {})
    assert not parts[1]._joined_completions
    assert not parts[1]._numerical_completions
    assert parts[0].read_bytes() == before


@pytest.mark.parametrize("sealed", [False, True])
def test_six_original_nowait_loops_after_a_real_join_do_not_borrow_its_completion(tmp_path, sealed):
    loop = SOURCE[SOURCE.index("!$omp do"):SOURCE.index("end subroutine")]
    prefix = "!$omp parallel private(i,j,k)\n" + loop.replace("end do nowait", "end do") + "!$omp end parallel\n"
    source = SOURCE[:SOURCE.index("!$omp do")] + prefix + loop*6 + "end subroutine\nend module\n"
    parts = fixture(tmp_path, source, sealed)
    proof = issue(parts)
    assert len(proof.public()["regions"]) == 6
    assert all(row["retained_nowait"] for row in proof.public()["regions"])
    assert not parts[1]._joined_completions


@pytest.mark.parametrize("replacement", [
    "!$omp do schedule(static)", "!$omp do private(i)", "!$omp do firstprivate(i)",
    "!$omp do lastprivate(i)", "!$omp do reduction(+:i)", "!$omp do collapse(0)",
    "!$omp do collapse(5)", "!$omp do collapse(n)", "!$omp do collapse(3) collapse(3)",
    "!$omp parallel", "!$omp task", "!$omp simd", "!$omp do ordered", "!$omp do allocate(i)",
])
@pytest.mark.parametrize("sealed", [False, True])
def test_unsupported_clauses_and_missing_team_are_not_repaired(tmp_path, replacement, sealed):
    with pytest.raises(CompilationError):
        issue(fixture(tmp_path, SOURCE.replace("!$omp do collapse(3)", replacement), sealed))


@pytest.mark.parametrize("change", ["missing_end", "wrong_end", "late_end", "missing_loop",
                                   "while", "imperfect", "call", "nested_task", "exit"])
@pytest.mark.parametrize("sealed", [False, True])
def test_original_loop_and_end_association_must_remain_exact(tmp_path, change, sealed):
    source = SOURCE
    if change == "missing_end":
        source = source.replace("!$omp end do nowait\n", "")
    elif change == "wrong_end":
        source = source.replace("end do nowait", "end single nowait")
    elif change == "late_end":
        source = source.replace("!$omp end do nowait", "i=1\n!$omp end do nowait")
    elif change == "missing_loop":
        source = source.replace("!$omp do collapse(3)\n", "!$omp do collapse(3)\ni=1\n")
    elif change == "while":
        source = source.replace("do k=1,n", "do while(n>0)")
    elif change == "imperfect":
        source = source.replace("do j=1,n", "i=1\ndo j=1,n")
    elif change == "call":
        source = source.replace("b(i,j,k)=a(i,j,k)", "call external(b)")
    elif change == "nested_task":
        source = source.replace("b(i,j,k)=a(i,j,k)", "!$omp task\nb(i,j,k)=a(i,j,k)\n!$omp end task")
    else:
        source = source.replace("b(i,j,k)=a(i,j,k)", "exit")
    with pytest.raises(CompilationError):
        issue(fixture(tmp_path, source, sealed))


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("operation", ["issue", "validate"])
@pytest.mark.parametrize("mutation", ["namespace", "shape", "record", "source", "routine"])
def test_final_publication_cannot_reseed_source_or_token_receipts(tmp_path, monkeypatch, sealed, operation, mutation):
    parts = fixture(tmp_path, sealed=sealed)
    path, analysis, routine = parts
    proof = issue(parts) if operation == "validate" else None
    original = service._same_scope_state
    armed = True

    def late(state):
        nonlocal armed
        result = original(state)
        if armed:
            armed = False
            if mutation == "namespace":
                object.__setattr__(routine.scope, "imports", {"unknown": ("missing_module", "unknown")})
            elif mutation == "shape":
                axis = routine.scope.bindings["a"].shape_nodes[0]
                object.__setattr__(axis, "items", (None, None, "wrong"))
            elif mutation == "record":
                if proof is None:
                    object.__setattr__(routine.scope.bindings["i"], "root", "foreign::i")
                else:
                    object.__setattr__(proof, "_record", "{}")
            elif mutation == "source":
                path.write_text(path.read_text()+"! changed\n")
            else:
                object.__setattr__(routine, "qualified", "foreign::run")
        return result

    monkeypatch.setattr(service, "_same_scope_state", late)
    with pytest.raises(CompilationError):
        proof.validate(analysis, routine) if proof is not None else issue(parts)


@pytest.mark.parametrize("sealed", [False, True])
def test_copied_and_foreign_issued_tokens_are_not_query_authority(tmp_path, sealed):
    parts = fixture(tmp_path/"one", sealed=sealed)
    proof = issue(parts)
    foreign = issue(fixture(tmp_path/"two", sealed=sealed))
    registry = parts[1]._original_worksharing_storage
    for substitute in (copy(proof), foreign):
        registry[proof.identity] = substitute
        with pytest.raises(CompilationError):
            issue(parts)
        registry[proof.identity] = proof
    assert issue(parts) is proof


def test_no_new_native_completion_is_manufactured(tmp_path):
    parts = fixture(tmp_path)
    proof = issue(parts)
    summary = parts[1].summarize(ENTRY)
    assert not summary["native_completion"]["available"]
    assert not proof.public()["work_completion_authority"]


def test_pure_checkpoint_rejects_declaration_child_mutation(tmp_path):
    parts = fixture(tmp_path)
    proof = issue(parts)
    saved = proof.checkpoint()
    name = next(node for node in walk(parts[2].execution) if type(node).__name__ == "Name" and str(node) == "n")
    object.__setattr__(name, "string", "foreign_n")
    with pytest.raises(CompilationError):
        proof.require_checkpoint(saved)


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("where", ["issue", "validate", "reuse"])
@pytest.mark.parametrize("mutation", ["namespace", "bound", "registry", "graph", "token"])
def test_final_live_callback_keeps_original_raw_attachments(tmp_path, monkeypatch, sealed, where, mutation):
    parts = fixture(tmp_path, sealed=sealed)
    _path, analysis, routine = parts
    proof = issue(parts) if where != "issue" else None
    original = type(analysis.inputs).verify_live
    original_state = service._same_scope_state
    armed, at_publication = True, False

    def mark_publication(state):
        nonlocal at_publication
        result = original_state(state)
        at_publication = True
        return result

    monkeypatch.setattr(service, "_same_scope_state", mark_publication)

    def endpoint(inputs, *args, **kwargs):
        nonlocal armed
        result = original(inputs, *args, **kwargs)
        if inputs is analysis.inputs and armed and at_publication:
            armed = False
            if mutation == "namespace":
                object.__setattr__(routine.scope, "imports", {"x": ("unresolved_module", "x")})
            elif mutation == "bound":
                object.__setattr__(routine.scope.bindings["a"].shape_nodes[0], "items", (None, "foreign_bound"))
            elif mutation == "registry":
                analysis._original_worksharing_storage = dict(analysis._original_worksharing_storage)
            elif mutation == "graph":
                graph = proof._graph if proof is not None else next(iter(analysis._structures.values()))
                object.__setattr__(graph, "_ids", dict(graph._ids))
            elif proof is not None:
                object.__setattr__(proof, "_routine", copy(routine))
            else:
                object.__setattr__(routine.scope.bindings["i"], "kind", 8)
        return result

    monkeypatch.setattr(type(analysis.inputs), "verify_live", endpoint)
    with pytest.raises(CompilationError):
        proof.validate(analysis, routine) if where == "validate" else issue(parts)


@pytest.mark.parametrize("sealed", [False, True])
def test_default_no_clause_is_original_collapse_one(tmp_path, sealed):
    source = SOURCE.replace("do collapse(3)", "do")
    parts = fixture(tmp_path, source, sealed)
    assert issue(parts).public()["regions"][0]["collapse"] == 1


def test_public_record_is_detached_and_partial_or_copied_routine_is_not_authority(tmp_path):
    parts = fixture(tmp_path)
    proof = issue(parts)
    public = proof.public()
    public["regions"].clear()
    assert len(proof.public()["regions"]) == 1
    with pytest.raises(CompilationError):
        service.original_worksharing_storage(parts[1], copy(parts[2]))
    with pytest.raises(CompilationError):
        proof.validate(parts[1], copy(parts[2]))


@pytest.mark.parametrize("sealed", [False, True])
def test_existing_graph_is_baselined_before_first_original_validator(tmp_path, monkeypatch, sealed):
    parts = fixture(tmp_path, sealed=sealed)
    _path, analysis, routine = parts
    graph = analysis.structure(ENTRY)
    original = type(analysis)._require_original
    armed = True

    def before_query(owner, name, *args, **kwargs):
        nonlocal armed
        result = original(owner, name, *args, **kwargs)
        if owner is analysis and armed:
            armed = False
            object.__setattr__(graph, "_ids", dict(graph._ids))
        return result

    monkeypatch.setattr(type(analysis), "_require_original", before_query)
    with pytest.raises(CompilationError):
        issue(parts)


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("where", ["issue", "reuse", "validate"])
@pytest.mark.parametrize("change", ["metadata", "session"])
def test_publication_retains_input_and_session_fields(tmp_path, monkeypatch, sealed, where, change):
    parts = fixture(tmp_path, sealed=sealed)
    _path, analysis, routine = parts
    proof = issue(parts) if where != "issue" else None
    original = service._same_scope_state
    armed = True

    def endpoint(state):
        nonlocal armed
        result = original(state)
        if armed:
            armed = False
            if change == "metadata":
                object.__setattr__(analysis.inputs, "configuration", {"changed": True})
            elif sealed:
                object.__setattr__(analysis._source_session, "identity", "0"*64)
            else:
                analysis._source_session = object()
        return result

    monkeypatch.setattr(service, "_same_scope_state", endpoint)
    with pytest.raises(CompilationError):
        proof.validate(analysis, routine) if where == "validate" else issue(parts)


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("where", ["issue", "reuse", "validate"])
def test_record_encoding_precedes_final_live_source_fence(tmp_path, monkeypatch, sealed, where):
    parts = fixture(tmp_path, sealed=sealed)
    path, analysis, routine = parts
    proof = issue(parts) if where != "issue" else None
    original = service._encoded
    armed = True

    def late(record):
        nonlocal armed
        result = original(record)
        if armed:
            armed = False
            path.write_text(path.read_text()+"! changed at record encoding\n")
        return result

    monkeypatch.setattr(service, "_encoded", late)
    with pytest.raises(CompilationError):
        proof.validate(analysis, routine) if where == "validate" else issue(parts)
