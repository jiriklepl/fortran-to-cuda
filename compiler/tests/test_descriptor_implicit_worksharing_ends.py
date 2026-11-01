"""A missing OMP sentinel ends only at its genuine associated counted loop."""
from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.descriptor_teams import prove_descriptor_team
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

ENTRY = "implicit_team::run"
SOURCE = """module implicit_team
contains
subroutine run(a,n)
integer,intent(in)::n
real(8),intent(inout)::a(:,:)
integer::i,j,value
real(8)::total
!$omp parallel private(i,j,value) shared(a,n)
!$omp do collapse(2) schedule(dynamic,4)
do j=1,n
! ordinary comment before the collapsed loop
do i=1,n
a(i,j)=a(i,j)+1.0_8
enddo
enddo
value=1
!$omp barrier
!$omp do schedule(dynamic)
do j=1,n
a(1,j)=2.0_8*a(1,j)
enddo
!$omp do collapse(2) schedule(guided)
do j=1,n
do i=1,n
a(i,j)=3.0_8*a(i,j)
enddo
enddo
!$omp end parallel
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE, *, sealed=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "teams.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine = analysis.routines[ENTRY]
    groups = [item for item in grouped_original_sequence(routine.execution.content, {})
              if isinstance(item, NativeGroupCandidate)]
    assert len(groups) == 1
    return path, analysis, routine, groups[0]


def issue(parts):
    _path, analysis, routine, group = parts
    return prove_descriptor_team(analysis, routine.qualified, group.original_nodes)


@pytest.mark.parametrize("sealed", [False, True])
def test_three_original_implicit_worksharing_ends_keep_default_barriers_and_source(tmp_path, sealed):
    parts = fixture(tmp_path, sealed=sealed)
    before = parts[0].read_bytes()
    proof = issue(parts)
    public = proof.public()
    assert public["version"] == 4
    endings = public["implicit_worksharing_endings"]
    assert [row["collapse"] for row in endings] == [2, 1, 2]
    assert all(row["implicit_default_barrier_retained"] for row in endings)
    assert endings[0]["associated_loop_source_span"][1] < endings[1]["directive_source_span"][0]
    assert public["retained_internal_nowait"] == 0
    assert len(public["original_directives"]) == 6
    assert all(not public[name] for name in ("execution_authorized", "entry_definition_authority",
                                           "payload_access_authorized", "gpu_legality_established"))
    assert proof.validate(parts[1], ENTRY, proof.selected_node_ids) is proof
    assert issue(parts) is proof
    assert parts[0].read_bytes() == before


@pytest.mark.parametrize("ending", ["!$omp end do", "!$omp end do nowait"])
def test_mixed_implicit_and_explicit_endings_preserve_original_nowait(tmp_path, ending):
    source = SOURCE.replace("enddo\nvalue=1", "enddo\n" + ending + "\nvalue=1")
    public = issue(fixture(tmp_path, source)).public()
    assert public["version"] == 4
    assert len(public["implicit_worksharing_endings"]) == 2
    assert public["retained_internal_nowait"] == int("nowait" in ending)
    assert ending[6:] in public["original_directives"]


def test_one_implicit_worksharing_loop_closes_before_intervening_scalar_and_barrier(tmp_path):
    source = SOURCE[:SOURCE.index("!$omp do schedule(dynamic)")] + "!$omp end parallel\nend subroutine\nend module\n"
    public = issue(fixture(tmp_path, source)).public()
    assert len(public["implicit_worksharing_endings"]) == 1
    assert "barrier" in public["original_directives"]


@pytest.mark.parametrize("feature", ["master", "single", "reduction"])
def test_implicit_endings_retain_independent_master_single_reduction_features(tmp_path, feature):
    source = SOURCE
    if feature == "master":
        source = source.replace("value=1", "!$omp master\nvalue=1\n!$omp end master")
    elif feature == "single":
        source = source.replace("value=1", "!$omp single\nvalue=1\n!$omp end single nowait")
    else:
        source = source.replace("!$omp do schedule(dynamic)", "!$omp do schedule(dynamic) reduction(+:total)")
    public = issue(fixture(tmp_path, source)).public()
    assert public["version"] == 4
    if feature == "master":
        assert public["retained_master_regions"] == 1
        assert not public["master_barrier_assumed"]
        assert not public["master_completion_authorized"]
    elif feature == "single":
        assert public["retained_single_regions"] == 1
        assert public["retained_internal_nowait"] == 1
    else:
        assert len(public["intrinsic_scalar_reductions"]) == 1


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_next_block_leading_comments_keep_the_exact_explicit_or_implicit_loop_end(tmp_path, sealed, explicit):
    prefix = "!$omp end do\n" if explicit else ""
    source = SOURCE.replace("value=1", prefix + "!$omp master\nblock\ninteger::inner\ninner=1\nend block\n!$omp end master")
    public = issue(fixture(tmp_path, source, sealed=sealed)).public()
    assert public["version"] == 4
    assert public["retained_master_regions"] == 1
    assert len(public["implicit_worksharing_endings"]) == (2 if explicit else 3)
    assert public["original_directives"].count("end do") == int(explicit)


@pytest.mark.parametrize("change", ["late_end", "wrong_end", "repeated_end", "nested_task", "nested_parallel"])
def test_implicit_completion_does_not_repair_invalid_or_unsupported_directives(tmp_path, change):
    replacement = {"late_end": "value=1\n!$omp end do", "wrong_end": "!$omp end single\nvalue=1",
                   "repeated_end": "!$omp end do\n!$omp end do\nvalue=1",
                   "nested_task": "!$omp task\nvalue=1\n!$omp end task",
                   "nested_parallel": "!$omp parallel\nvalue=1\n!$omp end parallel"}[change]
    with pytest.raises(CompilationError):
        issue(fixture(tmp_path, SOURCE.replace("value=1", replacement)))


@pytest.mark.parametrize("header", ["do while(n>0)", "do", "do concurrent(i=1:n)"])
def test_implicit_end_requires_an_original_ordinary_counted_loop(tmp_path, header):
    source = SOURCE.replace("!$omp do collapse(2) schedule(dynamic,4)", "!$omp do schedule(dynamic,4)", 1)
    source = source.replace("do j=1,n", header, 1)
    with pytest.raises(CompilationError, match="counted DO"):
        issue(fixture(tmp_path, source))


def test_collapse_never_invents_a_missing_original_counted_loop(tmp_path):
    with pytest.raises(CompilationError, match="counted loop nest"):
        issue(fixture(tmp_path, SOURCE.replace("collapse(2)", "collapse(3)", 1)))


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("mutation", ["header", "ending", "loop_child", "iterator", "span"])
def test_original_loop_attachments_are_retained_before_source_callbacks(tmp_path, monkeypatch, sealed, mutation):
    parts = fixture(tmp_path, sealed=sealed)
    _path, analysis, _routine, group = parts
    loops = [node for root in group.original_nodes for node in walk(root)
             if type(node).__name__ == "Block_Nonlabel_Do_Construct"]
    loop = loops[0]
    header = next(node for node in loop.content if type(node).__name__ == "Nonlabel_Do_Stmt")
    ending = loop.content[-1]
    select = analysis._selected_source
    fired = False

    def selected(*args, **kwargs):
        nonlocal fired
        result = select(*args, **kwargs)
        if not fired:
            fired = True
            if mutation == "header":
                object.__setattr__(header, "items", tuple(list(header.items)))
            elif mutation == "ending":
                object.__setattr__(loop, "content", (*loop.content[:-1], copy(ending)))
            elif mutation == "loop_child":
                object.__setattr__(loop, "content", (*loop.content[:-1], copy(loop.content[-2]), ending))
            elif mutation == "iterator":
                iterator = header.items[1].items[1][0]
                object.__setattr__(iterator, "string", str(iterator) + "changed")
            else:
                object.__setattr__(ending.item, "span", (1, 1))
        return result

    monkeypatch.setattr(analysis, "_selected_source", selected)
    with pytest.raises(CompilationError):
        issue(parts)
    assert fired


def test_orphan_worksharing_outside_the_exact_original_join_is_not_absorbed(tmp_path):
    parts = fixture(tmp_path)
    orphan = "!$omp do\ndo j=1,n\na(1,j)=0\nenddo\n!$omp end do nowait\n"
    text = SOURCE.replace("!$omp end parallel", "!$omp end parallel\n" + orphan)
    _path, analysis, routine, group = fixture(tmp_path, text)
    tail = tuple(routine.execution.content)
    with pytest.raises(CompilationError, match="exact complete original joined region"):
        prove_descriptor_team(analysis, routine.qualified, tail)
    assert issue((_path, analysis, routine, group)).public()["version"] == 4
    assert parts[0].is_file()


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("mutation", ["header", "ending", "source"])
def test_implicit_association_rejects_final_publication_mutations(tmp_path, monkeypatch, sealed, mutation):
    parts = fixture(tmp_path, sealed=sealed)
    path, analysis, _routine, group = parts
    loop = next(node for root in group.original_nodes for node in walk(root)
                if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    header = next(node for node in loop.content if type(node).__name__ == "Nonlabel_Do_Stmt")
    ending = loop.content[-1]
    publish = analysis.verify_publication
    fired = False

    def publication():
        nonlocal fired
        publish()
        if not fired:
            fired = True
            if mutation == "header":
                object.__setattr__(header, "items", tuple(list(header.items)))
            elif mutation == "ending":
                object.__setattr__(ending.item, "span", (1, 1))
            else:
                path.write_text(path.read_text() + "! final callback changed the source\n")

    monkeypatch.setattr(analysis, "verify_publication", publication)
    with pytest.raises(CompilationError):
        issue(parts)
    assert fired


@pytest.mark.parametrize("sealed", [False, True])
def test_live_source_fence_follows_final_scope_serialization(tmp_path, monkeypatch, sealed):
    from compiler.frontend import optional_scalar_control

    parts = fixture(tmp_path, sealed=sealed)
    path = parts[0]
    same = optional_scalar_control._same_scope_state
    fired = False

    def same_scope(saved):
        nonlocal fired
        result = same(saved)
        if not fired:
            fired = True
            path.write_text(path.read_text() + "! source changed by final scope serializer\n")
        return result

    monkeypatch.setattr(optional_scalar_control, "_same_scope_state", same_scope)
    with pytest.raises(CompilationError):
        issue(parts)
    assert fired


def test_copied_or_foreign_tokens_do_not_replace_exact_registered_original_team(tmp_path):
    parts = fixture(tmp_path)
    proof = issue(parts)
    with pytest.raises(CompilationError, match="compiler-issued authority"):
        copy(proof).validate(parts[1], ENTRY, proof.selected_node_ids)
    foreign = fixture(tmp_path / "foreign")
    with pytest.raises(CompilationError, match="compiler-issued authority"):
        proof.validate(foreign[1], ENTRY, proof.selected_node_ids)


def mutate_namespace(analysis, routine, mutation):
    scope = routine.scope
    if mutation == "imports":
        object.__setattr__(scope, "imports", {"foreign": ("unresolved_module", "foreign")})
    elif mutation == "imports_entry":
        dict.__setitem__(scope.imports, "foreign", ("unresolved_module", "foreign"))
    elif mutation == "parent":
        object.__setattr__(scope, "parent", None)
    elif mutation == "wildcards":
        object.__setattr__(scope, "wildcards", (*scope.wildcards, "unresolved_module"))
    elif mutation == "kinds":
        object.__setattr__(scope.kinds, "values", {**scope.kinds.values, "foreign": 8})
    elif mutation == "routine_entry":
        dict.__setitem__(analysis.routines, ENTRY, copy(routine))
    else:
        raise AssertionError(mutation)


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("mutation", ["imports", "imports_entry", "parent", "wildcards", "kinds", "routine_entry"])
def test_final_namespace_serializer_cannot_detach_consumed_original_scope(tmp_path, monkeypatch, sealed, existing, mutation):
    from compiler.frontend import optional_scalar_control

    parts = fixture(tmp_path, sealed=sealed)
    _path, analysis, routine, _group = parts
    proof = issue(parts) if existing else None
    same = optional_scalar_control._same_scope_state
    fired = False

    def same_scope(saved):
        nonlocal fired
        result = same(saved)
        if not fired and saved[0] is routine.scope:
            fired = True
            mutate_namespace(analysis, routine, mutation)
        return result

    def consume():
        return issue(parts) if proof is None else proof.validate(analysis, ENTRY, proof.selected_node_ids)

    monkeypatch.setattr(optional_scalar_control, "_same_scope_state", same_scope)
    if sealed and existing:
        assert consume() is proof
        assert not fired
        return
    with pytest.raises(CompilationError):
        consume()
    assert fired


@pytest.mark.parametrize("sealed", [False, True])
def test_initial_scope_serializer_cannot_reseed_implicit_namespace_receipt(tmp_path, monkeypatch, sealed):
    from compiler.frontend import optional_scalar_control

    parts = fixture(tmp_path, sealed=sealed)
    _path, analysis, routine, _group = parts
    scope_state = optional_scalar_control._scope_state
    fired = False

    def state(scope):
        nonlocal fired
        result = scope_state(scope)
        if not fired and scope is routine.scope:
            fired = True
            mutate_namespace(analysis, routine, "imports")
        return result

    monkeypatch.setattr(optional_scalar_control, "_scope_state", state)
    with pytest.raises(CompilationError):
        issue(parts)
    assert fired


@pytest.mark.parametrize("sealed", [False, True])
def test_final_reduction_namespace_serializer_keeps_original_host_scope(tmp_path, monkeypatch, sealed):
    from compiler.frontend import optional_scalar_control

    source = SOURCE.replace("!$omp do schedule(dynamic)", "!$omp do schedule(dynamic) reduction(+:total)")
    parts = fixture(tmp_path, source, sealed=sealed)
    _path, _analysis, routine, _group = parts
    host = routine.scope.parent
    same = optional_scalar_control._same_scope_state
    fired = False

    def same_scope(saved):
        nonlocal fired
        result = same(saved)
        if not fired and saved[0] is host:
            fired = True
            object.__setattr__(host, "imports", {"foreign": ("unresolved_module", "foreign")})
        return result

    monkeypatch.setattr(optional_scalar_control, "_same_scope_state", same_scope)
    with pytest.raises(CompilationError):
        issue(parts)
    assert fired


@pytest.mark.parametrize("sealed", [False, True])
def test_last_live_file_callback_cannot_change_previously_checked_namespace(tmp_path, monkeypatch, sealed):
    from compiler.frontend import optional_scalar_control

    parts = fixture(tmp_path, sealed=sealed)
    _path, analysis, routine, _group = parts
    same = optional_scalar_control._same_scope_state
    live = type(analysis.inputs).verify_live
    ready, fired = False, False

    def same_scope(saved):
        nonlocal ready
        result = same(saved)
        if saved[0] is routine.scope:
            ready = True
        return result

    def verify_live(inputs, *args, **kwargs):
        nonlocal fired
        result = live(inputs, *args, **kwargs)
        if ready and not fired and inputs is analysis.inputs:
            fired = True
            mutate_namespace(analysis, routine, "imports_entry")
        return result

    monkeypatch.setattr(optional_scalar_control, "_same_scope_state", same_scope)
    monkeypatch.setattr(type(analysis.inputs), "verify_live", verify_live)
    with pytest.raises(CompilationError):
        issue(parts)
    assert fired


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("bound", ["shape", "lower"])
def test_late_namespace_serializer_cannot_mutate_private_declaration_syntax(tmp_path, monkeypatch, sealed, existing, bound):
    from compiler.frontend import optional_scalar_control

    source = SOURCE.replace("integer::i,j,value", "integer::i,j,value\nreal(8)::scratch(n:2*n)")
    source = source.replace("private(i,j,value)", "private(i,j,value,scratch)")
    parts = fixture(tmp_path, source, sealed=sealed)
    _path, analysis, routine, _group = parts
    binding = routine.scope.bindings["scratch"]
    if bound == "shape":
        leaf = next(node for node in walk(binding.shape_nodes[0].items[1]) if type(node).__name__ == "Name")
    else:
        leaf = next(node for node in walk(binding.lower_bound_nodes[0]) if type(node).__name__ == "Name")
    proof = issue(parts) if existing else None
    same = optional_scalar_control._same_scope_state
    fired = False

    def same_scope(saved):
        nonlocal fired
        result = same(saved)
        if not fired and saved[0] is routine.scope:
            fired = True
            object.__setattr__(leaf, "string", "unresolved_shape_name")
        return result

    def consume():
        return issue(parts) if proof is None else proof.validate(analysis, ENTRY, proof.selected_node_ids)

    monkeypatch.setattr(optional_scalar_control, "_same_scope_state", same_scope)
    if sealed and existing:
        assert consume() is proof
        assert not fired
        return
    with pytest.raises(CompilationError):
        consume()
    assert fired
