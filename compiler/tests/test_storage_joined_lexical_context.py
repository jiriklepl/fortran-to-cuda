"""Selected joined teams retain exact complete enclosing lexical bodies."""
from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.descriptor_associations import FixedScalarAssociation
from compiler.frontend.descriptor_block_storage import original_joined_block_storage
from compiler.frontend.descriptor_teams import prove_descriptor_team
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.storage_mutation_effects import _Scan, prove_storage_mutation_effects
from compiler.ir import CompilationError

ENTRY = "lexical_teams::run"
SOURCE = """module lexical_teams
type::reference_type
real(8)::point(3)
end type
contains
subroutine run(a)
real(8),intent(inout)::a(:)
type(reference_type)::reference
real(8),allocatable,save::hidden(:)
integer::counter,i
counter=0
associate(ref=>reference)
!$omp parallel private(i)
!$omp do
do i=1,size(a)
a(i)=a(i)
enddo
!$omp end do
!$omp master
block
real(8)::value
value=ref%point(1)
a(1)=value
end block
!$omp end master
!$omp end parallel
end associate
end subroutine
end module
"""


def fixture(tmp_path, text=SOURCE, *, sealed=True):
    path = tmp_path / "original.f90"
    path.write_text(text)
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine = analysis.routines[ENTRY]
    associates = tuple(node for node in walk(routine.execution) if type(node).__name__ == "Associate_Construct")
    parent = associates[-1] if associates else routine.execution
    items = parent.content[1:-1] if associates else parent.content
    groups = tuple(item for item in grouped_original_sequence(items, {}) if isinstance(item, NativeGroupCandidate))
    team = prove_descriptor_team(analysis, ENTRY, groups[0].original_nodes)
    return path, analysis, routine, associates, team


def storage(parts):
    _path, analysis, routine, _associates, team = parts
    return prove_storage_mutation_effects(analysis, routine.qualified,
                                         selection=team.selected_node_ids, joined_storage=team)


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_original_enclosing_fixed_associations_are_consumed_in_order(tmp_path, sealed, nested):
    text = SOURCE
    if nested:
        text = text.replace("associate(ref=>reference)", "associate(count=>counter)\nassociate(ref=>reference)")
        text = text.replace("end associate", "end associate\nend associate")
    parts = fixture(tmp_path, text, sealed=sealed)
    path, analysis, routine, associates, team = parts
    before = path.read_bytes()
    proof = storage(parts)
    public = proof.public()
    assert public["version"] == 8
    assert public["complete"], public["frontiers"]
    assert proof.preserves_descriptors()
    assert public["original_selection"]["selected_node_ids"] == list(team.selected_node_ids)
    assert len(public["enclosing_lexical_context"]) == len(associates)
    assert [row["association"]["original_header_sha256"] for row in public["enclosing_lexical_context"]] == [
        item[2]["original_header_sha256"] for item in _association_receipts(analysis, routine, associates)]
    for flag in ("entry_definition_authority", "payload_access_authorized", "capture_authorized",
                 "execution_authorized", "motion_authorized", "gpu_legality_established", "alias_authority"):
        assert public[flag] is False
    block = next(node for node in walk(routine.execution) if type(node).__name__ == "Block_Construct")
    graph = analysis.structure(ENTRY)
    composition = original_joined_block_storage(analysis, routine, graph, graph.node_id(block), block, team)
    assert composition.public()["storage_mutation"]["version"] == 8
    assert path.read_bytes() == before


def _association_receipts(analysis, routine, associates):
    from compiler.frontend.descriptor_associations import original_fixed_association

    return [(analysis, routine, original_fixed_association(analysis, routine, node).public()) for node in associates]


def test_no_enclosing_context_keeps_the_existing_record_shape(tmp_path):
    text = SOURCE.replace("associate(ref=>reference)\n", "").replace("end associate\n", "").replace("ref%", "reference%")
    record = storage(fixture(tmp_path, text)).public()
    assert record["version"] == 7
    assert record["complete"]
    assert "enclosing_lexical_context" not in record


@pytest.mark.parametrize("where", ["prefix", "suffix"])
@pytest.mark.parametrize("statement", ["deallocate(hidden)", "call unknown_export(ref)", "call unknown_export(a)"])
def test_entire_original_enclosing_body_is_scanned(tmp_path, where, statement):
    text = SOURCE.replace("!$omp parallel", statement + "\n!$omp parallel", 1) if where == "prefix" else SOURCE.replace(
        "end associate", statement + "\nend associate")
    record = storage(fixture(tmp_path, text)).public()
    assert record["mutations"] or record["frontiers"] or record["escapes"]


def test_unrelated_original_routine_prefix_and_suffix_are_not_substituted(tmp_path):
    text = SOURCE.replace("associate(ref=>reference)", "call unknown_outside(a)\nassociate(ref=>reference)")
    text = text.replace("end associate", "end associate\ncall another_unknown_outside(a)")
    assert storage(fixture(tmp_path, text)).public()["complete"]


@pytest.mark.parametrize("selector", ["reference%point", "reference%point(1)+1.0d0"])
def test_unsupported_selector_cannot_reach_joined_context(tmp_path, selector):
    text = SOURCE.replace("ref=>reference", "ref=>" + selector)
    # Existing structured association admission rejects these before a team
    # token exists; the new context route cannot manufacture that authority.
    with pytest.raises(CompilationError, match="original source authority"):
        storage(fixture(tmp_path, text))


@pytest.mark.parametrize("sealed", [False, True])
def test_unknown_original_enclosing_control_is_not_skipped(tmp_path, sealed):
    text = SOURCE.replace("associate(ref=>reference)", "if(counter==0)then\nassociate(ref=>reference)")
    text = text.replace("end associate", "end associate\nendif")
    with pytest.raises(CompilationError, match="enclosing lexical frame"):
        storage(fixture(tmp_path, text, sealed=sealed))


@pytest.mark.parametrize("mutation", ["analysis", "routine", "original", "association", "record"])
def test_late_association_validation_cannot_replace_consumed_context(tmp_path, monkeypatch, mutation):
    parts = fixture(tmp_path)
    original = FixedScalarAssociation.validate
    calls = {}

    def changed(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        calls[id(self)] = calls.get(id(self), 0) + 1
        if calls[id(self)] == 2:
            name = "_" + mutation
            object.__setattr__(self, name, "{}" if mutation == "record" else copy(getattr(self, name)))
        return result

    monkeypatch.setattr(FixedScalarAssociation, "validate", changed)
    with pytest.raises(CompilationError):
        storage(parts)


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("mutation", ["binding", "scope_table", "team_registry", "team_record", "graph", "graph_map"])
def test_final_callback_cannot_mutate_earlier_context_namespace_or_team(tmp_path, monkeypatch, sealed, mutation):
    parts = fixture(tmp_path, sealed=sealed)
    _path, analysis, routine, associates, team = parts
    finish = _Scan.finish

    def changed(self):
        result = finish(self)
        if self.enclosing_context is not None:
            if mutation == "binding":
                object.__setattr__(routine.scope.bindings["reference"], "root", "forged::reference")
            elif mutation == "scope_table":
                analysis._associate_scopes = dict(analysis._associate_scopes)
            elif mutation == "team_registry":
                analysis._descriptor_team_proofs[team.identity] = copy(team)
            elif mutation == "team_record":
                object.__setattr__(team, "_record", "{}")
            elif mutation == "graph":
                object.__setattr__(self.enclosing_context.graph, "identity", "a" * 64)
            else:
                graph = self.enclosing_context.graph
                object.__setattr__(graph, "_ids", dict(graph._ids))
        return result

    monkeypatch.setattr(_Scan, "finish", changed)
    with pytest.raises(CompilationError):
        storage(parts)


def test_a_cached_token_still_authenticates_original_selection_and_team(tmp_path):
    parts = fixture(tmp_path)
    proof = storage(parts)
    assert storage(parts) is proof
    with pytest.raises(CompilationError):
        proof.validate(joined_storage=copy(parts[-1]))
    assert proof.validate() is proof


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("mutation", ["alias", "scope", "original_child"])
def test_first_charged_callback_cannot_reseed_original_context(tmp_path, monkeypatch, sealed, mutation):
    parts = fixture(tmp_path, sealed=sealed)
    _path, analysis, _routine, associates, _team = parts
    original = _Scan.tick
    fired = False

    def changed(self, *args):
        nonlocal fired
        result = original(self, *args)
        if self.enclosing_context is not None and not fired:
            fired = True
            association = analysis._associate_scopes[id(associates[0])]
            if mutation == "alias":
                binding = association.scope.bindings["ref"]
                object.__setattr__(association.scope, "bindings", {**association.scope.bindings, "ref": copy(binding)})
            elif mutation == "scope":
                object.__setattr__(association, "scope", copy(association.scope))
            else:
                node = associates[0]
                object.__setattr__(node, "content", [*node.content[:-1], copy(node.content[-1])])
        return result

    monkeypatch.setattr(_Scan, "tick", changed)
    with pytest.raises(CompilationError):
        storage(parts)
    assert fired


@pytest.mark.parametrize("sealed", [False, True])
def test_namespace_serializer_cannot_mutate_root_before_baseline(tmp_path, monkeypatch, sealed):
    from compiler.frontend.source_effects import Binding

    parts = fixture(tmp_path, sealed=sealed)
    _path, _analysis, routine, _associates, _team = parts
    binding = routine.scope.bindings["reference"]
    original = Binding.public
    fired = False

    def changed(self):
        nonlocal fired
        if self is binding and not fired:
            fired = True
            object.__setattr__(self, "root", "forged::reference")
        return original(self)

    monkeypatch.setattr(Binding, "public", changed)
    with pytest.raises(CompilationError):
        storage(parts)
    assert fired


@pytest.mark.parametrize("sealed", [False, True])
def test_later_nested_token_cannot_mutate_already_consumed_outer_token(tmp_path, monkeypatch, sealed):
    text = SOURCE.replace("associate(ref=>reference)", "associate(count=>counter)\nassociate(ref=>reference)")
    text = text.replace("end associate", "end associate\nend associate")
    parts = fixture(tmp_path, text, sealed=sealed)
    outer, inner = parts[3]
    original = FixedScalarAssociation.validate
    retained = []

    def changed(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if self._original is outer:
            retained.append(self)
        elif self._original is inner and retained:
            object.__setattr__(retained[0], "_record", "{}")
        return result

    monkeypatch.setattr(FixedScalarAssociation, "validate", changed)
    with pytest.raises(CompilationError):
        storage(parts)


def test_unselected_task_directives_inside_enclosing_body_remain_frontiers(tmp_path):
    text = SOURCE.replace("end associate", "!$omp task\na(1)=0\n!$omp end task\nend associate")
    record = storage(fixture(tmp_path, text)).public()
    assert record["frontiers"]
