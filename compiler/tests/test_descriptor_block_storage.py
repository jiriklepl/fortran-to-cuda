"""Complete source-team storage discharges BLOCK, independently of graph size."""
from copy import copy
from inspect import currentframe

import pytest
from fparser.two.utils import walk

from compiler.frontend.descriptor_block_storage import original_joined_block_storage
from compiler.frontend.descriptor_teams import prove_descriptor_team
from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

ENTRY = "blocks::run"


def source(*, body="a(1)=value", tail="", region="master", count=1):
    loop = "!$omp do reduction(+:total)\ndo i=1,n\ntotal=total+a(i)\nenddo\n!$omp end do\n"
    return ("module blocks\ncontains\nsubroutine run(a,n)\nreal(8),intent(inout)::a(:)\n"
            "integer,intent(in)::n\ninteger::i\nreal(8)::total\nreal(8),allocatable,save::hidden(:)\n"
            "total=0\n!$omp parallel private(i)\n" + loop * count + "!$omp " + region + "\n"
            "block\nreal(8)::value\nvalue=total\n" + body + "\nend block\n!$omp end " + region
            + "\n" + tail + "\n!$omp end parallel\nend subroutine\nend module\n")


def fixture(tmp_path, *, text=None, sealed=True):
    path = tmp_path / "blocks.f90"
    path.write_text(source() if text is None else text)
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine = analysis.routines[ENTRY]
    graph = analysis.structure(ENTRY)
    original = next(node for node in walk(routine.execution) if type(node).__name__ == "Block_Construct")
    group, = (item for item in grouped_original_sequence(routine.execution.content, {})
              if isinstance(item, NativeGroupCandidate))
    team = prove_descriptor_team(analysis, ENTRY, group.original_nodes)
    return path, analysis, routine, graph, graph.node_id(original), original, team


def compose(data, **kwargs):
    _path, analysis, routine, graph, identity, original, team = data
    return original_joined_block_storage(analysis, routine, graph, identity, original, team, **kwargs)


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("region", ["single", "master"])
def test_small_original_graph_block_has_complete_team_and_no_execution_authority(tmp_path, sealed, region):
    data = fixture(tmp_path, text=source(region=region), sealed=sealed)
    _path, analysis, routine, graph, identity, original, team = data
    assert not graph.native_groups
    proof = compose(data)
    public = proof.public()
    assert proof.validate() is proof
    assert public["original_node_id"] == identity
    assert public["team_storage"] == team.public()
    assert public["storage_mutation"]["complete"]
    assert len(public["storage_mutation"]["fixed_scalar_blocks"]) == 1
    assert public["complete_original_group_required"]
    assert public["complete_original_block_required"]
    for key in ("execution_authorized", "entry_definition_authority", "payload_access_authorized",
                "alias_authority", "motion_authorized", "numerical_completion_authorized"):
        assert public[key] is False
    assert original is proof._block._original
    assert analysis.summarize(routine.qualified)["complete"] is False


@pytest.mark.parametrize("region", ["single", "master"])
def test_window20_uses_the_same_complete_body_contract_for_ordinary_graph(tmp_path, region):
    _path, analysis, routine, graph, _id, _node, team = fixture(tmp_path, text=source(region=region))
    public = prove_descriptor_window(analysis, ENTRY, team.selected_node_ids, ("argument::a",)).public()
    assert not graph.native_groups
    assert public["schema_version"] == 20
    requirement, = public["native_joined_block_requirements"]
    assert requirement["storage_mutation"]["complete"]
    assert requirement["team_storage"]["version"] == (3 if region == "master" else 2)
    assert analysis._summary_authority()["descriptor_window_joined_block_version"] == 20
    assert public["execution_authorized"] is False
    assert public["entry_definition_authority"] is False


@pytest.mark.parametrize("tail", ["deallocate(hidden)", "call unknown_export(a)"])
@pytest.mark.parametrize("where", ["body", "tail"])
def test_full_original_body_and_following_suffix_are_required(tmp_path, tail, where):
    data = fixture(tmp_path, text=source(**{where: tail}))
    with pytest.raises(CompilationError):
        compose(data)


@pytest.mark.parametrize("case", ["graph", "analysis", "team_copy", "wrong_node", "wrong_block", "wrong_team"])
def test_exact_source_analysis_graph_membership_and_issued_team_are_required(tmp_path, case):
    data = list(fixture(tmp_path))
    if case == "graph":
        data[3] = copy(data[3])
    elif case == "analysis":
        data[1] = data[1].private_projection()
    elif case == "team_copy":
        data[6] = copy(data[6])
    elif case == "wrong_node":
        data[4] = "not-an-original-node"
    elif case == "wrong_block":
        data[5] = copy(data[5])
    else:
        other = tmp_path / "other"
        other.mkdir()
        data[6] = fixture(other)[6]
    with pytest.raises(CompilationError):
        compose(data)


def test_outside_joined_team_block_does_not_inherit_the_team_receipt(tmp_path):
    data = list(fixture(tmp_path, text=source().replace("!$omp end parallel", "!$omp end parallel\ntotal=1\nblock\ninteger::unrelated\nunrelated=1\nend block")))
    _path, analysis, routine, graph, _identity, _original, team = data
    blocks = [node for node in walk(routine.execution) if type(node).__name__ == "Block_Construct"]
    data[4], data[5] = graph.node_id(blocks[1]), blocks[1]
    with pytest.raises(CompilationError, match="complete-team membership"):
        compose(data)


@pytest.mark.parametrize("case", ["team", "block", "storage", "selection", "record", "identity"])
def test_retained_independent_receipts_cannot_be_replaced_or_retagged(tmp_path, case):
    data = fixture(tmp_path)
    proof = compose(data)
    if case in {"team", "block", "storage"}:
        object.__setattr__(proof, "_" + case, copy(getattr(proof, "_" + case)))
    elif case == "selection":
        object.__setattr__(proof._storage, "_selection", proof._storage._selection[:-1])
    elif case == "identity":
        object.__setattr__(proof, "identity", "a" * 64)
    else:
        object.__setattr__(proof, "_record", "{}")
    with pytest.raises(CompilationError):
        proof.validate()


@pytest.mark.parametrize("change", ["team_copy", "team_registry", "team_record", "namespace", "root_registry"])
def test_later_issuer_callback_cannot_reseed_earlier_consumed_team_baseline(tmp_path, monkeypatch, change):
    from compiler.frontend.descriptor_blocks import FixedScalarBlock

    data = fixture(tmp_path, sealed=False)
    _path, analysis, routine, _graph, _identity, _original, team = data
    validate = FixedScalarBlock.validate
    fired = False

    def inspect(self, *args, **kwargs):
        nonlocal fired
        result = validate(self, *args, **kwargs)
        if not fired:
            fired = True
            if change == "team_copy":
                object.__setattr__(team, "_originals", tuple(list(team._originals)))
            elif change == "team_registry":
                analysis._descriptor_team_proofs[team.identity] = copy(team)
            elif change == "team_record":
                object.__setattr__(team, "_record", "{}")
            elif change == "namespace":
                routine.scope.imports["forged"] = ("unknown", "kind")
            else:
                analysis._block_scopes = dict(analysis._block_scopes)
        return result

    monkeypatch.setattr(FixedScalarBlock, "validate", inspect)
    with pytest.raises(CompilationError):
        compose(data)
    assert fired


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("change", ["storage_copy", "storage_registry", "own_team", "block_binding", "registry_owner", "source"])
def test_final_publication_keeps_exact_composition_nested_tokens_and_declarations(tmp_path, monkeypatch, sealed, change):
    data = fixture(tmp_path, sealed=sealed)
    path, analysis, _routine, _graph, _identity, _original, _team = data
    proof = compose(data)
    publish = analysis.verify_publication
    fired = False

    def inspect():
        nonlocal fired
        result = publish()
        caller = currentframe().f_back
        if not fired and caller.f_locals.get("self") is proof:
            fired = True
            if change == "storage_copy":
                object.__setattr__(proof, "_storage", copy(proof._storage))
            elif change == "storage_registry":
                analysis._storage_mutation_effects[proof._storage.identity] = copy(proof._storage)
            elif change == "own_team":
                object.__setattr__(proof, "_team", copy(proof._team))
            elif change == "block_binding":
                object.__setattr__(proof._block._scope.bindings["value"], "kind", 4)
            elif change == "registry_owner":
                analysis._descriptor_blocks_owner = object()
            else:
                path.write_text(path.read_text() + "! changed live source\n")
        return result

    monkeypatch.setattr(analysis, "verify_publication", inspect)
    with pytest.raises(CompilationError):
        proof.validate()
    assert fired


def test_final_window_callback_cannot_detach_previously_checked_block_composition(tmp_path, monkeypatch):
    from compiler.frontend.descriptor_windows import _Check

    _path, analysis, _routine, _graph, _identity, _original, team = fixture(tmp_path, sealed=False)
    validate = _Check.validate_native_intrinsics
    fired = False

    def inspect(self, *args, **kwargs):
        nonlocal fired
        result = validate(self, *args, **kwargs)
        if self.joined_blocks and kwargs.get("rederive") is False and not fired:
            fired = True
            proof, _baseline = next(iter(self.joined_blocks.values()))
            object.__setattr__(proof, "_storage", copy(proof._storage))
        return result

    monkeypatch.setattr(_Check, "validate_native_intrinsics", inspect)
    with pytest.raises(CompilationError, match="joined BLOCK"):
        prove_descriptor_window(analysis, ENTRY, team.selected_node_ids, ("argument::a",))
    assert fired


@pytest.mark.parametrize("change", ["copied_comment", "removed_comment", "missing_body", "extra_outside"])
def test_joined_comment_normalization_never_accepts_a_different_original_team(tmp_path, change):
    from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects

    _path, analysis, routine, graph, identity, original, team = fixture(tmp_path, sealed=False)
    originals = list(team._originals)
    comment = next(index for index, node in enumerate(originals)
                   if type(node).__name__ == "Comment" and not str(node).lstrip().startswith("!$"))
    if change == "copied_comment":
        # fparser comments have no pickle string in an unsealed diagnostic.
        foreign = object.__new__(type(originals[comment]))
        vars(foreign).update(vars(originals[comment]))
        originals[comment] = foreign
    elif change == "removed_comment":
        originals.pop(comment)
    elif change == "missing_body":
        originals.remove(original)
    else:
        originals.append(routine.execution.content[0])
    object.__setattr__(team, "_originals", tuple(originals))
    with pytest.raises(CompilationError):
        prove_storage_mutation_effects(analysis, ENTRY, selection=team.selected_node_ids, joined_storage=team)
    assert graph.nodes[identity].kind == "boundary"


def test_matrix18_outside_team_and_block20_keep_all_typed_requirements(tmp_path):
    from compiler.frontend.descriptor_windows import prove_descriptor_window

    text = source().replace("real(8)::total", "real(8)::total\nreal(8)::x(3,3),y(3,3),z(3,3)")
    text = text.replace("end subroutine", "z=matmul(x,y)\nend subroutine")
    _path, analysis, routine, graph, _identity, _original, _team = fixture(tmp_path, text=text)
    public = prove_descriptor_window(analysis, ENTRY, graph.nodes[graph.root_id].children, ("argument::a",)).public()
    assert public["schema_version"] == 20
    assert public["native_joined_block_requirements"][0]["team_storage"]["version"] == 3
    assert public["native_intrinsic_descriptor_requirements"][0]["intrinsic"] == "matmul"
    assert not public["execution_authorized"]


@pytest.mark.parametrize("sealed", [False, True])
def test_initial_original_authentication_cannot_reseed_a_consumed_declaration(tmp_path, monkeypatch, sealed):
    from inspect import currentframe

    data = fixture(tmp_path, sealed=sealed)
    _path, analysis, routine, _graph, _identity, _original, _team = data
    original = analysis._require_original
    fired = False

    def inspect(*args, **kwargs):
        nonlocal fired
        result = original(*args, **kwargs)
        caller = currentframe().f_back
        if not fired and caller.f_globals.get("__name__") == "compiler.frontend.descriptor_block_storage":
            fired = True
            object.__setattr__(routine.scope.bindings["a"], "root", "forged::a")
        return result

    monkeypatch.setattr(analysis, "_require_original", inspect)
    with pytest.raises(CompilationError):
        compose(data)
    assert fired


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("change", ["mapping", "entry"])
def test_final_live_input_callback_cannot_replace_the_dynamic_component_type_namespace(tmp_path, monkeypatch, sealed, change):
    text = source().replace("module blocks\n", "module blocks\ntype::original_type\nreal(8)::payload\nend type\n")
    data = fixture(tmp_path, text=text, sealed=sealed)
    _path, analysis, routine, _graph, _identity, _original, _team = data
    proof = compose(data)
    scope = routine.scope.parent
    assert "original_type" in scope.component_types
    input_type = type(analysis.inputs)
    verify_live = input_type.verify_live
    fired = False

    def inspect(inputs):
        nonlocal fired
        result = verify_live(inputs)
        if inputs is analysis.inputs and not fired and currentframe().f_back.f_locals.get("self") is proof:
            fired = True
            if change == "mapping":
                object.__setattr__(scope, "component_types", dict(scope.component_types))
            else:
                dict.__setitem__(scope.component_types, "original_type", copy(scope.component_types["original_type"]))
        return result

    monkeypatch.setattr(input_type, "verify_live", inspect)
    with pytest.raises(CompilationError, match="joined BLOCK source or retained receipt"):
        proof.validate()
    assert fired


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("change", ["binding", "team", "component_map"])
def test_baseline_serialization_cannot_reseed_previously_consumed_raw_attachments(tmp_path, monkeypatch, sealed, change):
    from compiler.frontend import descriptor_block_storage as storage

    data = fixture(tmp_path, sealed=sealed)
    _path, _analysis, routine, _graph, _identity, _original, team = data
    binding = routine.scope.bindings["a"]
    binding_type, public = type(binding), type(binding).public
    fired = False

    def inspect(value):
        nonlocal fired
        frame = currentframe().f_back
        while frame is not None:
            if frame.f_globals.get("__name__") == storage.__name__ and frame.f_code.co_name == "_scopes":
                if value is binding and not fired:
                    fired = True
                    if change == "binding":
                        object.__setattr__(binding, "root", "forged::a")
                    elif change == "team":
                        object.__setattr__(team, "_originals", tuple(list(team._originals)))
                    else:
                        object.__setattr__(routine.scope.parent, "component_types", dict(routine.scope.parent.component_types))
                break
            frame = frame.f_back
        return public(value)

    monkeypatch.setattr(binding_type, "public", inspect)
    with pytest.raises(CompilationError):
        compose(data)
    assert fired


@pytest.mark.parametrize("change", ["earlier_namespace", "own_storage"])
def test_last_namespace_serializer_cannot_mutate_an_earlier_final_receipt(tmp_path, monkeypatch, change):
    from compiler.frontend import descriptor_block_storage as storage

    data = fixture(tmp_path)
    _path, analysis, routine, _graph, _identity, _original, _team = data
    proof = compose(data)
    baseline = proof.checkpoint()
    value = proof._block._scope.bindings["value"]
    binding_type, public = type(value), type(value).public
    fired = False

    def inspect(binding):
        nonlocal fired
        frame = currentframe().f_back
        while frame is not None:
            if frame.f_globals.get("__name__") == storage.__name__ and frame.f_code.co_name == "_same_scopes":
                if binding is value and not fired:
                    fired = True
                    if change == "earlier_namespace":
                        object.__setattr__(routine.scope.bindings["a"], "root", "forged::a")
                    else:
                        object.__setattr__(proof, "_storage", copy(proof._storage))
                break
            frame = frame.f_back
        return public(binding)

    monkeypatch.setattr(binding_type, "public", inspect)
    with pytest.raises(CompilationError, match="joined BLOCK consumed source"):
        proof.require_checkpoint(baseline)
    assert fired
    assert analysis is proof._analysis
