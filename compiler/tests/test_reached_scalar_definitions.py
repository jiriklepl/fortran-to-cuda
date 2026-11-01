"""Local line bounds come from reached original assignments, never local facts."""
from copy import copy, deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from compiler.frontend.reached_scalar_definitions import prove_reached_scalar_definitions
from compiler.frontend.source_device_tasks import source_device_task
from compiler.ir import CompilationError
from compiler.tests.test_line_snapshot import source
from compiler.tests.test_line_source_tasks import source_builder


def local_source(prefix="mx=ubound(a,1)-3", *, axis=1, guard=None):
    text = source(axis)
    text = text.replace("integer::i,j,k,ii,jj,kk", "integer::i,j,k,ii,jj,kk,mx,undefined")
    text = text.replace("shared(a,mask,nx,ny,nz)", "shared(a,mask,nx,ny,nz,mx)")
    consumer = {1: "i", 2: "j", 3: "k"}[axis]
    extent = {1: "nx", 2: "ny", 3: "nz"}[axis]
    text = text.replace(f"do {consumer}=1,{extent}", f"do {consumer}=1,mx")
    text = text.replace("!$omp parallel do", prefix + "\n" + (guard + "\n" if guard else "") + "!$omp parallel do", 1)
    if guard:
        text = text.replace("!$omp end parallel do", "!$omp end parallel do\nendif")
    return text


def issue(builder, nodes):
    return source_device_task(builder.analysis, builder.entry, nodes, builder=builder)


@pytest.mark.parametrize("axis", [1, 2, 3])
@pytest.mark.parametrize("guard", [None, "if(nx>0)then"])
def test_original_descriptor_preparation_defines_only_reached_local_bound(tmp_path, axis, guard):
    prefix = "undefined=lbound(a,1)+3\nmx=ubound(a," + str(axis) + ")-3"
    builder, nodes = source_builder(tmp_path, local_source(prefix, axis=axis, guard=guard))
    before = copy(builder.facts["captures"])
    proof = issue(builder, nodes)
    assert proof.validate() is proof
    assert builder.facts["captures"] == before
    authority = next(row["authority"] for row in proof.public()["scalar_captures"]
                     if row["descriptor"]["name"] == "mx")
    assert authority["kind"] == "original reached scalar prefix definition"
    prefix = proof._scalar_prefix
    assert prefix.validate(builder, nodes, prefix.demanded) is prefix
    assert len(prefix.public()["assignments"]) == 2
    array_actions = {access.kind for effect in prefix._effects for access in effect.accesses
                     if access.storage.binding.rank}
    assert array_actions == {"descriptor"}
    assert prefix.public()["array_payload_definition_authorized"] is False
    assert prefix.public()["finite_or_constant_authorized"] is False
    # Initial field contents are unrelated to LBOUND/UBOUND preparation.
    builder.facts["captures"]["argument::a"]["initialized"] = "none"
    assert proof.validate() is proof


@pytest.mark.parametrize("prefix", [
    "",
    "if(nx>0) mx=nx",
    "if(nx>0)then\nmx=nx\nendif",
    "do ii=1,nx\nmx=nx\nenddo",
    "mx=undefined",
    "mx=nx\nmx=undefined",
    "mx=mx+1",
    "mx=nx\ncall reset(mx)",
    "mx=nx\ncall unknown()",
    "return\nmx=nx",
])
def test_missing_conditional_zero_trip_or_undefined_overwrite_never_defines(tmp_path, prefix):
    builder, nodes = source_builder(tmp_path, local_source(prefix))
    # Even a caller-authored local entry fact cannot repair absent source proof.
    builder.facts["captures"]["lines::advance::mx"] = dict(builder.facts["captures"]["argument::nx"])
    with pytest.raises(CompilationError):
        issue(builder, nodes)


def test_prefix_redefinition_uses_reached_current_value_without_range_inference(tmp_path):
    builder, nodes = source_builder(tmp_path, local_source("mx=nx\nmx=mx-1"))
    proof = issue(builder, nodes)
    assert len(proof._scalar_prefix.public()["assignments"]) == 2
    assert proof._scalar_prefix.public()["finite_or_constant_authorized"] is False


def test_local_definition_only_in_untaken_sibling_branch_rejects(tmp_path):
    text = local_source("if(nx>0)then\nmx=nx\nelse")
    text = text.replace("!$omp end parallel do", "!$omp end parallel do\nendif")
    builder, nodes = source_builder(tmp_path, text)
    with pytest.raises(CompilationError):
        issue(builder, nodes)


def test_local_predicate_requires_preceding_definition(tmp_path):
    builder, nodes = source_builder(tmp_path, local_source(guard="if(undefined>0)then"))
    with pytest.raises(CompilationError, match="before its original definition"):
        issue(builder, nodes)


def test_entry_predicate_requires_genuine_initialized_fact(tmp_path):
    builder, nodes = source_builder(tmp_path, local_source(guard="if(nx>0)then"))
    proof = issue(builder, nodes)
    builder.facts["captures"]["argument::nx"]["initialized"] = "none"
    with pytest.raises(CompilationError, match="whole invocation definition"):
        proof.validate()


@pytest.mark.parametrize("declaration", ["save mx", "equivalence(mx,undefined)", "common /bounds/ mx"])
def test_shared_or_associated_local_storage_never_uses_local_definition_rule(tmp_path, declaration):
    text = local_source().replace("integer::i,j,k,ii,jj,kk,mx,undefined",
                                  "integer::i,j,k,ii,jj,kk,mx,undefined\n" + declaration)
    builder, nodes = source_builder(tmp_path, text)
    with pytest.raises(CompilationError):
        issue(builder, nodes)


def test_original_prefix_identity_and_discarded_effect_proof_remain_authoritative(tmp_path):
    builder, nodes = source_builder(tmp_path, local_source(guard="if(nx>0)then"))
    proof = issue(builder, nodes)
    prefix = proof._scalar_prefix
    for forged in (copy(prefix), replace(prefix), replace(prefix, identity="0"*64),
                   replace(prefix, _prefix=()), replace(prefix, _controls=(nodes[0],))):
        with pytest.raises(CompilationError):
            forged.validate(builder, nodes, prefix.demanded)
    # Even replacing the registry cannot bless changed identity/source payload.
    for forged in (replace(prefix, identity="0"*64), replace(prefix, _prefix=()),
                   replace(prefix, _controls=(nodes[0],)), replace(prefix, _effects=()),
                   replace(prefix, _effects=tuple(reversed(prefix._effects)))):
        builder.analysis._reached_scalar_definition_proofs[forged.identity] = forged
        with pytest.raises(CompilationError):
            forged.validate(builder, nodes, prefix.demanded)
        builder.analysis._reached_scalar_definition_proofs[prefix.identity] = prefix
    builder.analysis._resource_reachability_proofs.clear()
    with pytest.raises(CompilationError, match="registered"):
        proof.validate()
    assert not builder.analysis._resource_reachability_proofs


def test_equal_spelling_foreign_selection_and_demand_reject(tmp_path):
    builder, nodes = source_builder(tmp_path, local_source())
    proof = issue(builder, nodes)._scalar_prefix
    with pytest.raises(CompilationError):
        proof.validate(builder, tuple(copy(node) for node in nodes), proof.demanded)
    with pytest.raises(CompilationError):
        prove_reached_scalar_definitions(builder, nodes, (copy(proof.demanded[0]),))


def test_equivalent_builder_reuses_source_token_but_changed_source_or_facts_reject(tmp_path):
    builder, nodes = source_builder(tmp_path, local_source(guard="if(nx>0)then"))
    proof = issue(builder, nodes)._scalar_prefix
    candidate = copy(builder)
    candidate.facts = deepcopy(builder.facts)
    assert prove_reached_scalar_definitions(candidate, nodes, proof.demanded) is proof
    candidate.facts["captures"]["argument::nx"]["initialized"] = "none"
    with pytest.raises(CompilationError, match="whole invocation definition"):
        proof.validate(candidate, nodes, proof.demanded)
    candidate.facts = deepcopy(builder.facts)
    candidate.analysis = copy(builder.analysis)
    with pytest.raises(CompilationError, match="registered original"):
        proof.validate(candidate, nodes, proof.demanded)


def test_source_task_cannot_substitute_duck_typed_prefix_authority(tmp_path):
    builder, nodes = source_builder(tmp_path, local_source())
    proof = issue(builder, nodes)
    forged = replace(proof, _scalar_prefix=SimpleNamespace(validate=lambda *args: None,
                                                          public=proof._scalar_prefix.public))
    builder.analysis._source_device_task_proofs[forged.identity] = forged
    with pytest.raises(CompilationError, match="original reached prefix authority"):
        forged.validate()


def test_source_file_mutation_rejects_publication_of_captured_scalar_proof(tmp_path):
    builder, nodes = source_builder(tmp_path, local_source())
    proof = issue(builder, nodes)
    path = tmp_path / "source.f90"
    path.write_text(path.read_text().replace("mx=ubound", "mx=1+ubound"))
    # ScopeBuilder owns immutable captured source; external edits are checked
    # at publication, as for existing source/session proof issuers.
    assert proof.validate() is proof
    with pytest.raises(CompilationError, match="configured source changed"):
        builder.analysis.verify_publication()


@pytest.mark.parametrize("axis", [1, 2, 3])
def test_actual_min_max_descriptor_prefix_and_original_predicate(tmp_path, axis):
    text = local_source("\n".join(f"min{d}=lbound(a,{i})+3\nmax{d}=ubound(a,{i})-3"
                                 for i, d in enumerate("ijk", 1)), axis=axis, guard="if(nx>0)then")
    text = text.replace("mx,undefined", "mx,undefined,mini,maxi,minj,maxj,mink,maxk")
    text = text.replace("shared(a,mask,nx,ny,nz,mx)", "shared(a,mask,nx,ny,nz,mini,maxi,minj,maxj,mink,maxk)")
    for d, n in zip("ijk", "xyz", strict=True):
        text = text.replace(f"do {d}=1,n{n}", f"do {d}=min{d},max{d}")
    text = text.replace("=1,mx", "=1,max" + "ijk"[axis-1])
    builder, nodes = source_builder(tmp_path, text)
    proof = issue(builder, nodes)
    assert len(proof._scalar_prefix.public()["assignments"]) == 6
    assert [item.name for item in proof._scalar_prefix.demanded] == ["max" + "ijk"[axis-1]]


def test_normal_builder_preserves_original_prefix_and_uses_reached_bound(tmp_path):
    builder, _nodes = source_builder(tmp_path, local_source(guard="if(nx>0)then"))
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report
    tasks = [task for scope in report["scopes"] for task in scope["source_device_tasks"]]
    assert len(tasks) == 1
    assert any(row["authority"]["kind"] == "original reached scalar prefix definition"
               for row in tasks[0]["scalar_captures"])
    rewritten = next(text for name, text in outputs.items()
                     if name.endswith(".f90") and "fort_source_task_" in text)
    assert rewritten.lower().index("mx=ubound(a,1)-3") < rewritten.index("fort_source_task_")


def test_checked_plan_borrows_exact_prefix_and_consumption_never_reissues(tmp_path, monkeypatch):
    from compiler.frontend import reached_scalar_definitions
    from compiler.scopes import execution_plan

    original = execution_plan._seal_execution
    sealed = []

    def seal(owner):
        proofs = {id(proof): proof for member in owner.members.values()
                  for proof in getattr(member.builder.analysis, "_source_device_task_proofs", {}).values()
                  if proof._scalar_prefix is not None}
        before = {key: proof._scalar_prefix for key, proof in proofs.items()}
        result = original(owner)
        for key, proof in proofs.items():
            assert proof._scalar_prefix is before[key]
            prefix = proof._scalar_prefix
            registry = prefix._analysis._reached_scalar_definition_proofs
            assert registry[prefix.identity] is prefix
            with monkeypatch.context() as patch:
                def forbidden(*args, **kwargs):
                    raise AssertionError("consumption must not issue another scalar proof")
                patch.setattr(reached_scalar_definitions, "prove_reached_scalar_definitions", forbidden)
                assert proof.validate() is proof
                del registry[prefix.identity]
                try:
                    with pytest.raises(CompilationError, match="registered"):
                        proof.validate()
                finally:
                    registry[prefix.identity] = prefix
            sealed.append(prefix.identity)
        return result

    monkeypatch.setattr(execution_plan, "_seal_execution", seal)
    builder, _nodes = source_builder(tmp_path, local_source(guard="if(nx>0)then"))
    _outputs, report = builder.run()
    assert report["scope_count"] == 1, report
    assert sealed


def branching_source():
    """The original three-axis shape with authentic reached local preparation."""
    groups = []
    for axis in (1, 2, 3):
        text = source(axis)
        first, last = text.index("!$omp parallel do"), text.index("!$omp end parallel do")
        group = text[first:last + len("!$omp end parallel do")]
        group = group.replace("shared(a,mask,nx,ny,nz)", "shared(a,mask,mini,maxi,minj,maxj,mink,maxk)")
        for d, n in zip("ijk", "xyz", strict=True):
            group = group.replace(f"do {d}=1,n{n}", f"do {d}=min{d},max{d}")
        d = "ijk"[axis-1]
        group = group.replace(f"do {d}=min{d},max{d}", f"do {d}=1,max{d}")
        groups.append(group)
    text = source().split("!$omp parallel do", 1)[0]
    text = text.replace("advance(a,mask,nx,ny,nz)", "advance(a,mask,nx,ny,nz,dir)")
    text = text.replace("integer,intent(in)::nx,ny,nz", "integer,intent(in)::nx,ny,nz,dir")
    text = text.replace("tmp(-2:ubound(a,1))", "tmp(-2:ubound(a,dir))")
    text += "integer::mini,maxi,minj,maxj,mink,maxk\n"
    text += "\n".join(f"min{d}=lbound(a,{i})+3\nmax{d}=ubound(a,{i})-3"
                      for i, d in enumerate("ijk", 1))
    return text + "\n\n! original blank/comment prefix remains unchanged\n\nif(dir==1)then\n" + groups[0] + \
        "\nelse if(dir==2)then\n" + groups[1] + "\nelse\n" + groups[2] + "\nendif\nend subroutine\nend module\n"


def branching_builder(tmp_path, text=None):
    from hashlib import sha256

    from compiler.driver.options import CompilerOptions
    from compiler.offload.config import OffloadConfig
    from compiler.scopes.source import ScopeBuilder
    from compiler.tests.test_graph_source_owner import FACT

    path = tmp_path / "branches.f90"
    path.write_text(branching_source() if text is None else text)
    return ScopeBuilder([path], "lines::advance", facts={"schema_version": 1, "participation": "serial",
        "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
        "captures": {"argument::"+n: dict(FACT) for n in ("a", "mask", "nx", "ny", "nz", "dir")}},
        options=CompilerOptions(), config=OffloadConfig(policy="sections", scope_execution="graph"))


def test_actual_if_elseif_else_prefix_history_is_three_checked_tasks_in_one_graph(tmp_path):
    from compiler.tests.test_line_snapshot import selections

    builder = branching_builder(tmp_path)
    nodes = selections(builder.entry)
    assert len(nodes) == 3
    for axis, group in enumerate(nodes, 1):
        proof = issue(builder, group)
        prefix = proof._scalar_prefix.public()
        assert prefix["version"] == 2
        assert len(prefix["effects_identities"]) == (7 if axis == 1 else 8)
        assert prefix["resources"][0]["name"] == "max"+"ijk"[axis-1]
        assert len(prefix["assignments"]) == 6
    _outputs, report = builder.run()
    assert report["scope_count"] == 1, report
    assert len(report["scopes"][0]["source_device_tasks"]) == 3
    assert not report["graph_owner_attempt"]["numerical_team_declines"]


@pytest.mark.parametrize("axis", [2, 3])
def test_skipped_earlier_sibling_body_never_defines_later_branch_bound(tmp_path, axis):
    from compiler.tests.test_line_snapshot import selections

    name = "max"+"ijk"[axis-1]
    assignment = f"{name}=ubound(a,{axis})-3"
    text = branching_source().replace(assignment, "")
    text = text.replace("if(dir==1)then", "if(dir==1)then\n"+assignment)
    builder = branching_builder(tmp_path, text)
    with pytest.raises(CompilationError, match="dominating original prefix definition"):
        issue(builder, selections(builder.entry)[axis-1])
