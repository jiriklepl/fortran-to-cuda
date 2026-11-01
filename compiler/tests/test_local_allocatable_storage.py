# The unsealed diagnostic route below retains its fresh-scan callback tests.
# Sealed production mutation/derivation/rollback/foreign-context tests are in
# test_source_authority_arena.py; no production callback is introduced for them.
"""Local storage evidence supplies neither reached allocation nor payload facts."""
from copy import copy
from dataclasses import replace

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.frontend.local_allocatable_storage import original_local_allocation
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

SOURCE = """module local_storage
implicit none
contains
subroutine producer(a,n)
real(8),intent(out)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
a(i)=real(i,8)
enddo
end subroutine
subroutine consumer(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
a(1:n)=2.0_8*a(1:n)
end subroutine
subroutine advance(n,steps)
integer,intent(in)::n,steps
real(8),save,allocatable::backing(:)
integer::stage
do stage=1,steps
if(.not.allocated(backing))then
allocate(backing(-2:n+1))
endif
call producer(backing,n)
call consumer(backing,n)
enddo
if(n<0.and.allocated(backing))deallocate(backing)
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE, *, sealed=False):
    path = tmp_path / "source.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine = analysis.routines["local_storage::advance"]
    return path, analysis, routine, routine.scope.bindings["backing"]


def issue(parts):
    _path, analysis, routine, binding = parts
    return original_local_allocation(analysis, routine, binding)


@pytest.mark.parametrize("sealed", [False, True])
def test_exact_local_save_allocation_has_no_reached_or_payload_authority(tmp_path, sealed):
    parts = fixture(tmp_path, sealed=sealed)
    path, analysis, routine, binding = parts
    original = path.read_bytes()
    proof = issue(parts)
    assert proof.validate(analysis, routine, binding) is proof
    assert issue(parts) is proof
    public = proof.public()
    assert public["resource"] == "local_storage::advance::backing"
    assert [call["procedure"] for call in public["direct_calls"]] == [
        "local_storage::producer", "local_storage::consumer"]
    assert public["selected_call_closure_requires_independent_descriptor_proof"] is True
    assert public["requires_original_allocation_presence_and_alias_guards"] is True
    assert public["requires_original_runtime_bounds"] is True
    for key in ("payload_definition_authority", "allocation_presence_authority", "selected_descriptor_stability_authority",
                "capture_authorized", "execution_authorized", "cross_frontier_retention_authorized"):
        assert public[key] is False
    # The zero-trip loop can leave backing unallocated. No proof executes the
    # original allocation or invents entry definitions/presence to admit it.
    assert "initialized" not in public
    assert analysis._allocation_authorizations == {}
    public["payload_definition_authority"] = True
    assert proof.public()["payload_definition_authority"] is False
    assert path.read_bytes() == original


def test_whole_prefix_assignment_never_becomes_current_payload_authority(tmp_path):
    source = SOURCE.replace("allocate(backing(-2:n+1))", "allocate(backing(-2:n+1))\nbacking=0.0_8")
    public = issue(fixture(tmp_path, source)).public()
    assert public["payload_definition_authority"] is False
    assert public["allocation_presence_authority"] is False
    assert public["selected_descriptor_stability_authority"] is False
    # The producer's INTENT(OUT) remains an original entry-definition kill.
    assert public["direct_calls"][0]["arguments"][0]["entry_definition_kill"] is True


@pytest.mark.parametrize("declaration", [
    "real(8),allocatable::backing(:)",
    "real(8),save,target,allocatable::backing(:)",
    "real(8),save,pointer::backing(:)",
    "real(8),save,volatile,allocatable::backing(:)",
    "real(8),save,asynchronous,allocatable::backing(:)",
    "real(8),save::backing(4)",
    "complex(8),save,allocatable::backing(:)",
])
def test_unproved_local_storage_controls_reject(tmp_path, declaration):
    parts = fixture(tmp_path, SOURCE.replace("real(8),save,allocatable::backing(:)", declaration))
    with pytest.raises(CompilationError, match="local SAVE ALLOCATABLE"):
        issue(parts)


def test_formal_or_module_storage_is_not_local_save_authority(tmp_path):
    path, analysis, routine, binding = fixture(tmp_path)
    with pytest.raises(CompilationError, match="local SAVE ALLOCATABLE"):
        original_local_allocation(analysis, routine, analysis.routines["local_storage::producer"].scope.bindings["a"])
    source = SOURCE.replace("contains", "real(8),allocatable::module_array(:)\ncontains", 1)
    path.write_text(source)
    analysis = SourceEffects([path])
    with pytest.raises(CompilationError, match="local SAVE ALLOCATABLE"):
        original_local_allocation(analysis, analysis.routines[routine.qualified],
                                  analysis.modules["local_storage"].bindings["module_array"])
    assert binding.root.endswith("::backing")


@pytest.mark.parametrize("specification", [
    "namelist /state/ backing", "common /state/ backing", "equivalence(backing,stage)",
])
def test_storage_association_and_namelist_specification_exposure_reject(tmp_path, specification):
    source = SOURCE.replace("integer::stage", "integer::stage\n" + specification)
    with pytest.raises(CompilationError, match="association, NAMELIST|original"):
        issue(fixture(tmp_path, source))


def test_unreached_contained_observer_rejects_host_association(tmp_path):
    source = SOURCE.replace("end subroutine\nend module", "contains\nsubroutine observe()\nprint *,backing\n"
                            "end subroutine\nend subroutine\nend module")
    with pytest.raises(CompilationError, match="contained host association"):
        issue(fixture(tmp_path, source))


@pytest.mark.parametrize("statement", [
    "call escape(backing)", "call move_alloc(backing,other)", "address=loc(backing)",
    "print *,backing", "associate(alias=>backing)\ncall consumer(alias,n)\nend associate",
])
def test_direct_unknown_address_or_alias_export_is_not_storage_evidence(tmp_path, statement):
    source = SOURCE.replace("integer::stage", "integer::stage,address\nreal(8),allocatable::other(:)")
    source = source.replace("call consumer(backing,n)", "call consumer(backing,n)\n" + statement)
    with pytest.raises(CompilationError):
        issue(fixture(tmp_path, source))


@pytest.mark.parametrize("attributes", ["target,", "pointer,", "allocatable,", "asynchronous,"])
def test_receiving_formal_must_not_expose_or_replace_original_allocation(tmp_path, attributes):
    source = SOURCE.replace("real(8),intent(inout)::a(:)", f"real(8),{attributes}intent(inout)::a(:)")
    with pytest.raises(CompilationError, match="nonescaping numeric receiving formal|association"):
        issue(fixture(tmp_path, source))


def test_same_array_cannot_be_associated_with_two_original_formals(tmp_path):
    source = SOURCE.replace("subroutine consumer(a,n)", "subroutine consumer(a,b,n)\nreal(8),intent(in)::b(:)")
    source = source.replace("call consumer(backing,n)", "call consumer(backing,backing,n)")
    with pytest.raises(CompilationError, match="distinct original storage associations"):
        issue(fixture(tmp_path, source))


def test_openmp_storage_and_concurrency_are_not_inferred_from_save(tmp_path):
    source = SOURCE.replace("integer::stage", "integer::stage\n!$omp threadprivate(backing)")
    with pytest.raises(CompilationError, match="OpenMP storage/concurrency"):
        issue(fixture(tmp_path, source))


def test_copy_foreign_projection_and_unregistered_token_cannot_authorize_storage(tmp_path):
    parts = fixture(tmp_path)
    _path, analysis, routine, binding = parts
    proof = issue(parts)
    for candidate in (copy(proof), replace(proof, binding=copy(binding))):
        with pytest.raises(CompilationError, match="exact compiler-issued"):
            candidate.validate(analysis, routine, binding)
    with pytest.raises(CompilationError, match="exact compiler-issued"):
        proof.validate(analysis.private_projection(), routine, binding)
    with pytest.raises(CompilationError, match="exact compiler-issued"):
        proof.validate(analysis, copy(routine), binding)
    del analysis._local_allocation_proofs[proof.identity]
    with pytest.raises(CompilationError, match="exact compiler-issued"):
        proof.validate(analysis, routine, binding)


@pytest.mark.parametrize("change", ["file", "syntax", "binding", "namespace", "registry", "module_role",
                                    "routine_role", "policy"])
def test_source_binding_and_namespace_changes_invalidate_exact_token(tmp_path, change):
    parts = fixture(tmp_path)
    path, analysis, routine, binding = parts
    proof = issue(parts)
    if change == "file":
        path.write_text(SOURCE + "!changed after issuance\n")
    elif change == "syntax":
        assignment = next(node for node in walk(routine.execution) if type(node).__name__ == "Allocate_Stmt")
        assignment.items = F.Allocate_Stmt("allocate(backing(1:4))").items
    elif change == "binding":
        binding.kind = 4
    elif change == "namespace":
        routine.scope.imports["producer"] = ("different_module", "producer")
    elif change == "registry":
        routine.scope.bindings["backing"] = copy(binding)
    elif change == "module_role":
        analysis.modules["local_storage"] = copy(analysis.modules["local_storage"])
    elif change == "routine_role":
        analysis.routines[routine.qualified] = copy(routine)
    else:
        analysis.operation_limit += 1
    with pytest.raises(CompilationError):
        proof.validate(analysis, routine, binding)


def contained_source(body="integer::unrelated\nunrelated=1", header="subroutine unused()"):
    return SOURCE.replace("end subroutine\nend module", "contains\n" + header + "\n" + body
                          + "\nend subroutine\nend subroutine\nend module")


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("shadow", [False, True])
def test_contained_original_nonuse_and_distinct_same_spelling_binding_are_private(tmp_path, sealed, shadow):
    source = (contained_source("real(8),intent(inout)::backing(:)\nbacking=0.0_8", "subroutine unused(backing)")
              if shadow else contained_source())
    _path, analysis, routine, binding = parts = fixture(tmp_path, source, sealed=sealed)
    proof = issue(parts)
    public = proof.public()
    assert public["version"] == 2
    assert len(public["contained_nonuse"]) == 1
    assert public["contained_nonuse"][0]["procedure"].endswith("::unused")
    assert public["joined_storage"] == []
    assert proof.validate(analysis, routine, binding) is proof
    assert not public["selected_descriptor_stability_authority"]
    assert not public["cross_frontier_retention_authorized"]


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("mode", ["unrelated", "shadow", "host_read", "specification"])
def test_internal_function_uses_its_authentic_function_namespace(tmp_path, sealed, mode):
    argument = "backing" if mode == "shadow" else "x"
    declaration = "real(8),intent(in)::" + argument + ("(:)" if mode == "shadow" else "")
    body = ("value=sum(backing)" if mode in {"shadow", "host_read"} else "value=1.0_8")
    if mode == "specification":
        declaration += "\nreal(8)::extent(size(backing))"
    source = SOURCE.replace("end subroutine\nend module",
        "contains\npure real(8) function unused(" + argument + ") result(value)\n" + declaration
        + "\n" + body + "\nend function\nend subroutine\nend module")
    parts = fixture(tmp_path, source, sealed=sealed)
    if mode in {"host_read", "specification"}:
        with pytest.raises(CompilationError, match="contained host association"):
            issue(parts)
    else:
        public = issue(parts).public()
        assert public["version"] == 2
        assert public["contained_nonuse"][0]["procedure"].endswith("::unused")


@pytest.mark.parametrize("body", [
    "print *,backing", "integer::extent(size(backing))", "call consumer(backing,1)",
    "if(.false.)then\nprint *,backing\nendif", "associate(alias=>backing)\nprint *,alias\nend associate",
])
def test_every_contained_header_specification_and_unreached_body_host_use_declines(tmp_path, body):
    with pytest.raises(CompilationError, match="contained host association"):
        issue(fixture(tmp_path, contained_source(body)))


def joined_source():
    return SOURCE.replace("integer::stage", "integer::stage,i").replace(
        "if(n<0.and.allocated(backing))deallocate(backing)",
        "!$omp parallel do private(i)\ndo i=1,n\nbacking(i)=backing(i)+1.0_8\n"
        "enddo\n!$omp end parallel do\nif(n<0.and.allocated(backing))deallocate(backing)")


def joined_teams(parts):
    from compiler.frontend.descriptor_teams import prove_descriptor_team
    from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence

    _path, analysis, routine, _binding = parts
    return tuple(prove_descriptor_team(analysis, routine.qualified, group.original_nodes)
                 for group in grouped_original_sequence(routine.execution.content, {})
                 if isinstance(group, NativeGroupCandidate))


@pytest.mark.parametrize("sealed", [False, True])
def test_only_exact_full_original_joined_receipts_admit_openmp_declaration_privacy(tmp_path, sealed):
    parts = fixture(tmp_path, joined_source(), sealed=sealed)
    _path, analysis, routine, binding = parts
    with pytest.raises(CompilationError, match="OpenMP storage/concurrency"):
        issue(parts)
    teams = joined_teams(parts)
    assert len(teams) == 1
    proof = original_local_allocation(analysis, routine, binding, joined_storage=teams)
    assert original_local_allocation(analysis, routine, binding, joined_storage=teams) is proof
    public = proof.public()
    assert public["version"] == 2
    assert public["joined_storage"] == [teams[0].public()]
    assert public["parent_join_is_not_execution_authority"]
    assert not public["payload_definition_authority"]
    assert not public["selected_descriptor_stability_authority"]
    assert not public["execution_authorized"]


@pytest.mark.parametrize("sealed", [False, True])
def test_unused_internal_function_and_joined_body_compose_without_lifetime_authority(tmp_path, sealed):
    text = joined_source().replace("end subroutine\nend module",
        "contains\npure real(8) function unused(x) result(value)\nreal(8),intent(in)::x\n"
        "value=x*x\nend function\nend subroutine\nend module")
    parts = fixture(tmp_path, text, sealed=sealed)
    _path, analysis, routine, binding = parts
    teams = joined_teams(parts)
    public = original_local_allocation(analysis, routine, binding, joined_storage=teams).public()
    assert len(public["contained_nonuse"]) == len(public["joined_storage"]) == 1
    assert public["version"] == 2
    assert public["selected_call_closure_requires_independent_descriptor_proof"]
    assert not public["selected_descriptor_stability_authority"]
    assert not public["cross_frontier_retention_authorized"]


@pytest.mark.parametrize("change", ["copy", "foreign", "duplicate", "list", "uncovered", "registry"])
def test_joined_receipt_copies_foreign_context_and_partial_coverage_decline(tmp_path, change):
    parts = fixture(tmp_path, joined_source())
    _path, analysis, routine, binding = parts
    teams = joined_teams(parts)
    if change == "copy":
        teams = (copy(teams[0]),)
    elif change == "foreign":
        other = tmp_path / "other"
        other.mkdir()
        teams = joined_teams(fixture(other, joined_source()))
    elif change == "duplicate":
        teams = teams * 2
    elif change == "list":
        teams = list(teams)
    elif change == "uncovered":
        parts[0].write_text(joined_source().replace("!$omp end parallel do", "!$omp end parallel do\n!$omp taskwait"))
    else:
        analysis._descriptor_team_proofs[teams[0].identity] = copy(teams[0])
    with pytest.raises(CompilationError):
        original_local_allocation(analysis, routine, binding, joined_storage=teams)


@pytest.mark.parametrize("sealed", [False])
@pytest.mark.parametrize("stage", ["resolution", "serializer", "publication"])
def test_initial_raw_declarations_cannot_be_reseeded_by_callbacks(tmp_path, monkeypatch, sealed, stage):
    import compiler.frontend.local_allocatable_storage as local
    from compiler.frontend.source_effects import Binding

    parts = fixture(tmp_path, contained_source(), sealed=sealed)
    _path, analysis, _routine, binding = parts
    fired = False

    def mutate():
        nonlocal fired
        if not fired:
            fired = True
            object.__setattr__(binding, "root", binding.root + "::forged")

    if stage == "resolution":
        original = local._validate_resolution

        def inspect(*args, **kwargs):
            result = original(*args, **kwargs)
            mutate()
            return result

        monkeypatch.setattr(local, "_validate_resolution", inspect)
    elif stage == "serializer":
        original = Binding.public

        def inspect(self):
            if self is binding:
                mutate()
            return original(self)

        monkeypatch.setattr(Binding, "public", inspect)
    else:
        original = type(analysis.inputs).verify_live

        def inspect(self):
            result = original(self)
            if self is analysis.inputs:
                mutate()
            return result

        monkeypatch.setattr(type(analysis.inputs), "verify_live", inspect)
    with pytest.raises(CompilationError):
        issue(parts)
    assert fired


@pytest.mark.parametrize("change", ["binding", "namespace", "registry", "inputs", "token", "team", "parser", "team_registry"])
def test_pure_final_checkpoint_retains_binding_parser_registry_and_joined_receipts(tmp_path, monkeypatch, change):
    from compiler.frontend.source_effects import Binding

    parts = fixture(tmp_path, joined_source())
    _path, analysis, routine, binding = parts
    teams = joined_teams(parts)
    proof = original_local_allocation(analysis, routine, binding, joined_storage=teams)
    saved = proof.checkpoint()

    def forbid(*args, **kwargs):
        raise AssertionError("pure checkpoint invoked a serializer or validator")

    monkeypatch.setattr(Binding, "public", forbid)
    monkeypatch.setattr(analysis, "verify_publication", forbid)
    proof.require_checkpoint(saved)
    if change == "binding":
        binding.root += "::changed"
    elif change == "namespace":
        routine.scope.bindings = dict(routine.scope.bindings)
    elif change == "registry":
        analysis._local_allocation_proofs[proof.identity] = copy(proof)
    elif change == "inputs":
        analysis.inputs.entries = dict(analysis.inputs.entries)
    elif change == "token":
        object.__setattr__(proof, "_joined_storage", ())
    elif change == "team":
        object.__setattr__(teams[0], "_originals", tuple(list(teams[0]._originals)))
    elif change == "parser":
        name = next(node for node in walk(routine.execution) if type(node).__name__ == "Name")
        object.__setattr__(name, "string", "different")
    else:
        analysis._descriptor_team_proofs = dict(analysis._descriptor_team_proofs)
    with pytest.raises(CompilationError, match="checkpoint"):
        proof.require_checkpoint(saved)


def test_full_validation_retains_original_joined_registry_not_a_later_equal_map(tmp_path):
    parts = fixture(tmp_path, joined_source())
    _path, analysis, routine, binding = parts
    teams = joined_teams(parts)
    proof = original_local_allocation(analysis, routine, binding, joined_storage=teams)
    analysis._descriptor_team_proofs = dict(analysis._descriptor_team_proofs)
    with pytest.raises(CompilationError, match="exact compiler-issued"):
        proof.validate(analysis, routine, binding)


@pytest.mark.parametrize("suffix", ["!$omp taskwait", "!$omp do\ndo i=1,n\nbacking(i)=0.0_8\nenddo\n!$omp end do nowait"])
def test_joined_region_cannot_discharge_a_later_original_orphan_or_task_directive(tmp_path, suffix):
    from compiler.frontend.descriptor_teams import prove_descriptor_team
    from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence

    source = joined_source().replace("!$omp end parallel do", "!$omp end parallel do\n" + suffix)
    _path, analysis, routine, binding = fixture(tmp_path, source)
    groups = [group for group in grouped_original_sequence(routine.execution.content, {})
              if isinstance(group, NativeGroupCandidate)
              and "parallel" in str(group.original_nodes[0]).lower()]
    assert len(groups) == 1
    team = prove_descriptor_team(analysis, routine.qualified, groups[0].original_nodes)
    with pytest.raises(CompilationError, match="OpenMP storage/concurrency"):
        original_local_allocation(analysis, routine, binding, joined_storage=(team,))


@pytest.mark.parametrize("sealed", [False])
def test_composed_proof_callbacks_cannot_change_an_earlier_allocation_source(tmp_path, monkeypatch, sealed):
    from compiler.frontend.local_allocatable_storage import (
        allocation_source_checkpoint,
        require_allocation_source_checkpoint,
    )
    from compiler.frontend.source_effects import Binding

    _path, analysis, _routine, binding = parts = fixture(tmp_path, contained_source(), sealed=sealed)
    proof = issue(parts)
    saved = allocation_source_checkpoint(analysis)
    original = Binding.public
    fired = False

    def inspect(self):
        nonlocal fired
        result = original(self)
        if self is binding and not fired:
            fired = True
            object.__setattr__(binding, "root", binding.root + "::late")
        return result

    monkeypatch.setattr(Binding, "public", inspect)
    # A later independent serializer is sufficient; final composition must
    # still fence source consumed only by an earlier proof.
    binding.public()
    assert fired
    with pytest.raises(CompilationError, match="participating source"):
        require_allocation_source_checkpoint(analysis, saved)
    assert proof.public()["version"] == 2


@pytest.mark.parametrize("change", ["projected_base", "projected_overrides", "session_identity", "session_roots",
                                    "session_object_entry", "root_registry"])
def test_immutable_source_and_projection_anchors_reject_supported_mutations(tmp_path, change):
    from dataclasses import FrozenInstanceError

    from compiler.frontend.local_allocatable_storage import (
        allocation_source_checkpoint,
        require_allocation_source_checkpoint,
    )

    _path, original, routine, _binding = fixture(tmp_path, sealed=True)
    analysis = original.private_projection(routines=True)
    saved = allocation_source_checkpoint(analysis)
    if change == "projected_base":
        with pytest.raises(CompilationError, match="identity is immutable"):
            analysis.routines.base = dict(analysis.routines.base)
    elif change == "projected_overrides":
        with pytest.raises(CompilationError, match="identity is immutable"):
            analysis.routines._overrides = dict(analysis.routines.overrides)
    elif change == "session_identity":
        with pytest.raises(FrozenInstanceError):
            analysis._source_session.identity = "different"
    elif change == "session_roots":
        with pytest.raises(TypeError):
            analysis._source_session.roots["routines"] = dict(analysis.routines)
    elif change == "session_object_entry":
        with pytest.raises(TypeError):
            analysis._source_session.objects[id(routine.scope)] = copy(routine.scope)
    else:
        with pytest.raises(CompilationError, match="source registry is immutable"):
            analysis.modules = dict(analysis.modules)
    require_allocation_source_checkpoint(analysis, saved)


@pytest.mark.parametrize("sealed", [False, True])
def test_cached_same_public_foreign_analysis_token_never_satisfies_current_query(tmp_path, sealed):
    first = fixture(tmp_path, contained_source(), sealed=sealed)
    foreign = issue(first)
    second = fixture(tmp_path, contained_source(), sealed=sealed)
    current = issue(second)
    assert foreign.identity == current.identity
    second[1]._local_allocation_proofs[current.identity] = foreign
    with pytest.raises(CompilationError, match="another original query|stale or foreign authority context"):
        issue(second)


def test_every_original_team_is_required_even_when_the_allocation_use_is_elsewhere(tmp_path):
    text = joined_source()
    group = "!$omp parallel do private(i)\ndo i=1,n\nbacking(i)=backing(i)+1.0_8\nenddo\n!$omp end parallel do"
    parts = fixture(tmp_path, text.replace(group, group + "\n" + group))
    _path, analysis, routine, binding = parts
    teams = joined_teams(parts)
    assert len(teams) == 2
    with pytest.raises(CompilationError, match="OpenMP storage/concurrency"):
        original_local_allocation(analysis, routine, binding, joined_storage=teams[:1])
    public = original_local_allocation(analysis, routine, binding, joined_storage=teams).public()
    assert len(public["joined_storage"]) == 2


@pytest.mark.parametrize("existing", [False, True])
def test_joined_validator_cannot_replace_an_earlier_consumed_team_or_cached_local_receipt(tmp_path, monkeypatch, existing):
    from compiler.frontend.descriptor_teams import DescriptorTeamProof

    parts = fixture(tmp_path, joined_source())
    _path, analysis, routine, binding = parts
    teams = joined_teams(parts)
    proof = original_local_allocation(analysis, routine, binding, joined_storage=teams) if existing else None
    original = DescriptorTeamProof.validate
    fired = False

    def inspect(self, *args, **kwargs):
        nonlocal fired
        result = original(self, *args, **kwargs)
        if self is teams[0] and not fired:
            fired = True
            if proof is None:
                object.__setattr__(self, "_originals", tuple(list(self._originals)))
            else:
                object.__setattr__(proof, "_routine", copy(routine))
        return result

    monkeypatch.setattr(DescriptorTeamProof, "validate", inspect)
    with pytest.raises(CompilationError):
        original_local_allocation(analysis, routine, binding, joined_storage=teams)
    assert fired


@pytest.mark.parametrize("sealed", [False, True])
def test_issuer_uses_one_fresh_derivation_but_standalone_validation_still_rescans(tmp_path, monkeypatch, sealed):
    from compiler.frontend import local_allocatable_storage as local

    parts = fixture(tmp_path, sealed=sealed)
    derive = local._derive
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs.get("_return_fence", False))
        return derive(*args, **kwargs)

    monkeypatch.setattr(local, "_derive", counted)
    proof = issue(parts)
    before = proof.public()
    assert calls == ([False] if sealed else [True])
    assert issue(parts) is proof
    assert calls == ([False] if sealed else [True, True])
    assert proof.validate(parts[1], parts[2], parts[3]) is proof
    assert calls == ([False] if sealed else [True, True, False])
    assert proof.public() == before


@pytest.mark.parametrize("sealed", [False])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("mutation", ["binding", "parser_axis", "registry", "input_metadata", "live_file",
                                     "depth_limit", "procedure_limit", "operation_limit", "descriptor_source"])
def test_publication_after_the_fresh_scan_still_fences_exact_source(tmp_path, monkeypatch, sealed, existing, mutation):
    from compiler.frontend import local_allocatable_storage as local

    parts = fixture(tmp_path, sealed=sealed)
    _path, analysis, _routine, binding = parts
    if existing:
        issue(parts)
    derive = local._derive
    scope_roles = local._scope_roles
    validate_roles = local._validate_scope_roles
    ready = fired = False

    def derived(*args, **kwargs):
        nonlocal ready
        result = derive(*args, **kwargs)
        if kwargs.get("_return_fence"):
            ready = True
        return result

    def mutate():
        nonlocal fired
        if not ready or fired:
            return
        fired = True
        if mutation == "binding":
            object.__setattr__(binding, "root", "forged::backing")
        elif mutation == "parser_axis":
            object.__setattr__(binding, "shape_nodes", tuple(list(binding.shape_nodes)))
        elif mutation == "registry":
            analysis._local_allocation_proofs = dict(getattr(analysis, "_local_allocation_proofs", {}))
        elif mutation == "input_metadata":
            object.__setattr__(analysis.inputs, "configuration", {"forged": True})
        elif mutation == "live_file":
            _path.write_text(_path.read_text() + "! late original source change\n")
        elif mutation == "descriptor_source":
            analysis._descriptor_source = object()
        else:
            setattr(analysis, mutation, getattr(analysis, mutation) - 1)

    def checked(*args, **kwargs):
        result = validate_roles(*args, **kwargs)
        mutate()
        return result

    def roles(*args, **kwargs):
        result = scope_roles(*args, **kwargs)
        mutate()
        return result

    monkeypatch.setattr(local, "_derive", derived)
    monkeypatch.setattr(local, "_validate_scope_roles", checked)
    monkeypatch.setattr(local, "_scope_roles", roles)
    with pytest.raises(CompilationError):
        issue(parts)
    assert fired


@pytest.mark.parametrize("sealed", [False])
@pytest.mark.parametrize("mutation", ["owner", "record", "roles", "role_entry", "policy", "issuer"])
def test_issuer_baselines_retained_cached_attachments_before_any_fresh_callbacks(tmp_path, monkeypatch, sealed, mutation):
    from compiler.frontend import local_allocatable_storage as local

    parts = fixture(tmp_path, sealed=sealed)
    proof = issue(parts)
    derive = local._derive
    fired = False

    def changed(*args, **kwargs):
        nonlocal fired
        if not fired:
            fired = True
            if mutation == "owner":
                object.__setattr__(proof, "_routine", copy(proof._routine))
            elif mutation == "record":
                object.__setattr__(proof, "_record", "{}")
            elif mutation == "roles":
                object.__setattr__(proof, "_roles", copy(proof._roles))
            elif mutation == "role_entry":
                roles = proof._roles
                if isinstance(roles, dict):
                    key = next(iter(roles))
                    roles[key] = copy(roles[key])
                else:
                    object.__setattr__(roles, "overrides", tuple(list(roles.overrides)) + (("foreign", parts[2].scope),))
            elif mutation == "policy":
                object.__setattr__(proof, "_limits", (1, 1, 1))
            else:
                object.__setattr__(proof, "_issuer", object())
        return derive(*args, **kwargs)

    monkeypatch.setattr(local, "_derive", changed)
    with pytest.raises(CompilationError):
        issue(parts)
    assert fired


@pytest.mark.parametrize("sealed", [False])
@pytest.mark.parametrize("mutation", ["forged_return", "late_role_change"])
def test_new_proof_authenticates_and_retains_fresh_roles_after_derivation(tmp_path, monkeypatch, sealed, mutation):
    from compiler.frontend import local_allocatable_storage as local

    parts = fixture(tmp_path, sealed=sealed)
    derive, scope_roles, validate_roles = local._derive, local._scope_roles, local._validate_scope_roles
    ready = fired = False

    def change(roles):
        if isinstance(roles, dict):
            key = next(iter(roles))
            roles[key] = copy(roles[key])
        else:
            object.__setattr__(roles, "session", copy(roles.session))

    def derived(*args, **kwargs):
        nonlocal ready
        result = derive(*args, **kwargs)
        if kwargs.get("_return_fence"):
            ready = True
        return result

    def roles(*args, **kwargs):
        nonlocal fired
        result = scope_roles(*args, **kwargs)
        if ready and not fired and mutation == "forged_return":
            fired = True
            change(result)
        return result

    def checked(analysis, roles, consulted):
        nonlocal fired
        result = validate_roles(analysis, roles, consulted)
        if ready and not fired and mutation == "late_role_change":
            fired = True
            change(roles)
        return result

    monkeypatch.setattr(local, "_derive", derived)
    monkeypatch.setattr(local, "_scope_roles", roles)
    monkeypatch.setattr(local, "_validate_scope_roles", checked)
    with pytest.raises(CompilationError):
        issue(parts)
    assert fired


@pytest.mark.parametrize("sealed", [False])
@pytest.mark.parametrize("mutation", ["binding", "record"])
def test_final_live_check_cannot_replace_the_new_unpublished_proof(tmp_path, monkeypatch, sealed, mutation):
    from compiler.frontend import local_allocatable_storage as local

    parts = fixture(tmp_path, sealed=sealed)
    _path, analysis, _routine, _binding = parts
    initialize = local.OriginalLocalAllocationProof.__init__
    verify = analysis.verify_source
    retained = []
    fired = False

    def initialize_and_retain(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        retained.append(self)

    def changed():
        nonlocal fired
        result = verify()
        if retained and not fired:
            fired = True
            proof = retained[-1]
            if mutation == "binding":
                object.__setattr__(proof, "binding", copy(proof.binding))
            else:
                object.__setattr__(proof, "_record", "{}")
        return result

    monkeypatch.setattr(local.OriginalLocalAllocationProof, "__init__", initialize_and_retain)
    monkeypatch.setattr(analysis, "verify_source", changed)
    with pytest.raises(CompilationError):
        issue(parts)
    assert fired


@pytest.mark.parametrize("sealed", [False, True])
def test_registry_subclass_cannot_execute_a_publication_callback_after_the_final_fence(tmp_path, sealed):
    parts = fixture(tmp_path, sealed=sealed)
    path, analysis, _routine, _binding = parts
    before = path.read_bytes()
    calls = []

    class CallbackRegistry(dict):
        def __setitem__(self, key, value):
            calls.append("after final fence")
            path.write_text(path.read_text() + "! registry callback changed live source\n")
            dict.__setitem__(self, key, value)

    foreign = CallbackRegistry()
    analysis._local_allocation_proofs = foreign
    if sealed:
        proof = issue(parts)
        assert proof.validate(analysis, parts[2], parts[3]) is proof
        assert type(analysis._local_allocation_proofs) is dict
        assert analysis._local_allocation_proofs is not foreign
    else:
        with pytest.raises(CompilationError):
            issue(parts)
    assert not calls
    assert path.read_bytes() == before


@pytest.mark.parametrize("sealed", [False])
def test_roles_subclass_cannot_execute_after_the_final_live_source_check(tmp_path, monkeypatch, sealed):
    from compiler.frontend import local_allocatable_storage as local

    parts = fixture(tmp_path, sealed=sealed)
    path, analysis, _routine, _binding = parts
    before = path.read_bytes()
    derive, scope_roles = local._derive, local._scope_roles
    verify_live = type(analysis.inputs).verify_live
    ready = metadata = False
    armed = [False]
    calls = []

    def callback():
        if armed[0] and not calls:
            calls.append("after final live source check")
            path.write_text(path.read_text() + "! role callback changed live source\n")

    class CallbackRoles(dict):
        def __len__(self):
            callback()
            return dict.__len__(self)

    class CallbackSessionRoles(local._SessionScopeRoles):
        def __getattribute__(self, name):
            if name in {"session", "overrides"}:
                callback()
            return object.__getattribute__(self, name)

    def derived(*args, **kwargs):
        nonlocal ready
        result = derive(*args, **kwargs)
        if kwargs.get("_return_fence"):
            ready = True
        return result

    def roles(*args, **kwargs):
        nonlocal metadata
        result = scope_roles(*args, **kwargs)
        if ready:
            metadata = True
            return (CallbackRoles(result) if isinstance(result, dict)
                    else CallbackSessionRoles(result.session, result.overrides))
        return result

    def verified(self, *args, **kwargs):
        result = verify_live(self, *args, **kwargs)
        if self is analysis.inputs and metadata:
            armed[0] = True
        return result

    monkeypatch.setattr(local, "_derive", derived)
    monkeypatch.setattr(local, "_scope_roles", roles)
    monkeypatch.setattr(type(analysis.inputs), "verify_live", verified)
    with pytest.raises(CompilationError):
        issue(parts)
    assert metadata
    assert not calls
    assert path.read_bytes() == before
