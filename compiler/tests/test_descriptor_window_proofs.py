"""Original interval storage authority excludes tails, never seeds freshness."""

from copy import copy
from dataclasses import replace

import pytest

from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.native_prefix import prove_native_prefix
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

ENTRY = "intervals::advance"
ROOT = "argument::a"


def fixture(tmp_path, *, selected="call read_modify(a)", tail="call move_alloc(a,temporary)",
            prefix="counter=2", child="a(1)=a(1)+1.0_8", child_decl="real(8),intent(inout)::a(:)",
            extra="", owner_attrs="allocatable", operations=256, procedures=64):
    path = tmp_path / "intervals.f90"
    path.write_text(f"""module intervals
contains
subroutine read_modify(a)
{child_decl}
{child}
end subroutine
{extra}
subroutine advance(a,flag)
real(8),{owner_attrs},intent(inout)::a(:)
logical,intent(in)::flag
integer::counter,stage
real(8)::total
real(8),allocatable::temporary(:)
{prefix}
do stage=1,3
{selected}
{tail}
end do
end subroutine
end module
""")
    analysis = SourceEffects([path], operations=operations, procedures=procedures)
    graph = analysis.structure(ENTRY)
    routine = analysis.routines[ENTRY]
    loop = next(item for item in routine.execution.content if type(item).__name__ == "Block_Nonlabel_Do_Construct")
    source = tuple(item for item in loop.content[1:-1] if type(item).__name__ != "Comment")
    selected_count = len(selected.splitlines())
    # Tests with IF/DO selections choose their complete original construct.
    nodes = source[:1] if selected.lstrip().lower().startswith(("if", "do")) else source[:selected_count]
    identities = tuple(graph.node_id(node) for node in nodes)
    return path, analysis, graph, identities


def prove(analysis, selected, roots=(ROOT,)):
    return prove_descriptor_window(analysis, ENTRY, selected, roots)


def test_nested_interval_is_stable_despite_original_later_move_alloc(tmp_path):
    path, analysis, graph, selected = fixture(tmp_path)
    before = path.read_bytes()
    assert not analysis.descriptor_stability(ENTRY)["complete"]
    proof = prove(analysis, selected)
    assert prove(analysis, selected) is proof
    assert proof.validate(analysis, ENTRY, selected) is proof
    public = proof.public()
    assert public["selected_node_ids"] == list(selected)
    assert public["structured_identity"] == graph.identity
    assert public["analysis_identity"] == graph.authority_identity
    state = proof.resource_state(ROOT)
    assert state["stable"]
    assert state["original_descriptor_required"]
    assert state["requires_prior_noescape_and_current_allocation_proof"]
    assert state["requires_original_allocation_presence_and_alias_guards"]
    assert state["execution_requires_runtime_guard"]
    assert not state["descriptor_changes_within_selection"]
    assert not state["entry_definition_authority"]
    assert not public["prefix_effects_proved"]
    assert not public["reachability_authority"]
    assert not public["execution_authorized"]
    assert not public["gpu_legality_established"]
    assert not public["freshness_by_address_or_shape"]
    assert public["requires_close_before_excluded_tail"]
    assert path.read_bytes() == before


def test_unknown_prefix_is_excluded_but_never_claimed_harmless(tmp_path):
    _path, analysis, _graph, selected = fixture(tmp_path, prefix="call unknown_prefix(a)")
    public = prove(analysis, selected).public()
    assert not public["prefix_effects_proved"]
    assert public["resources"][0]["requires_prior_noescape_and_current_allocation_proof"]


@pytest.mark.parametrize("tail", ["deallocate(a)\nallocate(a(3))", "call unknown_tail(a)", "a=[1.0_8,2.0_8]"])
def test_excluded_tail_does_not_supply_or_invalidate_interval_authority(tmp_path, tail):
    _path, analysis, _graph, selected = fixture(tmp_path, tail=tail)
    assert prove(analysis, selected).resource_state(ROOT)["stable"]
    assert not analysis.descriptor_stability(ENTRY)["complete"]


@pytest.mark.parametrize("selected", ["allocate(a(3))", "deallocate(a)", "call move_alloc(a,temporary)",
                                      "a=3.0_8", "call unknown(a)", "call unknown()"])
def test_selected_lifetime_or_unknown_effects_reject(tmp_path, selected):
    _path, analysis, _graph, ids = fixture(tmp_path, selected=selected)
    with pytest.raises(CompilationError, match="unknown selected storage|whole allocatable"):
        prove(analysis, ids)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


@pytest.mark.parametrize("child", ["a=3.0_8", "deallocate(a)", "call move_alloc(a,b)", "call absent(a)"])
def test_direct_child_complete_source_is_required(tmp_path, child):
    declaration = "real(8),allocatable,intent(inout)::a(:)\nreal(8),allocatable::b(:)"
    _path, analysis, _graph, ids = fixture(tmp_path, child=child, child_decl=declaration)
    with pytest.raises(CompilationError, match="unknown selected storage|whole allocatable"):
        prove(analysis, ids)


def test_nonallocatable_out_payload_event_is_not_a_descriptor_change(tmp_path):
    _path, analysis, _graph, ids = fixture(tmp_path, child="a=3.0_8", child_decl="real(8),intent(out)::a(:)")
    public = prove(analysis, ids).public()
    assert not public["entry_definition_authority"]
    assert public["resources"][0]["stable"]


def test_allocatable_out_entry_rejects_even_without_body_access(tmp_path):
    _path, analysis, _graph, ids = fixture(tmp_path, child="continue", child_decl="real(8),allocatable,intent(out)::a(:)")
    with pytest.raises(CompilationError, match="allocatable OUT"):
        prove(analysis, ids)


def test_source_mappings_keep_original_keywords_and_numerical_payload_changes(tmp_path):
    _path, analysis, _graph, ids = fixture(tmp_path, selected="call read_modify(a=a)",
                                         child_decl="real(8),allocatable,intent(inout)::a(:)")
    assert prove(analysis, ids).public()["source_call_closure"]


def test_ordinary_loop_and_branch_are_checked_on_all_source_paths(tmp_path):
    _path, analysis, _graph, ids = fixture(tmp_path, selected="if(flag) then\na(1)=2.0_8\nelse\na(2)=3.0_8\nend if")
    assert prove(analysis, ids).resource_state(ROOT)["stable"]
    _path, analysis, _graph, ids = fixture(tmp_path, selected="if(flag) then\na(1)=2.0_8\nelse\ndeallocate(a)\nend if")
    with pytest.raises(CompilationError, match="unknown selected storage"):
        prove(analysis, ids)


def test_original_readonly_intrinsics_are_not_descriptor_writes(tmp_path):
    _path, analysis, _graph, ids = fixture(tmp_path, selected="total=sum(a)+real(size(a),8)")
    assert prove(analysis, ids).resource_state(ROOT)["stable"]


@pytest.mark.parametrize("body", ["call read_modify(a)", "real(8)::local(unknown_bound())\na(1)=2.0_8"])
def test_recursion_and_automatic_bound_effects_are_not_hidden(tmp_path, body):
    child, declaration = body, "real(8),intent(inout)::a(:)"
    if body.startswith("real"):
        declaration += "\n" + body.splitlines()[0]
        child = body.splitlines()[1]
    _path, analysis, _graph, ids = fixture(tmp_path, child=child, child_decl=declaration)
    with pytest.raises(CompilationError, match="recursive|unknown function/escape"):
        prove(analysis, ids)


def test_source_function_requires_registered_original_readonly_token(tmp_path):
    extra = """logical pure function check(x) result(ok)
real(8),intent(in)::x
ok=x>0.0_8
end function
"""
    _path, analysis, _graph, ids = fixture(tmp_path, selected="total=merge(1.0_8,0.0_8,check(total))", extra=extra)
    with pytest.raises(CompilationError, match="registered original readonly source-function"):
        prove(analysis, ids)
    # The ordinary source-effect provider mints the original scalar helper
    # proof. The descriptor query never calls whole descriptor_stability.
    analysis.summarize(ENTRY)
    assert analysis._native_functions
    original = analysis.descriptor_stability
    analysis.descriptor_stability = lambda *_args: pytest.fail("whole procedure descriptor recursion")
    try:
        proof = prove(analysis, ids)
        assert proof.public()["native_source_function_requirements"]
    finally:
        analysis.descriptor_stability = original


def test_interval_authority_is_registered_exact_and_source_bound(tmp_path):
    path, analysis, graph, ids = fixture(tmp_path)
    proof = prove(analysis, ids)
    for forged in (copy(proof), replace(proof)):
        with pytest.raises(CompilationError, match="registered original interval"):
            forged.validate(analysis, ENTRY, ids)
    with pytest.raises(CompilationError, match="registered original interval"):
        proof.validate(SourceEffects([path]), ENTRY, ids)
    with pytest.raises(CompilationError, match="resource was not proved"):
        proof.resource_state("argument::other")
    public = proof.public()
    public["resources"][0]["stable"] = False
    assert proof.resource_state(ROOT)["stable"]
    for wrong in ((graph.root_id,), (graph.entry_id,), (ids[0], ids[0]), ("forged",)):
        with pytest.raises(CompilationError, match="sequence|repeated|authority"):
            prove(analysis, wrong)
    path.write_text(path.read_text().replace("counter=2", "counter=3"))
    with pytest.raises(CompilationError, match="source changed"):
        proof.resource_state(ROOT)


def test_selection_cannot_skip_a_native_operation_in_its_immediate_sequence(tmp_path):
    _path, analysis, graph, ids = fixture(tmp_path, selected="call read_modify(a)\ncounter=3\ncall read_modify(a)")
    assert len(prove(analysis, ids).public()["selected_node_ids"]) == 3
    for wrong in ((ids[0], ids[2]), tuple(reversed(ids)), (graph.node_id(analysis.routines[ENTRY].execution.content[0]), ids[0])):
        with pytest.raises(CompilationError, match="sequence|source order|guarded segment"):
            prove(analysis, wrong)


@pytest.mark.parametrize("attributes", ["pointer", "allocatable,optional", "allocatable,volatile", "target"])
def test_only_original_numeric_allocatable_formals_are_initially_supported(tmp_path, attributes):
    _path, analysis, _graph, ids = fixture(tmp_path, owner_attrs=attributes)
    with pytest.raises(CompilationError, match="numeric allocatable formals"):
        prove(analysis, ids)


def test_small_budgets_decline_without_registering_authority(tmp_path):
    _path, analysis, _graph, ids = fixture(tmp_path, procedures=1)
    with pytest.raises(CompilationError, match="closure budget"):
        prove(analysis, ids)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


def test_counted_header_alone_has_no_body_or_execution_authority(tmp_path):
    _path, analysis, graph, _ids = fixture(tmp_path)
    loop = next(node for node in analysis.routines[ENTRY].execution.content
                if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    header = graph.node_id(loop.content[0], role="header")
    public = prove(analysis, (header,)).public()
    assert public["selection_role"] == "original counted header"
    assert public["parent_node"] == graph.node_id(loop)
    assert public["parent_sequence"] is None
    assert not public["execution_authorized"]
    assert not public["reachability_authority"]
    assert not public["source_call_closure"]


@pytest.mark.parametrize("before_issue", [False, True])
def test_resolution_mutation_cannot_mint_or_revalidate_interval_authority(tmp_path, before_issue):
    _path, analysis, _graph, ids = fixture(tmp_path)
    proof = None if before_issue else prove(analysis, ids)
    analysis.routines[ENTRY].scope.imports["read_modify"] = ("foreign", "overwrite")
    with pytest.raises(CompilationError, match="original.*authority"):
        prove(analysis, ids) if proof is None else proof.resource_state(ROOT)


def test_transitive_original_child_changes_invalidate_registered_proof(tmp_path):
    _path, analysis, _graph, ids = fixture(tmp_path)
    proof = prove(analysis, ids)
    analysis.routines["intervals::read_modify"].scope.imports["hidden"] = ("foreign", "state")
    with pytest.raises(CompilationError, match="original.*authority"):
        proof.resource_state(ROOT)


@pytest.mark.parametrize("child", ["!$omp parallel\na(1)=3.0_8\n!$omp end parallel",
                                    "q=>a", "a(1)=real(loc(a),8)"])
def test_team_pointer_or_address_escape_requires_distinct_authority(tmp_path, child):
    declaration = "real(8),target,intent(inout)::a(:)"
    if child == "q=>a":
        declaration += "\nreal(8),pointer::q(:)"
    _path, analysis, _graph, ids = fixture(tmp_path, child=child, child_decl=declaration)
    with pytest.raises(CompilationError, match="OpenMP storage|specification effects|unknown function/escape"):
        prove(analysis, ids)


def test_descriptor_query_does_not_call_or_mutate_payload_summary_cache(tmp_path):
    _path, analysis, _graph, ids = fixture(tmp_path)
    before = dict(analysis.summaries)
    analysis.summarize = lambda *_args, **_kwargs: pytest.fail("payload summary requested")
    assert prove(analysis, ids).resource_state(ROOT)["stable"]
    assert analysis.summaries == before


TEAM_DECLARATION = """real(8),intent(inout)::a(:)
integer::i
real(8)::p,total
logical::bad"""
TEAM_BODY = """total=0.0_8
bad=.false.
!$omp parallel do private(i,p) reduction(max:total) reduction(.or.:bad)
do i=1,size(a)
p=a(i)
total=max(total,p)
if(p/=p) bad=.true.
end do
!$omp end parallel do
a(1)=total"""


def test_registered_native_prefix_only_supplies_child_directive_authority(tmp_path):
    _path, analysis, _graph, ids = fixture(tmp_path, child=TEAM_BODY, child_decl=TEAM_DECLARATION)
    with pytest.raises(CompilationError, match="registered whole-closure"):
        prove(analysis, ids)
    assert not getattr(analysis, "_native_prefix_proofs", {})
    native = prove_native_prefix(analysis, "intervals::read_modify")
    # The native proof is already complete. The descriptor query neither
    # rebuilds payload effects nor inherits native execution authority.
    analysis.summarize = lambda *_args, **_kwargs: pytest.fail("payload summary requested")
    public = prove(analysis, ids).public()
    requirement, = public["native_prefix_descriptor_requirements"]
    assert requirement["proof_identity"] == native.identity
    assert requirement["whole_original_closure_scanned"]
    assert requirement["descriptor_evidence_only"]
    assert requirement["native_prefix_execution_requirements"]["requires_no_active_context"]
    assert not requirement["execution_authorized"]
    assert not requirement["allocation_freshness_authority"]
    assert not public["prefix_effects_proved"]
    assert not public["entry_definition_authority"]
    assert public["source_operations"] > 5


def test_registered_whole_native_prefix_covers_transitive_directives_only(tmp_path):
    extra = "subroutine team_reduce(a)\n" + TEAM_DECLARATION + "\n" + TEAM_BODY + "\nend subroutine"
    _path, analysis, _graph, ids = fixture(tmp_path, child="call team_reduce(a)", extra=extra)
    native = prove_native_prefix(analysis, "intervals::read_modify")
    public = prove(analysis, ids).public()
    requirement, = public["native_prefix_descriptor_requirements"]
    assert requirement["proof_identity"] == native.identity
    assert set(requirement["covered_original_procedures"]) == {"intervals::read_modify", "intervals::team_reduce"}
    assert set(public["source_call_closure"]) == {"intervals::read_modify", "intervals::team_reduce"}
    analysis._native_prefix_proofs.pop(native.identity)
    with pytest.raises(CompilationError, match="registered whole-closure"):
        prove(analysis, ids)


def test_copied_native_prefix_cannot_supply_directive_authority(tmp_path):
    _path, analysis, _graph, ids = fixture(tmp_path, child=TEAM_BODY, child_decl=TEAM_DECLARATION)
    native = prove_native_prefix(analysis, "intervals::read_modify")
    analysis._native_prefix_proofs["copy"] = copy(native)
    with pytest.raises(CompilationError, match="unique registered"):
        prove(analysis, ids)


def test_native_prefix_does_not_suppress_descriptor_specific_replacement_checks(tmp_path):
    declaration = TEAM_DECLARATION.replace("real(8),intent(inout)::a", "real(8),allocatable,intent(inout)::a")
    _path, analysis, _graph, ids = fixture(tmp_path, child=TEAM_BODY + "\na=3.0_8", child_decl=declaration)
    # SourceEffects already rejects this complete native proof. The interval
    # provider independently rejects the allocatable whole assignment as well.
    with pytest.raises(CompilationError):
        prove_native_prefix(analysis, "intervals::read_modify")
    with pytest.raises(CompilationError, match="registered whole-closure"):
        prove(analysis, ids)


def object_fixture(tmp_path, *, kind=8, component=None, body=None, actual_attributes="",
                   selected="call adjusted(control=control,a=a)"):
    sources = [
        f"""module object_types
type::state
{component or f'real({kind})::value'}
end type
end module
""",
        f"""module object_helpers
use object_types,only:formal_type=>state
contains
subroutine modify(a,control)
real({kind}),allocatable,intent(inout)::a(:)
type(formal_type),intent(inout)::control
{body or f'control%value=control%value+1.0_{kind}' + chr(10) + 'a(1)=a(1)+control%value'}
end subroutine
end module
""",
        f"""module intervals
use object_types,only:actual_type=>state
use object_helpers,only:adjusted=>modify
contains
subroutine advance(a,flag)
real({kind}),allocatable,intent(inout)::a(:)
logical,intent(in)::flag
type(actual_type){actual_attributes}::control
real({kind}),allocatable::temporary(:)
integer::stage
do stage=1,3
{selected}
call move_alloc(a,temporary)
end do
end subroutine
end module
"""]
    paths = []
    for index, source in enumerate(sources):
        path = tmp_path / f"objects{index}.f90"
        path.write_text(source)
        paths.append(path)
    analysis = SourceEffects(paths)
    graph = analysis.structure(ENTRY)
    loop = next(node for node in analysis.routines[ENTRY].execution.content
                if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    selected_ids = (graph.node_id(next(node for node in loop.content if type(node).__name__ == "Call_Stmt")),)
    return paths, analysis, graph, selected_ids


@pytest.mark.parametrize("kind", [4, 8])
def test_renamed_original_fixed_object_mapping_does_not_block_numeric_descriptor_proof(tmp_path, kind):
    _paths, analysis, _graph, ids = object_fixture(tmp_path, kind=kind)
    public = prove(analysis, ids).public()
    mapping, = public["source_object_mappings"]
    assert mapping["declaring_type"] == "object_types::state"
    assert mapping["source_only"]
    assert not mapping["numerical_abi"]
    assert not mapping["native_continuation_authority"]
    assert public["fixed_source_object_requirements"]
    assert not public["prefix_effects_proved"]
    assert not public["execution_authorized"]
    assert public["resources"][0]["stable"]
    # Public copies cannot change subsequent source authority.
    mapping["type_definition"]["fields"].clear()
    assert prove(analysis, ids).public()["source_object_mappings"][0]["type_definition"]["fields"]


@pytest.mark.parametrize("component", ["real(8),pointer::value", "real(8),allocatable::value(:)",
                                       "real(8)::value\nreal(8),pointer::hidden"])
def test_fixed_object_mapping_checks_entire_type_including_unused_descriptor_siblings(tmp_path, component):
    _paths, analysis, _graph, ids = object_fixture(tmp_path, component=component)
    with pytest.raises(CompilationError, match="source object|source component"):
        prove(analysis, ids)


@pytest.mark.parametrize("actual_attributes", [",pointer", ",allocatable", ",optional"])
def test_fixed_object_actual_association_remains_original_and_nonoptional(tmp_path, actual_attributes):
    _paths, analysis, _graph, ids = object_fixture(tmp_path, actual_attributes=actual_attributes)
    with pytest.raises(CompilationError, match="source object"):
        prove(analysis, ids)


@pytest.mark.parametrize("body", ["a=0.0_8", "deallocate(a)", "call unknown(a)", "control=control"])
def test_fixed_object_mapping_never_skips_child_descriptor_or_whole_object_effects(tmp_path, body):
    _paths, analysis, _graph, ids = object_fixture(tmp_path, body=body)
    with pytest.raises(CompilationError, match="whole allocatable|unknown selected storage|original expression storage"):
        prove(analysis, ids)


def test_fixed_object_schema_mutation_invalidates_interval_token(tmp_path):
    _paths, analysis, _graph, ids = object_fixture(tmp_path)
    proof = prove(analysis, ids)
    schema = analysis.modules["object_types"].component_types["state"]
    schema.members.clear()
    with pytest.raises(CompilationError, match="schema changed"):
        proof.resource_state(ROOT)


def test_fixed_object_original_import_mutation_cannot_issue_new_authority(tmp_path):
    _paths, analysis, _graph, ids = object_fixture(tmp_path)
    analysis.routines["object_helpers::modify"].scope.imports["formal_type"] = ("foreign", "state")
    with pytest.raises(CompilationError, match="original.*authority|declaration scope changed"):
        prove(analysis, ids)


JOINED_BODY = """!$omp parallel private(i)
!$omp do
do i=1,size(a)
a(i)=a(i)+1.0_8
end do
!$omp end do nowait
!$omp do
do i=1,size(a)
a(i)=a(i)+2.0_8
end do
!$omp end do
!$omp end parallel"""


def register_joined(analysis, procedure="intervals::read_modify"):
    originals = tuple(analysis.routines[procedure].execution.content)
    return analysis.joined_completion(procedure, originals)


def test_existing_exact_whole_joined_token_supplies_directive_evidence_only(tmp_path):
    declaration = "real(8),intent(inout)::a(:)\ninteger::i"
    _path, analysis, _graph, ids = fixture(tmp_path, child=JOINED_BODY, child_decl=declaration)
    discovered = prove(analysis, ids).public()
    assert not analysis._joined_completions
    token = register_joined(analysis)
    public = prove(analysis, ids).public()
    join, = public["native_joined_descriptor_requirements"]
    assert discovered["native_joined_descriptor_requirements"] == public["native_joined_descriptor_requirements"]
    assert join["proof_identity"] == token.identity
    assert join["whole_original_group_scanned"]
    assert join["retains_original_team_and_directives"]
    assert not join["internal_cuts_authorized"]
    assert not join["execution_authorized"]
    assert not public["native_prefix_descriptor_requirements"]


@pytest.mark.parametrize("tail", ["a=0.0_8", "deallocate(a)", "call unknown(a)"])
def test_joined_token_cannot_hide_original_later_child_descriptor_effects(tmp_path, tail):
    declaration = "real(8),allocatable,intent(inout)::a(:)\ninteger::i"
    _path, analysis, _graph, ids = fixture(tmp_path, child=JOINED_BODY + "\n" + tail, child_decl=declaration)
    routine = analysis.routines["intervals::read_modify"]
    group = tuple(routine.execution.content[:-1])
    analysis.joined_completion(routine.qualified, group)
    with pytest.raises(CompilationError, match="whole allocatable|unknown selected storage"):
        prove(analysis, ids)


def test_copied_joined_token_is_not_descriptor_authority(tmp_path):
    _path, analysis, _graph, ids = fixture(tmp_path, child=JOINED_BODY,
                                         child_decl="real(8),intent(inout)::a(:)\ninteger::i")
    token = register_joined(analysis)
    analysis._joined_completions["copied"] = copy(token)
    with pytest.raises(CompilationError, match="whole original joined-group authority"):
        prove(analysis, ids)


def test_joined_token_does_not_authorize_partial_original_group_selection(tmp_path):
    selected = JOINED_BODY.replace("private(i)", "private(counter)").replace("do i=", "do counter=").replace("a(i)", "a(counter)")
    _path, analysis, graph, _ids = fixture(tmp_path, selected=selected,
                                         prefix="counter=2", child="continue")
    routine = analysis.routines[ENTRY]
    loop = next(node for node in routine.execution.content if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    group = tuple(loop.content[1:-2])
    analysis.joined_completion(ENTRY, group)
    opening = graph.node_id(group[0])
    with pytest.raises(CompilationError, match="registered whole-closure"):
        prove(analysis, (opening,))


def large_unrelated_internal(*, reached=False):
    declarations = "\n".join(f"real(8)::unused_{index}" for index in range(300))
    return ("call inactive()" if reached else "a(1)=a(1)+1.0_8") + f"""
contains
subroutine inactive()
{declarations}
call unknown_inactive()
end subroutine
"""


def test_unreached_contains_sibling_does_not_spend_reached_descriptor_resolution_budget(tmp_path):
    _path, analysis, _graph, ids = fixture(tmp_path, child=large_unrelated_internal())
    assert analysis.operation_limit == 256
    assert analysis.depth_limit == 8
    public = prove(analysis, ids).public()
    assert set(public["source_call_closure"]) == {"intervals::read_modify"}
    assert public["source_operations"] == 2
    assert public["resources"][0]["stable"]


def test_reached_internal_child_still_rejects_unknown_original_effects(tmp_path):
    _path, analysis, _graph, ids = fixture(tmp_path, child=large_unrelated_internal(reached=True))
    with pytest.raises(CompilationError, match="unknown selected storage effects: boundary:.*unknown_inactive"):
        prove(analysis, ids)
    assert analysis.operation_limit == 256
    assert not getattr(analysis, "_descriptor_window_proofs", {})


@pytest.mark.parametrize("mutation", ["imports", "kinds", "binding"])
def test_own_original_resolution_mutations_still_reject_with_large_inactive_child(tmp_path, mutation):
    _path, analysis, _graph, ids = fixture(tmp_path, child=large_unrelated_internal())
    proof = prove(analysis, ids)
    scope = analysis.routines["intervals::read_modify"].scope
    if mutation == "imports":
        scope.imports["missing"] = ("foreign", "name")
    elif mutation == "kinds":
        scope.kinds.values["rk"] = 4
    else:
        scope.bindings["a"].intent = "out"
    with pytest.raises(CompilationError, match="original.*authority"):
        proof.resource_state(ROOT)


def test_own_automatic_specification_effects_are_not_skipped_with_inactive_child(tmp_path):
    declaration = "real(8),intent(inout)::a(:)\nreal(8)::scratch(unknown_bound())"
    _path, analysis, _graph, ids = fixture(tmp_path, child=large_unrelated_internal(), child_decl=declaration)
    with pytest.raises(CompilationError, match="unknown function/escape"):
        prove(analysis, ids)
