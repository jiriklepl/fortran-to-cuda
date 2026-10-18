"""Issued source routes bound admission to still-active original paths."""

from dataclasses import replace
from hashlib import sha256

import pytest
from fparser.two.utils import walk

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.closed_selection import ClosedIntervalOwner, interval_candidates
from compiler.scopes.lexical import LexicalBoundaryPatch, LexicalOwner
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

SOURCE = """module ordered_fields
implicit none
real(8)::old(-1:16),viscosity(-1:16),star(-1:16),out(-1:16)
integer::native_visits=0
contains
subroutine advance(n,unsupported)
integer,intent(in)::n
logical,intent(in)::unsupported
call producer(old,viscosity,n)
if(unsupported) then
call declined(old,star,n)
call closed_tail(old,star,n)
else
call diffusion(old,viscosity,star,n)
endif
call accumulate(star,out,n)
end subroutine
subroutine producer(a,v,n)
real(8),intent(in)::a(-1:)
real(8),intent(inout)::v(-1:)
integer,intent(in)::n
integer::i
!$omp parallel do private(i) schedule(runtime)
do i=0,n-1
v(i)=2.d0*a(i)
enddo
!$omp end parallel do
end subroutine
subroutine declined(a,s,n)
real(8),intent(in)::a(-1:)
real(8),intent(inout)::s(-1:)
integer,intent(in)::n
integer::i
real(8),allocatable::scratch(:)
allocate(scratch(0:n-1))
!$omp parallel do private(i) schedule(runtime)
do i=0,n-1
scratch(i)=sum(a)
s(i)=s(i)+scratch(i)
enddo
!$omp end parallel do
deallocate(scratch)
native_visits=native_visits+1
end subroutine
subroutine closed_tail(a,s,n)
real(8),intent(in)::a(-1:)
real(8),intent(inout)::s(-1:)
integer,intent(in)::n
integer::i
!$omp parallel do private(i) schedule(runtime)
do i=0,n-1
s(i)=s(i)+a(i)
enddo
!$omp end parallel do
end subroutine
subroutine diffusion(a,v,s,n)
real(8),intent(in)::a(-1:),v(-1:)
real(8),intent(inout)::s(-1:)
integer,intent(in)::n
integer::i
!$omp parallel do private(i) schedule(runtime)
do i=0,n-1
s(i)=v(i)+a(i)
enddo
!$omp end parallel do
end subroutine
subroutine accumulate(s,o,n)
real(8),intent(in)::s(-1:)
real(8),intent(inout)::o(-1:)
integer,intent(in)::n
integer::i
!$omp parallel do private(i) schedule(runtime)
do i=0,n-1
o(i)=o(i)+s(i)
enddo
!$omp end parallel do
end subroutine
end module
"""


def fixture(tmp_path, *, source=SOURCE, closed=True):
    path = tmp_path / "ordered.f90"
    path.write_text(source)
    builder = ScopeBuilder([path], "ordered_fields::advance",
        options=CompilerOptions(gpu_policy="sections"),
        config=OffloadConfig(policy="sections", scope_execution="reached"),
        facts={"schema_version": 1, "participation": "serial",
               "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
               "captures": {"ordered_fields::" + name: dict(FACT)
                            for name in ("old", "viscosity", "star", "out")}})
    originals = tuple(builder.entry.execution.content)
    if closed:
        site = next(site for site in interval_candidates(builder)
                    if len(site.originals) == len(originals)
                    and all(left is right for left, right in zip(site.originals, originals, strict=True)))
        owner = ClosedIntervalOwner(builder, site.originals, site.prefix, site.controls)
    else:
        owner = LexicalOwner(builder)
    return path, builder, owner, originals


def reject_reanalysis_after_draft(monkeypatch, owner, *, forbidden=("declined", "closed_tail")):
    """Watch real admission APIs; the draft still runs its real source proofs."""
    import compiler.scopes.interval_draft as interval
    import compiler.scopes.module_owner as module_owner

    forbidden_calls = {id(node) for node in walk(owner.routine.execution)
                       if type(node).__name__ == "Call_Stmt"
                       and str(node.items[0]).lower() in forbidden}
    declined_calls = {id(node) for node in walk(owner.routine.execution)
                      if type(node).__name__ == "Call_Stmt" and str(node.items[0]).lower() == "declined"}
    forbidden_procedures = {"ordered_fields::" + name for name in forbidden if name != "declined"}
    attempts = {"admit": 0, "native": 0}
    issued = []
    prepare = interval.prepare_interval_draft

    def check_nodes(nodes, *, native_attempt=None):
        identities = {id(node) for original in nodes for node in walk(original)}
        excluded = forbidden_calls - (declined_calls if native_attempt else set())
        assert not identities & excluded, "a declined child was borrowed or its closed tail was re-admitted"
        if native_attempt and identities & declined_calls:
            attempts[native_attempt] += 1
            assert attempts[native_attempt] == 1, "declined numerical child retried native admission"

    def prepare_once(candidate, originals):
        draft = prepare(candidate, originals)
        declined = [anchor for anchor in draft.numerical_anchors
                    if anchor.procedure == "ordered_fields::declined"]
        assert declined
        assert all(anchor.prepared is None for anchor in declined)
        assert any(anchor.procedure == "ordered_fields::closed_tail" and anchor.prepared is not None
                   for anchor in draft.numerical_anchors)
        issued.append(draft)

        original_admit = LexicalOwner.admit

        def admit(member, nodes, **kwargs):
            check_nodes(member.builder.inline.original_selection(tuple(nodes)), native_attempt="admit")
            return original_admit(member, nodes, **kwargs)

        monkeypatch.setattr(LexicalOwner, "admit", admit)
        for method_name in ("internal", "mapped_child"):
            original_method = getattr(LexicalOwner, method_name)

            def guarded(member, node, _method=original_method):
                check_nodes(member.builder.inline.original_selection((node,)))
                return _method(member, node)

            monkeypatch.setattr(LexicalOwner, method_name, guarded)
        original_borrow = module_owner.borrow_module

        def borrow(member, node, **kwargs):
            check_nodes(member.builder.inline.original_selection((node,)))
            return original_borrow(member, node, **kwargs)

        monkeypatch.setattr(module_owner, "borrow_module", borrow)
        original_native = interval.SourceIntervalDraft.native_operation

        def native_operation(current, member, nodes):
            check_nodes(member.builder.inline.original_selection(tuple(nodes)), native_attempt="native")
            return original_native(current, member, nodes)

        monkeypatch.setattr(interval.SourceIntervalDraft, "native_operation", native_operation)
        analysis_type = type(candidate.builder.analysis)
        original_summary = analysis_type.segment_summary

        def segment_summary(analysis, procedure, selected, *args, **kwargs):
            assert procedure not in forbidden_procedures, "closed child requested another native effect summary"
            return original_summary(analysis, procedure, selected, *args, **kwargs)

        monkeypatch.setattr(analysis_type, "segment_summary", segment_summary)
        return draft

    monkeypatch.setattr(interval, "prepare_interval_draft", prepare_once)
    return issued


def test_declined_branch_tries_native_once_and_retains_supported_sibling(tmp_path, monkeypatch):
    path, builder, owner, originals = fixture(tmp_path)
    issued = reject_reanalysis_after_draft(monkeypatch, owner)
    assert owner.run(originals)
    outputs, report = builder.finish()
    assert len(issued) == 1
    plan = owner.execution_plan.validate()
    assert plan.numerical_invocations == 3
    assert {member.routine.qualified for member in plan.members} == {
        "ordered_fields::advance", "ordered_fields::producer",
        "ordered_fields::diffusion", "ordered_fields::accumulate"}
    text = outputs[report["sources"][str(path)]["replacement"]].lower()
    assert text.count("call declined(old,star,n)") == 1
    assert text.count("call closed_tail(old,star,n)") == 1
    assert text.count("native_visits=native_visits+1") == 1
    patches = [patch for _first, _last, patch in plan.patches(owner) if isinstance(patch, LexicalBoundaryPatch)]
    assert patches
    close = next(patch for patch in patches if "call declined" in patch.original_source.lower())
    close.validate(owner, require_source=True)
    assert close.render(owner).index("fort_scope_close(") < close.render(owner).lower().index("call declined")
    assert owner.enabled + " = .false." in close.render(owner)
    suffix, = owner.closed_suffixes
    assert suffix.boundary is close
    assert [str(node.items[0]).lower() for node in suffix.originals] == ["closed_tail"]
    assert suffix.validate(owner) is suffix


def test_unconditional_close_keeps_entire_linear_suffix_native(tmp_path, monkeypatch):
    branch = "if(unsupported) then\ncall declined(old,star,n)\ncall closed_tail(old,star,n)\nelse\ncall diffusion(old,viscosity,star,n)\nendif"
    source = SOURCE.replace(branch, "call declined(old,star,n)\ncall closed_tail(old,star,n)")
    path, builder, owner, originals = fixture(tmp_path, source=source, closed=False)
    issued = reject_reanalysis_after_draft(monkeypatch, owner,
                                           forbidden=("declined", "closed_tail", "accumulate"))
    assert owner.run(originals)
    outputs, report = builder.finish()
    assert len(issued) == 1
    assert owner.execution_plan.numerical_invocations == 1
    assert {member.routine.qualified for member in owner.execution_plan.members} == {
        "ordered_fields::advance", "ordered_fields::producer"}
    text = outputs[report["sources"][str(path)]["replacement"]].lower()
    for call in ("call declined(old,star,n)", "call closed_tail(old,star,n)", "call accumulate(star,out,n)"):
        assert text.count(call) == 1


def conditional_source():
    branch = "if(unsupported) then\ncall declined(old,star,n)\ncall closed_tail(old,star,n)\nelse\ncall diffusion(old,viscosity,star,n)\nendif"
    return SOURCE.replace(branch, "if(unsupported) call declined(old,star,n)\ncall diffusion(old,viscosity,star,n)")


@pytest.mark.parametrize("mutation", ["position", "guard", "source"])
def test_source_directed_conditional_close_keeps_exact_boundary_authority(tmp_path, mutation):
    _path, builder, owner, originals = fixture(tmp_path, source=conditional_source())
    assert owner._prepare_lexical(originals)
    index, (first, last, patch) = next((index, patch) for index, patch in enumerate(owner.patches)
                                     if isinstance(patch[2], LexicalBoundaryPatch))
    assert patch.conditional_guard == "unsupported"
    patch.validate(owner, require_source=True)
    change = {"position": {"first": first+1}, "guard": {"conditional_guard": ".true."},
              "source": {"original_source": "call different()\n"}}[mutation]
    owner.patches[index] = first, last, replace(patch, **change)
    with pytest.raises(CompilationError, match="boundary|boundaries"):
        owner.validate_plan(builder.inline.prepare(originals))
    assert not builder.scopes


def test_source_directed_owner_rejects_changed_original_source_at_publication(tmp_path):
    path, builder, owner, originals = fixture(tmp_path)
    assert owner.run(originals)
    path.write_text(path.read_text().replace("if(unsupported) then", "if(.not.unsupported) then"))
    with pytest.raises(CompilationError, match="changed"):
        builder.finish()


@pytest.mark.parametrize("mutation", ["source", "boundary", "outer_continuation", "sequence"])
def test_closed_suffix_cannot_be_retargeted_to_another_original_path(tmp_path, mutation):
    _path, _builder, owner, originals = fixture(tmp_path)
    assert owner._prepare_lexical(originals)
    suffix, = owner.closed_suffixes
    assert suffix.validate(owner) is suffix
    changes = {
        "source": {"original_source": "call different()\n"},
        "boundary": {"boundary": replace(suffix.boundary)},
        "outer_continuation": {"originals": (originals[-1],)},
        "sequence": {"sequence": (*suffix.sequence, originals[-1])},
    }
    with pytest.raises(CompilationError, match="suffix|boundary|selection|source|guarded segment"):
        replace(suffix, **changes[mutation]).validate(owner)


@pytest.mark.parametrize("internal_child", [False, True])
def test_failed_prepared_child_route_closes_without_native_or_borrow_retry(tmp_path, monkeypatch, internal_child):
    import compiler.scopes.module_owner as module_owner

    source = SOURCE.replace("call declined(old,star,n)", "call diffusion(old,viscosity,star,n)")
    if internal_child:
        source = source.replace("call diffusion(old,viscosity,star,n)", "call local_diffusion()")
        source = source.replace("end subroutine\nsubroutine producer", """contains
subroutine local_diffusion()
integer::i
!$omp parallel do private(i) schedule(runtime)
do i=0,n-1
star(i)=viscosity(i)+old(i)
enddo
!$omp end parallel do
end subroutine
end subroutine
subroutine producer""", 1)
    path, builder, owner, originals = fixture(tmp_path, source=source)
    branch = next(node for node in originals if type(node).__name__ == "If_Construct")
    failed, tail, _sibling = [node for node in walk(branch) if type(node).__name__ == "Call_Stmt"]
    attempts = []
    original_borrow = module_owner.borrow_module
    original_admit = LexicalOwner.admit
    original_internal = LexicalOwner.internal

    def borrow(member, node, **kwargs):
        if node is failed:
            assert not internal_child, "failed internal route retried a module implementation"
            attempts.append(node)
            raise CompilationError("fixture prepared child association was rejected")
        assert node is not tail, "closed tail requested another borrowed implementation"
        return original_borrow(member, node, **kwargs)

    def internal(member, node):
        if internal_child and node is failed:
            attempts.append(node)
            raise CompilationError("fixture prepared internal child association was rejected")
        assert node is not tail, "closed tail requested another internal implementation"
        return original_internal(member, node)

    def admit(member, nodes, **kwargs):
        selected = member.builder.inline.original_selection(tuple(nodes))
        assert not any(node is failed or node is tail for original in selected for node in walk(original)), (
            "failed prepared route retried native admission or admitted its closed suffix")
        return original_admit(member, nodes, **kwargs)

    monkeypatch.setattr(module_owner, "borrow_module", borrow)
    monkeypatch.setattr(LexicalOwner, "admit", admit)
    monkeypatch.setattr(LexicalOwner, "internal", internal)
    assert owner.run(originals)
    outputs, report = builder.finish()
    assert attempts == [failed]
    assert owner.execution_plan.numerical_invocations == 3
    expected = "::local_diffusion" if internal_child else "::diffusion"
    assert any(member.routine.qualified.endswith(expected) for member in owner.execution_plan.members)
    assert "ordered_fields::closed_tail" not in {member.routine.qualified for member in owner.execution_plan.members}
    text = outputs[report["sources"][str(path)]["replacement"]].lower()
    assert text.count("call closed_tail(old,star,n)") == 1
    suffix, = owner.closed_suffixes
    assert suffix.boundary.originals == (failed,)
    assert suffix.originals == (tail,)


def test_declined_numerical_child_keeps_proved_native_correction_and_later_gpu_units(tmp_path, monkeypatch):
    import compiler.scopes.module_owner as module_owner

    first, last = SOURCE.index("subroutine declined("), SOURCE.index("subroutine closed_tail(")
    source = SOURCE[:first] + """subroutine correction()
do native_i=0,15
viscosity(native_i)=viscosity(native_i)+sum(old)
enddo
native_visits=native_visits+1
end subroutine
""" + SOURCE[last:]
    source = source.replace("integer::native_visits=0", "integer::native_visits=0,native_i")
    first, last = source.index("if(unsupported) then"), source.index("call accumulate(star,out,n)")
    source = source[:first] + "call correction()\ncall diffusion(old,viscosity,star,n)\n" + source[last:]
    path, builder, owner, originals = fixture(tmp_path, source=source)
    correction = next(node for node in originals if str(node.items[0]).lower() == "correction")
    attempts = []
    original_admit, original_borrow = LexicalOwner.admit, module_owner.borrow_module

    def admit(member, nodes, **kwargs):
        if any(node is correction for node in nodes):
            attempts.append(correction)
        return original_admit(member, nodes, **kwargs)

    def borrow(member, node, **kwargs):
        assert node is not correction, "native correction requested a borrowed numerical implementation"
        return original_borrow(member, node, **kwargs)

    monkeypatch.setattr(LexicalOwner, "admit", admit)
    monkeypatch.setattr(module_owner, "borrow_module", borrow)
    assert owner.run(originals)
    outputs, report = builder.finish()
    assert attempts == [correction]
    route = owner.interval_draft.call_route(correction)
    assert route.kind == "declined_numerical_child"
    assert route.anchor_ordinals
    assert all(owner.interval_draft.numerical_anchors[index].prepared is None for index in route.anchor_ordinals)
    plan = owner.execution_plan.validate()
    assert plan.numerical_invocations == 3
    assert all(not member.boundaries for member in plan.members)
    assert "ordered_fields::correction" not in {member.routine.qualified for member in plan.members}
    operation, = [operation for unit in plan.units(owner)
                  for operation in unit.scope.public()["native_operations"]]
    assert operation["preserves_original_source"]
    assert operation["resource_effects"] == {"ordered_fields::old": ["read"],
                                             "ordered_fields::viscosity": ["read", "write"]}
    assert operation["demand_relative_effects"]["native_completion"]["available"]
    text = outputs[report["sources"][str(path)]["replacement"]].lower()
    assert text.count("call correction()") == 1
    assert text.count("native_visits=native_visits+1") == 1
    assert "fort_scope_host_begin" in text
    assert "fort_scope_host_end" in text
