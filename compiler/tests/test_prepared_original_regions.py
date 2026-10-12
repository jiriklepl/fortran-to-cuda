"""Source-only numerical preparation retains original policy and context."""

from copy import copy
from dataclasses import replace
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend import numerical_preparation
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.numerical_preparation import (
    NumericalPreparationProvider,
    NumericalPreparationUnsupported,
    PreparedOriginalRegion,
    prepare_original_region,
)
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig

ENTRY = "renamed_region::form_field"
OPTIONS = CompilerOptions(gpu_policy="sections", memory_model="scoped", scope_execution="reached")
CONFIG = OffloadConfig(policy="sections", scope_execution="reached")
ASSOCIATION_SOURCE = """module original_aliases
implicit none
contains
subroutine form_field(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(out)::b(:)
integer,intent(in)::n
integer::i
associate(limit=>n)
do i=1,limit
b(i)=2.0_8*a(i)
enddo
end associate
end subroutine
end module
"""
SOURCE = """module renamed_region
implicit none
contains
pure subroutine leaf_value(x,work)
real(8),intent(in)::x
real(8),intent(out)::work(2)
work(1)=sqrt(abs(x)+1.0_8)
work(2)=work(1)*work(1)+x
end subroutine
subroutine form_field(a,b,n,width)
real(8),intent(in)::a(:)
real(8),intent(out)::b(:)
integer,intent(in)::n,width
integer::tile,i
real(8)::work(2)
continue
!$omp parallel do private(i,work)
do tile=1,n,width
do i=tile,min(tile+width-1,n)
call leaf_value(a(i),work)
b(i)=work(2)
enddo
enddo
!$omp end parallel do
end subroutine
end module
"""


def fixture(directory, source=SOURCE):
    path = directory / "renamed.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    routine = analysis.routines[ENTRY]
    groups = grouped_original_sequence(routine.execution.content, {})
    selection = next(item.original_nodes for item in groups if isinstance(item, NativeGroupCandidate))
    provider = NumericalPreparationProvider(analysis, options=OPTIONS, config=CONFIG, environment_root=ENTRY)
    return path, analysis, provider, selection


def configured_fixture(directory, *, padding=3):
    path = directory / "renamed.f90"
    prepared = directory / "prepared.f90"
    dependency = directory / "configuration.inc"
    text = SOURCE + "\n" * padding
    path.write_text(text)
    prepared.write_text(text)
    dependency.write_text("integer,parameter::configuration=1\n")
    document = {"schema_version": 1, "preserves_source_order": True,
                "source_inputs": {str(path): sha256(path.read_bytes()).hexdigest()},
                "configuration": {"defines": ["ORIGINAL_CONFIGURATION"]},
                "dependencies": {str(dependency): sha256(dependency.read_bytes()).hexdigest()},
                "entries": [{"source": str(path), "path": str(prepared),
                             "sha256": sha256(prepared.read_bytes()).hexdigest(),
                             "line_map": list(range(1, len(text.splitlines()) + 1))}]}
    analysis = SourceEffects([path], analysis_sources=document)
    groups = grouped_original_sequence(analysis.routines[ENTRY].execution.content, {})
    selection = next(item.original_nodes for item in groups if isinstance(item, NativeGroupCandidate))
    provider = NumericalPreparationProvider(analysis, options=OPTIONS, config=CONFIG, environment_root=ENTRY)
    return path, analysis, provider, selection


def test_large_configured_line_map_does_not_consume_executable_source_budget(tmp_path, monkeypatch):
    path, analysis, provider, selection = configured_fixture(tmp_path, padding=34000)
    original = analysis.inputs.identity()
    assert len(original["entries"][0]["line_map"]) > 34000
    # The previous identity representation visited every mapping scalar on
    # each source check, exhausting the shared budget for this small region.
    with monkeypatch.context() as old_identity:
        old_identity.setattr(numerical_preparation, "_input_identity", lambda analysis, _budget: analysis.inputs.identity())
        with pytest.raises(CompilationError, match="source-scan budget exhausted"):
            provider.prepare(analysis, ENTRY, selection)
    prepared = provider.prepare(analysis, ENTRY, selection)
    assert prepared.offload.available
    assert prepared.validate(analysis, ENTRY, selection) is prepared
    assert analysis.inputs.identity() == original
    assert type(analysis.inputs.entries[str(path)]["line_map"]) is list

    def exhausted(_count):
        raise CompilationError("parent aggregate source budget exhausted")

    with pytest.raises(CompilationError, match="parent aggregate"):
        provider.prepare(analysis, ENTRY, selection, consume=exhausted)
    with pytest.raises(CompilationError, match="parent aggregate"):
        prepared.validate(analysis, ENTRY, selection, consume=exhausted)
    # A mapping well beyond the selected AST still belongs to provenance.
    analysis.inputs.entries[str(path)]["line_map"][-1] = None
    with pytest.raises(CompilationError) as failure:
        prepared.validate(analysis, ENTRY, selection)
    assert not isinstance(failure.value, NumericalPreparationUnsupported)


@pytest.mark.parametrize("change", ["map_tail", "map_order", "map_none", "configuration", "dependency",
                                  "prepared_path", "original_file", "prepared_file", "dependency_file"])
def test_configured_provenance_changes_invalidate_registered_preparation(tmp_path, change):
    path, analysis, provider, selection = configured_fixture(tmp_path)
    prepared = provider.prepare(analysis, ENTRY, selection)
    entry = analysis.inputs.entries[str(path)]
    dependency = tmp_path / "configuration.inc"
    if change == "map_tail":
        entry["line_map"][-1] = entry["line_map"][-2]
    elif change == "map_order":
        entry["line_map"][-2:] = reversed(entry["line_map"][-2:])
    elif change == "map_none":
        entry["line_map"][-1] = None
    elif change == "configuration":
        analysis.inputs.configuration["defines"].append("CHANGED_CONFIGURATION")
    elif change == "dependency":
        alternate = tmp_path / "same_configuration.inc"
        alternate.write_bytes(dependency.read_bytes())
        analysis.inputs.dependencies = {str(alternate): sha256(alternate.read_bytes()).hexdigest()}
    elif change == "prepared_path":
        alternate = tmp_path / "same_prepared.f90"
        alternate.write_bytes((tmp_path / "prepared.f90").read_bytes())
        entry["path"] = str(alternate)
    else:
        changed = {"original_file": path, "prepared_file": tmp_path / "prepared.f90",
                   "dependency_file": dependency}[change]
        changed.write_text(changed.read_text() + "! changed after preparation\n")
    with pytest.raises(CompilationError) as failure:
        prepared.validate(analysis, ENTRY, selection)
    assert not isinstance(failure.value, NumericalPreparationUnsupported)


@pytest.mark.parametrize("change", ["tuple", "bool", "nested", "length", "text"])
def test_line_map_fingerprint_rejects_noninteger_or_unbounded_data(tmp_path, monkeypatch, change):
    path, analysis, _provider, _selection = configured_fixture(tmp_path)
    entry = analysis.inputs.entries[str(path)]
    if change == "tuple":
        entry["line_map"] = tuple(entry["line_map"])
    elif change == "bool":
        entry["line_map"][-1] = True
    elif change == "nested":
        entry["line_map"][-1] = [1]
    else:
        monkeypatch.setattr(numerical_preparation, "_IDENTITY_TEXT_LIMIT", 64)
        entry["line_map"] = [None] * 33 if change == "length" else [10**40, 10**40]
    with pytest.raises(CompilationError, match="line.map") as failure:
        numerical_preparation._input_identity(analysis, numerical_preparation._Budget())
    assert not isinstance(failure.value, NumericalPreparationUnsupported)


@pytest.mark.parametrize("mutation", [None, "scope_binding", "scope_parent", "selector_source"])
def test_original_association_scope_is_authenticated_during_numerical_preparation(tmp_path, mutation):
    from fparser.two import Fortran2003 as F
    from fparser.two.utils import walk

    path = tmp_path / "aliases.f90"
    path.write_text(ASSOCIATION_SOURCE)
    analysis = SourceEffects([path])
    entry = "original_aliases::form_field"
    routine = analysis.routines[entry]
    associate, = walk(routine.execution, F.Associate_Construct)
    loop, = walk(associate, F.Block_Nonlabel_Do_Construct)
    record = analysis._associate_scopes[id(associate)]
    assert record.available
    assert analysis._source_function_scopes[id(record.scope)][0] is record.scope
    provider = NumericalPreparationProvider(analysis, options=OPTIONS, config=CONFIG, environment_root=entry)
    prepared = provider.prepare(analysis, entry, (loop,))
    assert prepared.offload.available
    assert prepared.validate(analysis, entry, (loop,)) is prepared
    assert record.scope.bindings["limit"].root == "argument::n"
    assert path.read_text() == ASSOCIATION_SOURCE
    if mutation is None:
        return
    if mutation == "scope_binding":
        record.scope.bindings["limit"] = copy(record.scope.bindings["limit"])
    elif mutation == "scope_parent":
        record.scope.parent = analysis.modules["original_aliases"]
    else:
        header = associate.content[0]
        selector, = header.items[1].items
        selector.items = (*selector.items[:-1], F.Name("i"))
    with pytest.raises(CompilationError):
        prepared.validate(analysis, entry, (loop,))
    with pytest.raises(CompilationError):
        provider.prepare(analysis, entry, (loop,))


def test_rejected_association_remains_a_source_boundary_without_scope_authority(tmp_path):
    from fparser.two import Fortran2003 as F
    from fparser.two.utils import walk

    path = tmp_path / "aliases.f90"
    path.write_text(ASSOCIATION_SOURCE.replace("limit=>n", "limit=>a(1)"))
    analysis = SourceEffects([path])
    routine = analysis.routines["original_aliases::form_field"]
    associate, = walk(routine.execution, F.Associate_Construct)
    record = analysis._associate_scopes[id(associate)]
    assert not record.available
    assert record.scope is None
    graph = analysis.structure(routine.qualified)
    point = next(node for node in graph.nodes.values() if node.kind == "boundary")
    assert "original scalar variable selector" in point.details["reason"]
    assert graph.source_nodes(point.id) == (associate,)


def test_source_only_preparation_retains_exact_original_checked_ir(tmp_path, monkeypatch):
    path, analysis, provider, selection = fixture(tmp_path)
    before = path.read_bytes()

    def forbidden(*_args, **_kwargs):
        pytest.fail("source-only preparation acquired capture or artifact authority")

    monkeypatch.setattr("compiler.emission.generate_sources", forbidden)
    monkeypatch.setattr("compiler.scopes.source.ScopeBuilder.capture", forbidden)
    prepared = provider.prepare(analysis, ENTRY, selection)
    assert type(prepared) is PreparedOriginalRegion
    assert prepared.validate(analysis, ENTRY, selection) is prepared
    assert prepared.offload.available
    assert prepared.public()["physical_backend"] == "scoped_pointer_abi"
    assert prepared.offload.units
    assert prepared.extraction.numerical_helpers == ("renamed_region::leaf_value",)
    assert prepared.extraction.private_arrays == ("work",)
    assert not prepared.scalar_changes  # Original complete team has private iterators/temporaries.
    assert prepared.completion_proof.identity == prepared.completion["proof_identity"]
    assert not prepared.public()["execution_authorized"]
    assert not prepared.public()["capture_authorized"]
    assert path.read_bytes() == before
    assert {item.name for item in tmp_path.iterdir()} == {path.name}


def test_authenticated_non_numerical_team_declines_without_masking_source_changes(tmp_path):
    source = SOURCE.replace("implicit none\ncontains", """use,intrinsic::ieee_exceptions,only:ieee_set_halting_mode,ieee_invalid
implicit none
contains""", 1).replace("continue\n", """continue
!$omp parallel
call ieee_set_halting_mode(ieee_invalid,.false.)
!$omp end parallel
""", 1)
    path, analysis, provider, selection = fixture(tmp_path, source)
    with pytest.raises(NumericalPreparationUnsupported, match="contains no counted loop"):
        provider.prepare(analysis, ENTRY, selection)
    # This is only candidate classification; the original source is still
    # authenticated before any unsupported result can become a native fallback.
    path.write_text(path.read_text() + "! changed source\n")
    with pytest.raises(CompilationError) as error:
        provider.prepare(analysis, ENTRY, selection)
    assert not isinstance(error.value, NumericalPreparationUnsupported)


def test_tile_guards_are_typed_original_operands_and_exact_legacy_bindings(tmp_path):
    _path, analysis, provider, selection = fixture(tmp_path)
    prepared = provider.prepare(analysis, ENTRY, selection)
    guards = prepared.preconditions
    assert [item.role for item in guards] == ["positive_tile_step", "tile_integer_range"]
    assert guards[0].operands[0].resource == "argument::width"
    assert guards[1].operands[0].resource == "argument::n"
    assert not guards[0].activation_domains
    assert {item.resource for item in guards[0].dependencies} == {"argument::width"}
    assert prepared.all_runtime_guards_typed
    assert prepared.covered_runtime_guards == prepared.extraction.runtime_guards
    assert guards[0].render(lambda bound: bound.resource.split("::")[-1]) == "int((width),kind=c_int64_t) > 0_c_int64_t"
    assert "checked INTEGER64" in guards[0].public()["evaluation"]


@pytest.mark.parametrize("target", ["provider", "preparation", "analysis", "selection"])
def test_copied_or_foreign_authority_cannot_validate(tmp_path, target):
    _path, analysis, provider, selection = fixture(tmp_path)
    prepared = provider.prepare(analysis, ENTRY, selection)
    def invalid():
        if target == "provider":
            copy(provider).validate(analysis)
        elif target == "preparation":
            replace(prepared).validate(analysis, ENTRY, selection)
        elif target == "analysis":
            prepared.validate(copy(analysis), ENTRY, selection)
        else:
            forged = tuple(copy(node) if type(node).__name__ == "Block_Nonlabel_Do_Construct" else node
                           for node in selection)
            prepared.validate(analysis, ENTRY, forged)

    with pytest.raises(CompilationError):
        invalid()


@pytest.mark.parametrize("change", ["source", "binding", "declaring_scope", "imports", "IR", "guard"])
def test_original_source_binding_policy_and_checked_ir_mutations_are_hard(tmp_path, change):
    path, analysis, provider, selection = fixture(tmp_path)
    prepared = provider.prepare(analysis, ENTRY, selection)
    helper = analysis.numerical_helpers["renamed_region::leaf_value"]
    if change == "source":
        path.write_text(path.read_text().replace("work(2)=work(1)*work(1)+x", "work(2)=work(1)*work(1)-x"))
    elif change == "binding":
        helper.scope.bindings["x"] = replace(helper.scope.bindings["x"], attributes=frozenset({"volatile"}))
    elif change == "declaring_scope":
        object.__setattr__(helper.scope.bindings["x"], "declaring_scope", analysis.routines[ENTRY].scope)
    elif change == "imports":
        helper.scope.imports["unused_alias"] = ("not_original", "storage")
    elif change == "IR":
        object.__setattr__(prepared.function, "parameters", ())
    else:
        object.__setattr__(prepared.preconditions[0], "_runtime_expression", "width > -1")
    with pytest.raises(CompilationError):
        prepared.validate(analysis, ENTRY, selection)


@pytest.mark.parametrize("policy", ["options", "config", "root"])
def test_preparation_cannot_cross_compiler_policy_or_environment_root(tmp_path, policy):
    _path, analysis, provider, selection = fixture(tmp_path)
    prepared = provider.prepare(analysis, ENTRY, selection)
    options, config, root = OPTIONS, CONFIG, ENTRY
    if policy == "options":
        options = replace(options, opt_level=0)
    elif policy == "config":
        config = replace(config, host_threads=2)
    else:
        root = "another_original::entry"
    with pytest.raises(CompilationError, match="policy changed"):
        prepared.validate_configuration(options=options, config=config, environment_root=root)
    with pytest.raises(CompilationError, match="policy changed"):
        provider.validate_configuration(options=options, config=config, environment_root=root)


def test_cold_and_validation_work_is_charged_and_parent_budget_is_hard(tmp_path):
    _path, analysis, provider, selection = fixture(tmp_path)
    charged = []
    prepared = provider.prepare(analysis, ENTRY, selection, consume=charged.append)
    assert charged
    assert set(charged) == {1}
    charged.clear()
    prepared.validate(analysis, ENTRY, selection, consume=charged.append)
    assert charged

    def exhausted(_count):
        raise CompilationError("parent aggregate source budget exhausted")

    with pytest.raises(CompilationError, match="parent aggregate"):
        provider.prepare(analysis, ENTRY, selection, consume=exhausted)
    with pytest.raises(CompilationError, match="parent aggregate"):
        prepared.validate(analysis, ENTRY, selection, consume=exhausted)


def test_large_calibration_data_keeps_full_identity_without_spending_statement_budget(tmp_path):
    from compiler.numerical_contract import numerical_build_contract

    _path, analysis, _provider, selection = fixture(tmp_path)
    profile = {"numerical_contract": numerical_build_contract(), "retained_observations": list(range(70000))}
    config = replace(CONFIG, profile=profile)
    provider = NumericalPreparationProvider(analysis, options=OPTIONS, config=config, environment_root=ENTRY)
    prepared = provider.prepare(analysis, ENTRY, selection)
    prepared.validate(analysis, ENTRY, selection)
    assert provider.public()["schema_version"] == 2
    profile["retained_observations"][-1] += 1
    with pytest.raises(CompilationError, match="policy changed"):
        prepared.validate_configuration(options=OPTIONS, config=config, environment_root=ENTRY)
    with pytest.raises(CompilationError):
        prepared.validate(analysis, ENTRY, selection)


@pytest.mark.parametrize("defect", ["nonfinite", "cycle", "object", "oversized"])
def test_profile_identity_rejects_invalid_or_over_budget_data(tmp_path, monkeypatch, defect):
    from compiler.numerical_contract import numerical_build_contract

    _path, analysis, _provider, _selection = fixture(tmp_path)
    profile = {"numerical_contract": numerical_build_contract()}
    if defect == "nonfinite":
        profile["raw"] = float("nan")
    elif defect == "cycle":
        profile["raw"] = profile
    elif defect == "object":
        profile["raw"] = object()
    else:
        monkeypatch.setattr(numerical_preparation, "_PROFILE_TEXT_LIMIT", 1024)
        profile["raw"] = "x" * 2048
    with pytest.raises(CompilationError, match="profile.*(identity|budget)"):
        NumericalPreparationProvider(analysis, options=OPTIONS, config=replace(CONFIG, profile=profile),
                                     environment_root=ENTRY)


def test_budget_callback_cannot_change_already_fingerprinted_profile(tmp_path, monkeypatch):
    from compiler.numerical_contract import numerical_build_contract

    _path, analysis, _provider, _selection = fixture(tmp_path)
    profile = {"numerical_contract": numerical_build_contract(), "raw": [1, 2, 3]}
    config = replace(CONFIG, profile=profile)
    provider = NumericalPreparationProvider(analysis, options=OPTIONS, config=config, environment_root=ENTRY)
    fingerprint = numerical_preparation._profile_identity
    state = {"profile_hashed": False}

    def tracked(*args, **kwargs):
        result = fingerprint(*args, **kwargs)
        state["profile_hashed"] = True
        return result

    def mutate(_count):
        if state.pop("profile_hashed", False):
            profile["raw"][-1] += 1

    monkeypatch.setattr(numerical_preparation, "_profile_identity", tracked)
    with pytest.raises(CompilationError, match="profile changed"):
        provider.validate(analysis, consume=mutate)


def test_repeated_derivation_reuses_exact_token_without_avoiding_work(tmp_path):
    _path, analysis, provider, selection = fixture(tmp_path)
    cold, reused = [], []
    first = provider.prepare(analysis, ENTRY, selection, consume=cold.append)
    second = provider.prepare(analysis, ENTRY, selection, consume=reused.append)
    assert second is first
    assert reused == cold
    equivalent = NumericalPreparationProvider(analysis, options=OPTIONS, config=CONFIG, environment_root=ENTRY)
    assert equivalent.prepare(analysis, ENTRY, selection) is first


@pytest.mark.parametrize("change", ["intrinsic_reexport", "binding", "declaring_scope"])
def test_reused_preparation_revalidates_separate_environment_reexport_authority(tmp_path, change):
    path = tmp_path / "renamed.f90"
    path.write_text(SOURCE)
    environment = tmp_path / "environment.f90"
    environment.write_text("""module reshuffled_environment
use,intrinsic::ieee_exceptions,only:exposed_set=>ieee_set_halting_mode,ieee_invalid
implicit none
integer,parameter::marker=1
end module
module distinct_invocation
use reshuffled_environment,only:invoke_mode=>exposed_set,ieee_invalid
use renamed_region,only:form_field
implicit none
contains
subroutine start(a,b,n,width)
real(8),intent(in)::a(:)
real(8),intent(out)::b(:)
integer,intent(in)::n,width
call invoke_mode(ieee_invalid,.false.)
call form_field(a,b,n,width)
end subroutine
end module
""")
    analysis = SourceEffects([path, environment])
    groups = grouped_original_sequence(analysis.routines[ENTRY].execution.content, {})
    selection = next(item.original_nodes for item in groups if isinstance(item, NativeGroupCandidate))
    provider = NumericalPreparationProvider(analysis, options=OPTIONS, config=CONFIG,
                                            environment_root="distinct_invocation::start")
    prepared = provider.prepare(analysis, ENTRY, selection)
    assert prepared.requires_original_team_environment_preflight
    assert prepared.function.requires_numerical_environment
    assert prepared.extraction.numerical_environment_required
    assert provider.prepare(analysis, ENTRY, selection) is prepared

    facade = analysis.modules["reshuffled_environment"]
    if change == "intrinsic_reexport":
        facade.intrinsic_procedures["exposed_set"] = "$intrinsic::ieee_exceptions::ieee_get_halting_mode"
    elif change == "binding":
        facade.bindings["marker"] = replace(facade.bindings["marker"])
    else:
        object.__setattr__(facade.bindings["marker"], "declaring_scope", analysis.modules["distinct_invocation"])
    with pytest.raises(CompilationError, match="original resolution authority changed"):
        prepared.validate(analysis, ENTRY, selection)
    with pytest.raises(CompilationError, match="original resolution authority changed"):
        provider.prepare(analysis, ENTRY, selection)


@pytest.mark.parametrize("member", ["scale", "unrelated"], ids=["selected-field", "unused-sibling"])
def test_fixed_field_views_reuse_original_object_authority_and_reject_schema_mutation(tmp_path, member):
    source = SOURCE.replace("implicit none\ncontains", """implicit none
type renamed_weights
real(8)::scale,alternate
real(8),allocatable::unrelated(:)
end type
contains""").replace(
        "subroutine form_field(a,b,n,width)\n",
        "subroutine form_field(a,b,n,width,coeff)\ntype(renamed_weights),intent(in)::coeff\n").replace(
        "b(i)=work(2)", "b(i)=work(2)*coeff%scale")
    _path, analysis, provider, selection = fixture(tmp_path, source)
    prepared = provider.prepare(analysis, ENTRY, selection)
    field = next(binding for binding in prepared.extraction.bindings if binding.root == "argument::coeff%scale")
    assert field.component_object is analysis.routines[ENTRY].scope.bindings["coeff"]
    assert provider.prepare(analysis, ENTRY, selection) is prepared
    assert prepared.validate(analysis, ENTRY, selection) is prepared

    schema = analysis.modules["renamed_region"].component_types["renamed_weights"]
    assert "allocatable" in str(schema.members["unrelated"][1]).lower()
    schema.members[member] = schema.members["alternate"]
    with pytest.raises(CompilationError, match="schema changed"):
        prepared.validate(analysis, ENTRY, selection)
    with pytest.raises(CompilationError, match="schema changed"):
        provider.prepare(analysis, ENTRY, selection)


def test_factory_and_nested_tile_activation_domains_keep_original_order(tmp_path):
    source = SOURCE.replace("integer::tile,i", "integer::tile,i,row,j").replace(
        "do tile=1,n,width\ndo i=tile,min(tile+width-1,n)",
        "do row=1,n,width\ndo tile=1,n,width\ndo j=row,min(row+width-1,n)\ndo i=tile,min(tile+width-1,n)").replace(
        "enddo\nenddo\n!$omp end", "enddo\nenddo\nenddo\nenddo\n!$omp end").replace(
        "private(i,work)", "private(i,j,tile,work)").replace("b(i)=work(2)", "b(i)=work(2)+real(j,8)")
    # This source is intentionally overlapping across rows. Ordinary dependence
    # analysis must reject it, rather than obtaining definition credit from a
    # perfectly tiled coordinate shape alone.
    _path, analysis, _provider, selection = fixture(tmp_path, source)
    with pytest.raises(CompilationError):
        prepare_original_region(analysis, ENTRY, selection, options=OPTIONS, config=CONFIG, environment_root=ENTRY)


def test_full_original_context_keeps_excluded_iterator_live_out(tmp_path):
    source = SOURCE.replace("end subroutine\nend module", "print *, i\nend subroutine\nend module")
    _path, analysis, provider, selection = fixture(tmp_path, source)
    with pytest.raises(CompilationError):
        provider.prepare(analysis, ENTRY, selection)
