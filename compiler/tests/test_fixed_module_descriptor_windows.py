"""Fixed original module storage is independent of payload/entry authority."""
from copy import copy

import pytest
from fparser.two import Fortran2003 as F

from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

ENTRY = "driver::advance"
ROOT = "storage::backing"


def fixture(tmp_path, *, dtype="integer(4)", rank=1, attributes="", shape=None,
            module_extra="", body=None, helper_extra="", sealed=False,
            prefix="", imports="", parameters="integer,parameter::width=8", helper_type=None,
            coarray="", ordinary=False):
    shape = shape or ",".join(["-2:width-3"] * rank)
    attributes = "," + attributes if attributes else ""
    indices = ",".join(["1"] * rank)
    body = body or f"value=local({indices})"
    path = tmp_path / "fixed.f90"
    path.write_text(f"""{prefix}module storage
{imports}
implicit none
{parameters}
{dtype}{attributes}::backing({shape}){coarray}
{module_extra}
contains
subroutine touch()
backing=1
end subroutine
end module
module forwarding
use storage,only:forwarded=>backing
end module
module helper
use storage,only:local=>backing,touch
implicit none
contains
subroutine coordinate(n)
integer,intent(in)::n
{helper_type or dtype}::value
{body}
end subroutine
{helper_extra}
end module
module driver
use forwarding,only:slot=>forwarded
use helper,only:coordinate
implicit none
contains
subroutine advance({"a,n" if ordinary else "n"})
integer,intent(in)::n
{"real(8),intent(inout)::a(:)" if ordinary else ""}
call unknown_prefix()
call coordinate(n)
call unknown_tail()
end subroutine
end module
""")
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    graph = analysis.structure(ENTRY)
    call = next(node for node in analysis.routines[ENTRY].execution.content
                if str(node).lower().startswith("call coordinate"))
    return path, analysis, (graph.node_id(call),)


def prove(analysis, selected):
    return prove_descriptor_window(analysis, ENTRY, selected, module_roots=(ROOT,))


@pytest.mark.parametrize("dtype", ["integer(4)", "integer(8)", "real(4)", "real(8)"])
@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_exact_fixed_numeric_storage_is_not_an_allocation_or_definition(tmp_path, dtype, rank):
    path, analysis, selected = fixture(tmp_path, dtype=dtype, rank=rank)
    original = path.read_bytes()
    proof = prove(analysis, selected)
    public = proof.public()
    state = proof.resource_state(ROOT)
    assert public["schema_version"] == 11
    assert state["resource_kind"] == "original_module_fixed_array"
    assert state["original_visible_spelling"] == "slot"
    assert state["fixed_bounds"] == [[-2, 5]] * rank
    assert state["original_binding"]["rank"] == rank
    assert state["stable"]
    assert not state["descriptor_changes_within_selection"]
    assert not state["escapes_within_selection"]
    assert state["requires_prior_noescape_proof"]
    assert state["requires_original_alias_guards"]
    assert state["execution_requires_runtime_guard"]
    assert not state["entry_definition_authority"]
    assert not state["payload_access_authorized"]
    assert not state["alias_authority"]
    assert not public["prefix_effects_proved"]
    assert not public["execution_authorized"]
    assert not public["gpu_legality_established"]
    assert public["requires_close_before_excluded_tail"]
    assert analysis.stable_module_allocatables == frozenset()
    assert analysis._allocation_authorizations == {}
    assert proof is prove(analysis, selected)
    assert path.read_bytes() == original


@pytest.mark.parametrize("sealed", [False, True])
def test_tiling_shaped_protected_read_preserves_mutable_source_semantics(tmp_path, sealed):
    _path, analysis, selected = fixture(tmp_path, attributes="protected", sealed=sealed)
    proof = prove(analysis, selected)
    state = proof.resource_state(ROOT)
    assert "protected" in state["original_binding"]["attributes"]
    assert not state["payload_access_authorized"]
    private, _, _ = analysis._segment_projection(ENTRY, selected, descriptor_window=proof)
    assert not private.window_module_allocatables(ENTRY)
    assert not private.window_module_allocatables("helper::coordinate")
    summary = private.summarize("helper::coordinate")
    assert summary["complete"], summary["reasons"]
    assert any(op.get("resource") == ROOT and op["kind"] == "read" for op in summary["operations"])
    assert not analysis.window_module_allocatables(ENTRY)


def test_original_imported_kind_and_constant_bounds_are_authenticated(tmp_path):
    prefix = """module sizes
implicit none
integer,parameter::extent=8,precision=8
end module
"""
    _path, analysis, selected = fixture(tmp_path, dtype="real(rk)", prefix=prefix,
        imports="use sizes,only:width=>extent,rk=>precision", parameters="", helper_type="real(8)")
    proof = prove(analysis, selected)
    state = proof.resource_state(ROOT)
    assert state["original_binding"]["kind"] == 8
    assert state["fixed_bounds"] == [[-2, 5]]
    analysis.modules["sizes"].kinds.values["extent"] = F.Int_Literal_Constant("9")
    with pytest.raises(CompilationError):
        proof.validate(analysis, ENTRY, selected)


@pytest.mark.parametrize("fault", ["shape", "kind_environment", "binding"])
def test_consumed_fixed_declaration_cannot_change_at_final_publication(tmp_path, monkeypatch, fault):
    from compiler.frontend.descriptor_windows import _Check

    _path, analysis, selected = fixture(tmp_path)
    module = analysis.modules["storage"]
    binding = module.bindings["backing"]
    original = _Check.publish
    changed = []

    def mutate(check, procedure, authority):
        if check.fixed_module_scopes:
            if fault == "shape":
                binding.shape_nodes = tuple(copy(axis) for axis in binding.shape_nodes)
            elif fault == "kind_environment":
                module.kinds.values["width"] = F.Int_Literal_Constant("9")
            else:
                module.bindings["backing"] = copy(binding)
            changed.append(fault)
        return original(check, procedure, authority)

    monkeypatch.setattr(_Check, "publish", mutate)
    with pytest.raises(CompilationError):
        prove(analysis, selected)
    assert changed
    assert not getattr(analysis, "_descriptor_window_proofs", {})


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("stage", ["resolution_scope", "resolution_tick", "publication"])
def test_original_fixed_axes_cannot_be_rebased_by_callbacks(tmp_path, monkeypatch, sealed, stage):
    from compiler.frontend import descriptor_windows as windows

    _path, analysis, selected = fixture(tmp_path, sealed=sealed)
    binding = analysis.modules["storage"].bindings["backing"]
    changed = []

    def mutate():
        if not changed:
            changed.append(stage)
            # Text and declaration values stay equal. Unsealed input must
            # retain the initially consumed objects; sealed input must reject
            # mutation before it can replace that original authority.
            binding.shape_nodes = tuple(copy(axis) for axis in binding.shape_nodes)

    if stage == "publication":
        original = windows._Check.publish

        def publish(check, procedure, authority):
            mutate()
            return original(check, procedure, authority)

        monkeypatch.setattr(windows._Check, "publish", publish)
    else:
        original = windows._validate_resolution

        def resolution(*args, **kwargs):
            validator = kwargs.get("_scope_validator")
            if getattr(validator, "__qualname__", "") == "_fixed_module_bounds.<locals>.scope_authority":
                key = "_scope_validator" if stage == "resolution_scope" else "tick"
                callback = kwargs[key]

                def during(*values):
                    callback(*values)
                    mutate()

                kwargs[key] = during
            return original(*args, **kwargs)

        monkeypatch.setattr(windows, "_validate_resolution", resolution)
    with pytest.raises(CompilationError):
        prove(analysis, selected)
    assert changed == [stage]
    assert not getattr(analysis, "_descriptor_window_proofs", {})


def test_complete_original_module_payload_writer_is_storage_preserving(tmp_path):
    _path, analysis, selected = fixture(tmp_path, body="call touch()")
    proof = prove(analysis, selected)
    assert proof.resource_state(ROOT)["stable"]
    assert "storage::touch" in proof.public()["source_call_closure"]
    private, _, _ = analysis._segment_projection(ENTRY, selected, descriptor_window=proof)
    summary = private.summarize("storage::touch")
    assert summary["complete"], summary["reasons"]
    assert any(op.get("resource") == ROOT and op["kind"] in {"write", "overwrite"}
               for op in summary["operations"])
    assert not proof.public()["entry_definition_authority"]


@pytest.mark.parametrize("attributes", ["target", "pointer", "allocatable", "volatile", "asynchronous", "optional"])
def test_unsafe_or_nonfixed_storage_is_not_reclassified_as_fixed(tmp_path, attributes):
    _path, analysis, selected = fixture(tmp_path, attributes=attributes, shape=":" if attributes in {"pointer", "allocatable"} else "8")
    if attributes == "allocatable":
        # The established allocatable variant remains its original schema/kind.
        proof = prove(analysis, selected)
        assert proof.public()["schema_version"] == 6
        assert proof.resource_state(ROOT)["resource_kind"] == "original_module_allocatable"
    else:
        with pytest.raises(CompilationError):
            prove(analysis, selected)


@pytest.mark.parametrize("extra", [
    "common /arena/ backing", "integer::other(8)\nequivalence(backing,other)",
    "!$omp threadprivate(backing)",
])
def test_declared_alias_or_thread_storage_requires_separate_authority(tmp_path, extra):
    _path, analysis, selected = fixture(tmp_path, module_extra=extra)
    with pytest.raises(CompilationError):
        prove(analysis, selected)


@pytest.mark.parametrize("body", [
    "call unresolved(local)", "call hidden()",
    "deallocate(local)", "allocate(local(8))", "call move_alloc(local,other)",
])
def test_full_selected_closure_retains_unknown_and_lifetime_effects(tmp_path, body):
    _path, analysis, selected = fixture(tmp_path, body=body)
    with pytest.raises(CompilationError):
        prove(analysis, selected)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


def test_hidden_module_allocation_tail_is_not_omitted(tmp_path):
    extra = """subroutine hidden()
use other_storage,only:other
if(allocated(other)) deallocate(other)
end subroutine"""
    prefix = """module other_storage
implicit none
real(8),allocatable::other(:)
end module
"""
    _path, analysis, selected = fixture(tmp_path, body="call hidden()", helper_extra=extra, prefix=prefix)
    with pytest.raises(CompilationError):
        prove(analysis, selected)


@pytest.mark.parametrize("options", [
    {"rank": 5}, {"shape": ":"}, {"shape": "*"}, {"coarray": "[*]"},
    {"parameters": "integer::width=8"},
])
def test_rank_shape_or_nonconstant_declaration_is_outside_the_fixed_slice(tmp_path, options):
    def attempt():
        _path, analysis, selected = fixture(tmp_path, **options)
        prove(analysis, selected)

    with pytest.raises(CompilationError):
        attempt()


def test_mixed_borrowed_and_fixed_resources_choose_version11_without_widening_formal(tmp_path):
    _path, analysis, selected = fixture(tmp_path, ordinary=True)
    proof = prove_descriptor_window(analysis, ENTRY, selected, ("argument::a",), module_roots=(ROOT,))
    assert proof.public()["schema_version"] == 11
    state = proof.resource_state("argument::a")
    assert state["resource_kind"] == "original_ordinary_array_formal"
    assert state["borrowed_descriptor"]
    assert not state["caller_storage_lifetime_authority"]
    assert not state["physical_allocation_origin_authority"]


def test_mixed_clock_and_fixed_resources_choose_highest_used_version12(tmp_path):
    _path, analysis, selected = fixture(tmp_path, body="call system_clock(value)")
    proof = prove(analysis, selected)
    assert proof.public()["schema_version"] == 12
    assert proof.resource_state(ROOT)["resource_kind"] == "original_module_fixed_array"


@pytest.mark.parametrize("dtype", ["complex(8)", "logical(4)", "real(16)"])
def test_nonfirst_slice_type_or_kind_rejects(tmp_path, dtype):
    _path, analysis, selected = fixture(tmp_path, dtype=dtype)
    with pytest.raises(CompilationError):
        prove(analysis, selected)


@pytest.mark.parametrize("fault", ["binding", "shape", "kind_environment", "import", "source", "session"])
def test_registered_fixed_storage_reauthenticates_all_consumed_authority(tmp_path, fault):
    path, analysis, selected = fixture(tmp_path)
    proof = prove(analysis, selected)
    module = analysis.modules["storage"]
    binding = module.bindings["backing"]
    if fault == "binding":
        module.bindings["backing"] = copy(binding)
    elif fault == "shape":
        binding.shape_nodes = tuple(copy(axis) for axis in binding.shape_nodes)
    elif fault == "kind_environment":
        module.kinds.values["width"] = F.Int_Literal_Constant("9")
    elif fault == "import":
        analysis.modules["forwarding"].imports["forwarded"] = ("storage", "other")
    elif fault == "source":
        path.write_text(path.read_text() + "! changed original\n")
    else:
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        _path, other, _selected = fixture(foreign, sealed=True)
        original_session = getattr(analysis, "_source_session", None)
        with pytest.raises(CompilationError, match="original registry changed"):
            analysis._source_session = other._source_session
        assert getattr(analysis, "_source_session", None) is original_session
        assert proof.validate(analysis, ENTRY, selected) is proof
        # Also exercise the consumer fence after deliberate internal corruption.
        object.__setattr__(analysis, "_source_session", other._source_session)
    with pytest.raises(CompilationError):
        proof.validate(analysis, ENTRY, selected)
