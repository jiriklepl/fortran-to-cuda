"""Original source guards and ownership survive cheap fresh native decisions."""

import re
import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.native_preflight import source_native_preflight
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_compute_costs import profile_v2
from compiler.tests.test_source_scopes import FACT

SOURCE = """module renamed_gate
implicit none
contains
subroutine produce(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
a(i)=sqrt(a(i)*a(i)+1.0_8)+cos(a(i))
enddo
end subroutine
subroutine step(a,n,run)
real(8),allocatable,intent(inout)::a(:)
integer,intent(in)::n
logical,intent(in)::run
if(run) then
call produce(a,n)
endif
end subroutine
end module
"""


def builder_for(tmp_path, monkeypatch, source=SOURCE, *, mode="reached", profile=True,
                facts=None, numerical_sources=None):
    monkeypatch.setattr("compiler.emission.cuda.scoped.compiler_identity", lambda _: {})
    path = tmp_path / "original.f90"
    path.write_text(source)
    captures = facts or {"argument::a": FACT}
    builder = ScopeBuilder([path], "renamed_gate::step", options=CompilerOptions(),
        config=OffloadConfig(policy="auto", profile=profile_v2() if profile else None,
                             scope_execution=mode),
        facts={"schema_version": 1, "participation": "serial",
               "sources": {str(path): sha256(path.read_bytes()).hexdigest()}, "captures": captures},
        numerical_sources=numerical_sources)
    return path, builder


def emit(tmp_path, monkeypatch, source=SOURCE, **kwargs):
    path, builder = builder_for(tmp_path, monkeypatch, source, **kwargs)
    outputs, manifest = builder.run()
    text = outputs[manifest["sources"][str(path)]["replacement"]]
    return path, builder, outputs, manifest, text


def original_step(text):
    first = text.index("subroutine step(")
    return text[first:text.index("end subroutine", first)]


def reached_preflights(manifest):
    return [unit["native_preflight"] for owner in manifest["scopes"]
            for unit in owner.get("planning_segments", []) if unit.get("native_preflight")]


def inline_source(*, bounds="1,n", guarded=False):
    loop = f"do i={bounds}\na(i)=sqrt(a(i)*a(i)+1.0_8)+cos(a(i))\nenddo"
    if guarded:
        loop = "if(run) then\n" + loop + "\nendif"
    return SOURCE.replace("logical,intent(in)::run\n", "logical,intent(in)::run\ninteger::i\n").replace(
        "if(run) then\ncall produce(a,n)\nendif", loop)


@pytest.mark.parametrize("mode", ["bounded", "reached"])
def test_original_allocation_and_descriptor_guards_precede_metadata_and_context(tmp_path, monkeypatch, mode):
    _, _, _, manifest, text = emit(tmp_path, monkeypatch, inline_source(bounds="1,size(a)"), mode=mode)
    owner, = manifest["scopes"]
    proof = owner.get("native_preflight") if mode == "bounded" else reached_preflights(manifest)[0]
    assert proof["available"]
    assert proof["proofs"] == ["empty domain", "outside validated compute item range"]
    assert not proof["payload_reads"]
    assert proof["contexts_created"] == proof["registrations"] == 0
    assert any("size(" in argument for argument in proof["source_arguments"])
    if mode == "bounded":
        caller = original_step(text)
        assert caller.index("allocated(a)") < caller.index("lbound(a, 1, kind=8)")
        assert caller.index("lbound(a, 1, kind=8)") < caller.index("call " + owner["owner"])
        body = text[text.index("subroutine " + owner["owner"]):]
    else:
        body = original_step(text)
        assert body.index("allocated(a)") < body.index("lbound(a, 1, kind=8)")
        assert body.index("lbound(a, 1, kind=8)") < body.index("if (fort_native_preflight_")
    assert body.index("is_contiguous(") < body.index("if (fort_native_preflight_")
    assert body.index("if (fort_native_preflight_") < body.index("fort_scope_create(")
    assert body.index("fort_scope_create(") < body.index("fort_scope_register(")


def test_reached_original_action_guard_precedes_every_descriptor_and_native_decision(tmp_path, monkeypatch):
    _, _, _, manifest, text = emit(tmp_path, monkeypatch)
    proof, = reached_preflights(manifest)
    assert proof["available"]
    assert proof["source_arguments"] == ["n"]
    caller = original_step(text)
    assert caller.index("if(run) then") < caller.index("allocated(a)")
    assert caller.index("allocated(a)") < caller.index("if (fort_native_preflight_")
    assert caller.index("if (fort_native_preflight_") < caller.index("fort_scope_create(")
    assert caller.count("call produce(a,n)") == 1


@pytest.mark.parametrize("kind", ["module_procedure", "use_procedure", "generic"])
def test_metadata_native_flag_preserves_original_procedure_namespace(tmp_path, monkeypatch, kind):
    helper_name = "renamed_probe" if kind == "generic" else "fort_metadata_native_0"
    helper = f"pure real(8) function {helper_name}() result(y)\ny=0.0_8\nend function\n"
    source = inline_source()
    if kind == "use_procedure":
        source = ("module metadata_names\ncontains\n" + helper + "end module\n" +
            source.replace("module renamed_gate\n", "module renamed_gate\n"
                           "use metadata_names, only: fort_metadata_native_0\n"))
    else:
        source = source.replace("end module", helper + "end module")
        if kind == "generic":
            source = source.replace("implicit none\n", "implicit none\ninterface fort_metadata_native_0\n"
                                    "module procedure renamed_probe\nend interface\n", 1)
    _, _, _, manifest, text = emit(tmp_path, monkeypatch, source)
    proof, = reached_preflights(manifest)
    assert not proof["available"]
    assert proof["reason"] == "metadata flag conflicts with original source"
    caller = original_step(text)
    assert "logical :: fort_metadata_native_0" not in caller
    assert "if (fort_native_preflight_" not in caller
    assert "fort_scope_create(" in caller


def test_fresh_native_flag_keeps_later_units_enabled_and_skips_shortcut_on_continuation(tmp_path, monkeypatch):
    source = SOURCE.replace("integer,intent(in)::n\nlogical,intent(in)::run", "integer,intent(in)::n,m\nlogical,intent(in)::run,later")
    source = source.replace("subroutine step(a,n,run)", "subroutine step(a,n,m,run,later)")
    source = source.replace("call produce(a,n)\nendif", "call produce(a,n)\nendif\nif(later) then\ncall produce(a,m)\nendif")
    _, _, _, manifest, text = emit(tmp_path, monkeypatch, source)
    first, second = reached_preflights(manifest)
    assert first["source_arguments"] == ["n"]
    assert second["source_arguments"] == ["m"]
    caller = original_step(text)
    decisions = list(re.finditer(r"if \(fort_context == 0\) then\nif \(fort_native_preflight_[^\n]+\) then\n"
                               r"fort_metadata_native_(\d+) = \.true\.\nexit fort_reached_\d+\nendif\nendif", caller))
    assert len(decisions) == 2
    enabled = manifest["scopes"][0]["owner"].replace("fort_owner_context_", "fort_owner_enabled_")
    for decision in decisions:
        # The small/empty decision executes this original action and leaves the
        # owning invocation available to a later profitable reached segment.
        assert enabled not in decision.group()
        flag = "fort_metadata_native_" + decision.group(1)
        assert "logical :: " + flag in caller
        assert flag + " = .false." in caller
        assert f"if (.not. {enabled} .or. {flag}) then" in caller
    assert caller.index("if(later) then") < decisions[1].start()
    assert caller.count("call produce(a,n)") == caller.count("call produce(a,m)") == 1


def test_original_numerical_guard_does_not_become_an_eager_metadata_read(tmp_path, monkeypatch):
    source = SOURCE.replace("do i=1,n", "if(n>0)then\ndo i=1,n").replace("enddo\nend subroutine", "enddo\nendif\nend subroutine")
    _, _, _, manifest, text = emit(tmp_path, monkeypatch, source)
    proof, = reached_preflights(manifest)
    assert not proof["available"]
    assert "unconditional numerical unit" in proof["reason"]
    assert "fort_native_preflight_" not in original_step(text)
    assert "if(n>0)then" in text


def test_payload_bound_keeps_original_native_source_without_preflight(tmp_path, monkeypatch):
    source = SOURCE.replace("do i=1,n", "do i=1,int(a(1))")
    path, _, _, manifest, text = emit(tmp_path, monkeypatch, source)
    assert original_step(text) == original_step(path.read_text())
    assert not reached_preflights(manifest)
    assert "fort_native_preflight_" not in text
    assert "fort_scope_create(" not in text


def test_scalar_expression_actual_is_not_evaluated_by_metadata_preflight(tmp_path, monkeypatch):
    source = SOURCE.replace("call produce(a,n)", "call produce(a,n+1)")
    path, builder = builder_for(tmp_path, monkeypatch, source)
    outputs, manifest = builder.run()
    assert not manifest["scopes"]
    assert any("scalar-expression actual" in item["reason"] for item in manifest["boundaries"])
    assert not any("fort_native_preflight_" in text for text in outputs.values())
    assert path.read_text() == source


def test_static_all_native_has_no_source_guard_or_owner_side_effects(tmp_path, monkeypatch):
    path, _, _, manifest, text = emit(tmp_path, monkeypatch, profile=False)
    assert text == path.read_text() == SOURCE
    owner, = manifest["scopes"]
    assert owner["automatic_preflight"]["successful"]
    assert owner["automatic_preflight"]["runtime_decision_inputs"] == []
    assert not owner["automatic_preflight"]["caller_guards_evaluated"]
    assert "fort_native_preflight_" not in text
    assert "fort_scope_create(" not in text
    assert manifest["implementation_variants"]["generated_count"] == 0


@pytest.mark.parametrize("joined", [False, True])
def test_borrowed_root_view_keeps_original_native_compute_participation(tmp_path, monkeypatch, joined):
    builder, _ = normalized_bounds_builder(tmp_path, monkeypatch, joined=joined)
    full = builder.numerical("renamed_gate::produce").scoped
    borrowed, _ = builder.entry_artifacts("renamed_gate::produce", views=True)
    expected = "fork_join" if joined else "serial"
    assert full["compute_estimates"]["native_participation"] == expected
    assert borrowed["compute_estimates"]["native_participation"] == expected
    full_unit, = full["planning"]["units"]
    borrowed_unit, = borrowed["planning"]["units"]
    assert borrowed_unit["compute_model"] == full_unit["compute_model"]
    assert borrowed_unit["compute_model"]["native_fortran"]["backend_identity"] == "native_" + expected


def normalized_bounds_builder(tmp_path, monkeypatch, *, joined=False):
    original = """module saved_fields
real(8),allocatable,target::hidden(:)
end module
module renamed_gate
use saved_fields
implicit none
contains
subroutine produce(n)
integer,intent(in)::n
integer::i
do i=lbound(hidden,1),lbound(hidden,1)+n-1
hidden(i)=sqrt(hidden(i)*hidden(i)+1.0_8)+cos(hidden(i))
enddo
end subroutine
subroutine step(n)
integer,intent(in)::n
call produce(n)
end subroutine
end module
"""
    normalized = """module normalized_gate
contains
subroutine evaluate(hidden,n,origin)
real(8),intent(inout)::hidden(:)
integer,intent(in)::n,origin
integer::i
do i=origin,origin+n-1
hidden(i-origin+1)=sqrt(hidden(i-origin+1)*hidden(i-origin+1)+1.0_8)+cos(hidden(i-origin+1))
enddo
end subroutine
end module
"""
    if joined:
        original = original.replace("do i=lbound(hidden,1)", "!$omp parallel do private(i) schedule(static)\ndo i=lbound(hidden,1)")
        original = original.replace("enddo\nend subroutine", "enddo\n!$omp end parallel do\nend subroutine")
    original_path, normalized_path = tmp_path / "original.f90", tmp_path / "normalized.f90"
    original_path.write_text(original)
    normalized_path.write_text(normalized)
    source_hash = sha256(original_path.read_bytes()).hexdigest()
    package = {"schema_version": 1, "source_inputs": {str(original_path): source_hash}, "entries": [{
        "procedure": "renamed_gate::produce", "source_sha256": source_hash,
        "path": str(normalized_path), "sha256": sha256(normalized_path.read_bytes()).hexdigest(),
        "entry": "normalized_gate::evaluate", "normalization": "whole_storage_rebased_v1",
        "participation": "serial_coordinator", "capture_safe": True, "preserves_source_order": True,
        "parameters": [{"name": "hidden", "resource": "saved_fields::hidden", "physical_origin": [0]},
                       {"name": "n", "resource": "argument::n"},
                       {"name": "origin", "resource": "saved_fields::hidden", "lower_bound_dimension": 1}]}]}
    _, builder = builder_for(tmp_path, monkeypatch, original, facts={"saved_fields::hidden": FACT}, numerical_sources=package)
    call, = [builder.resolve(builder.entry, node) for node in builder.entry.execution.content]
    return builder, call


def test_synthetic_assumed_shape_rebasing_cannot_supply_original_hidden_lower_bounds(tmp_path, monkeypatch):
    builder, call = normalized_bounds_builder(tmp_path, monkeypatch)
    values = {"saved_fields::hidden": "synthetic_hidden", "argument::n": "n"}
    declined = source_native_preflight(builder, [call], {call.procedure}, values)
    assert declined.expression is None
    assert not declined.imports
    assert not declined.public["available"]
    assert "synthetic dummy rebasing" in declined.public["reason"]
    assert not builder.outputs


def test_original_descriptor_bound_mapping_uses_checked_integer_conversion(tmp_path, monkeypatch):
    builder, call = normalized_bounds_builder(tmp_path, monkeypatch)
    proof = source_native_preflight(builder, [call], {call.procedure},
        {"saved_fields::hidden": "hidden", "argument::n": "n"}, original_descriptors=True)
    assert proof.public["available"]
    assert proof.public["source_arguments"] == ["n", "int(lbound(hidden, 1, kind=c_int64_t), kind=c_int)"]
    assert "synthetic" not in proof.expression


@pytest.mark.parametrize("mode", ["bounded", "reached"])
def test_guarded_preflight_source_compiles_with_native_fortran_only(tmp_path, monkeypatch, mode):
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fc:
        pytest.skip("Fortran compiler unavailable")
    source = inline_source(bounds="lbound(a,1),ubound(a,1)", guarded=True)
    _, _, outputs, manifest, _ = emit(tmp_path, monkeypatch, source, mode=mode)
    assert manifest["scopes"], manifest["boundaries"]
    build = tmp_path / "compile"
    build.mkdir()
    for name, content in outputs.items():
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in manifest["build_sources"]:
            if item["role"] == role and item["language"] == "fortran":
                path = build / item["path"]
                result = subprocess.run([fc, "-std=f2018", "-fopenmp", "-fcheck=all", "-c", str(path),
                                         "-o", str(path.with_suffix(".o"))], cwd=build,
                                        capture_output=True, text=True, timeout=30)
                assert result.returncode == 0, result.stdout + result.stderr


def test_intent_out_entry_keeps_definition_registration_before_native_placement(tmp_path, monkeypatch):
    source = SOURCE.replace("real(8),intent(inout)::a(:)", "real(8),intent(out)::a(:)")
    source = source.replace("a(i)=sqrt(a(i)*a(i)+1.0_8)+cos(a(i))", "a(i)=sqrt(real(i,8)*real(i,8)+1.0_8)+cos(real(i,8))")
    _, builder, _, manifest, text = emit(tmp_path, monkeypatch, source)
    # The numerical metadata API itself is valid, but taking it before the
    # owner records INTENT(OUT) would lose the original definition event.
    assert builder.numerical("renamed_gate::produce").scoped["native_preflight"]["available"]
    proof, = reached_preflights(manifest)
    assert not proof["available"]
    assert "ordered definition changes" in proof["reason"]
    caller = original_step(text)
    assert "fort_native_preflight_" not in caller
    assert "fort_scope_register(" in caller
    unit, = manifest["scopes"][0]["planning_segments"]
    assert unit["ordered_definitions"]["possible_definition_changes"] == ["argument::a"]
    assert "fort_scope_plan_validate(" in caller


@pytest.mark.parametrize("attribute", ["volatile", "asynchronous", "optional", "pointer", "allocatable"])
def test_observable_or_guarded_scalar_storage_has_no_cheap_read(tmp_path, monkeypatch, attribute):
    from fparser.two.utils import walk

    source = SOURCE.replace("subroutine step(a,n,run)\nreal(8),allocatable,intent(inout)::a(:)\ninteger,intent(in)::n",
                            "subroutine step(a,n,run)\nreal(8),allocatable,intent(inout)::a(:)\n"
                            f"integer,{attribute},intent(in)::n")
    _, builder = builder_for(tmp_path, monkeypatch, source)
    node, = [node for node in walk(builder.entry.execution) if type(node).__name__ == "Call_Stmt"]
    if attribute == "optional":
        from compiler.ir import CompilationError
        with pytest.raises(CompilationError, match="optional actual requires an optional callee formal"):
            builder.resolve(builder.entry, node)
        assert not builder.outputs
        return
    call = builder.resolve(builder.entry, node)
    proof = source_native_preflight(builder, [call], {call.procedure},
                                    {"argument::a": "a", "argument::n": "n"}, original_descriptors=True)
    assert not proof.public["available"]
    assert "observable or guarded storage" in proof.public["reason"]
    assert proof.expression is None
    assert not proof.imports
    assert not builder.outputs


def test_straight_normalized_owner_declines_rebased_hidden_origin_shortcut(tmp_path, monkeypatch):
    builder, call = normalized_bounds_builder(tmp_path, monkeypatch)
    owner = builder.owner([call], {call.procedure})
    proof = owner["native_preflight"]
    assert not proof["available"]
    assert "synthetic dummy rebasing" in proof["reason"]
    builder.scopes.append(owner)
    outputs, manifest = builder.finish()
    text = outputs[manifest["sources"][str(builder.entry.scope.path)]["replacement"]]
    caller = original_step(text)
    assert caller.index("allocated(hidden)") < caller.index("lbound(hidden, 1, kind=8)")
    assert caller.index("lbound(hidden, 1, kind=8)") < caller.index("call " + owner["owner"])
    body = text[text.index("subroutine " + owner["owner"]):]
    assert "fort_native_preflight_" not in body
    assert "fort_scope_create(" in body
    assert "fort_scope_plan_validate(" in body
