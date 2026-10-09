"""Source scopes validate definition order before performing numerical work."""

from hashlib import sha256

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import form_source_scopes
from compiler.tests.test_source_scopes import FACT, PROGRAM


def sections(tmp_path, source=PROGRAM):
    path = tmp_path / "original.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::a": FACT,
                          "argument::b": {**FACT, "initialized": "none"},
                          "argument::out": {**FACT, "initialized": "none"}}}
    return form_source_scopes([path], "step", facts=facts,
                              options=CompilerOptions(fallback="host", gpu_policy="sections", memory_model="scoped"),
                              config=OffloadConfig("sections"))


def test_forced_sections_validate_complete_query_without_cost_profile(tmp_path):
    outputs, report = sections(tmp_path)
    scope, = report["scopes"]
    assert scope["definition_preflight"]["query_available"]
    assert not scope["estimate_available"]
    text = next(value for name, value in outputs.items() if name.startswith("sources/"))
    owner = text.split("subroutine fort_scope_owner_", 1)[1].split("end subroutine", 1)[0]
    assert owner.index("fort_scope_plan_reset(") < owner.index("call fort_scope_query_")
    assert owner.index("call fort_scope_query_") < owner.index("fort_scope_plan_validate(")
    assert owner.index("fort_scope_plan_validate(") < owner.index("call fort_scope_clone_")
    after_validation = owner.split("fort_scope_plan_validate(", 1)[1]
    cleanup = after_validation.split("call fort_scope_clone_", 1)[0]
    assert "if (fort_status /= FORT_SCOPE_OK) then" in cleanup
    assert "fort_scope_close(" in cleanup
    assert "call producer(" in cleanup
    assert "call transform(" in cleanup
    assert "call consumer(" in cleanup
    assert "return" in cleanup
    assert "fort_choose(" not in owner


def test_mutable_future_bounds_are_validated_after_the_original_adjustment(tmp_path):
    adjust = "subroutine adjust(n)\ninteger,intent(inout)::n\nn=n-1\nend subroutine\n"
    source = PROGRAM.replace("end module", adjust + "end module")
    before, owner = source.split("subroutine step(", 1)
    source = before + "subroutine step(" + owner.replace("integer,intent(in)::n", "integer,intent(inout)::n", 1)
    source = source.replace("call transform(b)", "call adjust(n)")
    outputs, report = sections(tmp_path, source)
    scope, = report["scopes"]
    assert scope["definition_preflight"]["query_available"]
    assert scope["definition_preflight"]["position"] == "when segment is reached"
    assert [segment["calls"] for segment in scope["planning_segments"]] == [
        ["original::producer"], ["original::adjust"], ["original::consumer"]]
    text = next(value for name, value in outputs.items() if name.startswith("sources/"))
    owner = text.split("subroutine fort_scope_owner_", 1)[1].split("end subroutine", 1)[0]
    adjustment = owner.rfind("call adjust(")
    query = owner.index("fort_scope_query_", adjustment)
    validation = owner.index("fort_scope_plan_validate(", query)
    numerical = owner.index("call fort_scope_clone_", validation)
    assert adjustment >= 0
    assert adjustment < query < validation < numerical
    assert "FORT_SCOPE_PLAN_CONTINUE" in owner
