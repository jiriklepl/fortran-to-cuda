"""Source-control coverage for the bounded owning native performance hint."""

import re
from hashlib import sha256
from unittest.mock import patch

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.native_owner_hint import latch, prepare
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_pristine_original_entry import emit, source
from compiler.tests.test_scoped_planning_entries import calibration


def test_hint_uses_original_descriptor_metadata_and_issued_static_identity(tmp_path):
    _original, _outputs, report, text, owner, child = emit(tmp_path, "combined")
    hint = report["scopes"][0]["native_owner_hint"]
    assert hint["eligible"]
    assert hint["execution_identity"] == owner.execution_plan.execution_identity
    assert not hint["estimate_available"]
    assert {item["resource"] for item in hint["entry_observations"]} == {"argument::a", "argument::b", "argument::c"}
    native = latch(owner)
    assert latch(child) is None  # Never reference the caller's local in another module entry.
    assert f"logical :: {native}" in text
    assert f"{native} = .false." in text
    assert "fort_scope_native_hint_begin_v1" in text
    # Each source CALL takes its exact original branch before any trace frame,
    # rearm, descriptor preflight or eager original-actual registration.
    call_branch = text.lower().index(f"if ({native}) then\ncall child")
    actuals = text.index("fort_actual_status = FORT_SCOPE_OK", call_branch)
    assert call_branch < text.index("fort_scope_trace_set_v1", call_branch) < actuals
    assert f"if ({native}) then\n{owner.enabled} = .false.\nelse" in text[call_branch:actuals]
    assert f".not. ({child.control_guard})" in text
    assert hint["metadata_words"] <= 480
    assert "descriptor bounds, extents and contiguity only" in str(hint["entry_observations"])


def test_root_early_return_declines_hint_without_changing_original_return(tmp_path):
    original = tmp_path / "return.f90"
    original.write_text(source("combined").replace("call child(a,b,c,n,flag)",
        "if(n<0)return\ncall child(a,b,c,n,flag)"))
    builder = ScopeBuilder([original], "original_entry_fixture::advance",
        facts={"schema_version": 1, "participation": "serial",
               "sources": {str(original): sha256(original.read_bytes()).hexdigest()},
               "captures": {"argument::" + name: dict(FACT) for name in ("a", "b", "c")}},
        options=CompilerOptions(opt_level=0), config=OffloadConfig("auto", calibration()))
    outputs, report = builder.run()
    public = report["scopes"][0]["native_owner_hint"]
    assert not public["eligible"]
    assert "normal completion" in public["reason"]
    text = outputs[report["sources"][str(original)]["replacement"]]
    assert "fort_scope_native_hint_begin_v1" not in text
    assert re.search(r"if\s*\(n\s*<\s*0\)\s*then", text, re.I)
    assert len(re.findall(r"\breturn\b", text, re.I)) == 1


def test_dormant_optional_payload_is_not_added_to_entry_hint_key(tmp_path):
    original = tmp_path / "optional.f90"
    raw = source("combined").replace("subroutine advance(a,b,c,n,flag)", "subroutine advance(a,b,c,n,flag,dormant)")
    raw = raw.replace("subroutine advance(a,b,c,n,flag,dormant)\n",
        "subroutine advance(a,b,c,n,flag,dormant)\nreal(8),optional,intent(in)::dormant(:)\n")
    original.write_text(raw)
    builder = ScopeBuilder([original], "original_entry_fixture::advance",
        facts={"schema_version": 1, "participation": "serial",
               "sources": {str(original): sha256(original.read_bytes()).hexdigest()},
               "captures": {"argument::" + name: dict(FACT) for name in ("a", "b", "c")}},
        options=CompilerOptions(opt_level=0), config=OffloadConfig("auto", calibration()))
    outputs, report = builder.run()
    public = report["scopes"][0]["native_owner_hint"]
    assert public["eligible"]
    assert not any(item["resource"] == "argument::dormant" for item in public["entry_observations"])
    text = outputs[report["sources"][str(original)]["replacement"]]
    assert "lbound(dormant" not in text.lower()
    assert "size(dormant" not in text.lower()


def test_assumed_size_last_dimension_is_not_observed_at_entry(tmp_path):
    original = tmp_path / "assumed_size.f90"
    raw = source("combined").replace("subroutine advance(a,b,c,n,flag)",
        "subroutine advance(a,b,c,n,flag,unknown)")
    raw = raw.replace("subroutine advance(a,b,c,n,flag,unknown)\n",
        "subroutine advance(a,b,c,n,flag,unknown)\nreal(8),intent(in)::unknown(2,*)\n")
    original.write_text(raw)
    builder = ScopeBuilder([original], "original_entry_fixture::advance",
        facts={"schema_version": 1, "participation": "serial",
               "sources": {str(original): sha256(original.read_bytes()).hexdigest()},
               "captures": {"argument::" + name: dict(FACT) for name in ("a", "b", "c")}},
        options=CompilerOptions(opt_level=0), config=OffloadConfig("auto", calibration()))
    outputs, report = builder.run()
    public = report["scopes"][0]["native_owner_hint"]
    assert public["eligible"]
    assert not any(item["resource"] == "argument::unknown" for item in public["entry_observations"])
    text = outputs[report["sources"][str(original)]["replacement"]]
    assert "lbound(unknown" not in text.lower()
    assert "size(unknown" not in text.lower()


def test_entry_observations_never_evaluate_a_deferred_index_expression(tmp_path):
    _original, _outputs, _report, _text, owner, _child = emit(tmp_path, "combined")
    original_visible = owner.builder.visible
    bindings = {root: owner.builder.visible_binding(owner.routine, root)
                for root in ("argument::a", "argument::b", "argument::c")}

    def expression(routine, root):
        return "a(undefined_index:)" if root == "argument::a" else original_visible(routine, root)

    with patch.object(owner.builder, "visible", expression), patch.object(owner.builder, "visible_binding",
            side_effect=lambda _routine, root: bindings[root]):
        hint, reason = prepare(owner)
    assert reason is None
    assert "argument::a" not in {item.binding.root for item in hint.descriptors}
    assert "undefined_index" not in str(hint.public())
