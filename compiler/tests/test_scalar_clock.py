"""A host clock observes ordered completion without publishing field payloads."""
import json
from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.scalar_clock import original_scalar_clock
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.tests.test_external_device_tasks_v2 import OPAQUE_SOURCE, emit

SOURCE = """module clock_fixture
implicit none
contains
subroutine advance(a)
real(8),intent(inout)::a(:)
integer(8)::first,last
integer(8),save::rate
call system_clock(count=first,count_rate=rate)
a=a+1
call system_clock(last)
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE):
    path = tmp_path / "clock.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    routine = analysis.routines["clock_fixture::advance"]
    node = next(node for node in walk(routine.execution) if type(node).__name__ == "Call_Stmt")
    return analysis, routine, node


def test_exact_original_clock_preserves_scalar_host_state_and_no_payload_authority(tmp_path):
    analysis, routine, node = fixture(tmp_path)
    proof = original_scalar_clock(analysis, routine.qualified, node)
    assert proof.validate(analysis, routine.qualified, node) is proof
    assert original_scalar_clock(analysis, routine.qualified, node) is proof
    assert set(proof.public()["outputs"]) == {"count", "count_rate"}
    assert proof.public()["payload_access_authorized"] is False
    summary = analysis.summarize(routine.qualified)
    assert summary["complete"] is True
    assert [item["kind"] for item in summary["operations"]].count("native_scalar_clock") == 2
    with pytest.raises(CompilationError, match="exact compiler-issued"):
        copy(proof).validate(analysis, routine.qualified, node)


def test_native_clock_keeps_original_reexported_kind_without_a_c_or_gpu_representation(tmp_path):
    source = ("module clock_kinds\nuse iso_fortran_env\nend module\n"
              "module clock_reexport\nuse clock_kinds\nend module\n" +
              SOURCE.replace("module clock_fixture", "module clock_fixture\nuse clock_reexport", 1)
              .replace("integer(8)", "integer(int64)"))
    analysis, routine, node = fixture(tmp_path, source)
    proof = original_scalar_clock(analysis, routine.qualified, node)
    assert proof.public()["outputs"]["count"] == json.loads(json.dumps(routine.scope.bindings["first"].public()))
    assert proof.public()["payload_access_authorized"] is False


@pytest.mark.parametrize("source", [
    SOURCE.replace("call system_clock(count=first,count_rate=rate)", "call system_clock(count=a(1))"),
    SOURCE.replace("integer(8)::first,last", "integer(8),target::first,last"),
    SOURCE.replace("integer(8)::first,last", "real(8)::first,last"),
    SOURCE.replace("count_rate=rate", "count_max=first"),
    SOURCE.replace("count_rate=rate", "unknown=rate"),
    SOURCE.replace("integer(8)::first,last", "integer(8)::first,last\nexternal system_clock"),
    SOURCE.replace("integer(8)::first,last", "integer(8)::first,last\ncommon /aliases/ first"),
    SOURCE.replace("subroutine advance(a)", "subroutine advance(a,first)").replace("integer(8)::first,last", "integer(8)::first,last"),
])
def test_unproved_storage_shadowing_presence_and_associations_fail_closed(tmp_path, source):
    analysis, routine, node = fixture(tmp_path, source)
    with pytest.raises(CompilationError, match="SYSTEM_CLOCK"):
        original_scalar_clock(analysis, routine.qualified, node)


def test_original_wrapper_clocks_do_not_cut_provider_lifetime_or_publish_payloads(tmp_path, monkeypatch):
    source = OPAQUE_SOURCE.replace("integer::i\n", "integer::i\ninteger(8)::t1,t2\n")
    source = source.replace("call original_transform(cache,b,a)",
                            "call system_clock(t1)\ncall original_transform(cache,b,a)\ncall system_clock(t2)")
    monkeypatch.setattr("compiler.tests.test_external_device_tasks_v2.OPAQUE_SOURCE", source)
    path, outputs, report = emit(tmp_path, child=True)
    owner, = report["scopes"]
    clock_lines = [index for index, line in enumerate(source.splitlines(), 1) if "call system_clock(" in line]
    boundaries = owner["boundaries"] + [boundary for child in owner["module_coordinators"]
                                       for boundary in child["boundaries"]]
    assert not any(boundary["first_line"] in clock_lines for boundary in boundaries)
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert text.count("call system_clock(t1)") == 1
    assert text.count("call system_clock(t2)") == 1
    assert "fort_scope_wait(" in text
    for name in ("t1", "t2"):
        position = text.index(f"call system_clock({name})")
        wait = text.rfind("fort_scope_wait(", 0, position)
        assert wait >= 0
        assert "fort_scope_resident_close_v1(" not in text[wait:position]
