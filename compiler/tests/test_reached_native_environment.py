"""Reached IEEE calls keep original typed storage and original callers."""

import pytest

from compiler.tests.test_lexical_source_owner import SOURCE, emit


def environment_source(*, team):
    before = "call ieee_get_halting_mode(ieee_all,modes)\ncall ieee_set_halting_mode(ieee_all,.false.)"
    after = "call ieee_set_halting_mode(ieee_all,modes)"
    if team:
        before = "!$omp parallel shared(modes)\n" + before + "\n!$omp end parallel"
        after = "!$omp parallel shared(modes)\n" + after + "\n!$omp end parallel"
    return (SOURCE.replace("subroutine step(a,b,out,n,escape)",
                           "subroutine step(a,b,out,n,escape)\nuse ieee_exceptions")
        .replace("integer::i", "integer::i\nlogical::modes(size(ieee_all))")
        .replace("if(escape) then\ncall opaque(b,n)\nendif", before+"\n"+after))


@pytest.mark.parametrize("team", [False, True])
def test_environment_operations_do_not_close_owner_or_capture_logical_state(tmp_path, team):
    source = environment_source(team=team)
    path, outputs, report = emit(tmp_path, source)
    owner, = report["scopes"]
    assert not owner["boundaries"]
    assert len(owner["gpu_leaves"]) == 2
    operations = [operation for segment in owner["planning_segments"]
                  for operation in segment["operations"]["native_operations"]]
    environment = [operation for operation in operations if operation["native_environment"]]
    assert len(environment) == (2 if team else 3)
    assert sum(len(operation["native_environment"]) for operation in environment) == 3
    assert all(not operation["managed_resources"] for operation in environment)
    assert all(operation["preserves_original_source"] for operation in environment)
    assert all(state["type"] == "logical" and not state["device_capture"]
               for operation in environment for state in operation["native_local_state"])
    assert all("modes" not in resource["resource"] for resource in owner["resource_bindings"]["resources"])
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert "logical::modes(size(ieee_all))" in text
    assert "c_loc(modes" not in text.lower()
    assert "target :: modes" not in text.lower()
    assert "c_bool" not in text.lower()
    for call in ("call ieee_get_halting_mode(ieee_all,modes)",
                 "call ieee_set_halting_mode(ieee_all,.false.)",
                 "call ieee_set_halting_mode(ieee_all,modes)"):
        assert text.count(call) == 1
    assert text.count("!$omp parallel shared(modes)") == (2 if team else 0)
    for operation in environment:
        first = operation["first_line"]
        original = path.read_text().splitlines()[first-1]
        position = text.index(original)
        assert "fort_scope_wait(fort_context)" in text[:position]


def test_automatic_environment_path_declines_unpriced_coordination_without_instrumentation(tmp_path):
    path, outputs, report = emit(tmp_path, environment_source(team=True), policy="auto")
    owner, = report["scopes"]
    assert owner["automatic_preflight"]["successful"]
    assert "IEEE environment coordination" in owner["planning_reason"]
    assert outputs[report["sources"][str(path)]["replacement"]] == path.read_text()


@pytest.mark.parametrize("use", ["call opaque(modes,n)", "if(any(modes)) visits=visits+1"])
def test_other_native_state_use_closes_before_environment_call(tmp_path, use):
    source = environment_source(team=False).replace("call ieee_set_halting_mode(ieee_all,modes)", use)
    _, _, report = emit(tmp_path, source)
    assert report["scopes"][0]["boundaries"]
    assert any("source use or escape" in boundary["reason"] for boundary in report["scopes"][0]["boundaries"])
