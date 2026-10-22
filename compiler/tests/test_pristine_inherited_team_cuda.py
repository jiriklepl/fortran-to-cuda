"""Original team before fork preserves an inherited mixed schedule and mirrors."""

import json
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_native_coefficient_cuda import binaries as coefficient_binaries
from compiler.tests.test_native_coefficient_cuda import (
    test_coefficient_cpu_team_preserves_future_gpu_choices_and_read_mirrors as check_fields,
)


@pytest.fixture(scope="module")
def pristine_binaries(tmp_path_factory):
    run = ScopeBuilder.run

    def with_stale_native_flag(builder):
        outputs, manifest = run(builder)
        assert any(query["workers"] == 3 for query in manifest["scopes"][0]["query_regions"])
        changed = []
        for item in manifest["build_sources"]:
            if item["role"] != "original_source":
                continue
            name = item["path"]
            text = outputs[name]
            assert "if (fort_team_query_plan) then" in text
            assert text.count("fort_team_pristine = .false.") == 3
            # A previous successful handoff must never authorize a later
            # mixed/applied-zero probe. Deliberately make the incoming local
            # flag stale immediately before each explicit preteam reset.
            text = text.replace("fort_team_pristine = .false.",
                "fort_team_native = .true.\nfort_team_pristine = .false.")
            outputs[name] = text
            changed.append({"path": name, "compiled_sha256": sha256(text.encode()).hexdigest(),
                            "issued_sha256": manifest["artifacts_sha256"][name]})
        assert changed
        return outputs, manifest

    with patch.object(ScopeBuilder, "run", with_stale_native_flag):
        cases = coefficient_binaries.__wrapped__(SimpleNamespace(param=(1, True)), tmp_path_factory)
    for case, _native, _candidate, manifest in cases.values():
        (case / "stale-flag-injection.json").write_text(json.dumps({
            "purpose": "correctness fault injection, never application performance evidence",
            "compiled_source": "fort_team_native=true before each explicit inherited probe reset",
            "issued_artifacts": manifest["artifacts_sha256"],
            "compiled_original_sources": {item["path"]: sha256((case / item["path"]).read_bytes()).hexdigest()
                for item in manifest["build_sources"] if item["role"] == "original_source"},
        }, indent=2))
    return cases


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
def test_inherited_pristine_team_after_prior_success_keeps_current_choices(pristine_binaries, disabled):
    # The shared fixture covers n=0/1/9, repeated calls, all-CPU versus a
    # GPU-producer/native-read/GPU-consumer chain, negative tile guards,
    # exact coefficient bounds, every field/halo byte and transfer receipts.
    check_fields(pristine_binaries, disabled)
