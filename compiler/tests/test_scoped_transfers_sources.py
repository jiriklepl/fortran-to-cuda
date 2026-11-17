"""Transfer configuration precedes placement and keeps missing costs explicit."""

from __future__ import annotations

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission.common.resources import read_scoped_runtime
from compiler.emission.cuda.scoped import generate_scoped
from compiler.frontend import lower_file
from compiler.offload.config import OffloadConfig
from compiler.tests.test_scoped_planning_entries import calibration
from compiler.tests.test_structured_offload import SOURCE


def emit(tmp_path, transfers, *, profile=None, collective=False):
    source = tmp_path / "source.f90"
    source.write_text(SOURCE)
    options = CompilerOptions(gpu_policy="auto", scope_transfers=transfers)
    function, plan = prepare_function(lower_file(source, "advance"), options=options)
    return generate_scoped(function, plan, OffloadConfig("auto", profile, 4, collective, scope_transfers=transfers),
                           "common_functions.cuh", runtime_id=read_scoped_runtime()[1]["runtime_id"])


@pytest.mark.parametrize(("requested", "selected", "reason", "constant"), [
    ("direct", "direct", None, "DIRECT"), ("pinned", "pinned", None, "PINNED"),
    ("auto", "direct", "transfer_estimates_unavailable", "AUTO"),
    ("coalesced", "runtime", None, "COALESCED"),
])
def test_public_configuration_retains_requested_mode_without_initializing_cuda(tmp_path, requested, selected, reason, constant):
    emitted = emit(tmp_path, requested)
    configuration = emitted.report["transfer_configuration"]
    assert configuration["requested"] == requested
    assert configuration["selected"] == selected
    assert configuration["reason"] == reason
    assert configuration["execution"] == "synchronous"
    assert configuration["pinned_budget_bytes"] == 64 * 1024 * 1024
    assert configuration["argument_order"] == ["context"]
    text = emitted.cuda.split('extern "C" int ' + configuration["entry"], 1)[1].split("}", 1)[0]
    assert "return fort_scope_set_transfers(fort_context, FORT_SCOPE_TRANSFERS_" + constant + ");" in text
    assert "cuda" not in text.lower()
    assert emitted.cuda.count("fort_scope_set_transfers(") == 1
    assert "function configure(fort_context)" in emitted.fortran
    assert "bind(C, name='" + configuration["entry"] + "')" in emitted.fortran
    assert emitted.report["entry_abi_version"] == 3


def test_removed_pipelined_configuration_rejects_before_source_generation(tmp_path):
    from compiler.ir import CompilationError

    with pytest.raises(CompilationError, match="pipelined slab batching was removed"):
        emit(tmp_path, "pipelined")


def test_pinned_bandwidth_in_v1_profile_cannot_estimate_staging_or_select_gpu(tmp_path):
    value = calibration()
    assert value["rates"]["h2d_pinned"]["bandwidth_bytes_per_second"] > 0
    emitted = emit(tmp_path, "pinned", profile=value)
    assert not emitted.report["automatic_estimate_available"]
    assert emitted.report["automatic_reason"] == "transfer_estimates_unavailable"
    assert not emitted.report["planning"]["profile_available"]
    assert emitted.report["planning"]["query_available"]
    assert not emitted.report["transfer_configuration"]["placement_estimate_available"]
    selector = emitted.cuda.split('extern "C" int ' + emitted.report["planning"]["selector"], 1)[1]
    assert "costs.valid = 1" not in selector
    assert "costs.h2d_bandwidth =" not in selector
    assert "!preview.available ||" in selector


@pytest.mark.parametrize("transfers", ["auto"])
def test_direct_transfer_fallback_retains_calibrated_direct_placement(tmp_path, transfers):
    emitted = emit(tmp_path, transfers, profile=calibration())
    assert emitted.report["automatic_estimate_available"]
    assert emitted.report["transfer_configuration"]["selected"] == "direct"
    assert emitted.report["transfer_configuration"]["reason"]
    assert "costs.valid = 1" in emitted.cuda


def test_coalesced_uploads_keep_calibrated_exact_direct_placement(tmp_path):
    emitted = emit(tmp_path, "coalesced", profile=calibration())
    assert emitted.report["automatic_estimate_available"]
    configuration = emitted.report["transfer_configuration"]
    assert configuration["selected"] == "runtime"
    assert configuration["placement_estimate_available"]
    limits = configuration["coalesced"]
    assert limits["minimum_physical_copies"] == 2
    assert limits["maximum_extra_bytes_fraction"] == 0.125
    assert "defined and host current" in limits["source_requirement"]
    assert limits["pinning"] is False
    assert "exact direct" in limits["placement"]
    assert limits["fallback_reason"] == "coalesced_upload_ineligible"
    assert "costs.valid = 1" in emitted.cuda
    assert not emitted.report["batch_execution"]["enabled"]
