"""Source-backed batching reuses numerical kernels and complete data-flow proofs."""

from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission.common.resources import read_scoped_runtime
from compiler.emission.cuda.scoped import generate_scoped
from compiler.frontend import lower_file
from compiler.offload.config import OffloadConfig
from compiler.offload.profile import SCOPED_BATCH_PAYLOADS, SCOPED_TRANSFER_COST_NAMES
from compiler.scopes.source import form_source_scopes
from compiler.tests.test_scoped_planning_entries import calibration
from compiler.tests.test_source_scopes import FACT

FLAT = """module numerical
contains
subroutine advance(a,b,c,n,m)
real(8),intent(in)::a(:,:)
real(8),intent(inout)::b(:,:),c(:,:)
integer,intent(in)::n,m
integer::i,j
do j=2,m-1
 do i=2,n-1
  b(i,j)=a(i-1,j)+a(i,j)+a(i+1,j)+real(i,8)
 enddo
enddo
do j=2,m-1
 do i=2,n-1
  c(i,j)=2*b(i,j)+a(i,j)
 enddo
enddo
end subroutine
end module
"""

CHAIN = """module original
contains
subroutine first(a,b,n,m)
real(8),intent(in)::a(:,:)
real(8),intent(out)::b(:,:)
integer,intent(in)::n,m
integer::i,j
do j=2,m-1
 do i=2,n-1
  b(i,j)=a(i-1,j)+a(i,j)+a(i+1,j)+real(i,8)
 enddo
enddo
end subroutine
subroutine second(b,c,n,m)
real(8),intent(in)::b(:,:)
real(8),intent(out)::c(:,:)
integer,intent(in)::n,m
integer::i,j
do j=2,m-1
 do i=2,n-1
  c(i,j)=2*b(i,j)+real(j,8)
 enddo
enddo
end subroutine
subroutine step(a,b,c,n,m)
real(8),intent(in)::a(:,:)
real(8),intent(inout)::b(:,:),c(:,:)
integer,intent(in)::n,m
call first(a,b,n,m)
call second(b,c,n,m)
end subroutine
end module
"""


def transfer_profile():
    value = calibration()
    value["hardware"]["async_engine_count"] = 2
    costs = {name: 1e-7 for name in SCOPED_TRANSFER_COST_NAMES}
    costs.update(staging_cold_seconds=[1e-6]*4, staging_reuse_seconds=[1e-7]*4,
                 pack_bytes_per_second=1e11, unpack_bytes_per_second=1e11,
                 pack_row_seconds=0.0, unpack_row_seconds=0.0)
    value["scoped"]["transfers"] = {"schema_version": 1, "batch_payload_bytes": list(SCOPED_BATCH_PAYLOADS),
                                      "max_slot_bytes": SCOPED_BATCH_PAYLOADS[-1], "costs": costs}
    return value


def numerical(tmp_path, source=FLAT, mode="pipelined", profile=None):
    path = tmp_path / "numerical.f90"
    path.write_text(source)
    options = CompilerOptions(opt_level=0, gpu_policy="sections", memory_model="scoped", scope_transfers=mode)
    function, plan = prepare_function(lower_file(path, "advance"), options=options)
    config = OffloadConfig("sections", transfer_profile() if profile is None else profile, 4, scope_transfers=mode)
    return generate_scoped(function, plan, config, "common_functions.cuh", runtime_id=read_scoped_runtime()[1]["runtime_id"])


def scope(tmp_path, source=CHAIN, mode="pipelined", *, negative=False):
    path = tmp_path / "original.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::"+name: FACT for name in ("a", "b", "c")}}
    options = CompilerOptions(opt_level=0, gpu_policy="sections", memory_model="scoped", scope_transfers=mode)
    numerical_sources=None
    if negative:
        from compiler.tests.test_scoped_batch_sources_cuda import negative_package
        numerical_sources=negative_package(tmp_path,path)
    outputs, report = form_source_scopes([path], "step", facts=facts, options=options,
        config=OffloadConfig("sections", transfer_profile(), 4, scope_transfers=mode), numerical_sources=numerical_sources)
    return path, outputs, report


def test_one_window_worker_reuses_kernels_and_preserves_full_layout(tmp_path):
    emitted = numerical(tmp_path)
    batch = emitted.report["batch_execution"]
    assert batch["enabled"]
    assert batch["eligible"]
    assert any(item["first_unit"] == 0 and item["stop_unit"] == 2 and item["eligible"] for item in batch["candidates"])
    assert emitted.cuda.count("__global__ void") == 2
    worker = emitted.cuda.split('extern "C" int '+batch["window_entry"], 1)[1].split("static int", 1)[0]
    assert "fort_window->begin" in worker
    assert "fort_window->count" in worker
    assert "fort_window->stream" in worker
    assert "_layout.extents[" in worker
    assert "fort_scope_" not in worker.replace("fort_scope_batch_window", "").replace("fort_scope_batch_view", "")
    assert "timing::measure" not in worker
    assert "++*fort_launches" in worker
    assert "scope_batch" not in emitted.fortran
    assert batch["complete_owner_estimate"].startswith("unavailable")


def test_immutable_overlapping_halo_is_prefix_and_intermediate_is_shifted(tmp_path):
    emitted = numerical(tmp_path)
    complete = next(item for item in emitted.report["batch_execution"]["candidates"] if item["stop_unit"] == 2 and item["first_unit"] == 0)
    assert complete["eligible"]
    arrays = {item["symbol"]: item for item in complete["slab"]["arrays"]}
    # Either the halo axis requires a prefix or the orthogonal axis makes the
    # input disjoint; both proofs use physical coordinates, not transformed IR.
    assert arrays["b"]["dimension"] is not None
    assert arrays["c"]["dimension"] is not None
    assert complete["slab"]["independence"].startswith("complete subchain")
    assert "preview.baseline_seconds-preview.execution_seconds" in emitted.cuda
    assert emitted.cuda.index("-1,nullptr,nullptr,&preview") < emitted.cuda.index("offload::compatible(profile,4,64)", emitted.cuda.index("-1,nullptr,nullptr,&preview"))


def test_boundary_domains_do_not_exclude_a_proved_interior_interval(tmp_path):
    source = FLAT.replace("integer::i,j", "integer::i,j\ndo j=2,m-1\nb(1,j)=a(1,j)\nenddo")
    emitted = numerical(tmp_path, source)
    candidates = emitted.report["batch_execution"]["candidates"]
    assert any(item["first_unit"] == 1 and item["stop_unit"] == 3 and item["eligible"] for item in candidates)
    assert not next(item for item in candidates if item["first_unit"] == 0 and item["stop_unit"] == 3)["eligible"]
    assert "fort_batch_skip" in emitted.cuda
    assert emitted.cuda.count("__global__ void") == 3


@pytest.mark.parametrize("mode", ["direct", "pinned"])
def test_synchronous_controls_do_not_emit_batch_workers(tmp_path, mode):
    emitted = numerical(tmp_path, mode=mode)
    assert not emitted.report["batch_execution"]["eligible"]
    assert "fort_scope_batch_execute_v1(" not in emitted.cuda
    if mode == "pinned":
        assert "fort_scope_set_transfer_costs_v1(" in emitted.cuda
        assert emitted.report["automatic_estimate_available"]


def test_old_profile_retains_direct_fallback(tmp_path):
    emitted = numerical(tmp_path, profile=calibration())
    assert emitted.report["batch_execution"]["eligible"]
    assert not emitted.report["batch_execution"]["enabled"]
    assert emitted.report["transfer_configuration"]["selected"] == "direct"
    assert "fort_batch_skip" not in emitted.cuda


@pytest.mark.parametrize("profile", [{}, {"schema_version": 1}])
def test_malformed_direct_api_profile_preserves_native_automatic_fallback(tmp_path, profile):
    emitted = numerical(tmp_path, profile=profile, mode="auto")
    assert not emitted.report["automatic_estimate_available"]
    assert not emitted.report["batch_execution"]["enabled"]
    helper = emitted.cuda.split('extern "C" int '+emitted.report["batch_execution"]["entry"],1)[1]
    assert "const auto profile=offload::Profile{};" in helper
    assert "fort_batch_skip" not in emitted.cuda


def test_cross_leaf_chain_has_one_callback_and_ordered_definition_events(tmp_path):
    original, outputs, report = scope(tmp_path)
    assert original.read_text() == CHAIN
    owner, = report["scopes"]
    chain, = owner["batch_subchains"]
    assert chain["eligible"]
    assert chain["calls"] == ["original::first", "original::second"]
    assert chain["units"] == 2
    artifact = outputs[chain["artifacts"][0]]
    assert "__global__" not in artifact
    assert artifact.count("event.value.kind=FORT_SCOPE_PLAN_FORGET") == 2
    callback = artifact.split("void *user,uint64_t *launches)", 1)[1].split("static bool", 1)[0]
    assert callback.count("_window_v1(window,0,1ULL,state.axis,launches") == 2
    assert "fort_scope_layout_get(" not in callback
    assert "fort_scope_access_begin(" not in callback
    assert "fort_scope_note_launch(" not in callback
    text = outputs[report["sources"][str(original)]["replacement"]]
    assert "integer(c_size_t)::fort_batch_stop" in text
    assert "if (fort_batch_stop == 0) then" in text
    assert text.index("fort_batch_stop == 0") < text.index("call fort_scope_clone_", text.index("fort_batch_stop == 0"))
    assert len([item for item in report["build_sources"] if item["path"].endswith("scope_batch.cu")]) == 1


def test_cross_chunk_written_halo_declines_chain_but_retains_native_source(tmp_path):
    changed = CHAIN.replace("c(i,j)=2*b(i,j)+real(j,8)", "c(i,j)=2*b(i-1,j-1)+real(j,8)")
    _original, _outputs, report = scope(tmp_path, changed)
    chain, = report["scopes"][0]["batch_subchains"]
    assert not chain["eligible"]
    assert "cross-chunk written-array dependence" in chain["reason"]


def test_reached_source_segments_can_have_independent_batch_chains(tmp_path):
    changed = CHAIN.replace("call second(b,c,n,m)\nend subroutine", "call second(b,c,n,m)\nif(n>3) then\n"
                            "call first(a,b,n,m)\ncall second(b,c,n,m)\nendif\nend subroutine")
    _original, _outputs, report = scope(tmp_path, changed)
    owner, = report["scopes"]
    assert owner["ownership"]["planning_mode"] == "continuation"
    assert len(owner["batch_subchains"]) == 2
    assert all(item["eligible"] for item in owner["batch_subchains"])


def test_native_consumer_separates_retained_and_terminal_publication(tmp_path):
    changed = CHAIN.replace("intent(out)::b", "intent(inout)::b").replace("intent(out)::c", "intent(inout)::c")
    changed = changed.replace("call second(b,c,n,m)\nend subroutine", "call second(b,c,n,m)\n"
                              "call observe(b,c)\ncall first(a,b,n,m)\ncall second(b,c,n,m)\nend subroutine")
    changed = changed.replace("end module", "subroutine observe(b,c)\nreal(8),intent(in)::b(:,:)\n"
                              "real(8),intent(inout)::c(:,:)\nif(size(c,1)>0.and.size(c,2)>0) c(1,1)=sum(b)+sum(c)\nend subroutine\nend module")
    _original, outputs, report = scope(tmp_path, changed)
    first, last = report["scopes"][0]["batch_subchains"]
    assert first["eligible"]
    assert last["eligible"]
    assert not first["terminal_exports"]
    assert len(last["terminal_exports"]) == 2
    assert "descriptor.exports=exports.data()" in outputs[last["artifacts"][0]]


def test_negative_logical_coordinates_are_public_for_a_multikernel_leaf(tmp_path):
    from compiler.tests.test_scoped_batch_sources_cuda import source_case
    _original, _outputs, report = scope(tmp_path, source_case("flat"), negative=True)
    leaf, = [item for item in report["numerical_decisions"] if item["procedure"] == "original::advance"]
    assert leaf["supported"], leaf
    assert leaf["batch_execution"]["enabled"]
    assert leaf["batch_execution"]["coordinates"] == "original logical coordinates and full-array pitches"
    assert any(item["first_unit"] == 0 and item["stop_unit"] == 2 and item["eligible"]
               for item in leaf["batch_execution"]["candidates"])
