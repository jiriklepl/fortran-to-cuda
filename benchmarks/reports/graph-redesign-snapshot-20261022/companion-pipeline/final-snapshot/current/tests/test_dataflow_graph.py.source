"""Tests for canonical handle value-numbering, context dataflow graph, and residency scheduler."""
from __future__ import annotations

import unittest
import pytest

from elmm_pipeline.dataflow_graph import (
    BufferState,
    CanonicalBuffer,
    CanonicalBufferCatalog,
    ContextDataflowGraph,
    DataflowEdge,
    ExecutionTarget,
    HardwareCostModel,
    PlacementOptimizer,
    PlacementPlan,
    ResidencyAction,
    ResidencyActionType,
    ResidencyCoherenceScheduler,
    StageNode,
)


class TestCanonicalBufferCatalog(unittest.TestCase):
    """Test Global Value Numbering / Canonical Handle generation across scopes."""

    def setUp(self):
        self.catalog = CanonicalBufferCatalog()

    def test_canonicalize_module_variables(self):
        h1 = self.catalog.get_or_create("parameters::viscosity", shape=(128, 64, 64), element_bytes=8)
        # Fortran symbols are case-insensitive
        h2 = self.catalog.get_or_create("parameters::Viscosity", shape=(128, 64, 64), element_bytes=8)
        self.assertEqual(h1.handle_id, h2.handle_id)
        self.assertEqual(h1.canonical_name, "parameters::viscosity")
        self.assertEqual(h1.bytes, 128 * 64 * 64 * 8)

    def test_distinct_arrays_receive_distinct_handles(self):
        h_visc = self.catalog.get_or_create("parameters::viscosity", shape=(128, 64, 64), element_bytes=8)
        h_u = self.catalog.get_or_create("parameters::u", shape=(128, 64, 64), element_bytes=8)
        self.assertNotEqual(h_visc.handle_id, h_u.handle_id)

    def test_unify_dummy_arguments_with_module_sources(self):
        """At callsite, dummy argument is bound to module variable or outer array."""
        h_mod = self.catalog.get_or_create("parameters::viscosity", shape=(128, 64, 64), element_bytes=8)
        h_arg = self.catalog.get_or_create("subgrid::sgs_sigma_uniform::viscosity", shape=(128, 64, 64), element_bytes=8)
        self.assertNotEqual(h_mod.handle_id, h_arg.handle_id)

        # Unify via value-numbering / alias binding
        self.catalog.unify("subgrid::sgs_sigma_uniform::viscosity", "parameters::viscosity")
        resolved = self.catalog.lookup("subgrid::sgs_sigma_uniform::viscosity")
        self.assertEqual(resolved.handle_id, h_mod.handle_id)

    def test_local_scope_id_remapping(self):
        """Map local arbitrary integer IDs (e.g. 4 in SGS, 1 in Convection) to canonical ID."""
        h_visc = self.catalog.get_or_create("parameters::viscosity", shape=(128, 64, 64), element_bytes=8)
        self.catalog.register_local_alias(scope="sgs", local_id=4, canonical_symbol="parameters::viscosity")
        self.catalog.register_local_alias(scope="convection", local_id=1, canonical_symbol="parameters::viscosity")

        self.assertEqual(self.catalog.resolve_local_id("sgs", 4).handle_id, h_visc.handle_id)
        self.assertEqual(self.catalog.resolve_local_id("convection", 1).handle_id, h_visc.handle_id)


class TestContextDataflowGraph(unittest.TestCase):
    """Test dataflow graph construction and cost modeling."""

    def setUp(self):
        self.graph = ContextDataflowGraph()
        self.catalog = CanonicalBufferCatalog()

    def test_add_stages_and_dependencies(self):
        sgs = self.graph.add_stage("sgs", cpu_cost=3.9, gpu_cost=0.15, has_gpu=True)
        conv = self.graph.add_stage("convection", cpu_cost=1.9, gpu_cost=0.40, has_gpu=True)
        poisson = self.graph.add_stage("poisson", cpu_cost=1.14, gpu_cost=None, has_gpu=False)

        h_visc = self.catalog.get_or_create("parameters::viscosity", shape=(128, 64, 64), element_bytes=8)
        edge = self.graph.add_dependency(
            producer="sgs",
            consumer="convection",
            buffer=h_visc,
            transfer_bytes=h_visc.bytes
        )

        self.assertEqual(len(self.graph.stages), 3)
        self.assertEqual(len(self.graph.edges), 1)
        self.assertEqual(edge.producer, "sgs")
        self.assertEqual(edge.consumer, "convection")
        self.assertEqual(edge.buffer.handle_id, h_visc.handle_id)


class TestPlacementOptimizer(unittest.TestCase):
    """Test min-cut / heterogeneous graph partitioning placement optimization."""

    def setUp(self):
        self.graph = ContextDataflowGraph()
        self.catalog = CanonicalBufferCatalog()
        self.hardware = HardwareCostModel(
            pcie_latency_seconds=1e-5,
            pcie_bandwidth_bytes_per_sec=16e9,  # 16 GB/s PCIe Gen4
            kernel_launch_overhead_seconds=5e-6
        )
        self.optimizer = PlacementOptimizer(hardware=self.hardware)

    def test_isolated_vs_joint_gpu_placement(self):
        """Verify that joint GPU placement collapses the intermediate cut edge to 0 cost."""
        h_visc = self.catalog.get_or_create("parameters::viscosity", shape=(128, 64, 64), element_bytes=8)
        h_vel = self.catalog.get_or_create("parameters::u2", shape=(128, 64, 64), element_bytes=8)

        self.graph.add_stage("sgs", cpu_cost=3.9, gpu_cost=0.15, has_gpu=True)
        self.graph.add_stage("convection", cpu_cost=1.9, gpu_cost=0.40, has_gpu=True)
        self.graph.add_stage("poisson", cpu_cost=1.14, gpu_cost=None, has_gpu=False)

        # SGS produces Viscosity for Convection
        self.graph.add_dependency("sgs", "convection", h_visc, h_visc.bytes)
        # Convection produces U2 for Poisson (CPU only)
        self.graph.add_dependency("convection", "poisson", h_vel, h_vel.bytes)

        plan = self.optimizer.optimize(self.graph)

        # SGS and Convection should both be placed on GPU
        self.assertEqual(plan.placements["sgs"], ExecutionTarget.GPU)
        self.assertEqual(plan.placements["convection"], ExecutionTarget.GPU)
        self.assertEqual(plan.placements["poisson"], ExecutionTarget.CPU)

        # Edge (sgs, convection) should NOT be cut (resident on device)
        sgs_conv_edge = plan.get_edge_plan("sgs", "convection")
        self.assertFalse(sgs_conv_edge.is_cut)
        self.assertEqual(sgs_conv_edge.transfer_cost_seconds, 0.0)

        # Edge (convection, poisson) MUST be cut (GPU -> CPU transfer)
        conv_pois_edge = plan.get_edge_plan("convection", "poisson")
        self.assertTrue(conv_pois_edge.is_cut)
        self.assertGreater(conv_pois_edge.transfer_cost_seconds, 0.0)

    def test_threshold_fallback_to_cpu_when_overhead_dominates(self):
        """If a stage has trivial GPU speedup and high transfer cost, fall back to CPU."""
        h_buf = self.catalog.get_or_create("temp_data", shape=(256, 256, 256), element_bytes=8)  # 134 MB

        # Tiny kernel: 1.00s CPU, 0.99s GPU (saves only 0.01s, but transferring 134MB takes ~0.01s + latency)
        self.graph.add_stage("heavy_cpu_producer", cpu_cost=5.0, gpu_cost=None, has_gpu=False)
        self.graph.add_stage("tiny_candidate", cpu_cost=1.00, gpu_cost=0.99, has_gpu=True)
        self.graph.add_stage("heavy_cpu_consumer", cpu_cost=5.0, gpu_cost=None, has_gpu=False)

        self.graph.add_dependency("heavy_cpu_producer", "tiny_candidate", h_buf, h_buf.bytes)
        self.graph.add_dependency("tiny_candidate", "heavy_cpu_consumer", h_buf, h_buf.bytes)

        plan = self.optimizer.optimize(self.graph)
        # Offloading tiny_candidate would require HtoD and DtoH of 134MB twice, which exceeds 0.01s savings!
        self.assertEqual(plan.placements["tiny_candidate"], ExecutionTarget.CPU)

    def test_elmm_multistage_rk_placement(self):
        """Verify full ELMM RK3 placement: collapses all internal GPU-GPU edges, cuts only to CPU Poisson."""
        self.graph.add_stage("sgs", cpu_cost=3.90, gpu_cost=0.15, has_gpu=True)
        self.graph.add_stage("convection_rk1", cpu_cost=0.63, gpu_cost=0.13, has_gpu=True)
        self.graph.add_stage("convection_rk2", cpu_cost=0.63, gpu_cost=0.13, has_gpu=True)
        self.graph.add_stage("convection_rk3", cpu_cost=0.63, gpu_cost=0.13, has_gpu=True)
        self.graph.add_stage("poisson", cpu_cost=1.14, gpu_cost=None, has_gpu=False)

        visc = self.catalog.get_or_create("parameters::viscosity", shape=(128, 64, 64), element_bytes=8)
        u = self.catalog.get_or_create("parameters::u", shape=(128, 64, 64), element_bytes=8)

        self.graph.add_dependency("sgs", "convection_rk1", visc)
        self.graph.add_dependency("convection_rk1", "convection_rk2", u)
        self.graph.add_dependency("convection_rk2", "convection_rk3", u)
        self.graph.add_dependency("convection_rk3", "poisson", u)

        plan = self.optimizer.optimize(self.graph)

        self.assertEqual(plan.placements["sgs"], ExecutionTarget.GPU)
        self.assertEqual(plan.placements["convection_rk1"], ExecutionTarget.GPU)
        self.assertEqual(plan.placements["convection_rk2"], ExecutionTarget.GPU)
        self.assertEqual(plan.placements["convection_rk3"], ExecutionTarget.GPU)
        self.assertEqual(plan.placements["poisson"], ExecutionTarget.CPU)

        self.assertFalse(plan.get_edge_plan("sgs", "convection_rk1").is_cut)
        self.assertFalse(plan.get_edge_plan("convection_rk1", "convection_rk2").is_cut)
        self.assertFalse(plan.get_edge_plan("convection_rk2", "convection_rk3").is_cut)
        self.assertTrue(plan.get_edge_plan("convection_rk3", "poisson").is_cut)



class TestResidencyCoherenceScheduler(unittest.TestCase):
    """Test memory residency state tracking, optimistic handoff, lazy fallback, and proactive DtoH."""

    def setUp(self):
        self.catalog = CanonicalBufferCatalog()
        self.scheduler = ResidencyCoherenceScheduler(self.catalog)

    def test_optimistic_gpu_to_gpu_handoff(self):
        """When producer and consumer are both on GPU, memory stays DEVICE_CURRENT with zero copy."""
        h_visc = self.catalog.get_or_create("parameters::viscosity", shape=(128, 64, 64), element_bytes=8)

        # 1. SGS runs on GPU
        actions_sgs = self.scheduler.on_stage_begin("sgs", target=ExecutionTarget.GPU, reads=[], writes=[h_visc])
        self.assertEqual(self.scheduler.get_state(h_visc), BufferState.DEVICE_CURRENT)

        actions_sgs_end = self.scheduler.on_stage_end("sgs", next_predicted_stage="convection", next_target=ExecutionTarget.GPU)
        # Zero DtoH transfer should be scheduled on SGS exit!
        dtoh_transfers = [a for a in actions_sgs_end if a.action_type == ResidencyActionType.DtoH_TRANSFER]
        self.assertEqual(len(dtoh_transfers), 0)

        # 2. Convection starts on GPU and reads Viscosity
        actions_conv = self.scheduler.on_stage_begin("convection", target=ExecutionTarget.GPU, reads=[h_visc], writes=[])
        # Should be an optimistic resident handoff, zero HtoD transfer
        self.assertTrue(any(a.action_type == ResidencyActionType.RESIDENT_HANDOFF for a in actions_conv))
        htod_transfers = [a for a in actions_conv if a.action_type == ResidencyActionType.HtoD_TRANSFER]
        self.assertEqual(len(htod_transfers), 0)

    def test_proactive_async_synchronization_for_likely_cpu_consumer(self):
        """If the predicted next consumer is CPU, schedule non-blocking async DtoH immediately."""
        h_u2 = self.catalog.get_or_create("parameters::u2", shape=(128, 64, 64), element_bytes=8)

        self.scheduler.on_stage_begin("convection", target=ExecutionTarget.GPU, reads=[], writes=[h_u2])
        # On exit, next stage is Poisson which is on CPU
        actions = self.scheduler.on_stage_end("convection", next_predicted_stage="poisson", next_target=ExecutionTarget.CPU)

        proactive_actions = [a for a in actions if a.action_type == ResidencyActionType.PROACTIVE_ASYNC_DtoH]
        self.assertEqual(len(proactive_actions), 1)
        self.assertEqual(proactive_actions[0].buffer.handle_id, h_u2.handle_id)

    def test_lazy_fallback_synchronization_fence_on_unlikely_cpu_read(self):
        """If an unexpected branch reaches a CPU stage while buffer is DEVICE_CURRENT, emit a lazy fence."""
        h_visc = self.catalog.get_or_create("parameters::viscosity", shape=(128, 64, 64), element_bytes=8)

        # SGS writes Viscosity on GPU
        self.scheduler.on_stage_begin("sgs", target=ExecutionTarget.GPU, reads=[], writes=[h_visc])
        self.scheduler.on_stage_end("sgs", next_predicted_stage="convection", next_target=ExecutionTarget.GPU)

        # Unexpected branch: a CPU diagnostics routine reads Viscosity
        actions = self.scheduler.on_stage_begin("cpu_diagnostics", target=ExecutionTarget.CPU, reads=[h_visc], writes=[])

        lazy_fences = [a for a in actions if a.action_type == ResidencyActionType.LAZY_SYNC_FENCE]
        self.assertEqual(len(lazy_fences), 1)
        self.assertEqual(lazy_fences[0].buffer.handle_id, h_visc.handle_id)
        # After the lazy fence, state becomes BOTH_VALID
        self.assertEqual(self.scheduler.get_state(h_visc), BufferState.BOTH_VALID)


class TestUnifiedScopeBridge(unittest.TestCase):
    """Test generating the shared Fortran/C++ residency bridge module."""

    def test_bridge_source_generation(self):
        from elmm_pipeline.dataflow_graph import UnifiedScopeBridge

        bridge = UnifiedScopeBridge()
        fortran_source = bridge.generate_fortran_bridge()

        self.assertIn("module fort_resident_bridge", fortran_source)
        self.assertIn("global_resident_context", fortran_source)
        self.assertIn("fort_get_or_create_resident_context", fortran_source)
        self.assertIn("fort_close_resident_context", fortran_source)
        self.assertIn("fort_ensure_host_buffer", fortran_source)
        self.assertIn("fort_mark_device_buffer", fortran_source)

    def test_patch_subgrid_source(self):
        from elmm_pipeline.dataflow_graph import UnifiedScopeBridge

        bridge = UnifiedScopeBridge()
        sample_subgrid = """
integer(fort_selection_kind_0a0443cf2c90) :: fort_selection_context_0a0443cf2c90
fort_selection_context_0a0443cf2c90 = 0
fort_actual_status = fort_scope_create(0_c_int, fort_selection_context_0a0443cf2c90)
fort_close_status = fort_scope_close(fort_selection_context_0a0443cf2c90)
"""
        patched = bridge.patch_subgrid_source(sample_subgrid)
        self.assertIn("use fort_resident_bridge", patched)
        self.assertIn("fort_get_or_create_resident_context", patched)
        self.assertIn("fort_mark_device_buffer", patched)
        self.assertNotIn("fort_close_status = fort_scope_close(fort_selection_context_0a0443cf2c90)", patched)

    def test_remap_dynamics_identities(self):
        from elmm_pipeline.dataflow_graph import UnifiedScopeBridge

        bridge = UnifiedScopeBridge()
        sample = """
fort_actual_status = fort_scope_register(fort_selection_context_2d2e69bd1df9, 3_c_int64_t, &
1_c_int64_t, fort_actual_layout, 1_c_int, fort_buffer_3)
fort_status = fort_scope_register(fort_context, 5_c_int64_t, &
1_c_int64_t, fort_layout, 1_c_int, fort_team_buffer_5)
"""
        remapped = bridge.remap_dynamics_identities(sample, offset=10)
        self.assertIn("13_c_int64_t", remapped)
        self.assertIn("15_c_int64_t", remapped)
        self.assertNotIn(" 3_c_int64_t", remapped)
        self.assertNotIn(" 5_c_int64_t", remapped)

    def test_bridge_has_invalidation_helper(self):
        from elmm_pipeline.dataflow_graph import UnifiedScopeBridge

        bridge = UnifiedScopeBridge()
        source = bridge.generate_fortran_bridge()
        self.assertIn("fort_invalidate_device_buffer", source)
        self.assertIn("fort_scope_forget_definition", source)


if __name__ == "__main__":
    unittest.main()
