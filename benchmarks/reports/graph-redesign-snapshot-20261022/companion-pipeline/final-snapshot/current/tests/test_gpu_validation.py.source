import json
import unittest

from elmm_pipeline.gpu_validation import (
    gpu_activity,
    offload_decisions,
    native_policy_exercised,
    scoped_modeled_decisions,
    scoped_modeled_evidence,
    trace_environment,
)


class GPUValidationTests(unittest.TestCase):

    def test_local_trace_needs_a_successful_kernel_event(self):
        log = "FORT_RUNTIME upload bytes=80\nFORT_RUNTIME download bytes=40\n"
        self.assertFalse(gpu_activity(log, "local", [])["verified"])
        result = gpu_activity(log + "FORT_RUNTIME kernel\n", "local", [])
        self.assertTrue(result["verified"])
        self.assertEqual((result["upload_bytes"], result["download_bytes"]), (80, 40))

    def test_timed_environment_cannot_inherit_gpu_tracing(self):
        original = {"FORT_RUNTIME_TRACE": "1", "FORT_OFFLOAD_TRACE": "1", "NVCOMPILER_ACC_TIME": "1", "CUDA_VISIBLE_DEVICES": "0"}
        self.assertEqual(trace_environment(original, "local"), {"CUDA_VISIBLE_DEVICES": "0"})
        self.assertEqual(trace_environment(original, "local", enabled=True)["FORT_RUNTIME_TRACE"], "1")
        self.assertEqual(trace_environment(original, "local", enabled=True)["FORT_OFFLOAD_TRACE"], "1")

    def test_shared_runtime_events_count_successful_launches_and_sections(self):
        log = ("FORT_SCOPED initialize bytes=0\n"
               "FORT_SCOPED allocate buffer=12 generation=3 bytes=4096\n"
               "FORT_SCOPED upload buffer=12 generation=3 bytes=160\n"
               "FORT_SCOPED launch bytes=0\nFORT_SCOPED wait bytes=0\n"
               "FORT_SCOPED download buffer=12 generation=3 bytes=80\n"
               "NOTFORT_SCOPED launch bytes=0\n")
        activity = gpu_activity(log, "local", [])
        self.assertTrue(activity["verified"])
        self.assertEqual(activity["kernel_launches"], 1)
        self.assertEqual((activity["upload_bytes"], activity["download_bytes"]), (160, 80))
        self.assertEqual(activity["allocations"], 1)
        self.assertEqual(activity["scoped_runtime"]["waits"], 1)
        self.assertFalse(native_policy_exercised(activity, []))

    def test_compiler_native_scope_contract_has_no_invented_runtime_calls(self):
        manifest = {"schema_version": 1, "scope_count": 1, "automatic_estimate_available": False,
                    "scopes": [{"mode": "auto", "estimate_available": False}]}
        activity = gpu_activity("", "local", [], scope_manifest=manifest)
        self.assertTrue(native_policy_exercised(activity, []))
        self.assertFalse(activity["verified"])
        self.assertEqual(activity["offload_decisions"]["calls"], 0)
        with self.assertRaisesRegex(ValueError, "contradicts"):
            gpu_activity("FORT_SCOPED launch bytes=0\n", "local", [], scope_manifest=manifest)
        manifest["scopes"][0]["estimate_available"] = True
        self.assertFalse(gpu_activity("", "local", [], scope_manifest=manifest)["compiler_native_selection"])

    def test_scoped_resource_fallback_uses_public_leaf_identity(self):
        region = {"entry": "scope_owner", "transformation": {"memory_model": "scoped"}}
        manifest = {"schema_version": 1, "scope_count": 1, "automatic_estimate_available": False,
                    "scopes": [{"mode": "sections", "gpu_leaves": ["module::leaf"]}],
                    "numerical_sources": [{"entry": "normalized_module::normalized_leaf"}]}
        log = "FORT_OFFLOAD entry=normalized_leaf mode=native-scoped-resource gpu_units=0 cpu_units=1\n"
        activity = gpu_activity(log, "local", [region], scope_manifest=manifest)
        self.assertTrue(native_policy_exercised(activity, [region]))
        self.assertFalse(activity["compiler_native_selection"])
        self.assertEqual(activity["offload_decisions"]["native_reasons"], {"native-scoped-resource": 1})
        with self.assertRaisesRegex(ValueError, "absent"):
            gpu_activity(log.replace("normalized_leaf", "unknown"), "local", [region], scope_manifest=manifest)

    def test_composed_native_and_gpu_packages_union_approved_entries(self):
        packages = [scope_package("native_leaf"), scope_package("gpu_leaf", available=True)]
        before = json.dumps(packages, sort_keys=True)
        log = ("FORT_OFFLOAD entry=native_leaf mode=native-scoped-unestimated gpu_units=0 cpu_units=1\n"
               "FORT_OFFLOAD entry=gpu_leaf mode=scoped-scheduled gpu_units=1 cpu_units=0\n"
               "FORT_SCOPED launch bytes=0\n")
        activity = gpu_activity(log, "local", [], scope_packages=packages)
        self.assertTrue(activity["verified"])
        self.assertFalse(activity["compiler_native_selection"])
        self.assertFalse(native_policy_exercised(activity, []))
        self.assertEqual(activity["scope_manifest_count"], 2)
        self.assertEqual(set(activity["offload_decisions"]["entries"]), {"native_leaf", "gpu_leaf"})
        self.assertEqual(offload_decisions(log, [], scope_packages=packages)["calls"], 2)
        self.assertEqual(json.dumps(packages, sort_keys=True), before)
        with self.assertRaisesRegex(ValueError, "absent"):
            gpu_activity(log.replace("gpu_leaf", "foreign_leaf"), "local", [], scope_packages=packages)

    def test_composed_all_native_contract_needs_every_package(self):
        packages = [scope_package("first"), scope_package("second")]
        activity = gpu_activity("", "local", [], scope_packages=packages)
        self.assertTrue(activity["compiler_native_selection"])
        self.assertTrue(native_policy_exercised(activity, []))
        self.assertFalse(activity["verified"])
        self.assertEqual(activity["offload_decisions"]["calls"], 0)
        with self.assertRaisesRegex(ValueError, "contradicts"):
            gpu_activity("FORT_SCOPED launch bytes=0\n", "local", [], scope_packages=packages)

    def test_single_manifest_and_packages_remain_separate_diagnostic_inputs(self):
        first, second = scope_package("first"), scope_package("second", available=True)
        log = "FORT_OFFLOAD entry=second mode=native-scoped-scheduled gpu_units=0 cpu_units=1\n"
        activity = gpu_activity(log, "local", [], scope_manifest=first["scope_manifest"], scope_packages=[second])
        self.assertFalse(activity["compiler_native_selection"])
        self.assertTrue(native_policy_exercised(activity, []))
        self.assertEqual(activity["scope_manifest_count"], 2)
        for packages in ({}, [{}], [{"scope_manifest": None}]):
            with self.subTest(packages=packages), self.assertRaises(ValueError):
                gpu_activity("", "local", [], scope_packages=packages)

    def test_scheduled_scopes_keep_policy_counts_and_cost_estimates_separate(self):
        region = {"entry": "scope_owner", "transformation": {"memory_model": "scoped"}}
        manifest = {"schema_version": 1, "scope_count": 1, "automatic_estimate_available": True,
                    "scopes": [{"mode": "auto", "gpu_leaves": ["module::producer"]}]}
        estimates = scoped_decision(gpu_units=1, cpu_units=1, upload_bytes=4096, download_bytes=2048)
        log = (estimates + "FORT_OFFLOAD entry=producer mode=scoped-scheduled gpu_units=1 cpu_units=1\n"
               "FORT_SCOPED upload buffer=4 generation=1 bytes=160\n"
               "FORT_SCOPED download buffer=4 generation=1 bytes=80\n")
        activity = gpu_activity(log, "local", [region], scope_manifest=manifest)
        self.assertFalse(activity["verified"])
        self.assertEqual(activity["offload_decisions"]["modes"]["scoped-scheduled"], 1)
        self.assertEqual((activity["upload_bytes"], activity["download_bytes"]), (160, 80))
        modeled = activity["scoped_runtime"]["modeled_totals"]
        self.assertEqual((modeled["upload_bytes"], modeled["download_bytes"]), (4096, 2048))
        self.assertEqual(modeled["available_calls"], 1)
        self.assertEqual(modeled["peak_device_bytes"], 4096)
        self.assertEqual(modeled["estimated_seconds"], 0.01)
        self.assertNotIn("wall_seconds", modeled)
        self.assertTrue(gpu_activity(log + "FORT_SCOPED launch bytes=0\n", "local", [region],
                                     scope_manifest=manifest)["verified"])
        with self.assertRaisesRegex(ValueError, "public source-scope manifest"):
            gpu_activity(log, "local", [{"entry": "producer"}])

    def test_available_native_schedule_is_observed_without_cuda(self):
        region = {"entry": "scope_owner", "transformation": {"memory_model": "scoped"}}
        manifest = {"schema_version": 1, "scope_count": 1, "automatic_estimate_available": True,
                    "scopes": [{"mode": "auto", "estimate_available": True, "gpu_leaves": ["module::producer"]}]}
        log = (scoped_decision(gpu_units=0, cpu_units=4, launches=0, reason="native_gpu_startup_lower_bound")
               + "FORT_OFFLOAD entry=producer mode=native-scoped-scheduled gpu_units=0 cpu_units=4\n")
        activity = gpu_activity(log, "local", [region], scope_manifest=manifest)
        self.assertTrue(native_policy_exercised(activity, [region]))
        self.assertFalse(activity["compiler_native_selection"])
        self.assertFalse(activity["verified"])
        self.assertEqual(activity["offload_decisions"]["native_reasons"], {"native-scoped-scheduled": 1})
        record = activity["scoped_runtime"]["modeled_decisions"][0]
        self.assertIs(record["available"], True)
        self.assertEqual(record["reason"], "native_gpu_startup_lower_bound")
        self.assertEqual(activity["scoped_runtime"]["modeled_totals"]["cpu_units"], 4)

    def test_unavailable_and_available_records_are_counted_without_inventing_availability(self):
        log = (scoped_decision(available=0, gpu_units=0, cpu_units=2, launches=0, reason="missing_costs")
               + scoped_decision(gpu_units=0, cpu_units=3, launches=0)
               + "NOTFORT_SCOPED decision available=1\n")
        activity = gpu_activity(log, "local", [])
        modeled = activity["scoped_runtime"]
        self.assertEqual(modeled["modeled_totals"]["calls"], 2)
        self.assertEqual(modeled["modeled_totals"]["available_calls"], 1)
        self.assertEqual(modeled["modeled_totals"]["cpu_units"], 5)
        self.assertFalse(modeled["modeled_decisions"][0]["available"])
        self.assertEqual(activity["kernel_launches"], 0)

    def test_malformed_decision_fields_fail_closed(self):
        log = scoped_decision()
        malformed = [log.replace("available=1", "available=2"),
                     log.replace("gpu_units=1", "gpu_units=-1"),
                     log.replace("cpu_units=1", "cpu_units=4294967296"),
                     log.replace("upload_bytes=0", "upload_bytes=18446744073709551616"),
                     log.replace("estimated_seconds=0.01", "estimated_seconds=nan"),
                     log.replace("estimated_seconds=0.01", "estimated_seconds=-0.01"),
                     log.replace("estimated_seconds=0.01", "estimated_seconds=unknown"),
                     log.replace("estimated_seconds=0.01 ", ""),
                     log.replace("reason=calibrated_selection", ""),
                     log.replace("reason=", "gpu_units=1 reason="),
                     log.replace("available=1", "available=0")]
        for candidate in malformed:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                scoped_modeled_decisions(candidate)

    def test_public_resource_evidence_is_preserved_without_becoming_observed_copies(self):
        copied = {"schema_version": 1, "context": 5, "sequence": 3, "event": "copy", "phase": "read",
                  "resource": 6, "identity": 17, "generation": 9, "element_bytes": 8,
                  "allocation_bytes": 256, "allocated": True, "extents": [32],
                  "initialized": [{"lower": [0], "upper": [32]}],
                  "host_current": [{"lower": [0], "upper": [32]}], "device_current": [],
                  "unit": 91, "direction": "h2d", "rectangle": {"lower": [2], "upper": [30]},
                  "bytes": 224, "copy_calls": 1}
        ended = {"schema_version": 1, "context": 5, "event": "end", "rows": 4096,
                 "complete": False, "truncated": True}
        log = "".join("FORT_SCOPED evidence " + json.dumps(record) + "\n" for record in [copied, ended])
        log += "NOTFORT_SCOPED evidence {}\n"
        activity = gpu_activity(log, "local", [])
        self.assertEqual(activity["scoped_runtime"]["modeled_evidence"], [copied, ended])
        self.assertEqual(activity["upload_bytes"], 0)
        self.assertEqual(activity["allocations"], 0)
        self.assertFalse(activity["verified"])

    def test_malformed_or_nonfinite_public_evidence_is_rejected(self):
        candidates = ['not-json', '[]', '{}',
                      '{"schema_version":2,"context":1,"event":"copy"}',
                      '{"schema_version":true,"context":1,"event":"copy"}',
                      '{"schema_version":1,"context":true,"event":"copy"}',
                      '{"schema_version":1,"context":18446744073709551616,"event":"copy"}',
                      '{"schema_version":1,"context":1,"event":[]}',
                      '{"schema_version":1,"context":1,"event":"unknown"}',
                      '{"schema_version":1,"context":1,"context":2,"event":"copy"}',
                      '{"schema_version":1,"context":1,"event":"copy","seconds":NaN}',
                      '{"schema_version":1,"context":1,"event":"copy","values":[1e999]}',
                      '{"schema_version":1,"context":1,"event":"end","rows":1,"complete":"true","truncated":false}']
        for candidate in candidates:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                scoped_modeled_evidence("FORT_SCOPED evidence " + candidate + "\n")

    def test_definition_validation_evidence_cannot_verify_gpu_execution(self):
        for status, reason in [(0, "definition_plan_valid"), (7, "uninitialized_read")]:
            proof = {"schema_version": 1, "context": 5, "event": "definition_validation",
                     "status": status, "reason": reason, "evidence": "ordered_definition_preflight",
                     "mutates_live_state": False}
            activity = gpu_activity("FORT_SCOPED evidence " + json.dumps(proof) + "\n", "local", [])
            self.assertEqual(activity["scoped_runtime"]["modeled_evidence"], [proof])
            self.assertEqual(activity["upload_bytes"], 0)
            self.assertEqual(activity["download_bytes"], 0)
            self.assertEqual(activity["allocations"], 0)
            self.assertFalse(activity["verified"])

    def test_definition_validation_evidence_requires_public_proof_fields(self):
        valid = {"schema_version": 1, "context": 5, "event": "definition_validation",
                 "status": 7, "reason": "uninitialized_read", "evidence": "ordered_definition_preflight",
                 "mutates_live_state": False}
        for change in [{"status": True}, {"status": 9}, {"reason": ""},
                       {"evidence": "prediction"}, {"mutates_live_state": True},
                       {"mutates_live_state": 0}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                scoped_modeled_evidence("FORT_SCOPED evidence " + json.dumps({**valid, **change}) + "\n")

    def test_completed_transfer_statistics_are_preserved_without_double_counting(self):
        counts = ("fallbacks", "pinned_uploads", "pinned_downloads", "pinned_upload_bytes", "pinned_download_bytes",
                  "packed_bytes", "unpacked_bytes", "tiles", "events", "event_waits", "staging_allocations",
                  "staging_reuses", "slot_capacity", "staging_device_bytes", "process_reserved_bytes", "process_peak_bytes")
        record = {"schema_version": 1, "context": 5, "event": "transfer_statistics", "stats_version": 1,
                  "complete": True, "requested_mode": 1, "effective_mode": 1, "fallback_reason": 0,
                  "reason": "none", **dict.fromkeys(counts, 0), "pinned_upload_bytes": 256,
                  "packing_seconds": 0.01, "unpacking_seconds": 0.02, "event_wait_seconds": 0.03}
        activity = gpu_activity("FORT_SCOPED evidence " + json.dumps(record) + "\n", "local", [])
        self.assertEqual(activity["scoped_runtime"]["modeled_evidence"], [record])
        self.assertEqual(activity["upload_bytes"], 0)
        self.assertFalse(activity["verified"])
        for effective_mode, fallback_reason, reason in (
                (4, 0, "none"),
                (4, 5, "coalesced_upload_ineligible"),
                (0, 5, "coalesced_upload_ineligible"),
                (0, 1, "transfer_estimates_unavailable")):
            coalesced = {**record, "requested_mode": 4, "effective_mode": effective_mode,
                         "fallback_reason": fallback_reason, "reason": reason,
                         "pinned_upload_bytes": 0}
            log = ("FORT_SCOPED evidence " + json.dumps(coalesced) + "\n"
                   "FORT_SCOPED upload buffer=12 generation=3 bytes=288\n"
                   "FORT_SCOPED upload_coalesced buffer=12 generation=3 bytes=288\n"
                   "FORT_SCOPED upload_requested buffer=12 generation=3 bytes=256\n"
                   "FORT_SCOPED launch bytes=0\n")
            with self.subTest(effective_mode=effective_mode, fallback_reason=fallback_reason):
                parsed = gpu_activity(log, "local", [])
                self.assertEqual(parsed["scoped_runtime"]["modeled_evidence"], [coalesced])
                self.assertEqual(parsed["upload_bytes"], 288)
                self.assertEqual(parsed["kernel_launches"], 1)
                self.assertTrue(parsed["verified"])
        for change in ({"stats_version": True}, {"complete": False}, {"effective_mode": 5},
                       {"requested_mode": 5}, {"fallback_reason": 6},
                       {"pinned_upload_bytes": -1}, {"process_peak_bytes": 2**64}, {"packing_seconds": -1},
                       {"event_wait_seconds": float("nan")}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                scoped_modeled_evidence("FORT_SCOPED evidence " + json.dumps({**record, **change}) + "\n")

    def test_finalized_batch_statistics_preserve_costs_without_inventing_kernel_evidence(self):
        counts = ("preparation_operations", "batches", "chunk_iterations", "slot_bytes", "upload_bytes",
                  "download_bytes", "prefix_upload_bytes", "prefix_uploads", "launches", "completed_batches",
                  "actual_upload_bytes", "actual_download_bytes", "actual_launches")
        record = {"schema_version": 1, "context": 6, "event": "batch_statistics", "stats_version": 1,
                  "available": 1, "applied": 1, "selected_transfers": 2, "reason": "none",
                  "owner_cost_available": 0, "owner_cost_reason": "complete_owner_batch_reprice_unavailable",
                  **dict.fromkeys(counts, 0), "batches": 2, "completed_batches": 2,
                  "actual_upload_bytes": 1024, "actual_launches": 2, "baseline_seconds": 0.01,
                  "pinned_seconds": 0.012, "pipelined_seconds": 0.006, "execution_seconds": 0.006,
                  "terminal_delta_seconds": -0.008, "estimated_seconds": -0.002}
        activity = gpu_activity("FORT_SCOPED evidence " + json.dumps(record) + "\n", "local", [])
        self.assertEqual(activity["scoped_runtime"]["modeled_evidence"], [record])
        self.assertEqual(activity["upload_bytes"], 0)
        self.assertFalse(activity["verified"])
        for change in ({"stats_version": True}, {"applied": True}, {"available": 0},
                       {"completed_batches": 1}, {"selected_transfers": 3}, {"slot_bytes": 2**64},
                       {"actual_launches": -1}, {"execution_seconds": -1},
                       {"terminal_delta_seconds": float("inf")}, {"owner_cost_reason": ""}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                scoped_modeled_evidence("FORT_SCOPED evidence " + json.dumps({**record, **change}) + "\n")


    def test_current_runtime_statistics_preserve_complete_and_abandoned_records(self):
        statistics = {
            "completion_statistics": ("event_records", "event_queries", "event_waits", "stream_waits",
                                      "created_events", "reused_events", "independent_host_tasks"),
            "command_queue_statistics": ("accepted_tasks", "submissions", "conflict_flushes", "explicit_flushes",
                                         "pending_tasks", "peak_pending_tasks", "copied_argument_bytes",
                                         "direct_submissions", "direct_kernels", "graph_resource_fallbacks"),
            "graph_statistics": ("creations", "instantiations", "rebuilds", "submissions", "replays",
                                 "parameter_updates", "logical_kernels", "live_graphs", "destructions"),
        }
        records = []
        for context, (event, counters) in enumerate(statistics.items(), start=5):
            for complete in (True, False):
                record = {"schema_version": 1, "context": context, "event": event, "stats_version": 1,
                          "complete": complete, **dict.fromkeys(counters, 0)}
                record[counters[0]] = 2**64 - 1
                if event == "command_queue_statistics":
                    record.update(queue_enabled=int(complete), graphs_enabled=int(not complete),
                                  pending_tasks=int(not complete), peak_pending_tasks=1)
                records.append(record)
        log = "".join("FORT_SCOPED evidence " + json.dumps(record) + "\n" for record in records)
        activity = gpu_activity(log, "local", [])
        self.assertEqual(activity["scoped_runtime"]["modeled_evidence"], records)
        self.assertFalse(activity["verified"])
        self.assertEqual(activity["kernel_launches"], 0)
        self.assertEqual(activity["allocations"], 0)
        self.assertEqual((activity["upload_bytes"], activity["download_bytes"]), (0, 0))

    def test_current_runtime_statistics_reject_missing_unknown_or_wrong_typed_fields(self):
        statistics = {
            "completion_statistics": ("event_records", "event_queries", "event_waits", "stream_waits",
                                      "created_events", "reused_events", "independent_host_tasks"),
            "command_queue_statistics": ("accepted_tasks", "submissions", "conflict_flushes", "explicit_flushes",
                                         "pending_tasks", "peak_pending_tasks", "copied_argument_bytes",
                                         "direct_submissions", "direct_kernels", "graph_resource_fallbacks"),
            "graph_statistics": ("creations", "instantiations", "rebuilds", "submissions", "replays",
                                 "parameter_updates", "logical_kernels", "live_graphs", "destructions"),
        }
        for event, counters in statistics.items():
            valid = {"schema_version": 1, "context": 5, "event": event, "stats_version": 1,
                     "complete": True, **dict.fromkeys(counters, 0)}
            flags = ("queue_enabled", "graphs_enabled") if event == "command_queue_statistics" else ()
            valid.update(dict.fromkeys(flags, 1))
            changes = [{"stats_version": value} for value in (True, 0, 2, "1")]
            changes += [{"complete": value} for value in (0, 1, "true", None)]
            changes += [{"context": value} for value in (True, -1, 2**64, "5")]
            changes += [{"schema_version": value} for value in (True, 0, 2)]
            changes += [{field: value} for field in counters for value in (True, -1, 2**64, 0.0)]
            changes += [{field: value} for field in flags for value in (True, -1, 2, 0.0)]
            changes.append({"unknown_counter": 0})
            for change in changes:
                with self.subTest(event=event, change=change), self.assertRaises(ValueError):
                    scoped_modeled_evidence("FORT_SCOPED evidence " + json.dumps({**valid, **change}) + "\n")
            for field in valid:
                missing = {key: value for key, value in valid.items() if key != field}
                with self.subTest(event=event, missing=field), self.assertRaises(ValueError):
                    scoped_modeled_evidence("FORT_SCOPED evidence " + json.dumps(missing) + "\n")
            encoded = json.dumps(valid)
            duplicate = encoded[:-1] + ', "complete": false}'
            nonfinite = encoded.replace('"' + counters[0] + '": 0', '"' + counters[0] + '": NaN')
            for malformed in (duplicate, nonfinite):
                with self.subTest(event=event, malformed=malformed), self.assertRaises(ValueError):
                    scoped_modeled_evidence("FORT_SCOPED evidence " + malformed + "\n")

    def test_runtime_statistics_do_not_double_count_actual_launches_and_transfers(self):
        record = {"schema_version": 1, "context": 1, "event": "command_queue_statistics", "stats_version": 1,
                  "complete": True, "queue_enabled": 1, "graphs_enabled": 1, "accepted_tasks": 1,
                  "submissions": 1, "conflict_flushes": 0, "explicit_flushes": 1, "pending_tasks": 0,
                  "peak_pending_tasks": 1, "copied_argument_bytes": 68, "direct_submissions": 1,
                  "direct_kernels": 1, "graph_resource_fallbacks": 0}
        log = ("FORT_SCOPED evidence " + json.dumps(record) + "\n"
               "FORT_SCOPED upload buffer=4 generation=1 bytes=64\n"
               "FORT_SCOPED download buffer=4 generation=1 bytes=32\n"
               "FORT_SCOPED launch bytes=0\n")
        activity = gpu_activity(log, "local", [])
        self.assertEqual(activity["scoped_runtime"]["modeled_evidence"], [record])
        self.assertTrue(activity["verified"])
        self.assertEqual(activity["kernel_launches"], 1)
        self.assertEqual((activity["upload_bytes"], activity["download_bytes"]), (64, 32))


def scope_package(entry, *, available=False):
    return {"scope_manifest": {
        "schema_version": 1, "scope_count": 1, "automatic_estimate_available": available,
        "scopes": [{"mode": "auto", "estimate_available": available, "gpu_leaves": [f"renamed::{entry}"]}]}}


def scoped_decision(**changes):
    fields = {"available": 1, "gpu_units": 1, "cpu_units": 1, "candidates": 4,
              "simulated_operations": 10, "upload_bytes": 0, "download_bytes": 0,
              "launches": 1, "waits": 1, "peak_device_bytes": 4096,
              "estimated_seconds": 0.01, "native_seconds": 0.02, "reason": "calibrated_selection"}
    fields.update(changes)
    return "FORT_SCOPED decision " + " ".join(f"{name}={value}" for name, value in fields.items()) + "\n"


if __name__ == "__main__":
    unittest.main()
