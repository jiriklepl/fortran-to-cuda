"""ELMM case configuration, application logs and VTK output: portable checks without external tools."""

from __future__ import annotations

import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configuration import prepare_config  # noqa: E402
from validation import compare_vtk, read_vtk, validate_log  # noqa: E402


def digest_tree(path):
    return {str(item.relative_to(path)): item.read_bytes() for item in sorted(Path(path).rglob("*")) if item.is_file()}


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="elmm-config-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "a config with spaces"
        self.source.mkdir()
        (self.source / "main.conf").write_text("! chosen scheme\n4 ! fourth order\nother setting\n")
        (self.source / "grid.conf").write_text("F\nF\nF\n128\n64\n64\n")
        (self.source / "time_stepping.conf").write_text(
            "time_stepping(\n  end_time = 50,\n! max_number_of_time_steps = 3000,\n)\n")
        self.destination = self.root / "snapshot"

    def test_custom_configuration_file_path_copies_all_assets_and_dereferences_links(self):
        assets = self.source / "input" / "nested"
        assets.mkdir(parents=True)
        (assets / "profile.bin").write_bytes(b"\x00\x10\xff\x80")
        (self.source / "custom geometry.f90").write_text("subroutine custom\nend subroutine\n")
        external = self.root / "external.dat"
        external.write_bytes(b"external profile contents")
        (self.source / "input" / "external.dat").symlink_to(external)
        original = digest_tree(self.source)
        metadata = prepare_config(self.source / "main.conf", self.destination)
        self.assertEqual(metadata["source"], str(self.source))
        self.assertEqual(metadata["overrides"], {})
        self.assertEqual(digest_tree(self.source), original)
        self.assertEqual((self.destination / "input/nested/profile.bin").read_bytes(), b"\x00\x10\xff\x80")
        self.assertEqual((self.destination / "custom geometry.f90").read_text(), "subroutine custom\nend subroutine\n")
        link_copy = self.destination / "input/external.dat"
        self.assertFalse(link_copy.is_symlink())
        external.unlink()
        self.assertEqual(link_copy.read_bytes(), b"external profile contents")
        # Input contents are unchanged; removing the external file is outside
        # the input tree, so compare actual input files explicitly here.
        self.assertEqual((self.source / "main.conf").read_bytes(), (self.destination / "main.conf").read_bytes())

    def test_default_does_not_silently_change_method_or_duration(self):
        originals = {p.name: p.read_bytes() for p in self.source.iterdir()}
        metadata = prepare_config(self.source, self.destination)
        self.assertEqual(metadata["advection_method"], 4)
        self.assertNotIn("transformation_supported", metadata)
        self.assertIsNone(metadata["expected_max_steps"])
        self.assertEqual(metadata["overrides"], {})
        self.assertEqual(originals, {p.name: p.read_bytes() for p in self.source.iterdir()})
        self.assertEqual(originals, {p.name: p.read_bytes() for p in self.destination.iterdir()})

    def test_explicit_cap_preserves_commented_examples_and_original_input(self):
        path = self.source / "time_stepping.conf"
        original = path.read_text() + "# max_number_of_time_steps = 10,\n"
        path.write_text(original)
        metadata = prepare_config(self.source, self.destination, steps=7, advection=2)
        staged = (self.destination / path.name).read_text()
        self.assertIn("! max_number_of_time_steps = 3000,", staged)
        self.assertIn("# max_number_of_time_steps = 10,", staged)
        self.assertIn("max_number_of_time_steps = 7,", staged)
        self.assertEqual(path.read_text(), original)
        self.assertEqual(metadata["expected_max_steps"], 7)
        self.assertEqual(metadata["overrides"]["max_number_of_time_steps"], 7)
        self.assertEqual(metadata["overrides"]["advection_method"], {"original": 4, "effective": 2})
        self.assertNotIn("transformation_supported", metadata)

    def test_disabled_step_limits_are_not_treated_as_active(self):
        path = self.source / "time_stepping.conf"
        path.write_text("time_stepping(\n! max_number_of_time_steps = 3000,\n"
                        "# max_number_of_time_steps = 10,\n end_time = 50,\n)\n")
        metadata = prepare_config(self.source, self.destination)
        self.assertIsNone(metadata["expected_max_steps"])

    def test_inline_step_limit_is_replaced_without_duplicate(self):
        path = self.source / "time_stepping.conf"
        path.write_text("time_stepping(max_number_of_time_steps = 99, end_time = 50)\n")
        metadata = prepare_config(self.source, self.destination, steps=7)
        staged = (self.destination / path.name).read_text()
        self.assertEqual(staged.count("max_number_of_time_steps"), 1)
        self.assertIn("max_number_of_time_steps = 7,", staged)
        self.assertEqual(metadata["expected_max_steps"], 7)

    def test_multiline_step_assignment_is_replaced(self):
        path = self.source / "time_stepping.conf"
        path.write_text("time_stepping(max_number_of_time_steps =\n 99, end_time = 50)\n")
        metadata = prepare_config(self.source, self.destination, steps=7)
        self.assertEqual(metadata["expected_max_steps"], 7)
        self.assertNotIn("99", (self.destination / path.name).read_text())

    def test_duplicate_active_step_limits_are_rejected(self):
        (self.source / "time_stepping.conf").write_text(
            "time_stepping(max_number_of_time_steps=99, max_number_of_time_steps=6)\n")
        with self.assertRaisesRegex(ValueError, "Duplicate active"):
            prepare_config(self.source, self.destination, steps=7)

    def test_existing_snapshot_is_not_overwritten(self):
        self.destination.mkdir()
        marker = self.destination / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(ValueError):
            prepare_config(self.source, self.destination)
        self.assertEqual(marker.read_text(), "keep")

    def test_discretization_is_reported_without_deciding_structural_eligibility(self):
        (self.source / "cmd.conf").write_text("discretization_order=4\n")
        metadata = prepare_config(self.source, self.destination, advection=2)
        self.assertEqual(metadata["advection_method"], 4)
        self.assertNotIn("transformation_supported", metadata)

    def test_output_location_is_pinned_to_fresh_run_directory(self):
        (self.source / "cmd.conf").write_text("output_dir='/some/external/directory/'\n")
        metadata = prepare_config(self.source, self.destination)
        self.assertIn("output_dir='output/'", metadata["runtime_arguments"])

    def test_snapshot_inside_input_is_rejected_before_recursive_copy(self):
        with self.assertRaises(ValueError):
            prepare_config(self.source, self.source / "results" / "config")

    def test_grid_domain_time_stepping_and_re_overrides(self):
        (self.source / "grid.conf").write_text(
            "F\nF\nF\n-6\n-2\n0\n12\n4\n2\n128\n64\n64\n"
        )
        (self.source / "main.conf").write_text(
            "\n".join(["4", "3", "1.5", "1", "0", "0", "0", "40", "50", "70000", "1E-7", "8", "1E-9", "0.0001"]) + "\n"
        )
        metadata = prepare_config(
            self.source,
            self.destination,
            grid="64x32x32",
            domain_size="6,2,1",
            end_time=1.5,
            cfl=0.25,
            re_number=10000,
        )
        self.assertEqual(metadata["overrides"]["grid"], {"nx": 64, "ny": 32, "nz": 32})
        self.assertEqual(metadata["overrides"]["domain_size"], {"lx": 6.0, "ly": 2.0, "lz": 1.0})
        self.assertEqual(metadata["overrides"]["end_time"], 1.5)
        self.assertEqual(metadata["overrides"]["cfl"], 0.25)
        self.assertEqual(metadata["overrides"]["re"], 10000.0)
        grid_lines = [line.split("!")[0].strip() for line in (self.destination / "grid.conf").read_text().splitlines()
                      if line.split("!")[0].strip()]
        self.assertEqual(grid_lines[6:12], ["6", "2", "1", "64", "32", "32"])
        main_lines = [line.split("!")[0].strip() for line in (self.destination / "main.conf").read_text().splitlines()
                      if line.split("!")[0].strip()]
        self.assertEqual(main_lines[9], "10000")
        ts_text = (self.destination / "time_stepping.conf").read_text()
        self.assertIn("end_time = 1.5", ts_text)
        self.assertIn("CFL = 0.25", ts_text)


def successful_log(steps=7, ending=""):
    return (f"tstep: 1\ntstep: {steps}\n{ending}\n"
            "Total wall clock time for time steps: 1.23\n"
            "Wall clock time for Poisson solver: 0.41\nsaved.\n")


class ApplicationLogTests(unittest.TestCase):
    def test_step_capped_completion_has_timings(self):
        record = validate_log(successful_log(), expected_max_steps=7)
        self.assertEqual(record["completed_steps"], 7)
        self.assertEqual(record["time_steps_seconds"], 1.23)
        self.assertEqual(record["poisson_seconds"], 0.41)
        self.assertFalse(record["natural_end"])

    def test_abort_markers_fail_even_if_process_returned_zero_and_saved(self):
        for marker in ("ELMM Abort: bad configuration", "Solution diverged.",
                       "Maximum clock time exceeded.", "Fortran runtime error: bounds", "ERROR STOP 1"):
            with self.subTest(marker=marker):
                with self.assertRaises(ValueError):
                    validate_log(successful_log() + marker, expected_max_steps=7)

    def test_early_stop_is_not_a_successful_measurement(self):
        with self.assertRaisesRegex(ValueError, "stopped"):
            validate_log(successful_log(3) + "STOP\n", expected_max_steps=7)

    def test_missing_final_save_or_timing_is_rejected(self):
        for text in (successful_log().replace("saved.", ""),
                     successful_log().replace("saved.", "stagframes.conf not found, no staggered frames will be saved."),
                     successful_log().replace("Total wall clock time for time steps: 1.23", "")):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    validate_log(text, expected_max_steps=7)

    def test_natural_end_can_precede_explicit_cap(self):
        record = validate_log(successful_log(3, "Time limit reached."), expected_max_steps=7)
        self.assertTrue(record["natural_end"])
        self.assertEqual(record["completed_steps"], 3)

    def test_natural_end_must_still_match_baseline_workload(self):
        with self.assertRaisesRegex(ValueError, "Workload differs from baseline"):
            validate_log(successful_log(3, "Time limit reached."), expected_max_steps=7,
                         expected_completed_steps=7)
        self.assertEqual(validate_log(successful_log(3, "Time limit reached."),
                                     expected_max_steps=7, expected_completed_steps=3)["completed_steps"], 3)

    def test_no_positive_timestep_is_rejected(self):
        text = successful_log(0, "Time limit reached.").replace("tstep: 1\n", "")
        with self.assertRaises(ValueError):
            validate_log(text)

    def test_phase_timings_are_parsed_when_present_and_optional_otherwise(self):
        legacy = validate_log(successful_log(), expected_max_steps=7)
        self.assertIsNone(legacy["initialization_seconds"])
        self.assertIsNone(legacy["loop_seconds"])
        self.assertIsNone(legacy["finalization_seconds"])

        with_phases = (
            successful_log()
            + " Wall clock time for initialization: .125000\n"
            + " Wall clock time for loop: 1.230000\n"
            + " Wall clock time for finalization: .045000\n"
        )
        record = validate_log(with_phases, expected_max_steps=7)
        self.assertAlmostEqual(record["initialization_seconds"], 0.125)
        self.assertAlmostEqual(record["loop_seconds"], 1.23)
        self.assertAlmostEqual(record["finalization_seconds"], 0.045)


def vtk_bytes(*, pressure=(2.0, -3.0), velocity=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)):
    def values(items):
        return struct.pack(">" + "f" * len(items), *items) + b"\n"
    return b"".join([
        b"# vtk DataFile Version 2.0\nportable fixture\nBINARY\nDATASET RECTILINEAR_GRID\n",
        b"DIMENSIONS 2 1 1\nX_COORDINATES 2 float\n", values((0.0, 1.0)),
        b"Y_COORDINATES 1 float\n", values((0.0,)),
        b"Z_COORDINATES 1 float\n", values((0.0,)),
        b"POINT_DATA 2\nSCALARS p float\nLOOKUP_TABLE default\n", values(pressure),
        b"VECTORS u float\n", values(velocity),
    ])


class VTKTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="elmm-vtk-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def read(self, name="out.vtk", content=None):
        path = self.root / name
        path.write_bytes(vtk_bytes() if content is None else content)
        return read_vtk(path)

    def test_big_endian_values_and_all_fields_are_compared(self):
        reference = self.read()
        self.assertEqual(reference[0], (2, 1, 1))
        self.assertEqual(list(reference[1]["p"]), [2.0, -3.0])
        self.assertEqual(list(reference[1]["u"]), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        comparison = compare_vtk(reference, self.read("same.vtk"))
        self.assertTrue(comparison["passed"])
        self.assertEqual(comparison["fields"]["u"]["values"], 6)
        self.assertIn("X_COORDINATES", comparison["fields"])

    def test_changed_field_fails_and_reports_error(self):
        comparison = compare_vtk(self.read(), self.read("changed.vtk", vtk_bytes(pressure=(2.0, -2.5))))
        self.assertFalse(comparison["passed"])
        self.assertEqual(comparison["fields"]["p"]["failed_values"], 1)
        self.assertEqual(comparison["fields"]["p"]["max_absolute_error"], 0.5)

    def test_scalar_output_requires_explicit_nonfinal_mode(self):
        path = self.root / "U.vtk"
        path.write_bytes(vtk_bytes().split(b"VECTORS u float\n")[0])
        with self.assertRaisesRegex(ValueError, "velocity u and pressure p"):
            read_vtk(path)
        shape, fields = read_vtk(path, require_flow_fields=False)
        self.assertEqual(shape, (2, 1, 1))
        self.assertEqual(list(fields["p"]), [2.0, -3.0])

    def test_nonfinal_mode_still_requires_at_least_one_numeric_field(self):
        path = self.root / "empty.vtk"
        path.write_bytes(vtk_bytes().split(b"SCALARS p float\n")[0])
        with self.assertRaisesRegex(ValueError, "at least one numeric field"):
            read_vtk(path, require_flow_fields=False)

    def test_tolerance_is_applied_to_each_value(self):
        reference = self.read()
        candidate = self.read("nearby.vtk", vtk_bytes(pressure=(2.000001, -3.0)))
        self.assertTrue(compare_vtk(reference, candidate, atol=2e-6, rtol=0)["passed"])
        self.assertFalse(compare_vtk(reference, candidate, atol=1e-8, rtol=0)["passed"])

    def test_invalid_tolerances_cannot_turn_differences_into_passes(self):
        reference = self.read()
        candidate = self.read("changed.vtk", vtk_bytes(pressure=(200.0, -300.0)))
        for keyword in ("atol", "rtol"):
            for value in (math.nan, math.inf, -math.inf, -1.0):
                with self.subTest(keyword=keyword, value=value):
                    with self.assertRaises(ValueError):
                        compare_vtk(reference, candidate, **{keyword: value})

    def test_nonfinite_values_fail_before_comparison(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "[Nn]on-finite"):
                    self.read(content=vtk_bytes(pressure=(value, -3.0)))

    def test_direct_comparison_rejects_nonfinite_reference_and_candidate(self):
        reference = ((1, 1, 1), {"p": [1.], "u": [1., 2., 3.]})
        for value in (math.nan, math.inf, -math.inf):
            invalid = ((1, 1, 1), {"p": [value], "u": [1., 2., 3.]})
            for left, right in ((reference, invalid), (invalid, reference), (invalid, invalid)):
                with self.subTest(value=value, left=left, right=right):
                    with self.assertRaisesRegex(ValueError, "Non-finite comparison field"):
                        compare_vtk(left, right)

    def test_truncated_binary_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Truncated"):
            self.read(content=vtk_bytes()[:-8])

    def test_missing_binary_delimiter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "delimiter"):
            self.read(content=vtk_bytes()[:-1])

    def test_missing_velocity_and_wrong_point_count_are_rejected(self):
        variants = [vtk_bytes().split(b"VECTORS u float\n")[0],
                    vtk_bytes().replace(b"POINT_DATA 2", b"POINT_DATA 3")]
        for content in variants:
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    self.read(content=content)


if __name__ == "__main__":
    unittest.main()
