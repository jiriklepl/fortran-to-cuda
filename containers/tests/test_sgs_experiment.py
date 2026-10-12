"""Small checks for portable proof inputs and diagnostic/sample separation."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("sgs_experiment", Path(__file__).resolve().parents[1] / "sgs_experiment.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PortableSGSChecks(unittest.TestCase):
    def test_changed_include_rejects_contract_before_rebinding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("ELMM", "PoisFFT"):
                (root / name / "src").mkdir(parents=True)
                (root / name / "src/SConstruct").write_text("pinned build\n")
            source = root / "ELMM/src/subgrid.f90"
            include = root / "ELMM/src/kinds.inc"
            source.write_text("pinned source\n")
            include.write_text("pinned include\n")
            contract = {"schema_version": 1, "entry": "subgrid::sgs_sigma_uniform", "defines": ["DPREC"],
                        "source": "elmm/src/subgrid.f90", "source_sha256": {"elmm/src/subgrid.f90": runner.digest(source)},
                        "include_sha256": {"elmm/src/kinds.inc": runner.digest(include)}, "build_sha256": {}}
            runner.verify_contract(root, contract)
            include.write_text("changed caller proof\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                runner.verify_contract(root, contract)

    def test_contract_cannot_escape_application_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "invalid portable"):
                runner.contract_path({"elmm": root}, "elmm/../outside.f90")

    def test_whole_checkout_rejects_wrong_pin_dirty_or_untracked_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            source = root / "src/native_only.f90"
            source.write_text("outside the SGS proof closure\n")
            contract = {"repositories": {"elmm": {"url": "example.invalid/elmm", "revision": "pinned"}}}
            answers = {("rev-parse", "--show-toplevel"): str(root),
                       ("rev-parse", "--verify", "HEAD"): "pinned",
                       ("status", "--porcelain=v1", "--untracked-files=no", "--ignore-submodules=none"): "",
                       ("ls-files", "--others", "-z"): "scratch.o\0",
                       ("ls-files", "-z"): "src/native_only.f90\0"}

            def command(argv, **_kwargs):
                return answers[tuple(argv[3:])], 0.

            provenance = runner.repository_provenance({"elmm": root}, contract, command)
            self.assertEqual(provenance["repositories"]["elmm"]["tracked_sha256"],
                             {"src/native_only.f90": runner.digest(source)})
            answers[("rev-parse", "--verify", "HEAD")] = "other"
            with self.assertRaisesRegex(ValueError, "revision mismatch"):
                runner.repository_provenance({"elmm": root}, contract, command)
            answers[("rev-parse", "--verify", "HEAD")] = "pinned"
            status_key = ("status", "--porcelain=v1", "--untracked-files=no", "--ignore-submodules=none")
            answers[status_key] = " M src/native_only.f90"
            with self.assertRaisesRegex(ValueError, "changed tracked"):
                runner.repository_provenance({"elmm": root}, contract, command)
            answers[status_key] = ""
            answers[("ls-files", "--others", "-z")] = "scratch.o\0src/ignored.f90\0"
            with self.assertRaisesRegex(ValueError, "untracked input"):
                runner.repository_provenance({"elmm": root}, contract, command)

    def test_warmup_excluded_and_diagnostic_cannot_be_a_timing_sample(self):
        rows = []
        for mode, value in (("native", 10.), ("current", 8.)):
            rows.extend([{"mode": mode, "warmup": True, "traced": False, "wall_seconds": 100.},
                         {"mode": mode, "warmup": False, "traced": False, "wall_seconds": value}])
        self.assertEqual(runner.summarize(rows)["current"]["ratio_to_native"], .8)
        rows[-1]["traced"] = True
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            runner.summarize(rows)

    def test_averaged_fields_are_required_when_native_emits_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "out.vtk").touch()
            (root / "avg.vtk").touch()
            reference = runner.read_outputs(root, lambda path: path.name)
            self.assertEqual(set(reference), {"out.vtk", "avg.vtk"})
            (root / "avg.vtk").unlink()
            observed = runner.read_outputs(root, lambda path: path.name)
            with self.assertRaisesRegex(ValueError, "output files differ"):
                runner.compare_outputs(reference, observed, lambda *a, **kw: {"passed": True})

    def test_averaged_only_mismatch_fails_complete_output_validation(self):
        reference = {"out.vtk": 1., "avg.vtk": 2.}
        observed = {"out.vtk": 1., "avg.vtk": 3.}
        def compare(a, b, *, atol, rtol):
            self.assertEqual((atol, rtol), (1e-6, 1e-5))
            return {"passed": a == b}
        result = runner.compare_outputs(reference, observed, compare)
        self.assertFalse(result["passed"])
        self.assertTrue(result["files"]["out.vtk"]["passed"])
        self.assertFalse(result["files"]["avg.vtk"]["passed"])


if __name__ == "__main__":
    unittest.main()
