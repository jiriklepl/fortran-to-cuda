"""Semantic regression checks for the deliberately narrow PSyclone recipe.

Run with the pinned PSyclone environment:
  /tmp/fortran-comparison-psyclone/bin/python -m unittest discover \
      -s benchmarks/tests -p test_psyclone.py -v

GPU and resident-data correctness are exercised by the shared comparison runner.
"""

# Stdlib unittest keeps recipe checks runnable without another dependency.
# ruff: noqa: PT009, PT027
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from benchmarks.harness.paths import CASES, SOURCES
from benchmarks.tools.psyclone import transform

HAS_PSYCLONE = importlib.util.find_spec("psyclone") is not None


@unittest.skipUnless(HAS_PSYCLONE, "requires pinned PSyclone environment")
class PSycloneRecipeTests(unittest.TestCase):
    def test_rejects_unproven_lower_bound(self):
        source = (CASES / "CDU/Fortran/cdu.f90").read_text()
        with self.assertRaisesRegex(ValueError, "rank-three assumed-shape"):
            transform(source.replace("U2(:,:,:)", "U2(0:,:,:)"), "CDU", target="cpu")

    def test_rejects_unexpected_call_graph(self):
        source = (CASES / "CDU/Fortran/cdu.f90").read_text()
        source = source.replace("call set(U2, zero, Unx, Uny, Unz)", "")
        with self.assertRaisesRegex(ValueError, "call sequence"):
            transform(source, "CDU", target="cpu")

    @unittest.skipUnless(shutil.which("gfortran"), "requires gfortran")
    def test_fusion_and_emission_preserve_all_interior_values(self):
        # Noncubic and singleton grids catch iteration-space, inlining-bound and
        # indexing mistakes; exact equality uses identical CPU compiler flags.
        for case, filename in SOURCES.items():
            source_path = CASES / case / "Fortran" / filename
            original = source_path.read_text()
            fused, info = transform(original, case, target="openacc", fuse=True)
            unfused, other_info = transform(original, case, target="openacc", fuse=False)
            self.assertEqual(info["loop_nests"], 1)
            self.assertEqual(other_info["loop_nests"], 4)
            self.assertEqual(info["loop_fusions"], 9)
            self.assertEqual(source_path.read_text(), original)
            self.assertIn("copyin(u,v,w)", fused)
            self.assertIn(f"copyout({case[-1].lower()}2)", fused)
            for shape in ((1, 1, 1), (7, 3, 5)):
                with self.subTest(case=case, shape=shape):
                    results = []
                    for code in (original, fused, unfused):
                        with tempfile.TemporaryDirectory(prefix="psyclone-test-") as directory:
                            path = Path(directory)
                            (path / "kernel.f90").write_text(code)
                            command = ["gfortran", "-O2", "-ffp-contract=off", "-fcheck=all", "-cpp"]
                            command += [f"-DVAR_N{axis}={extent}" for axis, extent in zip("XYZ", shape, strict=True)]
                            command += ["kernel.f90", str(CASES / case / "test_main.f90"), "-o", "test"]
                            subprocess.run(command, cwd=path, check=True, capture_output=True, text=True)
                            results.append(
                                subprocess.run([str(path / "test")], check=True, capture_output=True, text=True).stdout
                            )
                    self.assertEqual(results[0], results[1])
                    self.assertEqual(results[0], results[2])


if __name__ == "__main__":
    unittest.main()
