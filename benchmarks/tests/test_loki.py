"""Scope and emission regressions for the deliberately narrow Loki recipe.

Run with the pinned Loki environment:
  /tmp/loki-comparison-env/bin/python -m unittest discover \
      -s benchmarks/tests -p test_loki.py -v

The shared comparison runner checks numerical results on CPU and GPU.
"""

# Stdlib unittest keeps recipe checks runnable without another dependency.
# ruff: noqa: PT009, PT027
from __future__ import annotations

import importlib.util
import unittest

from benchmarks.harness.paths import CASES

HAS_LOKI = importlib.util.find_spec("loki") is not None
if HAS_LOKI:
    from loki import FP, FindVariables, Loop, Pragma, Sourcefile

    from benchmarks.tools.loki import transform

SOURCE = CASES / "CDU/Fortran/cdu.f90"


@unittest.skipUnless(HAS_LOKI, "requires pinned Loki environment")
class LokiRecipeTests(unittest.TestCase):
    def assert_rejected_change(self, old, new, message):
        original = SOURCE.read_text()
        changed = original.replace(old, new)
        self.assertNotEqual(original, changed)
        with self.assertRaisesRegex(ValueError, message):
            transform(changed, "CDU")

    def test_rejects_neighbor_output_access(self):
        old = "arr(i,j,k) = arr(i,j,k) * val"
        for new in (
            "arr(i+1,j,k) = arr(i,j,k) * val",
            "arr(i,j,k) = arr(i+1,j,k) * val",
        ):
            with self.subTest(statement=new):
                self.assert_rejected_change(old, new, "output must only be accessed pointwise")

    def test_rejects_modified_input_intent(self):
        self.assert_rejected_change(
            "real(knd), contiguous, intent(in)  :: U(:,:,:), V(:,:,:), W(:,:,:)",
            "real(knd), contiguous, intent(inout)  :: U(:,:,:), V(:,:,:), W(:,:,:)",
            "one output and three input arrays",
        )

    def test_rejects_changed_call_graph(self):
        self.assert_rejected_change("call set(U2, zero, Unx, Uny, Unz)", "", "Expected entry calls")

    def test_fused_output_privatizes_temporaries_and_preserves_indices(self):
        generated, metadata = transform(SOURCE.read_text(), "CDU", target="openacc")
        self.assertEqual(metadata["loop_nests"], 1)
        routine = Sourcefile.from_source(generated, frontend=FP)["MomentumAdvection"]["CDU"]
        loops = [node for node in routine.body.body if isinstance(node, Loop)]
        self.assertEqual(len(loops), 1)
        directives = [
            "".join(node.content.lower().split())
            for node in routine.body.body
            if isinstance(node, Pragma) and node.keyword.lower() == "acc"
        ]
        self.assertTrue(any("private(vadv,wadv)" in directive for directive in directives))
        # The pinned inliner and fusion utility must agree on index names.
        # Unrenamed helper indices once produced compilable, incorrect GPU
        # code; checking the emitted tree catches this without launching it.
        outputs = [v for v in FindVariables().visit(loops[0]) if v.name.lower() == "u2"]
        self.assertTrue(outputs)
        for variable in outputs:
            self.assertEqual(tuple(str(d).lower() for d in variable.dimensions), ("i", "j", "k"))


if __name__ == "__main__":
    unittest.main()
