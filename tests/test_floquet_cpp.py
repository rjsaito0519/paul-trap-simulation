"""Regression of the C++ floquet_scan against scripts/floquet_reference.py.

Requires ./build.sh first; skipped when the executable is missing.
Run from the repository root:  python3 -m unittest discover -s tests
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import floquet_reference as fr  # noqa: E402

SCAN = REPO / ".build" / "bin" / "floquet_scan"


def run_scan(out, *args):
    subprocess.run([str(SCAN), "--output", str(out), *map(str, args)],
                   check=True, capture_output=True)
    return {name: np.load(out / f"{name}.npy") for name in
            ("a_z", "q_z", "axial", "radial", "combined", "det_error")
            if (out / f"{name}.npy").exists()}


@unittest.skipUnless(SCAN.exists(), "floquet_scan not built")
class TestCppAgainstReference(unittest.TestCase):
    CASES = [
        # (a_range, q_range, b)
        ((-0.4, 0.2), (0.0, 1.2), 0.0),
        ((-0.4, 0.2), (0.0, 1.6), 0.5),
        ((-1.0, 1.0), (0.0, 20.0), 6.0),
    ]

    def test_matrices_and_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            for k, (ar, qr, b) in enumerate(self.CASES):
                out = Path(tmp) / f"case{k}"
                d = run_scan(out, "--a-range", *ar, "--q-range", *qr, "--na", 9, "--nq", 9,
                             "--b", b, "--save-matrices", "--threads", 4)
                m_ax = np.load(out / "monodromy_axial.npy")
                m_ra = np.load(out / "monodromy_radial.npy")
                meta = json.loads((out / "metadata.json").read_text())
                self.assertEqual(meta["b"], b)
                for i, a_z in enumerate(d["a_z"]):
                    for j, q_z in enumerate(d["q_z"]):
                        a_r, q_r = fr.radial_params(a_z, q_z)
                        ref_ax = fr.monodromy(a_z, q_z, b)
                        ref_ra = fr.monodromy(a_r, q_r, b)
                        for got, ref in ((m_ax[i, j], ref_ax), (m_ra[i, j], ref_ra)):
                            scale = max(1.0, np.abs(ref).max())
                            np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9 * scale,
                                                       err_msg=f"a_z={a_z} q_z={q_z} b={b}")
                        self.assertEqual(d["axial"][i, j], fr.classify(ref_ax, b))
                        self.assertEqual(d["radial"][i, j], fr.classify(ref_ra, b))
                np.testing.assert_array_equal(d["combined"], np.maximum(d["axial"], d["radial"]))

    def test_thread_count_does_not_change_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = ("--na", 40, "--nq", 40, "--b", 0.2, "--save-matrices")
            d1 = run_scan(Path(tmp) / "t1", *args, "--threads", 1)
            d8 = run_scan(Path(tmp) / "t8", *args, "--threads", 8)
            for name in d1:
                np.testing.assert_array_equal(d1[name], d8[name], err_msg=name)
            for name in ("monodromy_axial", "monodromy_radial"):
                np.testing.assert_array_equal(np.load(Path(tmp) / "t1" / f"{name}.npy"),
                                              np.load(Path(tmp) / "t8" / f"{name}.npy"))


if __name__ == "__main__":
    unittest.main()
