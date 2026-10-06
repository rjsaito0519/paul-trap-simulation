"""Validation of the C++ floquet_boundary tracer (Phase 3).

Requires ./build.sh first; skipped when the executable is missing.
Run from the repository root:  python3 -m unittest discover -s tests
"""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.special import mathieu_a, mathieu_b

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import floquet_reference as fr  # noqa: E402

BOUNDARY = REPO / ".build" / "bin" / "floquet_boundary"
TARGET = 1e-6  # boundary position target (docs/simulation/validation.md)


def run_boundary(out, *args):
    subprocess.run([str(BOUNDARY), "--output", str(out), *map(str, args)],
                   check=True, capture_output=True)
    with open(out / "boundary.csv", newline="") as f:
        return list(csv.DictReader(f))


def trap_margin(a_z, q_z, b):
    a_r, q_r = fr.radial_params(a_z, q_z)
    margin = []
    for a, q in ((a_z, q_z), (a_r, q_r)):
        m = fr.monodromy(a, q, b)
        margin.append(abs(np.trace(m)) - 2.0 if b == 0.0
                      else np.log(np.abs(fr.floquet_multipliers(m)).max()))
    return max(margin)


@unittest.skipUnless(BOUNDARY.exists(), "floquet_boundary not built")
class TestBoundary(unittest.TestCase):
    def test_b0_matches_mathieu_characteristic_values(self):
        # Every b = 0 boundary point must lie on a characteristic curve a_n(|q|) or b_n(|q|)
        # of the limiting direction (independent reference: scipy.special).
        with tempfile.TemporaryDirectory() as tmp:
            rows = run_boundary(Path(tmp), "--b-range", 0, 0, "--nb", 1, "--a-range", -1.5, 1.5,
                                "--q-range", 0, 4, "--na", 61, "--nq", 81)
        self.assertGreater(len(rows), 20)
        for r in rows:
            a_z, q_z = float(r["a_z"]), float(r["q_z"])
            axial = r["limiting"] == "axial"
            a, q, scale = (a_z, q_z, 1.0) if axial else (-a_z / 2, -q_z / 2, 2.0)
            if q_z < 0.05:
                continue  # all boundaries meet at the origin; see validation.md
            q = abs(q)
            curves = [mathieu_a(n, q) for n in range(12)] + [mathieu_b(n, q) for n in range(1, 12)]
            self.assertLess(scale * min(abs(a - c) for c in curves), TARGET, msg=str(r))

    def test_damped_points_bracket_reference_sign_change(self):
        # Moving 1e-6 either way along the refined grid edge must change the sign of the
        # Python reference margin.
        with tempfile.TemporaryDirectory() as tmp:
            rows = run_boundary(Path(tmp), "--b-range", 2, 6, "--nb", 2, "--a-range", -1, 1,
                                "--q-range", 0.5, 12, "--na", 41, "--nq", 41)
        rng = np.random.default_rng(0)
        for r in rng.choice(rows, size=min(40, len(rows)), replace=False):
            a_z, q_z, b = float(r["a_z"]), float(r["q_z"]), float(r["b"])
            da, dq = (TARGET, 0.0) if r["edge_axis"] == "a" else (0.0, TARGET)
            lo = trap_margin(a_z - da, q_z - dq, b)
            hi = trap_margin(a_z + da, q_z + dq, b)
            self.assertLess(lo * hi, 0.0, msg=str(r))

    def test_thread_count_does_not_change_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = ("--b-range", 0, 4, "--nb", 3, "--na", 81, "--nq", 81, "--q-range", 0, 10)
            r1 = run_boundary(Path(tmp) / "t1", *args, "--threads", 1)
            r8 = run_boundary(Path(tmp) / "t8", *args, "--threads", 8)
        self.assertEqual(r1, r8)

    def test_polylines_are_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = run_boundary(Path(tmp), "--b-range", 0, 6, "--nb", 4, "--na", 81, "--nq", 81,
                                "--q-range", 0, 20)
        seen = {}
        for r in rows:
            key = (r["b"], r["polyline"])
            idx = int(r["index"])
            # Indices run 0, 1, 2, ... within each polyline, and each polyline borders one region.
            self.assertEqual(idx, seen.get(key, (-1, None))[0] + 1)
            if key in seen:
                self.assertEqual(r["region"], seen[key][1])
            seen[key] = (idx, r["region"])


if __name__ == "__main__":
    unittest.main()
