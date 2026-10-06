"""Tests for the Monte Carlo capture scan (Phase 5b).

Requires ./build.sh first; skipped when the executable is missing.
"""

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import trajectory as tj  # noqa: E402

SCAN = REPO / ".build" / "bin" / "capture_scan"
SETUP = tj.Setup(radius=5e-6, density=1050.0, charge=1.6e-14, viscosity=1.81e-5, r0=5e-3,
                 v_dc=0.0, v_ac=1000.0, frequency=200.0)
UNIT = tj.Setup(**{**SETUP.__dict__, "v_dc": 1.0, "v_ac": 1.0, "particles": []})
COEFF = ["--a-per-vdc", UNIT.a_z, "--q-per-vac", UNIT.q_z, "--b", SETUP.b,
         "--gravity-term", SETUP.gravity_term, "--r0", SETUP.r0, "--omega", SETUP.omega]
NAMES = ("v_dc", "v_ac", "n_trapped", "n_ring", "n_endcap", "p_capture", "wilson_lo", "wilson_hi",
         "mean_escape_time", "p_capture_at", "floquet_class")
SAMPLE_NAMES = ("sample_rf_phase", "sample_position", "sample_velocity", "sample_electrode",
                "sample_t_end")


def run_scan(out, *args):
    cmd = [str(SCAN), "--output", str(out)]
    cmd += [repr(float(a)) if isinstance(a, (float, np.floating)) else str(a) for a in COEFF + list(args)]
    subprocess.run(cmd, check=True, capture_output=True)
    return {n: np.load(out / f"{n}.npy") for n in NAMES + SAMPLE_NAMES if (out / f"{n}.npy").exists()}


@unittest.skipUnless(SCAN.exists(), "capture_scan not built")
class TestCaptureScan(unittest.TestCase):
    def test_deterministic_and_thread_independent(self):
        args = ("--vdc-range", -30, 30, "--n-vdc", 3, "--vac-range", 500, 5000, "--n-vac", 4,
                "--samples", 60, "--rf-periods", 30, "--pos-radius", 1e-3, "--vel-sigma", 0.5,
                "--seed", 7, "--save-samples")
        with tempfile.TemporaryDirectory() as tmp:
            a = run_scan(Path(tmp) / "a", *args, "--threads", 1)
            b = run_scan(Path(tmp) / "b", *args, "--threads", 8)
            c = run_scan(Path(tmp) / "c", *args[:-3], "--seed", 8, "--save-samples")
        for n in a:
            np.testing.assert_array_equal(a[n], b[n], err_msg=n)
        self.assertFalse(np.array_equal(a["sample_position"], c["sample_position"]))

    def test_counts_and_checkpoints_are_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = run_scan(Path(tmp), "--vac-range", 0, 6000, "--n-vac", 7, "--samples", 80,
                         "--rf-periods", 40, "--pos-radius", 1e-3, "--vel-sigma", 0.5)
        np.testing.assert_array_equal(d["n_trapped"] + d["n_ring"] + d["n_endcap"], 80)
        np.testing.assert_allclose(d["p_capture"], d["n_trapped"] / 80)
        np.testing.assert_allclose(d["p_capture_at"][-1], d["p_capture"])
        self.assertTrue(np.all(np.diff(d["p_capture_at"], axis=0) <= 0))
        self.assertTrue(np.all((d["wilson_lo"] <= d["p_capture"]) & (d["p_capture"] <= d["wilson_hi"])))

    def test_wilson_interval(self):
        z = 1.959963984540054
        with tempfile.TemporaryDirectory() as tmp:
            d = run_scan(Path(tmp), "--vac-range", 0, 6000, "--n-vac", 7, "--samples", 50,
                         "--rf-periods", 20, "--pos-radius", 1.5e-3, "--vel-sigma", 1.0)
        n = 50
        for k, lo, hi in zip(d["n_trapped"].ravel(), d["wilson_lo"].ravel(), d["wilson_hi"].ravel()):
            p = k / n
            centre = (p + z * z / (2 * n)) / (1 + z * z / n)
            half = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
            self.assertAlmostEqual(lo, max(0.0, centre - half), places=12)
            self.assertAlmostEqual(hi, min(1.0, centre + half), places=12)
            self.assertEqual(lo == 0.0, k == 0)
            self.assertEqual(hi == 1.0, k == n)

    def test_sampling_distributions(self):
        N = 20000
        centre, R, vmean, vsig = np.array([1e-4, -2e-4, 3e-4]), 1e-3, np.array([0.1, 0.0, -0.2]), 0.4
        with tempfile.TemporaryDirectory() as tmp:
            d = run_scan(Path(tmp), "--samples", N, "--rf-periods", 0.01, "--pos-center", *centre,
                         "--pos-radius", R, "--vel-mean", *vmean, "--vel-sigma", vsig, "--save-samples")
        ph, pos, vel = d["sample_rf_phase"], d["sample_position"], d["sample_velocity"]
        tol = 5.0 / math.sqrt(N)  # about 5 standard errors
        self.assertTrue(np.all((ph >= 0) & (ph < 2 * math.pi)))
        self.assertAlmostEqual(ph.mean() / math.pi, 1.0, delta=tol * 0.6)
        rel = np.linalg.norm(pos - centre, axis=1) / R
        self.assertLessEqual(rel.max(), 1.0)
        self.assertAlmostEqual(np.mean(rel < 0.5), 0.125, delta=tol * 0.35)  # volume fraction 1/8
        np.testing.assert_allclose(pos.mean(axis=0), centre, atol=tol * R * 0.5)
        np.testing.assert_allclose(vel.mean(axis=0), vmean, atol=tol * vsig)
        np.testing.assert_allclose(vel.std(axis=0), vsig, rtol=tol)

    def test_fates_match_python_reference(self):
        # Re-integrate the sampled initial conditions with scripts/trajectory.py.
        with tempfile.TemporaryDirectory() as tmp:
            d = run_scan(Path(tmp), "--vac-range", 1000, 1000, "--samples", 40, "--rf-periods", 30,
                         "--pos-radius", 2e-3, "--vel-sigma", 3.0, "--steps-per-rf", 800, "--save-samples")
        electrodes = {0: "", 1: "ring", 2: "endcap"}
        n_escaped = 0
        for k in range(40):
            s = tj.Setup(**{**SETUP.__dict__, "rf_phase": float(d["sample_rf_phase"][k]),
                            "duration": 30 / SETUP.frequency, "particles": []})
            tr = tj.integrate(s, tj.Particle("p", d["sample_position"][k], d["sample_velocity"][k]))
            self.assertEqual(electrodes[int(d["sample_electrode"][0, 0, k])], tr.electrode, msg=f"sample {k}")
            if tr.escaped:
                n_escaped += 1
                self.assertAlmostEqual(d["sample_t_end"][0, 0, k] / tr.escape_time, 1.0, delta=1e-6)
        self.assertGreater(n_escaped, 3)   # the case must exercise both outcomes
        self.assertLess(n_escaped, 37)

    def test_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            # No RF: every particle eventually hits an electrode (falls at ~3 mm/s).
            none = run_scan(Path(tmp) / "none", "--vac-range", 0, 0, "--samples", 30,
                            "--rf-periods", 400, "--pos-radius", 1e-3, "--vel-sigma", 0.1)
            # Floquet-unstable point: nothing survives.
            unstable = run_scan(Path(tmp) / "unst", "--vac-range", 6000, 6000, "--samples", 30,
                                "--rf-periods", 50, "--pos-radius", 1e-3, "--vel-sigma", 0.1)
            # Deep inside the stable region with a small spread: everything is captured.
            stable = run_scan(Path(tmp) / "st", "--vac-range", 1000, 1000, "--samples", 30,
                              "--rf-periods", 100, "--pos-radius", 0.2e-3, "--vel-sigma", 0.02)
        self.assertEqual(int(none["n_trapped"][0, 0]), 0)
        self.assertEqual(int(unstable["floquet_class"][0, 0]), 2)
        self.assertEqual(int(unstable["n_trapped"][0, 0]), 0)
        self.assertEqual(int(stable["floquet_class"][0, 0]), 0)
        self.assertEqual(int(stable["n_trapped"][0, 0]), 30)


if __name__ == "__main__":
    unittest.main()
