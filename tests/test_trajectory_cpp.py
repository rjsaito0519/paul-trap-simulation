"""Cross-check of the C++ trajectory engine against scripts/trajectory.py (Phase 5a).

Requires ./build.sh first; skipped when the executable is missing.
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
import trajectory as tj  # noqa: E402

RUN = REPO / ".build" / "bin" / "trajectory_run"
# Position tolerance relative to r0 (0.5 nm for r0 = 5 mm): far below any physical
# scale of the capture decision, above the observed C++/Python difference (~1e-8 r0).
POS_TOL = 1e-7


def make_setup(**kw):
    base = dict(radius=5e-6, density=1050.0, charge=1.6e-14, viscosity=1.81e-5, r0=5e-3,
                v_dc=0.0, v_ac=1000.0, frequency=200.0, rf_phase=0.0, gravity=9.80665,
                duration=0.02)
    base.update(kw)
    return tj.Setup(**base)


def run_cpp(setup, particle, steps=200, record_every=1):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        args = [str(RUN), "--output", str(out), "--a-z", setup.a_z, "--q-z", setup.q_z,
                "--b", setup.b, "--gravity-term", setup.gravity_term, "--r0", setup.r0,
                "--omega", setup.omega, "--rf-phase", setup.rf_phase,
                "--duration", setup.duration, "--steps-per-rf", steps,
                "--record-every", record_every,
                "--position", *particle.position, "--velocity", *particle.velocity]
        # repr(float) keeps full double precision on the command line.
        subprocess.run([repr(float(a)) if isinstance(a, (float, np.floating)) else str(a) for a in args],
                       check=True, capture_output=True)
        return (np.load(out / "t.npy"), np.load(out / "position.npy"), np.load(out / "velocity.npy"),
                json.loads((out / "result.json").read_text()))


@unittest.skipUnless(RUN.exists(), "trajectory_run not built")
class TestCppTrajectory(unittest.TestCase):
    CASES = [
        # (setup overrides, position [m], velocity [m/s], expected electrode or None)
        ({}, [2e-3, 1e-3, 1.5e-3], [-0.5, 1.2, 0.5], None),
        ({"rf_phase": 1.1}, [1e-3, -1e-3, 0.5e-3], [0.3, 0.0, -0.4], None),
        ({}, [0.0, 0.0, 0.0], [0.0, 0.0, 15.0], "endcap"),
        ({"v_ac": 0.0, "gravity": 0.0}, [0.0, 0.0, 0.0], [30.0, 0.0, 0.0], "ring"),
        # No RF: falls at the terminal velocity g m / k ~ 3 mm/s onto the lower endcap.
        ({"v_ac": 0.0, "duration": 1.5}, [1e-3, 0.0, 0.0], [0.0, 0.0, 0.0], "endcap"),
        ({"v_dc": 30.0, "rf_phase": 2.5}, [1e-3, 0.5e-3, -1e-3], [0.2, -0.3, 0.1], None),
    ]

    def test_matches_python_reference(self):
        # 800 steps per RF period makes the RK4 error negligible, so this checks the
        # implementation; accuracy at the default 200 steps is in test_step_convergence.
        for over, pos, vel, electrode in self.CASES:
            s = make_setup(**over)
            part = tj.Particle("p", np.array(pos), np.array(vel))
            t, x, v, res = run_cpp(s, part, steps=800, record_every=4)
            ref = tj.integrate(s, part, t_eval=t[:-1] if res["escaped"] else t)
            self.assertEqual(res["escaped"], ref.escaped, msg=str(over))
            self.assertEqual(res["electrode"], electrode or "none", msg=str(over))
            if ref.escaped:
                self.assertEqual(res["electrode"], ref.electrode)
                self.assertAlmostEqual(res["t_end"] / ref.escape_time, 1.0, delta=1e-8)
            n = min(len(t), len(ref.t))
            np.testing.assert_allclose(x[:n], ref.position[:n], rtol=0, atol=POS_TOL * s.r0, err_msg=str(over))
            np.testing.assert_allclose(v[:n], ref.velocity[:n], rtol=0,
                                       atol=1e-6 * max(1.0, np.abs(vel).max()), err_msg=str(over))

    def test_escape_point_on_surface(self):
        s = make_setup()
        t, x, v, res = run_cpp(s, tj.Particle("p", np.zeros(3), np.array([0.0, 0.0, 15.0])))
        X, Y, Z = x[-1]
        self.assertAlmostEqual((2 * Z * Z - X * X - Y * Y) / s.r0 ** 2, 1.0, delta=1e-9)
        self.assertAlmostEqual(t[-1], res["t_end"], delta=1e-15)

    def test_escape_time_at_default_steps(self):
        # 200 steps per RF period (default): escape time within 1e-6 relative.
        s = make_setup()
        part = tj.Particle("p", np.zeros(3), np.array([0.0, 0.0, 15.0]))
        res = run_cpp(s, part, steps=200, record_every=10 ** 6)[3]
        ref = tj.integrate(s, part, rtol=1e-13, atol=1e-16)
        self.assertAlmostEqual(res["t_end"] / ref.escape_time, 1.0, delta=1e-6)

    def test_step_convergence(self):
        s = make_setup(duration=0.05)
        part = tj.Particle("p", np.array([2e-3, 1e-3, 1.5e-3]), np.array([-0.5, 1.2, 0.5]))
        finals = {n: run_cpp(s, part, steps=n, record_every=10 ** 6)[1][-1] for n in (50, 100, 200, 400)}
        err = {n: np.abs(finals[n] - finals[400]).max() for n in (50, 100, 200)}
        # Fourth-order convergence: halving the step reduces the error by about 16.
        self.assertGreater(err[50] / err[100], 10.0)
        self.assertLess(err[200], POS_TOL * s.r0)


if __name__ == "__main__":
    unittest.main()
