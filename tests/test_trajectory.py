"""Tests for scripts/trajectory.py (Phase 4, docs/simulation/validation.md)."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import floquet_reference as fr  # noqa: E402
import trajectory as tj  # noqa: E402


def make_setup(**kw):
    base = dict(radius=5e-6, density=1050.0, charge=1.6e-14, viscosity=1.81e-5, r0=5e-3,
                v_dc=0.0, v_ac=1000.0, frequency=200.0, rf_phase=0.0, gravity=9.80665,
                duration=0.02)
    base.update(kw)
    return tj.Setup(**base)


class TestConversions(unittest.TestCase):
    def test_example_config_loads(self):
        s = tj.load_setup(REPO / "configs" / "example_microparticle.toml")
        self.assertEqual(len(s.particles), 3)
        self.assertAlmostEqual(s.b, 9 * s.viscosity / (s.radius ** 2 * s.density * s.omega), delta=1e-12)

    def test_time_and_velocity_round_trip(self):
        s = make_setup(rf_phase=0.7)
        t = np.array([0.0, 1.3e-3, 0.02])
        np.testing.assert_allclose(s.time(s.tau(t)), t, rtol=0, atol=1e-15)
        v = np.array([0.3, -1.2, 5.0])
        np.testing.assert_allclose(s.to_physical_velocity(s.to_tau_velocity(v)), v, rtol=1e-15)
        self.assertAlmostEqual(float(s.to_tau_velocity(1.0)), 2.0 / s.omega)

    def test_march_parameters(self):
        # a_z and q_z from model.md; radial parameters by the quadrupole relation.
        s = make_setup(v_dc=10.0)
        den = s.mass * s.omega ** 2 * s.r0 ** 2
        self.assertAlmostEqual(s.a_z, -8 * s.charge * 10.0 / den)
        self.assertAlmostEqual(s.q_z, 4 * s.charge * 1000.0 / den)


class TestLimits(unittest.TestCase):
    def test_zero_voltage_is_free_fall_with_stokes_drag(self):
        s = make_setup(v_ac=0.0, duration=2e-3)
        v0 = np.array([0.1, -0.05, 0.4])
        p0 = np.array([1e-4, 2e-4, -3e-4])
        t = np.linspace(0, s.duration, 50)
        tr = tj.integrate(s, tj.Particle("p", p0, v0), t)
        tau_d = s.mass / s.drag
        v_term = np.array([0.0, 0.0, -s.gravity * tau_d])
        e = np.exp(-t / tau_d)[:, None]
        exact = p0 + v_term * t[:, None] + (v0 - v_term) * tau_d * (1 - e)
        np.testing.assert_allclose(tr.position, exact, rtol=0, atol=1e-11)
        np.testing.assert_allclose(tr.velocity, v_term + (v0 - v_term) * e, rtol=0, atol=1e-8)

    def test_point_symmetry_without_gravity(self):
        s = make_setup(gravity=0.0)
        t = np.linspace(0, s.duration, 200)
        p, v = np.array([1e-3, -0.5e-3, 0.8e-3]), np.array([0.1, 0.2, -0.1])
        a = tj.integrate(s, tj.Particle("a", p, v), t)
        b = tj.integrate(s, tj.Particle("b", -p, -v), t)
        np.testing.assert_allclose(a.position, -b.position, rtol=0, atol=1e-15)

    def test_growth_rate_matches_floquet_multiplier(self):
        # Without gravity and electrodes, the stroboscopic growth rate per RF period of
        # |[x, dx/dtau]| equals log(max|lambda|) of the corresponding direction.
        for v_ac in (1000.0, 3000.0, 6000.0):
            s = make_setup(gravity=0.0, v_ac=v_ac, duration=0.2)
            t = np.arange(41) / s.frequency
            for direction, p0 in (("radial", [1e-4, 0, 0]), ("axial", [0, 0, 1e-4])):
                tr = tj.integrate(s, tj.Particle("p", np.array(p0), np.zeros(3)), t,
                                  detect_escape=False)
                state = np.concatenate([tr.position, s.to_tau_velocity(tr.velocity)], axis=1)
                norm = np.linalg.norm(state, axis=1)
                rate = (np.log(norm[40]) - np.log(norm[20])) / 20.0
                a, q = (s.a_z, s.q_z) if direction == "axial" else fr.radial_params(s.a_z, s.q_z)
                expected = np.log(np.abs(fr.floquet_multipliers(fr.monodromy(a, q, s.b))).max())
                self.assertAlmostEqual(rate, expected, delta=1e-3 * abs(expected),
                                       msg=f"V_AC={v_ac} {direction}")


class TestEscape(unittest.TestCase):
    def test_fast_particle_hits_endcap_on_surface(self):
        s = make_setup()
        tr = tj.integrate(s, tj.Particle("p", np.zeros(3), np.array([0.0, 0.0, 15.0])),
                          np.linspace(0, s.duration, 500))
        self.assertTrue(tr.escaped)
        self.assertEqual(tr.electrode, "endcap")
        x, y, z = tr.position[-1]
        self.assertAlmostEqual((2 * z * z - x * x - y * y) / s.r0 ** 2, 1.0, delta=1e-9)
        self.assertAlmostEqual(tr.t[-1], tr.escape_time, delta=1e-15)

    def test_radial_escape_hits_ring(self):
        s = make_setup(v_ac=0.0, gravity=0.0)
        tr = tj.integrate(s, tj.Particle("p", np.zeros(3), np.array([30.0, 0.0, 0.0])),
                          np.linspace(0, s.duration, 500))
        self.assertTrue(tr.escaped)
        self.assertEqual(tr.electrode, "ring")
        self.assertAlmostEqual(np.hypot(*tr.position[-1, :2]), s.r0, delta=1e-12)

    def test_slow_particle_is_trapped_and_sags(self):
        s = make_setup(duration=0.1)
        tr = tj.integrate(s, tj.Particle("p", np.array([1e-4, 0, 1e-4]), np.zeros(3)),
                          np.linspace(0, s.duration, 200))
        self.assertFalse(tr.escaped)
        self.assertTrue(math.isnan(tr.escape_time))
        self.assertLess(np.mean(tr.position[-20:, 2]), 0.0)  # gravity pulls the mean below 0


if __name__ == "__main__":
    unittest.main()
