"""Validation tests for scripts/floquet_reference.py (docs/simulation/validation.md).

Run from the repository root:  python3 -m unittest discover -s tests
"""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import floquet_reference as fr  # noqa: E402

# First stability region upper edge for b = 0, a = 0 (Mathieu b1(q) = 0).
Q_EDGE_A0 = 0.908046


def _bisect(f, lo, hi, tol=1e-10):
    flo = f(lo)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


class TestMonodromy(unittest.TestCase):
    def test_constant_coefficient_matches_matrix_exponential(self):
        # q = 0: x'' + b x' + a x = 0 has M = expm(A pi) exactly.
        for a, b in [(0.3, 0.0), (0.3, 0.5), (-0.2, 0.1), (2.0, 1.5)]:
            exact = expm(np.array([[0.0, 1.0], [-a, -b]]) * np.pi)
            np.testing.assert_allclose(fr.monodromy(a, 0.0, b), exact,
                                       rtol=1e-10, atol=1e-11)

    def test_liouville_determinant(self):
        for a, q, b in [(0.0, 0.5, 0.0), (0.1, 0.7, 0.3), (-0.3, 1.5, 2.0), (0.0, 5.0, 6.0)]:
            det = np.linalg.det(fr.monodromy(a, q, b))
            self.assertAlmostEqual(det / np.exp(-b * np.pi), 1.0, delta=1e-9)

    def test_damping_scales_multipliers(self):
        # x = exp(-b tau/2) y maps the damped equation to undamped Mathieu
        # with a' = a - b^2/4, so lambda = exp(-b pi/2) mu.
        for a, q, b in [(0.1, 0.6, 0.4), (0.0, 1.2, 1.0), (-0.1, 0.3, 0.2)]:
            lam = np.sort_complex(fr.floquet_multipliers(fr.monodromy(a, q, b)))
            mu = np.sort_complex(fr.floquet_multipliers(fr.monodromy(a - b * b / 4.0, q, 0.0)))
            np.testing.assert_allclose(lam, np.exp(-b * np.pi / 2.0) * mu, atol=1e-9)

    def test_tolerance_convergence(self):
        m_ref = fr.monodromy(0.05, 0.8, 0.2, rtol=1e-13, atol=1e-15)
        m = fr.monodromy(0.05, 0.8, 0.2, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(m, m_ref, atol=1e-8)


class TestKnownBoundaries(unittest.TestCase):
    def test_first_region_upper_edge_a0_b0(self):
        # Edge of the first region at a = 0 has a period-2pi solution: tr M = -2.
        q = _bisect(lambda q: np.trace(fr.monodromy(0.0, q, 0.0)) + 2.0, 0.8, 1.0)
        self.assertAlmostEqual(q, Q_EDGE_A0, delta=2e-6)

    def test_classification_near_edge(self):
        self.assertEqual(fr.classify(fr.monodromy(0.0, 0.90, 0.0), 0.0), fr.STABLE)
        self.assertEqual(fr.classify(fr.monodromy(0.0, 0.92, 0.0), 0.0), fr.UNSTABLE)

    def test_damped_classification(self):
        self.assertEqual(fr.classify(fr.monodromy(0.0, 0.5, 0.1), 0.1), fr.STABLE)
        # Damping widens the region: q = 0.92 is unstable at b = 0 but stable at b = 0.5.
        self.assertEqual(fr.classify(fr.monodromy(0.0, 0.92, 0.5), 0.5), fr.STABLE)
        self.assertEqual(fr.classify(fr.monodromy(0.0, 1.5, 0.1), 0.1), fr.UNSTABLE)

    def test_negative_b_rejected(self):
        with self.assertRaises(ValueError):
            fr.classify(np.eye(2), -0.1)


class TestTrap(unittest.TestCase):
    def test_radial_conversion(self):
        self.assertEqual(fr.radial_params(-0.2, 0.6), (0.1, -0.3))

    def test_trap_classification(self):
        self.assertEqual(fr.trap_classification(0.0, 0.5, 0.0)[2], fr.STABLE)
        # Axial edge at q_z = 0.908 limits the trap at a_z = 0.
        ax, ra, comb = fr.trap_classification(0.0, 0.95, 0.0)
        self.assertEqual((ax, ra, comb), (fr.UNSTABLE, fr.STABLE, fr.UNSTABLE))
        # Positive a_z makes the radial direction unstable at small q.
        ax, ra, comb = fr.trap_classification(0.2, 0.1, 0.0)
        self.assertEqual(ra, fr.UNSTABLE)
        self.assertEqual(comb, fr.UNSTABLE)


if __name__ == "__main__":
    unittest.main()
