"""End-to-end test of scripts/capture_map.py on a tiny grid (Phase 5c)."""

import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import capture_map as cm  # noqa: E402
import trajectory as tj  # noqa: E402

BUILT = (REPO / ".build" / "bin" / "capture_scan").exists() and (REPO / ".build" / "bin" / "floquet_boundary").exists()


class TestCoefficients(unittest.TestCase):
    def test_linear_in_voltage(self):
        s = tj.load_setup(REPO / "configs" / "example_microparticle.toml")
        co = cm.coefficients(s)
        for v_dc, v_ac in ((-300.0, 2000.0), (150.0, 4500.0)):
            t = tj.Setup(**{**s.__dict__, "v_dc": v_dc, "v_ac": v_ac, "particles": []})
            self.assertAlmostEqual(co["a_per_vdc"] * v_dc, t.a_z, delta=1e-12 * max(1.0, abs(t.a_z)))
            self.assertAlmostEqual(co["q_per_vac"] * v_ac, t.q_z, delta=1e-12 * max(1.0, abs(t.q_z)))
        self.assertEqual(co["b"], s.b)


@unittest.skipUnless(BUILT, "C++ programs not built")
class TestEndToEnd(unittest.TestCase):
    def test_small_map(self):
        text = (REPO / "configs" / "example_microparticle.toml").read_text()
        text = re.sub(r"n_v_dc = \d+", "n_v_dc = 5", text)
        text = re.sub(r"n_v_ac = \d+", "n_v_ac = 7", text)
        text = re.sub(r"samples = \d+", "samples = 20", text)
        text = re.sub(r"rf_periods = \d+", "rf_periods = 20", text)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "small.toml"
            cfg.write_text(text)
            out = Path(tmp) / "out"
            cm.main([str(cfg), "--threads", "2", "--output", str(out)])
            self.assertTrue((out / "capture_map.png").exists())
            p = np.load(out / "scan" / "p_capture.npy")
            self.assertEqual(p.shape, (5, 7))
            self.assertTrue(np.all((p >= 0) & (p <= 1)))


if __name__ == "__main__":
    unittest.main()
