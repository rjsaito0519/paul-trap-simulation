"""Trajectory file round trip and animation rendering (Phase 6a)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import animate_trap as at  # noqa: E402
import trajectory as tj  # noqa: E402

CONFIG = REPO / "configs" / "example_microparticle.toml"


class TestTrajectoryFile(unittest.TestCase):
    def test_round_trip(self):
        setup, trajs = tj.run_config(CONFIG, samples=50, duration=0.01)
        escape = tj.integrate(setup, tj.Particle("fast", np.zeros(3), np.array([0.0, 0.0, 15.0])),
                              np.linspace(0, 0.01, 50))
        trajs.append(escape)
        setup.particles.append(tj.Particle("fast", np.zeros(3), np.array([0.0, 0.0, 15.0])))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.npz"
            t_eval = np.linspace(0, 0.01, 50)
            tj.save_run(path, setup, trajs, CONFIG.read_text(), t_eval=t_eval)
            s2, tr2, meta = tj.load_run(path)
        self.assertEqual(meta["format"], tj.RUN_FORMAT)
        np.testing.assert_array_equal(meta["t_eval"], t_eval)
        for f in ("a_z", "q_z", "b", "gravity_term", "omega", "r0", "rf_phase", "duration"):
            self.assertEqual(getattr(s2, f), getattr(setup, f), msg=f)
        self.assertEqual([p.label for p in s2.particles], ["particle", "fast"])
        for a, b in zip(trajs, tr2):
            np.testing.assert_array_equal(a.t, b.t)
            np.testing.assert_array_equal(a.position, b.position)
            np.testing.assert_array_equal(a.velocity, b.velocity)
            self.assertEqual((a.escaped, a.electrode), (b.escaped, b.electrode))
        self.assertTrue(tr2[1].escaped)
        self.assertEqual(tr2[1].escape_time, escape.escape_time)
        self.assertTrue(np.isnan(tr2[0].escape_time))


class TestAnimation(unittest.TestCase):
    def test_render_both_views_from_config_and_file(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            g3 = Path(tmp) / "a.gif"
            at.main([str(CONFIG), "--rf-cycles", "1", "--frames-per-rf", "4", "--dpi", "40", "--output", str(g3)])
            saved = Path(tmp) / "a.npz"
            self.assertTrue(saved.exists())
            g2 = Path(tmp) / "b.gif"
            at.main([str(saved), "--view", "2d", "--dpi", "40", "--output", str(g2)])
            for g in (g3, g2):
                with Image.open(g) as im:
                    self.assertEqual(im.n_frames, 5)


if __name__ == "__main__":
    unittest.main()
