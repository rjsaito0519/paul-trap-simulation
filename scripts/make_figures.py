#!/usr/bin/env python3
"""Regenerate the standard figures, animations and README images (Phase 6d).

Results go to results/ (not tracked by Git). With --readme, small copies for
the GitHub README are written to docs/images/ (tracked). Build the C++
programs first (./build.sh). On shared login nodes, limit the Python numeric
libraries, e.g. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import animate_trap  # noqa: E402
import capture_map  # noqa: E402
import explainer_figures  # noqa: E402
import stability_overview  # noqa: E402
import trajectory as tj  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "configs" / "example_microparticle.toml"
RESULTS = REPO / "results"
IMAGES = REPO / "docs" / "images"

# Escape example: the example configuration with only V_AC raised just past the
# axial Floquet boundary (q_z = 15.0, max|lambda| = 1.17) and a start 50 um from
# the centre, so the growing oscillation is visible before the endcap is hit.
ESCAPE_V_AC = 5100.0
ESCAPE_START = tj.Particle("particle", np.array([50e-6, 25e-6, 50e-6]), np.zeros(3))
ESCAPE_RF_PERIODS = 15


def escape_run(path, frames_per_rf):
    setup = tj.load_setup(CONFIG)
    setup.v_ac = ESCAPE_V_AC
    setup.particles = [ESCAPE_START]
    setup.duration = ESCAPE_RF_PERIODS / setup.frequency
    t = animate_trap.frame_times(setup, ESCAPE_RF_PERIODS, frames_per_rf, strobe=False)
    trajs = [tj.integrate(setup, p, t) for p in setup.particles]
    tj.save_run(path, setup, trajs, CONFIG.read_text(), t_eval=t,
                command=sys.argv + [f"[escape example: v_ac={ESCAPE_V_AC}]"])
    return path


def downscale_png(src, dst, width):
    from PIL import Image
    with Image.open(src) as im:
        h = round(im.height * width / im.width)
        im.convert("RGB").resize((width, h), Image.LANCZOS).save(dst, optimize=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--readme", action="store_true", help="also write README images to docs/images/")
    p.add_argument("--reuse-capture", action="store_true", help="replot the existing capture scan")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["explainer", "overview", "capture", "animation"])
    args = p.parse_args(argv)
    th = str(args.threads)
    anim = RESULTS / "animation"
    anim.mkdir(parents=True, exist_ok=True)

    if "explainer" not in args.skip:
        explainer_figures.main(["--threads", th])
    if "overview" not in args.skip:
        stability_overview.main(["--threads", th])
    if "capture" not in args.skip:
        capture_map.main([str(CONFIG), "--threads", th] + (["--plot-only"] if args.reuse_capture else []))
    if "animation" not in args.skip:
        trapped = anim / "example_microparticle.gif"
        animate_trap.main([str(CONFIG), "--output", str(trapped)])
        animate_trap.main([str(anim / "example_microparticle.npz"), "--view", "2d",
                           "--output", str(anim / "example_microparticle_2d.gif")])
        animate_trap.main([str(escape_run(anim / "example_escape.npz", 12)), "--output",
                           str(anim / "example_escape.gif"), "--fps", "15"])

    if args.readme:
        IMAGES.mkdir(parents=True, exist_ok=True)
        # Small GIFs: fewer frames per RF period and low resolution keep the repository light.
        animate_trap.main([str(CONFIG), "--output", str(IMAGES / "trap_3d.gif"), "--frames-per-rf", "6",
                           "--dpi", "55", "--fps", "15"])
        (IMAGES / "trap_3d.npz").unlink(missing_ok=True)
        esc = escape_run(RESULTS / "animation" / "readme_escape.npz", 8)
        animate_trap.main([str(esc), "--output", str(IMAGES / "escape_3d.gif"), "--dpi", "55", "--fps", "12"])
        downscale_png(RESULTS / "stability_overview" / "stability_overview.png",
                      IMAGES / "stability_overview.png", 1100)
        downscale_png(RESULTS / "capture" / CONFIG.stem / "capture_map.png", IMAGES / "capture_map.png", 1100)
        for f in sorted(IMAGES.iterdir()):
            print(f"{f.relative_to(REPO)}: {f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
