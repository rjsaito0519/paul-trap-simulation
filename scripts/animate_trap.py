#!/usr/bin/env python3
"""Animate particles in an ideal ring-endcap Paul trap (3D GIF / MP4).

Left: 3D view with cut-away hyperbolic electrodes and particle trails.
Right: z(t) and r(t) = sqrt(x^2 + y^2) with a moving time cursor.
Trajectories come from scripts/trajectory.py (Python reference).
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trajectory as tj  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf"]


def frame_times(setup, rf_cycles, frames_per_rf, strobe):
    period = 1.0 / setup.frequency
    if strobe:
        # One frame per RF period at the same RF phase: micromotion is hidden.
        return np.arange(0, int(rf_cycles) + 1) * period
    n = int(round(rf_cycles * frames_per_rf)) + 1
    return np.linspace(0.0, rf_cycles * period, n)


def electrode_surfaces(r0, cut=(0.25 * math.pi, 1.75 * math.pi), n=40):
    """Mesh grids (in mm) for the ring and the two endcaps, with a cut-away wedge."""
    z0 = r0 / math.sqrt(2.0)
    phi = np.linspace(cut[0], cut[1], n)
    zr = np.linspace(-1.1 * z0, 1.1 * z0, n)
    P, Z = np.meshgrid(phi, zr)
    R = np.sqrt(r0 ** 2 + 2.0 * Z ** 2)                  # ring: r^2 - 2 z^2 = r0^2
    ring = (R * np.cos(P) * 1e3, R * np.sin(P) * 1e3, Z * 1e3)
    rr = np.linspace(0.0, 1.1 * r0, n)
    P2, R2 = np.meshgrid(phi, rr)
    Zc = np.sqrt((r0 ** 2 + R2 ** 2) / 2.0)              # endcaps: 2 z^2 - r^2 = r0^2
    caps = [(R2 * np.cos(P2) * 1e3, R2 * np.sin(P2) * 1e3, s * Zc * 1e3) for s in (1.0, -1.0)]
    return ring, caps


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("config", type=Path)
    p.add_argument("--output", type=Path, default=None,
                   help=".gif or .mp4 (default results/animation/<config>.gif)")
    p.add_argument("--rf-cycles", type=float, default=None,
                   help="animated duration in RF periods (default: config duration)")
    p.add_argument("--frames-per-rf", type=float, default=12.0)
    p.add_argument("--strobe", action="store_true", help="one frame per RF period")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--trail", type=float, default=2.0, help="trail length in RF periods")
    p.add_argument("--extent", type=float, default=None,
                   help="half width of the 3D view in mm (default: electrode size)")
    p.add_argument("--rotate", type=float, default=0.0,
                   help="camera azimuth step per frame [deg] (default 0: fixed view)")
    p.add_argument("--elev", type=float, default=18.0)
    p.add_argument("--azim", type=float, default=-60.0)
    p.add_argument("--dpi", type=int, default=90)
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 9})

    setup = tj.load_setup(args.config)
    period = 1.0 / setup.frequency
    rf_cycles = args.rf_cycles if args.rf_cycles is not None else setup.duration / period
    setup.duration = rf_cycles * period
    t = frame_times(setup, rf_cycles, args.frames_per_rf, args.strobe)
    trajs = [tj.integrate(setup, part, t) for part in setup.particles]
    print(tj.describe(setup))
    for tr in trajs:
        print(f"{tr.label:>10s}: " + (f"escaped at {tr.escape_time * 1e3:.3f} ms ({tr.electrode})"
                                     if tr.escaped else "trapped"))

    fig = plt.figure(figsize=(11, 6))
    ax3 = fig.add_axes([0.0, 0.02, 0.58, 0.9], projection="3d")
    axz = fig.add_axes([0.66, 0.55, 0.32, 0.33])
    axr = fig.add_axes([0.66, 0.1, 0.32, 0.33], sharex=axz)

    # Cut a 90-degree wedge out of the electrodes, centred on the camera direction.
    view = math.radians(args.azim)
    ring, caps = electrode_surfaces(setup.r0, cut=(view + 0.25 * math.pi, view + 1.75 * math.pi))
    ax3.plot_surface(*ring, color="#b08d57", alpha=0.18, linewidth=0, shade=True)
    for cap in caps:
        ax3.plot_surface(*cap, color="#7f8c99", alpha=0.18, linewidth=0, shade=True)
    ext = args.extent if args.extent else 1.15 * setup.r0 * 1e3
    ax3.set_xlim(-ext, ext)
    ax3.set_ylim(-ext, ext)
    ax3.set_zlim(-ext, ext)
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_xlabel("x [mm]")
    ax3.set_ylabel("y [mm]")
    ax3.set_zlabel("z [mm]（上が鉛直上向き）")

    t_ms = t * 1e3
    trail_n = max(1, int(round(args.trail * (1 if args.strobe else args.frames_per_rf))))
    dots, trails, zlines, rlines = [], [], [], []
    for k, tr in enumerate(trajs):
        c = COLORS[k % len(COLORS)]
        dots.append(ax3.plot([], [], [], "o", color=c, ms=5, label=tr.label)[0])
        trails.append(ax3.plot([], [], [], "-", color=c, lw=1.0, alpha=0.7)[0])
        zlines.append(axz.plot(tr.t * 1e3, tr.position[:, 2] * 1e3, color=c, lw=0.9)[0])
        rlines.append(axr.plot(tr.t * 1e3, np.hypot(tr.position[:, 0], tr.position[:, 1]) * 1e3,
                               color=c, lw=0.9)[0])
        if tr.escaped:
            axz.axvline(tr.escape_time * 1e3, color=c, ls=":", lw=0.8)
    ax3.legend(loc="upper left", fontsize=8)
    axz.axhline(setup.z0 * 1e3, color="0.6", lw=0.6, ls="--")
    axz.axhline(-setup.z0 * 1e3, color="0.6", lw=0.6, ls="--")
    axr.axhline(setup.r0 * 1e3, color="0.6", lw=0.6, ls="--")
    axz.set_ylabel("z [mm]")
    axr.set_ylabel(r"$\sqrt{x^2+y^2}$ [mm]")
    axr.set_xlabel("t [ms]")
    axz.set_title("破線: 電極（z0, r0）", fontsize=8, loc="right")
    cur_z = axz.axvline(0, color="k", lw=0.8)
    cur_r = axr.axvline(0, color="k", lw=0.8)
    title = fig.text(0.02, 0.95, "", fontsize=11, fontweight="bold")
    fig.text(0.02, 0.915,
             f"q_z = {setup.q_z:.3g}, a_z = {setup.a_z:.3g}, b = {setup.b:.3g}, f = {setup.frequency:g} Hz, "
             f"r0 = {setup.r0 * 1e3:g} mm" + ("（ストロボ: 各RF周期の同位相）" if args.strobe else ""),
             fontsize=9, color="0.3")

    def update(i):
        for k, tr in enumerate(trajs):
            j = min(i, len(tr.t) - 1)
            pos = tr.position * 1e3
            dots[k].set_data_3d([pos[j, 0]], [pos[j, 1]], [pos[j, 2]])
            dots[k].set_marker("x" if tr.escaped and i >= len(tr.t) - 1 else "o")
            s = max(0, j - trail_n)
            trails[k].set_data_3d(pos[s:j + 1, 0], pos[s:j + 1, 1], pos[s:j + 1, 2])
        cur_z.set_xdata([t_ms[i]])
        cur_r.set_xdata([t_ms[i]])
        title.set_text(f"t = {t_ms[i]:7.3f} ms   （RF {t[i] / period:6.1f} 周期）")
        ax3.view_init(elev=args.elev, azim=args.azim + args.rotate * i)
        return dots + trails

    out = args.output or REPO / "results" / "animation" / f"{args.config.stem}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    anim = animation.FuncAnimation(fig, update, frames=len(t), blit=False)
    if out.suffix == ".mp4":
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=2400)
    else:
        writer = animation.PillowWriter(fps=args.fps)
    anim.save(out, writer=writer, dpi=args.dpi)
    print(f"wrote {out} ({len(t)} frames)")


if __name__ == "__main__":
    main()
