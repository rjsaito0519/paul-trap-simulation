#!/usr/bin/env python3
"""Animate particles in an ideal ring-endcap Paul trap (GIF; .mp4 also possible).

Input is either a TOML configuration (trajectories are computed with
scripts/trajectory.py and saved next to the animation as .npz) or a saved
trajectory file (.npz, from trajectory.py --output or a previous run), so
the drawing is separated from the computation.

--view 3d (default): fixed-camera 3D view with cut-away hyperbolic electrodes.
--view 2d: the r-z half plane, r = sqrt(x^2 + y^2), where the electrodes are
exact hyperbolas, so contact in the picture is contact in the simulation.
Both views show z(t) and r(t) with a moving time cursor.
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


def load_or_compute(path, args):
    """Return (setup, trajectories, frame times, config text or None)."""
    if path.suffix == ".npz":
        setup, trajs, meta = tj.load_run(path)
        t = meta["t_eval"] if meta["t_eval"] is not None else max(trajs, key=lambda tr: len(tr.t)).t
        return setup, trajs, np.asarray(t), None
    setup = tj.load_setup(path)
    period = 1.0 / setup.frequency
    rf_cycles = args.rf_cycles if args.rf_cycles is not None else setup.duration / period
    setup.duration = rf_cycles * period
    t = frame_times(setup, rf_cycles, args.frames_per_rf, args.strobe)
    trajs = [tj.integrate(setup, part, t) for part in setup.particles]
    return setup, trajs, t, path.read_text()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input", type=Path, help="TOML configuration or saved trajectory .npz")
    p.add_argument("--output", type=Path, default=None,
                   help=".gif (default) or .mp4; default results/animation/<input stem>[_2d].gif")
    p.add_argument("--view", choices=("3d", "2d"), default="3d")
    p.add_argument("--rf-cycles", type=float, default=None,
                   help="TOML input: animated duration in RF periods (default: config duration)")
    p.add_argument("--frames-per-rf", type=float, default=12.0, help="TOML input: frames per RF period")
    p.add_argument("--strobe", action="store_true", help="TOML input: one frame per RF period")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--trail", type=float, default=2.0, help="trail length in RF periods")
    p.add_argument("--extent", type=float, default=None,
                   help="half width of the trap view in mm (default: electrode size)")
    p.add_argument("--rotate", type=float, default=0.0,
                   help="3D: camera azimuth step per frame [deg] (default 0: fixed view)")
    p.add_argument("--elev", type=float, default=18.0)
    p.add_argument("--azim", type=float, default=-60.0)
    p.add_argument("--dpi", type=int, default=90)
    p.add_argument("--no-timeseries", action="store_true", help="draw only the trap view")
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 9})

    setup, trajs, t, config_text = load_or_compute(args.input, args)
    period = 1.0 / setup.frequency
    stem = args.input.stem + ("_2d" if args.view == "2d" else "")
    out = args.output or REPO / "results" / "animation" / f"{stem}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    if config_text is not None:
        tj.save_run(out.parent / (out.stem + ".npz"), setup, trajs, config_text, t_eval=t)
    print(tj.describe(setup))
    for tr in trajs:
        print(f"{tr.label:>10s}: " + (f"escaped at {tr.escape_time * 1e3:.3f} ms ({tr.electrode})"
                                     if tr.escaped else "trapped"))

    ts = not args.no_timeseries
    fig = plt.figure(figsize=(11, 6) if ts else (6.5, 6))
    left = [0.0, 0.02, 0.58, 0.9] if ts else [0.0, 0.02, 1.0, 0.86]
    ext = args.extent if args.extent else 1.15 * setup.r0 * 1e3
    if args.view == "3d":
        ax = fig.add_axes(left, projection="3d")
        # Cut a 90-degree wedge out of the electrodes, centred on the camera direction.
        view = math.radians(args.azim)
        ring, caps = electrode_surfaces(setup.r0, cut=(view + 0.25 * math.pi, view + 1.75 * math.pi))
        ax.plot_surface(*ring, color="#b08d57", alpha=0.18, linewidth=0, shade=True)
        for cap in caps:
            ax.plot_surface(*cap, color="#7f8c99", alpha=0.18, linewidth=0, shade=True)
        ax.set_xlim(-ext, ext)
        ax.set_ylim(-ext, ext)
        ax.set_zlim(-ext, ext)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]（上が鉛直上向き）")
    else:
        ax = fig.add_axes([left[0] + 0.06, left[1] + 0.06, left[2] - 0.1, left[3] - 0.1])
        r0_mm = setup.r0 * 1e3
        zz = np.linspace(-ext, ext, 400)
        rr = np.linspace(0.0, ext, 400)
        ax.fill_betweenx(zz, np.sqrt(r0_mm ** 2 + 2 * zz ** 2), ext * 1.2, color="#b08d57", alpha=0.35,
                         lw=0, label="リング電極")
        zc = np.sqrt((r0_mm ** 2 + rr ** 2) / 2)
        ax.fill_between(rr, zc, ext * 1.2, color="#7f8c99", alpha=0.35, lw=0, label="エンドキャップ")
        ax.fill_between(rr, -ext * 1.2, -zc, color="#7f8c99", alpha=0.35, lw=0)
        ax.set_xlim(0, ext)
        ax.set_ylim(-ext, ext)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$r=\sqrt{x^2+y^2}$ [mm]")
        ax.set_ylabel("z [mm]（上が鉛直上向き）")
        ax.legend(loc="lower right", fontsize=8)

    if ts:
        axz = fig.add_axes([0.66, 0.55, 0.32, 0.33])
        axr = fig.add_axes([0.66, 0.1, 0.32, 0.33], sharex=axz)
    t_ms = t * 1e3
    dt = float(np.median(np.diff(t))) if len(t) > 1 else period
    trail_n = max(1, int(round(args.trail * period / dt)))
    dots, trails = [], []
    for k, tr in enumerate(trajs):
        c = COLORS[k % len(COLORS)]
        if args.view == "3d":
            dots.append(ax.plot([], [], [], "o", color=c, ms=5, label=tr.label)[0])
            trails.append(ax.plot([], [], [], "-", color=c, lw=1.0, alpha=0.7)[0])
        else:
            dots.append(ax.plot([], [], "o", color=c, ms=5)[0])
            trails.append(ax.plot([], [], "-", color=c, lw=1.0, alpha=0.7)[0])
        if ts:
            axz.plot(tr.t * 1e3, tr.position[:, 2] * 1e3, color=c, lw=0.9)
            axr.plot(tr.t * 1e3, np.hypot(tr.position[:, 0], tr.position[:, 1]) * 1e3, color=c, lw=0.9)
            if tr.escaped:
                axz.axvline(tr.escape_time * 1e3, color=c, ls=":", lw=0.8)
    if args.view == "3d" and len(trajs) > 1:
        ax.legend(loc="upper left", fontsize=8)
    if ts:
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
            s = max(0, j - trail_n)
            if args.view == "3d":
                dots[k].set_data_3d([pos[j, 0]], [pos[j, 1]], [pos[j, 2]])
                trails[k].set_data_3d(pos[s:j + 1, 0], pos[s:j + 1, 1], pos[s:j + 1, 2])
            else:
                r = np.hypot(pos[:, 0], pos[:, 1])
                dots[k].set_data([r[j]], [pos[j, 2]])
                trails[k].set_data(r[s:j + 1], pos[s:j + 1, 2])
            dots[k].set_marker("x" if tr.escaped and i >= len(tr.t) - 1 else "o")
        if ts:
            cur_z.set_xdata([t_ms[i]])
            cur_r.set_xdata([t_ms[i]])
        title.set_text(f"t = {t_ms[i]:7.3f} ms   （RF {t[i] / period:6.1f} 周期）")
        if args.view == "3d":
            ax.view_init(elev=args.elev, azim=args.azim + args.rotate * i)
        return dots + trails

    anim = animation.FuncAnimation(fig, update, frames=len(t), blit=False)
    writer = (animation.FFMpegWriter(fps=args.fps, bitrate=2400) if out.suffix == ".mp4"
              else animation.PillowWriter(fps=args.fps))
    anim.save(out, writer=writer, dpi=args.dpi)
    print(f"wrote {out} ({len(t)} frames)")


if __name__ == "__main__":
    main()
