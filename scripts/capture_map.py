#!/usr/bin/env python3
"""Capture-probability maps in (V_AC, V_DC) from a TOML configuration (Phase 5c).

Reads the [particle], [gas], [trap] and [capture] sections, converts the
physical inputs to dimensionless coefficients with scripts/trajectory.py,
runs the C++ capture_scan, and plots the capture probability with the
Floquet stability boundary (floquet_boundary) mapped back to voltages.
Build the C++ programs first (./build.sh).
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trajectory as tj  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BIN = REPO / ".build" / "bin"


def _fmt(x):
    return repr(float(x)) if isinstance(x, (float, np.floating)) else str(x)


def coefficients(setup):
    """Dimensionless coefficients; a_z and q_z are linear in V_DC and V_AC."""
    unit = tj.Setup(**{**setup.__dict__, "v_dc": 1.0, "v_ac": 1.0, "particles": []})
    return {"a_per_vdc": unit.a_z, "q_per_vac": unit.q_z, "b": setup.b,
            "gravity_term": setup.gravity_term, "r0": setup.r0, "omega": setup.omega}


def load_capture(config_path):
    import tomllib
    with open(config_path, "rb") as f:
        cap = tomllib.load(f)["capture"]
    setup = tj.load_setup(config_path)
    return setup, cap, coefficients(setup)


def scan_command(config_path, out, threads, chunk=None, n_chunks=None):
    """capture_scan command line (list of strings) for a configuration."""
    setup, cap, co = load_capture(config_path)
    args = [BIN / "capture_scan", "--output", out, "--threads", threads,
            "--vdc-range", *cap["v_dc_range"], "--n-vdc", cap["n_v_dc"],
            "--vac-range", *cap["v_ac_range"], "--n-vac", cap["n_v_ac"],
            "--samples", cap["samples"], "--seed", cap["seed"], "--rf-periods", cap["rf_periods"],
            "--steps-per-rf", cap.get("steps_per_rf", 200), "--rf-phase", cap["rf_phase"],
            "--pos-center", *cap["pos_center"], "--pos-radius", cap["pos_radius"],
            "--vel-mean", *cap["vel_mean"], "--vel-sigma", cap["vel_sigma"]]
    for k, v in co.items():
        args += [f"--{k.replace('_', '-')}", v]
    # Optional per-sample particle radius and charge distributions (physical units).
    if "radius" in cap:
        args += ["--radius-ref", setup.radius, "--radius", cap["radius"]]
    if "charge" in cap:
        args += ["--charge-ref", setup.charge, "--charge", cap["charge"]]
    if chunk is not None:
        args += ["--chunk", chunk, "--n-chunks", n_chunks]
    return [_fmt(a) for a in args]


def run_capture(config_path, out, threads):
    subprocess.run(scan_command(config_path, out, threads), check=True)
    return load_capture(config_path)


# ---- batch jobs: the scripts are written here and submitted by the user ----
GRID_ARRAYS = ("n_trapped", "n_ring", "n_endcap", "p_capture", "wilson_lo", "wilson_hi",
               "mean_escape_time", "floquet_class")


def emit_jobs(config_path, out, n_chunks, threads, queue=None):
    """Write one LSF job script per chunk into out/jobs; nothing is submitted."""
    jobs = out / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    scripts = []
    for k in range(n_chunks):
        chunk_dir = out / "chunks" / f"chunk_{k:04d}"
        cmd = " ".join(f"'{a}'" for a in scan_command(config_path, chunk_dir, threads, k, n_chunks))
        lines = ["#!/bin/bash",
                 f"#BSUB -J capture_{config_path.stem}_{k:04d}",
                 f"#BSUB -n {threads}",
                 '#BSUB -R "span[hosts=1]"',
                 f"#BSUB -o {out / 'logs' / f'chunk_{k:04d}.out'}",
                 f"#BSUB -e {out / 'logs' / f'chunk_{k:04d}.err'}"]
        if queue:
            lines.append(f"#BSUB -q {queue}")
        lines += ["set -euo pipefail", f"export OMP_NUM_THREADS={threads}", cmd, ""]
        path = jobs / f"chunk_{k:04d}.sh"
        path.write_text("\n".join(lines))
        path.chmod(0o755)
        scripts.append(path)
    return scripts


def merge_chunks(chunk_dirs, scan):
    """Merge chunk outputs into one scan directory; every grid point must be computed exactly once."""
    metas = [json.loads((d / "metadata.json").read_text()) for d in chunk_dirs]

    def strip(cmd):
        drop = {"--output", "--threads", "--chunk", "--n-chunks"}
        out, skip = [], False
        for a in cmd[1:]:
            if skip:
                skip = False
            elif a in drop:
                skip = True
            else:
                out.append(a)
        return out
    ref = strip(metas[0]["command"])
    for d, m in zip(chunk_dirs, metas):
        if strip(m["command"]) != ref:
            raise ValueError(f"{d}: scan settings differ from {chunk_dirs[0]}")
    first = chunk_dirs[0]
    count = np.zeros(np.load(first / "computed.npy").shape, dtype=int)
    merged = {n: np.load(first / f"{n}.npy").copy() for n in GRID_ARRAYS + ("p_capture_at",)}
    for d in chunk_dirs:
        mask = np.load(d / "computed.npy").astype(bool)
        count += mask
        for n in GRID_ARRAYS:
            merged[n][mask] = np.load(d / f"{n}.npy")[mask]
        merged["p_capture_at"][:, mask] = np.load(d / "p_capture_at.npy")[:, mask]
    if not np.all(count == 1):
        raise ValueError(f"grid points computed {count.min()}..{count.max()} times; expected exactly once")
    scan.mkdir(parents=True, exist_ok=True)
    for n in ("v_dc", "v_ac", "a_z", "q_z", "checkpoint_time"):
        np.save(scan / f"{n}.npy", np.load(first / f"{n}.npy"))
    for n, v in merged.items():
        np.save(scan / f"{n}.npy", v)
    np.save(scan / "computed.npy", count.astype(np.int8))
    meta = dict(metas[0])
    meta.update({"chunk": None, "n_chunks": len(chunk_dirs), "point_range": None,
                 "elapsed_seconds": sum(m["elapsed_seconds"] for m in metas),
                 "merged_from": [str(d) for d in chunk_dirs],
                 "chunk_git": sorted({m["git"]["revision"] for m in metas})})
    (scan / "metadata.json").write_text(json.dumps(meta, indent=2))
    return scan


def floquet_boundary_in_volts(co, cap, out, threads, n=241):
    """Floquet boundary of the scanned window, converted to (V_AC, V_DC) polylines."""
    a_lim = sorted(co["a_per_vdc"] * np.array(cap["v_dc_range"]))
    q_lim = sorted(co["q_per_vac"] * np.array(cap["v_ac_range"]))
    if a_lim[0] == a_lim[1]:
        return []
    subprocess.run([_fmt(a) for a in [BIN / "floquet_boundary", "--output", out, "--threads", threads,
                                      "--a-range", *a_lim, "--q-range", *q_lim, "--na", n, "--nq", n,
                                      "--b-range", co["b"], co["b"], "--nb", 1]],
                   check=True, capture_output=True)
    lines = {}
    with open(out / "boundary.csv", newline="") as f:
        for r in csv.DictReader(f):
            lines.setdefault(r["polyline"], []).append(
                (float(r["q_z"]) / co["q_per_vac"], float(r["a_z"]) / co["a_per_vdc"], r["closed"] == "1"))
    result = []
    for pts in lines.values():
        vac = [p[0] for p in pts]
        vdc = [p[1] for p in pts]
        if pts[0][2]:
            vac, vdc = vac + vac[:1], vdc + vdc[:1]
        result.append((vac, vdc))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("config", type=Path)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--output", type=Path, default=None,
                   help="output directory (default results/capture/<config stem>)")
    p.add_argument("--plot-only", action="store_true", help="reuse an existing scan in the output directory")
    p.add_argument("--emit-jobs", type=int, metavar="N_CHUNKS", default=None,
                   help="write N_CHUNKS LSF job scripts to <output>/jobs and stop (nothing is submitted)")
    p.add_argument("--queue", default=None, help="LSF queue written into the job scripts")
    p.add_argument("--merge", action="store_true",
                   help="merge <output>/chunks/chunk_* into <output>/scan, then plot")
    args = p.parse_args(argv)
    out = args.output or REPO / "results" / "capture" / args.config.stem
    scan = out / "scan"

    if args.emit_jobs is not None:
        scripts = emit_jobs(args.config, out, args.emit_jobs, args.threads, args.queue)
        print(f"wrote {len(scripts)} job scripts to {out / 'jobs'} (not submitted)")
        print(f"submit each with: bsub < {scripts[0]}  ...; then run with --merge")
        return
    if args.merge:
        chunks = sorted((out / "chunks").glob("chunk_*"))
        if not chunks:
            raise SystemExit(f"no chunks in {out / 'chunks'}")
        merge_chunks(chunks, scan)
        setup, cap, co = load_capture(args.config)
    elif args.plot_only:
        setup, cap, co = load_capture(args.config)
    else:
        setup, cap, co = run_capture(args.config, scan, args.threads)
    meta = json.loads((scan / "metadata.json").read_text())
    d = {n: np.load(scan / f"{n}.npy") for n in
         ("v_dc", "v_ac", "p_capture", "wilson_lo", "wilson_hi", "n_ring", "n_endcap", "n_trapped",
          "p_capture_at", "checkpoint_time", "floquet_class")}
    bnd = floquet_boundary_in_volts(co, cap, out / "floquet_boundary", args.threads)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 9})
    vac, vdc = d["v_ac"], d["v_dc"]
    N = meta["samples"]
    info = (f"b = {co['b']:.3g}, f = {setup.frequency:g} Hz, r0 = {setup.r0 * 1e3:g} mm, "
            f"T_obs = RF {meta['rf_periods']:g} 周期 ({meta['observation_time'] * 1e3:.4g} ms), "
            f"N = {N}/点, seed = {meta['seed']}; 初期位置 半径 {cap['pos_radius'] * 1e3:g} mm の球内一様, "
            f"初速度 σ = {cap['vel_sigma']:g} m/s, RF位相 一様"
            + (f", 粒径 {cap['radius']}" if "radius" in cap else "")
            + (f", 電荷 {cap['charge']}" if "charge" in cap else ""))

    def overlay(ax):
        for k, (x, y) in enumerate(bnd):
            ax.plot(x, y, color="k", lw=0.9, label="Floquet 安定境界" if k == 0 else None)
        ax.set_xlim(vac.min(), vac.max())
        ax.set_ylim(vdc.min(), vdc.max())
        ax.set_xlabel(r"$V_{AC}$ [V]（振幅）")
        ax.set_ylabel(r"$V_{DC}$ [V]")

    fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))
    fig.suptitle("捕獲確率マップ（幾何学的捕獲: 観測時間内に電極へ到達しない割合）",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.935, info, fontsize=8.5, color="0.3")

    ax = axs[0, 0]
    im = ax.pcolormesh(vac, vdc, d["p_capture"], cmap="viridis", vmin=0, vmax=1, shading="nearest",
                       rasterized=True)
    fig.colorbar(im, ax=ax, label="捕獲確率 P")
    overlay(ax)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("捕獲確率 P")

    ax = axs[0, 1]
    half = (d["wilson_hi"] - d["wilson_lo"]) / 2
    im = ax.pcolormesh(vac, vdc, half, cmap="magma", shading="nearest", rasterized=True)
    fig.colorbar(im, ax=ax, label="95% Wilson 区間の半幅")
    overlay(ax)
    ax.set_title("統計誤差")

    ax = axs[1, 0]
    escaped = d["n_ring"] + d["n_endcap"]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_ring = np.where(escaped > 0, d["n_ring"] / escaped, np.nan)
    im = ax.pcolormesh(vac, vdc, frac_ring, cmap="coolwarm", vmin=0, vmax=1, shading="nearest", rasterized=True)
    fig.colorbar(im, ax=ax, label="脱出のうちリング電極に当たった割合")
    overlay(ax)
    ax.set_title("脱出先（赤: リング、青: エンドキャップ、白: 脱出なし）")

    ax = axs[1, 1]
    i0 = int(np.argmin(np.abs(vdc)))
    tc = d["checkpoint_time"] * 1e3
    for c in range(len(tc)):
        ax.plot(vac, d["p_capture_at"][c, i0], lw=1, label=f"T = {tc[c]:.4g} ms")
    ax.fill_between(vac, d["wilson_lo"][i0], d["wilson_hi"][i0], color="C3", alpha=0.2, label="95% 区間（最終）")
    stable = d["floquet_class"][i0] == 0
    ax.fill_between(vac, 0, 1.05, where=stable, color="0.85", step="mid", label="Floquet 安定", zorder=0)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(vac.min(), vac.max())
    ax.set_xlabel(r"$V_{AC}$ [V]")
    ax.set_ylabel("捕獲確率")
    ax.set_title(f"V_DC = {vdc[i0]:g} V での観測時間依存")
    ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "capture_map.png", dpi=130)
    fig.savefig(out / "capture_map.pdf")
    plt.close(fig)
    print(f"wrote {out / 'capture_map.png'} (scan {meta['elapsed_seconds']:.1f} s)")


if __name__ == "__main__":
    main()
