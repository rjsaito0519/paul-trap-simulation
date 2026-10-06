#!/usr/bin/env python3
"""Whole-region stability plots, one panel per b, with automatic plot ranges.

For each b the (a_z, q_z) window is enlarged until no stable grid node touches
the a_z or upper q_z edge, then the region is filled (floquet_scan) and its
boundary traced (floquet_boundary). Boundary segments are coloured by the
direction that limits stability. Build the C++ programs first (./build.sh).
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BIN = REPO / ".build" / "bin"


def _run(prog, out, args, threads):
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(BIN / prog), "--output", str(out), "--threads", str(threads),
                    *map(str, args)], check=True, capture_output=True)
    return out


def find_window(b, work, threads, n=161, a0=1.0, q0=2.0, max_doublings=10):
    """Smallest doubled window [-A, A] x [0, Q] whose edges contain no stable node."""
    A, Q = a0, q0
    for _ in range(max_doublings):
        d = _run("floquet_scan", work / "window", ["--a-range", -A, A, "--q-range", 0, Q,
                                                   "--na", n, "--nq", n, "--b", b], threads)
        s = np.load(d / "combined.npy") == 0
        touch_a = s[0].any() or s[-1].any()
        touch_q = s[:, -1].any()
        if not (touch_a or touch_q):
            if not s.any():
                return None
            a = np.load(d / "a_z.npy")
            q = np.load(d / "q_z.npy")
            ia, iq = np.nonzero(s)
            da, dq = a[1] - a[0], q[1] - q[0]
            return (a[ia].min() - da, a[ia].max() + da, 0.0, q[iq].max() + dq)
        if touch_a:
            A *= 2
        if touch_q:
            Q *= 2
    raise RuntimeError(f"stable region at b={b} not bounded within the search limit")


def padded(win, frac=0.08):
    a_lo, a_hi, q_lo, q_hi = win
    pa, pq = frac * (a_hi - a_lo), frac * (q_hi - q_lo)
    return a_lo - pa, a_hi + pa, q_lo, q_hi + pq


def compute(b_values, work, threads, n_fill=401, n_trace=401):
    """Return a list of per-b results with window, fill grid and boundary rows."""
    import csv
    results = []
    for b in b_values:
        wdir = work / f"b{b:g}".replace(".", "p")
        win = find_window(b, wdir, threads)
        if win is None:
            results.append({"b": b, "window": None})
            continue
        a_lo, a_hi, q_lo, q_hi = padded(win)
        rng = ["--a-range", a_lo, a_hi, "--q-range", q_lo, q_hi]
        fill = _run("floquet_scan", wdir / "fill", rng + ["--na", n_fill, "--nq", n_fill, "--b", b], threads)
        bnd = _run("floquet_boundary", wdir / "boundary",
                   rng + ["--na", n_trace, "--nq", n_trace, "--b-range", b, b, "--nb", 1], threads)
        with open(bnd / "boundary.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        meta = json.loads((bnd / "metadata.json").read_text())["slices"][0]
        results.append({"b": b, "window": (a_lo, a_hi, q_lo, q_hi),
                        "a_z": np.load(fill / "a_z.npy"), "q_z": np.load(fill / "q_z.npy"),
                        "combined": np.load(fill / "combined.npy"), "rows": rows,
                        "regions": meta["regions"]})
    return results


def draw_panel(ax, res, cls_cmap):
    """Filled stable region plus boundary coloured by the limiting direction."""
    ax.pcolormesh(res["q_z"], res["a_z"], res["combined"], cmap=cls_cmap, vmin=0, vmax=2,
                  shading="nearest", rasterized=True)
    polylines = {}
    for r in res["rows"]:
        polylines.setdefault(r["polyline"], []).append(r)
    for pts in polylines.values():
        q = [float(r["q_z"]) for r in pts]
        a = [float(r["a_z"]) for r in pts]
        lim = [r["limiting"] for r in pts]
        if pts[0]["closed"] == "1":
            q, a, lim = q + q[:1], a + a[:1], lim + lim[:1]
        for k in range(len(q) - 1):
            ax.plot(q[k:k + 2], a[k:k + 2], color="k" if lim[k] == "axial" else "#d62728", lw=1.0)
    ax.set_xlim(res["window"][2], res["window"][3])
    ax.set_ylim(res["window"][0], res["window"][1])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--b", type=float, nargs="+", default=[0, 0.5, 1, 2, 3, 4, 6, 8, 10])
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--output", type=Path, default=REPO / "results" / "stability_overview")
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 9})
    cls_cmap = ListedColormap(["#a9cdf2", "#f2c14e", "#f7f7f7"])
    results = compute(args.b, args.output / "work", args.threads)

    n = len(results)
    ncol = 3
    nrow = -(-n // ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(13, 3.9 * nrow), squeeze=False)
    for ax, res in zip(axs.flat, results):
        if res["window"] is None:
            ax.set_title(f"b = {res['b']:g}: 安定点なし")
            continue
        draw_panel(ax, res, cls_cmap)
        ax.set_title(f"b = {res['b']:g}", fontsize=10)
        ax.set_xlabel(r"$q_z$")
        ax.set_ylabel(r"$a_z$")
    for ax in axs.flat[n:]:
        ax.axis("off")
    handles = [Line2D([], [], color="k", label="軸方向で不安定化"),
               Line2D([], [], color="#d62728", label="半径方向で不安定化"),
               plt.Rectangle((0, 0), 1, 1, color=cls_cmap(0), label="安定")]
    fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=9)
    fig.suptitle("トラップ全体の Floquet 安定領域（b ごとに範囲を自動調整）",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.005, "注: b が大きいと細い舌状の安定帯が遠くまで伸び、格子解像度のため点線状に途切れて見える。"
             "範囲は粗格子で検出できた安定点を含むように決めているため、細い帯は範囲外へ続くことがある。",
             fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "stability_overview.png", dpi=130)
    fig.savefig(args.output / "stability_overview.pdf")
    plt.close(fig)

    for res in results:
        if res["window"] is None:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        draw_panel(ax, res, cls_cmap)
        ax.set_title(f"Floquet 安定領域 b = {res['b']:g}")
        ax.set_xlabel(r"$q_z$")
        ax.set_ylabel(r"$a_z$")
        ax.legend(handles=handles, fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(args.output / (f"b{res['b']:g}".replace(".", "p") + ".png"), dpi=130)
        plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
