#!/usr/bin/env python3
"""Explanatory figures for the Floquet stability work (Phase 1-3).

Runs the C++ programs (build first with ./build.sh) and the Python reference,
then writes a multi-page PDF and one PNG per page to results/explainer/.
"""

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import mathieu_a, mathieu_b

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import floquet_reference as fr  # noqa: E402
from plot_stability import read_boundary  # noqa: E402

BIN = REPO / ".build" / "bin"


def run(prog, out, *args, threads=8):
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(BIN / prog), "--output", str(out), "--threads", str(threads),
                    *map(str, args)], check=True, capture_output=True)
    return out


def load_scan(d):
    return {k: np.load(d / f"{k}.npy") for k in ("a_z", "q_z", "axial", "radial", "combined")}


def trajectory(a, q, b, tau_max, x0=1.0, v0=0.0):
    tau = np.linspace(0, tau_max, 4000)
    sol = solve_ivp(lambda t, s: [s[1], -b * s[1] - (a - 2 * q * np.cos(2 * t)) * s[0]],
                    (0, tau_max), [x0, v0], t_eval=tau, method="DOP853", rtol=1e-10, atol=1e-12)
    return sol.t, sol.y[0]


def page_title(fig, title, subtitle=None):
    fig.suptitle(title, fontsize=15, fontweight="bold", x=0.02, ha="left")
    if subtitle:
        fig.text(0.02, 0.905, subtitle, fontsize=9.5, ha="left", va="top", color="0.25")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--output", type=Path, default=REPO / "results" / "explainer")
    args = p.parse_args(argv)
    out = args.output
    work = out / "work"
    th = args.threads

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import ListedColormap

    plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    cls_cmap = ListedColormap(["#4c9be8", "#f2c14e", "#f4f4f4"])
    pages = []

    # ---- Page 1: what stable / unstable means -------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    page_title(fig, "1. 1方向の運動: 安定と不安定",
               r"$x'' + b\,x' + (a - 2q\cos 2\tau)\,x = 0$,  $\tau = \Omega t/2$.  "
               r"RF 1周期は $\tau$ で $\pi$。初期条件 $x=1, x'=0$。")
    cases = [(0.0, 0.5, 0.0, "a=0, q=0.5, b=0: 安定（有界な振動。マイクロモーション + 永年運動）"),
             (0.0, 0.95, 0.0, "a=0, q=0.95, b=0: 不安定（q > 0.908 で指数的に発散）"),
             (0.0, 0.95, 0.5, "a=0, q=0.95, b=0.5: 減衰で安定化（同じ q でも空気抵抗があれば減衰）")]
    for ax, (a, q, b, label) in zip(axs, cases):
        t, x = trajectory(a, q, b, 20 * np.pi)
        ax.plot(t / np.pi, x, lw=1, color="#1f5fa8")
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_ylabel("x")
        ax.axhline(0, color="0.7", lw=0.5)
    axs[-1].set_xlabel(r"$\tau/\pi$ （RF周期数）")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    pages.append(("01_trajectories", fig))

    # ---- Page 2: monodromy and Floquet multipliers -------------------------
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.6))
    page_title(fig, "2. モノドロミー行列 M と Floquet 乗数 λ",
               "1周期 π 積分して [x, x'] を写す 2×2 行列 M の固有値 λ が判定量。"
               "b=0 では λ は単位円上（安定）か実軸上（不安定）。b>0 では全体が exp(-bπ/2) 倍に縮む。")
    th_ = np.linspace(0, 2 * np.pi, 400)
    for ax, b in zip(axs, (0.0, 0.3)):
        qs = np.linspace(0.0, 1.2, 241)
        lam = np.array([fr.floquet_multipliers(fr.monodromy(0.0, q, b)) for q in qs])
        ax.plot(np.cos(th_), np.sin(th_), color="0.6", lw=1, label="|λ| = 1")
        if b > 0:
            r = np.exp(-b * np.pi / 2)
            ax.plot(r * np.cos(th_), r * np.sin(th_), color="0.6", lw=1, ls="--",
                    label=r"|λ| = exp(-bπ/2)")
        for k in range(2):
            sc = ax.scatter(lam[:, k].real, lam[:, k].imag, c=qs, cmap="viridis", s=8)
        ax.set_aspect("equal")
        ax.set_xlim(-2.6, 1.4)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel("Re λ")
        ax.set_ylabel("Im λ")
        ax.set_title(f"a = 0, b = {b:g}, q を 0→1.2 に動かす", fontsize=10)
        ax.legend(loc="lower left", fontsize=8)
    fig.colorbar(sc, ax=axs, label="q", shrink=0.8)
    fig.text(0.02, 0.03, "q≈0.908 で 2つの λ が -1 で衝突し実軸上に分かれる → |λ|>1 の側が発散。"
             "判定: b>0 は max|λ|<1、b=0 は |tr M|<2（stability.md）。", fontsize=9)
    pages.append(("02_floquet_multipliers", fig))

    # ---- Page 3: axial, radial, combined at b = 0 --------------------------
    s0 = load_scan(run("floquet_scan", work / "scan_b0", "--a-range", -0.8, 0.3, "--q-range", 0, 1.5,
                       "--na", 361, "--nq", 421, "--b", 0, threads=th))
    fig, axs = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    page_title(fig, "3. 軸方向・半径方向・トラップ全体 (b = 0)",
               r"半径方向は $a_r=-a_z/2,\ q_r=-q_z/2$ で同じ Mathieu 方程式。両方が安定な点だけがトラップの安定領域。"
               "  破線: Mathieu 特性値 a0(|q|), b1(|q|)（scipy、独立な参照。黒: 軸方向、赤: 半径方向）。")
    q = s0["q_z"]
    qq = np.linspace(1e-6, 1.5, 300)
    for ax, key, title in zip(axs, ("axial", "radial", "combined"),
                              ("軸方向 z", "半径方向 r", "トラップ全体（共通部分）")):
        ax.pcolormesh(q, s0["a_z"], s0[key], cmap=cls_cmap, vmin=0, vmax=2, shading="nearest",
                      rasterized=True)
        if key in ("axial", "combined"):
            ax.plot(qq, mathieu_a(0, qq), "k--", lw=0.8)
            ax.plot(qq, mathieu_b(1, qq), "k--", lw=0.8)
        if key in ("radial", "combined"):
            ax.plot(qq, -2 * mathieu_a(0, qq / 2), "r--", lw=0.8)
            ax.plot(qq, -2 * mathieu_b(1, qq / 2), "r--", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$q_z$")
        ax.set_ylim(-0.8, 0.3)
    axs[0].set_ylabel(r"$a_z$")
    axs[2].annotate("q_z = 0.908", (0.908, 0.0), (1.05, 0.15), arrowprops=dict(arrowstyle="->"), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    pages.append(("03_axial_radial_b0", fig))

    # ---- Page 4: effect of damping (boundary slices) ------------------------
    bdir = run("floquet_boundary", work / "boundary", "--na", 481, "--nq", 481,
               "--b-range", 0, 10, "--nb", 11, threads=th)
    slices = read_boundary(bdir / "boundary.csv")
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
    page_title(fig, "4. 空気抵抗 b による安定境界の変化",
               "floquet_boundary: 粗格子 481² で符号変化を検出 → 二分法で境界点を 1e-9 まで追い込む。"
               "左: 全体、右: 第一安定領域付近の拡大。")
    cmap = plt.get_cmap("viridis")
    bs = list(slices)
    for ax in axs:
        for k, (b, pls) in enumerate(slices.items()):
            for n, pl in enumerate(pls.values()):
                qz = pl["q_z"] + (pl["q_z"][:1] if pl["closed"] else [])
                az = pl["a_z"] + (pl["a_z"][:1] if pl["closed"] else [])
                ax.plot(qz, az, color=cmap(k / (len(bs) - 1)), lw=0.9,
                        label=f"b = {b:g}" if n == 0 else None)
        ax.set_xlabel(r"$q_z$")
        ax.set_ylabel(r"$a_z$")
    axs[0].legend(fontsize=8, ncol=2, loc="lower right")
    axs[1].set_xlim(0, 4)
    axs[1].set_ylim(-1.5, 1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    pages.append(("04_damping_slices", fig))

    # ---- Page 5: 3D boundary -----------------------------------------------
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(projection="3d")
    page_title(fig, "5. (q_z, b, a_z) 空間の境界面", "各 b の境界曲線を b 方向に積み重ねたもの（b = 0, 1, …, 10）。")
    for k, (b, pls) in enumerate(slices.items()):
        for pl in pls.values():
            qz = pl["q_z"] + (pl["q_z"][:1] if pl["closed"] else [])
            az = pl["a_z"] + (pl["a_z"][:1] if pl["closed"] else [])
            ax.plot(qz, [b] * len(qz), az, color=cmap(k / (len(bs) - 1)), lw=0.8)
    ax.set_xlabel(r"$q_z$")
    ax.set_ylabel(r"$b$")
    ax.set_zlabel(r"$a_z$")
    ax.view_init(elev=22, azim=-60)
    pages.append(("05_boundary_3d", fig))

    # ---- Page 6: a_z = 0 line, q_z vs b -------------------------------------
    b_list = np.linspace(0, 10, 201)
    q_list = None
    rows = []
    for b in b_list:
        d = load_scan(run("floquet_scan", work / "line" / f"b{b:.3f}", "--a-range", 0, 0, "--na", 1,
                          "--q-range", 0, 30, "--nq", 1501, "--b", b, threads=th))
        q_list = d["q_z"]
        rows.append(d["combined"][0])
    grid = np.array(rows)
    fig, ax = plt.subplots(figsize=(11, 6))
    page_title(fig, "6. a_z = 0（DC なし）での安定な q_z と b",
               "横軸 q_z、縦軸 b。青が安定。b が大きいと高い q_z まで安定になり、上側に安定な帯が離れて現れる。")
    ax.pcolormesh(q_list, b_list, grid, cmap=cls_cmap, vmin=0, vmax=2, shading="nearest", rasterized=True)
    ax.plot(2.854288 * b_list, b_list, "r--", lw=1, label="既存ノートのフィット q_max = 2.854 b（参考値）")
    ax.plot([18.47], [6.04667], "ko", ms=5)
    ax.annotate("b = 6.05 で最初の不安定化 q_z = 18.47\n（ノートは 17.26）", (18.47, 6.05), (20, 3.5),
                arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.set_xlabel(r"$q_z$")
    ax.set_ylabel(r"$b$")
    ax.set_xlim(0, 30)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    pages.append(("06_az0_q_vs_b", fig))

    # ---- Page 7: validation -------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8))
    page_title(fig, "7. 検証のまとめ",
               "左: b=0 境界と Mathieu 特性値の差。中: 1周期 step 数に対する境界点の移動（16000 step 基準、q_z>0.5）。"
               "右: OpenMP スレッド数と実行時間（約10万点）。")
    b0 = run("floquet_boundary", work / "b0check", "--b-range", 0, 0, "--nb", 1, "--a-range", -1.5, 1.5,
             "--q-range", 0, 4, "--na", 121, "--nq", 161, threads=th)
    res_q, res = [], []
    with open(b0 / "boundary.csv", newline="") as f:
        for r in csv.DictReader(f):
            az, qz = float(r["a_z"]), float(r["q_z"])
            axial = r["limiting"] == "axial"
            a, qv, scale = (az, qz, 1.0) if axial else (-az / 2, -qz / 2, 2.0)
            qa = abs(qv)
            vals = [mathieu_a(n, qa) for n in range(12)] + [mathieu_b(n, qa) for n in range(1, 12)]
            res_q.append(qz)
            res.append(scale * min(abs(a - v) for v in vals) if qa > 0 else np.nan)
    axs[0].semilogy(res_q, np.maximum(res, 1e-17), ".", ms=3)
    axs[0].axhline(1e-6, color="r", ls="--", lw=0.8, label="目標 1e-6")
    axs[0].set_xlabel(r"$q_z$")
    axs[0].set_ylabel(r"$|a_z - $特性値$|$")
    axs[0].legend(fontsize=8)

    def load_pts(d):
        pts = {}
        with open(d / "boundary.csv", newline="") as f:
            for r in csv.DictReader(f):
                pts[(r["b"], r["edge_axis"], r["edge_i"], r["edge_j"])] = (float(r["a_z"]), float(r["q_z"]))
        return pts
    conv_args = ("--b-range", 0, 8, "--nb", 3)
    ref = load_pts(run("floquet_boundary", work / "conv16000", *conv_args, "--steps", 16000, threads=th))
    steps = [500, 1000, 2000, 4000, 8000]
    shift = defaultdict(list)
    for n in steps:
        d = load_pts(run("floquet_boundary", work / f"conv{n}", *conv_args, "--steps", n, threads=th))
        per_b = defaultdict(float)
        for k in set(d) & set(ref):
            if ref[k][1] > 0.5:
                per_b[k[0]] = max(per_b[k[0]], np.hypot(d[k][0] - ref[k][0], d[k][1] - ref[k][1]))
        for b, v in per_b.items():
            shift[b].append(v)
    for b, v in sorted(shift.items(), key=lambda x: float(x[0])):
        axs[1].loglog(steps, np.maximum(v, 1e-17), "o-", label=f"b = {float(b):g}")
    axs[1].axhline(1e-6, color="r", ls="--", lw=0.8)
    axs[1].axvline(2000, color="0.5", ls=":", lw=0.8)
    axs[1].set_xlabel("1周期あたり step 数")
    axs[1].set_ylabel("境界点の最大移動")
    axs[1].legend(fontsize=8)

    import json
    threads_list = [1, 2, 4, 8]
    times = []
    for t in threads_list:
        d = run("floquet_scan", work / f"perf{t}", "--na", 316, "--nq", 316, threads=t)
        times.append(json.loads((d / "metadata.json").read_text())["elapsed_seconds"])
    axs[2].plot(threads_list, times, "o-", label="実測")
    axs[2].plot(threads_list, times[0] / np.array(threads_list), "k--", lw=0.8, label="理想 (1/N)")
    axs[2].set_xlabel("スレッド数")
    axs[2].set_ylabel("時間 [s]")
    axs[2].set_xticks(threads_list)
    axs[2].legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    pages.append(("07_validation", fig))

    out.mkdir(parents=True, exist_ok=True)
    with PdfPages(out / "explainer.pdf") as pdf:
        for name, fig in pages:
            pdf.savefig(fig)
            fig.savefig(out / f"{name}.png", dpi=130)
            plt.close(fig)
    print(f"wrote {out / 'explainer.pdf'} and {len(pages)} PNG files")


if __name__ == "__main__":
    main()
