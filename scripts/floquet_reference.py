#!/usr/bin/env python3
"""Python reference implementation of the Floquet stability test (Phase 1).

Each direction obeys the damped Mathieu equation in dimensionless time tau:

    x'' + b x' + (a - 2 q cos(2 tau)) x = 0

The monodromy matrix M maps [x, dx/dtau] at tau = 0 to tau = pi.
Conventions follow docs/simulation/model.md and docs/simulation/stability.md.

This module is an explicit cross-check reference, not the production engine.
"""

import argparse
import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp

PERIOD = np.pi

STABLE = 0
BOUNDARY = 1
UNSTABLE = 2
CLASS_NAMES = {STABLE: "stable", BOUNDARY: "boundary", UNSTABLE: "unstable"}

# Provisional classification tolerance (docs/simulation/stability.md, 2026-10-06).
DEFAULT_EPS = 1e-8
DEFAULT_RTOL = 1e-12
DEFAULT_ATOL = 1e-14


def _rhs(tau, s, a, q, b):
    # s = [x1, v1, x2, v2]: two fundamental solutions integrated together.
    k = a - 2.0 * q * np.cos(2.0 * tau)
    return [s[1], -b * s[1] - k * s[0], s[3], -b * s[3] - k * s[2]]


def monodromy(a, q, b, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, method="DOP853"):
    """Return the 2x2 monodromy matrix over one period pi.

    Columns are the solutions started from [1, 0] and [0, 1].
    """
    sol = solve_ivp(_rhs, (0.0, PERIOD), [1.0, 0.0, 0.0, 1.0],
                    method=method, args=(a, q, b), rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"integration failed at a={a}, q={q}, b={b}: {sol.message}")
    x1, v1, x2, v2 = sol.y[:, -1]
    return np.array([[x1, x2], [v1, v2]])


def floquet_multipliers(m):
    """Eigenvalues of a 2x2 matrix from its trace and determinant."""
    tr = m[0, 0] + m[1, 1]
    det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
    disc = np.sqrt(complex(tr * tr - 4.0 * det))
    return np.array([(tr + disc) / 2.0, (tr - disc) / 2.0])


def classify(m, b, eps=DEFAULT_EPS):
    """Classify one direction from its monodromy matrix.

    b > 0: asymptotic stability, rho = max|lambda| compared with 1.
    b = 0: det M = 1 and rho = 1 throughout the stable region, so bounded
           (Lyapunov) stability |tr M| < 2 is used instead. This is the
           b -> 0+ limit of the b > 0 criterion.
    """
    if b > 0.0:
        x = np.max(np.abs(floquet_multipliers(m)))
        lim = 1.0
    elif b == 0.0:
        x = abs(m[0, 0] + m[1, 1])
        lim = 2.0
    else:
        raise ValueError("b must be non-negative")
    if x < lim - eps:
        return STABLE
    if x > lim + eps:
        return UNSTABLE
    return BOUNDARY


def radial_params(a_z, q_z):
    """Radial (a_r, q_r) for an ideal quadrupole potential proportional to r^2 - 2 z^2."""
    return -a_z / 2.0, -q_z / 2.0


def trap_classification(a_z, q_z, b, eps=DEFAULT_EPS, **kw):
    """Return (axial, radial, combined) classes for one (a_z, q_z, b).

    The combined class is the worst of the two directions.
    """
    a_r, q_r = radial_params(a_z, q_z)
    c_ax = classify(monodromy(a_z, q_z, b, **kw), b, eps)
    c_ra = classify(monodromy(a_r, q_r, b, **kw), b, eps)
    return c_ax, c_ra, max(c_ax, c_ra)


def _git_info(repo):
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    rev = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"revision": rev or "unknown",
            "dirty": None if status is None else bool(status)}


def scan(a_values, q_values, b, eps=DEFAULT_EPS, **kw):
    """Classify a rectangular (a_z, q_z) grid. Arrays are indexed [i_a, i_q]."""
    shape = (len(a_values), len(q_values))
    axial = np.empty(shape, dtype=np.int8)
    radial = np.empty(shape, dtype=np.int8)
    det_err = np.empty(shape)
    ref_det = np.exp(-b * PERIOD)
    for i, a_z in enumerate(a_values):
        for j, q_z in enumerate(q_values):
            a_r, q_r = radial_params(a_z, q_z)
            m_ax = monodromy(a_z, q_z, b, **kw)
            m_ra = monodromy(a_r, q_r, b, **kw)
            axial[i, j] = classify(m_ax, b, eps)
            radial[i, j] = classify(m_ra, b, eps)
            det_err[i, j] = max(abs(np.linalg.det(m_ax) - ref_det),
                                abs(np.linalg.det(m_ra) - ref_det))
    return axial, radial, np.maximum(axial, radial), det_err


def plot_scan(path, a_values, q_values, b, combined):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(6, 4.5))
    cmap = ListedColormap(["#4c9be8", "#f2c14e", "#eeeeee"])
    ax.pcolormesh(q_values, a_values, combined, cmap=cmap, vmin=0, vmax=2,
                  shading="nearest")
    ax.set_xlabel(r"$q_z$")
    ax.set_ylabel(r"$a_z$")
    ax.set_title(f"Floquet stability (axial and radial), b = {b:g}")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(k)) for k in range(3)]
    ax.legend(handles, [CLASS_NAMES[k] for k in range(3)], loc="upper right",
              fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--a-range", type=float, nargs=2, default=(-0.4, 0.2), metavar=("MIN", "MAX"))
    p.add_argument("--q-range", type=float, nargs=2, default=(0.0, 1.2), metavar=("MIN", "MAX"))
    p.add_argument("--na", type=int, default=61)
    p.add_argument("--nq", type=int, default=61)
    p.add_argument("--b", type=float, default=0.0)
    p.add_argument("--eps", type=float, default=DEFAULT_EPS)
    p.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    p.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    p.add_argument("--output", type=Path, default=None,
                   help="output stem (default: results/floquet_reference_b<b>)")
    args = p.parse_args(argv)

    repo = Path(__file__).resolve().parent.parent
    tag = f"{args.b:g}".replace(".", "p")
    stem = args.output or repo / "results" / f"floquet_reference_b{tag}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    # Append suffixes explicitly: with_suffix() would treat "b0.5" as ".5".
    npz_path = stem.parent / (stem.name + ".npz")
    png_path = stem.parent / (stem.name + ".png")

    a_values = np.linspace(*args.a_range, args.na)
    q_values = np.linspace(*args.q_range, args.nq)
    axial, radial, combined, det_err = scan(a_values, q_values, args.b, args.eps,
                                            rtol=args.rtol, atol=args.atol)

    meta = {
        "created": datetime.datetime.now().astimezone().isoformat(),
        "command": sys.argv,
        "git": _git_info(repo),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "solver": "scipy.integrate.solve_ivp/DOP853",
        "rtol": args.rtol, "atol": args.atol, "eps": args.eps,
        "b": args.b,
        "classes": CLASS_NAMES,
        "max_det_error": float(det_err.max()),
    }
    np.savez_compressed(npz_path, a_z=a_values, q_z=q_values,
                        axial=axial, radial=radial, combined=combined,
                        det_error=det_err, metadata=json.dumps(meta))
    plot_scan(png_path, a_values, q_values, args.b, combined)
    print(f"wrote {npz_path} and {png_path}")
    print(f"max |det M - exp(-b pi)| = {det_err.max():.3e}")


if __name__ == "__main__":
    main()
