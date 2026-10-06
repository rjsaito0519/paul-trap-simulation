#!/usr/bin/env python3
"""Plot stability boundaries written by floquet_boundary (boundary.csv).

Produces a 2D overlay of (q_z, a_z) boundaries coloured by b and a 3D view
of the boundary curves stacked along b.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_boundary(path):
    """Return {b: {polyline: dict(a_z, q_z, closed, limiting, region)}} in CSV order."""
    slices = defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            b = float(row["b"])
            pl = slices[b].setdefault(int(row["polyline"]), {
                "a_z": [], "q_z": [], "limiting": [], "closed": row["closed"] == "1",
                "region": int(row["region"])})
            pl["a_z"].append(float(row["a_z"]))
            pl["q_z"].append(float(row["q_z"]))
            pl["limiting"].append(row["limiting"])
    return dict(sorted(slices.items()))


def _closed(xs, closed):
    return xs + xs[:1] if closed else xs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("directory", type=Path, help="floquet_boundary output directory")
    p.add_argument("--b", type=float, nargs="*", help="only these b values")
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slices = read_boundary(args.directory / "boundary.csv")
    if args.b:
        slices = {b: v for b, v in slices.items() if any(abs(b - x) < 1e-12 for x in args.b)}
    bs = list(slices)
    cmap = plt.get_cmap("viridis")
    color = {b: cmap(k / max(1, len(bs) - 1)) for k, b in enumerate(bs)}

    fig, ax = plt.subplots(figsize=(7, 5))
    for b, pls in slices.items():
        for k, pl in enumerate(pls.values()):
            ax.plot(_closed(pl["q_z"], pl["closed"]), _closed(pl["a_z"], pl["closed"]),
                    color=color[b], lw=1, label=f"b = {b:g}" if k == 0 else None)
    ax.set_xlabel(r"$q_z$")
    ax.set_ylabel(r"$a_z$")
    ax.set_title("Floquet stability boundary (axial and radial)")
    ax.legend(fontsize="small", ncol=2)
    fig.tight_layout()
    out2d = args.directory / "boundary_2d.png"
    fig.savefig(out2d, dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    for b, pls in slices.items():
        for pl in pls.values():
            q = _closed(pl["q_z"], pl["closed"])
            ax.plot(q, [b] * len(q), _closed(pl["a_z"], pl["closed"]), color=color[b], lw=0.8)
    ax.set_xlabel(r"$q_z$")
    ax.set_ylabel(r"$b$")
    ax.set_zlabel(r"$a_z$")
    fig.tight_layout()
    out3d = args.directory / "boundary_3d.png"
    fig.savefig(out3d, dpi=150)
    plt.close(fig)
    print(f"wrote {out2d} and {out3d}")


if __name__ == "__main__":
    main()
