#!/usr/bin/env python3
"""Python reference for 3D single-particle trajectories in an ideal Paul trap (Phase 4).

Equations (docs/simulation/model.md), dimensionless time tau = (Omega t + phase) / 2,
positions in metres, derivatives with respect to tau:

    x'' + b x' + (a_r - 2 q_r cos 2 tau) x = 0          (same for y)
    z'' + b z' + (a_z - 2 q_z cos 2 tau) z + 4 g / Omega^2 = 0

A particle escapes when it reaches an ideal hyperbolic electrode surface
(docs/simulation/stability.md):

    ring:    x^2 + y^2 - 2 z^2 = r0^2
    endcaps: 2 z^2 - (x^2 + y^2) = 2 z0^2 = r0^2
"""

import argparse
import math
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import floquet_reference as fr  # noqa: E402


@dataclass
class Particle:
    label: str
    position: np.ndarray  # m
    velocity: np.ndarray  # m/s (physical time)


@dataclass
class Setup:
    radius: float
    density: float
    charge: float
    viscosity: float
    r0: float
    v_dc: float
    v_ac: float
    frequency: float
    rf_phase: float = 0.0
    gravity: float = 9.80665
    duration: float = 0.1
    particles: list = field(default_factory=list)

    # ---- derived quantities (SI) ----
    @property
    def omega(self):
        return 2.0 * math.pi * self.frequency

    @property
    def mass(self):
        return 4.0 / 3.0 * math.pi * self.radius ** 3 * self.density

    @property
    def drag(self):
        return 6.0 * math.pi * self.viscosity * self.radius

    @property
    def z0(self):
        return self.r0 / math.sqrt(2.0)

    @property
    def a_z(self):
        # "+ 0.0" avoids printing -0 when V_DC = 0.
        return -8.0 * self.charge * self.v_dc / (self.mass * self.omega ** 2 * self.r0 ** 2) + 0.0

    @property
    def q_z(self):
        return 4.0 * self.charge * self.v_ac / (self.mass * self.omega ** 2 * self.r0 ** 2)

    @property
    def b(self):
        return 2.0 * self.drag / (self.mass * self.omega)

    @property
    def gravity_term(self):
        """4 g / Omega^2 [m], the constant term in the dimensionless z equation."""
        return 4.0 * self.gravity / self.omega ** 2

    # ---- time and velocity conversions ----
    def tau(self, t):
        return (self.omega * np.asarray(t) + self.rf_phase) / 2.0

    def time(self, tau):
        return (2.0 * np.asarray(tau) - self.rf_phase) / self.omega

    def to_tau_velocity(self, v):
        """Physical velocity [m/s] -> d/dtau [m]."""
        return 2.0 * np.asarray(v) / self.omega

    def to_physical_velocity(self, u):
        return self.omega * np.asarray(u) / 2.0


def load_setup(path):
    with open(path, "rb") as f:
        c = tomllib.load(f)
    s = Setup(radius=c["particle"]["radius"], density=c["particle"]["density"],
              charge=c["particle"]["charge"], viscosity=c["gas"]["viscosity"],
              r0=c["trap"]["r0"], v_dc=c["trap"]["v_dc"], v_ac=c["trap"]["v_ac"],
              frequency=c["trap"]["frequency"], rf_phase=c["trap"].get("rf_phase", 0.0),
              gravity=c["trap"].get("gravity", 9.80665),
              duration=c["simulation"]["duration"])
    for p in c.get("particles", []):
        s.particles.append(Particle(p.get("label", f"p{len(s.particles)}"),
                                    np.array(p["position"], float), np.array(p["velocity"], float)))
    return s


def _rhs(tau, s, a_r, q_r, a_z, q_z, b, gt):
    c = 2.0 * math.cos(2.0 * tau)
    x, vx, y, vy, z, vz = s
    return [vx, -b * vx - (a_r - q_r * c) * x,
            vy, -b * vy - (a_r - q_r * c) * y,
            vz, -b * vz - (a_z - q_z * c) * z - gt]


def _ring_event(r0):
    def ev(tau, s, *args):
        return s[0] ** 2 + s[2] ** 2 - 2.0 * s[4] ** 2 - r0 ** 2
    ev.terminal = True
    ev.direction = 1.0
    return ev


def _endcap_event(r0):
    def ev(tau, s, *args):
        return 2.0 * s[4] ** 2 - s[0] ** 2 - s[2] ** 2 - r0 ** 2
    ev.terminal = True
    ev.direction = 1.0
    return ev


@dataclass
class Trajectory:
    label: str
    t: np.ndarray         # s, sample times
    position: np.ndarray  # [n, 3] m
    velocity: np.ndarray  # [n, 3] m/s
    escaped: bool
    escape_time: float    # s, nan if trapped
    electrode: str        # "ring", "endcap" or ""


def integrate(setup, particle, t_eval=None, rtol=1e-10, atol=1e-13, detect_escape=True):
    """Integrate one particle until setup.duration or an electrode hit.

    detect_escape=False ignores the electrodes (unbounded ideal field), for tests.
    """
    a_r, q_r = fr.radial_params(setup.a_z, setup.q_z)
    tau0 = float(setup.tau(0.0))
    tau1 = float(setup.tau(setup.duration))
    u0 = setup.to_tau_velocity(particle.velocity)
    p0 = particle.position
    s0 = [p0[0], u0[0], p0[1], u0[1], p0[2], u0[2]]
    tau_eval = None if t_eval is None else setup.tau(t_eval)
    # Keep the step well below the RF period pi so the drive is always resolved.
    sol = solve_ivp(_rhs, (tau0, tau1), s0, method="DOP853", t_eval=tau_eval,
                    args=(a_r, q_r, setup.a_z, setup.q_z, setup.b, setup.gravity_term),
                    events=[_ring_event(setup.r0), _endcap_event(setup.r0)] if detect_escape else None,
                    rtol=rtol, atol=atol, max_step=math.pi / 20.0)
    if sol.status < 0:
        raise RuntimeError(f"integration failed for {particle.label}: {sol.message}")
    escaped, t_esc, electrode = False, math.nan, ""
    tau_out, y = sol.t, sol.y
    if sol.status == 1:
        hits = [(e[0], name, ye[0]) for e, ye, name in zip(sol.t_events, sol.y_events, ("ring", "endcap"))
                if len(e)]
        tau_hit, electrode, y_hit = min(hits, key=lambda h: h[0])
        escaped, t_esc = True, float(setup.time(tau_hit))
        # Append the state on the electrode surface as the final sample.
        tau_out = np.append(tau_out, tau_hit)
        y = np.concatenate([y, y_hit[:, None]], axis=1)
    pos = np.stack([y[0], y[2], y[4]], axis=1)
    vel = setup.to_physical_velocity(np.stack([y[1], y[3], y[5]], axis=1))
    return Trajectory(particle.label, setup.time(tau_out), pos, vel, escaped, t_esc, electrode)


def describe(setup):
    c_ax, c_ra, c = fr.trap_classification(setup.a_z, setup.q_z, setup.b)
    names = fr.CLASS_NAMES
    return (f"m = {setup.mass:.4g} kg, Omega = {setup.omega:.4g} rad/s, z0 = {setup.z0 * 1e3:.4g} mm\n"
            f"a_z = {setup.a_z:.6g}, q_z = {setup.q_z:.6g}, b = {setup.b:.6g}, "
            f"4g/Omega^2 = {setup.gravity_term:.4g} m\n"
            f"Floquet: axial {names[c_ax]}, radial {names[c_ra]}, trap {names[c]}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("config", type=Path)
    p.add_argument("--output", type=Path, default=None, help="write trajectories to this .npz")
    p.add_argument("--samples", type=int, default=4000)
    args = p.parse_args(argv)

    setup = load_setup(args.config)
    print(describe(setup))
    t_eval = np.linspace(0.0, setup.duration, args.samples)
    data = {}
    for k, part in enumerate(setup.particles):
        tr = integrate(setup, part, t_eval)
        fate = (f"escaped at t = {tr.escape_time * 1e3:.3f} ms ({tr.electrode})" if tr.escaped
                else f"trapped until {setup.duration * 1e3:.1f} ms, "
                     f"final |r| = {np.linalg.norm(tr.position[-1]) * 1e6:.3g} um")
        print(f"{tr.label:>10s}: {fate}")
        data[f"p{k}_t"] = tr.t
        data[f"p{k}_position"] = tr.position
        data[f"p{k}_velocity"] = tr.velocity
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, config=args.config.read_text(), **data)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
