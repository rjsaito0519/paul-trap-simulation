// 3D single-particle trajectory in an ideal ring-endcap Paul trap (production engine).
//
// Dimensionless time tau = (Omega t + rf_phase) / 2, positions in metres and
// velocities u = d/dtau in metres (docs/simulation/model.md):
//
//   x'' + b x' + (a_r - 2 q_r cos 2 tau) x = 0          (same for y)
//   z'' + b z' + (a_z - 2 q_z cos 2 tau) z + G = 0,     G = 4 g / Omega^2
//
// Escape when the particle reaches an ideal hyperbolic electrode
// (docs/simulation/stability.md):
//   ring:    x^2 + y^2 - 2 z^2 = r0^2
//   endcaps: 2 z^2 - (x^2 + y^2) = r0^2
//
// The conversion from voltages and particle properties to (a_z, q_z, b, G)
// lives in scripts/trajectory.py; this engine takes the dimensionless values.
// scripts/trajectory.py is also the independent Python cross-check.
#pragma once

#include <array>
#include <vector>

namespace paultrap {

struct TrapParams {
    double a_z = 0.0;
    double q_z = 0.0;
    double b = 0.0;
    double gravity_term = 0.0;  // G = 4 g / Omega^2 [m]
    double r0 = 0.0;            // [m]
};

// [x, u_x, y, u_y, z, u_z] with u = d/dtau.
using State = std::array<double, 6>;

enum class Electrode : int { None = 0, Ring = 1, Endcap = 2 };

const char* to_string(Electrode e);

struct TrajectoryResult {
    bool escaped = false;
    Electrode electrode = Electrode::None;
    double tau_end = 0.0;  // escape time, or the end of integration
    State final_state{};   // on the electrode surface if escaped
    double max_r = 0.0;    // max sqrt(x^2 + y^2) over step boundaries [m]
    double max_abs_z = 0.0;
};

struct TrajectorySample {
    double tau;
    State state;
};

// Fixed-step classical RK4 with steps_per_rf steps per RF period (pi in tau).
// After each step the electrode functions are checked; on a sign change the
// crossing inside the step is located by bisection on the RK4 step length.
// A crossing that enters and leaves within one step is not detected, so the
// step must be small compared with the motion near the electrodes.
class TrajectoryIntegrator {
public:
    explicit TrajectoryIntegrator(int steps_per_rf, double bisection_tol = 1e-12);

    int steps_per_rf() const { return steps_; }

    // Integrate from tau0 to tau1. If record_every > 0, every record_every-th
    // step boundary (and the escape point) is appended to *record.
    TrajectoryResult run(const TrapParams& p, const State& s0, double tau0, double tau1,
                         int record_every = 0,
                         std::vector<TrajectorySample>* record = nullptr) const;

private:
    int steps_;
    double h_;
    double tol_;
    std::vector<double> cos_half_;  // cos(2 s) at s = k h / 2, k = 0 .. 2 steps - 1
    std::vector<double> sin_half_;  // sin(2 s), same grid
};

}  // namespace paultrap
