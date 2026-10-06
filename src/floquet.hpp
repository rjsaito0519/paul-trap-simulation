// Floquet stability of the damped Mathieu equation (production engine).
//
//   x'' + b x' + (a - 2 q cos(2 tau)) x = 0
//
// Conventions follow docs/simulation/model.md and docs/simulation/stability.md.
// scripts/floquet_reference.py is the independent Python cross-check.
#pragma once

#include <array>
#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

namespace paultrap {

// Row-major 2x2 matrix acting on [x, dx/dtau].
struct Matrix2 {
    double m00 = 1.0;
    double m01 = 0.0;
    double m10 = 0.0;
    double m11 = 1.0;

    double trace() const { return m00 + m11; }
    double det() const { return m00 * m11 - m01 * m10; }
};

enum class Stability : std::int8_t { Stable = 0, Boundary = 1, Unstable = 2 };

const char* to_string(Stability s);

// Provisional classification tolerance (docs/simulation/stability.md, 2026-10-06).
constexpr double kDefaultEps = 1e-8;

// Fixed-step classical RK4 over one period pi. The cos(2 tau) values at all
// RK4 stages are tabulated once, so one integrator can be shared read-only
// between threads.
class MonodromyIntegrator {
public:
    explicit MonodromyIntegrator(int steps_per_period);

    int steps() const { return steps_; }
    Matrix2 monodromy(double a, double q, double b) const;

private:
    int steps_;
    double h_;
    std::vector<double> cos_half_;  // cos(2 tau) at tau = k h / 2, k = 0..2 steps
};

std::array<std::complex<double>, 2> floquet_multipliers(const Matrix2& m);

// b > 0: asymptotic stability, max|lambda| compared with 1.
// b = 0: bounded stability, |tr M| compared with 2.
Stability classify(const Matrix2& m, double b, double eps = kDefaultEps);

// Signed stability margin whose sign agrees with classify() for eps -> 0:
// negative inside the stable region, zero on the boundary.
// b > 0: log(max|lambda|).  b = 0: |tr M| - 2.
double stability_margin(const Matrix2& m, double b);

// Radial (a_r, q_r) for an ideal quadrupole potential proportional to r^2 - 2 z^2.
inline std::pair<double, double> radial_params(double a_z, double q_z)
{
    return {-a_z / 2.0, -q_z / 2.0};
}

struct TrapMargin {
    double axial = 0.0;
    double radial = 0.0;
    // The trap is stable only if both directions are stable.
    double combined() const { return axial > radial ? axial : radial; }
};

TrapMargin trap_margin(const MonodromyIntegrator& integ, double a_z, double q_z, double b);

}  // namespace paultrap
