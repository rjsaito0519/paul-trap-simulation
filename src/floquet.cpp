#include "floquet.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace paultrap {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

const char* to_string(Stability s)
{
    switch (s) {
    case Stability::Stable: return "stable";
    case Stability::Boundary: return "boundary";
    case Stability::Unstable: return "unstable";
    }
    return "unknown";
}

MonodromyIntegrator::MonodromyIntegrator(int steps_per_period)
    : steps_(steps_per_period), h_(0.0)
{
    if (steps_per_period <= 0) {
        throw std::invalid_argument("steps_per_period must be positive");
    }
    h_ = kPi / steps_;
    const auto n = static_cast<std::size_t>(2 * steps_ + 1);
    cos_half_.resize(n);
    for (std::size_t k = 0; k < n; ++k) {
        cos_half_[k] = std::cos(2.0 * (static_cast<double>(k) * h_ / 2.0));
    }
}

Matrix2 MonodromyIntegrator::monodromy(double a, double q, double b) const
{
    // Two fundamental solutions [x1, v1] and [x2, v2] advanced together.
    double x1 = 1.0, v1 = 0.0, x2 = 0.0, v2 = 1.0;
    const double h = h_;
    const double hh = 0.5 * h;
    auto acc = [b](double k, double x, double v) { return -b * v - k * x; };

    for (std::size_t i = 0; i < static_cast<std::size_t>(steps_); ++i) {
        const double k0 = a - 2.0 * q * cos_half_[2 * i];
        const double km = a - 2.0 * q * cos_half_[2 * i + 1];
        const double k1 = a - 2.0 * q * cos_half_[2 * i + 2];

        // Solution 1
        const double ax1 = v1, av1 = acc(k0, x1, v1);
        const double bx1 = v1 + hh * av1, bv1 = acc(km, x1 + hh * ax1, v1 + hh * av1);
        const double cx1 = v1 + hh * bv1, cv1 = acc(km, x1 + hh * bx1, v1 + hh * bv1);
        const double dx1 = v1 + h * cv1, dv1 = acc(k1, x1 + h * cx1, v1 + h * cv1);
        x1 += h / 6.0 * (ax1 + 2.0 * bx1 + 2.0 * cx1 + dx1);
        v1 += h / 6.0 * (av1 + 2.0 * bv1 + 2.0 * cv1 + dv1);

        // Solution 2
        const double ax2 = v2, av2 = acc(k0, x2, v2);
        const double bx2 = v2 + hh * av2, bv2 = acc(km, x2 + hh * ax2, v2 + hh * av2);
        const double cx2 = v2 + hh * bv2, cv2 = acc(km, x2 + hh * bx2, v2 + hh * bv2);
        const double dx2 = v2 + h * cv2, dv2 = acc(k1, x2 + h * cx2, v2 + h * cv2);
        x2 += h / 6.0 * (ax2 + 2.0 * bx2 + 2.0 * cx2 + dx2);
        v2 += h / 6.0 * (av2 + 2.0 * bv2 + 2.0 * cv2 + dv2);
    }
    return Matrix2{x1, x2, v1, v2};
}

std::array<std::complex<double>, 2> floquet_multipliers(const Matrix2& m)
{
    const double tr = m.trace();
    const std::complex<double> disc = std::sqrt(std::complex<double>(tr * tr - 4.0 * m.det()));
    return {(tr + disc) / 2.0, (tr - disc) / 2.0};
}

double stability_margin(const Matrix2& m, double b)
{
    if (b > 0.0) {
        const auto lam = floquet_multipliers(m);
        return std::log(std::max(std::abs(lam[0]), std::abs(lam[1])));
    }
    if (b == 0.0) {
        return std::abs(m.trace()) - 2.0;
    }
    throw std::invalid_argument("b must be non-negative");
}

TrapMargin trap_margin(const MonodromyIntegrator& integ, double a_z, double q_z, double b)
{
    const auto [a_r, q_r] = radial_params(a_z, q_z);
    return {stability_margin(integ.monodromy(a_z, q_z, b), b),
            stability_margin(integ.monodromy(a_r, q_r, b), b)};
}

Stability classify(const Matrix2& m, double b, double eps)
{
    double x = 0.0;
    double lim = 0.0;
    if (b > 0.0) {
        const auto lam = floquet_multipliers(m);
        x = std::max(std::abs(lam[0]), std::abs(lam[1]));
        lim = 1.0;
    } else if (b == 0.0) {
        x = std::abs(m.trace());
        lim = 2.0;
    } else {
        throw std::invalid_argument("b must be non-negative");
    }
    if (x < lim - eps) {
        return Stability::Stable;
    }
    if (x > lim + eps) {
        return Stability::Unstable;
    }
    return Stability::Boundary;
}

}  // namespace paultrap
