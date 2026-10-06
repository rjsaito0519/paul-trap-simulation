#include "trajectory.hpp"

#include "floquet.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace paultrap {

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Coefficients {
    double a_r, q_r, a_z, q_z, b, g;
};

// Derivative of the state for a given value of cos(2 tau).
inline State derivative(const Coefficients& k, double c, const State& s)
{
    const double kr = k.a_r - 2.0 * k.q_r * c;
    const double kz = k.a_z - 2.0 * k.q_z * c;
    return {s[1], -k.b * s[1] - kr * s[0],
            s[3], -k.b * s[3] - kr * s[2],
            s[5], -k.b * s[5] - kz * s[4] - k.g};
}

inline State axpy(const State& s, double h, const State& d)
{
    State r;
    for (std::size_t i = 0; i < 6; ++i) {
        r[i] = s[i] + h * d[i];
    }
    return r;
}

// One RK4 step of length h given cos(2 tau) at the start, middle and end.
inline State rk4(const Coefficients& k, const State& s, double h, double c0, double cm, double c1)
{
    const State d1 = derivative(k, c0, s);
    const State d2 = derivative(k, cm, axpy(s, 0.5 * h, d1));
    const State d3 = derivative(k, cm, axpy(s, 0.5 * h, d2));
    const State d4 = derivative(k, c1, axpy(s, h, d3));
    State r;
    for (std::size_t i = 0; i < 6; ++i) {
        r[i] = s[i] + h / 6.0 * (d1[i] + 2.0 * d2[i] + 2.0 * d3[i] + d4[i]);
    }
    return r;
}

inline double ring_function(const State& s, double r0)
{
    return s[0] * s[0] + s[2] * s[2] - 2.0 * s[4] * s[4] - r0 * r0;
}

inline double endcap_function(const State& s, double r0)
{
    return 2.0 * s[4] * s[4] - s[0] * s[0] - s[2] * s[2] - r0 * r0;
}

}  // namespace

const char* to_string(Electrode e)
{
    switch (e) {
    case Electrode::None: return "none";
    case Electrode::Ring: return "ring";
    case Electrode::Endcap: return "endcap";
    }
    return "unknown";
}

TrajectoryIntegrator::TrajectoryIntegrator(int steps_per_rf, double bisection_tol)
    : steps_(steps_per_rf), h_(0.0), tol_(bisection_tol)
{
    if (steps_per_rf <= 0) {
        throw std::invalid_argument("steps_per_rf must be positive");
    }
    h_ = kPi / steps_;
    const auto n = static_cast<std::size_t>(2 * steps_);
    cos_half_.resize(n);
    sin_half_.resize(n);
    for (std::size_t k = 0; k < n; ++k) {
        const double s = static_cast<double>(k) * h_ / 2.0;
        cos_half_[k] = std::cos(2.0 * s);
        sin_half_[k] = std::sin(2.0 * s);
    }
}

TrajectoryResult TrajectoryIntegrator::run(const TrapParams& p, const State& s0, double tau0,
                                           double tau1, int record_every,
                                           std::vector<TrajectorySample>* record) const
{
    if (!(tau1 >= tau0)) {
        throw std::invalid_argument("tau1 must not precede tau0");
    }
    const auto [a_r, q_r] = radial_params(p.a_z, p.q_z);
    const Coefficients k{a_r, q_r, p.a_z, p.q_z, p.b, p.gravity_term};

    // cos(2 (tau0 + s)) = cos(2 tau0) cos(2 s) - sin(2 tau0) sin(2 s); the
    // tables cover one period in s, so the drive is exactly periodic.
    const double c0 = std::cos(2.0 * tau0);
    const double s0c = std::sin(2.0 * tau0);
    const std::size_t period = cos_half_.size();
    auto drive = [&](std::size_t half_index) {
        const std::size_t m = half_index % period;
        return c0 * cos_half_[m] - s0c * sin_half_[m];
    };

    // Number of full steps, plus a final partial step to land on tau1.
    const double span = tau1 - tau0;
    const auto n_full = static_cast<std::size_t>(std::floor(span / h_));
    const double h_last = span - static_cast<double>(n_full) * h_;

    TrajectoryResult res;
    State s = s0;
    auto track = [&res](const State& st) {
        res.max_r = std::max(res.max_r, std::hypot(st[0], st[2]));
        res.max_abs_z = std::max(res.max_abs_z, std::abs(st[4]));
    };
    track(s);
    if (record && record_every > 0) {
        record->push_back({tau0, s});
    }

    // Locate an electrode crossing inside the step [tau_a, tau_a + h] from state sa.
    auto locate = [&](const State& sa, double tau_a, double h, Electrode e) {
        auto f = [&](const State& st) {
            return e == Electrode::Ring ? ring_function(st, p.r0) : endcap_function(st, p.r0);
        };
        auto partial = [&](double hh) {
            const double cs = std::cos(2.0 * tau_a);
            const double cmid = std::cos(2.0 * (tau_a + 0.5 * hh));
            const double ce = std::cos(2.0 * (tau_a + hh));
            return rk4(k, sa, hh, cs, cmid, ce);
        };
        double lo = 0.0, hi = h;
        while (hi - lo > tol_) {
            const double mid = 0.5 * (lo + hi);
            (f(partial(mid)) < 0.0 ? lo : hi) = mid;
        }
        res.escaped = true;
        res.electrode = e;
        res.tau_end = tau_a + hi;
        res.final_state = partial(hi);
    };

    for (std::size_t i = 0; i <= n_full; ++i) {
        const bool last = i == n_full;
        const double h = last ? h_last : h_;
        if (h <= 0.0) {
            break;
        }
        const double tau_a = tau0 + static_cast<double>(i) * h_;
        State next;
        if (!last) {
            next = rk4(k, s, h, drive(2 * i), drive(2 * i + 1), drive(2 * i + 2));
        } else {
            next = rk4(k, s, h, std::cos(2.0 * tau_a), std::cos(2.0 * (tau_a + 0.5 * h)),
                       std::cos(2.0 * (tau_a + h)));
        }
        if (ring_function(next, p.r0) >= 0.0) {
            locate(s, tau_a, h, Electrode::Ring);
        } else if (endcap_function(next, p.r0) >= 0.0) {
            locate(s, tau_a, h, Electrode::Endcap);
        }
        if (res.escaped) {
            track(res.final_state);
            if (record && record_every > 0) {
                record->push_back({res.tau_end, res.final_state});
            }
            return res;
        }
        s = next;
        track(s);
        if (record && record_every > 0 && ((i + 1) % static_cast<std::size_t>(record_every) == 0 || last)) {
            record->push_back({tau_a + h, s});
        }
    }
    if (record && record_every > 0 && record->back().tau < tau1) {
        record->push_back({tau1, s});
    }
    res.tau_end = tau1;
    res.final_state = s;
    return res;
}

}  // namespace paultrap
