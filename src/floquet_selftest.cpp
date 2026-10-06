// Self-contained checks of the C++ Floquet engine (docs/simulation/validation.md).
#include "floquet.hpp"

#include <cmath>
#include <cstdio>
#include <string>

namespace {

constexpr double kPi = 3.14159265358979323846;
int failures = 0;

void check(bool ok, const std::string& what)
{
    std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what.c_str());
    if (!ok) {
        ++failures;
    }
}

// exp(A pi) for A = [[0, 1], [-a, -b]] (q = 0), with w^2 = a - b^2/4.
paultrap::Matrix2 constant_coefficient_exact(double a, double b)
{
    const double w2 = a - b * b / 4.0;
    double c = 1.0;
    double s_over_w = kPi;  // w -> 0 limit
    if (w2 > 0.0) {
        const double w = std::sqrt(w2);
        c = std::cos(w * kPi);
        s_over_w = std::sin(w * kPi) / w;
    } else if (w2 < 0.0) {
        const double w = std::sqrt(-w2);
        c = std::cosh(w * kPi);
        s_over_w = std::sinh(w * kPi) / w;
    }
    const double e = std::exp(-b * kPi / 2.0);
    // exp(A t) = e^{-bt/2} [c I + (s/w)(A + b/2 I)]
    return {e * (c + s_over_w * b / 2.0), e * s_over_w,
            e * (-a * s_over_w), e * (c - s_over_w * b / 2.0)};
}

double max_abs_diff(const paultrap::Matrix2& x, const paultrap::Matrix2& y)
{
    return std::fmax(std::fmax(std::abs(x.m00 - y.m00), std::abs(x.m01 - y.m01)),
                     std::fmax(std::abs(x.m10 - y.m10), std::abs(x.m11 - y.m11)));
}

}  // namespace

int main()
{
    using paultrap::Stability;
    const paultrap::MonodromyIntegrator integ(2000);

    const double cc[][2] = {{0.3, 0.0}, {0.3, 0.5}, {-0.2, 0.1}, {2.0, 1.5}, {0.0625, 0.5}};
    for (const auto& p : cc) {
        const double d = max_abs_diff(integ.monodromy(p[0], 0.0, p[1]),
                                      constant_coefficient_exact(p[0], p[1]));
        check(d < 1e-10, "q=0 exact, a=" + std::to_string(p[0]) + " b=" + std::to_string(p[1])
                             + " diff=" + std::to_string(d));
    }

    // For large b, det M = exp(-b pi) is far smaller than the matrix entries,
    // so its error is dominated by cancellation in m00 m11 - m01 m10. Normalise
    // by that cancellation scale rather than by exp(-b pi).
    const double ld[][3] = {{0.0, 0.5, 0.0}, {0.1, 0.7, 0.3}, {-0.3, 1.5, 2.0},
                            {0.0, 5.0, 6.0}, {0.0, 17.0, 6.0}, {0.0, 30.0, 10.0}};
    for (const auto& p : ld) {
        const auto m = integ.monodromy(p[0], p[1], p[2]);
        const double scale = std::abs(m.m00 * m.m11) + std::abs(m.m01 * m.m10);
        const double err = std::abs(m.det() - std::exp(-p[2] * kPi)) / scale;
        check(err < 1e-9, "Liouville det, q=" + std::to_string(p[1]) + " b=" + std::to_string(p[2])
                              + " normalised err=" + std::to_string(err * 1e12) + "e-12");
    }

    // First region upper edge at a = 0, b = 0: tr M = -2 at q = 0.908046.
    double lo = 0.8, hi = 1.0;
    while (hi - lo > 1e-10) {
        const double mid = 0.5 * (lo + hi);
        (integ.monodromy(0.0, mid, 0.0).trace() + 2.0 > 0.0 ? lo : hi) = mid;
    }
    const double q_edge = 0.5 * (lo + hi);
    check(std::abs(q_edge - 0.908046) < 2e-6, "q edge a=0 b=0: " + std::to_string(q_edge));

    check(paultrap::classify(integ.monodromy(0.0, 0.90, 0.0), 0.0) == Stability::Stable, "q=0.90 b=0 stable");
    check(paultrap::classify(integ.monodromy(0.0, 0.92, 0.0), 0.0) == Stability::Unstable, "q=0.92 b=0 unstable");
    check(paultrap::classify(integ.monodromy(0.0, 0.92, 0.5), 0.5) == Stability::Stable, "q=0.92 b=0.5 stable");
    check(paultrap::classify(integ.monodromy(0.0, 1.5, 0.1), 0.1) == Stability::Unstable, "q=1.5 b=0.1 unstable");

    const auto [a_r, q_r] = paultrap::radial_params(-0.2, 0.6);
    check(a_r == 0.1 && q_r == -0.3, "radial conversion");

    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
