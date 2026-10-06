// Deterministic, portable random sampling for Monte Carlo capture studies.
//
// Every sample owns an independent SplitMix64 stream whose state is derived
// from (seed, stream index), so results do not depend on thread count or
// execution order. Uniform and normal variates are implemented here rather
// than with <random> distributions, whose outputs differ between standard
// libraries. (Box-Muller uses std::log/cos/sin, so last-bit differences
// between math libraries remain possible.)
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace paultrap {

class SplitMix64 {
public:
    explicit SplitMix64(std::uint64_t state) : state_(state) {}

    std::uint64_t next()
    {
        std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    // Uniform in [0, 1) with 53 random bits.
    double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

    // Standard normal by Box-Muller (one variate per call; the pair partner is discarded
    // so that each call consumes a fixed number of raw draws).
    double normal()
    {
        constexpr double kTwoPi = 6.283185307179586476925286766559;
        const double u1 = 1.0 - uniform();  // (0, 1]
        const double u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(kTwoPi * u2);
    }

private:
    std::uint64_t state_;
};

// Independent stream for (seed, index): the index is mixed through SplitMix64
// so that neighbouring indices give unrelated states.
inline SplitMix64 make_stream(std::uint64_t seed, std::uint64_t index)
{
    SplitMix64 mix(seed ^ (0xd1b54a32d192ed03ULL * (index + 1)));
    return SplitMix64(mix.next());
}

// One scalar distribution: "fixed:v", "uniform:min:max", "normal:mean:sigma" or
// "lognormal:median:sigma_ln" (median * exp(sigma_ln * N(0, 1))).
struct Distribution {
    enum class Kind { Fixed, Uniform, Normal, Lognormal } kind = Kind::Fixed;
    double p1 = 0.0;
    double p2 = 0.0;

    static Distribution parse(const std::string& spec)
    {
        auto field = [&spec](std::size_t k) {
            std::size_t start = 0;
            for (std::size_t i = 0; i < k; ++i) {
                start = spec.find(':', start);
                if (start == std::string::npos) {
                    throw std::invalid_argument("bad distribution: " + spec);
                }
                ++start;
            }
            return std::stod(spec.substr(start, spec.find(':', start) - start));
        };
        Distribution d;
        const std::string kind = spec.substr(0, spec.find(':'));
        if (kind == "fixed") { d.kind = Kind::Fixed; d.p1 = field(1); }
        else if (kind == "uniform") { d.kind = Kind::Uniform; d.p1 = field(1); d.p2 = field(2); }
        else if (kind == "normal") { d.kind = Kind::Normal; d.p1 = field(1); d.p2 = field(2); }
        else if (kind == "lognormal") { d.kind = Kind::Lognormal; d.p1 = field(1); d.p2 = field(2); }
        else { throw std::invalid_argument("unknown distribution: " + spec); }
        return d;
    }

    // Always consumes the same number of draws for a given kind.
    double sample(SplitMix64& rng) const
    {
        switch (kind) {
        case Kind::Fixed: return p1;
        case Kind::Uniform: return p1 + (p2 - p1) * rng.uniform();
        case Kind::Normal: return p1 + p2 * rng.normal();
        case Kind::Lognormal: return p1 * std::exp(p2 * rng.normal());
        }
        return p1;
    }
};

// 95% Wilson score interval for k successes in n trials.
inline void wilson_interval(std::uint64_t k, std::uint64_t n, double& lo, double& hi)
{
    if (n == 0) {
        lo = 0.0;
        hi = 1.0;
        return;
    }
    constexpr double z = 1.959963984540054;
    const double nn = static_cast<double>(n);
    const double p = static_cast<double>(k) / nn;
    const double denom = 1.0 + z * z / nn;
    const double centre = (p + z * z / (2.0 * nn)) / denom;
    const double half = z / denom * std::sqrt(p * (1.0 - p) / nn + z * z / (4.0 * nn * nn));
    lo = std::max(0.0, centre - half);
    hi = std::min(1.0, centre + half);
    // The bounds are exactly 0 and 1 at k = 0 and k = n; avoid rounding just inside.
    if (k == 0) { lo = 0.0; }
    if (k == n) { hi = 1.0; }
}

}  // namespace paultrap
