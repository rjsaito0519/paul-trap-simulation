// Monte Carlo geometric-capture probability over a (V_DC, V_AC) grid (Phase 5b).
//
// Capture = no electrode hit within the observation time (docs/simulation/stability.md).
// Physical inputs are converted to dimensionless coefficients by the Python
// driver (scripts/capture_map.py, via scripts/trajectory.py); here
//   a_z = a_per_vdc * V_DC,  q_z = q_per_vac * V_AC.
//
// Sample k uses the random stream (seed, k) at every grid point, so all points
// share the same initial conditions (common random numbers). Draw order per
// sample: RF phase, position (direction 2 draws, radius 1 draw), velocity (3 normals).
//
// Output directory (grid arrays indexed [i_dc, i_ac]):
//   v_dc.npy, v_ac.npy, a_z.npy, q_z.npy        axes
//   n_trapped.npy, n_ring.npy, n_endcap.npy     counts (int64)
//   p_capture.npy, wilson_lo.npy, wilson_hi.npy 95% Wilson interval
//   mean_escape_time.npy                        over escaped samples [s] (nan if none)
//   p_capture_at.npy [n_checkpoint, i_dc, i_ac] capture fraction at checkpoint times
//   checkpoint_time.npy                         [s]
//   floquet_class.npy                           0 stable, 1 boundary, 2 unstable (int8)
//   computed.npy                                1 where this run computed the point (int8)
// With --chunk K --n-chunks M only grid points [K n / M, (K + 1) n / M) in
// row-major order are computed (for batch jobs); the others are filled with -1
// (counts, classes) or NaN, and chunks are merged by scripts/capture_map.py.
//   with --save-samples: sample_rf_phase.npy, sample_position.npy [N, 3] m,
//     sample_velocity.npy [N, 3] m/s, sample_electrode.npy [i_dc, i_ac, N] (0 none,
//     1 ring, 2 endcap), sample_t_end.npy [i_dc, i_ac, N] s
//   metadata.json
#include "cli_util.hpp"
#include "floquet.hpp"
#include "npy.hpp"
#include "sampling.hpp"
#include "trajectory.hpp"

#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
const double kCheckpoints[] = {0.125, 0.25, 0.5, 1.0};

struct Options {
    double vdc_min = 0.0, vdc_max = 0.0, vac_min = 0.0, vac_max = 0.0;
    std::size_t n_vdc = 1, n_vac = 1;
    double a_per_vdc = 0.0, q_per_vac = 0.0, b = 0.0, gravity_term = 0.0, r0 = 0.0, omega = 0.0;
    double rf_periods = 200.0;
    std::size_t samples = 1000;
    std::uint64_t seed = 1;
    int steps_per_rf = 200;
    int threads = 0;
    double pos_center[3] = {0.0, 0.0, 0.0};
    double pos_radius = 0.0;
    double vel_mean[3] = {0.0, 0.0, 0.0};
    double vel_sigma = 0.0;
    std::string rf_phase = "uniform:0:6.283185307179586";
    bool save_samples = false;
    std::size_t chunk = 0, n_chunks = 1;
    std::string output;
};

Options parse(int argc, char** argv)
{
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto str = [&]() -> std::string {
            if (i + 1 >= argc) { throw std::invalid_argument("missing value for " + a); }
            return argv[++i];
        };
        auto num = [&]() { return std::stod(str()); };
        if (a == "--vdc-range") { o.vdc_min = num(); o.vdc_max = num(); }
        else if (a == "--vac-range") { o.vac_min = num(); o.vac_max = num(); }
        else if (a == "--n-vdc") { o.n_vdc = std::stoul(str()); }
        else if (a == "--n-vac") { o.n_vac = std::stoul(str()); }
        else if (a == "--a-per-vdc") { o.a_per_vdc = num(); }
        else if (a == "--q-per-vac") { o.q_per_vac = num(); }
        else if (a == "--b") { o.b = num(); }
        else if (a == "--gravity-term") { o.gravity_term = num(); }
        else if (a == "--r0") { o.r0 = num(); }
        else if (a == "--omega") { o.omega = num(); }
        else if (a == "--rf-periods") { o.rf_periods = num(); }
        else if (a == "--samples") { o.samples = std::stoul(str()); }
        else if (a == "--seed") { o.seed = std::stoull(str()); }
        else if (a == "--steps-per-rf") { o.steps_per_rf = std::stoi(str()); }
        else if (a == "--threads") { o.threads = std::stoi(str()); }
        else if (a == "--pos-center") { for (double& x : o.pos_center) { x = num(); } }
        else if (a == "--pos-radius") { o.pos_radius = num(); }
        else if (a == "--vel-mean") { for (double& x : o.vel_mean) { x = num(); } }
        else if (a == "--vel-sigma") { o.vel_sigma = num(); }
        else if (a == "--rf-phase") { o.rf_phase = str(); }
        else if (a == "--save-samples") { o.save_samples = true; }
        else if (a == "--chunk") { o.chunk = std::stoul(str()); }
        else if (a == "--n-chunks") { o.n_chunks = std::stoul(str()); }
        else if (a == "--output") { o.output = str(); }
        else { throw std::invalid_argument("unknown option " + a); }
    }
    if (o.output.empty()) { throw std::invalid_argument("--output is required"); }
    if (!(o.omega > 0.0) || !(o.r0 > 0.0)) { throw std::invalid_argument("--omega and --r0 must be positive"); }
    if (o.b < 0.0) { throw std::invalid_argument("b must be non-negative"); }
    if (o.n_vdc == 0 || o.n_vac == 0 || o.samples == 0) { throw std::invalid_argument("empty grid or no samples"); }
    if (o.pos_radius < 0.0 || o.vel_sigma < 0.0) { throw std::invalid_argument("negative spread"); }
    if (o.n_chunks == 0 || o.chunk >= o.n_chunks) { throw std::invalid_argument("need 0 <= chunk < n-chunks"); }
    return o;
}

struct InitialCondition {
    double rf_phase;
    double position[3];  // m
    double velocity[3];  // m/s
};

InitialCondition draw(const Options& o, const paultrap::Distribution& phase, std::uint64_t k)
{
    auto rng = paultrap::make_stream(o.seed, k);
    InitialCondition ic{};
    ic.rf_phase = phase.sample(rng);
    // Uniform in a sphere: isotropic direction and radius R u^(1/3).
    const double cos_t = 2.0 * rng.uniform() - 1.0;
    const double phi = 2.0 * kPi * rng.uniform();
    const double rad = o.pos_radius * std::cbrt(rng.uniform());
    const double sin_t = std::sqrt(std::max(0.0, 1.0 - cos_t * cos_t));
    const double dir[3] = {sin_t * std::cos(phi), sin_t * std::sin(phi), cos_t};
    for (int i = 0; i < 3; ++i) {
        ic.position[i] = o.pos_center[i] + rad * dir[i];
    }
    for (int i = 0; i < 3; ++i) {
        ic.velocity[i] = o.vel_mean[i] + o.vel_sigma * rng.normal();
    }
    return ic;
}

}  // namespace

int main(int argc, char** argv)
{
    Options o;
    paultrap::Distribution phase;
    try {
        o = parse(argc, argv);
        phase = paultrap::Distribution::parse(o.rf_phase);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
    if (o.threads > 0) {
        omp_set_num_threads(o.threads);
    }

    const auto v_dc = paultrap::cli::linspace(o.vdc_min, o.vdc_max, o.n_vdc);
    const auto v_ac = paultrap::cli::linspace(o.vac_min, o.vac_max, o.n_vac);
    const std::size_t n_points = o.n_vdc * o.n_vac;
    const std::size_t N = o.samples;
    const std::size_t p_begin = o.chunk * n_points / o.n_chunks;
    const std::size_t p_end = (o.chunk + 1) * n_points / o.n_chunks;
    const double t_obs = o.rf_periods * 2.0 * kPi / o.omega;

    std::vector<InitialCondition> ics(N);
    for (std::size_t k = 0; k < N; ++k) {
        ics[k] = draw(o, phase, k);
    }

    const paultrap::TrajectoryIntegrator integ(o.steps_per_rf);
    // -1 / NaN mark samples of grid points outside this chunk.
    std::vector<std::int8_t> electrode(n_points * N, -1);
    std::vector<double> t_end(n_points * N, std::numeric_limits<double>::quiet_NaN());

    const auto t0 = std::chrono::steady_clock::now();
    const auto total = static_cast<std::int64_t>((p_end - p_begin) * N);
#pragma omp parallel for schedule(dynamic, 16)
    for (std::int64_t idx = 0; idx < total; ++idx) {
        const auto u = p_begin * N + static_cast<std::size_t>(idx);
        const std::size_t pt = u / N, k = u % N;
        paultrap::TrapParams p;
        p.a_z = o.a_per_vdc * v_dc[pt / o.n_vac];
        p.q_z = o.q_per_vac * v_ac[pt % o.n_vac];
        p.b = o.b;
        p.gravity_term = o.gravity_term;
        p.r0 = o.r0;
        const auto& ic = ics[k];
        // tau = (Omega t + phase) / 2, u = 2 v / Omega (model.md).
        const double tau0 = ic.rf_phase / 2.0;
        const double tau1 = tau0 + o.omega * t_obs / 2.0;
        const paultrap::State s0{ic.position[0], 2.0 * ic.velocity[0] / o.omega,
                                 ic.position[1], 2.0 * ic.velocity[1] / o.omega,
                                 ic.position[2], 2.0 * ic.velocity[2] / o.omega};
        const auto r = integ.run(p, s0, tau0, tau1);
        electrode[u] = static_cast<std::int8_t>(r.electrode);
        t_end[u] = (2.0 * r.tau_end - ic.rf_phase) / o.omega;
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // Aggregate serially (deterministic).
    const std::size_t n_ck = sizeof(kCheckpoints) / sizeof(kCheckpoints[0]);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<std::int64_t> n_trapped(n_points, -1), n_ring(n_points, -1), n_endcap(n_points, -1);
    std::vector<double> p_capture(n_points, nan), lo(n_points, nan), hi(n_points, nan), mean_esc(n_points, nan);
    std::vector<double> p_at(n_ck * n_points, nan), ck_time(n_ck);
    std::vector<std::int8_t> fclass(n_points, -1), computed(n_points, 0);
    std::vector<double> a_axis(o.n_vdc), q_axis(o.n_vac);
    for (std::size_t c = 0; c < n_ck; ++c) {
        ck_time[c] = kCheckpoints[c] * t_obs;
    }
    for (std::size_t i = 0; i < o.n_vdc; ++i) { a_axis[i] = o.a_per_vdc * v_dc[i]; }
    for (std::size_t j = 0; j < o.n_vac; ++j) { q_axis[j] = o.q_per_vac * v_ac[j]; }
    const paultrap::MonodromyIntegrator floquet(2000);
    for (std::size_t pt = p_begin; pt < p_end; ++pt) {
        computed[pt] = 1;
        std::int64_t trapped = 0, ring = 0, endcap = 0;
        double sum_esc = 0.0;
        std::vector<std::int64_t> alive(n_ck, 0);
        for (std::size_t k = 0; k < N; ++k) {
            const std::size_t u = pt * N + k;
            const auto e = static_cast<paultrap::Electrode>(electrode[u]);
            if (e == paultrap::Electrode::None) { ++trapped; }
            else {
                (e == paultrap::Electrode::Ring ? ring : endcap) += 1;
                sum_esc += t_end[u];
            }
            for (std::size_t c = 0; c < n_ck; ++c) {
                if (e == paultrap::Electrode::None || t_end[u] > ck_time[c]) { ++alive[c]; }
            }
        }
        n_trapped[pt] = trapped;
        n_ring[pt] = ring;
        n_endcap[pt] = endcap;
        p_capture[pt] = static_cast<double>(trapped) / static_cast<double>(N);
        paultrap::wilson_interval(static_cast<std::uint64_t>(trapped), N, lo[pt], hi[pt]);
        mean_esc[pt] = ring + endcap > 0 ? sum_esc / static_cast<double>(ring + endcap)
                                         : std::numeric_limits<double>::quiet_NaN();
        for (std::size_t c = 0; c < n_ck; ++c) {
            p_at[c * n_points + pt] = static_cast<double>(alive[c]) / static_cast<double>(N);
        }
        const double a_z = a_axis[pt / o.n_vac], q_z = q_axis[pt % o.n_vac];
        const auto [a_r, q_r] = paultrap::radial_params(a_z, q_z);
        const auto c_ax = paultrap::classify(floquet.monodromy(a_z, q_z, o.b), o.b);
        const auto c_ra = paultrap::classify(floquet.monodromy(a_r, q_r, o.b), o.b);
        fclass[pt] = std::max(static_cast<std::int8_t>(c_ax), static_cast<std::int8_t>(c_ra));
    }

    namespace fs = std::filesystem;
    const fs::path dir(o.output);
    fs::create_directories(dir);
    using paultrap::npy::write;
    const std::vector<std::size_t> grid{o.n_vdc, o.n_vac};
    write((dir / "v_dc.npy").string(), v_dc, {o.n_vdc});
    write((dir / "v_ac.npy").string(), v_ac, {o.n_vac});
    write((dir / "a_z.npy").string(), a_axis, {o.n_vdc});
    write((dir / "q_z.npy").string(), q_axis, {o.n_vac});
    write((dir / "n_trapped.npy").string(), n_trapped, grid);
    write((dir / "n_ring.npy").string(), n_ring, grid);
    write((dir / "n_endcap.npy").string(), n_endcap, grid);
    write((dir / "p_capture.npy").string(), p_capture, grid);
    write((dir / "wilson_lo.npy").string(), lo, grid);
    write((dir / "wilson_hi.npy").string(), hi, grid);
    write((dir / "mean_escape_time.npy").string(), mean_esc, grid);
    write((dir / "p_capture_at.npy").string(), p_at, {n_ck, o.n_vdc, o.n_vac});
    write((dir / "checkpoint_time.npy").string(), ck_time, {n_ck});
    write((dir / "floquet_class.npy").string(), fclass, grid);
    write((dir / "computed.npy").string(), computed, grid);
    if (o.save_samples) {
        std::vector<double> ph(N), pos(3 * N), vel(3 * N);
        for (std::size_t k = 0; k < N; ++k) {
            ph[k] = ics[k].rf_phase;
            for (std::size_t i = 0; i < 3; ++i) {
                pos[3 * k + i] = ics[k].position[i];
                vel[3 * k + i] = ics[k].velocity[i];
            }
        }
        write((dir / "sample_rf_phase.npy").string(), ph, {N});
        write((dir / "sample_position.npy").string(), pos, {N, 3});
        write((dir / "sample_velocity.npy").string(), vel, {N, 3});
        write((dir / "sample_electrode.npy").string(), electrode, {o.n_vdc, o.n_vac, N});
        write((dir / "sample_t_end.npy").string(), t_end, {o.n_vdc, o.n_vac, N});
    }

    std::ofstream meta(dir / "metadata.json");
    meta << std::setprecision(17)
         << "{\n"
         << "  \"program\": \"capture_scan\",\n"
         << "  \"version\": \"" << ANALYSIS_VERSION << "\",\n"
         << "  \"git\": {\"revision\": \"" << ANALYSIS_GIT_COMMIT
         << "\", \"dirty_at_configure\": " << ANALYSIS_GIT_DIRTY << "},\n"
         << "  \"compiler\": \"" << paultrap::cli::json_escape(__VERSION__) << "\",\n"
         << "  \"openmp\": " << _OPENMP << ",\n"
         << "  \"threads\": " << omp_get_max_threads() << ",\n"
         << "  \"command\": " << paultrap::cli::json_command(argc, argv) << ",\n"
         << "  \"solver\": \"fixed-step classical RK4 with electrode bisection\",\n"
         << "  \"steps_per_rf\": " << o.steps_per_rf << ",\n"
         << "  \"observation_time\": " << t_obs << ", \"rf_periods\": " << o.rf_periods << ",\n"
         << "  \"samples\": " << N << ", \"seed\": " << o.seed << ",\n"
         << "  \"sampling\": {\"rf_phase\": \"" << o.rf_phase << "\", \"pos_center\": [" << o.pos_center[0]
         << ", " << o.pos_center[1] << ", " << o.pos_center[2] << "], \"pos_radius\": " << o.pos_radius
         << ", \"vel_mean\": [" << o.vel_mean[0] << ", " << o.vel_mean[1] << ", " << o.vel_mean[2]
         << "], \"vel_sigma\": " << o.vel_sigma << ", \"common_random_numbers\": true},\n"
         << "  \"coefficients\": {\"a_per_vdc\": " << o.a_per_vdc << ", \"q_per_vac\": " << o.q_per_vac
         << ", \"b\": " << o.b << ", \"gravity_term\": " << o.gravity_term << ", \"r0\": " << o.r0
         << ", \"omega\": " << o.omega << "},\n"
         << "  \"interval\": \"95% Wilson score\",\n"
         << "  \"chunk\": " << o.chunk << ", \"n_chunks\": " << o.n_chunks << ", \"point_range\": [" << p_begin
         << ", " << p_end << "],\n"
         << "  \"elapsed_seconds\": " << elapsed << "\n"
         << "}\n";
    std::cout << "points: " << (p_end - p_begin) << "/" << n_points << "  samples/point: " << N << "  threads: " << omp_get_max_threads()
              << "  elapsed: " << elapsed << " s\nwrote " << dir.string() << "\n";
    return 0;
}
