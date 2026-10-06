// Scan the trap Floquet stability over a rectangular (a_z, q_z) grid at fixed b.
//
// Output directory contents (arrays indexed [i_a, i_q]):
//   a_z.npy, q_z.npy                 grid axes (float64)
//   axial.npy, radial.npy, combined.npy  classes 0 stable, 1 boundary, 2 unstable (int8)
//   det_error.npy                    max |det M - exp(-b pi)| over both directions
//   monodromy_axial.npy, monodromy_radial.npy  [i_a, i_q, 2, 2], with --save-matrices
//   metadata.json                    provenance
#include "floquet.hpp"
#include "cli_util.hpp"
#include "npy.hpp"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Options {
    double a_min = -0.4, a_max = 0.2;
    double q_min = 0.0, q_max = 1.2;
    std::size_t na = 61, nq = 61;
    double b = 0.0;
    int steps = 2000;
    double eps = paultrap::kDefaultEps;
    int threads = 0;  // 0: OpenMP default
    bool save_matrices = false;
    std::string output;
};

void usage(const char* prog)
{
    std::cerr
        << "usage: " << prog << " --output DIR [options]\n"
        << "  --a-range MIN MAX   a_z range (default -0.4 0.2)\n"
        << "  --q-range MIN MAX   q_z range (default 0 1.2)\n"
        << "  --na N --nq N       grid points (default 61 61)\n"
        << "  --b B               damping b >= 0 (default 0)\n"
        << "  --steps N           RK4 steps per period pi (default 2000)\n"
        << "  --eps E             classification tolerance (default 1e-8)\n"
        << "  --threads N         OpenMP threads (default: OMP_NUM_THREADS or all)\n"
        << "  --save-matrices     also write monodromy matrices\n";
}

Options parse(int argc, char** argv)
{
    Options o;
    auto need = [&](int& i, int n) {
        if (i + n >= argc) {
            throw std::invalid_argument(std::string("missing value for ") + argv[i]);
        }
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--a-range") { need(i, 2); o.a_min = std::stod(argv[++i]); o.a_max = std::stod(argv[++i]); }
        else if (arg == "--q-range") { need(i, 2); o.q_min = std::stod(argv[++i]); o.q_max = std::stod(argv[++i]); }
        else if (arg == "--na") { need(i, 1); o.na = std::stoul(argv[++i]); }
        else if (arg == "--nq") { need(i, 1); o.nq = std::stoul(argv[++i]); }
        else if (arg == "--b") { need(i, 1); o.b = std::stod(argv[++i]); }
        else if (arg == "--steps") { need(i, 1); o.steps = std::stoi(argv[++i]); }
        else if (arg == "--eps") { need(i, 1); o.eps = std::stod(argv[++i]); }
        else if (arg == "--threads") { need(i, 1); o.threads = std::stoi(argv[++i]); }
        else if (arg == "--save-matrices") { o.save_matrices = true; }
        else if (arg == "--output") { need(i, 1); o.output = argv[++i]; }
        else if (arg == "-h" || arg == "--help") { usage(argv[0]); std::exit(0); }
        else { throw std::invalid_argument("unknown option " + arg); }
    }
    if (o.output.empty()) { throw std::invalid_argument("--output is required"); }
    if (o.na == 0 || o.nq == 0) { throw std::invalid_argument("grid must be non-empty"); }
    if (o.b < 0.0) { throw std::invalid_argument("b must be non-negative"); }
    return o;
}

}  // namespace

int main(int argc, char** argv)
{
    Options o;
    try {
        o = parse(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        usage(argv[0]);
        return 2;
    }
    if (o.threads > 0) {
        omp_set_num_threads(o.threads);
    }

    const paultrap::MonodromyIntegrator integ(o.steps);
    const auto a_values = paultrap::cli::linspace(o.a_min, o.a_max, o.na);
    const auto q_values = paultrap::cli::linspace(o.q_min, o.q_max, o.nq);
    const std::size_t n = o.na * o.nq;
    const double ref_det = std::exp(-o.b * kPi);

    std::vector<std::int8_t> axial(n), radial(n), combined(n);
    std::vector<double> det_error(n);
    std::vector<double> mat_ax(o.save_matrices ? 4 * n : 0);
    std::vector<double> mat_ra(o.save_matrices ? 4 * n : 0);

    const auto t0 = std::chrono::steady_clock::now();
    // Each grid point is independent and written to its own slot, so the
    // result does not depend on the thread count or schedule.
    const auto n_signed = static_cast<std::int64_t>(n);
#pragma omp parallel for schedule(dynamic, 64)
    for (std::int64_t idx = 0; idx < n_signed; ++idx) {
        const auto k = static_cast<std::size_t>(idx);
        const double a_z = a_values[k / o.nq];
        const double q_z = q_values[k % o.nq];
        const auto [a_r, q_r] = paultrap::radial_params(a_z, q_z);
        const auto m_ax = integ.monodromy(a_z, q_z, o.b);
        const auto m_ra = integ.monodromy(a_r, q_r, o.b);
        const auto c_ax = paultrap::classify(m_ax, o.b, o.eps);
        const auto c_ra = paultrap::classify(m_ra, o.b, o.eps);
        axial[k] = static_cast<std::int8_t>(c_ax);
        radial[k] = static_cast<std::int8_t>(c_ra);
        combined[k] = std::max(axial[k], radial[k]);
        det_error[k] = std::max(std::abs(m_ax.det() - ref_det), std::abs(m_ra.det() - ref_det));
        if (o.save_matrices) {
            const double ax[4] = {m_ax.m00, m_ax.m01, m_ax.m10, m_ax.m11};
            const double ra[4] = {m_ra.m00, m_ra.m01, m_ra.m10, m_ra.m11};
            std::copy(ax, ax + 4, mat_ax.begin() + static_cast<std::ptrdiff_t>(4 * k));
            std::copy(ra, ra + 4, mat_ra.begin() + static_cast<std::ptrdiff_t>(4 * k));
        }
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double max_det_error = *std::max_element(det_error.begin(), det_error.end());

    namespace fs = std::filesystem;
    const fs::path dir(o.output);
    fs::create_directories(dir);
    using paultrap::npy::write;
    write((dir / "a_z.npy").string(), a_values, {o.na});
    write((dir / "q_z.npy").string(), q_values, {o.nq});
    write((dir / "axial.npy").string(), axial, {o.na, o.nq});
    write((dir / "radial.npy").string(), radial, {o.na, o.nq});
    write((dir / "combined.npy").string(), combined, {o.na, o.nq});
    write((dir / "det_error.npy").string(), det_error, {o.na, o.nq});
    if (o.save_matrices) {
        write((dir / "monodromy_axial.npy").string(), mat_ax, {o.na, o.nq, 2, 2});
        write((dir / "monodromy_radial.npy").string(), mat_ra, {o.na, o.nq, 2, 2});
    }

    std::ofstream meta(dir / "metadata.json");
    meta << std::setprecision(17)
         << "{\n"
         << "  \"program\": \"floquet_scan\",\n"
         << "  \"version\": \"" << ANALYSIS_VERSION << "\",\n"
         << "  \"git\": {\"revision\": \"" << ANALYSIS_GIT_COMMIT
         << "\", \"dirty_at_configure\": " << ANALYSIS_GIT_DIRTY << "},\n"
         << "  \"compiler\": \"" << paultrap::cli::json_escape(__VERSION__) << "\",\n"
         << "  \"openmp\": " << _OPENMP << ",\n"
         << "  \"threads\": " << omp_get_max_threads() << ",\n"
         << "  \"command\": " << paultrap::cli::json_command(argc, argv) << ",\n"
         << "  \"solver\": \"fixed-step classical RK4\",\n"
         << "  \"steps_per_period\": " << o.steps << ",\n"
         << "  \"eps\": " << o.eps << ",\n"
         << "  \"b\": " << o.b << ",\n"
         << "  \"a_range\": [" << o.a_min << ", " << o.a_max << "], \"na\": " << o.na << ",\n"
         << "  \"q_range\": [" << o.q_min << ", " << o.q_max << "], \"nq\": " << o.nq << ",\n"
         << "  \"classes\": {\"0\": \"stable\", \"1\": \"boundary\", \"2\": \"unstable\"},\n"
         << "  \"max_det_error\": " << max_det_error << ",\n"
         << "  \"elapsed_seconds\": " << elapsed << "\n"
         << "}\n";

    std::cout << "points: " << n << "  threads: " << omp_get_max_threads()
              << "  elapsed: " << elapsed << " s\n"
              << "max |det M - exp(-b pi)| = " << max_det_error << "\n"
              << "wrote " << dir.string() << "\n";
    return 0;
}
