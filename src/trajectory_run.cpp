// Integrate one particle with the C++ trajectory engine and write the samples.
// Used to cross-check against scripts/trajectory.py; physical inputs are
// converted there, and only the dimensionless coefficients are passed here.
//
// Output directory: t.npy [s], position.npy [n, 3] m, velocity.npy [n, 3] m/s,
// result.json.
#include "cli_util.hpp"
#include "npy.hpp"
#include "trajectory.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    paultrap::TrapParams p;
    double omega = 0.0, rf_phase = 0.0, duration = 0.0;
    double pos[3] = {0, 0, 0}, vel[3] = {0, 0, 0};
    int steps = 200, record_every = 1;
    std::string output;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() {
                if (i + 1 >= argc) { throw std::invalid_argument("missing value for " + a); }
                return std::stod(argv[++i]);
            };
            if (a == "--a-z") { p.a_z = next(); }
            else if (a == "--q-z") { p.q_z = next(); }
            else if (a == "--b") { p.b = next(); }
            else if (a == "--gravity-term") { p.gravity_term = next(); }
            else if (a == "--r0") { p.r0 = next(); }
            else if (a == "--omega") { omega = next(); }
            else if (a == "--rf-phase") { rf_phase = next(); }
            else if (a == "--duration") { duration = next(); }
            else if (a == "--position") { for (double& x : pos) { x = next(); } }
            else if (a == "--velocity") { for (double& x : vel) { x = next(); } }
            else if (a == "--steps-per-rf") { steps = static_cast<int>(next()); }
            else if (a == "--record-every") { record_every = static_cast<int>(next()); }
            else if (a == "--output") { if (i + 1 >= argc) { throw std::invalid_argument("missing --output"); } output = argv[++i]; }
            else { throw std::invalid_argument("unknown option " + a); }
        }
        if (output.empty() || !(omega > 0.0) || !(p.r0 > 0.0)) {
            throw std::invalid_argument("--output, --omega and --r0 are required");
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }

    // tau = (Omega t + rf_phase) / 2 and u = d/dtau = 2 v / Omega (model.md).
    const double tau0 = rf_phase / 2.0;
    const double tau1 = tau0 + omega * duration / 2.0;
    paultrap::State s0{pos[0], 2.0 * vel[0] / omega, pos[1], 2.0 * vel[1] / omega,
                       pos[2], 2.0 * vel[2] / omega};
    const paultrap::TrajectoryIntegrator integ(steps);
    std::vector<paultrap::TrajectorySample> rec;
    const auto res = integ.run(p, s0, tau0, tau1, record_every, &rec);

    std::vector<double> t, x, v;
    for (const auto& r : rec) {
        t.push_back((2.0 * r.tau - rf_phase) / omega);
        for (int k = 0; k < 3; ++k) {
            x.push_back(r.state[static_cast<std::size_t>(2 * k)]);
            v.push_back(omega * r.state[static_cast<std::size_t>(2 * k + 1)] / 2.0);
        }
    }
    namespace fs = std::filesystem;
    const fs::path dir(output);
    fs::create_directories(dir);
    paultrap::npy::write((dir / "t.npy").string(), t, {t.size()});
    paultrap::npy::write((dir / "position.npy").string(), x, {t.size(), 3});
    paultrap::npy::write((dir / "velocity.npy").string(), v, {t.size(), 3});
    std::ofstream js(dir / "result.json");
    js << std::setprecision(17) << "{\"escaped\": " << (res.escaped ? "true" : "false")
       << ", \"electrode\": \"" << paultrap::to_string(res.electrode) << "\""
       << ", \"t_end\": " << (2.0 * res.tau_end - rf_phase) / omega
       << ", \"max_r\": " << res.max_r << ", \"max_abs_z\": " << res.max_abs_z
       << ", \"steps_per_rf\": " << steps
       << ", \"command\": " << paultrap::cli::json_command(argc, argv) << "}\n";
    std::cout << (res.escaped ? "escaped (" + std::string(paultrap::to_string(res.electrode)) + ")" : "trapped")
              << " t_end = " << (2.0 * res.tau_end - rf_phase) / omega << " s\n";
    return 0;
}
