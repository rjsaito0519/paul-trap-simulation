// Trace the trap Floquet stability boundary in (a_z, q_z) for a list of b values.
//
// For each b slice independently (docs/simulation/stability.md):
//   1. Evaluate the combined margin g = max(g_axial, g_radial) on a coarse grid.
//   2. Refine every grid edge whose sign of g changes by bisection.
//   3. Join the refined points into polylines with marching squares; saddle
//      cells are resolved with the margin at the cell centre.
//   4. Label connected stable regions on the grid (consistent with step 3).
// g < 0 is stable; g >= 0 (including exactly on the boundary) is not.
//
// Output directory contents:
//   boundary.csv   one row per boundary point, ordered along each polyline
//   metadata.json  provenance and per-slice summary
#include "cli_util.hpp"
#include "floquet.hpp"

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
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct Options {
    double a_min = -3.0, a_max = 3.0;
    double q_min = 0.0, q_max = 30.0;
    std::size_t na = 241, nq = 241;
    double b_min = 0.0, b_max = 10.0;
    std::size_t nb = 11;
    int steps = 2000;
    double tol = 1e-9;  // bisection interval length in parameter units
    int threads = 0;
    std::string output;
};

void usage(const char* prog)
{
    std::cerr
        << "usage: " << prog << " --output DIR [options]\n"
        << "  --a-range MIN MAX   a_z range (default -3 3)\n"
        << "  --q-range MIN MAX   q_z range (default 0 30)\n"
        << "  --na N --nq N       coarse grid nodes (default 241 241)\n"
        << "  --b-range MIN MAX   b range (default 0 10)\n"
        << "  --nb N              number of b slices (default 11)\n"
        << "  --steps N           RK4 steps per period pi (default 2000)\n"
        << "  --tol T             bisection tolerance (default 1e-9)\n"
        << "  --threads N         OpenMP threads (default: OMP_NUM_THREADS or all)\n";
}

Options parse(int argc, char** argv)
{
    Options o;
    auto need = [&](int i, int n) {
        if (i + n >= argc) {
            throw std::invalid_argument(std::string("missing value for ") + argv[i]);
        }
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--a-range") { need(i, 2); o.a_min = std::stod(argv[++i]); o.a_max = std::stod(argv[++i]); }
        else if (arg == "--q-range") { need(i, 2); o.q_min = std::stod(argv[++i]); o.q_max = std::stod(argv[++i]); }
        else if (arg == "--b-range") { need(i, 2); o.b_min = std::stod(argv[++i]); o.b_max = std::stod(argv[++i]); }
        else if (arg == "--na") { need(i, 1); o.na = std::stoul(argv[++i]); }
        else if (arg == "--nq") { need(i, 1); o.nq = std::stoul(argv[++i]); }
        else if (arg == "--nb") { need(i, 1); o.nb = std::stoul(argv[++i]); }
        else if (arg == "--steps") { need(i, 1); o.steps = std::stoi(argv[++i]); }
        else if (arg == "--tol") { need(i, 1); o.tol = std::stod(argv[++i]); }
        else if (arg == "--threads") { need(i, 1); o.threads = std::stoi(argv[++i]); }
        else if (arg == "--output") { need(i, 1); o.output = argv[++i]; }
        else if (arg == "-h" || arg == "--help") { usage(argv[0]); std::exit(0); }
        else { throw std::invalid_argument("unknown option " + arg); }
    }
    if (o.output.empty()) { throw std::invalid_argument("--output is required"); }
    if (o.na < 2 || o.nq < 2 || o.nb == 0) { throw std::invalid_argument("grid too small"); }
    if (o.b_min < 0.0) { throw std::invalid_argument("b must be non-negative"); }
    if (!(o.tol > 0.0)) { throw std::invalid_argument("tol must be positive"); }
    return o;
}

// Union-find over grid nodes for stable-region labelling.
struct DisjointSet {
    std::vector<std::size_t> parent;
    explicit DisjointSet(std::size_t n) : parent(n) { std::iota(parent.begin(), parent.end(), 0); }
    std::size_t find(std::size_t x)
    {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }
    void unite(std::size_t x, std::size_t y)
    {
        x = find(x);
        y = find(y);
        if (x != y) {
            parent[std::max(x, y)] = std::min(x, y);
        }
    }
};

// A grid edge whose endpoints differ in stability. axis 'q': node (i, j) to
// (i, j+1) at fixed a_i; axis 'a': node (i, j) to (i+1, j) at fixed q_j.
struct Crossing {
    char axis = 'q';
    std::size_t i = 0, j = 0;
    double a_z = 0.0, q_z = 0.0;
    bool axial_limited = true;
    std::size_t stable_node = 0;  // the stable endpoint
};

struct Point {
    std::size_t polyline = 0, index = 0, crossing = 0;
    bool closed = false;
};

struct SliceSummary {
    double b = 0.0;
    std::size_t regions = 0, polylines = 0, points = 0, saddles = 0;
    std::uint64_t monodromy_evaluations = 0;
    double elapsed = 0.0;
};

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
    const auto b_values = paultrap::cli::linspace(o.b_min, o.b_max, o.nb);
    const std::size_t na = o.na, nq = o.nq, nn = na * nq;
    auto node = [nq](std::size_t i, std::size_t j) { return i * nq + j; };

    namespace fs = std::filesystem;
    const fs::path dir(o.output);
    fs::create_directories(dir);
    std::ofstream csv(dir / "boundary.csv");
    csv << "b,polyline,index,closed,a_z,q_z,limiting,region,edge_axis,edge_i,edge_j\n";
    csv << std::setprecision(17);

    std::vector<SliceSummary> summaries;
    const auto t_all = std::chrono::steady_clock::now();

    for (const double b : b_values) {
        const auto t0 = std::chrono::steady_clock::now();
        SliceSummary sum;
        sum.b = b;

        // 1. Coarse grid.
        std::vector<double> g(nn);
#pragma omp parallel for schedule(dynamic, 64)
        for (std::int64_t idx = 0; idx < static_cast<std::int64_t>(nn); ++idx) {
            const auto k = static_cast<std::size_t>(idx);
            g[k] = paultrap::trap_margin(integ, a_values[k / nq], q_values[k % nq], b).combined();
        }
        sum.monodromy_evaluations += 2 * nn;
        auto stable = [&g](std::size_t k) { return g[k] < 0.0; };

        // 2. Sign-changing edges, enumerated in a fixed order (q-edges, then a-edges).
        std::vector<Crossing> crossings;
        std::map<std::tuple<char, std::size_t, std::size_t>, std::size_t> crossing_id;
        auto add_edge = [&](char axis, std::size_t i, std::size_t j) {
            const std::size_t k0 = node(i, j);
            const std::size_t k1 = axis == 'q' ? node(i, j + 1) : node(i + 1, j);
            if (stable(k0) != stable(k1)) {
                crossing_id[{axis, i, j}] = crossings.size();
                Crossing c;
                c.axis = axis;
                c.i = i;
                c.j = j;
                c.stable_node = stable(k0) ? k0 : k1;
                crossings.push_back(c);
            }
        };
        for (std::size_t i = 0; i < na; ++i) {
            for (std::size_t j = 0; j + 1 < nq; ++j) { add_edge('q', i, j); }
        }
        for (std::size_t i = 0; i + 1 < na; ++i) {
            for (std::size_t j = 0; j < nq; ++j) { add_edge('a', i, j); }
        }

        std::vector<std::uint64_t> evals(crossings.size(), 0);
#pragma omp parallel for schedule(dynamic, 4)
        for (std::int64_t idx = 0; idx < static_cast<std::int64_t>(crossings.size()); ++idx) {
            auto& c = crossings[static_cast<std::size_t>(idx)];
            // Parameterise the edge from its stable end (t = 0) to its unstable end (t = 1).
            const double a0 = a_values[c.i], q0 = q_values[c.j];
            const double a1 = c.axis == 'a' ? a_values[c.i + 1] : a0;
            const double q1 = c.axis == 'q' ? q_values[c.j + 1] : q0;
            const bool start_stable = c.stable_node == node(c.i, c.j);
            const double as = start_stable ? a0 : a1, qs = start_stable ? q0 : q1;
            const double au = start_stable ? a1 : a0, qu = start_stable ? q1 : q0;
            const double len = std::hypot(au - as, qu - qs);
            double lo = 0.0, hi = 1.0;
            while ((hi - lo) * len > o.tol) {
                const double t = 0.5 * (lo + hi);
                const double gm = paultrap::trap_margin(integ, as + t * (au - as), qs + t * (qu - qs), b).combined();
                (gm < 0.0 ? lo : hi) = t;
                evals[static_cast<std::size_t>(idx)] += 2;
            }
            const double t = 0.5 * (lo + hi);
            c.a_z = as + t * (au - as);
            c.q_z = qs + t * (qu - qs);
            const auto m = paultrap::trap_margin(integ, c.a_z, c.q_z, b);
            c.axial_limited = m.axial >= m.radial;
            evals[static_cast<std::size_t>(idx)] += 2;
        }
        sum.monodromy_evaluations += std::accumulate(evals.begin(), evals.end(), std::uint64_t{0});

        // 3. Marching squares. Cell (i, j) has corners c0 (i, j), c1 (i, j+1),
        // c2 (i+1, j+1), c3 (i+1, j) and edges e0 q(i, j), e1 a(i, j+1),
        // e2 q(i+1, j), e3 a(i, j).
        DisjointSet regions(nn);
        for (std::size_t i = 0; i < na; ++i) {
            for (std::size_t j = 0; j + 1 < nq; ++j) {
                if (stable(node(i, j)) && stable(node(i, j + 1))) { regions.unite(node(i, j), node(i, j + 1)); }
            }
        }
        for (std::size_t i = 0; i + 1 < na; ++i) {
            for (std::size_t j = 0; j < nq; ++j) {
                if (stable(node(i, j)) && stable(node(i + 1, j))) { regions.unite(node(i, j), node(i + 1, j)); }
            }
        }

        std::vector<std::vector<std::size_t>> adj(crossings.size());
        auto link = [&adj](std::size_t x, std::size_t y) {
            adj[x].push_back(y);
            adj[y].push_back(x);
        };
        for (std::size_t i = 0; i + 1 < na; ++i) {
            for (std::size_t j = 0; j + 1 < nq; ++j) {
                const std::size_t c[4] = {node(i, j), node(i, j + 1), node(i + 1, j + 1), node(i + 1, j)};
                const std::tuple<char, std::size_t, std::size_t> e[4] = {
                    {'q', i, j}, {'a', i, j + 1}, {'q', i + 1, j}, {'a', i, j}};
                std::size_t id[4];
                int n_cross = 0;
                for (int k = 0; k < 4; ++k) {
                    const auto it = crossing_id.find(e[k]);
                    id[k] = it == crossing_id.end() ? SIZE_MAX : it->second;
                    n_cross += it != crossing_id.end();
                }
                if (n_cross == 2) {
                    std::size_t pair[2], n = 0;
                    for (int k = 0; k < 4; ++k) {
                        if (id[k] != SIZE_MAX) { pair[n++] = id[k]; }
                    }
                    link(pair[0], pair[1]);
                } else if (n_cross == 4) {
                    // Saddle: c0, c2 share one state and c1, c3 the other.
                    ++sum.saddles;
                    const double gc = paultrap::trap_margin(
                        integ, 0.5 * (a_values[i] + a_values[i + 1]),
                        0.5 * (q_values[j] + q_values[j + 1]), b).combined();
                    sum.monodromy_evaluations += 2;
                    const bool centre_stable = gc < 0.0;
                    if (centre_stable == stable(c[0])) {
                        // c0 and c2 connected through the centre: cut off c1 and c3.
                        link(id[0], id[1]);
                        link(id[2], id[3]);
                        if (centre_stable) { regions.unite(c[0], c[2]); }
                    } else {
                        link(id[3], id[0]);
                        link(id[1], id[2]);
                        if (centre_stable) { regions.unite(c[1], c[3]); }
                    }
                }
            }
        }

        // 4. Region labels, numbered by first stable node in grid order.
        std::map<std::size_t, std::size_t> region_label;
        for (std::size_t k = 0; k < nn; ++k) {
            if (stable(k)) {
                region_label.emplace(regions.find(k), region_label.size());
            }
        }
        sum.regions = region_label.size();

        // Walk polylines: open ones from their endpoints first, then closed loops.
        std::vector<Point> points;
        std::vector<bool> visited(crossings.size(), false);
        auto walk = [&](std::size_t start, bool closed) {
            std::size_t prev = SIZE_MAX, cur = start, index = 0;
            while (cur != SIZE_MAX && !visited[cur]) {
                visited[cur] = true;
                points.push_back({sum.polylines, index++, cur, closed});
                std::size_t next = SIZE_MAX;
                for (std::size_t nb : adj[cur]) {
                    if (nb != prev && !visited[nb]) { next = nb; break; }
                }
                prev = cur;
                cur = next;
            }
            ++sum.polylines;
        };
        for (std::size_t k = 0; k < crossings.size(); ++k) {
            if (!visited[k] && adj[k].size() <= 1) { walk(k, false); }
        }
        for (std::size_t k = 0; k < crossings.size(); ++k) {
            if (!visited[k]) { walk(k, true); }
        }
        sum.points = points.size();

        for (const auto& p : points) {
            const auto& c = crossings[p.crossing];
            csv << b << ',' << p.polyline << ',' << p.index << ',' << (p.closed ? 1 : 0) << ','
                << c.a_z << ',' << c.q_z << ',' << (c.axial_limited ? "axial" : "radial") << ','
                << region_label.at(regions.find(c.stable_node)) << ',' << c.axis << ',' << c.i << ','
                << c.j << '\n';
        }
        sum.elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        summaries.push_back(sum);
        std::cout << "b=" << b << "  regions=" << sum.regions << "  polylines=" << sum.polylines
                  << "  points=" << sum.points << "  saddles=" << sum.saddles
                  << "  elapsed=" << sum.elapsed << " s\n";
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all).count();

    std::ofstream meta(dir / "metadata.json");
    meta << std::setprecision(17)
         << "{\n"
         << "  \"program\": \"floquet_boundary\",\n"
         << "  \"version\": \"" << ANALYSIS_VERSION << "\",\n"
         << "  \"git\": {\"revision\": \"" << ANALYSIS_GIT_COMMIT
         << "\", \"dirty_at_configure\": " << ANALYSIS_GIT_DIRTY << "},\n"
         << "  \"compiler\": \"" << paultrap::cli::json_escape(__VERSION__) << "\",\n"
         << "  \"openmp\": " << _OPENMP << ",\n"
         << "  \"threads\": " << omp_get_max_threads() << ",\n"
         << "  \"command\": " << paultrap::cli::json_command(argc, argv) << ",\n"
         << "  \"solver\": \"fixed-step classical RK4\",\n"
         << "  \"steps_per_period\": " << o.steps << ",\n"
         << "  \"bisection_tol\": " << o.tol << ",\n"
         << "  \"margin\": \"b > 0: log(max|lambda|); b = 0: |tr M| - 2; stable if max(axial, radial) < 0\",\n"
         << "  \"a_range\": [" << o.a_min << ", " << o.a_max << "], \"na\": " << o.na << ",\n"
         << "  \"q_range\": [" << o.q_min << ", " << o.q_max << "], \"nq\": " << o.nq << ",\n"
         << "  \"b_range\": [" << o.b_min << ", " << o.b_max << "], \"nb\": " << o.nb << ",\n"
         << "  \"elapsed_seconds\": " << elapsed << ",\n"
         << "  \"slices\": [\n";
    for (std::size_t k = 0; k < summaries.size(); ++k) {
        const auto& s = summaries[k];
        meta << "    {\"b\": " << s.b << ", \"regions\": " << s.regions << ", \"polylines\": " << s.polylines
             << ", \"points\": " << s.points << ", \"saddles\": " << s.saddles
             << ", \"monodromy_evaluations\": " << s.monodromy_evaluations
             << ", \"elapsed_seconds\": " << s.elapsed << "}" << (k + 1 < summaries.size() ? "," : "") << "\n";
    }
    meta << "  ]\n}\n";

    std::cout << "total elapsed: " << elapsed << " s\nwrote " << dir.string() << "\n";
    return 0;
}
