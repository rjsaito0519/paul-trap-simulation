// Small helpers shared by the command-line programs.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace paultrap::cli {

inline std::vector<double> linspace(double lo, double hi, std::size_t n)
{
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = n == 1 ? lo : lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(n - 1);
    }
    return v;
}

inline std::string json_escape(const std::string& s)
{
    std::string r;
    for (char c : s) {
        if (c == '"' || c == '\\') { r += '\\'; }
        r += c;
    }
    return r;
}

// JSON array of the command-line arguments.
inline std::string json_command(int argc, char** argv)
{
    std::string r = "[";
    for (int i = 0; i < argc; ++i) {
        r += (i ? ", \"" : "\"") + json_escape(argv[i]) + "\"";
    }
    return r + "]";
}

}  // namespace paultrap::cli
