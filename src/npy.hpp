// Minimal writer for NumPy .npy (format version 1.0, C order, little endian).
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace paultrap::npy {

template <typename T> const char* descr();
template <> inline const char* descr<double>() { return "<f8"; }
template <> inline const char* descr<std::int8_t>() { return "|i1"; }

template <typename T>
void write(const std::string& path, const std::vector<T>& data,
           const std::vector<std::size_t>& shape)
{
    const std::uint16_t probe = 1;
    unsigned char first = 0;
    std::memcpy(&first, &probe, 1);
    if (first != 1) {
        throw std::runtime_error("npy writer supports little-endian hosts only");
    }

    std::size_t count = 1;
    std::string shape_str = "(";
    for (std::size_t n : shape) {
        count *= n;
        shape_str += std::to_string(n) + ", ";
    }
    if (shape.size() > 1) {
        shape_str.erase(shape_str.size() - 1);  // "(2, 3, " -> "(2, 3,"
        shape_str.back() = ')';
    } else {
        shape_str.erase(shape_str.size() - 1);  // "(5, " -> "(5,"
        shape_str += ')';
    }
    if (count != data.size()) {
        throw std::invalid_argument("npy shape does not match data size: " + path);
    }

    std::string header = std::string("{'descr': '") + descr<T>()
        + "', 'fortran_order': False, 'shape': " + shape_str + ", }";
    // Magic (6) + version (2) + header length (2) + header must be a multiple of 64.
    const std::size_t total = 10 + header.size() + 1;
    header.append((64 - total % 64) % 64, ' ');
    header += '\n';

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot open " + path);
    }
    const auto len = static_cast<std::uint16_t>(header.size());
    out.write("\x93NUMPY\x01\x00", 8);
    out.put(static_cast<char>(len & 0xff));
    out.put(static_cast<char>(len >> 8));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(T)));
    if (!out) {
        throw std::runtime_error("failed writing " + path);
    }
}

}  // namespace paultrap::npy
