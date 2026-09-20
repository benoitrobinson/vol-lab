#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>
#include "vollab/philox.hpp"

namespace nb = nanobind;

static nb::ndarray<nb::numpy, double> uniforms(std::uint64_t seed,
                                               std::uint64_t path, std::uint64_t n) {
    double* data = new double[n];
    vollab::Philox(seed, path).uniforms(n, data);
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return nb::ndarray<nb::numpy, double>(data, {static_cast<size_t>(n)}, owner);
}

NB_MODULE(_probe, m) { m.def("uniforms", &uniforms); }
