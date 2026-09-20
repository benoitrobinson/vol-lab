#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <cmath>
#include <tuple>
#include <vector>
#include "vollab/hedge.hpp"
#include "vollab/philox.hpp"
#include "vollab/ndtri.hpp"

namespace nb = nanobind;

template <typename T>
static nb::ndarray<nb::numpy, T> own(T* data, size_t n) {
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return nb::ndarray<nb::numpy, T>(data, {n}, owner);
}

static nb::ndarray<nb::numpy, double> uniforms(std::uint64_t seed,
                                               std::uint64_t path, size_t n) {
    double* d = new double[n];
    vollab::Philox(seed, path).uniforms(n, d);
    return own(d, n);
}

static nb::ndarray<nb::numpy, double> normals(std::uint64_t seed,
                                              std::uint64_t path, size_t n) {
    double* d = new double[n];
    vollab::Philox(seed, path).uniforms(n, d);
    for (size_t i = 0; i < n; ++i) d[i] = vollab::ndtri(d[i]);
    return own(d, n);
}

static std::tuple<nb::ndarray<nb::numpy, double>, nb::ndarray<nb::numpy, std::int64_t>>
hedge_gbm(double S0, double K, double T, double r, double q,
          double s_imp, double s_hedge, double s_real,
          double mu, double cost_bps,
          std::uint64_t n_mon, std::uint64_t every,
          std::uint64_t seed, std::uint64_t path_start, size_t n_paths) {
    // mu is NaN for "use r - q"; nb::object would need an explicit .none()
    // annotation and buys nothing here.
    vollab::HedgeParams p{S0, K, T, r, q, s_imp, s_hedge, s_real, mu,
                          cost_bps * 1e-4, n_mon, every, seed};
    double* pnl = new double[n_paths];
    std::int64_t* nre = new std::int64_t[n_paths];
    std::vector<double> buf;
    for (size_t i = 0; i < n_paths; ++i)
        pnl[i] = vollab::hedge_one(p, path_start + i, nre[i], buf);
    return {own(pnl, n_paths), own(nre, n_paths)};
}

NB_MODULE(_core, m) {
    m.def("uniforms", &uniforms);
    m.def("normals", &normals);
    m.def("hedge_gbm", &hedge_gbm);
}
