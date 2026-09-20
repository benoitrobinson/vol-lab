// GBM path generation and the delta-hedging loop, scalar per path.
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>
#include "vollab/philox.hpp"
#include "vollab/ndtri.hpp"

namespace vollab {

inline double norm_cdf(double x) { return 0.5 * std::erfc(-x * M_SQRT1_2); }

inline double bs_call_delta(double S, double K, double tau, double r,
                            double q, double s) {
    if (tau <= 0.0) return S > K ? 1.0 : 0.0;
    double v = s * std::sqrt(tau);
    double d1 = (std::log(S / K) + (r - q + 0.5 * s * s) * tau) / v;
    return std::exp(-q * tau) * norm_cdf(d1);
}

inline double bs_call_price(double S, double K, double tau, double r,
                            double q, double s) {
    if (tau <= 0.0) return S > K ? S - K : 0.0;
    double v = s * std::sqrt(tau);
    double d1 = (std::log(S / K) + (r - q + 0.5 * s * s) * tau) / v;
    double d2 = d1 - v;
    return S * std::exp(-q * tau) * norm_cdf(d1) - K * std::exp(-r * tau) * norm_cdf(d2);
}

struct HedgeParams {
    double S0, K, T, r, q;
    double s_imp, s_hedge, s_real;
    double mu;             // NaN means r - q
    double k;              // proportional cost
    std::uint64_t n_mon, every, seed;
};

// One path: returns terminal P&L and writes the rehedge count.
inline double hedge_one(const HedgeParams& p, std::uint64_t path,
                        std::int64_t& n_reh, std::vector<double>& buf) {
    const double dt = p.T / static_cast<double>(p.n_mon);
    const double drift = std::isnan(p.mu) ? (p.r - p.q) : p.mu;
    const double sq = std::sqrt(dt);

    buf.resize(p.n_mon);
    Philox(p.seed, path).uniforms(p.n_mon, buf.data());

    double logS = std::log(p.S0);
    double S = p.S0;
    double cash = bs_call_price(p.S0, p.K, p.T, p.r, p.q, p.s_imp);
    double held = 0.0;
    n_reh = 0;

    // Step 0: open the hedge.
    {
        double want = bs_call_delta(S, p.K, p.T, p.r, p.q, p.s_hedge);
        double d = want - held;
        cash -= d * S + p.k * std::fabs(d) * S;
        if (d != 0.0) ++n_reh;
        held = want;
    }

    for (std::uint64_t i = 1; i <= p.n_mon; ++i) {
        double S_prev = S;
        logS += (drift - 0.5 * p.s_real * p.s_real) * dt
                + p.s_real * sq * ndtri(buf[i - 1]);
        S = std::exp(logS);

        cash = cash * std::exp(p.r * dt) + p.q * held * S_prev * dt;

        double target;
        if (i == p.n_mon) {
            target = 0.0;
        } else if (i % p.every == 0) {
            target = bs_call_delta(S, p.K, p.T - static_cast<double>(i) * dt,
                                   p.r, p.q, p.s_hedge);
        } else {
            target = held;
        }
        double d = target - held;
        cash -= d * S + p.k * std::fabs(d) * S;
        if (d != 0.0) ++n_reh;
        held = target;
    }

    double payoff = S > p.K ? S - p.K : 0.0;
    return cash - payoff;
}

}  // namespace vollab
