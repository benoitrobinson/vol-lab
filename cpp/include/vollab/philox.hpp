// Philox4x64-10, matching numpy's Philox bit for bit.
//
// numpy emits the block for counter c+1 when constructed with counter=c, so a
// straight Random123 call with ctr=c is off by one block and yields a stream
// that is correct in distribution and wrong in parity. block(i) below returns
// numpy's i-th block, with the offset folded in.
#pragma once
#include <cstdint>
#include <array>

namespace vollab {

constexpr std::uint64_t PHILOX_M0 = 0xD2E7470EE14C6C93ULL;
constexpr std::uint64_t PHILOX_M1 = 0xCA5A826395121157ULL;
constexpr std::uint64_t PHILOX_W0 = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t PHILOX_W1 = 0xBB67AE8584CAA73BULL;

inline std::uint64_t mulhilo(std::uint64_t a, std::uint64_t b, std::uint64_t& lo) {
    __uint128_t p = static_cast<__uint128_t>(a) * b;
    lo = static_cast<std::uint64_t>(p);
    return static_cast<std::uint64_t>(p >> 64);
}

class Philox {
public:
    Philox(std::uint64_t k0, std::uint64_t k1) : k0_(k0), k1_(k1) {}

    // numpy's block index i (its counter=i-1 construction).
    std::array<std::uint64_t, 4> block(std::uint64_t i) const {
        std::uint64_t c[4] = {i, 0, 0, 0};
        std::uint64_t key0 = k0_, key1 = k1_;
        for (int r = 0; r < 10; ++r) {
            if (r > 0) { key0 += PHILOX_W0; key1 += PHILOX_W1; }
            std::uint64_t lo0, lo1;
            std::uint64_t hi0 = mulhilo(PHILOX_M0, c[0], lo0);
            std::uint64_t hi1 = mulhilo(PHILOX_M1, c[2], lo1);
            std::uint64_t n0 = hi1 ^ c[1] ^ key0;
            std::uint64_t n1 = lo1;
            std::uint64_t n2 = hi0 ^ c[3] ^ key1;
            std::uint64_t n3 = lo0;
            c[0] = n0; c[1] = n1; c[2] = n2; c[3] = n3;
        }
        return {c[0], c[1], c[2], c[3]};
    }

    // numpy's uniform: (u64 >> 11) * 2^-53, exact in double.
    static double to_uniform(std::uint64_t u) {
        return static_cast<double>(u >> 11) * (1.0 / 9007199254740992.0);
    }

    void uniforms(std::uint64_t n, double* out) const {
        std::uint64_t produced = 0, blk = 1;   // numpy's first block is index 1
        while (produced < n) {
            auto b = block(blk++);
            for (int j = 0; j < 4 && produced < n; ++j)
                out[produced++] = to_uniform(b[j]);
        }
    }

private:
    std::uint64_t k0_, k1_;
};

}  // namespace vollab
