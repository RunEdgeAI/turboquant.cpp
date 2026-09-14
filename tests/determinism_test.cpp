// Copyright (c) 2026 Edge AI
// SPDX-License-Identifier: MIT

// Pins the exact output for fixed seeds, so a standard library, compiler or
// platform that changes it fails here instead of silently producing codes that
// another build — same seed, same parameters — cannot read.
//
// CI runs this on both libstdc++ (Linux) and libc++ (macOS). If a change to the
// output is intentional, bump kAlgorithmVersion and regenerate the goldens:
//   bazel run //:determinism_test -- --print

#include <turboquant/turboquant.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <random>
#include <vector>

namespace {

using namespace turboquant;

std::uint64_t fnv1a(const void* data, std::size_t n,
                    std::uint64_t h = 1469598103934665603ull) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// Gaussian matrix bits: exact by construction, independent of Eigen's QR.
std::uint64_t gaussianHash(int dim) {
    std::mt19937 rng(42);
    const Mat s = make_gaussian_matrix(dim, dim, rng);
    return fnv1a(s.data(), static_cast<std::size_t>(s.size()) * sizeof(float));
}

// Codes (MSE indices + QJL signs) for 8 test vectors built from integers only.
std::uint64_t codesHash(int dim) {
    std::mt19937 rng(42);
    const QuantizerProd qp(dim, 3, rng);
    std::mt19937 vec_rng(7);
    std::uint64_t h = 1469598103934665603ull;
    for (int k = 0; k < 8; ++k) {
        Vec x(dim);
        for (int i = 0; i < dim; ++i)
            x(i) = static_cast<float>(static_cast<std::int64_t>(vec_rng()) - 2147483648LL) /
                   2147483648.0f;
        x.normalize();
        const QuantizedProd q = qp.quantize(x);
        h = fnv1a(q.mse_part.indices.data(), q.mse_part.indices.size(), h);
        h = fnv1a(q.qjl_signs.data(), q.qjl_signs.size(), h);
    }
    return h;
}

struct Golden {
    int dim;
    std::uint64_t gaussian;
    std::uint64_t codes;
};

// Generated with --print, algorithm version 2. Verified identical on macOS arm64
// (libc++, Apple clang 21) and Linux arm64 (libstdc++, g++ 11.4), each in
// fastbuild, -c opt, and -c opt with -std=c++20.
constexpr Golden kGoldens[] = {
    {64, 0x4a335dee61b821bcull, 0xcfda2dd49e33c8bbull},
    {256, 0x04e52ec727f12707ull, 0xae17ac1b26b87814ull},
};

}  // namespace

int main(int argc, char** argv) {
    const bool print = argc > 1 && std::strcmp(argv[1], "--print") == 0;
    if (print) {
        std::printf("// kAlgorithmVersion = %d\n", kAlgorithmVersion);
        for (const auto& g : kGoldens)
            std::printf("    {%d, 0x%016llxull, 0x%016llxull},\n", g.dim,
                        static_cast<unsigned long long>(gaussianHash(g.dim)),
                        static_cast<unsigned long long>(codesHash(g.dim)));
        return 0;
    }

    int failures = 0;
    for (const auto& g : kGoldens) {
        const std::uint64_t gh = gaussianHash(g.dim);
        const std::uint64_t ch = codesHash(g.dim);
        if (gh != g.gaussian) {
            std::printf("FAIL d=%d gaussian matrix: got 0x%016llx, want 0x%016llx\n", g.dim,
                        static_cast<unsigned long long>(gh),
                        static_cast<unsigned long long>(g.gaussian));
            ++failures;
        }
        if (ch != g.codes) {
            std::printf("FAIL d=%d codes: got 0x%016llx, want 0x%016llx\n", g.dim,
                        static_cast<unsigned long long>(ch),
                        static_cast<unsigned long long>(g.codes));
            ++failures;
        }
    }
    if (failures == 0) std::printf("PASS: %zu dims match goldens\n", std::size(kGoldens));
    return failures == 0 ? 0 : 1;
}
