// Copyright (c) 2026 Edge AI
// SPDX-License-Identifier: MIT

#include "turboquant/rotation.h"

#include <cmath>
#include <cstdint>
#include <numbers>

namespace turboquant {

namespace {

// N(0, 1) samples defined only in terms of mt19937's integer output.
//
// std::normal_distribution is not portable: the standard leaves its algorithm
// unspecified, and libc++ and libstdc++ return different sequences from the
// same engine, so a rotation built from one seed differed between macOS and
// Linux. This is Box-Muller in double over 53-bit uniforms. The integer
// handling is exact; log, sqrt, sin and cos were measured bit-identical on
// libc++/Apple clang and libstdc++/GCC (see tests/determinism_test.cpp).
class PortableNormal {
public:
    explicit PortableNormal(std::mt19937& rng) : rng_(rng) {}

    float operator()() {
        if (has_spare_) {
            has_spare_ = false;
            return static_cast<float>(spare_);
        }
        const double u1 = open01();
        const double u2 = open01();
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double theta = 2.0 * std::numbers::pi * u2;
        spare_ = r * std::sin(theta);
        has_spare_ = true;
        return static_cast<float>(r * std::cos(theta));
    }

private:
    // Uniform on the open interval (0, 1) from two 32-bit draws; never 0 or 1.
    double open01() {
        const std::uint64_t a = rng_() >> 5;  // 27 bits
        const std::uint64_t b = rng_() >> 6;  // 26 bits
        const double k = static_cast<double>((a << 26) | b);  // < 2^53, exact
        return (k + 0.5) / 9007199254740992.0;                // / 2^53, exact
    }

    std::mt19937& rng_;
    bool has_spare_ = false;
    double spare_ = 0.0;
};

// Q from a Householder QR is the product of one reflector per nonzero tau, so
// det(Q) = (-1)^(number of nonzero taus): exact, O(dim). Q.determinant() cost
// another O(dim^3) and, in float, collapsed to -0 from dim ~384 up, so the
// sign fix below silently never ran at practical dimensions.
bool is_reflection(const Eigen::HouseholderQR<Mat>& qr) {
    int reflectors = 0;
    for (Eigen::Index i = 0; i < qr.hCoeffs().size(); ++i)
        if (qr.hCoeffs()(i) != 0.0f)
            ++reflectors;
    return (reflectors % 2) == 1;
}

}  // namespace

Mat make_rotation_matrix(int dim, std::mt19937& rng) {
    PortableNormal normal(rng);

    Mat G(dim, dim);
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            G(i, j) = normal();

    Eigen::HouseholderQR<Mat> qr(G);
    Mat Q = qr.householderQ() * Mat::Identity(dim, dim);

    // Ensure det = +1 (proper rotation, not reflection)
    if (is_reflection(qr))
        Q.col(0) *= -1.0f;

    return Q;
}

Mat make_gaussian_matrix(int rows, int cols, std::mt19937& rng) {
    PortableNormal normal(rng);

    Mat S(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            S(i, j) = normal();

    return S;
}

}  // namespace turboquant