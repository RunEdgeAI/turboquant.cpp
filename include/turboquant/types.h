// Copyright (c) 2026 Edge AI
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vector>
#include <span>
#include <Eigen/Dense>

namespace turboquant {

// Version of the numeric output: bumped whenever the rotation, projection or
// codes produced for a given (dim, bitwidth, seed) change. Anything that
// persists codes should record it, so codes from another version are refused.
//   1: rotation and projection drawn with std::normal_distribution, whose
//      output differs between standard libraries (libc++ vs libstdc++).
//   2: portable Box-Muller sampler; reflection sign from the QR coefficients.
//      Pinned by tests/determinism_test.cpp.
inline constexpr int kAlgorithmVersion = 2;

using Vec = Eigen::VectorXf;
using Mat = Eigen::MatrixXf;

struct QuantizedMSE {
    std::vector<uint8_t> indices;
    float norm;
    int dim;
    int bitwidth;
};

struct QuantizedProd {
    QuantizedMSE mse_part;
    std::vector<int8_t> qjl_signs;
    float residual_norm;
};

struct ScalarCodebook {
    int bitwidth;
    int num_centroids;
    std::vector<float> centroids;
    std::vector<float> boundaries;
};

}  // namespace turboquant