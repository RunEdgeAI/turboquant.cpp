// Copyright (c) 2026 Edge AI
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vector>
#include <span>
#include <Eigen/Dense>

namespace turboquant {

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