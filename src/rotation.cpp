// Copyright (c) 2026 Edge AI
// SPDX-License-Identifier: MIT

#include "turboquant/rotation.h"

namespace turboquant {

Mat make_rotation_matrix(int dim, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, 1.0f);

    Mat G(dim, dim);
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            G(i, j) = normal(rng);

    Eigen::HouseholderQR<Mat> qr(G);
    Mat Q = qr.householderQ() * Mat::Identity(dim, dim);

    // Ensure det = +1 (proper rotation, not reflection)
    if (Q.determinant() < 0.0f)
        Q.col(0) *= -1.0f;

    return Q;
}

Mat make_gaussian_matrix(int rows, int cols, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, 1.0f);

    Mat S(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            S(i, j) = normal(rng);

    return S;
}

}  // namespace turboquant