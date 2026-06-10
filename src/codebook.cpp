// Copyright (c) 2026 Edge AI
// SPDX-License-Identifier: MIT

#include "turboquant/codebook.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace turboquant {

// Lloyd-Max optimal centroids for N(0,1), solved via continuous 1D k-means (Eq. 4).
std::vector<float> normalized_centroids(int bitwidth) {
    switch (bitwidth) {
        case 1:
            return {-0.7979f, 0.7979f};
        case 2:
            return {-1.5104f, -0.4528f, 0.4528f, 1.5104f};
        case 3:
            return {-2.1520f, -1.3439f, -0.7560f, -0.2451f,
                     0.2451f,  0.7560f,  1.3439f,  2.1520f};
        case 4:
            return {-2.7326f, -2.0690f, -1.6180f, -1.2562f,
                    -0.9424f, -0.6568f, -0.3881f, -0.1284f,
                     0.1284f,  0.3881f,  0.6568f,  0.9424f,
                     1.2562f,  1.6180f,  2.0690f,  2.7326f};
        default:
            throw std::invalid_argument("bitwidth must be 1-4, got " + std::to_string(bitwidth));
    }
}

ScalarCodebook make_codebook(int bitwidth, int dim) {
    if (bitwidth < 1 || bitwidth > 4)
        throw std::invalid_argument("bitwidth must be 1-4");
    if (dim < 2)
        throw std::invalid_argument("dimension must be >= 2");

    auto centroids = normalized_centroids(bitwidth);
    float scale = 1.0f / std::sqrt(static_cast<float>(dim));

    for (auto& c : centroids)
        c *= scale;

    std::vector<float> boundaries;
    boundaries.reserve(centroids.size() - 1);
    for (std::size_t i = 0; i + 1 < centroids.size(); ++i)
        boundaries.push_back(0.5f * (centroids[i] + centroids[i + 1]));

    return ScalarCodebook{
        .bitwidth = bitwidth,
        .num_centroids = static_cast<int>(centroids.size()),
        .centroids = std::move(centroids),
        .boundaries = std::move(boundaries),
    };
}

}  // namespace turboquant