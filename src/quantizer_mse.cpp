// Copyright (c) 2026 Edge AI
// SPDX-License-Identifier: MIT

#include "turboquant/quantizer_mse.h"

#include <cmath>
#include <algorithm>

namespace turboquant {

QuantizerMSE::QuantizerMSE(int dim, int bitwidth, std::mt19937& rng)
    : dim_(dim)
    , Pi_(make_rotation_matrix(dim, rng))
    , codebook_(make_codebook(bitwidth, dim)) {}

uint8_t QuantizerMSE::quantize_scalar(float val) const {
    auto it = std::upper_bound(codebook_.boundaries.begin(),
                               codebook_.boundaries.end(), val);
    return static_cast<uint8_t>(std::distance(codebook_.boundaries.begin(), it));
}

QuantizedMSE QuantizerMSE::quantize(const Vec& x) const {
    float norm = x.norm();
    Vec y = Pi_ * (x / norm);

    std::vector<uint8_t> indices(dim_);
    for (int j = 0; j < dim_; ++j)
        indices[j] = quantize_scalar(y(j));

    return QuantizedMSE{
        .indices = std::move(indices),
        .norm = norm,
        .dim = dim_,
        .bitwidth = codebook_.bitwidth,
    };
}

Vec QuantizerMSE::dequantize(const QuantizedMSE& q) const {
    Vec y_hat(dim_);
    for (int j = 0; j < dim_; ++j)
        y_hat(j) = codebook_.centroids[q.indices[j]];

    return q.norm * (Pi_.transpose() * y_hat);
}

}  // namespace turboquant