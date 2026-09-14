// Copyright (c) 2026 Edge AI
// SPDX-License-Identifier: MIT

#pragma once

#include "types.h"
#include "codebook.h"
#include "rotation.h"

#include <random>

namespace turboquant {

class QuantizerMSE {
public:
    QuantizerMSE(int dim, int bitwidth, std::mt19937& rng);

    QuantizedMSE quantize(const Vec& x) const;
    Vec dequantize(const QuantizedMSE& q) const;

    // <y, dequantize(q)> given rotated_y = rotation() * y, in O(dim): the
    // rotation is orthogonal, so <y, Pi^T y_hat> = <Pi y, y_hat>.
    float inner_product_rotated(const Vec& rotated_y, const QuantizedMSE& q) const;

    int dim() const { return dim_; }
    int bitwidth() const { return codebook_.bitwidth; }
    const ScalarCodebook& codebook() const { return codebook_; }
    const Mat& rotation() const { return Pi_; }

private:
    int dim_;
    Mat Pi_;
    ScalarCodebook codebook_;

    uint8_t quantize_scalar(float val) const;
};

}  // namespace turboquant