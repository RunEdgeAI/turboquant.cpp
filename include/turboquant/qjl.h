// Copyright (c) 2026 Edge AI
// SPDX-License-Identifier: MIT

#pragma once

#include "types.h"

namespace turboquant {

class QJL {
public:
    QJL(int dim, const Mat& projection_matrix);

    std::vector<int8_t> quantize(const Vec& x) const;
    Vec dequantize(std::span<const int8_t> signs, float residual_norm) const;
    float estimate_inner_product(const Vec& y, std::span<const int8_t> signs,
                                 float residual_norm) const;

private:
    int dim_;
    Mat S_;
    float scale_;
};

}  // namespace turboquant