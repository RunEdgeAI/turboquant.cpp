// Copyright (c) 2026 Edge AI
// SPDX-License-Identifier: MIT

#include "turboquant/qjl.h"

#include <cmath>
#include <numbers>

namespace turboquant {

QJL::QJL(int dim, const Mat& projection_matrix)
    : dim_(dim)
    , S_(projection_matrix)
    , scale_(std::sqrt(std::numbers::pi_v<float> / 2.0f) / static_cast<float>(dim)) {}

std::vector<int8_t> QJL::quantize(const Vec& x) const {
    Vec projected = S_ * x;

    std::vector<int8_t> signs(dim_);
    for (int i = 0; i < dim_; ++i)
        signs[i] = projected(i) >= 0.0f ? 1 : -1;

    return signs;
}

Vec QJL::dequantize(std::span<const int8_t> signs, float residual_norm) const {
    Vec z(dim_);
    for (int i = 0; i < dim_; ++i)
        z(i) = static_cast<float>(signs[i]);

    return (scale_ * residual_norm) * (S_.transpose() * z);
}

float QJL::estimate_inner_product(const Vec& y, std::span<const int8_t> signs,
                                   float residual_norm) const {
    return estimate_inner_product_projected(project(y), signs, residual_norm);
}

Vec QJL::project(const Vec& y) const {
    return S_ * y;
}

float QJL::estimate_inner_product_projected(const Vec& projected_y,
                                            std::span<const int8_t> signs,
                                            float residual_norm) const {
    // Avoids full dequantization: <y, dequant> = scale * gamma * <S*y, z>
    float dot = 0.0f;
    for (int i = 0; i < dim_; ++i)
        dot += projected_y(i) * static_cast<float>(signs[i]);

    return scale_ * residual_norm * dot;
}

}  // namespace turboquant