// Copyright (c) 2026 Edge AI
// SPDX-License-Identifier: MIT

#include "turboquant/quantizer_prod.h"

namespace turboquant {

QuantizerProd::QuantizerProd(int dim, int bitwidth, std::mt19937& rng)
    : dim_(dim)
    , total_bitwidth_(bitwidth)
    , mse_quantizer_(dim, bitwidth - 1, rng)
    , qjl_(dim, make_gaussian_matrix(dim, dim, rng)) {
    if (bitwidth < 2)
        throw std::invalid_argument("QuantizerProd requires bitwidth >= 2");
}

QuantizedProd QuantizerProd::quantize(const Vec& x) const {
    auto mse_q = mse_quantizer_.quantize(x);
    Vec x_hat = mse_quantizer_.dequantize(mse_q);

    Vec r = x - x_hat;
    float r_norm = r.norm();

    std::vector<int8_t> signs;
    if (r_norm > 1e-10f) {
        signs = qjl_.quantize(r / r_norm);
    } else {
        signs.assign(dim_, 1);
        r_norm = 0.0f;
    }

    return QuantizedProd{
        .mse_part = std::move(mse_q),
        .qjl_signs = std::move(signs),
        .residual_norm = r_norm,
    };
}

Vec QuantizerProd::dequantize(const QuantizedProd& q) const {
    return mse_quantizer_.dequantize(q.mse_part)
         + qjl_.dequantize(q.qjl_signs, q.residual_norm);
}

float QuantizerProd::estimate_inner_product(const Vec& y, const QuantizedProd& q) const {
    float ip_mse = y.dot(mse_quantizer_.dequantize(q.mse_part));
    float ip_qjl = qjl_.estimate_inner_product(y, q.qjl_signs, q.residual_norm);
    return ip_mse + ip_qjl;
}

}  // namespace turboquant