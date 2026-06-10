// Copyright (c) 2026 Edge AI
// SPDX-License-Identifier: MIT

#pragma once

#include "types.h"
#include "quantizer_mse.h"
#include "qjl.h"

#include <memory>
#include <random>

namespace turboquant {

class QuantizerProd {
public:
    QuantizerProd(int dim, int bitwidth, std::mt19937& rng);

    QuantizedProd quantize(const Vec& x) const;
    Vec dequantize(const QuantizedProd& q) const;
    float estimate_inner_product(const Vec& y, const QuantizedProd& q) const;

    int dim() const { return dim_; }
    int bitwidth() const { return total_bitwidth_; }

private:
    int dim_;
    int total_bitwidth_;
    QuantizerMSE mse_quantizer_;
    QJL qjl_;
};

}  // namespace turboquant