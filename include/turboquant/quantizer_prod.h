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

    // The query-only work of estimate_inner_product — the rotation and the QJL
    // projection of y, both O(dim^2). Prepare once per query, then score many
    // codes at O(dim) each.
    struct PreparedQuery {
        Vec rotated;    // Pi * y
        Vec projected;  // S * y
    };

    QuantizedProd quantize(const Vec& x) const;
    Vec dequantize(const QuantizedProd& q) const;
    float estimate_inner_product(const Vec& y, const QuantizedProd& q) const;

    PreparedQuery prepare_query(const Vec& y) const;
    // Same arithmetic as estimate_inner_product(y, q) for p = prepare_query(y),
    // so the two agree exactly.
    float estimate_inner_product(const PreparedQuery& p, const QuantizedProd& q) const;

    int dim() const { return dim_; }
    int bitwidth() const { return total_bitwidth_; }

private:
    int dim_;
    int total_bitwidth_;
    QuantizerMSE mse_quantizer_;
    QJL qjl_;
};

}  // namespace turboquant