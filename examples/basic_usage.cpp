// Copyright (c) 2026 Edge AI
// SPDX-License-Identifier: MIT

#include <turboquant/turboquant.h>

#include <cmath>
#include <iostream>
#include <iomanip>
#include <numbers>
#include <random>

using namespace turboquant;

namespace {

Vec random_unit_vector(int dim, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    Vec v(dim);
    for (int i = 0; i < dim; ++i)
        v(i) = normal(rng);
    v.normalize();
    return v;
}

void demo_mse_quantizer(int dim, int bitwidth, std::mt19937& rng) {
    std::cout << "--- MSE Quantizer (d=" << dim << ", b=" << bitwidth << ") ---\n";

    QuantizerMSE quantizer(dim, bitwidth, rng);

    Vec x = random_unit_vector(dim, rng);
    auto q = quantizer.quantize(x);
    Vec x_hat = quantizer.dequantize(q);

    float mse = (x - x_hat).squaredNorm();
    float upper_bound = std::sqrt(3.0f * std::numbers::pi_v<float>) / 2.0f
                        * std::pow(0.25f, static_cast<float>(bitwidth));
    float lower_bound = std::pow(0.25f, static_cast<float>(bitwidth));

    std::cout << "  MSE distortion: " << std::fixed << std::setprecision(6) << mse << "\n";
    std::cout << "  Lower bound:    " << lower_bound << "\n";
    std::cout << "  Upper bound:    " << upper_bound << "\n\n";
}

void demo_prod_quantizer(int dim, int bitwidth, std::mt19937& rng) {
    std::cout << "--- Inner Product Quantizer (d=" << dim << ", b=" << bitwidth << ") ---\n";

    QuantizerProd quantizer(dim, bitwidth, rng);

    Vec x = random_unit_vector(dim, rng);
    Vec y = random_unit_vector(dim, rng);

    float true_ip = y.dot(x);
    auto q = quantizer.quantize(x);
    float estimated_ip = quantizer.estimate_inner_product(y, q);

    std::cout << "  True inner product:      " << std::fixed << std::setprecision(6)
              << true_ip << "\n";
    std::cout << "  Estimated inner product: " << estimated_ip << "\n";
    std::cout << "  Absolute error:          " << std::abs(true_ip - estimated_ip) << "\n\n";
}

void demo_batch_mse_stats(int dim, int bitwidth, int num_samples, std::mt19937& rng) {
    std::cout << "--- Batch MSE Statistics (d=" << dim << ", b=" << bitwidth
              << ", n=" << num_samples << ") ---\n";

    QuantizerMSE quantizer(dim, bitwidth, rng);

    float total_mse = 0.0f;
    for (int i = 0; i < num_samples; ++i) {
        Vec x = random_unit_vector(dim, rng);
        auto q = quantizer.quantize(x);
        Vec x_hat = quantizer.dequantize(q);
        total_mse += (x - x_hat).squaredNorm();
    }

    std::cout << "  Average MSE: " << std::fixed << std::setprecision(6)
              << total_mse / static_cast<float>(num_samples) << "\n\n";
}

}  // namespace

int main() {
    std::mt19937 rng(42);
    constexpr int dim = 256;

    for (int b = 1; b <= 4; ++b)
        demo_mse_quantizer(dim, b, rng);

    for (int b = 2; b <= 4; ++b)
        demo_prod_quantizer(dim, b, rng);

    demo_batch_mse_stats(dim, 2, 1000, rng);

    return 0;
}