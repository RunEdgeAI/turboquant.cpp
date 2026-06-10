// Copyright (c) 2026 Edge AI
// SPDX-License-Identifier: MIT

// Throughput and compression-ratio benchmark.
// Build with optimization: bazel run -c opt //:turboquant_benchmark

#include <turboquant/turboquant.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace turboquant;

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kMinSeconds = 0.3;
constexpr int kMinIters = 50;

// Sink accumulated across all timed loops so the compiler cannot
// eliminate the work; printed at the end.
volatile float g_sink = 0.0f;

std::vector<Vec> random_unit_vectors(int n, int dim, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<Vec> vs(n);
    for (auto& v : vs) {
        v.resize(dim);
        for (int i = 0; i < dim; ++i)
            v(i) = normal(rng);
        v.normalize();
    }
    return vs;
}

// Runs fn(i) repeatedly for at least kMinSeconds and returns calls per second.
template <typename F>
double measure_per_sec(F&& fn) {
    int i = 0;
    for (; i < 10; ++i)  // warmup
        fn(i);

    auto start = Clock::now();
    double elapsed = 0.0;
    int iters = 0;
    while (elapsed < kMinSeconds || iters < kMinIters) {
        fn(i++);
        ++iters;
        elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    }
    return iters / elapsed;
}

// Encoded size of the wire format in bytes: bit-packed indices plus
// float metadata. The in-memory structs are not bit-packed (one byte
// per index); this reports what the encoding represents.
int encoded_bytes_mse(int dim, int bitwidth) {
    return (dim * bitwidth + 7) / 8 + 4;  // indices + norm
}

int encoded_bytes_prod(int dim, int total_bitwidth) {
    return (dim * (total_bitwidth - 1) + 7) / 8  // MSE indices
         + (dim + 7) / 8                         // QJL signs
         + 8;                                    // norm + residual_norm
}

void print_header(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n"
              << std::setw(6) << "d" << std::setw(4) << "b"
              << std::setw(16) << "quantize/s" << std::setw(16) << "dequantize/s"
              << std::setw(16) << "inner_prod/s"
              << std::setw(10) << "bytes" << std::setw(8) << "ratio" << "\n";
}

void bench_mse(int dim, int bitwidth, std::mt19937& rng) {
    QuantizerMSE quantizer(dim, bitwidth, rng);
    auto inputs = random_unit_vectors(64, dim, rng);

    std::vector<QuantizedMSE> qs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
        qs[i] = quantizer.quantize(inputs[i]);

    double quant_per_sec = measure_per_sec([&](int i) {
        auto q = quantizer.quantize(inputs[i % inputs.size()]);
        g_sink = g_sink + q.norm;
    });
    double dequant_per_sec = measure_per_sec([&](int i) {
        Vec x_hat = quantizer.dequantize(qs[i % qs.size()]);
        g_sink = g_sink + x_hat(0);
    });

    int bytes = encoded_bytes_mse(dim, bitwidth);
    std::cout << std::setw(6) << dim << std::setw(4) << bitwidth
              << std::setw(16) << std::fixed << std::setprecision(0) << quant_per_sec
              << std::setw(16) << dequant_per_sec
              << std::setw(16) << "-"
              << std::setw(10) << bytes
              << std::setw(7) << std::setprecision(1)
              << static_cast<double>(dim * 4) / bytes << "x\n";
}

void bench_prod(int dim, int bitwidth, std::mt19937& rng) {
    QuantizerProd quantizer(dim, bitwidth, rng);
    auto inputs = random_unit_vectors(64, dim, rng);
    auto queries = random_unit_vectors(64, dim, rng);

    std::vector<QuantizedProd> qs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
        qs[i] = quantizer.quantize(inputs[i]);

    double quant_per_sec = measure_per_sec([&](int i) {
        auto q = quantizer.quantize(inputs[i % inputs.size()]);
        g_sink = g_sink + q.residual_norm;
    });
    double dequant_per_sec = measure_per_sec([&](int i) {
        Vec x_hat = quantizer.dequantize(qs[i % qs.size()]);
        g_sink = g_sink + x_hat(0);
    });
    double ip_per_sec = measure_per_sec([&](int i) {
        g_sink = g_sink + quantizer.estimate_inner_product(
            queries[i % queries.size()], qs[i % qs.size()]);
    });

    int bytes = encoded_bytes_prod(dim, bitwidth);
    std::cout << std::setw(6) << dim << std::setw(4) << bitwidth
              << std::setw(16) << std::fixed << std::setprecision(0) << quant_per_sec
              << std::setw(16) << dequant_per_sec
              << std::setw(16) << ip_per_sec
              << std::setw(10) << bytes
              << std::setw(7) << std::setprecision(1)
              << static_cast<double>(dim * 4) / bytes << "x\n";
}

}  // namespace

int main() {
    std::mt19937 rng(42);
    const std::vector<int> dims = {256, 768, 1536};

    std::cout << "turboquant benchmark (single thread)\n"
              << "ratio = fp32 size / encoded size (bit-packed wire format)\n";

    print_header("QuantizerMSE");
    for (int dim : dims)
        for (int b = 1; b <= 4; ++b)
            bench_mse(dim, b, rng);

    print_header("QuantizerProd");
    for (int dim : dims)
        for (int b = 2; b <= 4; ++b)
            bench_prod(dim, b, rng);

    std::cout << "\n(sink: " << g_sink << ")\n";
    return 0;
}
