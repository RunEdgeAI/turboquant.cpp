# turboquant.cpp

[![CI](https://github.com/RunEdgeAI/turboquant.cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/RunEdgeAI/turboquant.cpp/actions/workflows/ci.yml)

C++ implementation of TurboQuant, a near-optimal online vector quantization algorithm.

Based on [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)

## What it does

Compresses high-dimensional floating-point vectors to low-bitwidth integers (1-4 bits per coordinate) while preserving inner products and distances. No preprocessing, no training, no codebook learning. Vectors are quantized independently on arrival.

Two quantizers are provided:

- **QuantizerMSE** minimizes reconstruction error (mean-squared error).
- **QuantizerProd** provides unbiased inner product estimates with low variance.

Both achieve distortion within a constant factor (~2.7x) of the information-theoretic lower bound.

## Build

Built with [Bazel](https://bazel.build). Depends on Eigen 3.4, resolved automatically from the [Bazel Central Registry](https://registry.bazel.build) — no manual install needed.

```
bazel build //...
```

Run the example:

```
bazel run //:turboquant_example
```

To depend on turboquant from another Bazel project, add it to your `MODULE.bazel` and use `@turboquant` as a dependency.

## Usage

```cpp
#include <turboquant/turboquant.h>
#include <random>

using namespace turboquant;

std::mt19937 rng(42);
int dim = 1536;
int bitwidth = 3;

// MSE quantizer
QuantizerMSE qmse(dim, bitwidth, rng);
auto compressed = qmse.quantize(x);
Vec reconstructed = qmse.dequantize(compressed);

// Inner product quantizer (unbiased)
QuantizerProd qprod(dim, bitwidth, rng);
auto compressed = qprod.quantize(x);
float ip = qprod.estimate_inner_product(y, compressed);
```

## Theoretical guarantees

For unit-norm vectors at bitwidth `b`:

| Metric | Upper bound | Lower bound |
|--------|-------------|-------------|
| MSE | sqrt(3pi)/2 * 4^(-b) | 4^(-b) |
| Inner product error | sqrt(3pi^2) * norm(y)^2 / (d * 4^b) | norm(y)^2 / (d * 4^b) |

## Benchmarks

Single-threaded, Apple M4 Max, `bazel run -c opt //:turboquant_benchmark`. Compression ratio is fp32 size vs. the bit-packed wire format (indices + norm metadata).

| Quantizer | d | b | quantize/s | dequantize/s | inner product/s | compression |
|-----------|------|---|-----------:|-------------:|----------------:|------------:|
| MSE | 1536 | 1 | 10.0k | 9.6k | — | 31.3x |
| MSE | 1536 | 2 | 9.8k | 9.5k | — | 15.8x |
| MSE | 1536 | 4 | 9.7k | 9.6k | — | 8.0x |
| Prod | 1536 | 2 | 2.9k | 4.1k | 4.2k | 15.7x |
| Prod | 1536 | 4 | 2.9k | 4.0k | 4.1k | 7.9x |
| MSE | 256 | 2 | 319k | 312k | — | 15.1x |
| Prod | 256 | 2 | 108k | 154k | 162k | 14.2x |

Throughput is dominated by the dense d x d rotation matvec, so it scales as O(d^2) and is nearly independent of bitwidth (see Limitations). Run `bazel run -c opt //:turboquant_benchmark` for the full sweep (d = 256/768/1536, b = 1-4).

## Limitations

- Bitwidths 1-4 only (precomputed codebooks). Extending to higher bitwidths requires solving the Lloyd-Max optimization for the Beta distribution at the desired precision.
- Stores full d x d rotation matrix in memory. For large dimensions, a randomized Hadamard transform would reduce this to O(d log d).
- No SIMD optimization yet. The quantization loop is straightforward to vectorize with AVX2/AVX-512.
