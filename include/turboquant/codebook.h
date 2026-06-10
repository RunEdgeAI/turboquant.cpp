// Copyright (c) 2026 Edge AI
// SPDX-License-Identifier: MIT

#pragma once

#include "types.h"

namespace turboquant {

ScalarCodebook make_codebook(int bitwidth, int dim);
std::vector<float> normalized_centroids(int bitwidth);

}  // namespace turboquant