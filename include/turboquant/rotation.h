// Copyright (c) 2026 Edge AI
// SPDX-License-Identifier: MIT

#pragma once

#include "types.h"
#include <random>

namespace turboquant {

Mat make_rotation_matrix(int dim, std::mt19937& rng);
Mat make_gaussian_matrix(int rows, int cols, std::mt19937& rng);

}  // namespace turboquant