#pragma once
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "gpu_arch.hpp"

namespace oz2_const {

size_t grids_invscaling;
size_t grids_conv32i8u;

size_t threads_scaling;
size_t threads_conv32i8u;
size_t threads_invscaling;

} // namespace oz2_const
