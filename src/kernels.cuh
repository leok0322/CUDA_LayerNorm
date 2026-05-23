#pragma once

#include "kernels/0_LayerNorm_kernel_base.cuh"
#include "kernels/1_LayerNrom_kernel_naive.cuh"
#include "kernels/2_LayerNorm_kernel_double_warp_reduction.cuh"
#include "kernels/3_LayerNorm_kernel_double_warp_reduction_unroll.cuh"
#include "kernels/4_LayerNorm_kernel_welford_double_warp_reduction.cuh"
#include "kernels/5_LayerNorm_kernel_welford_double_warp_reduction_unroll.cuh"


