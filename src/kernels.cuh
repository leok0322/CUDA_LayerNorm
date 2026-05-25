#pragma once

#include "kernels/0_LayerNorm_kernel_base.cuh"
#include "kernels/1_LayerNrom_kernel_naive.cuh"
#include "kernels/2_LayerNorm_kernel_double_warp_reduction.cuh"
#include "kernels/3_LayerNorm_kernel_double_warp_reduction_unroll.cuh"
#include "kernels/4_LayerNorm_kernel_welford_double_warp_reduction.cuh"
#include "kernels/5_LayerNorm_kernel_welford_double_warp_reduction_unroll.cuh"
#include "kernels/6_layerNorm_kernel_double_warp_redcuction_unroll_SMEM.cuh"
#include "kernels/7_LayerNorm_kernel_cooprative_groups_warp.cuh"
#include "kernels/8_LayerNorm_kernel_cooprative_groups_block.cuh"
#include "kernels/9_LayerNorm_kernel_cooperative_warp_advanced.cuh"
#include "kernels/10_LayerNorm_kernel_warp_reduction_unroll_SMEM.cuh"

