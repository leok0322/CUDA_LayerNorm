#pragma once


void run_LayerNorm_kernel_naive(uint totalRow, uint totalCol, const float *A,
                float *out, float* mean, float* rstd, const float* weight, const float* bias);

void run_LayerNorm_kernel_base(uint totalRow, uint totalCol, const float *A,
                float *out, float* mean, float* rstd, const float* weight, const float* bias);


void run_LayerNorm_kernel_double_warp_reduction(uint totalRow, uint totalCol, float *A,
                float *out, float* mean, float* rstd, const float* weight, const float* bias);

void run_LayerNorm_kernel_double_warp_reduction_unroll(uint totalRow, uint totalCol, float *A,
                float *out, float* mean, float* rstd, const float* weight, const float* bias);

void run_LayerNorm_kernel_welford_double_warp_reduction(uint totalRow, uint totalCol, float *A,
                float *out, float* mean, float* rstd, const float* weight, const float* bias);


void run_LayerNorm_kernel_welford_double_warp_reduction_unroll(uint totalRow, uint totalCol, float *A,
                float *out, float* mean, float* rstd, const float* weight, const float* bias);


void run_LayerNorm_kernel_double_warp_reduction_unroll_SMEM(uint totalRow, uint totalCol, float *A,
                float *out, float* mean, float* rstd, float* weight, float* bias);