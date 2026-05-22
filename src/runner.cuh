#pragma once


void run_LayerNorm_kernel_naive(uint totalRow, uint totalCol, const float *A,
                float *out, float* mean, float* rstd, const float* weight, const float* bias);

void run_LayerNorm_kernel_base(uint totalRow, uint totalCol, const float *A,
                float *out, float* mean, float* rstd, const float* weight, const float* bias);
