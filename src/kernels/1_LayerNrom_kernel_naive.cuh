#pragma once

#include <cuda_runtime.h>
#include "common.cuh"

template <typename scalar_t, typename scalar_i>
__global__ void LayerNorm_kernel_naive(const scalar_i totalRow, const scalar_i totalCol, const scalar_t *A,
                scalar_t *out, scalar_t* mean, scalar_t* rstd, const scalar_t* weight, const scalar_t* bias) {
  const scalar_i row = blockDim.y * blockIdx.y + threadIdx.y;
  const scalar_i col = threadIdx.x;
  const scalar_i threaNum = blockDim.x;
  if (row < totalRow) {
    scalar_t rowMean {};
    scalar_t rowVar {};
    A += row * totalCol;
    out += row * totalCol;
    //计算行均值
    for (scalar_i i {0}; i < totalCol; ++i) {
       rowMean += A[i];
    }
    rowMean /= static_cast<scalar_t>(totalCol);
    if (col == 0) {
      mean[row] = rowMean;
    }
    // 计算行方差
    for (scalar_i i {0}; i < totalCol; ++i) {
      scalar_t diff { A[i] - rowMean };
      rowVar += diff * diff;
    }
    rowVar /= static_cast<scalar_t>(totalCol);
    // rsqrtf：CUDA 设备端内置函数，无需 #include，直接映射硬件倒数平方根指令
    //   rsqrtf(x) = 1 / sqrtf(x)，仅接受 float，double 版本为 rsqrt(x)
    //   存倒数（rstd）而非标准差，后续归一化用 * rstd 代替 / std，避免除法开销
    //   模板类型 scalar_t 可能为 double，需先 cast 为 float 再 cast 回 scalar_t
    rowVar  = static_cast<scalar_t>(rsqrtf(static_cast<float>(rowVar + 1e-5f)));
    if (col == 0) {
      rstd[row] = rowVar;
    }
    for (scalar_i i {col}; i < totalCol; i+=threaNum) {
      out[i] = weight[i] * (A[i] - rowMean) * rowVar + bias[i];
    }
  }
}