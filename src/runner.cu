#include "kernels.cuh"
#include "runner.cuh"
#include <cuda_runtime.h>
#include "error_check.cuh"

#include <cudnn.h>         // cuDNN API：cudnnNormalizationForwardInference 等
#include <cstdio>          // fprintf / stderr
#include <cuda/cmath>


// ── 基准实现：使用 cuDNN LayerNorm API 完成前向推理 ───────────────────────────
//
// 参数：
//   totalRow : 样本数（batch × seq_len），即归一化的行数 N
//   totalCol : 特征维度（hidden_dim），即每行归一化的元素数 C
//   A        : 输入矩阵 [N, C]，设备内存
//   out      : 输出矩阵 [N, C]，设备内存
//   mean     : 每行均值 [N]，设备内存（供反向传播 / 验证使用）
//   rstd     : 每行倒数标准差 [N]，设备内存（供反向传播 / 验证使用）
//   weight   : 可学习参数 γ [C]，设备内存（默认初始值 1.0f）
//   bias     : 可学习参数 β [C]，设备内存（默认初始值 0.0f）
void run_LayerNorm_kernel_base(const uint totalRow, const uint totalCol, const float *A,
                float *out, float* mean, float* rstd, const float* weight, const float* bias) {
  // run_LayerNorm_kernel_base_graph(totalRow,totalCol,A,out,mean,rstd,weight,bias);
  dim3 grid(cuda::ceil_div(totalCol, 32), cuda::ceil_div(totalRow, 32), 1);
  dim3 block(32, 32, 1);
  LayerNorm_kernel_base<float, uint><<<grid, block, 0, 0>>>(totalRow,totalCol,A,out,mean,rstd,weight,bias);
  cudaCheck(cudaGetLastError());

}

void run_LayerNorm_kernel_naive(const uint totalRow, const uint totalCol, const float *A,
                float *out, float* mean, float* rstd, const float* weight, const float* bias) {

  dim3 grid(1, totalRow, 1);
  dim3 block(BLOCK_SIZE, 1, 1);
  LayerNorm_kernel_naive<float, uint><<<grid, block, 0, 0>>>(totalRow,totalCol,A,out,mean,rstd,weight,bias);
  cudaCheck(cudaGetLastError());
}


void run_LayerNorm_kernel_double_warp_reduction(const uint totalRow, const uint totalCol, float *A,
                float *out, float* mean, float* rstd, const float* weight, const float* bias) {
  dim3 grid(1, totalRow, 1);
  dim3 block(BLOCK_SIZE, 1, 1);
  assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");
  LayerNorm_kernel_double_warp_reduction<float, float4,uint><<<grid, block, 0, 0>>>(totalRow,totalCol,A,out,mean,rstd,weight,bias);
  cudaCheck(cudaGetLastError());
}

void run_LayerNorm_kernel_double_warp_reduction_unroll(const uint totalRow, const uint totalCol, float *A,
                float *out, float* mean, float* rstd, const float* weight, const float* bias) {
  dim3 grid(1, totalRow, 1);
  dim3 block(BLOCK_SIZE, 1, 1);
  assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");
  LayerNorm_kernel_double_warp_reduction_unroll<float, float4,uint><<<grid, block, 0, 0>>>(totalRow,totalCol,A,out,mean,rstd,weight,bias);
  cudaCheck(cudaGetLastError());
}

void run_LayerNorm_kernel_welford_double_warp_reduction(const uint totalRow, const uint totalCol, float *A,
                float *out, float* mean, float* rstd, const float* weight, const float* bias) {
  dim3 grid(1, totalRow, 1);
  dim3 block(BLOCK_SIZE, 1, 1);
  assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");
  LayerNorm_kernel_welford_double_warp_reduction<float, float4,uint><<<grid, block, 0, 0>>>(totalRow,totalCol,A,out,mean,rstd,weight,bias);
  cudaCheck(cudaGetLastError());
}

void run_LayerNorm_kernel_welford_double_warp_reduction_unroll(const uint totalRow, const uint totalCol, float *A,
                float *out, float* mean, float* rstd, const float* weight, const float* bias) {
  dim3 grid(1, totalRow, 1);
  dim3 block(BLOCK_SIZE, 1, 1);
  assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");
  LayerNorm_kernel_welford_double_warp_reduction_unroll<float, float4,uint><<<grid, block, 0, 0>>>(totalRow,totalCol,A,out,mean,rstd,weight,bias);
  cudaCheck(cudaGetLastError());
}


void run_LayerNorm_kernel_double_warp_reduction_unroll_SMEM(const uint totalRow, const uint totalCol, float *A,
                float *out, float* mean, float* rstd, float* weight, float* bias) {
  dim3 grid(1, totalRow, 1);
  dim3 block(BLOCK_SIZE, 1, 1);
  assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");
  LayerNorm_kernel_double_warp_reduction_unroll_SMEM<float, float4,uint><<<grid, block, 0, 0>>>(totalRow,totalCol,A,out,mean,rstd,weight,bias);
  cudaCheck(cudaGetLastError());
}