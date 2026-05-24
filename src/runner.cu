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
  dim3 block(BLOCK_SIZE_X, 1, 1);
  LayerNorm_kernel_naive<float, uint><<<grid, block, 0, 0>>>(totalRow,totalCol,A,out,mean,rstd,weight,bias);
  cudaCheck(cudaGetLastError());
}


void run_LayerNorm_kernel_double_warp_reduction(const uint totalRow, const uint totalCol, float *A,
                float *out, float* mean, float* rstd, const float* weight, const float* bias) {
  // 该 kernel 的 reduction[] 数组仅按 threadIdx.x 索引（共 BLOCK_SIZE_X/32=32 槽）：
  //   若 blockDim.y > 1，不同行（threadIdx.y 不同）会写同一槽，造成数据竞争。
  // 同时 BLOCK_SIZE_X=1024，blockDim.y>1 会超过 CUDA 每 block 1024 线程上限。
  // 因此 blockDim.y 必须为 1：一个 block 处理一行。
  // CUDA 规定每个 block 的总线程数 blockDim.x × blockDim.y × blockDim.z ≤ 1024，这来自 SM（流式多处理器）的硬件资源约束：
  dim3 grid(1, totalRow, 1);
  dim3 block(BLOCK_SIZE_X, 1, 1);
  assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");
  LayerNorm_kernel_double_warp_reduction<float, float4,uint><<<grid, block, 0, 0>>>(totalRow,totalCol,A,out,mean,rstd,weight,bias);
  cudaCheck(cudaGetLastError());
}

void run_LayerNorm_kernel_double_warp_reduction_unroll(const uint totalRow, const uint totalCol, float *A,
                float *out, float* mean, float* rstd, const float* weight, const float* bias) {
  dim3 grid(1, totalRow, 1);
  dim3 block(BLOCK_SIZE_X_kernel3, 1, 1);
  assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");
  LayerNorm_kernel_double_warp_reduction_unroll<float, float4,uint><<<grid, block, 0, 0>>>(totalRow,totalCol,A,out,mean,rstd,weight,bias);
  cudaCheck(cudaGetLastError());
}

void run_LayerNorm_kernel_welford_double_warp_reduction(const uint totalRow, const uint totalCol, float *A,
                float *out, float* mean, float* rstd, const float* weight, const float* bias) {
  dim3 grid(1, totalRow, 1);
  dim3 block(BLOCK_SIZE_X, 1, 1);
  assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");
  LayerNorm_kernel_welford_double_warp_reduction<float, float4,uint><<<grid, block, 0, 0>>>(totalRow,totalCol,A,out,mean,rstd,weight,bias);
  cudaCheck(cudaGetLastError());
}

void run_LayerNorm_kernel_welford_double_warp_reduction_unroll(const uint totalRow, const uint totalCol, float *A,
                float *out, float* mean, float* rstd, const float* weight, const float* bias) {
  dim3 grid(1, totalRow, 1);
  dim3 block(BLOCK_SIZE_X, 1, 1);
  assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");
  LayerNorm_kernel_welford_double_warp_reduction_unroll<float, float4,uint><<<grid, block, 0, 0>>>(totalRow,totalCol,A,out,mean,rstd,weight,bias);
  cudaCheck(cudaGetLastError());
}


void run_LayerNorm_kernel_warp_reduction_unroll_SMEM(const uint totalRow, const uint totalCol, float *A,
                float *out, float* mean, float* rstd, float* weight, float* bias) {
  dim3 grid(1, cuda::ceil_div(totalRow,BLOCK_SIZE_Y_SMEM), 1);
  dim3 block(BLOCK_SIZE_X_SMEM, BLOCK_SIZE_Y_SMEM, 1);
  assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");
  // per-block SMEM 上限 = 静态 + MaxDynamic。
  // 该 kernel 动态 SMEM = 0，故 MaxDynamic = 0，上限 = 静态 SMEM = 40.5 KB。
  // 默认上限 48 KB > 40.5 KB，当前无需调用；
  // 防御性保留：若 MAX_TOTALCOL 增大导致静态超过 48 KB，此调用解除默认限制。
  constexpr size_t static_smem = sizeof(float) * (BLOCK_SIZE_Y_SMEM * 32 + 2 * (MAX_TOTALCOL / 4) * 5);
  constexpr size_t max_dynamic = static_smem < 48 * 1024 ? 48 * 1024 - static_smem: 0;
  cudaCheck(cudaFuncSetAttribute(
    (const void*)LayerNorm_kernel_warp_reduction_unroll_SMEM<float, float4, uint>,
    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)max_dynamic));
  LayerNorm_kernel_warp_reduction_unroll_SMEM<float, float4,uint><<<grid, block, 0, 0>>>(totalRow,totalCol,A,out,mean,rstd,weight,bias);
  cudaCheck(cudaGetLastError());
}

void run_layernorm_forward_kernel_cg_warp(const uint totalRow, const uint totalCol, float *A,
                float *out, float* mean, float* rstd, float* weight, float* bias) {
  // 每个 warp（32 线程）负责一行；block_size 决定每个 block 打包的 warp 数
  // warps_per_block = block_size / 32；grid.x = ceil(totalRow / warps_per_block)
  // idx = blockIdx.x * warps_per_block + warp.meta_group_rank()
  // 注意：该 kernel 只使用 blockIdx.x，不能沿用其他 kernel 的 grid(1, totalRow, 1)
  const uint block_size = 128;                               // 每 block 4 个 warp，覆盖 4 行
  const uint warps_per_block = block_size / 32;
  dim3 block(block_size, 1, 1);
  dim3 grid(cuda::ceil_div(totalRow, warps_per_block), 1, 1);
  LayerNorm_forward_kernel_cg_warp<float, float4, uint><<<grid, block, 0, 0>>>(totalRow, totalCol, A, out, mean, rstd, weight, bias);
  cudaCheck(cudaGetLastError());
}


void run_layernorm_forward_kernel_cg_block(const uint totalRow, const uint totalCol, float *A,
                float *out, float* mean, float* rstd, float* weight, float* bias) {
  // 1 block = 1 行，idx = blockIdx.x
  // block.x = BLOCK_SIZE（1024 线程 = 32 个 warp），SMEM 恰好有 32 个槽
  // grid.x  = totalRow，每行一个 block
  // 注意：不能用 grid(1, totalRow, 1)，该 kernel 只读 blockIdx.x
  dim3 block(BLOCK_SIZE_X, 1, 1);
  dim3 grid(totalRow, 1, 1);
  LayerNorm_forward_kernel_cg_block<float, float4, uint><<<grid, block, 0, 0>>>(totalRow, totalCol, A, out, mean, rstd, weight, bias);
  cudaCheck(cudaGetLastError());
}

void run_layernorm_forward_kernel_cg_warp_advanced(const uint totalRow, const uint totalCol, float *A,
                float *out, float* mean, float* rstd, float* weight, float* bias) {
  // block: dim3(WARP_SIZE=32, block_y)
  //   threadIdx.x = warp 内 lane（负责列），threadIdx.y = block 内 warp 编号（负责行）
  //   idx = blockIdx.x * block_y + threadIdx.y
  // shared memory: s_weight(C) + s_bias(C) + s_in(block_y * C) = (2 + block_y) * C * sizeof(float)
  const uint block_y = 4;                                  // 每 block 处理 4 行（4 个 warp）
  dim3 block(WARP_SIZE, block_y);
  dim3 grid(cuda::ceil_div(totalRow, block_y), 1, 1);

  size_t smem = (size_t)(2 + block_y) * totalCol * sizeof(float);
  // 当 totalCol 较大时 smem 可能超过默认 48 KB，需显式请求更大动态 shared memory
  cudaFuncSetAttribute(
    (const void*)LayerNorm_forward_kernel_cg_warp_advanced<float, x128, uint>,
    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  LayerNorm_forward_kernel_cg_warp_advanced<float, x128, uint><<<grid, block, smem>>>(totalRow, totalCol, A, out, mean, rstd, weight, bias);
  cudaCheck(cudaGetLastError());
}