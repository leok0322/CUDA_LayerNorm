# pragma once

#include "common.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>          // cg::this_thread_block, cg::tiled_partition, thread_block_tile
#include <cooperative_groups/reduce.h>   // cg::reduce, cg::plus<T> 等规约操作符



// ============================================================
// Kernel 6：x128 向量化 + weight/bias 预加载到 shared memory
//   灵感来源：fused_residual_forward_kernel5
//
// block 维度：dim3(WARP_SIZE=32, block_y)
//   threadIdx.x：warp 内 lane，负责列方向（C 维度）
//   threadIdx.y：warp 编号，负责行方向（不同 token）
//
// 关键设计：
//   1. 将 weight 和 bias 预加载到 shared memory，多个 warp/行共享同一份副本，
//      彻底消除 weight/bias 的重复全局内存读取
//   2. 使用 x128（4 个 float 打包）做向量化读写，单次指令传输 128-bit，
//      提升内存带宽利用率并降低指令数
//   3. 将 inp 的当前行缓存在 shared memory（s_in），
//      均值计算完毕后直接从 shared memory 读取计算方差，避免再次访问全局内存
//   4. shared memory 超过默认 48KB 上限时需调用 cudaFuncSetAttribute；
//      失败则自动回退到 kernel5
// ============================================================
template <typename scalar_t, typename scalar_t4, typename scalar_i>
__global__ void LayerNorm_forward_kernel_cg_warp_advanced(scalar_i totalRow, scalar_i totalCol, const scalar_t*  __restrict__ A, scalar_t*
  __restrict__ out, scalar_t* __restrict__ mean, scalar_t* __restrict__ rstd, const scalar_t*  __restrict__ weight,
                                    const scalar_t* __restrict__ bias) {
    assert(blockDim.x == WARP_SIZE); // x 维度必须恰好是一个 warp（32 线程）

    // shared memory 布局（按 x128 对齐）：
    //   [0,          C/4)        → s_weight：共享的 weight
    //   [C/4,        2*C/4)      → s_bias：共享的 bias
    //   [2*C/4 + threadIdx.y*C/4, ...)  → s_in[threadIdx.y]：当前 warp 行的输入
    extern __shared__ char params[];
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias   = reinterpret_cast<x128*>(params) + (totalCol / x128::size);
    // s_in 为每个 row-warp 分配独立的 C 大小缓冲，避免不同 warp 间干扰
    x128* s_in     = reinterpret_cast<x128*>(params) + ((2 + threadIdx.y) * totalCol / x128::size);

    // 协作加载 weight 和 bias 到 shared memory（所有线程参与，确保在任何线程 early return 前完成）
    int sidx = (threadIdx.x + WARP_SIZE * threadIdx.y) * x128::size;
    for(int i = sidx; i < totalCol; i += blockDim.y * WARP_SIZE * x128::size) {
        s_weight[i/x128::size] = load128(weight + i);
        s_bias[i/x128::size]   = load128(bias + i);
    }
    __syncthreads(); // 保证 s_weight/s_bias 对所有线程可见后再处理各自的行

    // 每个 warp（threadIdx.y）负责一行
    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if(idx >= totalRow) { return; } // 越界保护（需在 __syncthreads() 之后）

    A += idx * totalCol;
    out += idx * totalCol;

    const float eps = 1e-5f;

    // ── 第一遍：向量化读取 inp，缓存到 s_in，同时累加 Σx ──────────────
    float sum = 0.0f;
    for(int c = threadIdx.x * x128::size; c < totalCol; c += WARP_SIZE * x128::size) {
        const x128 in_data = load128cs(A + c); // streaming load，不污染 L1
        for(int k = 0; k < x128::size; ++k) {
            sum += (float)in_data[k];
        }
        s_in[c / x128::size] = in_data; // 缓存本行数据，避免后续再次访问全局内存
    }
    // warpReduceSum 而非 cg::reduce 的原因：
    //   cg::reduce(warp, val, op) 只接受标量类型（float/int 等）
    //   本 kernel 内层用 x128（4×float 结构体）展开累加，无法直接传给 cg::reduce；
    //   必须先把 x128 分量手动累加到标量 sum，再调用 warpReduceSum（__shfl_down_sync 实现）
    //   kernel 7 block 一维、规约对象始终是 float 标量，才可以直接用 cg::reduce
    sum = warpReduceSum(sum);
    float m = sum / totalCol;

    // ── 第二遍：从 s_in（shared memory）计算方差，无全局内存访问 ─────────
    float v = 0.f;
    for(int c = threadIdx.x * x128::size; c < totalCol; c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size]; // 从 shared memory 读
        for(int k = 0; k < x128::size; ++k) {
            v += ((float)in_data[k] - m) * ((float)in_data[k] - m);
        }
    }
    v = warpReduceSum(v) / totalCol; // 同上，x128 展开后标量规约
    float s = rsqrtf(v + eps);

    // ── 第三遍：归一化 + 仿射变换，weight/bias 从 shared memory 读 ──────
    for(int c = threadIdx.x * x128::size; c < totalCol; c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];
        const x128 w       = s_weight[c / x128::size]; // 来自 shared memory，无全局读
        const x128 b       = s_bias[c / x128::size];
        x128 out_data;
        for(int k = 0; k < x128::size; ++k) {
            float n    = s * ((float)in_data[k] - m);
            out_data[k] = n * (float)w[k] + (float)b[k];
        }
        store128cs(out + c, out_data); // streaming store
    }

    // 仅 lane 0 写出统计量
    if(threadIdx.x == 0 && mean != nullptr) { __stcs(mean + idx, m); }
    if(threadIdx.x == 0 && rstd != nullptr) { __stcs(rstd + idx, s); }
}