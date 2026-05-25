# pragma once

#include "common.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>          // cg::this_thread_block, cg::tiled_partition, thread_block_tile
#include <cooperative_groups/reduce.h>   // cg::reduce, cg::plus<T> 等规约操作符


// ============================================================
// Kernel 5：block 粒度处理一行，算法同 kernel4
//
// 与 kernel4（warp/行）的区别：
//   每个 block 包含多个 warp，共同处理一行；适合 C 远大于 32 的情况
//   （kernel4 中单个 warp 的 thread coarsening 比率更高，ILP 更低）
//
// 两阶段归约：
//   阶段 1：warp 内用 cg::reduce 得到每个 warp 的局部和
//   阶段 2：将各 warp 结果写入 shared_sum/shared_sum2，
//           再由第一个 warp 做最终归约
// ============================================================
template <typename scalar_t, typename scalar_t4, typename scalar_i>
__global__ void LayerNorm_forward_kernel_cg_block(scalar_i totalRow, scalar_i totalCol, const scalar_t*  __restrict__ A, scalar_t*
  __restrict__ out, scalar_t* __restrict__ mean, scalar_t* __restrict__ rstd, const scalar_t*  __restrict__ weight,
                                    const scalar_t* __restrict__ bias) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // shared memory 各存 32 个 float，对应最多 1024/32 = 32 个 warp 的中间结果
    __shared__ float shared_sum[32];
    __shared__ float shared_sum2[32];

    int num_warps = blockDim.x / 32;
    int warp_id   = threadIdx.x / 32; // 当前线程属于第几个 warp
    int lane_id   = threadIdx.x % 32; // 当前线程在 warp 内的 lane 编号
    int idx       = blockIdx.x;       // 一个 block 负责一行

    if (idx >= totalRow) return;

    const float* x = A + idx * totalCol;

    // 阶段 1a：线程局部累加
    float thread_sum  = 0.0;
    float thread_sum2 = 0.0;
    for (int i = threadIdx.x; i < totalCol; i += blockDim.x) {
        float xi = x[i];
        thread_sum  += xi;
        thread_sum2 += xi * xi;
    }

    // 阶段 1b：warp 内归约
    float warp_sum  = cg::reduce(warp, thread_sum,  cg::plus<float>{});
    float warp_sum2 = cg::reduce(warp, thread_sum2, cg::plus<float>{});

    // 将各 warp 的结果写入 shared memory（所有 lane 均写，lane != 0 的写入会被覆盖，无需 guard）
    shared_sum[warp_id]  = warp_sum;
    shared_sum2[warp_id] = warp_sum2;
    __syncthreads(); // 确保所有 warp 均已写完

    // 阶段 2：第一个 warp 读取各 warp 结果并做最终归约
    // 超出 warp 数量的 lane 补零（不参与实际求和）
    warp_sum  = (lane_id < num_warps) ? shared_sum[lane_id]  : 0.0f;
    warp_sum2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;

    float block_sum  = cg::reduce(warp, warp_sum,  cg::plus<float>{}); // 全行 Σx
    float block_sum2 = cg::reduce(warp, warp_sum2, cg::plus<float>{}); // 全行 Σx²

    block_sum  /= totalCol;
    block_sum2 /= totalCol;
    float m   = block_sum;
    float var = block_sum2 - m * m; // Var(x) = E[x²] - E[x]²
    float s   = rsqrtf(var + 1e-5f);

    // 线程 0 写出统计量（block 内 lane 0 同时也是整体 threadIdx.x == 0）
    if(threadIdx.x == 0 && mean != nullptr) { __stcs(mean + idx, m); }
    if(threadIdx.x == 0 && rstd != nullptr) { __stcs(rstd + idx, s); }

    // 归一化 + 仿射变换，所有线程参与
    float* o = out + idx * totalCol;
    for (int i = threadIdx.x; i < totalCol; i += blockDim.x) {
        float n = s * (__ldcs(x+i) - m);
        __stcs(o+i, n * weight[i] + bias[i]);
    }
}