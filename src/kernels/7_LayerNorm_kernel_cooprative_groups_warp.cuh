# pragma once

#include "common.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>          // cg::this_thread_block, cg::tiled_partition, thread_block_tile
#include <cooperative_groups/reduce.h>   // cg::reduce, cg::plus<T> 等规约操作符


template <typename scalar_t, typename scalar_t4, typename scalar_i>
__global__ void LayerNorm_forward_kernel_cg_warp(scalar_i totalRow, scalar_i totalCol, const scalar_t*  __restrict__ A, scalar_t*
  __restrict__ out, scalar_t* __restrict__ mean, scalar_t* __restrict__ rstd, const scalar_t*  __restrict__ weight,
                                    const scalar_t* __restrict__ bias) {
  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  // 将 block 切分为 32 线程的 warp tile；每个 warp 负责一行
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

  // meta_group_rank()：当前 warp 在 block 内的编号（warp index）
  // meta_group_size()：通用公式 = (blockDim.x × blockDim.y × blockDim.z) / 32
  //                   本 kernel block 为一维 (block_size, 1, 1)，简化为 blockDim.x / 32
  int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
  if(idx >= totalRow) { return; }

  const float* x = A + idx * totalCol; // 当前 warp 负责的行

  // ── 计算均值：warp 内各线程负责步长为 32 的子集，最后 warp reduce ──
  float sum = 0.0f;
  for (int i = warp.thread_rank(); i < totalCol; i += warp.size()) {
    sum += x[i];
  }
  // cg::reduce(group, val, op)：
  //   group              ：参与规约的线程组（此处为 32 线程的 warp tile）
  //   val                ：每条 lane 贡献的局部值（各自的 sum）
  //   cg::plus<float>{}  ：规约操作符，对所有 lane 的 val 求和
  //   底层实现（编译器自动展开，等价于 common.cuh 的 warpReduceSum）：
  //     for (int offset = 16; offset > 0; offset >>= 1)
  //       val += __shfl_xor_sync(0xffffffff, val, offset);
  //     offset = 16,8,4,2,1 → 5 轮蝶形；lane i 与 lane(i XOR offset) 互换并累加
  //     每轮交换对称，5 轮后所有 lane 均持有全局和（all-reduce，非 reduce-to-lane-0）
  //   返回值             ：规约结果广播到 warp 内所有 lane（每条 lane 都拿到全局和）
  sum = cg::reduce(warp, sum, cg::plus<float>{});
  float m = sum / totalCol;
  // 仅 lane 0 写回全局内存，避免重复写
  if(warp.thread_rank() == 0 && mean != nullptr) {
    // __stcs(T* addr, T val)
    //   addr ：目标地址，必须是全局内存指针（__device__ 或 cudaMalloc 分配）
    //   val  ：要写入的值，类型 T 由 addr 的指针类型推导（此处 float*→float）
    //   返回值：void
    //
    //   展开为 PTX：st.global.cs.f32 [addr], val
    //     st      ：store 指令
    //     global  ：目标地址空间为全局内存
    //     cs      ：cache streaming hint，写入后尽快逐出 L1
    //   对应的其他变体（后缀不同，cache 策略不同）：
    //     __stwb  → st.global.wb   写回（默认）：写入留在 L1，可被后续读命中
    //     __stcg  → st.global.cg   绕过 L1，直接写 L2
    //     __stcs  → st.global.cs   写入但快速逐出 L1（streaming，本处所用）
    //     __stwt  → st.global.wt   write-through，直写到 L2/DRAM，不缓存
    //   mean/rstd 写完即不再使用，.cs 防止其占据 L1 挤走 weight/bias 等高复用数据
    __stcs(mean + idx, m);
  }

  // ── 计算 rstd：复用 m，第二遍扫描 ──────────────────────────────────
  sum = 0.0f;
  for (int i = warp.thread_rank(); i < totalCol; i += warp.size()) {
    float diff = x[i] - m;
    sum += diff * diff;
  }
  sum = cg::reduce(warp, sum, cg::plus<float>{}); // 同上，对 diff² 求全 warp 之和
  float s = rsqrtf(sum / totalCol + 1e-5f); // rsqrtf：CUDA 硬件倒数平方根指令
  if(warp.thread_rank() == 0 && rstd != nullptr) {
    __stcs(rstd + idx, s);
  }

  // ── 归一化 + 仿射变换，以 .cs hint 流式加载/存储 ────────────────────
  float* o = out + idx * totalCol;
  for (int c = warp.thread_rank(); c < totalCol; c += warp.size()) {
    // __ldcs：streaming load，绕过 L1，使 weight/bias 更易命中 cache
    float n = s * (__ldcs(x+c) - m);
    __stcs(o+c, n * weight[c] + bias[c]);
  }
}