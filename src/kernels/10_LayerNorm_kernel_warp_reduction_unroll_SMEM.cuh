# pragma once

#include "common.cuh"
#include <cuda_runtime.h>


template <typename scalar_t, typename scalar_t4, typename scalar_i>
__global__ void LayerNorm_kernel_warp_reduction_unroll_SMEM(const scalar_i totalRow, const scalar_i totalCol, scalar_t *A,
                scalar_t *out, scalar_t* mean, scalar_t* rstd, scalar_t* weight, scalar_t* bias) {
  // 该线程负责的行
  scalar_i row { blockIdx.y * blockDim.y + threadIdx.y };
  // 该线程的线程号
  scalar_i threadIDX { threadIdx.x };
  // block的所有线程数
  scalar_i threadNum { blockDim.x };
  // 二级规约的线程数
  static_assert(WARP_SIZE == 32, "x维度32个线程，覆盖一个warp");


  // __align__(16) 只做一件事：令编译器把该变量的基地址放在 16 字节对齐的位置，
  // 不影响数组内部布局，也不影响每次访问的偏移计算。
  //
  // 为什么需要它：
  //   编译器生成 st.shared.v4.f32（128-bit 合并 store）的条件是
  //   绝对地址 = 基地址 + 偏移 均为 16 的倍数。
  //   访问 smemWeight[col][0] 时偏移 = col×5×4 = col×20 字节，
  //   col×20 不总是 16 的倍数（gcd(20,16)=4），因此此布局无法触发合并指令；
  //   __align__(16) 在这里仅保证基地址对齐，对合并指令无实际帮助。
  //
  // 当前文件中是否必须：
  //   smemWeight 前方的 reduction[BLOCK_SIZE_Y_SMEM][32] 占
  //   4×32×4 = 512 字节（512 是 16 的倍数），不写 __align__(16) 时
  //   基地址也恰好 16 字节对齐，实际效果相同。
  //   保留它是防御性写法：防止将来在 smemWeight 前插入奇数大小的 SMEM 变量
  //   导致基地址偏移，破坏潜在的向量化对齐假设。
  // __shared__ __align__(16) scalar_t smemWeight[MAX_TOTALCOL / 4][5];
  // __shared__ __align__(16) scalar_t smemBias[MAX_TOTALCOL / 4][5];

  // 动态SMEM：单一声明，手动三路切分
  // 布局：[smemWeight: (totalCol/4)×5] [smemBias: (totalCol/4)×5] [smemA: BLOCK_SIZE_Y_SMEM×(totalCol/4)×5]
  // dynSmem = sizeof(scalar_t) * (totalCol/4) * 5 * (2 + BLOCK_SIZE_Y_SMEM)
  extern __shared__ __align__(16) scalar_t smem[];
  scalar_t (*smemWeight)[5] = reinterpret_cast<scalar_t(*)[5]>(smem);
  scalar_t (*smemBias)[5]   = smemWeight + totalCol / 4;
  //                          ↑ 偏移 totalCol/4 个 [5] 块，即跳过 smemWeight 的全部元素
  // smemA 紧接 smemBias 之后，flat index: [y][col][k] → y*(totalCol/4)*5 + col*5 + k
  scalar_t *smemA = reinterpret_cast<scalar_t*>(smemBias + totalCol / 4);

  if (row < totalRow) {
    // 求均值
    scalar_t rowMean {};

#pragma Unroll URF
    //每个线程向量化加载
    for (scalar_i col {threadIDX}; col < totalCol / 4; col+=threadNum) {
      // totalCol是4的倍数，16字节对齐
      // reinterpret_cast cannot cast away const or other type qualifiers，A不能是const指针
      scalar_t4 vecA = reinterpret_cast<scalar_t4*>(&A[row * totalCol + col * 4])[0];
      scalar_t4 vecWeight = reinterpret_cast<scalar_t4*>(&weight[col * 4])[0];
      scalar_t4 vecBias = reinterpret_cast<scalar_t4*>(&bias[col * 4])[0];

      rowMean += vecA.x;
      rowMean += vecA.y;
      rowMean += vecA.z;
      rowMean += vecA.w;

      smemA[threadIdx.y * totalCol / 4 * 5 + col * 5] = vecA.x;
      smemA[threadIdx.y * totalCol / 4 * 5 + col * 5 + 1] = vecA.y;
      smemA[threadIdx.y * totalCol / 4 * 5 + col * 5 + 2] = vecA.z;
      smemA[threadIdx.y * totalCol / 4 * 5 + col * 5 + 3] = vecA.w;



      if (threadIdx.y == 0) {
        // ── bank conflict 分析 ────────────────────────────────────────────────
        // SMEM 有 32 个 bank，每 bank 宽 4 字节，bank 编号 = 元素下标 % 32
        // 线程 t 写 smemWeight[t*4]，对应 bank = (t*4) % 32：
        //   t=0→bank0, t=1→bank4, ..., t=7→bank28（前 8 线程无冲突）
        //   t=8→bank0（与 t=0 冲突），t=16→bank0，t=24→bank0 → 4-way bank conflict
        //
        // ── 128-bit 合并指令说明 ──────────────────────────────────────────────
        // 编译器可将下方 4 次连续标量 store 合并为一条 st.shared.v4.f32 指令。
        // 合并的前提：
        //   1. 地址连续：4 次 store 依次写 [col*4], [col*4+1], [col*4+2], [col*4+3]，
        //               中间无间隔，满足连续性要求
        //   smemWeight[col*4] 的字节偏移 = 元素下标 × sizeof(float)
        //                                = (col×4) × 4字节 = col×16 字节
        //   2. 16 字节对齐：smemWeight[col*4] 的字节偏移 = col*4*4 = col*16，
        //               无论 col 取何值，偏移量始终是 16 的倍数，满足对齐要求
        //   3. 编译器可静态验证以上两点（下标表达式在编译期可分析）


        // 合并的效果：4 条指令 → 1 条指令，降低指令流水压力，但：
        //   bank conflict 是多线程之间的问题，128-bit store 让单线程一次占 4 个
        //   bank，并不能消除不同线程争用同一 bank 的冲突，4-way conflict 依然存在
        // smemWeight[col * 4] = vecWeight.x;
        // smemWeight[col * 4 + 1] = vecWeight.y;
        // smemWeight[col * 4 + 2] = vecWeight.z;
        // smemWeight[col * 4 + 3] = vecWeight.w;
        //
        // smemBias[col * 4] = vecBias.x;
        // smemBias[col * 4 + 1] = vecBias.y;
        // smemBias[col * 4 + 2] = vecBias.z;
        // smemBias[col * 4 + 3] = vecBias.w;

        smemWeight[col][0] = vecWeight.x;
        smemWeight[col][1] = vecWeight.y;
        smemWeight[col][2] = vecWeight.z;
        smemWeight[col][3] = vecWeight.w;

        smemBias[col][0] = vecBias.x;
        smemBias[col][1] = vecBias.y;
        smemBias[col][2] = vecBias.z;
        smemBias[col][3] = vecBias.w;
      }
    }


#pragma Unroll
    // warp树形规约
    for (scalar_i i {16}; i>=1; i>>=1) {
      // 所有线程得到的rowMean完全相同
      // 如果total小于32*4,rowMean为0，不影响rowMean的求和
      rowMean += __shfl_xor_sync(0xffffffff,rowMean,i,32);
    }

    rowMean /= totalCol;

    if (threadIDX == 0) {
      __stcs(mean+row,rowMean);
    }


    // SMEM加载同步，需要在计算rowVar之前执行
    __syncthreads();

    // 求均方差
    scalar_t rowVar {};
    scalar_t diff {};

#pragma Unroll URF
    // 向量化加载
    for (scalar_i col {threadIDX}; col< totalCol / 4; col+=threadNum) {
      // scalar_t4 vecA {reinterpret_cast<scalar_t4*>(&A[row * totalCol + col * 4])[0]};
      // diff = vecA.x - rowMean;
      // rowVar += diff * diff;
      // diff = vecA.y - rowMean;
      // rowVar += diff * diff;
      // diff = vecA.z - rowMean;
      // rowVar += diff * diff;
      // diff = vecA.w - rowMean;
      // rowVar += diff * diff;


      scalar_t x  = smemA[threadIdx.y * totalCol / 4 * 5 + col * 5];
      scalar_t y  = smemA[threadIdx.y * totalCol / 4 * 5 + col * 5 + 1];
      scalar_t z  = smemA[threadIdx.y * totalCol / 4 * 5 + col * 5 + 2];
      scalar_t w  = smemA[threadIdx.y * totalCol / 4 * 5 + col * 5 + 3];

      diff = x - rowMean;
      rowVar += diff * diff;
      diff = y - rowMean;
      rowVar += diff * diff;
      diff = z - rowMean;
      rowVar += diff * diff;
      diff = w - rowMean;
      rowVar += diff * diff;

    }
    // warp树形规约
    for (scalar_i i {16}; i>=1; i>>=1) {
      // 所有线程得到的rowVar完全相同
      // 如果total小于32*4,rowVar为0，不影响rowVar的求和
      rowVar += __shfl_xor_sync(0xffffffff,rowVar,i,32);
    }

    rowVar /= static_cast<scalar_t>(totalCol);
    // + 1e-5f（ε）：防止方差为 0 时 rsqrtf(0) = Inf，导致输出 NaN
    // 当输入行所有元素相同时（如全 0、全 1），(xᵢ - μ)² 均为 0，方差精确为 0
    // ε 是极小正数，对归一化结果的精度影响可忽略，但保证了数值稳定性
    rowVar = static_cast<scalar_t>(rsqrtf(static_cast<float>(rowVar) + 1e-5f));

    if (threadIDX == 0) {
      // rstd[row] = rowVar;
      __stcs(rstd + row,rowVar);
    }

#pragma Unroll URF
    // 对行元素进行归一化
    for (scalar_i col {threadIDX}; col< totalCol / 4; col+=threadNum) {
      scalar_t4 vecA {};

      // 每行元素复用
      vecA.x  = smemA[threadIdx.y * totalCol / 4 * 5 + col * 5];
      vecA.y  = smemA[threadIdx.y * totalCol / 4 * 5 + col * 5 + 1];
      vecA.z  = smemA[threadIdx.y * totalCol / 4 * 5 + col * 5 + 2];
      vecA.w  = smemA[threadIdx.y * totalCol / 4 * 5 + col * 5 + 3];

      //smemWeight、smemBias没有任何复用，GMEM加载没有少，反而多了一次从SMEM加载的开销。
      vecA.x = smemWeight[col][0] * ((vecA.x - rowMean) * rowVar) + smemBias[col][0];
      vecA.y = smemWeight[col][1] * ((vecA.y - rowMean) * rowVar) + smemBias[col][1];
      vecA.z = smemWeight[col][2] * ((vecA.z - rowMean) * rowVar) + smemBias[col][2];
      vecA.w = smemWeight[col][3] * ((vecA.w - rowMean) * rowVar) + smemBias[col][3];
      reinterpret_cast<scalar_t4*>(&out[row * totalCol + col * 4])[0] = vecA;
    }
  }
}

