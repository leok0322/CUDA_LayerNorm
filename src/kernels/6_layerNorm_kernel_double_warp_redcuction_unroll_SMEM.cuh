# pragma once

#include "common.cuh"
#include <cuda_runtime.h>


template <typename scalar_t, typename scalar_t4, typename scalar_i>
__global__ void LayerNorm_kernel_double_warp_reduction_unroll_SMEM(const scalar_i totalRow, const scalar_i totalCol, scalar_t *A,
                scalar_t *out, scalar_t* mean, scalar_t* rstd, scalar_t* weight, scalar_t* bias) {
  // 该线程负责的行
  scalar_i row { blockIdx.y * blockDim.y + threadIdx.y };
  // 该线程的线程号
  scalar_i threadIDX { threadIdx.x };
  // block的所有线程数
  scalar_i threadNum { blockDim.x };
  // 二级规约的线程数
  static_assert(BLOCK_SIZE_X_SMEM % 32 == 0, "block的线程需要覆盖完整的warp");
  scalar_i warp0threadNum {BLOCK_SIZE_X_SMEM / 32};


  // 因为一个block处理多行，所以静态SMEM需要放到循环的外面

  // static_assert 和 __shared__ 数组大小均要求编译期常量：
  //   static_assert(cond)：cond 必须在编译期可求值，否则编译报错
  //   __shared__ T arr[N]：N 必须是编译期常量，CUDA 不支持动态大小的共享内存数组（VLA）
  //     原因：编译器在编译期就需要确定每个 block 的 SMEM 布局，分配固定偏移
  //
  // threadNum = blockDim.x 是运行时变量：
  //   blockDim 在 kernel 启动时由 host 传入，编译器无法在编译期知道其值
  //   因此 __shared__ scalar_t reduction[threadNum / 32] 会编译报错
  //   必须改用编译期已知的宏 BLOCK_SIZE（在 common.cuh 中定义为 1024）
  // 因为是二级warp，reduction的长度固定是32，如果
  __shared__ scalar_t reduction[BLOCK_SIZE_Y_SMEM][32];


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
  // extern __shared__ 是翻译单元级全局符号：同一名称只能声明为一种类型。
  // 若写 extern __shared__ scalar_t smem[]，float/double 两次实例化产生两种类型声明 → 冲突。
  // 解法：始终用 char（类型唯一），再 reinterpret_cast 到目标类型。
  extern __shared__ __align__(16) char smem_raw[];
  scalar_t (*smemWeight)[5] = reinterpret_cast<scalar_t(*)[5]>(smem_raw);
  scalar_t (*smemBias)[5]   = smemWeight + totalCol / 4;
  //                          ↑ 偏移 totalCol/4 个 [5] 块，即跳过 smemWeight 的全部元素
  // smemA 紧接 smemBias 之后，flat index: [y][col][k] → y*(totalCol/4)*5 + col*5 + k
  scalar_t *smemA = reinterpret_cast<scalar_t*>(smemBias + totalCol / 4);


  scalar_t rowMean {};


  if (row < totalRow) {
    // 求均值

#pragma Unroll URF
    //每个线程向量化加载
    for (scalar_i col {threadIDX}; col < totalCol / 4; col+=threadNum) {
      // totalCol是4的倍数，16字节对齐
      // reinterpret_cast cannot cast away const or other type qualifiers，A不能是const指针
      scalar_t4 vecA    = reinterpret_cast<scalar_t4*>(&A[row * totalCol + col * 4])[0];

      // // A：每行读一次不复用，.cs hint 令数据加载后快速逐出 L1，避免污染 cache
      // scalar_t4 vecA;
      // asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];"
      //              : "=f"(vecA.x), "=f"(vecA.y), "=f"(vecA.z), "=f"(vecA.w)
      //              : "l"(&A[row * totalCol + col * 4]));
      // // weight/bias：所有 block 共享同一份，.ca hint 缓存到 L1+L2，供后续 block 复用
      // scalar_t4 vecWeight;
      // asm volatile("ld.global.ca.v4.f32 {%0,%1,%2,%3}, [%4];"
      //              : "=f"(vecWeight.x), "=f"(vecWeight.y), "=f"(vecWeight.z), "=f"(vecWeight.w)
      //              : "l"(&weight[col * 4]));
      // scalar_t4 vecBias;
      // asm volatile("ld.global.ca.v4.f32 {%0,%1,%2,%3}, [%4];"
      //              : "=f"(vecBias.x), "=f"(vecBias.y), "=f"(vecBias.z), "=f"(vecBias.w)
      //              : "l"(&bias[col * 4]));

      rowMean += vecA.x;
      rowMean += vecA.y;
      rowMean += vecA.z;
      rowMean += vecA.w;

      smemA[threadIdx.y * totalCol / 4 * 5 + col * 5] = vecA.x;
      smemA[threadIdx.y * totalCol / 4 * 5 + col * 5 + 1] = vecA.y;
      smemA[threadIdx.y * totalCol / 4 * 5 + col * 5 + 2] = vecA.z;
      smemA[threadIdx.y * totalCol / 4 * 5 + col * 5 + 3] = vecA.w;



      if (threadIdx.y == 0) {
        scalar_t4 vecWeight = reinterpret_cast<scalar_t4*>(&weight[col * 4])[0];
        scalar_t4 vecBias   = reinterpret_cast<scalar_t4*>(&bias[col * 4])[0];
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


        // 无bank conflict
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
      rowMean += __shfl_xor_sync(0xffffffff,rowMean,i,32);
    }
    if (threadIDX % 32 == 0) {
      reduction[threadIdx.y][threadIDX / 32] = rowMean;
    }
  }


  // SMEM的同步可以放在这里，因为最后计算的时候才会用到
  __syncthreads();

  if (row < totalRow) {
    // warp0二级规约，
    if (threadIDX < 32) {
      rowMean = threadIDX < warp0threadNum ? reduction[threadIdx.y][threadIDX]: static_cast<scalar_t>(0.f);
    }



    if (threadIDX < 32) {
#pragma Unroll
      for (scalar_i i {16}; i>=1; i>>=1) {
        rowMean += __shfl_xor_sync(0xffffffff,rowMean,i,32);
      }
    }

    if (threadIDX == 0) {
      rowMean /= static_cast<scalar_t>(totalCol);
      reduction[threadIdx.y][threadIDX] = rowMean;
    }
  }


  __syncthreads();

  scalar_t rowVar {};
  scalar_t diff {};
  if (row < totalRow) {
    rowMean = reduction[threadIdx.y][0];
    if (threadIDX == 0) {
      // mean[row] = rowMean;
      __stcs(mean+row,rowMean);
    }

    // 求均方差

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
      rowVar += __shfl_xor_sync(0xffffffff,rowVar,i,32);
    }

    if (threadIDX % 32 == 0) {
      reduction[threadIdx.y][threadIDX / 32] = rowVar;
    }
  }

  __syncthreads();

  if (row < totalRow) {
    // warp0树形规约
    if (threadIDX < 32) {
      rowVar = threadIDX < warp0threadNum ? reduction[threadIdx.y][threadIDX] : static_cast<scalar_t>(0.f);
    }

    // 一个warp不需要同步
    // __syncthreads();

    if (threadIDX < 32) {
#pragma Unroll
      for (scalar_i i {16}; i>=1; i>>=1) {
        rowVar += __shfl_xor_sync(0xffffffff,rowVar,i,32);
      }
    }

    if (threadIDX == 0) {
      rowVar /= static_cast<scalar_t>(totalCol);
      // + 1e-5f（ε）：防止方差为 0 时 rsqrtf(0) = Inf，导致输出 NaN
      // 当输入行所有元素相同时（如全 0、全 1），(xᵢ - μ)² 均为 0，方差精确为 0
      // ε 是极小正数，对归一化结果的精度影响可忽略，但保证了数值稳定性
      rowVar = static_cast<scalar_t>(rsqrtf(static_cast<float>(rowVar) + 1e-5f));
      reduction[threadIdx.y][threadIDX] = rowVar;
    }
  }

  __syncthreads();
  if (row < totalRow) {
    rowVar = reduction[threadIdx.y][0];

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
      // output：写一次即完成，.cs hint 写入后快速逐出 L1，减少 cache 污染
      // asm volatile("st.global.cs.v4.f32 [%0], {%1,%2,%3,%4};"
      //              : : "l"(&out[row * totalCol + col * 4]),
      //                  "f"(vecA.x), "f"(vecA.y), "f"(vecA.z), "f"(vecA.w));
    }
  }
}

