# pragma once

#include "common.cuh"
#include <cuda_runtime.h>


template <typename scalar_t, typename scalar_t4, typename scalar_i>
__global__ void LayerNorm_kernel_welford_double_warp_reduction_unroll(const scalar_i totalRow, const scalar_i totalCol, scalar_t *A,
                scalar_t *out, scalar_t* mean, scalar_t* rstd, const scalar_t* weight, const scalar_t* bias) {

  // 该线程负责的行
  scalar_i row { blockDim.y * blockIdx.y + threadIdx.y };
  // block中的线程号
  scalar_i threadIDX { threadIdx.x };
  // block中总线程数
  scalar_i threadNum { blockDim.x };

  // 判断行数是否越界
  if (row < totalRow) {

    // 静态SMEM分配
    static_assert(BLOCK_SIZE / 32 == 32, "block的线程需要是1024，刚好覆盖两次warp规约");
    __shared__ scalar_t reductionMean[BLOCK_SIZE / 32];
    __shared__ scalar_t reductionVar[BLOCK_SIZE / 32];
    __shared__ uint reductionCount[BLOCK_SIZE / 32];
    scalar_t rowMean {};
    scalar_t rowMeanOld {};
    scalar_t rowVar {};
    scalar_t rowVarOld {};
    scalar_t elementDiff {};
    scalar_t meanDiff {};
    uint count {};
    uint countXor {};
    scalar_t rowMeanXor {};
    scalar_t rowVarXor {};

#pragma unroll URF
    for (scalar_i col {threadIDX}; col < totalCol / 4; col += threadNum) {
      count+=4;

      // 16字节对齐
      scalar_t4 vecA { reinterpret_cast<scalar_t4*>(&A[row * totalCol + col * 4])[0] };
      rowMean += vecA.x;
      rowMean += vecA.y;
      rowMean += vecA.z;
      rowMean += vecA.w;

      rowMean /= 4;

      elementDiff  = vecA.x - rowMean;
      rowVar += elementDiff * elementDiff;
      elementDiff  = vecA.y - rowMean;
      rowVar += elementDiff * elementDiff;
      elementDiff  = vecA.z - rowMean;
      rowVar += elementDiff * elementDiff;
      elementDiff  = vecA.w - rowMean;
      rowVar += elementDiff * elementDiff;

      if (col == threadIDX) {
        rowMeanOld = rowMean;
      }

      // 第一轮次为meanDiff=0
      meanDiff = rowMean - rowMeanOld;
      rowMean = rowMeanOld + meanDiff * 4 / count;
      rowVar += rowVarOld + meanDiff * meanDiff * (count - 4) * 4 / count;


      rowMeanOld = rowMean;
      rowVarOld = rowVar;
      rowMean = static_cast<scalar_t>(0);
      rowVar = static_cast<scalar_t>(0);
    }

    // 需要在 warp 归约前补充：
    rowMean = rowMeanOld;
    rowVar  = rowVarOld;

    // warp树形规约
    // ── 曾犯的错误：将 __shfl_xor_sync 放在 else 分支内 ──────────────────────
    // 错误写法：
    //   if (count_xor == 0) { meanDiff = 0; }
    //   else { meanDiff = __shfl_xor_sync(0xffffffff, rowMean, i, 32) - rowMean; }
    //
    // 原因：__shfl_xor_sync(mask=0xffffffff, ...) 要求 warp 内全部 32 条线程
    // 同时执行该指令。若同一 warp 内 count_xor 值不一致（部分线程=0，部分≠0），
    // 则线程发生分歧（divergence）：走 else 的线程调用 shfl 并等待对方，
    // 走 if 的线程永远不调用 shfl → 双方永久互等 → kernel 卡死（hang）。
    //
    // 正确做法：所有 __shfl_xor_sync 调用必须在任何条件分支之前无条件执行，
    // 保证 warp 内 32 条线程同步到达每一条 shfl 指令，再根据 count 决定是否合并。
#pragma unroll
    for (scalar_i i {16}; i>=1; i>>=1) {
      countXor   = __shfl_xor_sync(0xffffffff, count,   i, 32);
      rowMeanXor = __shfl_xor_sync(0xffffffff, rowMean, i, 32);
      rowVarXor  = __shfl_xor_sync(0xffffffff, rowVar,  i, 32);

      // ── 为什么 count 可能为 0 ────────────────────────────────────────────────
      // 本地循环按步长 threadNum(=BLOCK_SIZE=1024) 分配列：
      //   thread t 处理 col = t, t+threadNum, t+2*threadNum, ...
      // 若 totalCol/4 < threadNum（如 128/4=32 < 1024），
      // 则只有 threadIDX ∈ [0, totalCol/4) 的线程处理了数据，count > 0；
      // threadIDX ≥ totalCol/4 的线程一次都没进循环，count 保持初始值 0。
      //
      // ── 三种情况的处理逻辑 ──────────────────────────────────────────────────
      // case 1: count == 0          → 本线程没有数据，不能作为合并基准，跳过
      // case 2: count != 0，countXor == 0 → 对端线程没有数据，无需合并，跳过
      // case 3: count != 0，countXor != 0 → 双方都有数据，执行 Welford 批量合并
      if (count != 0  && countXor != 0) {
        meanDiff = rowMeanXor - rowMean;
        rowMean += meanDiff * countXor / (count + countXor);
        rowVar += rowVarXor + meanDiff * meanDiff  * count * countXor / (count + countXor);
        count += countXor;
      }
    }

    if (threadIDX % 32 == 0) {
      reductionMean[threadIDX / 32] = rowMean;
      reductionVar[threadIDX / 32] = rowVar;
      reductionCount[threadIDX / 32] = count;
    }

    __syncthreads();

    if (threadIDX < 32) {
      rowMean = reductionMean[threadIDX];
      rowVar = reductionVar[threadIDX];
      count = reductionCount[threadIDX];
    }

    // 第二轮 warp 规约（同样遵循：shfl 必须无条件执行，见第一轮注释）
#pragma unroll
    for (scalar_i i {16}; i>=1; i>>=1) {
      countXor = __shfl_xor_sync(0xffffffff,count,i,32);
      rowMeanXor = __shfl_xor_sync(0xffffffff,rowMean,i,32);
      rowVarXor = __shfl_xor_sync(0xffffffff,rowVar,i,32);

      // ── 为什么 count 可能为 0 ────────────────────────────────────────────────
      // 上方仅 threadIDX < 32 的线程从 SMEM 加载了数据（32 个 warp-leader）；
      // threadIDX ≥ 32 的线程仍持有第一轮结束时的旧值，其 count 通常为 0。
      // 即使在 threadIDX < 32 内，若对应 warp 在第一轮中未处理任何列，
      // 其写入 SMEM 的 count 同样为 0（见第一轮注释）。
      //
      // ── 三种情况的处理逻辑 ──────────────────────────────────────────────────
      // case 1: count == 0          → 本线程没有数据，不能作为合并基准，跳过
      // case 2: count != 0，countXor == 0 → 对端线程没有数据，无需合并，跳过
      // case 3: count != 0，countXor != 0 → 双方都有数据，执行 Welford 批量合并
      if (count != 0  && countXor != 0) {
        meanDiff = rowMeanXor - rowMean;
        rowMean += meanDiff * countXor / (count + countXor);
        rowVar += rowVarXor + meanDiff * meanDiff  * count * countXor / (count + countXor);
        count += countXor;
      }
    }

    if (threadIDX  == 0) {
      reductionMean[threadIDX] = rowMean;
      reductionVar[threadIDX] = rowVar;
      reductionCount[threadIDX] = count;
    }

    __syncthreads();

    rowMean = reductionMean[0];
    rowVar = reductionVar[0];
    count = reductionCount[0];

    rowVar  /= static_cast<scalar_t>(count);
    rowVar = static_cast<scalar_t>(rsqrtf(static_cast<float>(rowVar) + 1e-5f));

    mean[row] = rowMean;
    rstd[row] = rowVar;

#pragma unroll
    for (scalar_i i { threadIDX}; i < totalCol / 4; i+=threadNum) {
      scalar_t4 vecA { reinterpret_cast<scalar_t4*>(&A[row * totalCol + i * 4])[0] };
      vecA.x = weight[i * 4]  * (vecA.x - rowMean) * rowVar + bias[i * 4];
      vecA.y = weight[i * 4 + 1]  * (vecA.y - rowMean) * rowVar + bias[i * 4 + 1];
      vecA.z = weight[i * 4 + 2]  * (vecA.z - rowMean) * rowVar + bias[i * 4 + 2];
      vecA.w = weight[i * 4 + 3]  * (vecA.w - rowMean) * rowVar + bias[i * 4 + 3];
      reinterpret_cast<scalar_t4*>(&out[row * totalCol + i * 4])[0] = vecA;
    }
  }
}