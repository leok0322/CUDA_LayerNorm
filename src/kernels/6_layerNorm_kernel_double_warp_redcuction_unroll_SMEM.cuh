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

  if (row < totalRow) {
    // static_assert 和 __shared__ 数组大小均要求编译期常量：
    //   static_assert(cond)：cond 必须在编译期可求值，否则编译报错
    //   __shared__ T arr[N]：N 必须是编译期常量，CUDA 不支持动态大小的共享内存数组（VLA）
    //     原因：编译器在编译期就需要确定每个 block 的 SMEM 布局，分配固定偏移
    //
    // threadNum = blockDim.x 是运行时变量：
    //   blockDim 在 kernel 启动时由 host 传入，编译器无法在编译期知道其值
    //   因此 __shared__ scalar_t reduction[threadNum / 32] 会编译报错
    //   必须改用编译期已知的宏 BLOCK_SIZE（在 common.cuh 中定义为 1024）
    static_assert(BLOCK_SIZE / 32 == 32, "block的线程需要覆盖完整的warp且刚好是32个warp，以便第二次warp树形规约");
    __shared__ scalar_t reduction[BLOCK_SIZE / 32];
    __shared__ scalar_t smemWeight[4096];
    __shared__ scalar_t smemBias[4096];
    // 求均值
    scalar_t rowMean {};

#pragma Unroll URF
    //每个线程向量化加载
    for (scalar_i col {threadIDX}; col < totalCol / 4; col+=threadNum) {
      // totalCol是4的倍数，16字节对齐
      // reinterpret_cast cannot cast away const or other type qualifiers，A不能是const指针
      scalar_t4 vecA = reinterpret_cast<scalar_t4*>(&A[row * totalCol + col * 4])[0];
      rowMean += vecA.x;
      rowMean += vecA.y;
      rowMean += vecA.z;
      rowMean += vecA.w;

      scalar_t4 vecWeight = reinterpret_cast<scalar_t4*>(&weight[col * 4])[0];
      scalar_t4 vecBias = reinterpret_cast<scalar_t4*>(&bias[col * 4])[0];


      // 一个warp32个线程，每个线程之间间隔4个bank，
      smemWeight[col * 4] = vecWeight.x;
      smemWeight[col * 4 + 1] = vecWeight.y;
      smemWeight[col * 4 + 2] = vecWeight.z;
      smemWeight[col * 4 + 3] = vecWeight.w;

      smemBias[col * 4] = vecBias.x;
      smemBias[col * 4 + 1] = vecBias.y;
      smemBias[col * 4 + 2] = vecBias.z;
      smemBias[col * 4 + 3] = vecBias.w;
    }

#pragma Unroll
    // warp树形规约
    for (scalar_i i {16}; i>=1; i>>=1) {
      rowMean += __shfl_xor_sync(0xffffffff,rowMean,i,32);
    }
    if (threadIDX % 32 == 0) {
      reduction[threadIDX / 32] = rowMean;
    }

    __syncthreads();

    //warp0树形规约
    if (threadIDX < 32) {
      rowMean = reduction[threadIDX];
    }



    if (threadIDX < 32) {
#pragma Unroll
      for (scalar_i i {16}; i>=1; i>>=1) {
        rowMean += __shfl_xor_sync(0xffffffff,rowMean,i,32);
      }
    }

    if (threadIDX == 0) {
      rowMean /= static_cast<scalar_t>(totalCol);
      reduction[threadIDX] = rowMean;
    }

    __syncthreads();

    rowMean = reduction[0];
    mean[row] = rowMean;

    // 求均方差
    scalar_t rowVar {};
    scalar_t diff {};

#pragma Unroll URF
    // 向量化加载
    for (scalar_i col {threadIDX}; col< totalCol / 4; col+=threadNum) {
      scalar_t4 vecA {reinterpret_cast<scalar_t4*>(&A[row * totalCol + col * 4])[0]};
      diff = vecA.x - rowMean;
      rowVar += diff * diff;
      diff = vecA.y - rowMean;
      rowVar += diff * diff;
      diff = vecA.z - rowMean;
      rowVar += diff * diff;
      diff = vecA.w - rowMean;
      rowVar += diff * diff;
    }
    // warp树形规约
    for (scalar_i i {16}; i>=1; i>>=1) {
      rowVar += __shfl_xor_sync(0xffffffff,rowVar,i,32);
    }

    if (threadIDX % 32 == 0) {
      reduction[threadIDX / 32] = rowVar;
    }

    __syncthreads();

    // warp0树形规约
    if (threadIDX < 32) {
      rowVar = reduction[threadIDX];
    }

    __syncthreads();

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
      reduction[threadIDX] = rowVar;
    }

    __syncthreads();

    rowVar = reduction[0];
    rstd[row] = rowVar;

#pragma Unroll URF
    // 对行元素进行归一化
    for (scalar_i col {threadIDX}; col< totalCol / 4; col+=threadNum) {
      scalar_t4 vecA {reinterpret_cast<scalar_t4*>(&A[row * totalCol + col * 4])[0]};
      //smemWeight、smemBias没有任何复用，GMEM加载没有少，反而多了一次从SMEM加载的开销。
      vecA.x = smemWeight[col * 4] * ((vecA.x - rowMean) * rowVar) + smemBias[col * 4];
      vecA.y = smemWeight[col * 4 + 1] * ((vecA.y - rowMean) * rowVar) + smemBias[col * 4 + 1];
      vecA.z = smemWeight[col * 4 + 2] * ((vecA.z - rowMean) * rowVar) + smemBias[col * 4 + 2];
      vecA.w = smemWeight[col * 4 + 3] * ((vecA.w - rowMean) * rowVar) + smemBias[col * 4 + 3];
      reinterpret_cast<scalar_t4*>(&out[row * totalCol + col * 4])[0] = vecA;
    }
  }
}

