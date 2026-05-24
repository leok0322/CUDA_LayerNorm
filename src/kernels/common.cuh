#pragma once

#include <cuda_runtime.h>

#ifndef BLOCK_SIZE_X
#define BLOCK_SIZE_X 1024
#endif


#ifndef BLOCK_SIZE_X_kernel3
#define BLOCK_SIZE_X_kernel3 512
#endif


#ifndef BLOCK_SIZE_X_SMEM
#define BLOCK_SIZE_X_SMEM 512
#endif


#ifndef BLOCK_SIZE_Y_SMEM
#define BLOCK_SIZE_Y_SMEM 2
#endif

#ifndef MAX_TOTALCOL
#define MAX_TOTALCOL 4096
#endif


#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

constexpr int URF {4};

// ── x128：4 个 float 打包为 128-bit 向量，用于向量化读写 ─────────────────────
// alignas(16) 保证结构体按 16 字节对齐，满足 128-bit 指令的地址对齐要求
struct alignas(16) x128 {
    static constexpr int size = 4; // 4 × float = 128-bit
    float data[size];

    __device__ float& operator[](int i) { return data[i]; }
    __device__ const float& operator[](int i) const { return data[i]; }
};

// 标准 128-bit 全局内存加载（经 L1/L2 cache，适合重复访问的数据如 weight/bias）
__device__ inline x128 load128(const float* addr) {
    x128 x;
    *reinterpret_cast<float4*>(x.data) = *reinterpret_cast<const float4*>(addr);
    return x;
}

// Streaming 128-bit 加载（.cs hint：绕过 L1 只进 L2，保留 cache 给其他高复用数据）
__device__ inline x128 load128cs(const float* addr) {
    x128 x;
    asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(x.data[0]), "=f"(x.data[1]), "=f"(x.data[2]), "=f"(x.data[3])
                 : "l"(addr));
    return x;
}

// Streaming 128-bit 存储（.cs hint：写入后数据尽快逐出 L1，减少 cache 污染）
__device__ inline void store128cs(float* addr, x128 val) {
    asm volatile("st.global.cs.v4.f32 [%0], {%1,%2,%3,%4};"
                 : : "l"(addr), "f"(val.data[0]), "f"(val.data[1]), "f"(val.data[2]), "f"(val.data[3]));
}

// warp 内全归约求和（all-reduce）：蝶形 __shfl_xor_sync，5 轮，结果广播到全部 32 条 lane
//
// 与 __shfl_down_sync 的关键区别：
//   __shfl_down_sync：lane i 读 lane(i+offset)，对称性缺失 → 仅 lane 0 得到完整和
//   __shfl_xor_sync ：lane i 与 lane(i XOR offset) 互换  → 每轮交换均对称，
//                     5 轮后所有 lane 都持有相同的全局和
//
// 调用前确保整个 warp（mask=0xffffffff）均到达此指令，不能置于发散分支内
__device__ inline float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}