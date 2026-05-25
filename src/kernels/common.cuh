#pragma once

#include <cuda_runtime.h>

#ifndef BLOCK_SIZE_X
#define BLOCK_SIZE_X 1024
#endif


#ifndef BLOCK_SIZE_Y
#define BLOCK_SIZE_Y 2
#endif

#ifndef BLOCK_SIZE_X_kernel3
#define BLOCK_SIZE_X_kernel3 512
#endif


#ifndef BLOCK_SIZE_X_SMEM
#define BLOCK_SIZE_X_SMEM 256
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
    // asm volatile(PTX指令字符串  : 输出操作数  : 输入操作数);
    //
    // PTX 指令：ld.global.cs.v4.f32 {dst0,dst1,dst2,dst3}, [src]
    //   ld          ：load 指令
    //   global      ：地址空间为全局内存（cudaMalloc / __device__ 变量）
    //   cs          ：cache streaming hint，数据加载后在 L1 快速逐出（不长期占用 L1）
    //   v4.f32      ：向量宽度 4，元素类型 float（单次指令传输 4×4=16 字节 / 128-bit）
    //   {%0,%1,%2,%3}：4 个目标寄存器，对应输出操作数列表中的 %0~%3
    //   [%4]        ：源地址寄存器，取自输入操作数列表中的 %4
    //
    // 输出操作数（冒号后第一段）：
    //   "=f"(x.data[0])  → %0，"=f" 表示写入（=）、float 寄存器（f），绑定到 x.data[0]
    //   "=f"(x.data[1])  → %1，同上，绑定到 x.data[1]
    //   "=f"(x.data[2])  → %2
    //   "=f"(x.data[3])  → %3
    //
    // 输入操作数（冒号后第二段）：
    //   "l"(addr)        → %4，"l" 表示 64-bit 整数寄存器（指针），绑定到 addr
    //
    // volatile：禁止编译器将此 asm 块重排或消除（即使输出值未被使用）
    asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(x.data[0]), "=f"(x.data[1]), "=f"(x.data[2]), "=f"(x.data[3])
                 : "l"(addr));
    return x;
}

// Streaming 128-bit 存储（.cs hint：写入后数据尽快逐出 L1，减少 cache 污染）
__device__ inline void store128cs(float* addr, x128 val) {
    // PTX 指令：st.global.cs.v4.f32 [dst], {src0,src1,src2,src3}
    //   st          ：store 指令
    //   global      ：目标地址空间为全局内存
    //   cs          ：cache streaming hint，写入后 L1 快速逐出（write-once 数据不应长占 L1）
    //   v4.f32      ：一次写 4×float = 128-bit
    //   [%0]        ：目标地址，取自输入操作数 %0
    //   {%1,%2,%3,%4}：4 个源寄存器，对应输入操作数 %1~%4
    //
    // 操作数列表（输出段为空，故两个冒号紧邻）：
    //   ": :"        ：输出操作数为空（store 无 C++ 侧输出变量）
    //   "l"(addr)          → %0，64-bit 指针寄存器
    //   "f"(val.data[0])   → %1，float 寄存器（只读，无 = 前缀）
    //   "f"(val.data[1])   → %2
    //   "f"(val.data[2])   → %3
    //   "f"(val.data[3])   → %4
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