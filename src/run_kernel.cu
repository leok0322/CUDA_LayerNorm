#include "kernels.cuh"
#include "error_check.cuh"
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda/cmath>


#ifndef LAYERNORM_VARIANT
#define LAYERNORM_VARIANT 6
#endif



// torch::Tensor：句柄（handle），不是数据本身
//
//   内部结构：
//     torch::Tensor
//       └─ intrusive_ptr<TensorImpl>   ← 引用计数智能指针，类似 shared_ptr 但侵入式
//            └─ TensorImpl
//                 ├─ Storage           ← 实际 GPU 内存块（data_ptr 指向这里）
//                 ├─ sizes[]           ← 各维度大小，如 [h, w]
//                 ├─ strides[]         ← 各维度步长（字节数），决定内存布局
//                 ├─ dtype             ← 元素类型（float / double 等）
//                 ├─ device            ← cuda:0 / cpu 等
//                 └─ ref_count         ← 引用计数，降为 0 时释放 Storage
//
//   拷贝语义（浅拷贝）：
//     torch::Tensor b = a;   // 只复制句柄，ref_count++，a 和 b 指向同一块 GPU 内存
//     按值传参同理：softmax_cu(torch::Tensor x) 传入时拷贝句柄，开销 O(1)
//
//   深拷贝（需要独立副本时）：
//     torch::Tensor b = a.clone();   // 分配新内存并复制数据，ref_count 各自独立
torch::Tensor LayerNorm_cu(torch::Tensor x) {
  // auto 推断为 torch::Tensor，与 x 共享同一 TensorImpl 结构（shape/dtype/device），
  // 但 empty_like 分配独立的 Storage（新的 GPU 内存块），ref_count 从 1 开始
  auto out = torch::empty_like(x);

  int64_t dim = x.dim();
  if (dim == 2) {
    int64_t totalRow = x.size(0);
    int64_t totalCol = x.size(1);

    // mean, rstd：每行一个标量（行数维度）
    // weight, bias：每列一个标量（列数维度），weight 初始化为 1，bias 初始化为 0
    auto mean   = torch::empty({totalRow}, x.options());
    auto rstd   = torch::empty({totalRow}, x.options());
    auto weight = torch::ones ({totalCol}, x.options());
    auto bias   = torch::zeros({totalCol}, x.options());


    #if LAYERNORM_VARIANT == 1
      dim3 grid(1, totalRow, 1);
      dim3 block(BLOCK_SIZE_X, 1, 1);
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "LayerNorm_cu", ([&] {
            LayerNorm_kernel_naive<scalar_t, int64_t><<<grid,block,0,0>>>(totalRow,totalCol,x.data_ptr<scalar_t>(),
              out.data_ptr<scalar_t>(),mean.data_ptr<scalar_t>(),rstd.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),bias.data_ptr<scalar_t>());
            cudaCheck(cudaGetLastError());
      }));
    #endif

    #if LAYERNORM_VARIANT == 2
    dim3 grid(1, totalRow, 1);
    dim3 block(BLOCK_SIZE_X, 1, 1);
    assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "LayerNorm_cu", ([&] {
      LayerNorm_kernel_double_warp_reduction<scalar_t, float4,int64_t><<<grid,block,0,0>>>(totalRow,totalCol,x.data_ptr<scalar_t>(),
          out.data_ptr<scalar_t>(),mean.data_ptr<scalar_t>(),rstd.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),bias.data_ptr<scalar_t>());
      cudaCheck(cudaGetLastError());
    }));
    #endif

    #if LAYERNORM_VARIANT == 3
      dim3 grid(1, totalRow, 1);
      dim3 block(BLOCK_SIZE_X, 1, 1);
      assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "LayerNorm_cu", ([&] {
        LayerNorm_kernel_double_warp_reduction_unroll<scalar_t, float4,int64_t><<<grid, block, 0, 0>>>(totalRow,totalCol,x.data_ptr<scalar_t>(),
          out.data_ptr<scalar_t>(),mean.data_ptr<scalar_t>(),rstd.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),bias.data_ptr<scalar_t>());
        cudaCheck(cudaGetLastError());
      }));
    #endif

    #if LAYERNORM_VARIANT == 4
      dim3 grid(1, totalRow, 1);
      dim3 block(BLOCK_SIZE_X, 1, 1);
      assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "LayerNorm_cu", ([&] {
        LayerNorm_kernel_welford_double_warp_reduction<scalar_t, float4,int64_t><<<grid, block, 0, 0>>>(totalRow,totalCol,x.data_ptr<scalar_t>(),
          out.data_ptr<scalar_t>(),mean.data_ptr<scalar_t>(),rstd.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),bias.data_ptr<scalar_t>());
        cudaCheck(cudaGetLastError());
      }));
    #endif

    #if LAYERNORM_VARIANT == 5
      dim3 grid(1, totalRow, 1);
      dim3 block(BLOCK_SIZE_X, 1, 1);
      assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "LayerNorm_cu", ([&] {
        LayerNorm_kernel_welford_double_warp_reduction_unroll<scalar_t, float4,int64_t><<<grid, block, 0, 0>>>(totalRow,totalCol,x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),mean.data_ptr<scalar_t>(),rstd.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),bias.data_ptr<scalar_t>());
        cudaCheck(cudaGetLastError());
      }));

    #endif
    #if LAYERNORM_VARIANT == 6
      dim3 grid(1, cuda::ceil_div(totalRow,BLOCK_SIZE_Y_SMEM), 1);
      dim3 block(BLOCK_SIZE_X_SMEM, BLOCK_SIZE_Y_SMEM, 1);
      assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");

      size_t dynSmem;
      switch (x.scalar_type()) {
        case at::ScalarType::Float: {
          constexpr size_t static_smem = sizeof(float) * BLOCK_SIZE_Y_SMEM * 32;
          // SMEM动态存储A的每一列元素和weight、bias以供复用，因为totalCol不是编译器常量，所以需要用动态smem
          dynSmem  = sizeof(float)  * (totalCol / 4) * 5 *  (2 + BLOCK_SIZE_Y_SMEM);
          constexpr size_t SM86_PER_BLOCK_MAX = 99 * 1024;
          if (static_smem + dynSmem > SM86_PER_BLOCK_MAX) {
            throw std::runtime_error(
                "[kernel6] dynSmem 超过 sm_86 per-block 上限 (99 KB)，请减小 totalCol 或 BLOCK_SIZE_Y_SMEM");
          }
          size_t maxDynamicDiff = static_smem + dynSmem < 48 * 1024 ? 48 * 1024 - (static_smem + dynSmem): 0;
          size_t maxDynamic {dynSmem + maxDynamicDiff};
          cudaCheck(cudaFuncSetAttribute(
            (const void*)LayerNorm_kernel_double_warp_reduction_unroll_SMEM<float, float4, int64_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(maxDynamic)));
          break;
        }
      }
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "LayerNorm_cu", ([&] {
        LayerNorm_kernel_double_warp_reduction_unroll_SMEM<scalar_t, float4, int64_t><<<grid, block, dynSmem,0>>>(totalRow,totalCol,x.data_ptr<scalar_t>(),
          out.data_ptr<scalar_t>(),mean.data_ptr<scalar_t>(),rstd.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),bias.data_ptr<scalar_t>());
        cudaCheck(cudaGetLastError());
      }));
    #endif

    #if LAYERNORM_VARIANT == 7
      const uint block_size = 128;
      const uint warps_per_block = block_size / 32;
      dim3 block(block_size, 1, 1);
      dim3 grid(cuda::ceil_div(totalRow, warps_per_block), 1, 1);

      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "LayerNorm_cu", ([&] {
        LayerNorm_forward_kernel_cg_warp<scalar_t, float4, int64_t><<<grid, block, 0, 0>>>(totalRow,totalCol,x.data_ptr<scalar_t>(),
          out.data_ptr<scalar_t>(),mean.data_ptr<scalar_t>(),rstd.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),bias.data_ptr<scalar_t>());
        cudaCheck(cudaGetLastError());
      }));
    #endif

    #if LAYERNORM_VARIANT == 8
        dim3 block(BLOCK_SIZE_X, 1, 1);
        dim3 grid(totalRow, 1, 1);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "LayerNorm_cu", ([&] {
          LayerNorm_forward_kernel_cg_block<scalar_t, float4, int64_t><<<grid, block, 0, 0>>>(totalRow,totalCol,x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),mean.data_ptr<scalar_t>(),rstd.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),bias.data_ptr<scalar_t>());
          cudaCheck(cudaGetLastError());
        }));
    #endif


    #if LAYERNORM_VARIANT == 9
        const uint block_y = 4;
        dim3 block(WARP_SIZE, block_y);
        dim3 grid(cuda::ceil_div(totalRow, block_y), 1, 1);

        size_t smem;
        switch (x.scalar_type()) {
          case at::ScalarType::Float: {
            smem = (size_t)(2 + block_y) * totalCol * sizeof(float);
            // 当 totalCol 较大时 smem 可能超过默认 48 KB，需显式请求更大动态 shared memory
            cudaFuncSetAttribute(
              (const void*)LayerNorm_forward_kernel_cg_warp_advanced<float, x128, int64_t>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            break;
          }
        }

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "LayerNorm_cu", ([&] {
          LayerNorm_forward_kernel_cg_warp_advanced<scalar_t, x128, int64_t><<<grid, block, smem>>>(totalRow,totalCol,x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),mean.data_ptr<scalar_t>(),rstd.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),bias.data_ptr<scalar_t>());
          cudaCheck(cudaGetLastError());
        }));
    #endif

    #if LAYERNORM_VARIANT == 10
        dim3 grid(1, cuda::ceil_div(totalRow,BLOCK_SIZE_Y), 1);
        dim3 block(WARP_SIZE, BLOCK_SIZE_Y, 1);
        assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");


        size_t dynSmem;
        switch (x.scalar_type()) {
          case at::ScalarType::Float: {
            constexpr size_t static_smem = 0;
            dynSmem = sizeof(float)  * (totalCol / 4) * 5 *  (2 + BLOCK_SIZE_Y);
            constexpr size_t SM86_PER_BLOCK_MAX = 99 * 1024;
            if (static_smem + dynSmem > SM86_PER_BLOCK_MAX) {
              throw std::runtime_error(
                  "[kernel10] dynSmem 超过 sm_86 per-block 上限 (99 KB)，请减小 totalCol 或 BLOCK_SIZE_Y");
            }
            size_t maxDynamicDiff = static_smem + dynSmem < 48 * 1024 ? 48 * 1024 - (static_smem + dynSmem): 0;
            size_t maxDynamic = std::min(dynSmem + maxDynamicDiff, SM86_PER_BLOCK_MAX - static_smem);
            cudaCheck(cudaFuncSetAttribute(
              (const void*)LayerNorm_kernel_warp_reduction_unroll_SMEM<float, float4, int64_t>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(maxDynamic)));
            break;
          }
        }

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "LayerNorm_cu", ([&] {
          LayerNorm_kernel_warp_reduction_unroll_SMEM<scalar_t, float4, int64_t><<<grid, block, dynSmem,0>>>(totalRow,totalCol,x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),mean.data_ptr<scalar_t>(),rstd.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),bias.data_ptr<scalar_t>());
          cudaCheck(cudaGetLastError());
        }));
    #endif

  }
  if (dim == 3) {
    assert(x.is_contiguous() && "x的内存必须连续");
    int64_t totalBatch {x.size(0)};
    int64_t totalRow {x.size(1)};
    int64_t totalCol {x.size(2)};

    // mean, rstd：每行一个标量（行数维度）
    // weight, bias：每列一个标量（列数维度），weight 初始化为 1，bias 初始化为 0
    auto mean   = torch::empty({totalBatch * totalRow}, x.options());
    auto rstd   = torch::empty({totalBatch * totalRow}, x.options());
    auto weight = torch::ones ({totalBatch * totalCol}, x.options());
    auto bias   = torch::zeros({totalBatch * totalCol}, x.options());


    #if LAYERNORM_VARIANT == 1
      dim3 grid(1, totalBatch * totalRow, 1);
      dim3 block(BLOCK_SIZE_X, 1, 1);
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "LayerNorm_cu", ([&] {
            LayerNorm_kernel_naive<scalar_t, int64_t><<<grid,block,0,0>>>(totalBatch * totalRow,totalCol,x.data_ptr<scalar_t>(),
              out.data_ptr<scalar_t>(),mean.data_ptr<scalar_t>(),rstd.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),bias.data_ptr<scalar_t>());
            cudaCheck(cudaGetLastError());
      }));
    #endif

    #if LAYERNORM_VARIANT == 2
    dim3 grid(1, totalBatch * totalRow, 1);
    dim3 block(BLOCK_SIZE_X, 1, 1);
    assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "LayerNorm_cu", ([&] {
      LayerNorm_kernel_double_warp_reduction<scalar_t, float4,int64_t><<<grid,block,0,0>>>(totalBatch * totalRow,totalCol,x.data_ptr<scalar_t>(),
          out.data_ptr<scalar_t>(),mean.data_ptr<scalar_t>(),rstd.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),bias.data_ptr<scalar_t>());
      cudaCheck(cudaGetLastError());
    }));
    #endif

    #if LAYERNORM_VARIANT == 3
      dim3 grid(1, totalBatch * totalRow, 1);
      dim3 block(BLOCK_SIZE_X, 1, 1);
      assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "LayerNorm_cu", ([&] {
        LayerNorm_kernel_double_warp_reduction_unroll<scalar_t, float4,int64_t><<<grid, block, 0, 0>>>(totalBatch * totalRow,totalCol,x.data_ptr<scalar_t>(),
          out.data_ptr<scalar_t>(),mean.data_ptr<scalar_t>(),rstd.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),bias.data_ptr<scalar_t>());
        cudaCheck(cudaGetLastError());
      }));
    #endif

    #if LAYERNORM_VARIANT == 4
      dim3 grid(1, totalBatch * totalRow, 1);
      dim3 block(BLOCK_SIZE_X, 1, 1);
      assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "LayerNorm_cu", ([&] {
        LayerNorm_kernel_welford_double_warp_reduction<scalar_t, float4,int64_t><<<grid, block, 0, 0>>>(totalBatch * totalRow,totalCol,x.data_ptr<scalar_t>(),
          out.data_ptr<scalar_t>(),mean.data_ptr<scalar_t>(),rstd.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),bias.data_ptr<scalar_t>());
        cudaCheck(cudaGetLastError());
      }));
    #endif

    #if LAYERNORM_VARIANT == 5
      dim3 grid(1, totalBatch * totalRow, 1);
      dim3 block(BLOCK_SIZE_X, 1, 1);
      assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "LayerNorm_cu", ([&] {
        LayerNorm_kernel_welford_double_warp_reduction_unroll<scalar_t, float4,int64_t><<<grid, block, 0, 0>>>(totalBatch * totalRow,totalCol,x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),mean.data_ptr<scalar_t>(),rstd.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),bias.data_ptr<scalar_t>());
        cudaCheck(cudaGetLastError());
      }));

    #endif
    #if LAYERNORM_VARIANT == 6
      dim3 grid(1, cuda::ceil_div(totalBatch * totalRow,BLOCK_SIZE_Y_SMEM), 1);
      dim3 block(BLOCK_SIZE_X_SMEM, BLOCK_SIZE_Y_SMEM, 1);
      assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");

      size_t dynSmem;
      switch (x.scalar_type()) {
        case at::ScalarType::Float: {
          constexpr size_t static_smem = sizeof(float) * BLOCK_SIZE_Y_SMEM * 32;
          // SMEM动态存储A的每一列元素和weight、bias以供复用，因为totalCol不是编译器常量，所以需要用动态smem
          dynSmem  = sizeof(float)  * (totalCol / 4) * 5 *  (2 + BLOCK_SIZE_Y_SMEM);
          constexpr size_t SM86_PER_BLOCK_MAX = 99 * 1024;
          if (static_smem + dynSmem > SM86_PER_BLOCK_MAX) {
            throw std::runtime_error(
                "[kernel6] dynSmem 超过 sm_86 per-block 上限 (99 KB)，请减小 totalCol 或 BLOCK_SIZE_Y_SMEM");
          }
          size_t maxDynamicDiff = static_smem + dynSmem < 48 * 1024 ? 48 * 1024 - (static_smem + dynSmem): 0;
          size_t maxDynamic {dynSmem + maxDynamicDiff};
          cudaCheck(cudaFuncSetAttribute(
            (const void*)LayerNorm_kernel_double_warp_reduction_unroll_SMEM<float, float4, int64_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(maxDynamic)));
          break;
        }
      }

      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "LayerNorm_cu", ([&] {
        LayerNorm_kernel_double_warp_reduction_unroll_SMEM<scalar_t, float4, int64_t><<<grid, block, dynSmem,0>>>(totalBatch * totalRow,totalCol,x.data_ptr<scalar_t>(),
          out.data_ptr<scalar_t>(),mean.data_ptr<scalar_t>(),rstd.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),bias.data_ptr<scalar_t>());
        cudaCheck(cudaGetLastError());
      }));
    #endif

    #if LAYERNORM_VARIANT == 7
      const uint block_size = 128;
      const uint warps_per_block = block_size / 32;
      dim3 block(block_size, 1, 1);
      dim3 grid(cuda::ceil_div(totalBatch * totalRow, warps_per_block), 1, 1);

      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "LayerNorm_cu", ([&] {
        LayerNorm_forward_kernel_cg_warp<scalar_t, float4, int64_t><<<grid, block, 0, 0>>>(totalBatch * totalRow,totalCol,x.data_ptr<scalar_t>(),
          out.data_ptr<scalar_t>(),mean.data_ptr<scalar_t>(),rstd.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),bias.data_ptr<scalar_t>());
        cudaCheck(cudaGetLastError());
      }));
    #endif

    #if LAYERNORM_VARIANT == 8
        dim3 block(BLOCK_SIZE_X, 1, 1);
        dim3 grid(totalBatch * totalRow, 1, 1);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "LayerNorm_cu", ([&] {
          LayerNorm_forward_kernel_cg_block<scalar_t, float4, int64_t><<<grid, block, 0, 0>>>(totalBatch * totalRow,totalCol,x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),mean.data_ptr<scalar_t>(),rstd.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),bias.data_ptr<scalar_t>());
          cudaCheck(cudaGetLastError());
        }));
    #endif


    #if LAYERNORM_VARIANT == 9
        const uint block_y = 4;
        dim3 block(WARP_SIZE, block_y);
        dim3 grid(cuda::ceil_div(totalBatch * totalRow, block_y), 1, 1);

        size_t smem;
        switch (x.scalar_type()) {
          case at::ScalarType::Float: {
          smem = (size_t)(2 + block_y) * totalCol * sizeof(float);
          // 当 totalCol 较大时 smem 可能超过默认 48 KB，需显式请求更大动态 shared memory
          cudaFuncSetAttribute(
            (const void*)LayerNorm_forward_kernel_cg_warp_advanced<float, x128, int64_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            break;
          }
        }

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "LayerNorm_cu", ([&] {
          LayerNorm_forward_kernel_cg_warp_advanced<scalar_t, x128, int64_t><<<grid, block, smem>>>(totalBatch * totalRow,totalCol,x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),mean.data_ptr<scalar_t>(),rstd.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),bias.data_ptr<scalar_t>());
          cudaCheck(cudaGetLastError());
        }));
    #endif

    #if LAYERNORM_VARIANT == 10
        dim3 grid(1, cuda::ceil_div(totalBatch * totalRow,BLOCK_SIZE_Y), 1);
        dim3 block(WARP_SIZE, BLOCK_SIZE_Y, 1);
        assert(totalCol % 4 == 0 && "向量化加载不能完整覆盖所有列");

        size_t dynSmem;
        switch (x.scalar_type()) {
          case at::ScalarType::Float: {
            constexpr size_t static_smem = 0;
            dynSmem  = sizeof(float)  * (totalCol / 4) * 5 *  (2 + BLOCK_SIZE_Y);
            constexpr size_t SM86_PER_BLOCK_MAX = 99 * 1024;
            if (static_smem + dynSmem > SM86_PER_BLOCK_MAX) {
              throw std::runtime_error(
                  "[kernel10] dynSmem 超过 sm_86 per-block 上限 (99 KB)，请减小 totalCol 或 BLOCK_SIZE_Y");
            }
            size_t maxDynamicDiff = static_smem + dynSmem < 48 * 1024 ? 48 * 1024 - (static_smem + dynSmem): 0;
            size_t maxDynamic = std::min(dynSmem + maxDynamicDiff, SM86_PER_BLOCK_MAX - static_smem);
            cudaCheck(cudaFuncSetAttribute(
              (const void*)LayerNorm_kernel_warp_reduction_unroll_SMEM<float, float4, int64_t>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(maxDynamic)));
            break;
          }
        }

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "LayerNorm_cu", ([&] {
          LayerNorm_kernel_warp_reduction_unroll_SMEM<scalar_t, float4, int64_t><<<grid, block, dynSmem,0>>>(totalBatch * totalRow,totalCol,x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),mean.data_ptr<scalar_t>(),rstd.data_ptr<scalar_t>(),weight.data_ptr<scalar_t>(),bias.data_ptr<scalar_t>());
          cudaCheck(cudaGetLastError());
        }));
    #endif

  }
  return out;
}
