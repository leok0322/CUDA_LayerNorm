#pragma once

#include <cudnn.h>         // cuDNN backend graph API：cudnnBackendCreateDescriptor 等
#include <cuda_runtime.h>  // cudaMalloc / cudaMemcpy / cudaFree 等 CUDA Runtime API
#include <cstdio>          // fprintf / stderr
#include <cstdlib>         // exit / EXIT_FAILURE
#include "../error_check.cuh"

// ── cuDNN 错误检查辅助函数 ────────────────────────────────────────────────────
// 与 error_check.cuh 中的 cudaCheck 设计相同：失败时打印文件名、行号、错误描述并终止
// inline 的根本作用：此 .cuh 头文件会被多个翻译单元 #include，每个翻译单元都会得到
//   一份 cudnnCheck 的函数定义。inline 告知链接器这是合法的多份定义并自动合并，
//   防止链接时报 "multiple definition of cudnnCheck" 的 ODR 重复定义错误。
//   次要作用：建议编译器内联展开，消除函数调用开销。
inline void cudnnCheck(cudnnStatus_t status, const char* file, int line) {
    if (status != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "[cuDNN ERROR] at file %s:%d:\n%s\n",
                file, line, cudnnGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}
// 单参数宏：调用处只写 cudnnCheck(expr)，自动填入 __FILE__ 和 __LINE__
#define cudnnCheck(s) (cudnnCheck(s, __FILE__, __LINE__))

// ════════════════════════════════════════════════════════════════════════════
// cuDNN Backend Graph API LayerNorm 前向推理
// ════════════════════════════════════════════════════════════════════════════
//
// ── 背景：为什么需要 Backend Graph API ───────────────────────────────────────
//
// cuDNN 有两套 API：
//   Legacy API（已废弃）：cudnnNormalizationForwardInference，cudnnNormMode_t
//     只有 CUDNN_NORM_PER_ACTIVATION / CUDNN_NORM_PER_CHANNEL（均为 BatchNorm 变体）
//     不支持 LayerNorm。
//
//   Backend Graph API（cuDNN 9.0 新增）：cudnnBackendXxx 系列函数
//     cudnnBackendNormMode_t 包含 CUDNN_LAYER_NORM，支持真正的 LayerNorm。
//     编程模型是"声明式图"：先描述张量和运算，由 cuDNN 选择最优 kernel 执行。
//
// ── Backend Graph API 的执行流程（5 层抽象）────────────────────────────────
//
//   ┌─ 1. Tensor Descriptor ─────────────────────────────────────────────┐
//   │   描述每个张量的形状、步长、数据类型、唯一 ID（UID）               │
//   │   UID 用于在 VariantPack 中将描述符与设备指针绑定                  │
//   └────────────────────────────────────────────────────────────────────┘
//              ↓ 作为属性传入
//   ┌─ 2. Operation Descriptor ──────────────────────────────────────────┐
//   │   描述运算类型（此处为 NORM_FORWARD）及输入/输出张量               │
//   │   CUDNN_LAYER_NORM：每个样本 N 在 C 维度上独立归一化               │
//   │   CUDNN_NORM_FWD_INFERENCE：推理模式，实时计算统计量               │
//   └────────────────────────────────────────────────────────────────────┘
//              ↓ 打包成图
//   ┌─ 3. OperationGraph ────────────────────────────────────────────────┐
//   │   将一个或多个 Operation 组成有向无环图（DAG）                     │
//   │   cuDNN 在 Finalize 时分析图结构，准备引擎选择                     │
//   └────────────────────────────────────────────────────────────────────┘
//              ↓ 启发式选择最优引擎
//   ┌─ 4. EngineHeuristic → EngineCfg → ExecutionPlan ───────────────────┐
//   │   EngineHeuristic：查询 cuDNN 内置启发式，返回按性能排序的引擎列表 │
//   │   EngineCfg：选取第一个（最优）引擎配置                           │
//   │   ExecutionPlan：基于 EngineCfg 编译成可直接执行的 plan            │
//   │     Finalize 时确定所需 workspace 大小                             │
//   └────────────────────────────────────────────────────────────────────┘
//              ↓ 绑定运行时指针
//   ┌─ 5. VariantPack → Execute ─────────────────────────────────────────┐
//   │   VariantPack：将 UID 数组与设备内存指针数组一一对应               │
//   │   cudnnBackendExecute(handle, plan, varPack)：提交执行              │
//   └────────────────────────────────────────────────────────────────────┘
//
// ── Tensor UID 常量（图内各张量的唯一标识符）──────────────────────────────
// UID 是任意 int64_t 整数，只要在同一图中不重复即可。
// VariantPack 通过 UID 数组把运行时指针与描述符对应起来：
//   uids[i] 对应的设备内存地址 = ptrs[i]
static constexpr int64_t UID_X       = 1;   // 输入张量 x
static constexpr int64_t UID_SCALE   = 2;   // 可学习参数 γ（weight）
static constexpr int64_t UID_BIAS    = 3;   // 可学习参数 β（bias）
static constexpr int64_t UID_Y       = 4;   // 输出张量 y
static constexpr int64_t UID_MEAN    = 5;   // 输出统计量：每行均值
static constexpr int64_t UID_RSTD    = 6;   // 输出统计量：每行倒数标准差
static constexpr int64_t UID_EPSILON = 7;   // 标量常量 ε（数值稳定项）

// ── 辅助函数：创建并 Finalize 一个 Tensor Descriptor ────────────────────────
//
// 参数说明：
//   uid       : 该张量在图中的唯一 ID，VariantPack 通过此值绑定设备指针
//   dims      : 各维度大小，int64_t 数组，长度为 ndim
//   strides   : 各维度步长（元素为单位，非字节），行主序时 strides[i] = dims[i+1]*...
//   ndim      : 维度数（此处统一用 4 维）
//   isByValue : true 表示标量常量（如 ε），值通过 VariantPack 的指针传入设备内存
//
// 关键属性：
//   CUDNN_ATTR_TENSOR_UNIQUE_ID      : UID，图内唯一
//   CUDNN_ATTR_TENSOR_DATA_TYPE      : CUDNN_DATA_FLOAT（float32）
//   CUDNN_ATTR_TENSOR_DIMENSIONS     : 形状数组
//   CUDNN_ATTR_TENSOR_STRIDES        : 步长数组（决定内存布局）
//   CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT : 内存对齐字节数，float = 4
//   CUDNN_ATTR_TENSOR_IS_BY_VALUE    : 仅对标量常量（ε）设为 true
//
// cudnnBackendFinalize：锁定描述符，之后不可再修改属性。
//   未 Finalize 的描述符不能传给 Operation 或 OperationGraph。
inline cudnnBackendDescriptor_t createTensorDesc(
        int64_t uid, const int64_t* dims, const int64_t* strides,
        int ndim, bool isByValue = false) {

    cudnnBackendDescriptor_t tensor;
    cudnnCheck(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &tensor));

    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnCheck(cudnnBackendSetAttribute(tensor, CUDNN_ATTR_TENSOR_DATA_TYPE,
                                       CUDNN_TYPE_DATA_TYPE, 1, &dtype));
    cudnnCheck(cudnnBackendSetAttribute(tensor, CUDNN_ATTR_TENSOR_UNIQUE_ID,
                                       CUDNN_TYPE_INT64, 1, &uid));
    cudnnCheck(cudnnBackendSetAttribute(tensor, CUDNN_ATTR_TENSOR_DIMENSIONS,
                                       CUDNN_TYPE_INT64, ndim, dims));
    cudnnCheck(cudnnBackendSetAttribute(tensor, CUDNN_ATTR_TENSOR_STRIDES,
                                       CUDNN_TYPE_INT64, ndim, strides));
    int64_t alignment = 4;  // sizeof(float)
    cudnnCheck(cudnnBackendSetAttribute(tensor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                                       CUDNN_TYPE_INT64, 1, &alignment));
    if (isByValue) {
        // IS_BY_VALUE = true：该张量是标量常量，不代表设备内存缓冲区
        // VariantPack 中对应的指针指向存放该标量值的设备内存（1 个 float）
        cudnnCheck(cudnnBackendSetAttribute(tensor, CUDNN_ATTR_TENSOR_IS_BY_VALUE,
                                           CUDNN_TYPE_BOOLEAN, 1, &isByValue));
    }
    cudnnCheck(cudnnBackendFinalize(tensor));
    return tensor;
}

// ── 主函数：cuDNN Backend Graph API LayerNorm 前向推理 ───────────────────────
//
// 输入/输出布局：[totalRow, totalCol] 行主序矩阵
//   A      : 输入 x，设备内存 [totalRow × totalCol]
//   out    : 输出 y，设备内存 [totalRow × totalCol]（由 cuDNN 计算）
//   mean   : 每行均值 [totalRow]（由辅助 kernel 计算，见 Step 11）
//   rstd   : 每行倒数标准差 [totalRow]（由辅助 kernel 计算，见 Step 11）
//   weight : γ，设备内存 [totalCol]
//   bias   : β，设备内存 [totalCol]
//
// 设计说明：mean/rstd 不通过 cuDNN 图输出，原因：
//   cuDNN 9.x 的 INFERENCE 和 TRAINING 模式均在 ExecutionPlan::Finalize 阶段
//   对输出 mean/rstd 报 CUDNN_STATUS_NOT_SUPPORTED（引擎不支持此输出组合）。
//   解决方案：cuDNN 图只负责计算 y；mean/rstd 由 layernorm_base_mean/rstd_kernel
//   单独计算，两者结果在数学上完全等价。
inline void run_LayerNorm_kernel_base_graph(
        const uint totalRow, const uint totalCol, const float* A,
        float* out, float* mean, float* rstd,
        const float* weight, const float* bias) {

    const int64_t N = static_cast<int64_t>(totalRow);
    const int64_t C = static_cast<int64_t>(totalCol);

    // ── Step 1：创建 cuDNN handle ─────────────────────────────────────────
    cudnnHandle_t handle;
    cudnnCheck(cudnnCreate(&handle));

    // ── Step 2：定义各张量的形状和步长 ────────────────────────────────────
    //
    // 张量布局：[N, C, 1, 1]，行主序（row-major）
    //   strides[0]=C：相邻两行相差 C 个 float
    //   strides[1]=1：同行相邻两列相差 1 个 float
    int64_t xDims[4]      = {N, C, 1, 1};
    int64_t xStrides[4]   = {C, 1, 1, 1};

    // γ/β 形状 [1, C, 1, 1]：逐特征参数，在 N 维广播
    int64_t scaleDims[4]    = {1, C, 1, 1};
    int64_t scaleStrides[4] = {C, 1, 1, 1};

    // ε 形状 [1, 1, 1, 1]：标量常量，IS_BY_VALUE=true
    int64_t epsDims[4]    = {1, 1, 1, 1};
    int64_t epsStrides[4] = {1, 1, 1, 1};

    // ── Step 3：创建张量描述符（仅 x、γ、β、y、ε，不含 mean/rstd）──────
    cudnnBackendDescriptor_t xTensor     = createTensorDesc(UID_X,       xDims,     xStrides,     4);
    cudnnBackendDescriptor_t scaleTensor = createTensorDesc(UID_SCALE,   scaleDims, scaleStrides, 4);
    cudnnBackendDescriptor_t biasTensor  = createTensorDesc(UID_BIAS,    scaleDims, scaleStrides, 4);
    cudnnBackendDescriptor_t yTensor     = createTensorDesc(UID_Y,       xDims,     xStrides,     4);
    // ε 是标量常量：IS_BY_VALUE=true，VariantPack 中对应指针指向设备内存上的 float 值
    cudnnBackendDescriptor_t epsTensor   = createTensorDesc(UID_EPSILON, epsDims,   epsStrides,   4,
                                                            /*isByValue=*/true);

    // ── Step 4：创建 LayerNorm 前向 Operation ─────────────────────────────
    //
    //   CUDNN_LAYER_NORM        : 每个样本 N 在 C,H,W 维度上独立归一化
    //   CUDNN_NORM_FWD_INFERENCE: 推理模式，实时计算 mean/rstd 用于归一化，
    //                             但不作为用户可见输出（不设置 MEAN_DESC/INV_VARIANCE_DESC）
    //
    // 只绑定 x、γ、β、ε、y 五个角色，不设置 MEAN_DESC 和 INV_VARIANCE_DESC：
    //   去掉这两个输出后，cuDNN 引擎能正常选到支持的实现。
    //   mean/rstd 改由 Step 11 的辅助 kernel 单独计算。
    cudnnBackendDescriptor_t normOp;
    cudnnCheck(cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR, &normOp));

    cudnnBackendNormMode_t normMode = CUDNN_LAYER_NORM;
    cudnnCheck(cudnnBackendSetAttribute(normOp, CUDNN_ATTR_OPERATION_NORM_FWD_MODE,
                                       CUDNN_TYPE_NORM_MODE, 1, &normMode));
    cudnnBackendNormFwdPhase_t phase = CUDNN_NORM_FWD_INFERENCE;
    cudnnCheck(cudnnBackendSetAttribute(normOp, CUDNN_ATTR_OPERATION_NORM_FWD_PHASE,
                                       CUDNN_TYPE_NORM_FWD_PHASE, 1, &phase));
    cudnnCheck(cudnnBackendSetAttribute(normOp, CUDNN_ATTR_OPERATION_NORM_FWD_XDESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xTensor));
    cudnnCheck(cudnnBackendSetAttribute(normOp, CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &scaleTensor));
    cudnnCheck(cudnnBackendSetAttribute(normOp, CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &biasTensor));
    cudnnCheck(cudnnBackendSetAttribute(normOp, CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &epsTensor));
    cudnnCheck(cudnnBackendSetAttribute(normOp, CUDNN_ATTR_OPERATION_NORM_FWD_YDESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &yTensor));
    cudnnCheck(cudnnBackendFinalize(normOp));

    // ── Step 5：创建 OperationGraph ───────────────────────────────────────
    //
    // OperationGraph 是一个有向无环图（DAG），包含一个或多个 Operation 节点。
    // 此处只有一个 normOp 节点（单操作图）。
    // Finalize 后 cuDNN 分析图拓扑，为引擎选择做准备。
    // 注意：cuDNN 9.x 中 CUDNN_ATTR_OPERATIONGRAPH_HANDLE 已废弃，
    //       handle 改为在 cudnnBackendExecute 时直接传入。
    cudnnBackendDescriptor_t opGraph;
    cudnnCheck(cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &opGraph));
    cudnnCheck(cudnnBackendSetAttribute(opGraph, CUDNN_ATTR_OPERATIONGRAPH_OPS,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &normOp));
    cudnnCheck(cudnnBackendFinalize(opGraph));

    // ── Step 6：引擎启发式选择 → 最优 EngineCfg ──────────────────────────
    //
    // EngineHeuristic：向 cuDNN 查询适合该 OperationGraph 的引擎列表。
    //   CUDNN_HEUR_MODE_INSTANT：快速决策树模式，CPU 开销极小（微秒级），
    //     不做实际 kernel 基准测试，直接基于硬件参数和图结构估计最优引擎。
    //   另有 CUDNN_HEUR_MODE_B（更全面但慢）可用于离线调优。
    //
    // GetAttribute 取回第 0 个引擎配置（启发式排名第一）存入 engCfg。
    //   第 4 参数 = 1：只取 1 个结果；第 5 参数 &retCount：实际返回数量。
    cudnnBackendDescriptor_t engHeur;
    cudnnCheck(cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &engHeur));
    cudnnCheck(cudnnBackendSetAttribute(engHeur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &opGraph));
    cudnnBackendHeurMode_t heurMode = CUDNN_HEUR_MODE_INSTANT;
    cudnnCheck(cudnnBackendSetAttribute(engHeur, CUDNN_ATTR_ENGINEHEUR_MODE,
                                       CUDNN_TYPE_HEUR_MODE, 1, &heurMode));
    cudnnCheck(cudnnBackendFinalize(engHeur));

    cudnnBackendDescriptor_t engCfg;
    cudnnCheck(cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engCfg));
    int64_t retCount = 0;
    cudnnCheck(cudnnBackendGetAttribute(engHeur, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1, &retCount, &engCfg));

    // ── Step 7：创建 ExecutionPlan ────────────────────────────────────────
    //
    // ExecutionPlan 基于 EngineCfg 做最终编译（JIT PTX 或选择预编译 kernel），
    // 生成可直接执行的计划。
    // Finalize 后可查询 CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE，
    // 得到该 plan 需要的临时 workspace 字节数。
    cudnnBackendDescriptor_t plan;
    cudnnCheck(cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan));
    cudnnCheck(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engCfg));
    cudnnCheck(cudnnBackendFinalize(plan));

    int64_t workspaceSize = 0;
    cudnnCheck(cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                       CUDNN_TYPE_INT64, 1, nullptr, &workspaceSize));
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        cudaCheck(cudaMalloc(&workspace, static_cast<size_t>(workspaceSize)));
    }

    // ── Step 8：在设备内存上准备 ε 值 ────────────────────────────────────
    //
    // ε 张量标记了 IS_BY_VALUE=true，意味着它是一个标量常量，
    // VariantPack 中该 UID 对应的指针指向一块设备内存，内存中存放该 float 值。
    // 与普通张量（指向大块数据缓冲区）不同，这里只需 1 个 float 的设备内存。
    float epsVal = 1e-5f;
    float* dEps  = nullptr;
    cudaCheck(cudaMalloc(&dEps, sizeof(float)));
    cudaCheck(cudaMemcpy(dEps, &epsVal, sizeof(float), cudaMemcpyHostToDevice));

    // ── Step 9：创建 VariantPack，绑定 UID → 设备指针 ─────────────────────
    //
    // VariantPack 是运行时的"参数包"：
    //   CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS    : UID 数组，与 Step 2 中各张量 UID 对应
    //   CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS : 设备指针数组，与 UID 数组一一对应
    //   CUDNN_ATTR_VARIANT_PACK_WORKSPACE     : workspace 设备内存指针
    //
    // 每次 Execute 可以复用同一 plan，只更换 VariantPack 中的指针（不同输入批次）。
    // 只绑定参与图运算的 5 个张量（x、γ、β、y、ε），不含 mean/rstd。
    const int numTensors = 5;
    int64_t uids[numTensors] = {UID_X, UID_SCALE, UID_BIAS, UID_Y, UID_EPSILON};
    void*   ptrs[numTensors] = {(void*)A, (void*)weight, (void*)bias, (void*)out, (void*)dEps};

    cudnnBackendDescriptor_t varPack;
    cudnnCheck(cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &varPack));
    cudnnCheck(cudnnBackendSetAttribute(varPack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                       CUDNN_TYPE_INT64, numTensors, uids));
    cudnnCheck(cudnnBackendSetAttribute(varPack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                       CUDNN_TYPE_VOID_PTR, numTensors, ptrs));
    cudnnCheck(cudnnBackendSetAttribute(varPack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                       CUDNN_TYPE_VOID_PTR, 1, &workspace));
    cudnnCheck(cudnnBackendFinalize(varPack));

    // ── Step 10：执行 ─────────────────────────────────────────────────────
    //
    // cudnnBackendExecute 将 plan 提交到 handle 绑定的 CUDA 流上异步执行。
    // 调用返回后 kernel 可能尚未完成，需 cudaDeviceSynchronize 等待。
    cudnnCheck(cudnnBackendExecute(handle, plan, varPack));

    // ── Step 11：辅助 kernel 计算 mean / rstd ─────────────────────────────
    //
    // cuDNN 图只计算了输出 y，mean/rstd 由独立的 __global__ kernel 补充计算。
    // 数学结果与 cuDNN 内部的统计量完全等价（相同公式），用于验证框架对比。
    // rstd kernel 依赖 mean kernel 的结果，两者之间加 cudaDeviceSynchronize。
    // const int block_size = 256;
    // const int grid_size  = (static_cast<int>(totalRow) + block_size - 1) / block_size;
    // layernorm_base_mean_kernel<<<grid_size, block_size>>>(A, mean, totalCol);
    // cudaCheck(cudaDeviceSynchronize());
    // layernorm_base_rstd_kernel<<<grid_size, block_size>>>(A, mean, rstd, totalCol);
    // cudaCheck(cudaDeviceSynchronize());

    // ── Step 12：释放所有资源 ─────────────────────────────────────────────
    cudnnBackendDestroyDescriptor(xTensor);
    cudnnBackendDestroyDescriptor(scaleTensor);
    cudnnBackendDestroyDescriptor(biasTensor);
    cudnnBackendDestroyDescriptor(yTensor);
    cudnnBackendDestroyDescriptor(epsTensor);
    cudnnBackendDestroyDescriptor(normOp);
    cudnnBackendDestroyDescriptor(opGraph);
    cudnnBackendDestroyDescriptor(engHeur);
    cudnnBackendDestroyDescriptor(engCfg);
    cudnnBackendDestroyDescriptor(plan);
    cudnnBackendDestroyDescriptor(varPack);
    if (workspace) cudaCheck(cudaFree(workspace));
    cudaCheck(cudaFree(dEps));
    cudnnCheck(cudnnDestroy(handle));
}

// ── 辅助 kernel：计算每行均值 ─────────────────────────────────────────────────
// cudnnNormalizationForwardInference 只输出归一化结果，不对外暴露统计量。
// 此 kernel 单独计算 mean，供验证框架对比各版本 kernel 的中间量是否一致。
// 并行策略：每个线程负责一行，在 C 维度上串行累加（等同于 kernel1 的方式）
// __global__ 不能加 inline/__forceinline__：nvcc 不支持对设备入口函数内联，
//   多翻译单元的重复定义由 nvcc 自动以弱符号（weak symbol）解决，无需开发者干预。
__global__ void layernorm_base_mean_kernel(const float* x, float* mean, uint C) {
    uint row = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (uint i = 0; i < C; i++) {
        sum += x[row * C + i];
    }
    mean[row] = sum / static_cast<float>(C);   // μ = Σxᵢ / C
}

// ── 辅助 kernel：计算每行 rstd ────────────────────────────────────────────────
// rstd = 1 / sqrt(σ² + ε)，存倒数以便反向传播中用乘法代替除法
// 依赖 layernorm_base_mean_kernel 已写入的 mean 结果
__global__ void layernorm_base_rstd_kernel(const float* x, const float* mean,
                                           float* rstd, uint C) {
    uint row = blockIdx.x * blockDim.x + threadIdx.x;
    float m   = mean[row];
    float var = 0.0f;
    for (uint i = 0; i < C; i++) {
        float diff = x[row * C + i] - m;
        var += diff * diff;
    }
    // rsqrtf：CUDA 硬件倒数平方根指令，等价于 1.0f / sqrtf(...)，速度更快
    rstd[row] = rsqrtf(var / static_cast<float>(C) + 1e-5f);
}

// 2D grid 版 LayerNorm kernel
// grid : (⌈totalCol/32⌉, ⌈totalRow/32⌉, 1)   block : (32, 32, 1)
// 每个线程负责一个输出元素 out[row, col]
// 均值/方差：同行内各线程冗余独立计算，无需跨线程同步
template <typename scalar_t, typename scalar_i>
__global__ void LayerNorm_kernel_base(const scalar_i totalRow, const scalar_i totalCol, const scalar_t *A,
                scalar_t *out, scalar_t* mean, scalar_t* rstd, const scalar_t* weight, const scalar_t* bias) {
  // x 维覆盖列，y 维覆盖行
  const scalar_i row = blockIdx.y * blockDim.y + threadIdx.y;
  const scalar_i col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= totalRow || col >= totalCol) return;

  // 计算本行均值（同行所有线程各自独立完成整行累加）
  scalar_t rowMean {};
  for (scalar_i i {0}; i < totalCol; ++i) {
    rowMean += A[row * totalCol + i];
  }
  rowMean /= static_cast<scalar_t>(totalCol);
  // 每行只有 col==0 的线程写 mean，避免同行多线程重复写同一地址
  if (col == 0) {
    mean[row] = rowMean;
  }

  // 计算本行方差
  scalar_t rowVar {};
  for (scalar_i i {0}; i < totalCol; ++i) {
    scalar_t diff { A[row * totalCol + i] - rowMean };
    rowVar += diff * diff;
  }
  rowVar /= static_cast<scalar_t>(totalCol);
  rowVar = static_cast<scalar_t>(rsqrtf(static_cast<float>(rowVar + 1e-5f)));
  if (col == 0) {
    rstd[row] = rowVar;
  }

  // 每个线程写自己负责的列元素
  out[row * totalCol + col] = weight[col] * (A[row * totalCol + col] - rowMean) * rowVar + bias[col];
}