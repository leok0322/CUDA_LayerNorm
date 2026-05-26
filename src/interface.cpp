// torch/extension.h：PyTorch C++ 扩展的一站式头文件，包含：
//   · pybind11          — Python ↔ C++ 绑定框架
//   · torch::Tensor     — Tensor 句柄类型及其方法（.size() / .data_ptr() / ...）
//   · AT_DISPATCH_*     — dtype 分发宏
//   · PYBIND11_MODULE   — 生成 Python 模块入口符号 PyInit_<name>
#include <torch/extension.h>

// 声明
// 定义在 run_kernels.cu 中的函数（跨编译单元，链接时由链接器解析）
// torch::Tensor 按值传递是浅拷贝（复制句柄，ref_count++），不复制 GPU 数据，开销 O(1)
torch::Tensor LayerNorm_cu(torch::Tensor x);

// Python 侧调用的入口函数，链接到 run_kernels.cu 中的实现
// 此处可做参数校验（shape / dtype / device 检查），当前直接透传
torch::Tensor LayerNorm_cuda(torch::Tensor x)
{
  return LayerNorm_cu(x);
}

// PYBIND11_MODULE：pybind11 宏，生成 Python 模块初始化函数 PyInit_softmax_cuda
//   · TORCH_EXTENSION_NAME：cmake target_compile_definitions 注入的宏，
//     展开为 softmax_cuda，与 .so 文件名一致，确保 import 时入口符号匹配
//   · m：pybind11::module_ 对象，代表当前 Python 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  // m.def：将 C++ 函数注册为 Python 可调用对象
  //   · "softmax_cuda"  — Python 侧的函数名：cuda.softmax_cuda(x)
  //   · &softmax_cuda   — C++ 函数指针
  //   · "Softmax (CUDA)"— docstring，Python 中 help(cuda.softmax_cuda) 可见
  // pybind11 自动处理 torch::Tensor ↔ Python tensor 的类型转换，
  // 两侧共享同一块 GPU 内存，无数据拷贝
  // m.def 第二个参数要求函数指针，类型为 torch::Tensor (*)(torch::Tensor)
  // &LayerNorm_cuda 显式取地址；不写 & 函数名也会隐式退化为指针，效果相同
  // pybind11 收到指针后生成 Python 可调用对象，调用时自动完成：
  //   Python tensor → torch::Tensor → LayerNorm_cuda() → torch::Tensor → Python tensor
  m.def("LayerNorm", &LayerNorm_cuda, "LayerNorm (CUDA)");
}
