import sys
import os
import datetime
import torch
import torch.nn.functional as F

# .so 由 cmake 编译输出，搜索 cmake-build-release / cmake-build-debug
_script_dir = os.path.dirname(os.path.abspath(__file__))
for _build_dir in ("cmake-build-release", "cmake-build-debug"):
    _candidate = os.path.join(_script_dir, _build_dir)
    if os.path.isdir(_candidate):
        sys.path.insert(0, _candidate)
        break

import LayerNorm_cuda  # LayerNorm_cuda.cpython-3XX-x86_64-linux-gnu.so

REPEATS = 100
WARMUP  = 10

# 测试矩阵尺寸 (rows, cols)
SIZES = [
    (128,   128),
    (128,   256),
    (128,   512),
    (128,  1024),
    (128,  4096),
    (256,   256),
    (256,   512),
    (256,  1024),
    (256,  4096),
    (512,   512),
    (512,  1024),
    (512,  4096),
    (1024, 1024),
    (1024, 4096),
    (2048, 2048),
    (4096, 4096),
]

# ── 计时（CUDA Event，精度 ~0.5 us） ─────────────────────────────────────────
def benchmark(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(REPEATS)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(REPEATS)]
    for i in range(REPEATS):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]  # ms

# ── 打印表头 ──────────────────────────────────────────────────────────────────
print(f"{'size':<14}  {'cuda mean':>10}  {'cuda median':>12}  "
      f"{'torch mean':>10}  {'torch median':>12}  {'torch/cuda':>10}  correctness")
print("-" * 100)

results = []

for rows, cols in SIZES:
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)

    # weight=1, bias=0 与 run_kernel.cu 中 torch::ones / torch::zeros 一致
    weight = torch.ones (cols, device="cuda", dtype=torch.float32)
    bias   = torch.zeros(cols, device="cuda", dtype=torch.float32)

    # ── 正确性验证 ────────────────────────────────────────────────────────────
    # std::runtime_error 经 pybind11 传到 Python 侧为 RuntimeError（如 SMEM 超限）
    try:
        out = LayerNorm_cuda.LayerNorm(x)
    except RuntimeError as e:
        print(f"{rows}x{cols:<8}  SKIP  {e}")
        results.append((rows, cols, None, None, "SKIP"))
        continue

    ref = F.layer_norm(x, normalized_shape=[cols], weight=weight, bias=bias)
    match = torch.allclose(out, ref, atol=1e-4, rtol=1e-4)

    if not match:
        max_diff = (out - ref).abs().max().item()
        print(f"{rows}x{cols:<8}  correctness FAIL  max_diff={max_diff:.2e}")
        results.append((rows, cols, None, None, False))
        continue

    # ── 计时 ─────────────────────────────────────────────────────────────────
    try:
        t_cuda  = benchmark(lambda: LayerNorm_cuda.LayerNorm(x))
    except RuntimeError as e:
        print(f"{rows}x{cols:<8}  SKIP (benchmark)  {e}")
        results.append((rows, cols, None, None, "SKIP"))
        continue
    t_torch = benchmark(lambda: F.layer_norm(x, [cols], weight, bias))

    tc = torch.tensor(t_cuda)
    tt = torch.tensor(t_torch)
    ratio = tt.median().item() / tc.median().item()

    print(f"{rows}x{cols:<8}"
          f"  {tc.mean().item():>9.3f}ms"
          f"  {tc.median().item():>11.3f}ms"
          f"  {tt.mean().item():>9.3f}ms"
          f"  {tt.median().item():>11.3f}ms"
          f"  {ratio:>9.2f}x"
          f"  PASS")
    results.append((rows, cols, t_cuda, t_torch, True))

# ── 追加写入结果文件 ──────────────────────────────────────────────────────────
RESULT_DIR  = os.path.join(_script_dir, "benchmark_results")
os.makedirs(RESULT_DIR, exist_ok=True)
RESULT_FILE = os.path.join(RESULT_DIR, "python_test_result.txt")

timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
lines = [f"\n[{timestamp}]  repeats={REPEATS}"]
for rows, cols, t_cuda, t_torch, ok in results:
    if ok == "SKIP":
        lines.append(f"  {rows}x{cols}  SKIP (RuntimeError)")
        continue
    if not ok:
        lines.append(f"  {rows}x{cols}  FAIL")
        continue
    tc = torch.tensor(t_cuda)
    tt = torch.tensor(t_torch)
    ratio = tt.median().item() / tc.median().item()
    lines.append(
        f"  {rows}x{cols:<8}"
        f"  cuda  mean={tc.mean().item():.3f}ms  median={tc.median().item():.3f}ms"
        f"  torch mean={tt.mean().item():.3f}ms  median={tt.median().item():.3f}ms"
        f"  torch/cuda={ratio:.2f}x"
    )

with open(RESULT_FILE, "a") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nresult appended to: {RESULT_FILE}")
