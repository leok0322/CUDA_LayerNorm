#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
# 上方 `# /// script` 块是 PEP 723 定义的 inline script metadata：
#   uv run plot_performance.py        → uv 解析此块，自动创建隔离环境并安装 matplotlib，再运行脚本
#   uv run python plot_performance.py → uv 把 python 当可执行命令，不解析此块，不自动安装依赖
#   python plot_performance.py        → 解释器直接忽略此块，需手动 pip install matplotlib
#
# 安装的包缓存在 ~/.cache/uv/，不会自动删除，下次运行直接复用；手动清理：uv cache clean
"""
用法：
  uv run plot_performance.py              # 绘制 m==n 的 6 个典型维度
  uv run plot_performance.py 128 512      # 只绘制指定的方阵维度
  uv run plot_performance.py --all-sizes  # 绘制 logs 中出现的全部方阵维度
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LOGS_DIR = Path(__file__).parent / "logs"

# 典型方阵维度（m == n）
DEFAULT_DIMS = [128, 256, 512, 1024, 2048, 4096]

# 典型非方阵维度（rows, cols）
DEFAULT_NONSQUARE = [
    (128,  4096),
    (256,  4096),
    (4096,  128),
    (4096, 1024),
]

KERNEL_IDS = list(range(0, 12))   # kernel 0–11

KERNEL_LABELS = {
    0:  "K0 base",
    1:  "K1 naive",
    2:  "K2 double_warp_reduction",
    3:  "K3 double_warp_reduction_unroll",
    4:  "K4 welford_double_warp_reduction",
    5:  "K5 welford_double_warp_reduction_unroll",
    6:  "K6 double_warp_reduction_unroll_SMEM",
    7:  "K7 cooperative_groups_warp",
    8:  "K8 cooperative_groups_block",
    9:  "K9 cooperative_warp_advanced",
    10: "K10 warp_reduction_unroll_SMEM",
    11: "K11 double_warp_reduction_unroll_SMEM_adv",
}


def parse_log(kernel_id: int) -> dict[tuple[int, int], float]:
    """解析 kernel_{id}.log，返回 {(m, n): gflops}。"""
    path = LOGS_DIR / f"kernel_{kernel_id}.log"
    if not path.exists():
        return {}
    # 匹配：performance: (  123.4) GFLOPS. size: (128, 256).
    pattern = re.compile(
        r"performance:\s*\(\s*([\d.]+)\s*\)\s*GFLOPS\.\s*size:\s*\((\d+),\s*(\d+)\)"
    )
    results: dict[tuple[int, int], float] = {}
    for line in path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            gflops = float(m.group(1))
            rows, cols = int(m.group(2)), int(m.group(3))
            results[(rows, cols)] = gflops
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="绘制 LayerNorm kernel 0–11 在方阵维度上的 GFLOPS 折线图"
    )
    parser.add_argument(
        "dims",
        nargs="*",
        type=int,
        metavar="DIM",
        help="方阵维度 m=n，不填则使用默认 6 个典型维度",
    )
    parser.add_argument(
        "--all-sizes",
        action="store_true",
        help="自动发现 logs 中出现的全部方阵维度",
    )
    args = parser.parse_args()

    # 加载所有 kernel 数据
    data: dict[int, dict[tuple[int, int], float]] = {}
    for kid in KERNEL_IDS:
        parsed = parse_log(kid)
        if parsed:
            data[kid] = parsed
        else:
            print(f"警告：未找到 kernel_{kid}.log，跳过", file=sys.stderr)

    if not data:
        print("错误：logs/ 目录下没有找到任何日志文件", file=sys.stderr)
        sys.exit(1)

    # 确定要绘制的方阵维度
    if args.all_sizes:
        all_keys: set[tuple[int, int]] = set()
        for d in data.values():
            all_keys.update(d.keys())
        dims = sorted({r for r, c in all_keys if r == c})
    elif args.dims:
        dims = sorted(set(args.dims))
    else:
        dims = DEFAULT_DIMS

    # kernel 0 作为基准线，折线只画 kernel 1–11
    plot_kernels = [k for k in KERNEL_IDS if k != 0]
    sq_colors   = [plt.cm.tab10(i) for i in range(len(dims))]
    nsq_colors  = [plt.cm.Set2(i)  for i in range(len(DEFAULT_NONSQUARE))]

    fig, ax = plt.subplots(figsize=(14, 7))

    # ── 方阵折线（实线 + 圆形标记）────────────────────────────────────────────
    for i, dim in enumerate(dims):
        color = sq_colors[i]
        key = (dim, dim)

        ys = [data.get(kid, {}).get(key) for kid in plot_kernels]
        valid = [(kid, v) for kid, v in zip(plot_kernels, ys) if v is not None]
        if not valid:
            print(f"警告：维度 {dim}×{dim} 在所有 kernel 中均无数据，跳过", file=sys.stderr)
            continue
        xs, ys_valid = zip(*valid)
        ax.plot(
            xs, ys_valid,
            label=f"{dim}×{dim}",
            marker="o",
            linewidth=1.8,
            markersize=5,
            color=color,
        )

        # K0 基准：同色水平虚线
        baseline = data.get(0, {}).get(key)
        if baseline is not None:
            ax.axhline(
                baseline,
                linestyle="--",
                linewidth=1.0,
                color=color,
                alpha=0.5,
            )

    # ── 非方阵折线（点划线 + 三角形标记）─────────────────────────────────────
    for i, (rows, cols) in enumerate(DEFAULT_NONSQUARE):
        color = nsq_colors[i]
        key = (rows, cols)

        ys = [data.get(kid, {}).get(key) for kid in plot_kernels]
        valid = [(kid, v) for kid, v in zip(plot_kernels, ys) if v is not None]
        if not valid:
            print(f"警告：维度 {rows}×{cols} 在所有 kernel 中均无数据，跳过", file=sys.stderr)
            continue
        xs, ys_valid = zip(*valid)
        ax.plot(
            xs, ys_valid,
            label=f"{rows}×{cols}",
            marker="^",
            linewidth=1.5,
            markersize=5,
            color=color,
            linestyle="-.",
        )

    # 各 kernel 在所有绘制维度（方阵 + 非方阵）上的平均 GFLOPS
    all_keys = [(d, d) for d in dims] + DEFAULT_NONSQUARE
    avg_ys = []
    for kid in plot_kernels:
        vals = [data.get(kid, {}).get(key) for key in all_keys]
        valid_vals = [v for v in vals if v is not None]
        avg_ys.append(sum(valid_vals) / len(valid_vals) if valid_vals else None)

    avg_valid = [(kid, v) for kid, v in zip(plot_kernels, avg_ys) if v is not None]
    if avg_valid:
        xs_avg, ys_avg = zip(*avg_valid)
        ax.plot(
            xs_avg, ys_avg,
            label="Average",
            marker="D",
            linewidth=2.5,
            markersize=7,
            color="black",
            linestyle="--",
            zorder=5,
        )
        for x, y in zip(xs_avg, ys_avg):
            ax.annotate(
                f"{y:.0f}",
                xy=(x, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="black",
                zorder=6,
            )

    ax.set_xlabel("Kernel", fontsize=12)
    ax.set_ylabel("Performance (GFLOPS)", fontsize=12)
    ax.set_title("CUDA LayerNorm Kernel Performance (K0–K11, ○ square  △ non-square)", fontsize=13)
    ax.set_xticks(plot_kernels)
    ax.set_xticklabels(
        [KERNEL_LABELS[k] for k in plot_kernels], rotation=25, ha="right"
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # 图例：在维度条目后追加基准线说明
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(
        Line2D([0], [0], linestyle="--", linewidth=1.0, color="gray", alpha=0.7)
    )
    labels.append("K0 baseline")
    ax.legend(handles, labels, loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = Path(__file__).parent / "performance_plot.png"
    plt.savefig(out, dpi=150)
    print(f"已保存：{out}")
    plt.show()


if __name__ == "__main__":
    main()
