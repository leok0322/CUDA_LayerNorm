#!/usr/bin/env bash

set -u
set -o pipefail

# 搜索空间：BLOCK_SIZE_X_SMEM 必须是 32 的倍数（覆盖完整 warp）
#           BLOCK_SIZE_Y_SMEM * BLOCK_SIZE_X_SMEM <= 1024（CUDA 每 block 线程数上限）
BSX_VALUES=(32 64 128 256 512 1024)   # BLOCK_SIZE_X_SMEM：block X 方向线程数（warps = BSX/32）
BSY_VALUES=(1 2 4 8 16 32)            # BLOCK_SIZE_Y_SMEM：block 内同时处理的行数

# 切换到项目根目录
cd "$(dirname "$0")"

COMMON="src/kernels/common.cuh"
OUTPUT="autotune/kernel_6_autotune_results.txt"

mkdir -p "$(dirname "$OUTPUT")"
echo "" > "$OUTPUT"

export DEVICE="0"

# 预先统计合法组合数，用于进度提示
TOTAL_CONFIGS=0
for bsx in "${BSX_VALUES[@]}"; do
  for bsy in "${BSY_VALUES[@]}"; do
    if [[ $(( bsx * bsy )) -le 1024 ]]; then
      TOTAL_CONFIGS=$(( TOTAL_CONFIGS + 1 ))
    fi
  done
done

CONFIG_NUM=0

for bsx in "${BSX_VALUES[@]}"; do
  for bsy in "${BSY_VALUES[@]}"; do
    echo ""

    config="BLOCK_SIZE_X_SMEM=$bsx BLOCK_SIZE_Y_SMEM=$bsy"

    # 约束：每 block 总线程数 = BSX * BSY <= 1024（硬件上限）
    # BSX 已由候选列表保证是 32 的倍数，此处无需再检查
    if [[ $(( bsx * bsy )) -gt 1024 ]]; then
      echo "THREADS: Skipping $config because BSX * BSY = $(( bsx * bsy )) > 1024"
      continue
    fi

    CONFIG_NUM=$(( CONFIG_NUM + 1 ))

    # 原地替换 common.cuh 中的宏定义
    # common.cuh 使用 #ifndef 守卫，sed 直接替换 #define 行的数值即可
    sed -i "s/#define BLOCK_SIZE_X_SMEM .*/#define BLOCK_SIZE_X_SMEM $bsx/" "$COMMON"
    sed -i "s/#define BLOCK_SIZE_Y_SMEM .*/#define BLOCK_SIZE_Y_SMEM $bsy/" "$COMMON"

    # 重新编译（每次参数变化都需要完整重编译，宏是编译期常量）
    if ! cmake --build cmake-build-release --target validation -- -j 18 2>&1 | tee -a "$OUTPUT"; then
      echo "COMPILE FAILED: $config" | tee -a "$OUTPUT"
      continue
    fi

    echo "($CONFIG_NUM/$TOTAL_CONFIGS): $config" |& tee -a "$OUTPUT"
    timeout -v 10 ./validation 6 2>&1 | tee -a "$OUTPUT"
    echo "-------------------" | tee -a "$OUTPUT"
    echo "" | tee -a "$OUTPUT"
  done
done

# 恢复默认值，避免脚本结束后 common.cuh 停留在最后一个测试配置
sed -i "s/#define BLOCK_SIZE_X_SMEM .*/#define BLOCK_SIZE_X_SMEM 256/" "$COMMON"
sed -i "s/#define BLOCK_SIZE_Y_SMEM .*/#define BLOCK_SIZE_Y_SMEM 4/" "$COMMON"

echo ""
echo "Autotune complete. Results in $OUTPUT"
