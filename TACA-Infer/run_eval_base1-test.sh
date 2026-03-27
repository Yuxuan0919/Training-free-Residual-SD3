#!/bin/bash
set -euo pipefail

# # =============== 阶段 0：环境 ===============
# source /inspire/hdd/project/chineseculture/public/yuxuan/miniconda3/etc/profile.d/conda.sh
# conda activate geneval_1
# cd /inspire/hdd/project/chineseculture/public/yuxuan/benches/geneval



MODEL='flux'
NFE=50
CFG=3.5
IMGSIZE=1024
BATCHSIZE=16




BASE_GENEVAL_DIR="/inspire/hdd/project/chineseculture/public/yuxuan/benches/geneval/outputs_flux_TACA"
DPG_SAVE_BASE="/inspire/hdd/project/chineseculture/public/yuxuan/benches/ELLA/dpg_bench/outputs_flux_TACA"
mkdir -p  "$BASE_GENEVAL_DIR" "$DPG_SAVE_BASE"





EXP_NAME="TACA-lora64"

GENEVAL_OUTDIR="${BASE_GENEVAL_DIR}/${EXP_NAME}"
DPG_OUTDIR="${DPG_SAVE_BASE}/${EXP_NAME}"

mkdir -p "$GENEVAL_OUTDIR" "$DPG_OUTDIR" 

echo "→ GENEVAL_OUTDIR: $GENEVAL_OUTDIR"
echo "→ DPG_OUTDIR:     $DPG_OUTDIR"



DPG_BENCH_DIR="/inspire/hdd/project/chineseculture/public/yuxuan/benches/ELLA/dpg_bench"
DPG_RESOLUTION=1024   # 单格尺寸，官方要求


DPG_EVAL_RES="${DPG_SAVE_BASE}/results/${EXP_NAME}.txt"

echo "----------------------------------------------------"
echo " Evaluating DPG directory: $DPG_OUTDIR"
echo "----------------------------------------------------"

python compute_dpg_bench.py \
    --image-root-path "$DPG_OUTDIR" \
    --resolution $DPG_RESOLUTION
    # --res-path "$DPG_EVAL_RES" \

echo "DPG evaluation finished: $DPG_OUTDIR"
echo "    → Log file: "$DPG_EVAL_RES""
echo












