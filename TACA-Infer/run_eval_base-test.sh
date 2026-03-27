#!/bin/bash
set -euo pipefail

# =============== 阶段 0：环境 ===============
source /inspire/hdd/project/chineseculture/public/yuxuan/miniconda3/etc/profile.d/conda.sh
conda activate geneval_1
cd /inspire/hdd/project/chineseculture/public/yuxuan/benches/geneval



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



echo "----------------------------------------------------"
echo " Evaluating Geneval directory:"
echo "   $GENEVAL_OUTDIR"
echo "----------------------------------------------------"

MASK2FORMER_PATH="/inspire/hdd/project/chineseculture/public/yuxuan/benches/geneval/mask2former"

STEP_NAME=$(basename "$GENEVAL_OUTDIR")
OUTFILE_PARENT=$(dirname "$GENEVAL_OUTDIR")
GENEVAL_OUTFILE="${OUTFILE_PARENT}/results_${STEP_NAME}.jsonl"

python evaluation/evaluate_images.py \
    "$GENEVAL_OUTDIR" \
    --outfile "$GENEVAL_OUTFILE" \
    --model-path "$MASK2FORMER_PATH" \

python evaluation/summary_scores.py \
    "$GENEVAL_OUTFILE"

echo "🎉 Geneval evaluation finished: $STEP_NAME"
echo















EXP_NAME="TACA-lora16"

GENEVAL_OUTDIR="${BASE_GENEVAL_DIR}/${EXP_NAME}"
DPG_OUTDIR="${DPG_SAVE_BASE}/${EXP_NAME}"

mkdir -p "$GENEVAL_OUTDIR" "$DPG_OUTDIR" 

echo "→ GENEVAL_OUTDIR: $GENEVAL_OUTDIR"
echo "→ DPG_OUTDIR:     $DPG_OUTDIR"



echo "----------------------------------------------------"
echo " Evaluating Geneval directory:"
echo "   $GENEVAL_OUTDIR"
echo "----------------------------------------------------"

MASK2FORMER_PATH="/inspire/hdd/project/chineseculture/public/yuxuan/benches/geneval/mask2former"

STEP_NAME=$(basename "$GENEVAL_OUTDIR")
OUTFILE_PARENT=$(dirname "$GENEVAL_OUTDIR")
GENEVAL_OUTFILE="${OUTFILE_PARENT}/results_${STEP_NAME}.jsonl"

python evaluation/evaluate_images.py \
    "$GENEVAL_OUTDIR" \
    --outfile "$GENEVAL_OUTFILE" \
    --model-path "$MASK2FORMER_PATH" \

python evaluation/summary_scores.py \
    "$GENEVAL_OUTFILE"

echo "🎉 Geneval evaluation finished: $STEP_NAME"
echo

