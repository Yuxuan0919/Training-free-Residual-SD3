#!/bin/bash
set -euo pipefail

# =============== 阶段 0：环境 ===============
source /inspire/hdd/project/chineseculture/public/yuxuan/miniconda3/etc/profile.d/conda.sh
conda activate flux_taca
cd /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/TACA-Infer

MODEL='flux'
NFE=50
CFG=3.5
IMGSIZE=1024
BATCHSIZE=16


WORLD_SIZE=8



BASE_GENEVAL_DIR="/inspire/hdd/project/chineseculture/public/yuxuan/benches/geneval/outputs_flux_TACA"
DPG_SAVE_BASE="/inspire/hdd/project/chineseculture/public/yuxuan/benches/ELLA/dpg_bench/outputs_flux_TACA"
mkdir -p  "$BASE_GENEVAL_DIR" "$DPG_SAVE_BASE"


GENEVAL_DIR_LIST=()
DPG_DIR_LIST=()    


EXP_NAME="TACA-lora16"

GENEVAL_OUTDIR="${BASE_GENEVAL_DIR}/${EXP_NAME}"
DPG_OUTDIR="${DPG_SAVE_BASE}/${EXP_NAME}"

mkdir -p "$GENEVAL_OUTDIR" "$DPG_OUTDIR" 

echo "→ GENEVAL_OUTDIR: $GENEVAL_OUTDIR"
echo "→ DPG_OUTDIR:     $DPG_OUTDIR"




# echo "📌 Running GenEval bench generation on 8 GPUs..."
# for GPU_ID in $(seq 0 $((WORLD_SIZE-1))); do
#     CUDA_VISIBLE_DEVICES=$GPU_ID python generate_geneval_flux.py \
#         --lora_weights "/inspire/hdd/project/chineseculture/public/yuxuan/TACA/TACA/flux-dev-lora-rank-16.safetensors" \
#         --guidance_scale $CFG \
#         --num_inference_steps $NFE \
#         --img_size $IMGSIZE \
#         --batch_size $BATCHSIZE \
#         --outdir "$GENEVAL_OUTDIR" \
#         --world_size $WORLD_SIZE \
#         --rank $GPU_ID \
#         > "${GENEVAL_OUTDIR}/log_gpu${GPU_ID}.txt" 2>&1 &
# done

# wait   
# echo "🎉 GenEval generation finished."
# echo


echo "📌 Running DPG bench generation on 8 GPUs..."
for GPU_ID in $(seq 0 $((WORLD_SIZE-1))); do
    CUDA_VISIBLE_DEVICES=$GPU_ID python generate_dpg_flux.py \
        --lora_weights "/inspire/hdd/project/chineseculture/public/yuxuan/TACA/TACA/flux-dev-lora-rank-16.safetensors" \
        --guidance_scale $CFG \
        --num_inference_steps $NFE \
        --img_size $IMGSIZE \
        --save_dir "$DPG_OUTDIR" \
        --world_size $WORLD_SIZE \
        --rank $GPU_ID \
        > "${DPG_OUTDIR}/log_gpu${GPU_ID}.txt" 2>&1 &
done

wait   
echo "🎉 DPG generation finished."
echo

