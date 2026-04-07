#!/bin/bash

GPU_ID=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
    | awk '$1 == 0 {print NR-1}' \
    | head -n1)

if [ -z "$GPU_ID" ]; then
    echo "[ERROR] No available GPU found. Exiting."
    exit 1
fi

echo "Using GPU: $GPU_ID"

PATH_TO_OBJAVERSE='/project/ricky/objaverse/glbs/000-000'
PATH_TO_OUTPUT='/project/ricky/splatformer-data/trainset-256/colmap'

for obj_path in ${PATH_TO_OBJAVERSE}/*.glb; do
    obj_file=$(basename "$obj_path")
    obj_id="${obj_file%.glb}"

    echo "Processing $obj_id"

    CUDA_VISIBLE_DEVICES=$GPU_ID blender-3.2.2-linux-x64/blender --background --python render_full.py \
        -- --object_path="$obj_path" \
        --output_folder="${PATH_TO_OUTPUT}/${obj_id}" \
        --resolution=512 \
        --train_views=256 \
        --test_elevation_range=0-90 \
        --train_elevation_sin_amplitude_max_levels=15 \
        --test_num_per_floor=3 \
        --use_gpu
done