#!/bin/bash

GPU_ID=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
    | awk '$1 == 0 {print NR-1}' \
    | head -n1)

# GPU_ID=5

if [ -z "$GPU_ID" ]; then
    echo "[ERROR] No available GPU found. Exiting."
    exit 1
fi

echo "Using GPU: $GPU_ID"

PATH_TO_OBJAVERSE='/project/ricky/objaverse/glbs/000-004'
PATH_TO_OUTPUT='/project/ricky/splatformer-data/trainset-256/colmap'

for obj_path in ${PATH_TO_OBJAVERSE}/*.glb; do
    obj_file=$(basename "$obj_path")
    obj_id="${obj_file%.glb}"
    obj_output_dir="${PATH_TO_OUTPUT}/${obj_id}"
    images_dir="${obj_output_dir}/images"
    image_count=0

    echo "Processing $obj_id"

    if [ -d "$images_dir" ]; then
        image_count=$(find "$images_dir" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d '[:space:]')
    fi

    if [ "$image_count" -ge 256 ]; then
        echo "[SKIP] ${obj_id}: found ${image_count} images in ${images_dir}, skipping render."
    else
        CUDA_VISIBLE_DEVICES=$GPU_ID blender-3.2.2-linux-x64/blender --background --python render_full.py \
            -- --object_path="$obj_path" \
            --output_folder="${obj_output_dir}" \
            --resolution=512 \
            --train_views=256 \
            --test_elevation_range=0-90 \
            --train_elevation_sin_amplitude_max_levels=15 \
            --test_num_per_floor=3 \
            --use_gpu
    fi

    # In nerfstudio
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    for df in 1 2 4
    do
    colmap_dir=/project/ricky/splatformer-data/trainset-256/colmap/${obj_id}
    output_dir=/project/ricky/splatformer-data/trainset-256/nerfstudio/
    experiment_dir=${obj_id}/df-${df}
    ns-train splatfacto \
            --logging.local-writer.enable=False --logging.profiler=none \
            --pipeline.datamanager.data=${colmap_dir} \
            --pipeline.model.sh_degree=1 \
            --pipeline.save_img=True --test_after_train True \
            --output_dir=$output_dir --experiment-name=${experiment_dir} \
            --relative-model-dir=nerfstudio_models  --vis viewer+tensorboard \
            --steps_per_eval_image=100000 --steps_per_eval_all_images=1000000 --max_num_iterations=30000 \
            --save_only_latest_checkpoint False  --steps_per_save=100000 --save_last_checkpoint True \
            --early_stop_steps=10000 \
            --save_only_gs_params True \
            --viewer.quit-on-train-completion True \
            colmap \
            --downscale_factor=${df} \
            --downscale-rounding-mode=floor \
            --load_3D_points True \
            --eval-mode all \
            --auto_scale_poses=False --orientation_method=none --center_method=none \
            --load_bbox True --num_points_from_bbox 50000 \
            --assume_colmap_world_coordinate_convention False
    done
done
