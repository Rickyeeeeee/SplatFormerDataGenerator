#!/bin/bash

GPU_ID=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
    | awk '$1 == 0 {print NR-1}' \
    | head -n1)

if [ -z "$GPU_ID" ]; then
    echo "[ERROR] No available GPU found. Exiting."
    exit 1
fi

echo "Using GPU: $GPU_ID"

# In nerfstudio
export CUDA_VISIBLE_DEVICES=$GPU_ID
for obj in $(ls /project/ricky/splatformer-data/trainset-256/colmap)
do
for df in 1 2 4
do
colmap_dir=/project/ricky/splatformer-data/trainset-256/colmap/$obj
output_dir=/project/ricky/splatformer-data/trainset-256/nerfstudio/
experiment_dir=${obj}/df-${df}
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
