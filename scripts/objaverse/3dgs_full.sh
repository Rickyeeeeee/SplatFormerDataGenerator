# In nerfstudio
export CUDA_VISIBLE_DEVICES=9
for obj in $(ls /project/ricky/splatformer-data/trainset_sr)
do
for df in 1 2 4
do
colmap_dir=/project/ricky/splatformer-data/overfit-data/colmap/$obj
output_dir=/project/ricky/splatformer-data/overfit-data/nerfstudio/
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
        colmap \
        --downscale_factor=${df} \
        --load_3D_points True \
        --auto_scale_poses=False --orientation_method=none --center_method=none \
        --load_bbox True --num_points_from_bbox 50000 \
        --assume_colmap_world_coordinate_convention False
done
done
