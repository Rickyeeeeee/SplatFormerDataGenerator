PATH_TO_OBJAVERSE='/project/ricky/splatformer-data/glbs/000-000'
PATH_TO_OUTPUT='/project/ricky/splatformer-data/overfit-data/colmap'
obj_id='000b76f2b03e44e8ab44e1a1614be0f4'
CUDA_VISIBLE_DEVICES=9 blender-3.2.2-linux-x64/blender --background --python render_full.py \
    -- --object_path=${PATH_TO_OBJAVERSE}/$obj_id.glb \
    --output_folder=${PATH_TO_OUTPUT}/$obj_id \
    --resolution=512 \
    --train_views=32 \
    --test_elevation_range=20-90 \
    --train_elevation_sin_amplitude_max_levels=15 \
    --test_num_per_floor=3 \
    --use_gpu