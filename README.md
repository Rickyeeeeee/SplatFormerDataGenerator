# SplatFormer-OOD-BlenderRender
This repository is used by [SplatFormer](https://github.com/ChenYutongTHU/SplatFormer) to generate OOD data. It contains 
* The blender rendering scripts to render OOD-NVS test and training set. 
* The nerfstudio script to train 3DGS on the rendered OOD-NVS sets.
You can also download **the prepared test set** [here](https://drive.google.com/file/d/1-mUCl-yxe1aE0rrQDHKlXk1J2n8d1-60/view?usp=sharing). 

## 1. GSplat Installation
```bash
conda create -n datagenerator-gsplat python=3.8 -y
conda activate datagenerator-gsplat

# Install the pytorch version for your cuda version.
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install gsplat
pip install gsplat==1.5.3 
pip install -r requirements.txt
cd ./pycolmap
python setup.py install
```

## 1. Installation
### Blender (for render)
```
wget https://download.blender.org/release/Blender2.90/blender-2.90.0-linux64.tar.xz # For ShapeNet
tar -xvf blender-2.90.0-linux64.tar.xz
wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz # For Objaverse-v1 and GSO
tar -xvf blender-3.2.2-linux-x64.tar.xz
```
### nerfstudio (for 3DGS training)
```
cd nerfstudio
pip install -e .
```

## 2. Rendering 3D Scenes
### ShapeNet
#### Download
Download ShapeNet-Core from [the huggingface repository](https://huggingface.co/datasets/ShapeNet/ShapeNetCore). Or directly
```
## In Python
from huggingface_hub import snapshot_download
download_dir = 'path/to/the/dir'
snapshot_download(repo_id="ShapeNet/ShapeNetCore",repo_type="dataset",local_dir=download_dir, cache_dir=download_dir)
```
We select ShapeNet objects with texture for OOD training and test. The training (29,587 objects) and test (20 objects) split can be found in [](traintest_splits).
#### Render the training set
To render each object for OOD training and put the camera poses and images in a colmap structure
```
obj_id='03046257/8c00d87bcc8f034aa1b95b258b5f7139' #Replace other scenes in OOD-BlenderRender/traintest_splits/shapenet_train.txt
blender-2.90.0-linux64/blender --background --python render_shapenet.py  \
    -- --object_path=${PATH_TO_SHAPENET}/${obj_id}/models/model_normalized.obj \
    --output_folder=render_outputs/shapenet/trainset/${obj_id} \
    --train_elevation_sin_amplitude_max_levels=10 \
    --test_num_per_floor=2 \
    --test_elevation_range=70-90 \
    --generate_trainset  --use_gpu
```
#### Render the test set
For OOD test sets, we render each object with two sets of elevation ranges, 10 and 20. We also render more test views with various elevation degree.
```
obj_id='02691156/2628b6cfcf1a53465569af4484881d20' #Replace other scenes in OOD-BlenderRender/traintest_splits/shapenet_test.txt
blender-2.90.0-linux64/blender --background --python render_shapenet.py  \
    -- --object_path=${PATH_TO_SHAPENET}/${obj_id}/models/model_normalized.obj \
    --output_folder=render_outputs/shapenet/trainset/${obj_id} \
    --train_elevation_sin_amplitude_max_levels=10,20 \
    --test_num_per_floor=3 \
    --test_elevation_range=20-90  \
    --use_gpu
```
In SplatFormer, we use only 48k scenes for training. The training (48,354 objects) and test (20 objects) split can be found in [traintest_splits](traintest_splits).
### Objaverse-v1
#### Download
Download objaverse-v1 from [the huggingface repository](https://huggingface.co/datasets/allenai/objaverse/tree/main). For the OOD training set, we used [000-000,...,000-039]. For the OOD evaluation set, we choose objects from [000-100](https://huggingface.co/datasets/allenai/objaverse/tree/main/glbs/000-100). To download these subsets, you can use [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli).
```
# Take 000-100 for examples
huggingface-cli download allenai/objaverse \
--include "glbs/000-100/*.glb" \
--repo-type dataset \
--token='your-huggging-face-token'  \
--use_gpu
```

#### Render training set
```
obj_id='fff5b37e11e74f00a6459b37451950b4'
blender-3.2.2-linux-x64/blender --background --python render.py \
    -- --object_path=${PATH_TO_OBJAVERSE}/$obj_id.glb \
    --output_folder=render_outputs/objaverse/trainset/$obj_id \
    --train_elevation_sin_amplitude_max_levels=15 \
    --test_num_per_floor=2 \
    --test_elevation_range=70-90 \
    --generate_trainset  \
    --use_gpu
```
#### Render Test set
```
obj_id='0a604c1ee9b245c7b2d797a910e53219'
blender-3.2.2-linux-x64/blender --background --python render.py \
    -- --object_path=${PATH_TO_OBJAVERSE}/$obj_id.glb \
    --output_folder=render_outputs/objaverse/testset/$obj_id \
    --train_views=32 \
    --train_elevation_sin_amplitude_max_levels=10-20 \
    --test_num_per_floor=3 \
    --test_elevation_range=70-90  \
    --use_gpu 
```

### Google Scanned Objects
#### Download
Download [google scanned objects](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research) and extract the files. We choose 20 objects and their names are listed in [traintest_splits/gso_test.txt](traintest_splits/gso_test.txt).

#### Render Test set
```
obj_id=Sushi_Mat
blender-3.2.2-linux-x64/blender --background --python render.py \
    -- --object_path=${PATH_TO_GSO}/${obj_id}/meshes/model.obj \
    --output_folder=render_outputs/gso/testset/$obj_id \
    --train_elevation_sin_amplitude_max_levels=10-20 \
    --test_num_per_floor=3 \
    --test_elevation_range=70-90 \
    --use_gpu
```

## 3. Train 3DGS on the OOD scenes
All the nerfstudio training scripts can be found in [scripts](scripts/). Here we take 3DGS of GSO-OOD test set for example.
```
export CUDA_VISIBLE_DEVICES=0 
for obj in $(ls render_outputs/shapenet/testset) # You can modify this to enable multi-gpu processing.
do
colmap_dir=render_outputs/gso/testset/$obj # path to the directory of rendered images
output_dir=nerfstudio_outputs/gso/testset/$obj
ns-train splatfacto \
        --logging.local-writer.enable=False --logging.profiler=none \
        --pipeline.datamanager.data=${colmap_dir} \
        --pipeline.model.sh_degree=1 \
        --pipeline.save_img=True --test_after_train True \
        --output_dir=./ --experiment-name=${output_dir} \
        --relative-model-dir=nerfstudio_models  --vis wandb \
        --max_num_iterations=10000 \
        colmap \
        --downscale_factor=1 \
        --load_3D_points True --load_bbox True --num_points_from_bbox 50000 \
        --auto_scale_poses=False --orientation_method=none --center_method=none \
        --assume_colmap_world_coordinate_convention False \
        --eval_mode filename
done
```
Note that when generating the input 3DGS for SplatFormer, we stop the optimization at the 10k step (--early_stop_steps=10000), and reduce sh_degree to save computational budgets. We find that this does not affect 3DGS's rendering performance in these datasets. However, we report the metrics of 3DGS with default configuration (--max_num_iterations=30000, --pipeline.model.sh_degree=3) in our paper's table.

### ShapeNetCore v2: gsplat data generation

Extract models under `/project/ricky/ShapeNetCore/<synset>/<model_id>/models/`,
including their OBJ, MTL, and texture files. The image and gsplat roots default
to `/project/ricky/shapenet-sr-data` and
`/project/ricky/shapenet-sr-data-mcmc-4096`. Outputs are independent of data
splits and follow this layout:

```text
<root>/shapenet/<resolution>/<colmap|gsplat>/<synset>/<model_id>/
```

Use the `srdatagenerator` environment for the installed capped gsplat strategy.
For example, process the chair category on two GPUs with two scenes per GPU:

```bash
conda activate srdatagenerator
python gen_data_shapenet.py --strategy default --categories 03001627 \
  --gpu-ids 0 1 --scenes-per-gpu 2
```

GPU IDs are physical NVIDIA indices and must be visible in any existing
`CUDA_VISIBLE_DEVICES` setting. Without GPU options, one idle visible GPU is
selected. `--gpu-id 0` selects one GPU; `--limit 1` processes one model for a
pilot. Each scene remains on one GPU, and per-scene file locks prevent
simultaneous writers. Failures are appended to `shapenet_failures.jsonl` in the
image root. Re-running skips complete renders and checkpoints.

Resume checks count PNG names and check for the requested gsplat checkpoint.
They do not check image sizes or require render metrics. Preprocessing also
counts PNG names and checks the checkpoint and render metrics for filtering.

The default is 128 views at 128 × 128, 15,000 training iterations and at most
4,096 splats. Use `--resolutions 256 128` to render the largest and downscale
others, `--train-steps` to change training length, and `--skip-gs` to render
only. MCMC is the default; `--strategy default` uses the capped DefaultStrategy
and the root `/project/ricky/shapenet-sr-data-default-4096`. The default root
reflects `--strategy` and `--max-splats`; either root can be explicitly set.

Preprocess the generated pool to filter scenes and replace the split CSVs:

```bash
python preprocess_shapenet.py --strategy default --resolutions 128 \
  --split-ratios 0.9 0.05 0.05 --seed 42
```

Preprocessing displays progress bars for discovery, scene checks, and training
checkpoint statistics, followed by per-resolution counts and the report path.

When using multiple resolutions, a scene must pass the completeness and PSNR
checks at every selected resolution. It receives the same split at each
resolution. `<gsplat-root>/shapenet/<resolution>/train_scenes.csv`,
`val_scenes.csv`, and `test_scenes.csv` contain scene IDs, category IDs, and
metrics. Aggregate statistics in `gs_statistics.json` are calculated only from
the training split. Add `--skip-statistics` to generate only the split CSVs and
counts; the statistics file is marked as skipped so old statistics are not
mistaken for the new split. Use the same `--strategy`, `--max-splats`, `--train-steps`, and root
options as generation. Splits are regenerated on every invocation.

`<gsplat-root>/shapenet/gen_data.json` records discovered, valid, PSNR-passing,
PSNR-rejected and invalid counts per resolution, the number passing at every
resolution, split counts, and the preprocessing configuration.

#### Train noise-to-GS on ShapeNet

Run preprocessing as above without `--skip-statistics` before training. The
config `configs/dataset/shapenet-sr.gin` uses the filtered train and test CSVs,
training statistics, and separate image and checkpoint roots. Override
`dataset_root`, `shapenet_gs_root`, the CSV paths, and
`flow_matching.gs_statistics_path` for other roots. Both resolutions are 128;
the source checkpoint is still read to normalize the coordinate frame.
Set `SplatFactoSRDataset.image_dataset_root` to override the image root
independently; it defaults to `dataset_root` in the ShapeNet layout.

From the repository root, choose a scene ID from `test_scenes.csv` and run:

```bash
SCENE='03001627/<model-id>'
python overfit-noise-to-gs.py --scene_name="$SCENE" \
  --gin_file=configs/model/dit_gaussian.gin \
  --gin_file=configs/dataset/shapenet-sr.gin \
  --gin_file=configs/overfit/noise_to_gs.gin \
  --gin_param='SplatFactoSRDataset.max_gs_num=1000'
```

For distributed training, point both checkpoint directory flags at the
unsplit gsplat directory and select an evaluation scene from the test CSV:

```bash
GS_DIR=/project/ricky/shapenet-sr-data-default-4096/shapenet/128/gsplat
torchrun --standalone --nproc_per_node=1 train-noise-to-gs.py \
  --train_splat_dir="$GS_DIR" --test_splat_dir="$GS_DIR" --eval_scene_name="$SCENE" \
  --gin_file=configs/model/dit_gaussian.gin \
  --gin_file=configs/dataset/shapenet-sr.gin \
  --gin_file=configs/train/noise_to_gs.gin
```

To evaluate against the validation split instead, add
`--gin_param="test_dataset/SplatFactoSRDataset.test_scene_list='/project/ricky/shapenet-sr-data-default-4096/shapenet/128/val_scenes.csv'"`
and choose a scene from that file. Overfit checkpoint root overrides point to
the dataset root, whereas the distributed trainer's `*_splat_dir` flags point
directly to the resolution's `gsplat` directory.

#### Copy outputs from the old split layout

Stop any generators using the old layout before copying. First check for scene
directory collisions across old splits or in the new destination:

```bash
python - <<'CHECK'
from pathlib import Path
import sys

datasets = [
    (Path('/project/ricky/shapenet-sr-data'), 'colmap'),
    (Path('/project/ricky/shapenet-sr-data-default-4096'), 'gsplat'),
]
collisions = []
for root, kind in datasets:
    seen = {}
    for prefix in ('train-set/shapenet', 'val-set/shapenet', 'test-set/shapenet', 'shapenet'):
        base = root / prefix
        if not base.is_dir():
            continue
        for scene in base.glob(f'*/{kind}/*/*'):
            if not scene.is_dir():
                continue
            key = (scene.parents[2].name, scene.parent.name, scene.name)
            if key in seen:
                collisions.append((seen[key], scene))
            else:
                seen[key] = scene
if collisions:
    for first, second in collisions:
        print(f'Collision: {first}\n           {second}')
    sys.exit('Resolve these collisions before copying.')
print('No scene-directory collisions found.')
CHECK
```

If there are no collisions, copy the data. This preserves the original files:

```bash
for pair in \
  '/project/ricky/shapenet-sr-data:colmap' \
  '/project/ricky/shapenet-sr-data-default-4096:gsplat'
do
  root="${pair%:*}"
  kind="${pair##*:}"
  for split in train-set val-set test-set
  do
    for resolution_dir in "$root/$split/shapenet/"*/
    do
      [ -d "$resolution_dir/$kind" ] || continue
      resolution="$(basename "$resolution_dir")"
      mkdir -p "$root/shapenet/$resolution/$kind"
      rsync -a --ignore-existing \
        "$resolution_dir/$kind/" "$root/shapenet/$resolution/$kind/"
    done
  done
done
```

Verify the copies before deleting any old directories. Repeat for other gsplat
strategy roots if needed.
