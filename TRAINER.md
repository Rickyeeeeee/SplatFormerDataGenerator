# `trainer.py` Guide

## Overview

`trainer.py` trains, evaluates, visualizes, exports, and optionally compresses a
3D Gaussian Splatting (3DGS) scene. It supports two Gaussian refinement
strategies, single- or multi-GPU execution, optional camera and appearance
optimization, optional depth supervision, bilateral-grid color correction,
checkpoint/PLY output, validation metrics, trajectory videos, and an interactive
viewer.

At a high level, the program performs this pipeline:

```text
CLI configuration
    -> parse COLMAP scene and build train/validation datasets
    -> initialize Gaussian parameters and one optimizer per parameter group
    -> initialize refinement strategy and optional modules
    -> repeatedly load images, rasterize, compute loss, and update parameters
    -> refine/prune Gaussians according to the selected strategy
    -> periodically checkpoint, export, evaluate, render video, and compress
```

If one or more checkpoints are supplied with `--ckpt`, training is skipped and
the program enters evaluation-only mode.

## Main structures

### `Config`

`Config` is the central dataclass used by the `tyro` command-line interface. Its
fields can be grouped as follows:

| Area | Important fields | Purpose |
|---|---|---|
| Run mode | `disable_viewer`, `ckpt`, `compression`, `render_traj_path` | Select training/evaluation behavior and output features. |
| Data | `data_dir`, `data_factor`, `test_every`, `patch_size`, `normalize_world_space` | Locate the COLMAP dataset, downsample it, split train/validation images, and optionally crop training patches. |
| Outputs | `result_dir`, `save_steps`, `eval_steps`, `save_ply`, `ply_steps`, `disable_video` | Control artifacts and their schedules. |
| Training | `batch_size`, `max_steps`, `steps_scaler`, `tb_every`, `tb_save_image` | Set iteration count, effective batch size, and TensorBoard logging. |
| Gaussian initialization | `init_type`, `init_num_pts`, `init_extent`, `init_opa`, `init_scale` | Initialize Gaussians from structure-from-motion points or random points. |
| Gaussian appearance | `sh_degree`, `sh_degree_interval`, `app_opt`, `app_embed_dim` | Use spherical harmonics directly or an appearance network. |
| Rendering | `camera_model`, `near_plane`, `far_plane`, `packed`, `antialiased`, `with_ut`, `with_eval3d` | Configure rasterization and camera projection. |
| Refinement | `strategy` | Choose `DefaultStrategy` or `MCMCStrategy` for densification/pruning. |
| Optimization | `means_lr`, `scales_lr`, `opacities_lr`, `quats_lr`, `sh0_lr`, `shN_lr` | Learning rates for individual Gaussian parameter groups. |
| Losses | `ssim_lambda`, `opacity_reg`, `scale_reg`, `depth_loss`, `depth_lambda` | Configure image, depth, opacity, and scale objectives. |
| Optional corrections | `pose_opt`, `pose_noise`, `use_bilateral_grid`, `use_fused_bilagrid`, `random_bkgd` | Optimize cameras, test noisy poses, color-correct renders, or randomize backgrounds. |
| Optimizer variants | `sparse_grad`, `visible_adam` | Use sparse gradients or update only visible Gaussians. |
| Evaluation | `lpips_net` | Select AlexNet- or VGG-based LPIPS. |
| Viewer | `port` | Select the interactive viewer server port. |

`Config.adjust_steps(factor)` scales the total steps, evaluation/save/PLY
schedules, spherical-harmonic schedule, and the iteration settings inside the
selected refinement strategy. This is especially useful for distributed runs,
where a larger effective batch often needs fewer optimizer steps.

### Gaussian model: `self.splats`

The scene itself is a `torch.nn.ParameterDict`. Every row describes one
Gaussian:

| Parameter | Shape | Stored representation |
|---|---:|---|
| `means` | `[N, 3]` | 3D center position. |
| `scales` | `[N, 3]` | Logarithm of the three axis scales. |
| `quats` | `[N, 4]` | Orientation quaternion; normalization occurs inside rasterization. |
| `opacities` | `[N]` | Opacity logits. |
| `sh0` | `[N, 1, 3]` | Degree-zero spherical-harmonic color term. Present when appearance optimization is off. |
| `shN` | `[N, K-1, 3]` | Higher-order spherical-harmonic coefficients. Present when appearance optimization is off. |
| `features` | `[N, 32]` | Learned appearance features. Present when appearance optimization is on. |
| `colors` | `[N, 3]` | Base RGB logits used with appearance features. Present when appearance optimization is on. |

The code keeps scale and opacity unconstrained during optimization and converts
them at render time with `exp(scales)` and `sigmoid(opacities)`.

### Optimizer structure

`create_splats_with_optimizers()` creates a separate optimizer for each Gaussian
parameter group. The default is Adam; `SparseAdam` is used with `sparse_grad`,
and `SelectiveAdam` is used with `visible_adam`.

Learning rates are multiplied by the square root of the effective batch size:

```text
effective batch size = batch_size * world_size
scaled learning rate = configured learning rate * sqrt(effective batch size)
```

The Gaussian centers additionally scale their learning rate by `scene_scale`.
Adam epsilon and beta values are also adjusted for the effective batch size.

Optional subsystems have separate optimizers:

- Camera-pose adjustment: one Adam optimizer with weight decay.
- Appearance optimization: one Adam optimizer for image embeddings and another
  for the color head.
- Bilateral grid: one Adam optimizer.

Only the Gaussian-center optimizer, camera optimizer, and bilateral-grid
optimizer have schedulers. Their learning rates decay exponentially to 1% of
their initial values; the bilateral grid first uses a 1,000-step linear warmup.

## Initialization

### Gaussian initialization

`create_splats_with_optimizers()` supports two modes:

1. `sfm`: uses the COLMAP points and point colors from `Parser`.
2. `random`: samples points uniformly inside an extent based on
   `init_extent * scene_scale` and assigns random RGB colors.

Each Gaussian's initial size is based on the root-mean-square distance to its
three nearest neighbors. Initial quaternions are random, and opacity is stored
as the logit of `init_opa`.

In distributed mode, Gaussians are divided across ranks using strided slicing
(`world_rank::world_size`). Distributed rasterization later combines their
contributions.

### `Runner.__init__()`

The runner performs these setup steps:

1. Seeds randomness with `42 + local_rank` and selects `cuda:<local_rank>`.
2. Creates `ckpts`, `stats`, `renders`, `ply`, and TensorBoard output locations.
3. Parses the COLMAP scene and creates training and validation datasets.
4. Computes a scene scale from the parser scale, a 1.1 margin, and
   `global_scale`.
5. Creates the Gaussian parameters and optimizers.
6. Validates and initializes the selected refinement strategy.
7. Initializes optional PNG compression, pose adjustment/noise, appearance
   optimization, and bilateral grids.
8. Creates PSNR, SSIM, and LPIPS metrics.
9. Starts the Viser/Gsplat viewer unless disabled.

## Rendering: `rasterize_splats()`

This method converts stored parameters into renderable values, derives either
spherical-harmonic colors or appearance-module colors, and calls gsplat's
`rasterization()` function.

Inputs include camera-to-world matrices, camera intrinsics, image dimensions,
optional masks, render mode, clipping planes, and viewer overrides. It returns:

- Rendered channels, such as RGB and optional depth.
- Accumulated alpha.
- An `info` dictionary containing visibility/radius/Gaussian-ID data used by
  refinement and specialized optimizers.

When a mask is present, pixels outside it are set to zero in the rendered
output. Rasterization can run in packed, sparse-gradient, antialiased,
distributed, unscented-transform, and 3D-evaluation modes.

## Training loop, step by step

`Runner.train()` first writes `cfg.yml`, builds schedulers and a shuffled
`DataLoader`, and then executes `max_steps` iterations.

### 1. Coordinate with the viewer

If the viewer is active, training waits while it is paused and acquires the
viewer lock before modifying the scene.

### 2. Load a batch

The loop continuously cycles through the training loader and moves the following
data to the GPU:

- Camera-to-world matrices.
- Intrinsic matrices.
- RGB images normalized from `[0, 255]` to `[0, 1]`.
- Image IDs.
- Optional masks.
- Optional sparse 2D sample points and ground-truth depths.

### 3. Adjust camera poses

Optional pose noise is applied first. If pose optimization is enabled, the
learned camera correction is then applied. When both are active, the reported
pose error measures how well optimization recovers the original pose.

### 4. Increase spherical-harmonic complexity

The active degree is:

```text
min(step // sh_degree_interval, configured sh_degree)
```

Training therefore begins with view-independent color and progressively enables
higher-order view-dependent terms.

### 5. Rasterize the scene

The current Gaussians are rendered from the batch cameras. The usual output is
RGB. With depth supervision, the requested output is RGB plus expected depth.

Optional post-render operations are:

- Slice the image-specific bilateral grid to correct RGB values.
- Composite a random background through the remaining transparency.

### 6. Run the refinement pre-backward hook

The chosen strategy receives the parameters, optimizers, strategy state, step,
and rasterizer metadata. It can prepare visibility and gradient statistics
needed for later densification or pruning.

### 7. Compute the objective

The base image objective is a weighted combination:

```text
image loss = (1 - ssim_lambda) * L1(render, target)
           + ssim_lambda * (1 - fused_SSIM(render, target))
```

Optional additions are:

- **Depth loss:** samples rendered expected depth at sparse target locations,
  converts both values to disparity, computes L1 loss, multiplies by scene scale,
  and applies `depth_lambda`.
- **Bilateral-grid smoothness:** ten times the grid's total-variation loss.
- **Opacity regularization:** mean activated opacity times `opacity_reg`.
- **Scale regularization:** mean activated scale times `scale_reg`.

### 8. Backpropagate and log

`loss.backward()` computes gradients. The progress bar shows total loss, active
SH degree, and optional depth/pose diagnostics. Rank zero periodically writes
losses, Gaussian count, peak CUDA memory, and optional comparison images to
TensorBoard.

### 9. Save scheduled artifacts

Saving is tested against `configured_step - 1` because the loop is zero-based.
The final iteration is always checkpointed. A checkpoint contains:

- Current zero-based step.
- Gaussian parameter state.
- Optional camera-adjustment state.
- Optional appearance-module state.

Training statistics are stored separately as JSON. If PLY export is enabled,
the Gaussian scene is also exported. For appearance-optimized models, appearance
is evaluated at the origin and baked into degree-zero color for the PLY.

### 10. Prepare specialized gradients or visibility

- Sparse-gradient mode converts dense gradients into sparse COO tensors using
  the Gaussian IDs returned by packed rasterization. It requires `packed=True`.
- Visible-Adam mode builds a mask of Gaussians visible in the current batch and
  updates only those entries.

### 11. Optimize and schedule

All enabled optimizers take a step and clear their gradients. All schedulers then
advance.

### 12. Run the refinement post-backward hook

The strategy can now split, duplicate, relocate, or prune Gaussians and update
optimizer state:

- `DefaultStrategy` receives packed-mode information.
- `MCMCStrategy` receives the current Gaussian-center learning rate.

This is why the number of Gaussians can change during training.

### 13. Evaluate, render, and compress

At configured evaluation iterations, the runner evaluates the validation set and
renders a camera-trajectory video. If compression is configured, it also
compresses, decompresses, replaces the active splats with the decompressed
version, and evaluates that version.

### 14. Refresh the viewer

Finally, the viewer lock is released, training throughput is calculated in rays
per second, and the interactive scene is updated.

## Evaluation: `eval()`

Evaluation renders every validation image with the maximum configured SH degree.
Rank zero saves target/render pairs and calculates mean:

- PSNR (higher is better).
- SSIM (higher is better).
- LPIPS (lower is better).
- Render time per image.
- Number of Gaussians.

When bilateral-grid correction is active, an additional offline color correction
is applied and `cc_psnr`, `cc_ssim`, and `cc_lpips` are reported. Results are
written to JSON and TensorBoard.

## Trajectory rendering: `render_traj()`

Unless video is disabled, this method builds one of three camera paths:

- `interp`: interpolates between dataset cameras.
- `ellipse`: generates an elliptical orbit at the cameras' mean height.
- `spiral`: generates a spiral using scene bounds and the dataset's configured
  radius scale.

It renders RGB plus expected depth, normalizes depth per frame, places RGB and
grayscale depth side by side, and writes a 30 FPS MP4 to `videos/traj_<step>.mp4`.

## Compression: `run_compression()`

With `compression="png"`, `PngCompression` serializes the Gaussian parameters,
then reads them back. The decompressed tensors replace the current splats and are
evaluated under the `compress` stage. This path requires the optional `torchpq`
and `plas` dependencies.

Important: compression is not merely evaluated on a temporary copy; the active
model parameters are replaced by the decompressed values.

## Interactive viewer: `_viewer_render_fn()`

The viewer callback supports:

- RGB.
- Accumulated depth.
- Expected depth.
- Alpha.
- Preview and full viewer resolutions.
- Near/far normalization, inversion, and color maps.
- Background, radius clipping, 2D epsilon, rasterization-mode, camera-model, and
  maximum-SH controls.

It also reports total and currently rendered Gaussian counts. The viewer is
automatically disabled for distributed training. After a run completes, the
process remains alive to serve the viewer until interrupted.

## Entry point and run modes

The CLI exposes two presets:

- `default`: original-paper-style heuristic densification through
  `DefaultStrategy`.
- `mcmc`: MCMC refinement with different initial opacity/scale and nonzero
  opacity/scale regularization.

After parsing CLI overrides, the program scales schedules, conditionally imports
the standard or fused bilateral-grid implementation, validates compression
dependencies, checks that unscented-transform rendering also enables 3D
evaluation, and launches through gsplat's distributed CLI helper.

### Training mode

With no `--ckpt`, `main()` creates a `Runner` and calls `train()`.

### Evaluation-only mode

With `--ckpt`, each rank's Gaussian tensors are loaded and concatenated, then the
runner evaluates, renders a trajectory, and optionally tests compression. The
checkpoint's stored step is used for artifact names. Camera and appearance states
present in checkpoints are not restored by this branch; only splat tensors are
loaded.

## Output layout

```text
<result_dir>/
├── cfg.yml                         # Effective configuration
├── ckpts/ckpt_<step>_rank<r>.pt    # Per-rank checkpoints
├── stats/                          # Training/evaluation JSON summaries
├── renders/                        # Validation target/render images
├── ply/point_cloud_<step>.ply      # Optional Gaussian export
├── videos/traj_<step>.mp4          # Optional RGB/depth trajectory video
├── compression/rank<r>/            # Optional compressed representation
└── tb/                              # TensorBoard event files
```

## Implementation caveats in the current file

These behaviors follow directly from the current implementation and may deserve
fixes before using the affected features:

1. **Compression has an undefined variable.** `run_compression()` constructs its
   directory with `cfg.result_dir`, but does not define local `cfg`; it should use
   `self.cfg.result_dir` or assign `cfg = self.cfg`. As written, compression will
   raise `NameError`.
2. **The rasterizer ignores the resolved camera override.**
   `rasterize_splats()` computes its local `camera_model` value, including viewer
   overrides, but passes `self.cfg.camera_model` to `rasterization()`. Viewer or
   call-specific camera-model selections therefore have no effect.
3. **Evaluation-only mode does not restore optional modules.** Checkpoints save
   camera and appearance state when enabled, but `main()` restores only
   `splats`. Appearance-enabled evaluation can therefore use a newly initialized
   appearance module instead of the trained one.
4. **Compression permanently changes the active tensors.** Subsequent evaluation,
   video rendering, or viewer output uses the decompressed model.
5. **Depth-video normalization can divide by zero.** A constant-depth frame makes
   `depths.max() - depths.min()` zero because no epsilon is added in
   `render_traj()`.

## Short summary

`trainer.py` is a complete 3DGS experiment runner. It initializes a distributed
set of learnable Gaussians, differentiably renders batches, optimizes photometric
and optional auxiliary losses, and dynamically changes scene structure through a
densification strategy. Around that core loop it provides validation metrics,
checkpointing, PLY export, trajectory video generation, compression experiments,
TensorBoard telemetry, camera/appearance/color-correction modules, and an
interactive viewer.
