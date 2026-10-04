# ChromaDistill on 3D Gaussian Splatting

A 3D Gaussian Splatting implementation of ChromaDistill, built on
[gsplat](https://github.com/nerfstudio-project/gsplat).


The release also includes the single-stage **baseline**: fitting ordinary 3DGS on view-inconsistent colorized views from a pre-trained colorization model.

## Environment

gsplat 1.5.3 requires PyTorch 2.6 and a CUDA toolkit matching your driver. Training was
run on RTX 3090s (24 GB).

```bash
conda create -n chroma-gs python=3.10 -y
conda activate chroma-gs

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install gsplat==1.5.3
pip install numpy opencv-python imageio imageio-ffmpeg scipy plyfile tqdm
pip install fused-ssim pycolmap
```

Notes:

- `fused_ssim` is used for the baseline's SSIM term. If it is unavailable, the trainer
  falls back to the built-in implementation.
- `pycolmap` here is the lightweight `SceneManager` reader used to parse `sparse/0`.
- Stage 2's content loss uses torchvision's pretrained VGG16, downloaded on first use.
- If `import torch` fails with an `mpmath` error, install it: `pip install mpmath`. It
  is a declared dependency of `sympy` that some environments are missing, and without
  it `torch._dynamo` fails to import.

## Data

Each scene is a COLMAP reconstruction plus two image folders:

```
<data_root>/<scene>/
├── images/             monochrome captures          (the real measurements)
├── images_bigcolor/    per-view colorizations       (the teacher)
└── sparse/0/           COLMAP cameras, images, points3D
```

Use `prepare_scene.py` to prepare data:

```bash
export BIGCOLOR_DIR=/path/to/BigColor
export BIGCOLOR_PYTHON=/path/to/envs/bigcolor/bin/python

python prepare_scene.py --src /path/to/colmap_scene --scene truck \
    --data_root data/chromadistill --colour_root data/colour
```


Any colorizer works. We used [BigColor](https://github.com/KIMGEONUNG/BigColor).

## Training

Set the data roots once:

```bash
export CHROMA_GS_DATA=data/chromadistill
export CHROMA_GS_COLOR=data/colour      # real colour frames, for scoring only
```

### Baseline — colorize, then fit

```bash
python train_baseline.py <scene> --factor 4 --max_steps 30000
```

Plain 3DGS, SH degree 3, `L1 + 0.2 * SSIM` against the colorized images. Writes `ckpt_baseline/llff/<scene>/ckpt.pt`.

### Ours — two-stage multi-scale distillation

**Stage 1**, luminance-only field (required before stage 2):

```bash
python train_stage1.py <scene> --factor 4 --max_steps 30000
```

Writes `ckpt_stage1/llff/<scene>/ckpt.pt`.

**Stage 2**, multi-scale chroma distillation:

```bash
python train_stage2.py <scene> --factor 4 --ms_scales 1.0 0.5 0.25
```

Utilizes the stage-1 checkpoint, freezes `means`, `scales`, `quats`, `opacities`
and the stage-1 `shN`, and optimises `sh0` alone for 3000 steps. Writes
`ckpt_stage2_ms/llff/<scene>/ckpt.pt`.


### Useful flags

| flag | default | |
|---|---|---|
| `--factor` | 4 | image downscale; `1` for native resolution |
| `--data_root` / `--color_root` | `$CHROMA_GS_DATA` / `$CHROMA_GS_COLOR` | override per run |
| `--max_steps` | 30000 / 3000 | stage 1 / stage 2 |
| `--lr`, `--lr_final` | 1e-2, 1e-3 | stage-2 `sh0` learning rate schedule |
| `--ms_scales` | `0.5` | pass `1.0 0.5 0.25` for the multi-scale version |
| `--content_weight` | 1e-3 | raising this desaturates the result |
| `--device` | `cuda` | |

Rectified images are cached as lossless PNGs under `cache/<scene>_f<factor>/` on first
use; delete that folder if you change the source images.

## Citation

If you use this code, please cite the ChromaDistill paper.
