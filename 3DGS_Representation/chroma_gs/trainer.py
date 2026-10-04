"""The shared photometric fit: L1 + SSIM with gsplat's default densification.

Three of the runs in this folder are the same optimisation against a different target
image, so they share this loop:

  baseline       3 channels, target = the teacher's colorizations  (Exp 1)
  colour bound   3 channels, target = the real colour frames        (Exp 1, upper bound)
  stage 1        1 channel,  target = the monochrome inputs         (Exp 2)
  grey control   3 channels, target = the monochrome inputs         (Exp 2 control)

Stage 2 does not use this loop -- it freezes the geometry and optimises whole-image
perceptual losses instead, so it lives in train_stage2.py.
"""

import json
import os
import time
from dataclasses import dataclass, field

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from gsplat.strategy import DefaultStrategy

from . import metrics as M
from .model import create_splats, rasterize, save_checkpoint


@dataclass
class FitConfig:
    target: str = "teacher"  # which image stack supervises: grey | teacher | color
    channels: int = 3
    sh_degree: int = 3
    sh_degree_interval: int = 1000  # raise the active SH degree every N steps
    max_steps: int = 30000
    batch_size: int = 1
    ssim_lambda: float = 0.2
    init_extra_random: int = 0
    init_scale: float = 1.0
    init_opacity: float = 0.1
    packed: bool = False
    absgrad: bool = False
    antialiased: bool = False
    refine_stop_iter: int = 15000
    grow_grad2d: float = 2e-4
    prune_opa: float = 5e-3
    eval_steps: list = field(default_factory=lambda: [7000, 30000])
    save_steps: list = field(default_factory=lambda: [30000])
    lrs: dict = field(default_factory=dict)
    seed: int = 42


def _target_key(cfg):
    assert cfg.target in ("grey", "teacher", "color"), cfg.target
    return cfg.target


def _get_target(batch, cfg):
    """The supervision image, NCHW, matching the configured channel count."""
    img = batch[_target_key(cfg)]  # [B, H, W, C]
    img = img.permute(0, 3, 1, 2)
    if cfg.channels == 1 and img.shape[1] == 3:
        # Supervising a 1-channel field with a colour stack is a configuration error;
        # grey is already stored single-channel.
        raise ValueError(
            f"target '{cfg.target}' is 3-channel but channels=1; use target='grey'"
        )
    if cfg.channels == 3 and img.shape[1] == 1:
        img = img.expand(-1, 3, -1, -1)
    return img.contiguous()


def fit(parser, images, cfg, out_dir, device="cuda", verbose=True):
    """Run the photometric fit and return the final evaluation metrics."""
    from .colmap import Dataset

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "renders"), exist_ok=True)

    trainset = Dataset(parser, "train", images=images)
    testset = Dataset(parser, "test", images=images)
    scene_scale = parser.scene_scale * 1.1

    params, optimizers, lrs = create_splats(
        parser.points,
        parser.points_rgb,
        scene_scale=scene_scale,
        channels=cfg.channels,
        sh_degree=cfg.sh_degree,
        init_opacity=cfg.init_opacity,
        init_scale=cfg.init_scale,
        init_extra_random=cfg.init_extra_random,
        lrs=cfg.lrs,
        device=device,
    )
    if verbose:
        print(
            f"[fit] {cfg.target} target, {cfg.channels}ch, {params['means'].shape[0]} "
            f"gaussians, scene_scale={scene_scale:.3f}, {len(trainset)} train / "
            f"{len(testset)} test views, {parser.width}x{parser.height}"
        )

    strategy = DefaultStrategy(
        prune_opa=cfg.prune_opa,
        grow_grad2d=cfg.grow_grad2d,
        refine_stop_iter=cfg.refine_stop_iter,
        absgrad=cfg.absgrad,
        verbose=False,
    )
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    schedulers = {
        "means": torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=0.01 ** (1.0 / cfg.max_steps)
        )
    }

    loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        persistent_workers=False,
    )
    iterator = iter(loader)
    from fused_ssim import fused_ssim

    history = []
    t0 = time.time()
    for step in range(cfg.max_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        camtoworlds = batch["camtoworld"].to(device)
        Ks = batch["K"].to(device)
        target = _get_target(batch, cfg).to(device)
        active_sh = min(step // cfg.sh_degree_interval, cfg.sh_degree)

        colors, alphas, info = rasterize(
            params,
            camtoworlds,
            Ks,
            parser.width,
            parser.height,
            sh_degree=active_sh,
            channels=cfg.channels,
            packed=cfg.packed,
            absgrad=cfg.absgrad,
            rasterize_mode="antialiased" if cfg.antialiased else "classic",
        )
        pred = colors.permute(0, 3, 1, 2).clamp(0.0, 1.0)

        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

        l1 = F.l1_loss(pred, target)
        ssim_pred = pred if cfg.channels == 3 else pred.expand(-1, 3, -1, -1)
        ssim_target = target if cfg.channels == 3 else target.expand(-1, 3, -1, -1)
        ssim_val = fused_ssim(ssim_pred, ssim_target, padding="valid")
        loss = l1 * (1.0 - cfg.ssim_lambda) + (1.0 - ssim_val) * cfg.ssim_lambda

        loss.backward()

        if verbose and step % 500 == 0:
            print(
                f"  step {step:6d} loss {loss.item():.5f} l1 {l1.item():.5f} "
                f"ssim {ssim_val.item():.4f} n={params['means'].shape[0]} "
                f"[{time.time() - t0:.0f}s]"
            )

        strategy.step_post_backward(
            params, optimizers, strategy_state, step, info, packed=cfg.packed
        )

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        for sched in schedulers.values():
            sched.step()

        if (step + 1) in cfg.eval_steps or step + 1 == cfg.max_steps:
            stats = evaluate(
                parser, testset, params, cfg, out_dir, step + 1, device=device
            )
            stats["step"] = step + 1
            stats["num_gaussians"] = int(params["means"].shape[0])
            history.append(stats)
            if verbose:
                print(f"  [eval @ {step + 1}] " + json.dumps(stats))

        if (step + 1) in cfg.save_steps or step + 1 == cfg.max_steps:
            save_checkpoint(
                os.path.join(out_dir, "ckpt.pt"),
                params,
                meta={
                    "step": step + 1,
                    "channels": cfg.channels,
                    "sh_degree": cfg.sh_degree,
                    "target": cfg.target,
                    "scene_scale": scene_scale,
                    "width": parser.width,
                    "height": parser.height,
                    "transform": parser.transform.tolist(),
                },
            )

    elapsed = time.time() - t0
    result = {
        "config": {k: v for k, v in cfg.__dict__.items()},
        "history": history,
        "final": history[-1] if history else {},
        "train_seconds": elapsed,
        "num_gaussians": int(params["means"].shape[0]),
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(result, f, indent=2)
    if verbose:
        print(f"[fit] done in {elapsed / 60:.1f} min -> {out_dir}")
    return result


@torch.no_grad()
def evaluate(parser, testset, params, cfg, out_dir, step, device="cuda", lpips=None):
    """Score the held-out views against every reference the dataset provides.

    A luminance field is scored against the grey inputs only. A colour field is scored
    against the real colour frames when they are available, and against the grey inputs
    as well, since matching luminance is the part the monochrome capture actually
    measures.
    """
    acc = M.Accumulator()
    if lpips is None and cfg.channels == 3:
        try:
            lpips = M.Lpips(device=device)
        except Exception as exc:  # torchvision/torchmetrics unavailable
            print(f"  [eval] LPIPS unavailable, skipping it ({exc.__class__.__name__})")
            lpips = False

    for item in range(len(testset)):
        batch = testset[item]
        camtoworlds = batch["camtoworld"][None].to(device)
        Ks = batch["K"][None].to(device)
        colors, _, _ = rasterize(
            params,
            camtoworlds,
            Ks,
            parser.width,
            parser.height,
            sh_degree=cfg.sh_degree,
            channels=cfg.channels,
        )
        pred = colors.permute(0, 3, 1, 2).clamp(0.0, 1.0)

        grey = batch["grey"][None].permute(0, 3, 1, 2).to(device)
        if cfg.channels == 1:
            acc.add("psnr_grey", M.psnr(pred, grey))
            acc.add("ssim_grey", M.ssim(pred, grey))
        else:
            pred_grey = (
                0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
            )
            acc.add("psnr_grey", M.psnr(pred_grey, grey))
            acc.add("ssim_grey", M.ssim(pred_grey, grey))
            if "color" in batch:
                ref = batch["color"][None].permute(0, 3, 1, 2).to(device)
                acc.add("psnr", M.psnr(pred, ref))
                acc.add("ssim", M.ssim(pred, ref))
                if lpips:
                    acc.add("lpips", lpips(pred, ref))

        img = pred[0].permute(1, 2, 0).cpu().numpy()
        acc.add("sharpness", M.laplacian_variance(img))
        acc.add("colourfulness", M.colourfulness(img))

        if step is not None:
            name = os.path.splitext(batch["name"])[0]
            save = (img * 255).astype(np.uint8)
            imageio.imwrite(
                os.path.join(out_dir, "renders", f"{name}_{step}.png"),
                save.squeeze() if save.shape[-1] == 1 else save,
            )
    return acc.means()
