#!/usr/bin/env python
"""Experiments 3 and 4 -- chroma distillation into a frozen Gaussian geometry.

Takes a trained stage-1 luminance field and distils colour into it from the teacher's
2D colorizations, with the geometry held fixed. This is the port of opt/opt_style.py.

What is frozen and what is learned:

    means, scales, quats, opacities    frozen        (Plenoxels: lr_sigma = 0)
    shN  (view dependence)             frozen, as learned on luminance in stage 1
    sh0  (the DC term)                 trained, 3 channels -- this is the chroma

so colour_c(d) = Y_0 * sh0_c + sum_{k>=1} Y_k(d) * shN_k. The chroma is
view-independent, which is the property that makes a single consistent 3D colour the
only way to satisfy all the views at once; the view-dependent luminance stage 1 learned
is preserved rather than discarded. Densification is off: no Gaussian is created,
removed or moved, so any change in the render is colour and nothing else.

Because nothing is destroyed at promotion -- sh0 starts as the stage-1 luminance DC
replicated across RGB -- the step-0 render equals the stage-1 render exactly, and the
MSE warm-up opt_style.py needed (--mse_num_epoches 2, to recover luminance after
reset_basis_dim=1 wiped the SH) is unnecessary here. The script asserts that identity
before training rather than assuming it.

One deliberate departure from the port, measured on fern: opt_style.py matches the
Lab L channel to the *teacher's* luminance, but the teacher's luminance is not the
measured luminance -- on fern the two differ by 8.7/255 -- so that term drags the field
off the one quantity the monochrome capture establishes. The faithful setting gave up
1.58 dB of the grey PSNR stage 1 had already earned. Pointing the L term at the
monochrome input instead (--l_target grey, now the default) is better on every axis:

    stage-1 reference            grey PSNR 25.14   sharpness 869
    --l_target teacher (port)    grey PSNR 23.56   sharpness 811   colour PSNR 20.82
    --l_target grey  (default)   grey PSNR 24.84   sharpness 843   colour PSNR 21.35

Dropping the L term entirely (--lab_weights 0 1 1) recovers less grey PSNR (24.53) and
costs a lot of sharpness (764), so the term is doing useful work -- it was just aimed at
the wrong target. Raising --content_weight to 1e-1 scores best of all on PSNR/SSIM/LPIPS
but halves colourfulness (26.8 -> 13.3): it wins the reference metrics by distilling
less colour, which is the metric pathology this project has already been bitten by once.
Hence the default is l_target=grey at content_weight 1e-3, not the best-scoring row.

Experiment 4 is the same run with the distillation loss evaluated on an image pyramid:

    python train_stage2.py fern                              # Exp 3, single scale 0.5
    python train_stage2.py fern --ms_scales 1.0 0.5 0.25     # Exp 4, multi-scale
"""

import json
import os
import sys
import time

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chroma_gs.colmap import Dataset
from chroma_gs.losses import DistillationLoss
from chroma_gs.metrics import Accumulator, colourfulness, laplacian_variance, psnr, ssim
from chroma_gs.model import load_checkpoint, promote_to_stage2, rasterize, save_checkpoint
from common_args import base_parser, build_parser_and_images, default_out_dir


def exp_lr(step, max_steps, lr_init, lr_final):
    """Log-linear decay, the same shape Plenoxels used for lr_sh (1e-1 -> 1e-2)."""
    if max_steps <= 1:
        return lr_final
    t = step / (max_steps - 1)
    return float(np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t))


def render_view(params, batch, parser, sh_degree, shN_frozen, device, channels=3):
    camtoworlds = batch["camtoworld"].to(device)
    Ks = batch["K"].to(device)
    if camtoworlds.dim() == 2:
        camtoworlds, Ks = camtoworlds[None], Ks[None]
    colors, _, _ = rasterize(
        params,
        camtoworlds,
        Ks,
        parser.width,
        parser.height,
        sh_degree=sh_degree,
        channels=channels,
        shN_frozen=shN_frozen,
    )
    return colors.permute(0, 3, 1, 2)  # NCHW, unclamped


@torch.no_grad()
def check_promotion(stage1_params, params, shN_frozen, parser, batch, device, sh_degree):
    """The stage-2 render at step 0 must reproduce the stage-1 luminance render."""
    before = render_view(
        stage1_params, batch, parser, sh_degree, None, device, channels=1
    )
    after = render_view(params, batch, parser, sh_degree, shN_frozen, device)
    # after is grey at this point, so compare against any one channel.
    diff = (after[:, 0:1] - before).abs().max().item()
    spread = (after.amax(dim=1) - after.amin(dim=1)).abs().max().item()
    return diff, spread


@torch.no_grad()
def evaluate(params, shN_frozen, parser, testset, sh_degree, out_dir, step, device,
             lpips=None):
    acc = Accumulator()
    os.makedirs(os.path.join(out_dir, "renders"), exist_ok=True)
    for item in range(len(testset)):
        batch = testset[item]
        pred = render_view(params, batch, parser, sh_degree, shN_frozen, device).clamp(0, 1)
        grey = batch["grey"][None].permute(0, 3, 1, 2).to(device)
        pred_grey = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        acc.add("psnr_grey", psnr(pred_grey, grey))
        acc.add("ssim_grey", ssim(pred_grey, grey))
        if "color" in batch:
            ref = batch["color"][None].permute(0, 3, 1, 2).to(device)
            acc.add("psnr", psnr(pred, ref))
            acc.add("ssim", ssim(pred, ref))
            if lpips:
                acc.add("lpips", lpips(pred, ref))
        img = pred[0].permute(1, 2, 0).cpu().numpy()
        acc.add("sharpness", laplacian_variance(img))
        acc.add("colourfulness", colourfulness(img))
        if step is not None:
            name = os.path.splitext(batch["name"])[0]
            imageio.imwrite(
                os.path.join(out_dir, "renders", f"{name}_{step}.png"),
                (img * 255).astype(np.uint8),
            )
    return acc.means()


def main():
    p = base_parser("chroma distillation into a frozen Gaussian geometry")
    p.add_argument(
        "--init_ckpt",
        default=None,
        help="stage-1 checkpoint; defaults to ckpt_stage1/llff/<scene>/ckpt.pt",
    )
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-2, help="initial sh0 learning rate")
    p.add_argument("--lr_final", type=float, default=1e-3)
    p.add_argument("--content_weight", type=float, default=1e-3,
                   help="matches try_llff.sh")
    p.add_argument("--img_tv_weight", type=float, default=1.0,
                   help="opt_style.py's --img_tv_weight")
    p.add_argument("--vgg_block", type=int, nargs="+", default=[2])
    p.add_argument("--lab_weights", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                   metavar=("L", "A", "B"))
    p.add_argument("--l_target", default="grey", choices=["teacher", "grey"],
                   help="what the Lab L term matches. 'grey' (default) targets the "
                        "monochrome input, which is measured; 'teacher' reproduces "
                        "opt_style.py exactly but costs ~1.3 dB of grey PSNR -- see "
                        "the module docstring")
    p.add_argument("--ms_scales", type=float, nargs="+", default=[0.5],
                   help="[0.5] reproduces opt_style.py; [1.0 0.5 0.25] is Exp 4")
    p.add_argument("--ms_weights", type=float, nargs="+", default=None)
    p.add_argument("--eval_steps", type=int, nargs="*", default=[])
    p.add_argument("--warmup_iters", type=int, default=0,
                   help="optional L1-to-grey warm-up; not needed, see the docstring")
    p.add_argument("--tag", default=None, help="suffix for the output directory")
    p.add_argument("--skip_promotion_check", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    parser, images = build_parser_and_images(args)
    multi_scale = len(args.ms_scales) > 1
    tag = args.tag or ("stage2_ms" if multi_scale else "stage2")
    out_dir = default_out_dir(args, tag)
    os.makedirs(out_dir, exist_ok=True)

    init_ckpt = args.init_ckpt or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ckpt_stage1", "llff", args.scene, "ckpt.pt",
    )
    if not os.path.isfile(init_ckpt):
        raise SystemExit(
            f"no stage-1 checkpoint at {init_ckpt}; run train_stage1.py {args.scene} first"
        )
    stage1_params, meta, _ = load_checkpoint(init_ckpt, device=device)
    if meta.get("channels") != 1:
        raise SystemExit(
            f"{init_ckpt} is a {meta.get('channels')}-channel checkpoint; stage 2 "
            "expects the 1-channel luminance field from train_stage1.py"
        )
    sh_degree = meta["sh_degree"]
    n = stage1_params["means"].shape[0]
    print(
        f"[stage2] {args.scene}: {n} gaussians from {init_ckpt} "
        f"(stage-1 step {meta['step']}), scales={args.ms_scales}"
    )

    params, shN_frozen = promote_to_stage2(stage1_params, device=device)

    trainset = Dataset(parser, "train", images=images)
    testset = Dataset(parser, "test", images=images)

    if not args.skip_promotion_check:
        diff, spread = check_promotion(
            stage1_params, params, shN_frozen, parser, trainset[0], device, sh_degree
        )
        print(f"[stage2] promotion check: max|stage2 - stage1| = {diff:.2e}, "
              f"max channel spread = {spread:.2e}")
        assert diff < 1e-4, (
            f"promotion is not lossless (max diff {diff:.3e}); the step-0 render should "
            "reproduce the stage-1 luminance render exactly"
        )
        assert spread < 1e-4, f"promoted render is not grey (channel spread {spread:.3e})"

    del stage1_params

    loss_fn = DistillationLoss(
        device,
        vgg_block=args.vgg_block if len(args.vgg_block) > 1 else args.vgg_block[0],
        content_weight=args.content_weight,
        tv_weight=args.img_tv_weight,
        lab_weights=tuple(args.lab_weights),
        scales=tuple(args.ms_scales),
        scale_weights=args.ms_weights,
        l_target=args.l_target,
    )
    optimizer = torch.optim.Adam([params["sh0"]], lr=args.lr, eps=1e-15)

    try:
        from chroma_gs.metrics import Lpips

        lpips = Lpips(device=device)
    except Exception as exc:
        print(f"[stage2] LPIPS unavailable ({exc.__class__.__name__}), skipping it")
        lpips = None

    before = evaluate(params, shN_frozen, parser, testset, sh_degree, out_dir,
                      step=0, device=device, lpips=lpips)
    print(f"[stage2] before: {json.dumps(before)}")

    order = list(range(len(trainset)))
    history = []
    t0 = time.time()
    for step in range(args.max_steps):
        if step % len(order) == 0:
            np.random.shuffle(order)
        batch = trainset[order[step % len(order)]]

        lr = exp_lr(step, args.max_steps, args.lr, args.lr_final)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Unclamped, as in opt_style.py: the rasterizer's output is non-negative
        # already (gsplat clamps the SH colour at 0), and clamping the top end would
        # zero the gradient on exactly the saturated pixels that need correcting.
        pred = render_view(params, batch, parser, sh_degree, shN_frozen, device)
        teacher = batch["teacher"][None].permute(0, 3, 1, 2).to(device)
        grey = batch["grey"][None].permute(0, 3, 1, 2).to(device)

        if step < args.warmup_iters:
            losses = {"warmup_l1": torch.nn.functional.l1_loss(
                pred, grey.expand(-1, 3, -1, -1))}
        else:
            losses = loss_fn(pred, teacher, grey)
        loss = sum(losses.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            parts = " ".join(f"{k.replace('_loss', '')}={v.item():.4f}"
                             for k, v in losses.items())
            print(f"  step {step:5d} lr {lr:.2e} loss {loss.item():.4f}  {parts} "
                  f"[{time.time() - t0:.0f}s]")

        if (step + 1) in args.eval_steps:
            stats = evaluate(params, shN_frozen, parser, testset, sh_degree, out_dir,
                             step=step + 1, device=device, lpips=lpips)
            stats["step"] = step + 1
            history.append(stats)
            print(f"  [eval @ {step + 1}] {json.dumps(stats)}")

    final = evaluate(params, shN_frozen, parser, testset, sh_degree, out_dir,
                     step=args.max_steps, device=device, lpips=lpips)
    final["step"] = args.max_steps
    history.append(final)
    elapsed = time.time() - t0

    save_checkpoint(
        os.path.join(out_dir, "ckpt.pt"),
        params,
        meta={
            "step": args.max_steps,
            "channels": 3,
            "sh_degree": sh_degree,
            "target": "teacher",
            "stage": tag,
            "ms_scales": args.ms_scales,
            "init_ckpt": init_ckpt,
            "width": parser.width,
            "height": parser.height,
            "scene_scale": meta.get("scene_scale"),
            "transform": parser.transform.tolist(),
        },
        shN_frozen=shN_frozen,
    )
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(
            {
                "config": vars(args),
                "before": before,
                "history": history,
                "final": final,
                "train_seconds": elapsed,
                "num_gaussians": n,
            },
            f,
            indent=2,
        )
    print(f"[stage2] after:  {json.dumps(final)}")
    print(f"[stage2] done in {elapsed / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
