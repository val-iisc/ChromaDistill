#!/usr/bin/env python
"""Experiment 2 -- stage 1: can a Gaussian field learn luminance alone?

Fits a genuinely monochrome radiance field to the greyscale inputs: one appearance
degree of freedom per SH basis instead of three (sh0 [N,1,1], shN [N,15,1]). This is
the 3DGS analogue of the first ChromaDistill stage, which fits Plenoxels to the
monochrome capture before any colour is introduced.

The claim being tested is that dropping to one channel costs nothing in reconstruction
quality, so `--channels 3` runs the control: the identical model and schedule
supervised by the same grey images but carrying full RGB appearance. If the two agree
on held-out grey PSNR/SSIM, luminance-only is free.

    python train_stage1.py fern                   # 1-channel luminance field
    python train_stage1.py fern --channels 3      # 3-channel control on grey targets
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chroma_gs.trainer import FitConfig, fit
from common_args import base_parser, build_parser_and_images, default_out_dir


def main():
    p = base_parser(__doc__.split("\n")[0])
    p.add_argument(
        "--channels",
        type=int,
        default=1,
        choices=[1, 3],
        help="1 for the luminance field, 3 for the control",
    )
    p.add_argument("--max_steps", type=int, default=30000)
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--ssim_lambda", type=float, default=0.2)
    p.add_argument("--init_extra_random", type=int, default=0)
    p.add_argument("--antialiased", action="store_true")
    p.add_argument("--eval_steps", type=int, nargs="*", default=[7000])
    args = p.parse_args()

    parser, images = build_parser_and_images(args)
    kind = "stage1" if args.channels == 1 else "stage1_control3ch"
    out_dir = default_out_dir(args, kind)

    cfg = FitConfig(
        target="grey",
        channels=args.channels,
        sh_degree=args.sh_degree,
        max_steps=args.max_steps,
        ssim_lambda=args.ssim_lambda,
        init_extra_random=args.init_extra_random,
        antialiased=args.antialiased,
        eval_steps=list(args.eval_steps),
        save_steps=[args.max_steps],
        seed=args.seed,
    )
    result = fit(parser, images, cfg, out_dir, device=args.device)

    # The whole point of the experiment: how much appearance storage this buys.
    result["appearance_floats_per_gaussian"] = (args.sh_degree + 1) ** 2 * args.channels
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"[stage1] {args.scene} channels={args.channels} "
        f"appearance floats/gaussian="
        f"{result['appearance_floats_per_gaussian']} "
        f"final={json.dumps(result['final'])}"
    )


if __name__ == "__main__":
    main()
