#!/usr/bin/env python
"""Experiment 1 -- the colorize-then-fit baseline, and the colour upper bound.

Plain 3DGS, single stage. The training target is the teacher network's 2D
colorizations, so the radiance field never sees the monochrome inputs and no
distillation happens: whatever the teacher got wrong, and whatever it got
inconsistently between views, the Gaussians have to reconcile by themselves. This is
the 3DGS analogue of opt/try_llff_baseline.sh.

`--target color` instead fits the real colour frames. That is not a method, it is the
ceiling: it separates how much the colorization costs from how much the backbone costs,
which makes the result table readable.

    python train_baseline.py fern                    # colorize -> 3DGS baseline
    python train_baseline.py fern --target color     # real-colour upper bound
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
        "--target",
        default="teacher",
        choices=["teacher", "color"],
        help="teacher = the colorized views (the baseline); color = real colour (bound)",
    )
    p.add_argument("--max_steps", type=int, default=30000)
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--ssim_lambda", type=float, default=0.2)
    p.add_argument("--init_extra_random", type=int, default=0)
    p.add_argument("--antialiased", action="store_true")
    p.add_argument("--eval_steps", type=int, nargs="*", default=[7000])
    args = p.parse_args()

    parser, images = build_parser_and_images(args)
    if args.target == "color" and "color" not in images:
        raise SystemExit(
            f"--target color needs the real colour frames; none found under "
            f"{args.color_root}/{args.scene}"
        )

    kind = "baseline" if args.target == "teacher" else "colorbound"
    out_dir = default_out_dir(args, kind)

    cfg = FitConfig(
        target=args.target,
        channels=3,
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
    print(f"[{kind}] {args.scene} final={json.dumps(result['final'])}")


if __name__ == "__main__":
    main()
