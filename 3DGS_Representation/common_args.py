"""Argument parsing and Parser construction shared by the experiment entry points."""

import argparse
import os

# Set CHROMA_GS_DATA / CHROMA_GS_COLOR once instead of passing --data_root and
# --color_root on every command; an explicit flag still wins over both.
DEFAULT_DATA_ROOT = os.environ.get("CHROMA_GS_DATA", "data/chromadistill")
DEFAULT_COLOR_ROOT = os.environ.get("CHROMA_GS_COLOR", "data/colour")
HERE = os.path.dirname(os.path.abspath(__file__))


def base_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("scene", help="LLFF scene name, e.g. fern")
    p.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    p.add_argument(
        "--color_root",
        default=DEFAULT_COLOR_ROOT,
        help="release holding the real colour frames, used only for scoring "
        "(and as the target for --target color)",
    )
    p.add_argument("--out_dir", default=None)
    p.add_argument("--cache_dir", default=os.path.join(HERE, "cache"))
    p.add_argument("--factor", type=int, default=4)
    p.add_argument("--test_every", type=int, default=8)
    p.add_argument("--no_normalize", action="store_true")
    p.add_argument("--no_align_axes", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p


def build_parser_and_images(args, verbose=True):
    """Construct the COLMAP Parser and load the rectified image stacks."""
    from chroma_gs.colmap import Parser

    data_dir = os.path.join(args.data_root, args.scene)
    color_ref_dir = (
        os.path.join(args.color_root, args.scene) if args.color_root else None
    )
    if color_ref_dir and not os.path.isdir(color_ref_dir):
        if verbose:
            print(f"[data] no colour reference at {color_ref_dir}; scoring grey only")
        color_ref_dir = None

    parser = Parser(
        data_dir=data_dir,
        color_ref_dir=color_ref_dir,
        factor=args.factor,
        normalize=not args.no_normalize,
        align_axes=not args.no_align_axes,
        test_every=args.test_every,
    )
    images = parser.load_images(cache_dir=args.cache_dir, verbose=verbose)
    return parser, images


def default_out_dir(args, kind):
    """ckpt_<kind>/llff/<scene>, mirroring the Plenoxels side's ckpt_* convention."""
    return args.out_dir or os.path.join(HERE, f"ckpt_{kind}", "llff", args.scene)
