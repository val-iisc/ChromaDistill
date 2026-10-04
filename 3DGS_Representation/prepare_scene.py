#!/usr/bin/env python
"""Turn any COLMAP scene with colour images into the layout this pipeline expects.

The released ChromaDistill LLFF data ships the monochrome inputs and the teacher's
colorizations pre-made. Nothing else does, so running on a new dataset means producing
them. This does that end to end:

    <src>/images/            colour frames + <src>/sparse/0
        |
        |-- copy ----------> <colour_root>/<scene>/images/        scoring reference only
        |-- desaturate ----> <data_root>/<scene>/images/          the monochrome input
        |-- BigColor ------> <data_root>/<scene>/images_bigcolor/ the teacher
        '-- link ---------->  <data_root>/<scene>/sparse/

Images are kept at their original resolution, because COLMAP's intrinsics describe that
resolution and the loader derives everything by integer division from it. Choose the
working resolution at training time with --factor, not here.

BigColor runs in its own conda environment (it needs torch 1.x), so it is invoked as a
subprocess rather than imported. Two of its behaviours matter:

  * It colorizes at 256px and then upsamples only the predicted *chroma*, recombining it
    with the original full-resolution luminance (see `fusion` in colorize_real.py). The
    teacher therefore stays pixel-aligned with the monochrome input, which the Lab loss
    depends on, and its luminance is identical to the input's by construction.
  * It writes one file per top-k ImageNet class, named <stem>_c<class>.jpg. We ask for
    topk=1 and rename back to <stem>.jpg so the teacher pairs with the input by stem.

    python prepare_scene.py --src /data/ankit/Dataset/TNT/tandt/truck --scene truck
"""

import argparse
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# BigColor needs torch 1.x, so it lives in its own environment and is run as a
# subprocess. Point these at your checkout and its interpreter, via flags or env vars.
BIGCOLOR_DIR = os.environ.get("BIGCOLOR_DIR", "../BigColor")
BIGCOLOR_PY = os.environ.get("BIGCOLOR_PYTHON", "python")
EXTS = (".jpg", ".JPG", ".jpeg", ".png", ".PNG")


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.endswith(EXTS))


def step_copy_and_desaturate(src_images, colour_dir, grey_dir, overwrite):
    os.makedirs(colour_dir, exist_ok=True)
    os.makedirs(grey_dir, exist_ok=True)
    names = list_images(src_images)
    if not names:
        raise SystemExit(f"no images found in {src_images}")

    made = 0
    for name in names:
        stem = os.path.splitext(name)[0]
        out_colour = os.path.join(colour_dir, name)
        out_grey = os.path.join(grey_dir, f"{stem}.png")
        if not overwrite and os.path.isfile(out_colour) and os.path.isfile(out_grey):
            continue
        img = cv2.imread(os.path.join(src_images, name), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"could not read {name}")
        if not os.path.isfile(out_colour) or overwrite:
            shutil.copyfile(os.path.join(src_images, name), out_colour)
        # PNG for the monochrome input: it is the measurement every later stage is
        # anchored to, and a second JPEG generation would put artefacts into it.
        cv2.imwrite(out_grey, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        made += 1
    print(f"[prep] {len(names)} frames  (+{made} written)  colour -> {colour_dir}")
    print(f"[prep] {' ' * len(str(len(names)))}          grey   -> {grey_dir}")
    return names


def step_link_sparse(src, scene_dir, overwrite):
    src_sparse = os.path.join(src, "sparse")
    if not os.path.isdir(src_sparse):
        raise SystemExit(f"no COLMAP reconstruction at {src_sparse}")
    dst = os.path.join(scene_dir, "sparse")
    if os.path.islink(dst) or os.path.isdir(dst):
        if not overwrite:
            print(f"[prep] sparse/ already present")
            return
        (os.unlink if os.path.islink(dst) else shutil.rmtree)(dst)
    os.symlink(os.path.abspath(src_sparse), dst)
    print(f"[prep] sparse/ -> {src_sparse}")


def step_bigcolor(grey_dir, teacher_dir, device, topk, size, overwrite,
                  bigcolor_dir=BIGCOLOR_DIR, bigcolor_py=BIGCOLOR_PY):
    """Run BigColor over the monochrome frames, then rename to match input stems."""
    os.makedirs(teacher_dir, exist_ok=True)
    expected = {os.path.splitext(n)[0] for n in list_images(grey_dir)}
    have = {os.path.splitext(n)[0] for n in list_images(teacher_dir)}
    if not overwrite and expected and expected <= have:
        print(f"[prep] teacher already complete ({len(have)} frames)")
        return

    cmd = [
        bigcolor_py, "-W", "ignore", "colorize_real.py",
        "--path_ckpt", "ckpts/bigcolor",
        "--path_input", os.path.abspath(grey_dir),
        "--path_output", os.path.abspath(teacher_dir),
        "--epoch", "11",
        "--size_target", str(size),
        "--type_resize", "powerof",
        "--topk", str(topk),
        "--seed", "-1",
        "--use_ema",
        "--device", device,
    ]
    print(f"[prep] BigColor: {' '.join(cmd[2:])}")
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    env.pop("PYTHONPATH", None)  # the gsplat overlay must not leak into torch 1.x
    r = subprocess.run(cmd, cwd=bigcolor_dir, env=env)
    if r.returncode != 0:
        raise SystemExit(f"BigColor failed with exit code {r.returncode}")

    # <stem>_c<class>.jpg -> <stem>.jpg
    renamed = 0
    for name in list_images(teacher_dir):
        stem, ext = os.path.splitext(name)
        if "_c" not in stem:
            continue
        base = stem.rsplit("_c", 1)[0]
        dst = os.path.join(teacher_dir, base + ext)
        if os.path.abspath(dst) == os.path.abspath(os.path.join(teacher_dir, name)):
            continue
        os.replace(os.path.join(teacher_dir, name), dst)
        renamed += 1
    print(f"[prep] teacher -> {teacher_dir}  ({renamed} renamed to input stems)")


def step_verify(colour_dir, grey_dir, teacher_dir, sample=12):
    """Counts, sizes and the alignment the Lab loss depends on."""
    c, g, t = (list_images(d) for d in (colour_dir, grey_dir, teacher_dir))
    print(f"[prep] counts  colour {len(c)}  grey {len(g)}  teacher {len(t)}")
    if not (len(c) == len(g) == len(t)):
        raise SystemExit("count mismatch between colour, grey and teacher")

    stems = [os.path.splitext(n)[0] for n in g]
    missing = [s for s in stems if not any(
        os.path.isfile(os.path.join(teacher_dir, s + e)) for e in EXTS)]
    if missing:
        raise SystemExit(f"{len(missing)} frames have no teacher, e.g. {missing[:3]}")

    diffs, bad = [], []
    for stem in stems[:: max(1, len(stems) // sample)]:
        gi = cv2.imread(os.path.join(grey_dir, stem + ".png"), cv2.IMREAD_GRAYSCALE)
        tp = next(os.path.join(teacher_dir, stem + e) for e in EXTS
                  if os.path.isfile(os.path.join(teacher_dir, stem + e)))
        ti = cv2.imread(tp, cv2.IMREAD_COLOR)
        if gi.shape[:2] != ti.shape[:2]:
            bad.append((stem, gi.shape[:2], ti.shape[:2]))
            continue
        ty = cv2.cvtColor(ti, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diffs.append(float(np.abs(gi.astype(np.float32) - ty).mean()))
    if bad:
        raise SystemExit(f"grey/teacher size mismatch, e.g. {bad[:2]}")
    print(f"[prep] grey vs teacher luminance: mean |diff| = {np.mean(diffs):.3f}/255 "
          f"over {len(diffs)} sampled frames")
    print(f"[prep] resolution: {gi.shape[1]}x{gi.shape[0]}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", required=True,
                   help="COLMAP scene directory holding images/ and sparse/")
    p.add_argument("--scene", default=None, help="defaults to the src folder name")
    p.add_argument("--data_root",
                   default=os.environ.get("CHROMA_GS_DATA", "data/chromadistill"))
    p.add_argument("--colour_root",
                   default=os.environ.get("CHROMA_GS_COLOR", "data/colour"))
    p.add_argument("--bigcolor_dir", default=BIGCOLOR_DIR,
                   help="BigColor checkout (holds colorize_real.py and ckpts/)")
    p.add_argument("--bigcolor_python", default=BIGCOLOR_PY,
                   help="interpreter for the BigColor environment (torch 1.x)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--topk", type=int, default=1,
                   help="BigColor emits one image per class; 1 keeps the top choice")
    p.add_argument("--size", type=int, default=256, help="BigColor working resolution")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip_bigcolor", action="store_true")
    args = p.parse_args()

    scene = args.scene or os.path.basename(os.path.normpath(args.src))
    src_images = os.path.join(args.src, "images")
    if not os.path.isdir(src_images):
        raise SystemExit(f"no images/ under {args.src}")

    scene_dir = os.path.join(args.data_root, scene)
    grey_dir = os.path.join(scene_dir, "images")
    teacher_dir = os.path.join(scene_dir, "images_bigcolor")
    colour_dir = os.path.join(args.colour_root, scene, "images")
    os.makedirs(scene_dir, exist_ok=True)

    print(f"[prep] scene '{scene}' from {args.src}")
    step_copy_and_desaturate(src_images, colour_dir, grey_dir, args.overwrite)
    step_link_sparse(args.src, scene_dir, args.overwrite)
    if not args.skip_bigcolor:
        step_bigcolor(grey_dir, teacher_dir, args.device, args.topk, args.size,
                      args.overwrite, args.bigcolor_dir, args.bigcolor_python)
        step_verify(colour_dir, grey_dir, teacher_dir)

    print(f"\n[prep] ready. Train with:")
    print(f"  ./scripts/run_scene.sh {scene} 0 30000   # after setting:")
    print(f"    CHROMA_GS_DATA={args.data_root}")
    print(f"    CHROMA_GS_COLOR={args.colour_root}")


if __name__ == "__main__":
    main()
