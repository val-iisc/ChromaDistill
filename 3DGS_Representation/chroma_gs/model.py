"""The Gaussian parameter set, its rasterization, and the stage-1 -> stage-2 promotion.

Three configurations share one container:

  channels=1  stage 1. A genuinely monochrome field: sh0 [N,1,1], shN [N,15,1]. One
              appearance degree of freedom per basis instead of three.
  channels=3  the colorize-then-fit baseline and the colour-GT upper bound. Ordinary
              3DGS: sh0 [N,1,3], shN [N,15,3].
  stage 2     sh0 [N,1,3] trainable for chroma, plus the stage-1 shN carried forward
              as a frozen, channel-shared luminance term. See promote_to_stage2.

Parameters are held in a plain ParameterDict with one optimizer per key, because that
is the convention gsplat's densification strategies require: they rebuild each
parameter and its optimizer state in place when Gaussians are split, duplicated or
pruned (gsplat/strategy/ops.py).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization, spherical_harmonics
from scipy.spatial import cKDTree

C0 = 0.28209479177387814  # the degree-0 spherical harmonic, Y_0^0


def rgb_to_sh(rgb):
    """Invert gsplat's clamp_min(sh_eval + 0.5, 0) for the DC term."""
    return (rgb - 0.5) / C0


def sh_to_rgb(sh):
    return sh * C0 + 0.5


def create_splats(
    points,
    rgbs,
    scene_scale,
    channels=3,
    sh_degree=3,
    init_opacity=0.1,
    init_scale=1.0,
    init_extra_random=0,
    lrs=None,
    device="cuda",
    sparse_grad=False,
):
    """Initialise Gaussians at the SfM points and build one optimizer per parameter.

    Args:
        points: (P, 3) SfM point positions, already in normalized world space.
        rgbs: (P, 3) uint8 point colours.
        scene_scale: the Parser's camera-spread scale; the position learning rate and
            the initial Gaussian extent are both expressed relative to it.
        channels: 1 for a luminance field, 3 for colour.
        init_extra_random: pad the initialisation with this many uniformly random
            points inside the SfM bounding box. Needed for scenes whose
            reconstruction is too sparse to densify from (leaves has 3294 points).
    """
    points = np.asarray(points, dtype=np.float32)
    rgbs = np.asarray(rgbs, dtype=np.float32) / 255.0

    if init_extra_random > 0:
        lo, hi = points.min(axis=0), points.max(axis=0)
        extra = np.random.uniform(lo, hi, size=(init_extra_random, 3)).astype(np.float32)
        points = np.concatenate([points, extra], axis=0)
        rgbs = np.concatenate(
            [rgbs, np.full((init_extra_random, 3), 0.5, dtype=np.float32)], axis=0
        )

    # Isotropic initial scale from the mean distance to the three nearest neighbours,
    # the usual 3DGS initialisation.
    dists, _ = cKDTree(points).query(points, k=4)
    dist_avg = np.maximum(dists[:, 1:].mean(axis=1), 1e-7)
    scales = np.log(dist_avg * init_scale)[:, None].repeat(3, axis=1)

    n = points.shape[0]
    num_bases = (sh_degree + 1) ** 2
    if channels == 1:
        # Luminance: the same Rec.601 weighting used to make the monochrome inputs.
        dc = rgbs @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        dc = rgb_to_sh(dc)[:, None, None]
    else:
        dc = rgb_to_sh(rgbs)[:, None, :]

    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0

    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.from_numpy(points)),
            "scales": torch.nn.Parameter(torch.from_numpy(scales.astype(np.float32))),
            "quats": torch.nn.Parameter(torch.from_numpy(quats)),
            "opacities": torch.nn.Parameter(
                torch.full((n,), math.log(init_opacity / (1 - init_opacity)))
            ),
            "sh0": torch.nn.Parameter(torch.from_numpy(dc.astype(np.float32))),
            "shN": torch.nn.Parameter(
                torch.zeros(n, num_bases - 1, channels, dtype=torch.float32)
            ),
        }
    ).to(device)

    default_lrs = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
    }
    default_lrs.update(lrs or {})
    optimizer_cls = torch.optim.SparseAdam if sparse_grad else torch.optim.Adam
    optimizers = {
        name: optimizer_cls(
            [{"params": params[name], "lr": default_lrs[name], "name": name}],
            eps=1e-15,
        )
        for name in params
    }
    return params, optimizers, default_lrs


def rasterize(
    params,
    camtoworlds,
    Ks,
    width,
    height,
    sh_degree,
    channels=3,
    shN_frozen=None,
    near_plane=0.01,
    far_plane=1e10,
    packed=False,
    absgrad=False,
    rasterize_mode="classic",
    background=None,
):
    """Render a batch of cameras.

    Args:
        camtoworlds: (C, 4, 4).
        sh_degree: the SH degree to activate this step (warm-up raises it over training).
        channels: 1 renders a single luminance channel, 3 renders colour.
        shN_frozen: stage 2 only. A frozen (N, K-1, 3) tensor supplying the
            view-dependent term, with params["sh0"] supplying the trainable chroma DC.

    Returns:
        (colors (C, H, W, channels), alphas (C, H, W, 1), info).
    """
    means = params["means"]
    quats = params["quats"]
    scales = torch.exp(params["scales"])
    opacities = torch.sigmoid(params["opacities"])
    viewmats = torch.linalg.inv(camtoworlds)

    if shN_frozen is not None:
        # Stage 2: trainable 3-channel DC + frozen channel-shared view dependence.
        coeffs = torch.cat([params["sh0"], shN_frozen], dim=1)  # [N, K, 3]
        colors, use_sh = coeffs, sh_degree
    elif channels == 1:
        # gsplat's spherical_harmonics hard-codes 3 output channels, so evaluate on a
        # broadcast view and keep one channel. SH evaluation is linear and independent
        # per channel, so this is exact, and the rasterizer then carries 1 channel
        # instead of 3. The +0.5 / clamp matches gsplat's internal SH path
        # (gsplat/rendering.py: clamp_min(colors + 0.5, 0)).
        coeffs = torch.cat([params["sh0"], params["shN"]], dim=1)  # [N, K, 1]
        dirs = means[None, :, :] - camtoworlds[:, None, :3, 3]  # [C, N, 3]
        shs = coeffs.expand(-1, -1, 3)[None].expand(dirs.shape[0], -1, -1, -1)
        lum = spherical_harmonics(sh_degree, dirs, shs)[..., :1]  # [C, N, 1]
        colors, use_sh = torch.clamp_min(lum + 0.5, 0.0), None
    else:
        coeffs = torch.cat([params["sh0"], params["shN"]], dim=1)  # [N, K, 3]
        colors, use_sh = coeffs, sh_degree

    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=use_sh,
        near_plane=near_plane,
        far_plane=far_plane,
        packed=packed,
        absgrad=absgrad,
        rasterize_mode=rasterize_mode,
        backgrounds=background,
    )
    return render_colors, render_alphas, info


def promote_to_stage2(params, device="cuda"):
    """Turn a trained stage-1 luminance field into the stage-2 chroma parameterisation.

    Stage 1 holds sh0 [N,1,1] and shN [N,15,1]. Stage 2 wants a trainable 3-channel DC
    for chroma plus the stage-1 view-dependent term kept as it is:

        colour_c(d) = Y_0 * sh0_c  +  sum_{k>=1} Y_k(d) * shN_k

    sh0 starts as the luminance DC replicated across RGB, so the step-0 render is
    identical to the stage-1 luminance render -- nothing is destroyed and no MSE
    warm-up is needed to recover the luminance, unlike the Plenoxels stage 2 where
    reset_basis_dim=1 discarded the SH outright.

    shN is channel-shared and frozen, so it is materialised once as a [N,15,3] buffer
    rather than broadcast on every forward pass.

    Returns:
        (params, shN_frozen) where params holds only the geometry (frozen) and the
        trainable sh0, and shN_frozen is a detached [N, K-1, 3] tensor.
    """
    assert params["sh0"].shape[-1] == 1, (
        f"expected a 1-channel stage-1 checkpoint, got sh0 {tuple(params['sh0'].shape)}"
    )
    sh0 = params["sh0"].detach().expand(-1, -1, 3).contiguous().to(device)
    shN_frozen = params["shN"].detach().expand(-1, -1, 3).contiguous().to(device)

    promoted = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(params["means"].detach().clone(), requires_grad=False),
            "scales": torch.nn.Parameter(params["scales"].detach().clone(), requires_grad=False),
            "quats": torch.nn.Parameter(params["quats"].detach().clone(), requires_grad=False),
            "opacities": torch.nn.Parameter(
                params["opacities"].detach().clone(), requires_grad=False
            ),
            "sh0": torch.nn.Parameter(sh0.clone(), requires_grad=True),
        }
    ).to(device)
    return promoted, shN_frozen


def save_checkpoint(path, params, meta, shN_frozen=None):
    state = {k: v.detach().cpu() for k, v in params.items()}
    if shN_frozen is not None:
        state["shN_frozen"] = shN_frozen.detach().cpu()
    torch.save({"splats": state, "meta": meta}, path)


def load_checkpoint(path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["splats"]
    shN_frozen = state.pop("shN_frozen", None)
    params = torch.nn.ParameterDict(
        {k: torch.nn.Parameter(v.to(device)) for k, v in state.items()}
    )
    if shN_frozen is not None:
        shN_frozen = shN_frozen.to(device)
    return params, ckpt["meta"], shN_frozen
