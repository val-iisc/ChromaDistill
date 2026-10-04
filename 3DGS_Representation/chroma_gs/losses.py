"""The stage-2 distillation objective.

A port of opt/nnfm_loss.py with loss_names=["lab_loss_a","lab_loss_b","lab_loss_l",
"content_loss"], which is what opt_style.py actually runs:

  * L1 on the CIELab a, b and L channels between the render and the teacher
    colorization -- this is the chroma that gets distilled;
  * a VGG16 feature loss between the render and the *monochrome input*, which anchors
    structure and luminance to the real measurements rather than to the teacher;
  * total variation on the render.

The sRGB -> Lab conversion is carried over arithmetic-for-arithmetic from
opt/nnfm_loss.py (device hardcoding and the fixed NCHW batch index removed) so the two
backbones optimise the same colour space and their numbers stay comparable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# VGG16 feature layer indices per block, as in opt/nnfm_loss.py.
VGG_BLOCK_INDEXES = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]

_RGB_TO_XYZ = (
    (0.412453, 0.212671, 0.019334),
    (0.357580, 0.715160, 0.119193),
    (0.180423, 0.072169, 0.950227),
)
_FXFYFZ_TO_LAB = (
    (0.0, 500.0, 0.0),
    (116.0, -500.0, 200.0),
    (0.0, 0.0, -200.0),
)


def rgb_to_lab(srgb):
    """sRGB in [0, 1] with a trailing channel of 3 -> CIELab, shape preserved."""
    device, dtype = srgb.device, srgb.dtype
    pixels = srgb.reshape(-1, 3)

    linear_mask = (pixels <= 0.04045).to(dtype)
    exponential_mask = (pixels > 0.04045).to(dtype)
    rgb = (pixels / 12.92) * linear_mask + (
        ((pixels + 0.055) / 1.055) ** 2.4
    ) * exponential_mask

    xyz = rgb @ torch.tensor(_RGB_TO_XYZ, device=device, dtype=dtype)
    xyz = xyz * torch.tensor(
        [1 / 0.950456, 1.0, 1 / 1.088754], device=device, dtype=dtype
    )

    epsilon = 6.0 / 29.0
    linear_mask = (xyz <= epsilon**3).to(dtype)
    exponential_mask = (xyz > epsilon**3).to(dtype)
    fxfyfz = (xyz / (3 * epsilon**2) + 4.0 / 29.0) * linear_mask + (
        (xyz + 1e-6) ** (1.0 / 3.0)
    ) * exponential_mask

    lab = fxfyfz @ torch.tensor(
        _FXFYFZ_TO_LAB, device=device, dtype=dtype
    ) + torch.tensor([-16.0, 0.0, 0.0], device=device, dtype=dtype)
    return lab.reshape(srgb.shape)


def preprocess_lab(lab):
    """Split Lab into L, a, b each rescaled to roughly [-1, 1]. Expects [..., 3]."""
    L_chan, a_chan, b_chan = torch.unbind(lab, dim=-1)
    return L_chan / 50.0 - 1.0, a_chan / 110.0, b_chan / 110.0


def total_variation(img):
    """Isotropic TV on an NCHW image."""
    h_var = torch.mean((img[:, :, :-1, :] - img[:, :, 1:, :]) ** 2)
    w_var = torch.mean((img[:, :, :, :-1] - img[:, :, :, 1:]) ** 2)
    return (h_var + w_var) / 2.0


class DistillationLoss(nn.Module):
    """Lab chroma transfer from the teacher + VGG structure anchor to the grey input.

    Args:
        vgg_block: which VGG16 block supplies the content features (opt_style.py
            defaults to 2 -> layers [11, 13, 15]). A list activates several blocks.
        content_weight: try_llff.sh passes 1e-3.
        tv_weight: opt_style.py's --img_tv_weight, default 1.
        scales: the image scales the Lab and content terms are evaluated at. (0.5,)
            reproduces opt_style.py exactly; (1.0, 0.5, 0.25) is the multi-scale
            variant. The weighted sum is normalized by the total weight, so the loss
            magnitude -- and therefore the usable learning rate -- does not change
            with the number of scales.
    """

    def __init__(
        self,
        device,
        vgg_block=2,
        content_weight=1e-3,
        tv_weight=1.0,
        lab_weights=(1.0, 1.0, 1.0),
        scales=(0.5,),
        scale_weights=None,
        l_target="teacher",
    ):
        super().__init__()
        assert l_target in ("teacher", "grey"), l_target
        self.l_target = l_target
        self.blocks = [vgg_block] if isinstance(vgg_block, int) else sorted(vgg_block)
        self.layers = sorted(sum((VGG_BLOCK_INDEXES[b] for b in self.blocks), []))
        self.content_weight = content_weight
        self.tv_weight = tv_weight
        self.lab_weights = lab_weights
        self.scales = tuple(scales)
        weights = scale_weights or [1.0] * len(self.scales)
        assert len(weights) == len(self.scales), "one weight per scale"
        total = float(sum(weights))
        self.scale_weights = tuple(w / total for w in weights)

        self.vgg = (
            torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
            .features[: max(self.layers) + 1]
            .eval()
            .to(device)
        )
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def _feats(self, x):
        """Concatenated features from the activated layers. x is NCHW in [0, 1]."""
        x = self.normalize(x)
        out = []
        for ix, layer in enumerate(self.vgg):
            x = layer(x)
            if ix in self.layers:
                out.append(x)
        return out

    def forward(self, render, teacher, grey):
        """All inputs NCHW in [0, 1]; grey may be 1- or 3-channel.

        Returns a dict of scalar losses. Sum its values for the total.
        """
        if grey.shape[1] == 1:
            grey = grey.expand(-1, 3, -1, -1)

        losses = {
            "lab_loss_l": render.new_zeros(()),
            "lab_loss_a": render.new_zeros(()),
            "lab_loss_b": render.new_zeros(()),
            "content_loss": render.new_zeros(()),
        }

        for scale, weight in zip(self.scales, self.scale_weights):
            if scale == 1.0:
                r, t, g = render, teacher, grey
            else:
                kw = dict(scale_factor=scale, mode="bilinear", align_corners=False)
                r = F.interpolate(render, **kw)
                t = F.interpolate(teacher, **kw)
                g = F.interpolate(grey, **kw)

            # Lab terms operate on channel-last tensors.
            r_l, r_a, r_b = preprocess_lab(rgb_to_lab(r.permute(0, 2, 3, 1)))
            t_l, t_a, t_b = preprocess_lab(rgb_to_lab(t.permute(0, 2, 3, 1)))
            wl, wa, wb = self.lab_weights
            # The teacher's luminance is not the measured luminance -- on fern the two
            # differ by 8.7/255 -- so matching the teacher's L drags the field away
            # from the one quantity the monochrome capture actually establishes.
            # l_target='grey' matches the input instead; the a/b terms, which carry the
            # chroma the teacher is there to supply, always target the teacher.
            if self.l_target == "grey":
                l_ref, _, _ = preprocess_lab(rgb_to_lab(g.permute(0, 2, 3, 1)))
            else:
                l_ref = t_l
            losses["lab_loss_l"] += weight * wl * torch.mean(torch.abs(l_ref - r_l))
            losses["lab_loss_a"] += weight * wa * torch.mean(torch.abs(t_a - r_a))
            losses["lab_loss_b"] += weight * wb * torch.mean(torch.abs(t_b - r_b))

            if self.content_weight > 0:
                r_feats = self._feats(r)
                with torch.no_grad():
                    g_feats = self._feats(g)
                content = sum(
                    torch.mean((gf - rf) ** 2) for rf, gf in zip(r_feats, g_feats)
                )
                losses["content_loss"] += weight * self.content_weight * content

        if self.tv_weight > 0:
            # TV is a smoothness prior on the render itself, not a distillation term,
            # so it stays at full resolution outside the pyramid.
            losses["img_tv_loss"] = self.tv_weight * total_variation(render)
        return losses
