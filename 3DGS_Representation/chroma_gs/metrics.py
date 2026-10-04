"""Image metrics for the held-out views.

PSNR / SSIM / LPIPS, plus Laplacian variance. The sharpness column is not decoration:
in the Plenoxels evaluation the single-stage baselines beat the two-stage pipeline on
full-image PSNR purely by rendering a blurrier image, which full-reference metrics
reward under slight misalignment. Any table produced here reports sharpness alongside
the reference metrics so that failure mode is visible in the numbers.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def psnr(pred, target):
    """pred/target NCHW in [0, 1]."""
    mse = F.mse_loss(pred, target).item()
    return float("inf") if mse == 0 else -10.0 * np.log10(mse)


def ssim(pred, target):
    """Structural similarity via fused_ssim; falls back to skimage on 1-channel input."""
    from fused_ssim import fused_ssim

    if pred.shape[1] == 1:
        pred = pred.expand(-1, 3, -1, -1)
        target = target.expand(-1, 3, -1, -1)
    return float(fused_ssim(pred, target, padding="valid", train=False).item())


class Lpips:
    """Lazily built LPIPS (AlexNet backbone, weights come from the local torch cache)."""

    def __init__(self, device="cuda", net="alex"):
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.metric = LearnedPerceptualImagePatchSimilarity(
            net_type=net, normalize=True
        ).to(device)

    def __call__(self, pred, target):
        if pred.shape[1] == 1:
            pred = pred.expand(-1, 3, -1, -1)
            target = target.expand(-1, 3, -1, -1)
        return float(self.metric(pred.clamp(0, 1), target.clamp(0, 1)).item())


def laplacian_variance(img):
    """Sharpness proxy. img is HWC or HW in [0, 1]; returns variance on a 0-255 scale."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        arr = (arr.squeeze() * 255).astype(np.uint8)
    return float(cv2.Laplacian(arr, cv2.CV_64F).var())


def colourfulness(img):
    """Hasler-Suesstrunk colourfulness, for spotting a field that stayed grey."""
    arr = (np.asarray(img) * 255).astype(np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return 0.0
    rg = arr[..., 0] - arr[..., 1]
    yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
    return float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2)
        + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )


class Accumulator:
    """Running means keyed by metric name."""

    def __init__(self):
        self.sums, self.counts = {}, {}

    def add(self, name, value):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return
        self.sums[name] = self.sums.get(name, 0.0) + float(value)
        self.counts[name] = self.counts.get(name, 0) + 1

    def means(self):
        return {k: self.sums[k] / self.counts[k] for k in self.sums}
