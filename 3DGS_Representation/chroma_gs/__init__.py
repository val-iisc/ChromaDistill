"""ChromaDistill on a 3D Gaussian Splatting backbone (gsplat).

`losses` is deliberately not imported here. It pulls in torchvision for the VGG content
loss, which only stage 2 needs; the baseline and stage-1 fits must stay importable
without it. Import it directly: `from chroma_gs.losses import DistillationLoss`.
"""

from .colmap import Dataset, Parser
from .model import (
    create_splats,
    load_checkpoint,
    promote_to_stage2,
    rasterize,
    save_checkpoint,
)
from .trainer import FitConfig, evaluate, fit

__all__ = [
    "Parser",
    "Dataset",
    "create_splats",
    "rasterize",
    "promote_to_stage2",
    "save_checkpoint",
    "load_checkpoint",
    "FitConfig",
    "fit",
    "evaluate",
]
