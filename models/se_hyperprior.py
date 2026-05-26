"""ScaleHyperprior variants with SE attention in transform networks.

Defines:
- SEScaleHyperprior: encoder-only, decoder-only, or encoder+decoder attention
- load_pretrained_with_se: transfer pretrained weights to an SE variant
- load_pretrained_baseline: load vanilla pretrained model for baseline variant
"""

import torch.nn as nn
from compressai.models import ScaleHyperprior
from compressai.layers import GDN
from compressai.models.utils import conv, deconv
from compressai.zoo import (
    bmshj2018_factorized,
    bmshj2018_hyperprior,
    cheng2020_anchor,
    mbt2018,
    mbt2018_mean,
)

from .se_block import SEBlock

QUALITY_TO_PARAMS = {
    1: (128, 192), 2: (128, 192), 3: (128, 192), 4: (128, 192),
    5: (192, 320), 6: (192, 320), 7: (192, 320), 8: (192, 320),
}

ZOO_PRETRAINED_MODELS = {
    "bmshj2018_factorized": bmshj2018_factorized,
    "bmshj2018_hyperprior": bmshj2018_hyperprior,
    "mbt2018_mean": mbt2018_mean,
    "mbt2018": mbt2018,
    "cheng2020_anchor": cheng2020_anchor,
}


class SEScaleHyperprior(ScaleHyperprior):
    """ScaleHyperprior with optional SE blocks in g_a and/or g_s.

    Baseline g_a:
        conv(3,N) -> GDN -> conv(N,N) -> GDN -> conv(N,N) -> GDN -> conv(N,M)

    Encoder attention g_a:
        conv(3,N) -> GDN -> conv(N,N) -> SE(N) -> GDN -> conv(N,N) -> GDN -> conv(N,M)

    Baseline g_s:
        deconv(M,N) -> IGDN -> deconv(N,N) -> IGDN -> deconv(N,N) -> IGDN -> deconv(N,3)

    Decoder attention g_s:
        deconv(M,N) -> IGDN -> deconv(N,N) -> SE(N) -> IGDN -> deconv(N,N) -> IGDN -> deconv(N,3)
    """

    VALID_ATTENTION = {"encoder", "decoder", "both"}

    def __init__(self, N, M, attention="both", reduction=16, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        if attention not in self.VALID_ATTENTION:
            raise ValueError(f"attention must be one of {sorted(self.VALID_ATTENTION)}")
        self.attention = attention

        if attention in {"encoder", "both"}:
            self.g_a = nn.Sequential(
                conv(3, N),
                GDN(N),
                conv(N, N),
                SEBlock(N, reduction=reduction),
                GDN(N),
                conv(N, N),
                GDN(N),
                conv(N, M),
            )

        if attention in {"decoder", "both"}:
            self.g_s = nn.Sequential(
                deconv(M, N),
                GDN(N, inverse=True),
                deconv(N, N),
                SEBlock(N, reduction=reduction),
                GDN(N, inverse=True),
                deconv(N, N),
                GDN(N, inverse=True),
                deconv(N, 3),
            )


def _remap_transform_key(key, transform_name):
    """Map baseline transform indices to an SE-augmented transform."""
    parts = key.split(".")
    old_idx = int(parts[1])
    # Baseline: [0]=conv/deconv [1]=GDN [2]=conv/deconv [3]=GDN [4]=conv/deconv [5]=GDN [6]=conv/deconv
    # SE:       [0]=conv/deconv [1]=GDN [2]=conv/deconv [3]=SE  [4]=GDN [5]=conv/deconv [6]=GDN [7]=conv/deconv
    baseline_to_se = {0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 6, 6: 7}
    if old_idx not in baseline_to_se:
        return None
    new_idx = baseline_to_se[old_idx]
    return f"{transform_name}.{new_idx}." + ".".join(parts[2:])


def load_pretrained_with_se(quality=3, attention="both", reduction=16):
    """Load pretrained ScaleHyperprior and transfer weights to SEScaleHyperprior.

    The SE block is randomly initialized; all other weights come from the
    pretrained CompressAI checkpoint.
    """
    N, M = QUALITY_TO_PARAMS[quality]

    pretrained = bmshj2018_hyperprior(quality=quality, pretrained=True)
    se_model = SEScaleHyperprior(N=N, M=M, attention=attention, reduction=reduction)

    se_state = se_model.state_dict()
    pretrained_state = pretrained.state_dict()

    new_state = {}
    for key, value in pretrained_state.items():
        if key.startswith("g_a.") and attention in {"encoder", "both"}:
            new_key = _remap_transform_key(key, "g_a")
            if new_key is not None:
                new_state[new_key] = value
        elif key.startswith("g_s.") and attention in {"decoder", "both"}:
            new_key = _remap_transform_key(key, "g_s")
            if new_key is not None:
                new_state[new_key] = value
        else:
            new_state[key] = value

    se_state.update(new_state)
    se_model.load_state_dict(se_state)
    return se_model


def load_pretrained_baseline(quality=3):
    """Load a pretrained vanilla ScaleHyperprior baseline."""
    return bmshj2018_hyperprior(quality=quality, pretrained=True)


def load_pretrained_zoo_model(model_name, quality=3, metric="mse"):
    """Load a pretrained CompressAI zoo image model by name."""
    if model_name not in ZOO_PRETRAINED_MODELS:
        raise ValueError(
            f"Unknown pretrained zoo model: {model_name}. "
            f"Available: {sorted(ZOO_PRETRAINED_MODELS)}"
        )
    return ZOO_PRETRAINED_MODELS[model_name](
        quality=quality,
        metric=metric,
        pretrained=True,
    )
