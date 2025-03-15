import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Metric

from src.data.humanml.scripts.motion_process import (qrot,
                                                     recover_root_rot_pos)


class MLDLosses(nn.Module):
    """
    MLD Loss
    """

    def __init__(self, learned_var=False):
        super().__init__()
        self.learned_var = learned_var

    def forward(self, out):
        # mask padding positions or given prior infos
        losses = {"loss": 0}
        mask, noise_pred = out["mask"], out["output"]
        B, _, D = out["noise"].shape
        if self.learned_var:
            noise_pred = noise_pred[..., :D]
            losses["vb"] = mask_before_summery(
                torch.mean,
                normal_kl(out["true_mean"], out["true_log_var"],
                          out["pred_mean"], out["pred_log_var"], out["timestemps"]),
                mask,
            )
            losses["loss"] += losses["vb"] * out["num_timesteps"] / 1000.0
        losses["noise"] = mask_before_summery(torch.mean, (noise_pred - out["noise"])**2, mask)
        losses["loss"] += losses["noise"]
        return losses


def mask_before_summery(func, x, mask):
    if len(x.shape) == len(mask.shape) + 1:
        return func(x[mask].sum() / (mask.sum() * x.shape[-1]))
    return func(x[mask].sum() / mask.sum())


def normal_kl(mean1, logvar1, mean2, logvar2, timestemps=None):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    kl_div = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                    + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))
    if timestemps is None:
        return kl_div
    kl_div[timestemps==0] = 0
    return kl_div