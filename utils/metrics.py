"""Evaluation metrics: PSNR, MS-SSIM, and BD-rate."""

import math

import numpy as np
import torch
from pytorch_msssim import ms_ssim


def compute_psnr(x, x_hat):
    """Compute PSNR between two image tensors (values in [0, 1])."""
    mse = torch.mean((x - x_hat) ** 2).item()
    if mse == 0:
        return 100.0
    return -10 * math.log10(mse)


def compute_ms_ssim(x, x_hat, data_range=1.0):
    """Compute MS-SSIM between two image tensors."""
    return ms_ssim(x_hat, x, data_range=data_range, size_average=True).item()


def compute_bd_rate(rate_a, psnr_a, rate_b, psnr_b):
    """Compute BD-rate (Bjontegaard Delta Rate) between two RD curves.

    Negative BD-rate means B is better (lower rate at same quality).

    Args:
        rate_a, psnr_a: reference curve (4 operating points)
        rate_b, psnr_b: test curve (4 operating points)

    Returns:
        BD-rate in percent.
    """
    idx_a = np.argsort(psnr_a)
    idx_b = np.argsort(psnr_b)
    psnr_a = np.array(psnr_a)[idx_a]
    rate_a = np.log(np.array(rate_a)[idx_a])
    psnr_b = np.array(psnr_b)[idx_b]
    rate_b = np.log(np.array(rate_b)[idx_b])

    if len(psnr_a) < 4 or len(psnr_b) < 4:
        print("Warning: BD-rate requires 4 operating points, returning NaN")
        return float("nan")

    poly_a = np.polyfit(psnr_a, rate_a, 3)
    poly_b = np.polyfit(psnr_b, rate_b, 3)

    min_psnr = max(psnr_a.min(), psnr_b.min())
    max_psnr = min(psnr_a.max(), psnr_b.max())

    if min_psnr >= max_psnr:
        return float("nan")

    p_a = np.polyint(poly_a)
    p_b = np.polyint(poly_b)

    int_a = np.polyval(p_a, max_psnr) - np.polyval(p_a, min_psnr)
    int_b = np.polyval(p_b, max_psnr) - np.polyval(p_b, min_psnr)

    avg_diff = (int_b - int_a) / (max_psnr - min_psnr)
    bd_rate = (np.exp(avg_diff) - 1) * 100

    return bd_rate
