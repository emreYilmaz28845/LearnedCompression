"""
Evaluate trained models on the Kodak dataset.

Computes per-image PSNR, MS-SSIM, and bpp, then aggregates results.
Supports BD-rate computation between variants.

Usage:
  python evaluate.py --checkpoint checkpoints/variant_C_lmbda_0.0067_mse/checkpoint_best.pth.tar \
                     --variant C --data-dir datasets/kodak --output results/
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader

from utils.datasets import KodakDataset
from models import (
    QUALITY_TO_PARAMS,
    SEScaleHyperprior,
    ZOO_PRETRAINED_MODELS,
    load_pretrained_baseline,
    load_pretrained_zoo_model,
)


SUPPORTED_VARIANTS = ["A", "B", "C", *ZOO_PRETRAINED_MODELS.keys()]


def infer_run_name(checkpoint_path, variant, quality):
    if checkpoint_path:
        return Path(checkpoint_path).parent.name
    return f"variant_{variant}_quality_{quality}"


def select_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.empty(1, device="cuda").add_(1)
        torch.cuda.synchronize()
        return torch.device("cuda")
    except Exception as exc:
        print(f"CUDA is visible but unusable in this environment; falling back to CPU ({exc})")
        return torch.device("cpu")


def load_model(checkpoint_path, variant, quality=3):
    """Load a trained model from checkpoint."""
    if variant == "A":
        model = load_pretrained_baseline(quality=quality)
    elif variant == "B":
        N, M = QUALITY_TO_PARAMS[quality]
        model = SEScaleHyperprior(N=N, M=M, attention="encoder")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
    elif variant == "C":
        N, M = QUALITY_TO_PARAMS[quality]
        model = SEScaleHyperprior(N=N, M=M, attention="both")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
    elif variant in ZOO_PRETRAINED_MODELS:
        model = load_pretrained_zoo_model(variant, quality=quality, metric="mse")
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return model


def evaluate_model(model, data_loader, device):
    """Evaluate model on a dataset. Returns list of per-image metrics."""
    model.eval()
    model.update()
    results = []

    with torch.no_grad():
        for i, (x, (orig_h, orig_w)) in enumerate(data_loader):
            x = x.to(device)
            h, w = orig_h.item(), orig_w.item()

            out = model(x)
            x_hat = out["x_hat"].clamp(0, 1)

            # Crop back to original size
            x_cropped = x[:, :, :h, :w]
            x_hat_cropped = x_hat[:, :, :h, :w]

            # BPP from likelihoods
            num_pixels = h * w
            bpp = sum(
                -torch.log2(lk).sum().item() / num_pixels
                for lk in out["likelihoods"].values()
            )

            # PSNR
            mse = torch.mean((x_cropped - x_hat_cropped) ** 2).item()
            psnr = -10 * math.log10(mse) if mse > 0 else 100.0

            # MS-SSIM
            msssim_val = ms_ssim(
                x_hat_cropped, x_cropped, data_range=1.0, size_average=True
            ).item()

            results.append({
                "image": i + 1,
                "bpp": bpp,
                "psnr": psnr,
                "ms_ssim": msssim_val,
                "ms_ssim_db": -10 * math.log10(1 - msssim_val) if msssim_val < 1 else 100.0,
                "mse": mse,
            })

            print(f"  Image {i+1:2d}: bpp={bpp:.4f}  PSNR={psnr:.2f}dB  MS-SSIM={msssim_val:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Kodak dataset")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (not needed for pretrained variants)")
    parser.add_argument("--variant", choices=SUPPORTED_VARIANTS, required=True)
    parser.add_argument("--quality", type=int, default=3)
    parser.add_argument("--lmbda", type=float, default=None,
                        help="Lambda value (for labeling results)")
    parser.add_argument("--data-dir", default="datasets/kodak")
    parser.add_argument("--output", default="experiments/results",
                        help="Output directory for results")
    parser.add_argument("--run-name", default=None,
                        help="Optional run name for output filenames")
    args = parser.parse_args()

    device = select_device()
    print(f"Device: {device}")

    # Load model
    print(f"Loading variant {args.variant}...")
    model = load_model(args.checkpoint, args.variant, args.quality)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Load dataset
    dataset = KodakDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Evaluating on {len(dataset)} images...")

    # Evaluate
    results = evaluate_model(model, loader, device)

    # Aggregate
    avg_bpp = np.mean([r["bpp"] for r in results])
    avg_psnr = np.mean([r["psnr"] for r in results])
    avg_msssim = np.mean([r["ms_ssim"] for r in results])
    avg_msssim_db = np.mean([r["ms_ssim_db"] for r in results])

    print(f"\n--- Averages ---")
    print(f"  BPP:      {avg_bpp:.4f}")
    print(f"  PSNR:     {avg_psnr:.2f} dB")
    print(f"  MS-SSIM:  {avg_msssim:.4f}")
    print(f"  MS-SSIM (dB): {avg_msssim_db:.2f}")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    lmbda_str = f"{args.lmbda}" if args.lmbda else "unknown"
    run_name = args.run_name or infer_run_name(args.checkpoint, args.variant, args.quality)
    result_file = output_dir / f"{run_name}_lmbda_{lmbda_str}.csv"

    with open(result_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "bpp", "psnr", "ms_ssim", "ms_ssim_db", "mse"])
        writer.writeheader()
        writer.writerows(results)

    # Save summary
    summary = {
        "variant": args.variant,
        "run_name": run_name,
        "lmbda": args.lmbda,
        "quality": args.quality,
        "avg_bpp": avg_bpp,
        "avg_psnr": avg_psnr,
        "avg_ms_ssim": avg_msssim,
        "avg_ms_ssim_db": avg_msssim_db,
        "total_params": total_params,
    }
    summary_file = output_dir / f"{run_name}_lmbda_{lmbda_str}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {result_file}")
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
