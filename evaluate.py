"""
Evaluate trained models on the Kodak dataset.

Computes per-image PSNR, MS-SSIM, estimated bpp, and optionally true coded bpp.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from models import (
    QUALITY_TO_PARAMS,
    SEScaleHyperprior,
    ZOO_PRETRAINED_MODELS,
    load_pretrained_baseline,
    load_pretrained_zoo_model,
)
from utils.config import DEFAULT_CONFIG_PATH, get_config_value, load_config
from utils.datasets import KodakDataset


SUPPORTED_VARIANTS = ["A", "A_ft", "B", "C", *ZOO_PRETRAINED_MODELS.keys()]


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


def load_parser_defaults(config):
    return {
        "quality": get_config_value(config, "qualities", default=[1, 2, 3, 4])[2],
        "data_dir": get_config_value(config, "paths", "kodak_dir", default="datasets/kodak"),
        "output": get_config_value(config, "paths", "results_dir", default="experiments/results"),
        "recon_dir": get_config_value(config, "paths", "recon_dir", default="experiments/reconstructions"),
        "compute_coded_bpp": get_config_value(config, "evaluation", "compute_coded_bpp", default=True),
        "save_reconstructions": get_config_value(config, "evaluation", "save_reconstructions", default=True),
        "reconstruction_limit": get_config_value(config, "evaluation", "reconstruction_limit", default=4),
        "reconstruction_indices": get_config_value(config, "evaluation", "reconstruction_indices", default=[0, 1, 2, 3]),
    }


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    config_args, remaining = config_parser.parse_known_args()
    config = load_config(config_args.config)
    defaults = load_parser_defaults(config)

    parser = argparse.ArgumentParser(description="Evaluate on Kodak dataset", parents=[config_parser])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--variant", choices=SUPPORTED_VARIANTS, required=True)
    parser.add_argument("--quality", type=int, default=defaults["quality"])
    parser.add_argument("--lmbda", type=float, default=None)
    parser.add_argument("--data-dir", default=defaults["data_dir"])
    parser.add_argument("--output", default=defaults["output"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--recon-dir", default=defaults["recon_dir"])
    parser.add_argument("--compute-coded-bpp", dest="compute_coded_bpp", action="store_true")
    parser.add_argument("--no-compute-coded-bpp", dest="compute_coded_bpp", action="store_false")
    parser.set_defaults(compute_coded_bpp=defaults["compute_coded_bpp"])
    parser.add_argument("--save-reconstructions", dest="save_reconstructions", action="store_true")
    parser.add_argument("--no-save-reconstructions", dest="save_reconstructions", action="store_false")
    parser.set_defaults(save_reconstructions=defaults["save_reconstructions"])
    parser.add_argument("--reconstruction-limit", type=int, default=defaults["reconstruction_limit"])
    parser.add_argument("--reconstruction-indices", type=int, nargs="*", default=defaults["reconstruction_indices"])

    args = parser.parse_args(remaining)
    args.config = config_args.config
    args._config = config
    return args


def load_model(checkpoint_path, variant, quality=3):
    """Load a model and any checkpoint metadata."""
    metadata = {"freeze_mode": "reference"}

    if variant == "A":
        model = load_pretrained_baseline(quality=quality)
        metadata["freeze_mode"] = "pretrained_reference"
    elif variant == "A_ft":
        if not checkpoint_path:
            raise ValueError("Variant A_ft requires --checkpoint")
        model = load_pretrained_baseline(quality=quality)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        metadata.update({
            "freeze_mode": ckpt.get("freeze_mode", "none"),
            "run_name": ckpt.get("run_name"),
        })
    elif variant == "B":
        if not checkpoint_path:
            raise ValueError("Variant B requires --checkpoint")
        n, m = QUALITY_TO_PARAMS[quality]
        model = SEScaleHyperprior(N=n, M=m, attention="encoder")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        metadata.update({
            "freeze_mode": ckpt.get("freeze_mode", "none"),
            "run_name": ckpt.get("run_name"),
        })
    elif variant == "C":
        if not checkpoint_path:
            raise ValueError("Variant C requires --checkpoint")
        n, m = QUALITY_TO_PARAMS[quality]
        model = SEScaleHyperprior(N=n, M=m, attention="both")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        metadata.update({
            "freeze_mode": ckpt.get("freeze_mode", "none"),
            "run_name": ckpt.get("run_name"),
        })
    elif variant in ZOO_PRETRAINED_MODELS:
        model = load_pretrained_zoo_model(variant, quality=quality, metric="mse")
        metadata["freeze_mode"] = "pretrained_reference"
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return model, metadata


def estimate_bpp_from_likelihoods(output, num_pixels):
    return sum(
        -torch.log2(likelihoods).sum().item() / num_pixels
        for likelihoods in output["likelihoods"].values()
    )


def compute_coded_reconstruction(model, x, num_pixels):
    encoded = model.compress(x)
    decoded = model.decompress(encoded["strings"], encoded["shape"])
    coded_bits = sum(len(stream) * 8 for group in encoded["strings"] for stream in group)
    return decoded["x_hat"].clamp(0, 1), coded_bits / num_pixels


def get_image_stem(dataset, index):
    if hasattr(dataset, "images"):
        return Path(dataset.images[index]).stem
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "images"):
        return Path(dataset.dataset.images[index]).stem
    return f"image_{index + 1:02d}"


def maybe_save_reconstruction(save_root, image_stem, x_cropped, x_hat_cropped):
    save_root.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(x_cropped.squeeze(0).cpu()).save(save_root / f"{image_stem}_input.png")
    TF.to_pil_image(x_hat_cropped.squeeze(0).cpu()).save(save_root / f"{image_stem}_recon.png")


def evaluate_model(
    model,
    data_loader,
    device,
    compute_coded_bpp=False,
    recon_dir=None,
    reconstruction_indices=None,
):
    """Evaluate model on a dataset. Returns list of per-image metrics."""
    model.eval()
    model.update()
    results = []
    reconstruction_indices = set(reconstruction_indices or [])

    with torch.no_grad():
        for i, (x, (orig_h, orig_w)) in enumerate(data_loader):
            x = x.to(device)
            h, w = orig_h.item(), orig_w.item()
            num_pixels = h * w

            out = model(x)
            estimated_bpp = estimate_bpp_from_likelihoods(out, num_pixels)

            x_hat = out["x_hat"].clamp(0, 1)
            coded_bpp = None
            if compute_coded_bpp:
                x_hat, coded_bpp = compute_coded_reconstruction(model, x, num_pixels)

            x_cropped = x[:, :, :h, :w]
            x_hat_cropped = x_hat[:, :, :h, :w]

            mse = torch.mean((x_cropped - x_hat_cropped) ** 2).item()
            psnr = -10 * math.log10(mse) if mse > 0 else 100.0
            msssim_val = ms_ssim(
                x_hat_cropped,
                x_cropped,
                data_range=1.0,
                size_average=True,
            ).item()

            image_stem = get_image_stem(data_loader.dataset, i)
            results.append({
                "image": i + 1,
                "image_name": image_stem,
                "estimated_bpp": estimated_bpp,
                "coded_bpp": coded_bpp,
                "bpp": estimated_bpp,
                "psnr": psnr,
                "ms_ssim": msssim_val,
                "ms_ssim_db": -10 * math.log10(1 - msssim_val) if msssim_val < 1 else 100.0,
                "mse": mse,
            })

            if recon_dir and i in reconstruction_indices:
                maybe_save_reconstruction(recon_dir, image_stem, x_cropped, x_hat_cropped)

            coded_fragment = f" coded_bpp={coded_bpp:.4f}" if coded_bpp is not None else ""
            print(
                f"  Image {i+1:2d}: est_bpp={estimated_bpp:.4f}{coded_fragment} "
                f"PSNR={psnr:.2f}dB MS-SSIM={msssim_val:.4f}"
            )

    return results


def main():
    args = parse_args()
    device = select_device()
    print(f"Device: {device}")

    print(f"Loading variant {args.variant}...")
    model, metadata = load_model(args.checkpoint, args.variant, args.quality)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    dataset = KodakDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Evaluating on {len(dataset)} images...")

    run_name = args.run_name or metadata.get("run_name") or infer_run_name(
        args.checkpoint,
        args.variant,
        args.quality,
    )
    recon_run_dir = None
    if args.save_reconstructions:
        recon_run_dir = Path(args.recon_dir) / run_name

    results = evaluate_model(
        model,
        loader,
        device,
        compute_coded_bpp=args.compute_coded_bpp,
        recon_dir=recon_run_dir,
        reconstruction_indices=args.reconstruction_indices[: args.reconstruction_limit],
    )

    avg_estimated_bpp = np.mean([r["estimated_bpp"] for r in results])
    coded_values = [r["coded_bpp"] for r in results if r["coded_bpp"] is not None]
    avg_coded_bpp = np.mean(coded_values) if coded_values else None
    avg_psnr = np.mean([r["psnr"] for r in results])
    avg_msssim = np.mean([r["ms_ssim"] for r in results])
    avg_msssim_db = np.mean([r["ms_ssim_db"] for r in results])

    print("\n--- Averages ---")
    print(f"  Estimated BPP: {avg_estimated_bpp:.4f}")
    if avg_coded_bpp is not None:
        print(f"  Coded BPP:     {avg_coded_bpp:.4f}")
    print(f"  PSNR:          {avg_psnr:.2f} dB")
    print(f"  MS-SSIM:       {avg_msssim:.4f}")
    print(f"  MS-SSIM (dB):  {avg_msssim_db:.2f}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    lmbda_str = f"{args.lmbda}" if args.lmbda is not None else "unknown"
    result_file = output_dir / f"{run_name}_lmbda_{lmbda_str}.csv"

    with open(result_file, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "image_name",
                "estimated_bpp",
                "coded_bpp",
                "bpp",
                "psnr",
                "ms_ssim",
                "ms_ssim_db",
                "mse",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "variant": args.variant,
        "run_name": run_name,
        "lmbda": args.lmbda,
        "quality": args.quality,
        "freeze_mode": metadata.get("freeze_mode", "unknown"),
        "avg_bpp": avg_estimated_bpp,
        "avg_estimated_bpp": avg_estimated_bpp,
        "avg_coded_bpp": avg_coded_bpp,
        "avg_psnr": avg_psnr,
        "avg_ms_ssim": avg_msssim,
        "avg_ms_ssim_db": avg_msssim_db,
        "total_params": total_params,
        "compute_coded_bpp": args.compute_coded_bpp,
        "saved_reconstructions": bool(args.save_reconstructions),
        "recon_dir": str(recon_run_dir) if recon_run_dir else None,
    }
    summary_file = output_dir / f"{run_name}_lmbda_{lmbda_str}_summary.json"
    with open(summary_file, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nResults saved to {result_file}")
    print(f"Summary saved to {summary_file}")
    if recon_run_dir:
        print(f"Reconstructions saved to {recon_run_dir}")


if __name__ == "__main__":
    main()
