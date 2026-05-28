"""
Evaluate trained models on the Kodak dataset.

Computes per-image PSNR, MS-SSIM, estimated bpp, and optionally true coded bpp.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from models import (
    QUALITY_TO_PARAMS,
    SEScaleHyperprior,
    ZOO_PRETRAINED_MODELS,
    load_pretrained_baseline,
    load_pretrained_zoo_model,
)
from utils import compute_ms_ssim, compute_psnr
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


def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_forward(model, x, device):
    synchronize_device(device)
    start = time.perf_counter()
    out = model(x)
    synchronize_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return out, elapsed_ms


def timed_coded_reconstruction(model, x, num_pixels):
    encode_start = time.perf_counter()
    encoded = model.compress(x)
    encode_ms = (time.perf_counter() - encode_start) * 1000.0

    decode_start = time.perf_counter()
    decoded = model.decompress(encoded["strings"], encoded["shape"])
    decode_ms = (time.perf_counter() - decode_start) * 1000.0

    coded_bits = sum(len(stream) * 8 for group in encoded["strings"] for stream in group)
    return decoded["x_hat"].clamp(0, 1), coded_bits / num_pixels, encode_ms, decode_ms


def build_coding_model(model):
    """Create a CPU copy for reliable entropy coding during evaluation."""
    coding_model = copy.deepcopy(model).cpu()
    coding_model.eval()
    coding_model.update()
    return coding_model


def assert_finite_tensor(name, tensor):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")


def compute_metrics(x_cropped, x_hat_cropped):
    """Compute metrics on CPU for numerical stability."""
    x_metrics = x_cropped.detach().cpu()
    x_hat_metrics = x_hat_cropped.detach().cpu()
    assert_finite_tensor("reference image", x_metrics)
    assert_finite_tensor("reconstruction", x_hat_metrics)

    mse = torch.mean((x_metrics - x_hat_metrics) ** 2).item()
    psnr = compute_psnr(x_metrics, x_hat_metrics)
    msssim_val = compute_ms_ssim(x_metrics, x_hat_metrics, data_range=1.0)

    if not math.isfinite(mse) or not math.isfinite(psnr) or not math.isfinite(msssim_val):
        raise ValueError(
            f"Non-finite metric detected (mse={mse}, psnr={psnr}, ms_ssim={msssim_val})"
        )

    if msssim_val >= 1.0:
        msssim_db = 100.0
    else:
        clipped = min(max(msssim_val, 0.0), 1.0 - 1e-12)
        msssim_db = -10 * math.log10(1 - clipped)

    return mse, psnr, msssim_val, msssim_db


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
    coding_model = build_coding_model(model) if compute_coded_bpp else None

    with torch.no_grad():
        for i, (x, (orig_h, orig_w)) in enumerate(data_loader):
            x = x.to(device)
            h, w = orig_h.item(), orig_w.item()
            num_pixels = h * w

            out, forward_ms = timed_forward(model, x, device)
            estimated_bpp = estimate_bpp_from_likelihoods(out, num_pixels)

            x_hat = out["x_hat"].clamp(0, 1)
            coded_bpp = None
            encode_ms = None
            decode_ms = None
            if compute_coded_bpp:
                x_hat, coded_bpp, encode_ms, decode_ms = timed_coded_reconstruction(
                    coding_model,
                    x.cpu(),
                    num_pixels,
                )

            x_cropped = x[:, :, :h, :w]
            x_hat_cropped = x_hat[:, :, :h, :w]

            mse, psnr, msssim_val, msssim_db = compute_metrics(x_cropped, x_hat_cropped)

            image_stem = get_image_stem(data_loader.dataset, i)
            results.append({
                "image": i + 1,
                "image_name": image_stem,
                "estimated_bpp": estimated_bpp,
                "coded_bpp": coded_bpp,
                "bpp": estimated_bpp,
                "forward_ms": forward_ms,
                "encode_ms": encode_ms,
                "decode_ms": decode_ms,
                "codec_total_ms": (encode_ms + decode_ms) if encode_ms is not None and decode_ms is not None else None,
                "num_pixels": num_pixels,
                "psnr": psnr,
                "ms_ssim": msssim_val,
                "ms_ssim_db": msssim_db,
                "mse": mse,
            })

            if recon_dir and i in reconstruction_indices:
                maybe_save_reconstruction(recon_dir, image_stem, x_cropped, x_hat_cropped)

            coded_fragment = f" coded_bpp={coded_bpp:.4f}" if coded_bpp is not None else ""
            timing_fragment = (
                f" enc={encode_ms:.1f}ms dec={decode_ms:.1f}ms"
                if encode_ms is not None and decode_ms is not None
                else ""
            )
            print(
                f"  Image {i+1:2d}: est_bpp={estimated_bpp:.4f}{coded_fragment} "
                f"PSNR={psnr:.2f}dB MS-SSIM={msssim_val:.4f}"
                f" fwd={forward_ms:.1f}ms{timing_fragment}"
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
    avg_forward_ms = np.mean([r["forward_ms"] for r in results])
    encode_values = [r["encode_ms"] for r in results if r["encode_ms"] is not None]
    decode_values = [r["decode_ms"] for r in results if r["decode_ms"] is not None]
    total_codec_values = [r["codec_total_ms"] for r in results if r["codec_total_ms"] is not None]
    avg_encode_ms = np.mean(encode_values) if encode_values else None
    avg_decode_ms = np.mean(decode_values) if decode_values else None
    avg_codec_total_ms = np.mean(total_codec_values) if total_codec_values else None
    avg_num_pixels = np.mean([r["num_pixels"] for r in results])
    avg_megapixels = avg_num_pixels / 1_000_000.0
    avg_forward_mpix_per_s = avg_megapixels / (avg_forward_ms / 1000.0) if avg_forward_ms > 0 else None
    avg_encode_mpix_per_s = avg_megapixels / (avg_encode_ms / 1000.0) if avg_encode_ms else None
    avg_decode_mpix_per_s = avg_megapixels / (avg_decode_ms / 1000.0) if avg_decode_ms else None
    avg_codec_total_mpix_per_s = avg_megapixels / (avg_codec_total_ms / 1000.0) if avg_codec_total_ms else None

    print("\n--- Averages ---")
    print(f"  Estimated BPP: {avg_estimated_bpp:.4f}")
    if avg_coded_bpp is not None:
        print(f"  Coded BPP:     {avg_coded_bpp:.4f}")
    print(f"  PSNR:          {avg_psnr:.2f} dB")
    print(f"  MS-SSIM:       {avg_msssim:.4f}")
    print(f"  MS-SSIM (dB):  {avg_msssim_db:.2f}")
    print(f"  Forward time:  {avg_forward_ms:.2f} ms/image")
    if avg_encode_ms is not None and avg_decode_ms is not None:
        print(f"  Encode time:   {avg_encode_ms:.2f} ms/image")
        print(f"  Decode time:   {avg_decode_ms:.2f} ms/image")
        print(f"  Codec total:   {avg_codec_total_ms:.2f} ms/image")

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
                "forward_ms",
                "encode_ms",
                "decode_ms",
                "codec_total_ms",
                "num_pixels",
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
        "avg_num_pixels": avg_num_pixels,
        "avg_forward_ms": avg_forward_ms,
        "avg_encode_ms": avg_encode_ms,
        "avg_decode_ms": avg_decode_ms,
        "avg_codec_total_ms": avg_codec_total_ms,
        "avg_forward_mpix_per_s": avg_forward_mpix_per_s,
        "avg_encode_mpix_per_s": avg_encode_mpix_per_s,
        "avg_decode_mpix_per_s": avg_decode_mpix_per_s,
        "avg_codec_total_mpix_per_s": avg_codec_total_mpix_per_s,
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
