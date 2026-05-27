"""Fast end-to-end checks before launching long training runs."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from evaluate import evaluate_model, load_model
from plot import plot_rd_curves
from train import RateDistortionLoss, apply_freeze_mode, build_model, configure_optimizers, select_device
from utils.config import DEFAULT_CONFIG_PATH, get_config_value, load_config
from utils.datasets import CLICDataset, KodakDataset


class KodakCropDataset(Dataset):
    """Tiny Kodak wrapper used only for fast smoke evaluation."""

    def __init__(self, root, crop_size=256, limit=1):
        self.dataset = KodakDataset(root)
        self.crop_size = crop_size
        self.limit = min(limit, len(self.dataset))

    def __len__(self):
        return self.limit

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        crop = x[:, : self.crop_size, : self.crop_size]
        return crop, (self.crop_size, self.crop_size)


def load_parser_defaults(config):
    return {
        "train_dir": get_config_value(config, "paths", "clic_train_dir", default="datasets/clic_images/train"),
        "val_dir": get_config_value(config, "paths", "clic_val_dir", default="datasets/clic_images/validation"),
        "kodak_dir": get_config_value(config, "paths", "kodak_dir", default="datasets/kodak"),
    }


def assert_finite_metrics(metrics, label):
    for item in metrics:
        for key in ("estimated_bpp", "psnr", "ms_ssim", "mse"):
            value = item[key]
            if not torch.isfinite(torch.tensor(value)):
                raise RuntimeError(f"{label}: non-finite {key}: {value}")


def run_training_step(model, batch, criterion, device):
    optimizer, aux_optimizer = configure_optimizers(model, lr=1e-4, se_lr=1e-3)
    model.train()
    batch = batch.to(device)
    optimizer.zero_grad()
    if aux_optimizer:
        aux_optimizer.zero_grad()

    output = model(batch)
    output["x_hat"] = output["x_hat"].clamp(0, 1)
    losses = criterion(output, batch)
    losses["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    aux_value = 0.0
    if aux_optimizer:
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        aux_value = aux_loss.item()
    return losses["loss"].item(), aux_value


def write_dummy_summaries(results_dir):
    lmbdas = [0.0018, 0.0035, 0.0067, 0.013]
    base_bpps = [0.15, 0.25, 0.38, 0.55]
    base_psnr = [27.0, 28.5, 30.0, 31.5]
    base_msssim = [12.0, 13.5, 15.0, 16.5]
    offsets = {"A_ft": (0.00, 0.00), "B": (0.01, 0.10), "C": (0.02, 0.20)}

    results_dir.mkdir(parents=True, exist_ok=True)
    for variant, (bpp_offset, quality_offset) in offsets.items():
        for i, lmbda in enumerate(lmbdas):
            summary = {
                "variant": variant,
                "lmbda": lmbda,
                "quality": i + 1,
                "freeze_mode": "none",
                "avg_bpp": base_bpps[i] + bpp_offset,
                "avg_estimated_bpp": base_bpps[i] + bpp_offset,
                "avg_coded_bpp": base_bpps[i] + bpp_offset + 0.001,
                "avg_psnr": base_psnr[i] + quality_offset,
                "avg_ms_ssim": 0.9,
                "avg_ms_ssim_db": base_msssim[i] + quality_offset,
                "total_params": 1,
                "run_name": f"{variant}_{lmbda}",
            }
            path = results_dir / f"variant_{variant}_lmbda_{lmbda}_summary.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle)


def main():
    config = load_config(DEFAULT_CONFIG_PATH)
    defaults = load_parser_defaults(config)

    parser = argparse.ArgumentParser(description="Run fast project smoke tests")
    parser.add_argument("--train-dir", default=defaults["train_dir"])
    parser.add_argument("--val-dir", default=defaults["val_dir"])
    parser.add_argument("--kodak-dir", default=defaults["kodak_dir"])
    parser.add_argument("--quality", type=int, default=1)
    parser.add_argument("--lmbda", type=float, default=0.0018)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--eval-crop-size", type=int, default=256)
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = select_device()
    print(f"Device: {device}")

    train_dataset = CLICDataset(args.train_dir, patch_size=args.patch_size, training=True)
    val_dataset = CLICDataset(args.val_dir, patch_size=args.patch_size, training=False)
    kodak_dataset = KodakDataset(args.kodak_dir)
    print(f"CLIC train images: {len(train_dataset)}")
    print(f"CLIC validation images: {len(val_dataset)}")
    print(f"Kodak images: {len(kodak_dataset)}")

    train_loader = DataLoader(Subset(train_dataset, [0]), batch_size=1, num_workers=0)
    batch = next(iter(train_loader))
    criterion = RateDistortionLoss(lmbda=args.lmbda, distortion="mse")

    with tempfile.TemporaryDirectory(prefix="lc_smoke_") as tmp:
        tmp_dir = Path(tmp)
        eval_loader = DataLoader(
            KodakCropDataset(args.kodak_dir, crop_size=args.eval_crop_size, limit=1),
            batch_size=1,
            num_workers=0,
        )

        for variant in ("A", "A_ft", "B", "C"):
            print(f"\nChecking variant {variant}...")
            model = build_model(variant, args.quality)
            if variant == "B":
                apply_freeze_mode(model, variant, "frozen_hyperprior")
            model = model.to(device)
            model.update()

            with torch.no_grad():
                output = model(batch.to(device))
                output["x_hat"] = output["x_hat"].clamp(0, 1)
                losses = criterion(output, batch.to(device))
            print(f"  forward ok: x_hat={tuple(output['x_hat'].shape)} loss={losses['loss'].item():.4f}")

            if variant in {"A_ft", "B", "C"}:
                loss, aux_loss = run_training_step(model, batch, criterion, device)
                print(f"  one training step ok: loss={loss:.4f} aux={aux_loss:.4f}")

            ckpt_path = tmp_dir / f"variant_{variant}.pth.tar"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "variant": variant,
                    "quality": args.quality,
                    "lmbda": args.lmbda,
                    "freeze_mode": "none",
                    "run_name": f"smoke_{variant}",
                },
                ckpt_path,
            )

            checkpoint = None if variant == "A" else ckpt_path
            eval_model, _ = load_model(checkpoint, variant, quality=args.quality)
            eval_model = eval_model.to(device)
            metrics = evaluate_model(
                eval_model,
                eval_loader,
                device,
                compute_coded_bpp=True,
                recon_dir=tmp_dir / f"recon_{variant}",
                reconstruction_indices={0},
            )
            assert_finite_metrics(metrics, f"variant {variant}")
            print(
                f"  eval ok: est_bpp={metrics[0]['estimated_bpp']:.4f} "
                f"coded_bpp={metrics[0]['coded_bpp']:.4f} psnr={metrics[0]['psnr']:.2f}"
            )

        if not args.skip_plot:
            print("\nChecking plot generation...")
            results_dir = tmp_dir / "results"
            plots_dir = tmp_dir / "plots"
            write_dummy_summaries(results_dir)
            data = {}
            for path in results_dir.glob("*_summary.json"):
                with open(path, "r", encoding="utf-8") as handle:
                    summary = json.load(handle)
                data.setdefault(summary["variant"], []).append(summary)
            for summaries in data.values():
                summaries.sort(key=lambda item: item.get("avg_coded_bpp", item["avg_bpp"]))
            plot_rd_curves(data, plots_dir, metric="psnr")
            plot_rd_curves(data, plots_dir, metric="msssim")
            expected = [plots_dir / "rd_curve_psnr.png", plots_dir / "rd_curve_msssim.png"]
            missing = [path for path in expected if not path.exists()]
            if missing:
                raise RuntimeError(f"Plot smoke test did not create: {missing}")
            print("  plot generation ok")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
