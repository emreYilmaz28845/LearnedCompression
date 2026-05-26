"""
Training script for learned image compression variants.

Variants:
  A - Pretrained baseline (no training, just save for evaluation)
  B - Fine-tuned with SE block in the encoder
  C - Fine-tuned with SE blocks in both encoder and decoder

Usage:
  python train.py --variant C --lmbda 0.0067 --data-dir datasets/clic_images --epochs 100
"""

import argparse
import math
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ms_ssim

from utils.datasets import CLICDataset
from models import load_pretrained_with_se, load_pretrained_baseline


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


def get_gpu_stats(device):
    """Return GPU utilization and memory stats for logging."""
    if device.type != "cuda":
        return {}

    stats = {}
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    total_mb = props.total_memory / (1024**2)
    allocated_mb = torch.cuda.memory_allocated(index) / (1024**2)
    reserved_mb = torch.cuda.memory_reserved(index) / (1024**2)
    max_allocated_mb = torch.cuda.max_memory_allocated(index) / (1024**2)

    stats.update({
        "gpu/memory_allocated_mb": allocated_mb,
        "gpu/memory_reserved_mb": reserved_mb,
        "gpu/memory_max_allocated_mb": max_allocated_mb,
        "gpu/memory_total_mb": total_mb,
        "gpu/memory_allocated_pct": 100.0 * allocated_mb / total_mb,
        "gpu/memory_reserved_pct": 100.0 * reserved_mb / total_mb,
    })

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        gpu_util, memory_util, memory_used, memory_total = [
            float(part.strip()) for part in result.stdout.strip().split(",")
        ]
        stats.update({
            "gpu/utilization_pct": gpu_util,
            "gpu/memory_utilization_pct": memory_util,
            "gpu/nvidia_smi_memory_used_mb": memory_used,
            "gpu/nvidia_smi_memory_total_mb": memory_total,
            "gpu/nvidia_smi_memory_used_pct": 100.0 * memory_used / memory_total,
        })
    except (subprocess.SubprocessError, FileNotFoundError, ValueError, ZeroDivisionError):
        pass

    return stats


class GPUStatsSampler:
    """Background nvidia-smi sampler for epoch-level GPU utilization."""

    def __init__(self, device, interval=1.0):
        self.device = device
        self.interval = interval
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = None

    def __enter__(self):
        if self.device.type != "cuda":
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1.0)

    def _run(self):
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        while not self._stop_event.is_set():
            sample = self._sample_once(index)
            if sample is not None:
                self.samples.append(sample)
            self._stop_event.wait(self.interval)

    @staticmethod
    def _sample_once(index):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={index}",
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            gpu_util, memory_util, memory_used, memory_total = [
                float(part.strip()) for part in result.stdout.strip().split(",")
            ]
            return {
                "gpu_util": gpu_util,
                "memory_util": memory_util,
                "memory_used": memory_used,
                "memory_total": memory_total,
            }
        except (subprocess.SubprocessError, FileNotFoundError, ValueError):
            return None

    def summary(self):
        if not self.samples:
            return {}
        gpu_utils = [sample["gpu_util"] for sample in self.samples]
        mem_utils = [sample["memory_util"] for sample in self.samples]
        mem_used = [sample["memory_used"] for sample in self.samples]
        mem_total = self.samples[-1]["memory_total"]
        return {
            "gpu/epoch_avg_utilization_pct": sum(gpu_utils) / len(gpu_utils),
            "gpu/epoch_max_utilization_pct": max(gpu_utils),
            "gpu/epoch_avg_memory_utilization_pct": sum(mem_utils) / len(mem_utils),
            "gpu/epoch_max_memory_used_mb": max(mem_used),
            "gpu/epoch_memory_total_mb": mem_total,
            "gpu/epoch_samples": len(self.samples),
        }


def log_scalar_dict(writer, values, step):
    for key, value in values.items():
        writer.add_scalar(key, value, step)


def build_run_name(args):
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.name.startswith("checkpoint_"):
            return checkpoint_path.parent.name
    base = (
        f"variant_{args.variant}_lmbda_{args.lmbda}_"
        f"quality_{args.quality}_{args.distortion}"
    )
    if args.no_timestamp:
        return base
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{timestamp}"


def checkpoint_payload(
    epoch,
    model,
    optimizer,
    aux_optimizer,
    scheduler,
    best_loss,
    patience_counter,
    args,
    run_name,
):
    return {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "aux_optimizer": aux_optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_loss": best_loss,
        "patience_counter": patience_counter,
        "variant": args.variant,
        "lmbda": args.lmbda,
        "quality": args.quality,
        "distortion": args.distortion,
        "run_name": run_name,
    }


def load_training_checkpoint(path, model, optimizer, aux_optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_loss = float(checkpoint.get("best_loss", float("inf")))
    patience_counter = int(checkpoint.get("patience_counter", 0))
    return start_epoch, best_loss, patience_counter


class RateDistortionLoss(nn.Module):
    """Rate-distortion loss: L = lambda * D(x, x_hat) + R(y_hat) + R(z_hat)"""

    def __init__(self, lmbda=0.0067, distortion="mse"):
        super().__init__()
        self.lmbda = lmbda
        self.distortion = distortion

    def forward(self, output, target):
        N, _, H, W = target.size()
        num_pixels = N * H * W

        # Rate: bits per pixel from likelihoods
        bpp_loss = sum(
            -torch.log2(likelihoods).sum() / num_pixels
            for likelihoods in output["likelihoods"].values()
        )

        # Distortion
        if self.distortion == "mse":
            distortion = 255.0**2 * nn.functional.mse_loss(output["x_hat"], target)
        else:  # ms-ssim
            distortion = 1.0 - ms_ssim(
                output["x_hat"].clamp(0, 1),
                target,
                data_range=1.0,
                size_average=True,
            )

        loss = self.lmbda * distortion + bpp_loss
        return {
            "loss": loss,
            "bpp": bpp_loss,
            "distortion": distortion,
            "mse": nn.functional.mse_loss(output["x_hat"], target),
        }


def build_model(variant, quality):
    if variant == "A":
        return load_pretrained_baseline(quality=quality)
    if variant == "B":
        return load_pretrained_with_se(quality=quality, attention="encoder")
    if variant == "C":
        return load_pretrained_with_se(quality=quality, attention="both")
    raise ValueError(f"Unknown variant: {variant}")


def configure_optimizers(model, lr, se_lr):
    """Separate parameters into main, SE block, and auxiliary (entropy model) groups.

    The SE blocks get a higher learning rate since they are newly initialized,
    while the pretrained backbone uses the base learning rate.
    """
    params_dict = dict(model.named_parameters())

    # Match SE block parameters by checking for SEBlock's submodules
    se_parameters = {
        n for n, p in model.named_parameters()
        if ".excitation." in n
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n for n, p in model.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }
    backbone_parameters = {
        n for n, p in model.named_parameters()
        if n not in se_parameters and n not in aux_parameters and p.requires_grad
    }

    param_groups = [
        {"params": [params_dict[n] for n in sorted(backbone_parameters)], "lr": lr},
    ]
    if se_parameters:
        param_groups.append(
            {"params": [params_dict[n] for n in sorted(se_parameters)], "lr": se_lr}
        )

    optimizer = optim.Adam(param_groups)
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=1e-3,
    )
    return optimizer, aux_optimizer


def train_one_epoch(model, criterion, train_loader, optimizer, aux_optimizer, device):
    model.train()
    total_loss = 0.0
    total_bpp = 0.0
    total_mse = 0.0
    total_distortion = 0.0
    total_aux_loss = 0.0
    total_grad_norm = 0.0
    count = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for i, x in enumerate(train_loader):
        x = x.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out = model(x)
        out["x_hat"] = out["x_hat"].clamp(0, 1)
        losses = criterion(out, x)

        losses["loss"].backward()
        # Gradient clipping for stability
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Auxiliary loss for entropy bottleneck
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        total_loss += losses["loss"].item()
        total_bpp += losses["bpp"].item()
        total_mse += losses["mse"].item()
        total_distortion += losses["distortion"].item()
        total_aux_loss += aux_loss.item()
        total_grad_norm += float(grad_norm)
        count += 1

        if (i + 1) % 100 == 0:
            psnr = -10 * math.log10(total_mse / count) if total_mse > 0 else 0
            print(
                f"  Step {i+1}: loss={total_loss/count:.4f} "
                f"bpp={total_bpp/count:.4f} PSNR={psnr:.2f}dB "
                f"aux={total_aux_loss/count:.4f}"
            )

    return {
        "loss": total_loss / count,
        "bpp": total_bpp / count,
        "distortion": total_distortion / count,
        "aux_loss": total_aux_loss / count,
        "grad_norm": total_grad_norm / count,
        "psnr": -10 * math.log10(total_mse / count) if total_mse > 0 else 0,
    }


def validate(model, criterion, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_bpp = 0.0
    total_mse = 0.0
    total_distortion = 0.0
    count = 0

    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            out = model(x)
            out["x_hat"] = out["x_hat"].clamp(0, 1)
            losses = criterion(out, x)

            total_loss += losses["loss"].item()
            total_bpp += losses["bpp"].item()
            total_mse += losses["mse"].item()
            total_distortion += losses["distortion"].item()
            count += 1

    return {
        "loss": total_loss / count,
        "bpp": total_bpp / count,
        "distortion": total_distortion / count,
        "psnr": -10 * math.log10(total_mse / count) if total_mse > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Train learned image compression")
    parser.add_argument("--variant", choices=["A", "B", "C"], required=True,
                        help="A=pretrained baseline, B=encoder SE, C=encoder+decoder SE")
    parser.add_argument("--lmbda", type=float, default=0.0067,
                        help="Rate-distortion trade-off (default: 0.0067)")
    parser.add_argument("--quality", type=int, default=3,
                        help="CompressAI pretrained quality level (default: 3)")
    parser.add_argument("--distortion", choices=["mse", "msssim"], default="mse",
                        help="Distortion metric (default: mse)")
    parser.add_argument("--data-dir", default="datasets/clic_images",
                        help="Training data directory")
    parser.add_argument("--val-dir", default=None,
                        help="Validation data directory (default: use 10%% of train)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--se-lr", type=float, default=1e-3,
                        help="Learning rate for newly initialized SE blocks")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Optional cap for quick benchmark/debug runs")
    parser.add_argument("--max-val-samples", type=int, default=None,
                        help="Optional validation cap for quick benchmark/debug runs")
    parser.add_argument("--save-dir", default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--timestamp", default=None,
                        help="Optional timestamp string for run directory naming")
    parser.add_argument("--no-timestamp", action="store_true",
                        help="Use stable checkpoint directory names without timestamps")
    parser.add_argument("--resume", default=None,
                        help="Resume training from checkpoint_last/checkpoint_epoch file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = select_device()
    print(f"Device: {device}")

    run_name = build_run_name(args)
    save_root = Path(args.save_dir)
    save_dir = save_root / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    latest_pointer = save_root / (
        f"latest_variant_{args.variant}_lmbda_{args.lmbda}_"
        f"quality_{args.quality}_{args.distortion}.txt"
    )
    latest_pointer.write_text(str(save_dir / "checkpoint_best.pth.tar") + "\n")
    print(f"Run directory: {save_dir}")
    print(f"Latest checkpoint pointer: {latest_pointer}")

    # Load model
    if args.variant == "A":
        print("Variant A: saving pretrained baseline (no training)")
        model = load_pretrained_baseline(quality=args.quality)
        model.to(device)
        model.update()
        torch.save({
            "state_dict": model.state_dict(),
            "variant": "A",
            "lmbda": args.lmbda,
            "quality": args.quality,
            "run_name": run_name,
        }, save_dir / "checkpoint_best.pth.tar")
        print(f"Saved to {save_dir / 'checkpoint_best.pth.tar'}")
        return

    elif args.variant == "B":
        print("Variant B: fine-tuning with encoder SE attention")
        model = build_model(args.variant, args.quality)
    elif args.variant == "C":
        print("Variant C: fine-tuning with encoder+decoder SE attention")
        model = build_model(args.variant, args.quality)
    else:
        raise ValueError(f"Unknown variant: {args.variant}")

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Dataset
    full_dataset = CLICDataset(args.data_dir, patch_size=args.patch_size, training=True)

    if args.val_dir:
        train_dataset = full_dataset
        val_dataset = CLICDataset(args.val_dir, patch_size=args.patch_size, training=False)
    else:
        # Split train/val (90/10). Use the same image pool, but deterministic indices.
        val_size = max(1, len(full_dataset) // 10)
        train_size = len(full_dataset) - val_size
        train_indices, val_indices = torch.utils.data.random_split(
            range(len(full_dataset)), [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        train_dataset = torch.utils.data.Subset(
            CLICDataset(args.data_dir, patch_size=args.patch_size, training=True),
            train_indices.indices,
        )
        val_dataset = torch.utils.data.Subset(
            CLICDataset(args.data_dir, patch_size=args.patch_size, training=False),
            val_indices.indices,
        )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    if args.max_train_samples is not None:
        train_limit = min(args.max_train_samples, len(train_dataset))
        train_dataset = torch.utils.data.Subset(train_dataset, range(train_limit))
        print(f"Using {len(train_dataset)} training samples for this run")

    if args.max_val_samples is not None:
        val_limit = min(args.max_val_samples, len(val_dataset))
        val_dataset = torch.utils.data.Subset(val_dataset, range(val_limit))
        print(f"Using {len(val_dataset)} validation samples for this run")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Loss, optimizers, scheduler
    criterion = RateDistortionLoss(lmbda=args.lmbda, distortion=args.distortion)
    optimizer, aux_optimizer = configure_optimizers(model, args.lr, args.se_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=str(save_dir / "logs"))
    writer.add_text("run/variant", args.variant)
    writer.add_scalar("run/total_params", total_params, 0)
    writer.add_scalar("run/trainable_params", trainable_params, 0)
    writer.add_scalar("run/lambda", args.lmbda, 0)
    writer.add_scalar("run/quality", args.quality, 0)
    writer.add_scalar("run/batch_size", args.batch_size, 0)
    writer.add_scalar("run/patch_size", args.patch_size, 0)

    best_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 20
    start_epoch = 1

    if args.resume:
        start_epoch, best_loss, patience_counter = load_training_checkpoint(
            args.resume, model, optimizer, aux_optimizer, scheduler, device
        )
        print(f"Resumed from {args.resume}")
        print(f"Starting at epoch {start_epoch}; best val loss so far: {best_loss:.4f}")

    if start_epoch > args.epochs:
        print(
            f"Checkpoint already reached epoch {start_epoch - 1}; "
            f"requested epochs={args.epochs}. Nothing to train."
        )
        writer.close()
        return

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        with GPUStatsSampler(device) as gpu_sampler:
            train_metrics = train_one_epoch(
                model, criterion, train_loader, optimizer, aux_optimizer, device
            )
            val_metrics = validate(model, criterion, val_loader, device)

        scheduler.step(val_metrics["loss"])
        epoch_time = time.perf_counter() - epoch_start
        gpu_stats = get_gpu_stats(device)
        gpu_stats.update(gpu_sampler.summary())

        # Log to TensorBoard
        writer.add_scalar("train/loss", train_metrics["loss"], epoch)
        writer.add_scalar("train/bpp", train_metrics["bpp"], epoch)
        writer.add_scalar("train/psnr", train_metrics["psnr"], epoch)
        writer.add_scalar("train/distortion", train_metrics["distortion"], epoch)
        writer.add_scalar("train/aux_loss", train_metrics["aux_loss"], epoch)
        writer.add_scalar("train/grad_norm", train_metrics["grad_norm"], epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/bpp", val_metrics["bpp"], epoch)
        writer.add_scalar("val/psnr", val_metrics["psnr"], epoch)
        writer.add_scalar("val/distortion", val_metrics["distortion"], epoch)
        writer.add_scalar("time/epoch_seconds", epoch_time, epoch)
        writer.add_scalar("time/seconds_per_train_step", epoch_time / max(len(train_loader), 1), epoch)
        writer.add_scalar("lr/backbone", optimizer.param_groups[0]["lr"], epoch)
        if len(optimizer.param_groups) > 1:
            writer.add_scalar("lr/se", optimizer.param_groups[1]["lr"], epoch)
        writer.add_scalar("lr/aux", aux_optimizer.param_groups[0]["lr"], epoch)
        log_scalar_dict(writer, gpu_stats, epoch)

        print(
            f"  Train: loss={train_metrics['loss']:.4f} "
            f"bpp={train_metrics['bpp']:.4f} PSNR={train_metrics['psnr']:.2f}dB "
            f"aux={train_metrics['aux_loss']:.4f} grad={train_metrics['grad_norm']:.2f}"
        )
        print(
            f"  Val:   loss={val_metrics['loss']:.4f} "
            f"bpp={val_metrics['bpp']:.4f} PSNR={val_metrics['psnr']:.2f}dB"
        )
        print(
            f"  Epoch duration: {epoch_time:.1f}s "
            f"({epoch_time / max(len(train_loader), 1):.3f}s/step)"
        )
        if gpu_stats:
            avg_util = gpu_stats.get("gpu/epoch_avg_utilization_pct")
            max_util = gpu_stats.get("gpu/epoch_max_utilization_pct")
            if avg_util is None:
                avg_util = gpu_stats.get("gpu/utilization_pct", float("nan"))
                max_util = avg_util
            print(
                "  GPU:   "
                f"avg_util={avg_util:.0f}% max_util={max_util:.0f}% "
                f"mem={gpu_stats['gpu/memory_allocated_mb']:.0f}MB/"
                f"{gpu_stats['gpu/memory_total_mb']:.0f}MB "
                f"peak={gpu_stats['gpu/memory_max_allocated_mb']:.0f}MB"
            )

        # Save best model
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(
                checkpoint_payload(
                    epoch,
                    model,
                    optimizer,
                    aux_optimizer,
                    scheduler,
                    best_loss,
                    patience_counter,
                    args,
                    run_name,
                ),
                save_dir / "checkpoint_best.pth.tar",
            )
            print(f"  Saved best model (loss={best_loss:.4f})")
        else:
            patience_counter += 1

        torch.save(
            checkpoint_payload(
                epoch,
                model,
                optimizer,
                aux_optimizer,
                scheduler,
                best_loss,
                patience_counter,
                args,
                run_name,
            ),
            save_dir / "checkpoint_last.pth.tar",
        )

        # Save periodic checkpoint
        if epoch % 10 == 0:
            torch.save(
                checkpoint_payload(
                    epoch,
                    model,
                    optimizer,
                    aux_optimizer,
                    scheduler,
                    best_loss,
                    patience_counter,
                    args,
                    run_name,
                ),
                save_dir / f"checkpoint_epoch{epoch}.pth.tar",
            )

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
            break

    writer.close()
    print(f"\nTraining complete. Best val loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
