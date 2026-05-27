"""
Training script for learned image compression variants.

Variants:
  A    - Pretrained baseline (evaluation-only checkpoint export)
  A_ft - Fine-tuned vanilla baseline
  B    - Fine-tuned with SE block in the encoder
  C    - Fine-tuned with SE blocks in both encoder and decoder
"""

from __future__ import annotations

import argparse
import math
import random
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import load_pretrained_baseline, load_pretrained_with_se
from utils.config import DEFAULT_CONFIG_PATH, get_config_value, load_config
from utils.datasets import CLICDataset


TRAINABLE_VARIANTS = ["A_ft", "B", "C"]
SUPPORTED_VARIANTS = ["A", *TRAINABLE_VARIANTS]
FREEZE_MODES = ["none", "frozen_hyperprior", "attention_only"]
HYPERPRIOR_PREFIXES = ("h_a.", "h_s.", "entropy_bottleneck.", "gaussian_conditional.")


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


def load_parser_defaults(config):
    return {
        "lmbda": get_config_value(config, "lambdas", default=[0.0067])[2],
        "quality": get_config_value(config, "qualities", default=[1, 2, 3, 4])[2],
        "distortion": get_config_value(config, "training", "distortion", default="mse"),
        "train_dir": get_config_value(config, "paths", "clic_train_dir", default="datasets/clic_images/train"),
        "val_dir": get_config_value(config, "paths", "clic_val_dir", default="datasets/clic_images/validation"),
        "epochs": get_config_value(config, "training", "epochs", default=100),
        "batch_size": get_config_value(config, "training", "batch_size", default=32),
        "lr": get_config_value(config, "training", "lr", default=1e-4),
        "se_lr": get_config_value(config, "training", "se_lr", default=1e-3),
        "patch_size": get_config_value(config, "training", "patch_size", default=256),
        "num_workers": get_config_value(config, "training", "num_workers", default=4),
        "save_dir": get_config_value(config, "paths", "checkpoint_dir", default="checkpoints"),
        "seed": get_config_value(config, "training", "seed", default=42),
        "freeze_mode": get_config_value(config, "training", "freeze_mode", default="none"),
        "early_stop_patience": get_config_value(config, "training", "early_stop_patience", default=20),
        "deterministic": get_config_value(config, "training", "deterministic", default=True),
        "pin_memory": get_config_value(config, "training", "pin_memory", default=True),
        "drop_last": get_config_value(config, "training", "drop_last", default=True),
        "scheduler_factor": get_config_value(config, "scheduler", "factor", default=0.1),
        "scheduler_patience": get_config_value(config, "scheduler", "patience", default=10),
    }


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def build_run_name(args):
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.name.startswith("checkpoint_"):
            return checkpoint_path.parent.name
    base = (
        f"variant_{args.variant}_lmbda_{args.lmbda}_"
        f"quality_{args.quality}_{args.distortion}_freeze_{args.freeze_mode}"
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
        "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer else None,
        "scheduler": scheduler.state_dict(),
        "best_loss": best_loss,
        "patience_counter": patience_counter,
        "variant": args.variant,
        "lmbda": args.lmbda,
        "quality": args.quality,
        "distortion": args.distortion,
        "freeze_mode": args.freeze_mode,
        "run_name": run_name,
    }


def load_training_checkpoint(path, model, optimizer, aux_optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if aux_optimizer and checkpoint.get("aux_optimizer"):
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_loss = float(checkpoint.get("best_loss", float("inf")))
    patience_counter = int(checkpoint.get("patience_counter", 0))
    return start_epoch, best_loss, patience_counter


class RateDistortionLoss(nn.Module):
    """Rate-distortion loss: L = lambda * D(x, x_hat) + R(y_hat) + R(z_hat)."""

    def __init__(self, lmbda=0.0067, distortion="mse"):
        super().__init__()
        self.lmbda = lmbda
        self.distortion = distortion

    def forward(self, output, target):
        n, _, h, w = target.size()
        num_pixels = n * h * w

        bpp_loss = sum(
            -torch.log2(likelihoods).sum() / num_pixels
            for likelihoods in output["likelihoods"].values()
        )

        if self.distortion == "mse":
            distortion = 255.0**2 * nn.functional.mse_loss(output["x_hat"], target)
        else:
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
    if variant in {"A", "A_ft"}:
        return load_pretrained_baseline(quality=quality)
    if variant == "B":
        return load_pretrained_with_se(quality=quality, attention="encoder")
    if variant == "C":
        return load_pretrained_with_se(quality=quality, attention="both")
    raise ValueError(f"Unknown variant: {variant}")


def apply_freeze_mode(model, variant, freeze_mode):
    if freeze_mode == "none":
        return

    if freeze_mode == "frozen_hyperprior":
        for name, parameter in model.named_parameters():
            if name.startswith(HYPERPRIOR_PREFIXES):
                parameter.requires_grad = False
        return

    if freeze_mode == "attention_only":
        if variant not in {"B", "C"}:
            raise ValueError("freeze_mode=attention_only is only valid for variants B and C")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = ".excitation." in name
        return

    raise ValueError(f"Unknown freeze mode: {freeze_mode}")


def configure_optimizers(model, lr, se_lr):
    """Separate parameters into main, SE block, and auxiliary groups."""
    params_dict = dict(model.named_parameters())
    se_parameters = {
        name for name, parameter in model.named_parameters()
        if ".excitation." in name and parameter.requires_grad and not name.endswith(".quantiles")
    }
    aux_parameters = {
        name for name, parameter in model.named_parameters()
        if name.endswith(".quantiles") and parameter.requires_grad
    }
    backbone_parameters = {
        name for name, parameter in model.named_parameters()
        if name not in se_parameters and name not in aux_parameters and parameter.requires_grad
    }

    param_groups = []
    if backbone_parameters:
        param_groups.append(
            {"params": [params_dict[name] for name in sorted(backbone_parameters)], "lr": lr}
        )
    if se_parameters:
        param_groups.append(
            {"params": [params_dict[name] for name in sorted(se_parameters)], "lr": se_lr}
        )

    if not param_groups:
        raise ValueError("No trainable parameters remain after applying the freeze mode")

    optimizer = optim.Adam(param_groups)
    aux_optimizer = None
    if aux_parameters:
        aux_optimizer = optim.Adam(
            (params_dict[name] for name in sorted(aux_parameters)),
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
        if aux_optimizer:
            aux_optimizer.zero_grad()

        out = model(x)
        out["x_hat"] = out["x_hat"].clamp(0, 1)
        losses = criterion(out, x)

        losses["loss"].backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        aux_value = 0.0
        if aux_optimizer:
            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()
            aux_value = aux_loss.item()

        total_loss += losses["loss"].item()
        total_bpp += losses["bpp"].item()
        total_mse += losses["mse"].item()
        total_distortion += losses["distortion"].item()
        total_aux_loss += aux_value
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


def resolve_dataset_args(args):
    if args.data_dir:
        train_dir = args.data_dir
        val_dir = args.val_dir
        random_split = val_dir is None
    else:
        train_dir = args.train_dir
        val_dir = args.val_dir
        random_split = val_dir is None
    return train_dir, val_dir, random_split


def build_data_loaders(args):
    train_dir, val_dir, random_split = resolve_dataset_args(args)
    generator = torch.Generator().manual_seed(args.seed)

    if random_split:
        full_dataset = CLICDataset(train_dir, patch_size=args.patch_size, training=True)
        val_size = max(1, len(full_dataset) // 10)
        train_size = len(full_dataset) - val_size
        train_indices, val_indices = torch.utils.data.random_split(
            range(len(full_dataset)),
            [train_size, val_size],
            generator=generator,
        )
        train_dataset = torch.utils.data.Subset(
            CLICDataset(train_dir, patch_size=args.patch_size, training=True),
            train_indices.indices,
        )
        val_dataset = torch.utils.data.Subset(
            CLICDataset(train_dir, patch_size=args.patch_size, training=False),
            val_indices.indices,
        )
    else:
        train_dataset = CLICDataset(train_dir, patch_size=args.patch_size, training=True)
        val_dataset = CLICDataset(val_dir, patch_size=args.patch_size, training=False)

    if args.max_train_samples is not None:
        train_limit = min(args.max_train_samples, len(train_dataset))
        train_dataset = torch.utils.data.Subset(train_dataset, range(train_limit))

    if args.max_val_samples is not None:
        val_limit = min(args.max_val_samples, len(val_dataset))
        val_dataset = torch.utils.data.Subset(val_dataset, range(val_limit))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return train_loader, val_loader, len(train_dataset), len(val_dataset), random_split, train_dir, val_dir


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    config_args, remaining = config_parser.parse_known_args()
    config = load_config(config_args.config)
    defaults = load_parser_defaults(config)

    parser = argparse.ArgumentParser(
        description="Train learned image compression",
        parents=[config_parser],
    )
    parser.add_argument("--variant", choices=SUPPORTED_VARIANTS, required=True)
    parser.add_argument("--lmbda", type=float, default=defaults["lmbda"])
    parser.add_argument("--quality", type=int, default=defaults["quality"])
    parser.add_argument("--distortion", choices=["mse", "msssim"], default=defaults["distortion"])
    parser.add_argument("--train-dir", default=defaults["train_dir"])
    parser.add_argument("--val-dir", default=defaults["val_dir"])
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Legacy single-root dataset path. If used without --val-dir, training falls back to a random split.",
    )
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--se-lr", type=float, default=defaults["se_lr"])
    parser.add_argument("--patch-size", type=int, default=defaults["patch_size"])
    parser.add_argument("--num-workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--save-dir", default=defaults["save_dir"])
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--freeze-mode", choices=FREEZE_MODES, default=defaults["freeze_mode"])
    parser.add_argument("--early-stop-patience", type=int, default=defaults["early_stop_patience"])
    parser.add_argument("--scheduler-factor", type=float, default=defaults["scheduler_factor"])
    parser.add_argument("--scheduler-patience", type=int, default=defaults["scheduler_patience"])
    parser.add_argument("--deterministic", dest="deterministic", action="store_true")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=defaults["deterministic"])
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=defaults["pin_memory"])
    parser.add_argument("--drop-last", dest="drop_last", action="store_true")
    parser.add_argument("--no-drop-last", dest="drop_last", action="store_false")
    parser.set_defaults(drop_last=defaults["drop_last"])

    args = parser.parse_args(remaining)
    args.config = config_args.config
    args._config = config
    return args


def main():
    args = parse_args()
    seed_everything(args.seed, deterministic=args.deterministic)
    device = select_device()
    print(f"Device: {device}")

    run_name = build_run_name(args)
    save_root = Path(args.save_dir)
    save_dir = save_root / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    latest_pointer = save_root / (
        f"latest_variant_{args.variant}_lmbda_{args.lmbda}_"
        f"quality_{args.quality}_{args.distortion}_freeze_{args.freeze_mode}.txt"
    )
    latest_pointer.write_text(str(save_dir / "checkpoint_best.pth.tar") + "\n")
    print(f"Run directory: {save_dir}")
    print(f"Latest checkpoint pointer: {latest_pointer}")
    print(f"Config: {args.config}")

    if args.variant == "A":
        print("Variant A: saving pretrained baseline (no training)")
        model = build_model(args.variant, args.quality).to(device)
        model.update()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "variant": "A",
                "lmbda": args.lmbda,
                "quality": args.quality,
                "freeze_mode": args.freeze_mode,
                "run_name": run_name,
            },
            save_dir / "checkpoint_best.pth.tar",
        )
        print(f"Saved to {save_dir / 'checkpoint_best.pth.tar'}")
        return

    model = build_model(args.variant, args.quality)
    apply_freeze_mode(model, args.variant, args.freeze_mode)
    model = model.to(device)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    train_loader, val_loader, train_len, val_len, random_split, train_dir, val_dir = build_data_loaders(args)
    print(f"Training images: {train_len}")
    print(f"Validation images: {val_len}")
    print(f"Train dir: {train_dir}")
    print(f"Val dir:   {val_dir or '(random split from train dir)'}")
    print(f"Random split fallback: {random_split}")

    criterion = RateDistortionLoss(lmbda=args.lmbda, distortion=args.distortion)
    optimizer, aux_optimizer = configure_optimizers(model, args.lr, args.se_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    writer = SummaryWriter(log_dir=str(save_dir / "logs"))
    writer.add_text("run/variant", args.variant)
    writer.add_text("run/config_path", args.config)
    writer.add_text("run/train_dir", str(train_dir))
    writer.add_text("run/val_dir", str(val_dir))
    writer.add_text("run/freeze_mode", args.freeze_mode)
    writer.add_scalar("run/total_params", total_params, 0)
    writer.add_scalar("run/trainable_params", trainable_params, 0)
    writer.add_scalar("run/lambda", args.lmbda, 0)
    writer.add_scalar("run/quality", args.quality, 0)
    writer.add_scalar("run/batch_size", args.batch_size, 0)
    writer.add_scalar("run/patch_size", args.patch_size, 0)

    best_loss = float("inf")
    patience_counter = 0
    start_epoch = 1

    if args.resume:
        start_epoch, best_loss, patience_counter = load_training_checkpoint(
            args.resume,
            model,
            optimizer,
            aux_optimizer,
            scheduler,
            device,
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
            train_metrics = train_one_epoch(model, criterion, train_loader, optimizer, aux_optimizer, device)
            val_metrics = validate(model, criterion, val_loader, device)

        scheduler.step(val_metrics["loss"])
        epoch_time = time.perf_counter() - epoch_start
        gpu_stats = get_gpu_stats(device)
        gpu_stats.update(gpu_sampler.summary())

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
        writer.add_scalar("lr/aux", aux_optimizer.param_groups[0]["lr"] if aux_optimizer else 0.0, epoch)
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

        if patience_counter >= args.early_stop_patience:
            print(
                f"  Early stopping at epoch {epoch} "
                f"(no improvement for {args.early_stop_patience} epochs)"
            )
            break

    writer.close()
    print(f"\nTraining complete. Best val loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
