from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from PIL import Image
from tensorboard.backend.event_processing import event_accumulator


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"
PLOTS_DIR = ROOT / "experiments" / "plots"
RECON_DIR = ROOT / "experiments" / "reconstructions"
CHECKPOINT_DIR = ROOT / "checkpoints"
CLIC_TRAIN_DIR = ROOT / "datasets" / "clic_images" / "train"


def ensure_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_scalars(log_dir: Path, tag: str) -> tuple[list[int], list[float]]:
    acc = event_accumulator.EventAccumulator(str(log_dir))
    acc.Reload()
    events = acc.Scalars(tag)
    return [event.step for event in events], [event.value for event in events]


def generate_convergence_plot() -> None:
    runs = {
        "A_ft": CHECKPOINT_DIR / "variant_A_ft_lmbda_0.0067_quality_3_mse_freeze_none_20260528_011518" / "logs",
        "B": CHECKPOINT_DIR / "variant_B_lmbda_0.0067_quality_3_mse_freeze_none_20260528_011518" / "logs",
        "C": CHECKPOINT_DIR / "variant_C_lmbda_0.0067_quality_3_mse_freeze_none_20260528_011518" / "logs",
    }
    labels = {
        "A_ft": "Fine-tuned Baseline",
        "B": "Encoder SE",
        "C": "Encoder+Decoder SE",
    }
    colors = {
        "A_ft": "#111111",
        "B": "#0072B2",
        "C": "#D55E00",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    for variant, log_dir in runs.items():
        epochs, val_loss = load_scalars(log_dir, "val/loss")
        _, val_psnr = load_scalars(log_dir, "val/psnr")
        axes[0].plot(epochs, val_loss, color=colors[variant], linewidth=2.0, label=labels[variant])
        axes[1].plot(epochs, val_psnr, color=colors[variant], linewidth=2.0, label=labels[variant])

    axes[0].set_title("Validation RD Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation PSNR")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PSNR (dB)")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "convergence_q3_none.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "convergence_q3_none.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def style_axes(ax) -> None:
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def generate_preprocessing_panel() -> None:
    image_path = sorted(CLIC_TRAIN_DIR.glob("*.png"))[0]
    image = load_image(image_path)
    h, w = image.shape[:2]
    crop_size = 256
    x0 = max((w - crop_size) // 2, 0)
    y0 = max((h - crop_size) // 2, 0)
    x1 = min(x0 + crop_size, w)
    y1 = min(y0 + crop_size, h)
    crop = image[y0:y1, x0:x1]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
    axes[0].imshow(image)
    axes[0].add_patch(
        patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2.5, edgecolor="#D55E00", facecolor="none")
    )
    axes[0].set_title("Example CLIC image", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(crop)
    axes[1].set_title("256x256 training patch", fontsize=11)
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "preprocessing_patch_example.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "preprocessing_patch_example.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_reconstruction_panel() -> None:
    crop = (160, 100, 360, 260)  # x0, y0, x1, y1
    image_name = "kodim03"
    image_paths = {
        "Input": ROOT / "datasets" / "kodak" / f"{image_name}.png",
        "A_ft": RECON_DIR / "variant_A_ft_lmbda_0.0067_quality_3_mse_freeze_none_20260528_011518" / f"{image_name}_recon.png",
        "B": RECON_DIR / "variant_B_lmbda_0.0067_quality_3_mse_freeze_none_20260528_011518" / f"{image_name}_recon.png",
        "C": RECON_DIR / "variant_C_lmbda_0.0067_quality_3_mse_freeze_none_20260528_011518" / f"{image_name}_recon.png",
    }
    display_titles = {
        "Input": "Input",
        "A_ft": "Fine-tuned Baseline",
        "B": "Encoder SE",
        "C": "Encoder+Decoder SE",
    }

    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.6))
    x0, y0, x1, y1 = crop
    for col, (key, path) in enumerate(image_paths.items()):
        image = load_image(path)
        axes[0, col].imshow(image)
        axes[0, col].add_patch(
            patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="#D55E00", facecolor="none")
        )
        axes[0, col].set_title(display_titles[key], fontsize=10)
        axes[0, col].axis("off")

        crop_image = image[y0:y1, x0:x1]
        axes[1, col].imshow(crop_image)
        axes[1, col].axis("off")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "reconstruction_panel_q3_kodim03.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "reconstruction_panel_q3_kodim03.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_bitrate_gap_table() -> None:
    df = pd.read_csv(PLOTS_DIR / "rd_summary_table.csv")
    df = df[df["avg_coded_bpp"].notna()].copy()
    df["coded_minus_estimated_bpp"] = df["avg_coded_bpp"] - df["avg_estimated_bpp"]
    keep = [
        "variant",
        "freeze_mode",
        "quality",
        "avg_estimated_bpp",
        "avg_coded_bpp",
        "coded_minus_estimated_bpp",
    ]
    df[keep].to_csv(FIG_DIR / "coded_vs_estimated_bpp.csv", index=False)


def runtime_summary_df() -> pd.DataFrame:
    return pd.read_csv(PLOTS_DIR / "runtime_tradeoff_summary.csv")


def generate_runtime_components_plot() -> None:
    df = runtime_summary_df()
    df = df[df["variant"].isin(["A_ft", "B", "C"])].copy()
    labels = ["Fine-tuned\nbaseline", "Encoder\nSE", "Encoder+\nDecoder SE"]
    forward = df.set_index("variant").loc[["A_ft", "B", "C"], "mean_forward_ms"].to_numpy()
    encode = df.set_index("variant").loc[["A_ft", "B", "C"], "mean_encode_ms"].to_numpy()
    decode = df.set_index("variant").loc[["A_ft", "B", "C"], "mean_decode_ms"].to_numpy()

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(1, 1, figsize=(8.3, 4.6))
    ax.bar(x - width, forward, width=width, color="#4C4C4C", label="Forward")
    ax.bar(x, encode, width=width, color="#0072B2", label="Encode")
    ax.bar(x + width, decode, width=width, color="#D55E00", label="Decode")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Time (ms/image)")
    ax.set_title("Inference time breakdown")
    style_axes(ax)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "runtime_components_project.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "runtime_components_project.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_runtime_comparison_plot() -> None:
    df = runtime_summary_df().copy()
    label_map = {
        "A_ft": "Fine-tuned baseline",
        "A": "Pretrained baseline",
        "B": "Encoder SE",
        "C": "Encoder+Decoder SE",
        "bmshj2018_factorized": "BMSHJ factorized",
        "mbt2018_mean": "MBT mean",
        "mbt2018": "MBT 2018",
        "cheng2020_anchor": "Cheng 2020",
    }
    order = ["bmshj2018_factorized", "A_ft", "A", "B", "C", "mbt2018_mean", "mbt2018", "cheng2020_anchor"]
    df["label_clean"] = df["variant"].map(label_map)
    df = df.set_index("variant").loc[order].reset_index()

    colors = [
        "#009E73" if row["variant"] == "bmshj2018_factorized"
        else "#111111" if row["variant"] == "A_ft"
        else "#555555" if row["variant"] == "A"
        else "#0072B2" if row["variant"] == "B"
        else "#D55E00" if row["variant"] == "C"
        else "#CC79A7" if row["variant"] == "mbt2018_mean"
        else "#E69F00" if row["variant"] == "mbt2018"
        else "#888888"
        for _, row in df.iterrows()
    ]

    fig, ax = plt.subplots(1, 1, figsize=(8.6, 5.4))
    ax.barh(df["label_clean"], df["mean_codec_total_ms"], color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("Mean encode+decode time (ms/image, log scale)")
    ax.set_title("Evaluation runtime across local reference implementations")
    style_axes(ax)
    for idx, value in enumerate(df["mean_codec_total_ms"]):
        ax.text(value * 1.03, idx, f"{value:.0f} ms", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "runtime_comparison_all.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "runtime_comparison_all.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_runtime_bitrate_tradeoff_plot() -> None:
    df = runtime_summary_df().copy()
    label_map = {
        "A_ft": "A_ft",
        "A": "A",
        "B": "B",
        "C": "C",
        "bmshj2018_factorized": "Factorized",
        "mbt2018_mean": "MBT mean",
        "mbt2018": "MBT",
        "cheng2020_anchor": "Cheng",
    }
    colors = {
        "A_ft": "#111111",
        "A": "#555555",
        "B": "#0072B2",
        "C": "#D55E00",
        "bmshj2018_factorized": "#009E73",
        "mbt2018_mean": "#CC79A7",
        "mbt2018": "#E69F00",
        "cheng2020_anchor": "#888888",
    }
    offsets = {
        "A_ft": (16, 6),
        "A": (10, 6),
        "B": (10, -10),
        "C": (14, 16),
        "bmshj2018_factorized": (10, 6),
        "mbt2018_mean": (10, 6),
        "mbt2018": (10, 6),
        "cheng2020_anchor": (10, 6),
    }

    fig, ax = plt.subplots(1, 1, figsize=(8.6, 4.9))
    for _, row in df.iterrows():
        ax.scatter(
            row["mean_codec_total_ms"],
            row["mean_coded_bpp"],
            s=160,
            color=colors[row["variant"]],
            edgecolors="white",
            linewidths=0.8,
        )
        ax.annotate(
            label_map[row["variant"]],
            (row["mean_codec_total_ms"], row["mean_coded_bpp"]),
            xytext=offsets[row["variant"]],
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Mean encode+decode time (ms/image, log scale)")
    ax.set_ylabel("Mean coded bpp")
    ax.set_title("Bitrate-runtime tradeoff")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "runtime_bitrate_tradeoff_all.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "runtime_bitrate_tradeoff_all.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dir()
    generate_preprocessing_panel()
    generate_convergence_plot()
    generate_reconstruction_panel()
    generate_runtime_components_plot()
    generate_runtime_comparison_plot()
    generate_runtime_bitrate_tradeoff_plot()
    generate_bitrate_gap_table()


if __name__ == "__main__":
    main()
