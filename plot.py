"""
Plot rate-distortion curves comparing variants A, B, C.

Reads evaluation summary JSON files and generates PSNR vs bpp and MS-SSIM vs bpp plots.

Usage:
  python plot.py --results-dir results/ --output plots/
"""

import argparse
import json
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.metrics import compute_bd_rate


LAMBDAS = [0.0018, 0.0035, 0.0067, 0.013]
VARIANT_LABELS = {
    "A": "Baseline",
    "B": "Encoder SE",
    "C": "Encoder+Decoder SE",
    "bmshj2018_factorized": "BMSHJ 2018 Factorized",
    "bmshj2018_hyperprior": "BMSHJ 2018 Hyperprior",
    "mbt2018_mean": "MBT 2018 Mean",
    "mbt2018": "MBT 2018",
    "cheng2020_anchor": "Cheng 2020 Anchor",
}
VARIANT_ORDER = [
    "A",
    "B",
    "C",
    "bmshj2018_factorized",
    "bmshj2018_hyperprior",
    "mbt2018_mean",
    "mbt2018",
    "cheng2020_anchor",
]
VARIANT_COLORS = {
    "A": "#1f77b4",
    "B": "#ff7f0e",
    "C": "#2ca02c",
    "bmshj2018_factorized": "#d62728",
    "bmshj2018_hyperprior": "#9467bd",
    "mbt2018_mean": "#8c564b",
    "mbt2018": "#e377c2",
    "cheng2020_anchor": "#7f7f7f",
}
VARIANT_MARKERS = {
    "A": "o",
    "B": "s",
    "C": "D",
    "bmshj2018_factorized": "^",
    "bmshj2018_hyperprior": "v",
    "mbt2018_mean": "P",
    "mbt2018": "X",
    "cheng2020_anchor": "*",
}


def ordered_variants(data):
    known = [variant for variant in VARIANT_ORDER if variant in data]
    extra = sorted(variant for variant in data if variant not in VARIANT_ORDER)
    return known + extra


def load_summaries(results_dir):
    """Load all summary JSON files, organized by variant."""
    data = {}
    for f in glob.glob(str(Path(results_dir) / "*_summary.json")):
        with open(f) as fh:
            summary = json.load(fh)
        variant = summary["variant"]
        if variant not in data:
            data[variant] = []
        data[variant].append(summary)

    # Sort each variant's results by bpp
    for v in data:
        data[v].sort(key=lambda x: x["avg_bpp"])

    return data


def plot_rd_curves(data, output_dir, metric="psnr"):
    """Plot rate-distortion curves."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    bd_rates = {}
    for variant in ordered_variants(data):
        points = data[variant]
        bpps = [p["avg_bpp"] for p in points]
        if metric == "psnr":
            vals = [p["avg_psnr"] for p in points]
            ylabel = "PSNR (dB)"
        else:
            vals = [p["avg_ms_ssim_db"] for p in points]
            ylabel = "MS-SSIM (dB)"

        ax.plot(
            bpps, vals,
            marker=VARIANT_MARKERS.get(variant, "o"),
            color=VARIANT_COLORS.get(variant, None),
            label=VARIANT_LABELS.get(variant, variant),
            linewidth=2,
            markersize=8,
        )

        # Compute BD-rate vs variant A
        if variant != "A" and "A" in data and len(bpps) >= 4:
            ref_bpps = [p["avg_bpp"] for p in data["A"]]
            if metric == "psnr":
                ref_vals = [p["avg_psnr"] for p in data["A"]]
            else:
                ref_vals = [p["avg_ms_ssim_db"] for p in data["A"]]

            if len(ref_bpps) >= 4:
                bd = compute_bd_rate(ref_bpps, ref_vals, bpps, vals)
                bd_rates[variant] = bd

    ax.set_xlabel("Bits per pixel (bpp)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Rate-Distortion Performance on Kodak ({metric.upper()})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add BD-rate text
    if bd_rates:
        text_lines = []
        for v, bd in bd_rates.items():
            text_lines.append(f"BD-rate {VARIANT_LABELS.get(v, v)}: {bd:+.2f}%")
        ax.text(
            0.02, 0.02, "\n".join(text_lines),
            transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"rd_curve_{metric}.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / f"rd_curve_{metric}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / f'rd_curve_{metric}.pdf'}")
    plt.close(fig)

    return bd_rates


def main():
    parser = argparse.ArgumentParser(description="Plot RD curves")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--output", default="experiments/plots")
    args = parser.parse_args()

    data = load_summaries(args.results_dir)
    if not data:
        print(f"No summary files found in {args.results_dir}")
        return

    print(f"Loaded results for variants: {list(data.keys())}")
    for v in data:
        print(f"  {v}: {len(data[v])} operating points")

    bd_psnr = plot_rd_curves(data, args.output, metric="psnr")
    bd_msssim = plot_rd_curves(data, args.output, metric="msssim")

    print("\n--- BD-Rate Summary (vs. Variant A baseline) ---")
    for v in ordered_variants(data):
        if v == "A":
            continue
        if v in bd_psnr:
            print(f"  {VARIANT_LABELS.get(v, v)}:")
            print(f"    PSNR BD-rate:    {bd_psnr[v]:+.2f}%")
        if v in bd_msssim:
            print(f"    MS-SSIM BD-rate: {bd_msssim[v]:+.2f}%")


if __name__ == "__main__":
    main()
