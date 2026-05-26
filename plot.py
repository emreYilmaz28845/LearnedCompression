"""
Plot rate-distortion curves for project variants and pretrained reference models.

Reads evaluation summary JSON files and generates:
- PSNR vs bpp and MS-SSIM vs bpp RD curves
- BD-rate bar charts vs the project baseline
- A consolidated summary CSV for report writing

Usage:
  python plot.py --results-dir results/ --output plots/
"""

import argparse
import csv
import json
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
    "A": "#222222",
    "B": "#0072B2",
    "C": "#D55E00",
    "bmshj2018_factorized": "#009E73",
    "bmshj2018_hyperprior": "#56B4E9",
    "mbt2018_mean": "#CC79A7",
    "mbt2018": "#E69F00",
    "cheng2020_anchor": "#7A7A7A",
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
VARIANT_LINESTYLES = {
    "A": "-",
    "B": "-",
    "C": "-",
    "bmshj2018_factorized": "--",
    "bmshj2018_hyperprior": "--",
    "mbt2018_mean": "-.",
    "mbt2018": "-.",
    "cheng2020_anchor": ":",
}
SHORT_LABELS = {
    "A": "A",
    "B": "B",
    "C": "C",
    "bmshj2018_factorized": "Factorized",
    "bmshj2018_hyperprior": "Hyperprior",
    "mbt2018_mean": "MBT-Mean",
    "mbt2018": "MBT",
    "cheng2020_anchor": "Cheng",
}
ANNOTATION_OFFSETS = {
    "A": (6, -8),
    "B": (6, 2),
    "C": (6, 8),
    "bmshj2018_factorized": (8, 0),
    "bmshj2018_hyperprior": (8, -2),
    "mbt2018_mean": (8, 0),
    "mbt2018": (8, 2),
    "cheng2020_anchor": (8, 0),
}


def ordered_variants(data):
    known = [variant for variant in VARIANT_ORDER if variant in data]
    extra = sorted(variant for variant in data if variant not in VARIANT_ORDER)
    return known + extra


def load_summaries(results_dir):
    """Load all summary JSON files, organized by variant."""
    latest_records = {}
    for f in glob.glob(str(Path(results_dir) / "*_summary.json")):
        path = Path(f)
        with open(f) as fh:
            summary = json.load(fh)
        variant = summary["variant"]
        key = (variant, summary.get("lmbda"), summary.get("quality"))
        mtime = path.stat().st_mtime
        previous = latest_records.get(key)
        if previous is None or mtime >= previous[0]:
            latest_records[key] = (mtime, summary)

    data = {}
    for (_, _, _), (_, summary) in latest_records.items():
        variant = summary["variant"]
        if variant not in data:
            data[variant] = []
        data[variant].append(summary)

    # Sort each variant's results by bpp
    for v in data:
        data[v].sort(key=lambda x: x["avg_bpp"])

    return data


def style_axes(ax):
    ax.set_facecolor("#FAFAFA")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8B8B8")
    ax.spines["bottom"].set_color("#B8B8B8")
    ax.tick_params(colors="#333333", labelsize=11)
    ax.grid(True, which="major", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", color="#EEEEEE", linewidth=0.6, alpha=0.7)


def write_summary_table(data, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "rd_summary_table.csv"

    rows = []
    for variant in ordered_variants(data):
        for point in data[variant]:
            rows.append({
                "variant": variant,
                "label": VARIANT_LABELS.get(variant, variant),
                "lmbda": point.get("lmbda"),
                "quality": point.get("quality"),
                "avg_bpp": point.get("avg_bpp"),
                "avg_psnr": point.get("avg_psnr"),
                "avg_ms_ssim": point.get("avg_ms_ssim"),
                "avg_ms_ssim_db": point.get("avg_ms_ssim_db"),
                "total_params": point.get("total_params"),
                "run_name": point.get("run_name"),
            })

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "label",
                "lmbda",
                "quality",
                "avg_bpp",
                "avg_psnr",
                "avg_ms_ssim",
                "avg_ms_ssim_db",
                "total_params",
                "run_name",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {summary_path}")


def write_parameter_table(data, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "model_params_table.csv"
    md_path = output_dir / "model_params_table.md"
    png_path = output_dir / "model_params_table.png"
    pdf_path = output_dir / "model_params_table.pdf"
    svg_path = output_dir / "model_params_table.svg"

    rows = []
    for variant in ordered_variants(data):
        summaries = data[variant]
        if not summaries:
            continue
        total_params = summaries[0].get("total_params")
        rows.append({
            "variant": variant,
            "label": VARIANT_LABELS.get(variant, variant),
            "total_params": total_params,
            "params_millions": (total_params / 1_000_000.0) if total_params is not None else None,
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "label", "total_params", "params_millions"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(md_path, "w") as f:
        f.write("| Variant | Label | Parameters | Parameters (M) |\n")
        f.write("|---|---|---:|---:|\n")
        for row in rows:
            total_params = row["total_params"]
            params_millions = row["params_millions"]
            f.write(
                f"| `{row['variant']}` | {row['label']} | "
                f"{total_params:,} | {params_millions:.3f} |\n"
            )

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")

    fig_height = 1.2 + 0.55 * max(len(rows), 1)
    fig, ax = plt.subplots(1, 1, figsize=(9.5, fig_height))
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    cell_text = []
    for row in rows:
        cell_text.append([
            row["variant"],
            row["label"],
            f"{row['total_params']:,}",
            f"{row['params_millions']:.3f}",
        ])

    col_labels = ["Variant", "Label", "Parameters", "Parameters (M)"]
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.22, 0.34, 0.22, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.55)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.8)
        if row_idx == 0:
            cell.set_facecolor("#2F4858")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F7F7F7" if row_idx % 2 else "#ECEFF1")
        if col_idx in {2, 3}:
            cell.get_text().set_ha("right")

    fig.suptitle(
        "Model Parameter Counts",
        fontsize=18,
        color="#111111",
        y=0.98,
    )
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=180, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {svg_path}")


def plot_rd_curves(data, output_dir, metric="psnr"):
    """Plot rate-distortion curves."""
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 7.0))
    style_axes(ax)

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
            color=VARIANT_COLORS.get(variant, "#4C78A8"),
            label=VARIANT_LABELS.get(variant, variant),
            linewidth=2.4,
            linestyle=VARIANT_LINESTYLES.get(variant, "-"),
            markersize=8.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
        )

        dx, dy = ANNOTATION_OFFSETS.get(variant, (6, 0))
        ax.annotate(
            SHORT_LABELS.get(variant, variant),
            xy=(bpps[-1], vals[-1]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            color=VARIANT_COLORS.get(variant, "#333333"),
            va="center",
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

    ax.set_xlabel("Bits per pixel (bpp)", fontsize=13, color="#222222")
    ax.set_ylabel(ylabel, fontsize=13, color="#222222")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.margins(x=0.08, y=0.08)
    fig.suptitle(
        f"Kodak Rate-Distortion Comparison ({metric.upper()})",
        fontsize=19,
        color="#111111",
        y=0.97,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=False,
        fontsize=10,
        columnspacing=1.4,
        handlelength=2.8,
    )
    fig.subplots_adjust(top=0.82, right=0.92)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"rd_curve_{metric}.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / f"rd_curve_{metric}.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / f"rd_curve_{metric}.svg", bbox_inches="tight")
    print(f"Saved: {output_dir / f'rd_curve_{metric}.pdf'}")
    plt.close(fig)

    return bd_rates


def plot_bd_rate_bars(bd_rates, output_dir, metric):
    """Plot horizontal BD-rate bars vs the project baseline."""
    if not bd_rates:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = sorted(bd_rates.items(), key=lambda item: item[1])
    labels = [VARIANT_LABELS.get(variant, variant) for variant, _ in items]
    values = [value for _, value in items]
    colors = ["#0072B2" if value <= 0 else "#D55E00" for value in values]

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.5))
    style_axes(ax)

    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.axvline(0.0, color="#444444", linewidth=1.2)
    ax.set_xlabel("BD-rate vs variant A (%)", fontsize=12, color="#222222")
    ax.set_title(
        f"BD-rate Summary on Kodak ({metric.upper()})",
        fontsize=15,
        color="#111111",
        pad=12,
    )

    for bar, value in zip(bars, values):
        x = bar.get_width()
        ax.text(
            x + (0.6 if x >= 0 else -0.6),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}%",
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=10,
            color="#222222",
        )

    fig.tight_layout()
    fig.savefig(output_dir / f"bd_rate_{metric}.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / f"bd_rate_{metric}.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / f"bd_rate_{metric}.svg", bbox_inches="tight")
    print(f"Saved: {output_dir / f'bd_rate_{metric}.pdf'}")
    plt.close(fig)


def write_bd_rate_summary(bd_psnr, bd_msssim, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "bd_rate_summary.csv"

    rows = []
    all_variants = ordered_variants({**{k: [] for k in bd_psnr}, **{k: [] for k in bd_msssim}})
    for variant in all_variants:
        if variant == "A":
            continue
        rows.append({
            "variant": variant,
            "label": VARIANT_LABELS.get(variant, variant),
            "bd_rate_psnr": bd_psnr.get(variant),
            "bd_rate_msssim": bd_msssim.get(variant),
        })

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "label", "bd_rate_psnr", "bd_rate_msssim"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {summary_path}")


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

    write_summary_table(data, args.output)
    write_parameter_table(data, args.output)
    bd_psnr = plot_rd_curves(data, args.output, metric="psnr")
    bd_msssim = plot_rd_curves(data, args.output, metric="msssim")
    plot_bd_rate_bars(bd_psnr, args.output, metric="psnr")
    plot_bd_rate_bars(bd_msssim, args.output, metric="msssim")
    write_bd_rate_summary(bd_psnr, bd_msssim, args.output)

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
