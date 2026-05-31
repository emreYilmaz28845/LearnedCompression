"""
Plot rate-distortion curves for project variants and pretrained reference models.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from utils.config import DEFAULT_CONFIG_PATH, get_config_value, load_config
from utils.metrics import compute_bd_rate


matplotlib.use("Agg")

VARIANT_LABELS = {
    "A": "Pretrained Baseline",
    "A_ft": "Fine-tuned Baseline",
    "B": "Encoder SE",
    "C": "Encoder+Decoder SE",
    "bmshj2018_factorized": "BMSHJ 2018 Factorized",
    "bmshj2018_hyperprior": "BMSHJ 2018 Hyperprior",
    "mbt2018_mean": "MBT 2018 Mean",
    "mbt2018": "MBT 2018",
    "cheng2020_anchor": "Cheng 2020 Anchor",
}
VARIANT_ORDER = [
    "A_ft",
    "A",
    "B",
    "C",
    "bmshj2018_factorized",
    "bmshj2018_hyperprior",
    "mbt2018_mean",
    "mbt2018",
    "cheng2020_anchor",
]
PROJECT_VARIANTS = ["A_ft", "A", "B", "C"]
VARIANT_COLORS = {
    "A": "#4D4D4D",
    "A_ft": "#111111",
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
    "A_ft": "o",
    "B": "s",
    "C": "D",
    "bmshj2018_factorized": "^",
    "bmshj2018_hyperprior": "v",
    "mbt2018_mean": "P",
    "mbt2018": "X",
    "cheng2020_anchor": "*",
}
VARIANT_LINESTYLES = {
    "A": "--",
    "A_ft": "-",
    "B": "-",
    "C": "-",
    "bmshj2018_factorized": "--",
    "bmshj2018_hyperprior": "--",
    "mbt2018_mean": "-.",
    "mbt2018": "-.",
    "cheng2020_anchor": ":",
}


def load_parser_defaults(config):
    return {
        "results_dir": get_config_value(config, "paths", "results_dir", default="experiments/results"),
        "output": get_config_value(config, "paths", "plots_dir", default="experiments/plots"),
    }


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    config_args, remaining = config_parser.parse_known_args()
    config = load_config(config_args.config)
    defaults = load_parser_defaults(config)

    parser = argparse.ArgumentParser(description="Plot RD curves", parents=[config_parser])
    parser.add_argument("--results-dir", default=defaults["results_dir"])
    parser.add_argument("--output", default=defaults["output"])
    args = parser.parse_args(remaining)
    args.config = config_args.config
    return args


def ordered_variants(data):
    known = [variant for variant in VARIANT_ORDER if variant in data]
    extra = sorted(variant for variant in data if variant not in VARIANT_ORDER)
    return known + extra


def plot_bpp(summary):
    coded = summary.get("avg_coded_bpp")
    if coded is not None:
        return coded
    return summary.get("avg_estimated_bpp", summary.get("avg_bpp"))


def load_summaries(results_dir):
    latest_records = {}
    for filename in glob.glob(str(Path(results_dir) / "*_summary.json")):
        path = Path(filename)
        with open(path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        key = (
            summary.get("variant"),
            summary.get("lmbda"),
            summary.get("quality"),
            summary.get("freeze_mode", "unknown"),
        )
        previous = latest_records.get(key)
        mtime = path.stat().st_mtime
        if previous is None or mtime >= previous[0]:
            latest_records[key] = (mtime, summary)

    data = {}
    for (_, _, _, _), (_, summary) in latest_records.items():
        data.setdefault(summary["variant"], []).append(summary)

    for variant in data:
        data[variant].sort(key=plot_bpp)

    return data


def is_finite_point(summary, metric="psnr"):
    bpp = plot_bpp(summary)
    value = summary.get("avg_psnr") if metric == "psnr" else summary.get("avg_ms_ssim_db")
    return (
        bpp is not None
        and value is not None
        and math.isfinite(float(bpp))
        and math.isfinite(float(value))
    )


def curve_priority(freeze_mode):
    priorities = {
        "pretrained_reference": 0,
        "none": 1,
        "": 2,
        None: 2,
        "frozen_hyperprior": 3,
        "attention_only": 4,
    }
    return priorities.get(freeze_mode, 5)


def select_curve_data(data, metric):
    curve_data = {}
    for variant, summaries in data.items():
        groups = {}
        for summary in summaries:
            if not is_finite_point(summary, metric):
                continue
            freeze_mode = summary.get("freeze_mode")
            groups.setdefault(freeze_mode, []).append(summary)

        if not groups:
            continue

        selected_freeze, selected_points = sorted(
            groups.items(),
            key=lambda item: (-len(item[1]), curve_priority(item[0]), str(item[0])),
        )[0]
        selected_points = sorted(selected_points, key=plot_bpp)
        curve_data[variant] = selected_points
    return curve_data


def subset_curve_data(curve_data, variants):
    return {variant: curve_data[variant] for variant in variants if variant in curve_data}


def axis_bounds(curve_data, metric):
    all_bpps = []
    all_vals = []
    for points in curve_data.values():
        all_bpps.extend(plot_bpp(point) for point in points)
        all_vals.extend(
            point["avg_psnr"] if metric == "psnr" else point["avg_ms_ssim_db"]
            for point in points
        )

    if not all_bpps or not all_vals:
        return None, None

    x_min, x_max = min(all_bpps), max(all_bpps)
    y_min, y_max = min(all_vals), max(all_vals)
    x_pad = max((x_max - x_min) * 0.08, 0.005)
    y_pad = max((y_max - y_min) * 0.15, 0.08)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def style_axes(ax, tick_labelsize=11):
    ax.set_facecolor("#FAFAFA")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8B8B8")
    ax.spines["bottom"].set_color("#B8B8B8")
    ax.tick_params(colors="#333333", labelsize=tick_labelsize)
    ax.grid(True, which="major", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", color="#EEEEEE", linewidth=0.6, alpha=0.7)


def apply_horizontal_jitter(records, threshold=0.3, max_jitter=0.1):
    if not records:
        return records

    records = sorted(records, key=lambda record: record["x"])
    groups = []
    current = [records[0]]
    for record in records[1:]:
        if abs(record["x"] - current[-1]["x"]) <= threshold:
            current.append(record)
        else:
            groups.append(current)
            current = [record]
    groups.append(current)

    for group in groups:
        if len(group) == 1:
            group[0]["x_plot"] = group[0]["x"]
            continue
        offsets = np.linspace(-max_jitter, max_jitter, len(group))
        for record, offset in zip(group, offsets):
            record["x_plot"] = record["x"] + float(offset)
    return records


def broken_runtime_limits(values):
    finite = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if len(finite) < 3:
        return None

    top = finite[-1]
    second = finite[-2]
    if top < second * 2.5:
        return None

    lower_max = second * 1.18
    upper_min = max(second * 1.55, top * 0.82)
    upper_max = top * 1.08
    if upper_min >= upper_max:
        return None
    return (0.0, lower_max), (upper_min, upper_max)


def write_summary_table(data, output_dir):
    summary_path = Path(output_dir) / "rd_summary_table.csv"
    rows = []
    for variant in ordered_variants(data):
        for point in data[variant]:
            rows.append({
                "variant": variant,
                "label": VARIANT_LABELS.get(variant, variant),
                "freeze_mode": point.get("freeze_mode"),
                "lmbda": point.get("lmbda"),
                "quality": point.get("quality"),
                "avg_estimated_bpp": point.get("avg_estimated_bpp", point.get("avg_bpp")),
                "avg_coded_bpp": point.get("avg_coded_bpp"),
                "avg_psnr": point.get("avg_psnr"),
                "avg_ms_ssim": point.get("avg_ms_ssim"),
                "avg_ms_ssim_db": point.get("avg_ms_ssim_db"),
                "avg_forward_ms": point.get("avg_forward_ms"),
                "avg_encode_ms": point.get("avg_encode_ms"),
                "avg_decode_ms": point.get("avg_decode_ms"),
                "avg_codec_total_ms": point.get("avg_codec_total_ms"),
                "avg_forward_mpix_per_s": point.get("avg_forward_mpix_per_s"),
                "avg_encode_mpix_per_s": point.get("avg_encode_mpix_per_s"),
                "avg_decode_mpix_per_s": point.get("avg_decode_mpix_per_s"),
                "avg_codec_total_mpix_per_s": point.get("avg_codec_total_mpix_per_s"),
                "total_params": point.get("total_params"),
                "run_name": point.get("run_name"),
            })

    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved: {summary_path}")


def write_parameter_table(data, output_dir):
    output_dir = Path(output_dir)
    rows = []
    for variant in ordered_variants(data):
        for point in data[variant]:
            rows.append({
                "variant": variant,
                "label": VARIANT_LABELS.get(variant, variant),
                "quality": point.get("quality"),
                "freeze_mode": point.get("freeze_mode"),
                "total_params": point.get("total_params"),
                "params_millions": (
                    point.get("total_params") / 1_000_000.0
                    if point.get("total_params") is not None
                    else None
                ),
            })

    csv_path = output_dir / "model_params_table.csv"
    md_path = output_dir / "model_params_table.md"
    png_path = output_dir / "model_params_table.png"
    pdf_path = output_dir / "model_params_table.pdf"
    svg_path = output_dir / "model_params_table.svg"

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant", "label", "quality", "freeze_mode", "total_params", "params_millions"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("| Variant | Label | Quality | Freeze | Parameters | Parameters (M) |\n")
        handle.write("|---|---|---:|---|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| `{row['variant']}` | {row['label']} | {row['quality']} | "
                f"{row['freeze_mode']} | {row['total_params']:,} | {row['params_millions']:.3f} |\n"
            )

    fig_height = 1.2 + 0.5 * max(len(rows), 1)
    fig, ax = plt.subplots(1, 1, figsize=(10.5, fig_height))
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")
    table = ax.table(
        cellText=[
            [
                row["variant"],
                row["label"],
                row["quality"],
                row["freeze_mode"],
                f"{row['total_params']:,}",
                f"{row['params_millions']:.3f}",
            ]
            for row in rows
        ],
        colLabels=["Variant", "Label", "Quality", "Freeze", "Parameters", "Parameters (M)"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=180, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {svg_path}")


def write_runtime_tradeoff_tables(data, output_dir):
    output_dir = Path(output_dir)
    curve_data = select_curve_data(data, metric="psnr")
    point_rows = []
    summary_rows = []

    for variant in ordered_variants(curve_data):
        points = curve_data[variant]
        timed_points = [
            point for point in points
            if point.get("avg_forward_ms") is not None
            and point.get("avg_encode_ms") is not None
            and point.get("avg_decode_ms") is not None
            and point.get("avg_codec_total_ms") is not None
        ]
        if not timed_points:
            continue

        points_for_summary = []
        for point in timed_points:
            point_rows.append({
                "variant": variant,
                "label": VARIANT_LABELS.get(variant, variant),
                "freeze_mode": point.get("freeze_mode"),
                "quality": point.get("quality"),
                "lmbda": point.get("lmbda"),
                "total_params": point.get("total_params"),
                "params_millions": (
                    point.get("total_params") / 1_000_000.0
                    if point.get("total_params") is not None
                    else None
                ),
                "avg_coded_bpp": point.get("avg_coded_bpp"),
                "avg_psnr": point.get("avg_psnr"),
                "avg_forward_ms": point.get("avg_forward_ms"),
                "avg_encode_ms": point.get("avg_encode_ms"),
                "avg_decode_ms": point.get("avg_decode_ms"),
                "avg_codec_total_ms": point.get("avg_codec_total_ms"),
                "avg_forward_mpix_per_s": point.get("avg_forward_mpix_per_s"),
                "avg_encode_mpix_per_s": point.get("avg_encode_mpix_per_s"),
                "avg_decode_mpix_per_s": point.get("avg_decode_mpix_per_s"),
                "avg_codec_total_mpix_per_s": point.get("avg_codec_total_mpix_per_s"),
            })
            points_for_summary.append(point)

        summary_rows.append({
            "variant": variant,
            "label": VARIANT_LABELS.get(variant, variant),
            "freeze_mode": points_for_summary[0].get("freeze_mode"),
            "params_millions": (
                points_for_summary[0].get("total_params") / 1_000_000.0
                if points_for_summary[0].get("total_params") is not None
                else None
            ),
            "timed_points": len(points_for_summary),
            "mean_coded_bpp": float(np.mean([p.get("avg_coded_bpp") for p in points_for_summary])),
            "mean_psnr": float(np.mean([p.get("avg_psnr") for p in points_for_summary])),
            "mean_forward_ms": float(np.mean([p.get("avg_forward_ms") for p in points_for_summary])),
            "mean_encode_ms": float(np.mean([p.get("avg_encode_ms") for p in points_for_summary])),
            "mean_decode_ms": float(np.mean([p.get("avg_decode_ms") for p in points_for_summary])),
            "mean_codec_total_ms": float(np.mean([p.get("avg_codec_total_ms") for p in points_for_summary])),
            "mean_forward_mpix_per_s": float(np.mean([p.get("avg_forward_mpix_per_s") for p in points_for_summary])),
            "mean_encode_mpix_per_s": float(np.mean([p.get("avg_encode_mpix_per_s") for p in points_for_summary])),
            "mean_decode_mpix_per_s": float(np.mean([p.get("avg_decode_mpix_per_s") for p in points_for_summary])),
            "mean_codec_total_mpix_per_s": float(np.mean([p.get("avg_codec_total_mpix_per_s") for p in points_for_summary])),
        })

    point_csv = output_dir / "runtime_tradeoff_table.csv"
    point_md = output_dir / "runtime_tradeoff_table.md"
    summary_csv = output_dir / "runtime_tradeoff_summary.csv"
    summary_md = output_dir / "runtime_tradeoff_summary.md"

    with open(point_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(point_rows[0].keys()) if point_rows else [])
        if point_rows:
            writer.writeheader()
            writer.writerows(point_rows)

    with open(point_md, "w", encoding="utf-8") as handle:
        handle.write("| Variant | Freeze | Q | Params (M) | Coded bpp | PSNR | Forward ms | Encode ms | Decode ms | Codec total ms |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in point_rows:
            handle.write(
                f"| {row['label']} | {row['freeze_mode']} | {row['quality']} | "
                f"{row['params_millions']:.3f} | {row['avg_coded_bpp']:.4f} | {row['avg_psnr']:.3f} | "
                f"{row['avg_forward_ms']:.2f} | {row['avg_encode_ms']:.2f} | {row['avg_decode_ms']:.2f} | "
                f"{row['avg_codec_total_ms']:.2f} |\n"
            )

    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)

    with open(summary_md, "w", encoding="utf-8") as handle:
        handle.write("| Variant | Freeze | Params (M) | Mean coded bpp | Mean PSNR | Mean forward ms | Mean encode ms | Mean decode ms | Mean codec total ms |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            handle.write(
                f"| {row['label']} | {row['freeze_mode']} | {row['params_millions']:.3f} | "
                f"{row['mean_coded_bpp']:.4f} | {row['mean_psnr']:.3f} | {row['mean_forward_ms']:.2f} | "
                f"{row['mean_encode_ms']:.2f} | {row['mean_decode_ms']:.2f} | {row['mean_codec_total_ms']:.2f} |\n"
            )

    print(f"Saved: {point_csv}")
    print(f"Saved: {point_md}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_md}")


def timed_curve_data(data):
    curve_data = select_curve_data(data, metric="psnr")
    timed = {}
    for variant, points in curve_data.items():
        filtered = [
            point for point in points
            if point.get("avg_codec_total_ms") is not None
            and point.get("avg_forward_ms") is not None
            and point.get("avg_encode_ms") is not None
            and point.get("avg_decode_ms") is not None
        ]
        if filtered:
            timed[variant] = filtered
    return timed


def plot_runtime_tradeoff(data, output_dir, variants=None, suffix=""):
    curve_data = timed_curve_data(data)
    if variants is not None:
        curve_data = subset_curve_data(curve_data, variants)

    if not curve_data:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 7.0))
    style_axes(ax)

    for variant in ordered_variants(curve_data):
        points = curve_data[variant]
        runtimes = [point["avg_codec_total_ms"] for point in points]
        psnrs = [point["avg_psnr"] for point in points]
        bpps = [plot_bpp(point) for point in points]
        scatter = ax.scatter(
            runtimes,
            psnrs,
            s=[90 + 220 * bpp for bpp in bpps],
            marker=VARIANT_MARKERS.get(variant, "o"),
            color=VARIANT_COLORS.get(variant, "#4C78A8"),
            edgecolors="white",
            linewidths=0.9,
            alpha=0.95,
            label=VARIANT_LABELS.get(variant, variant),
        )
        ax.plot(
            runtimes,
            psnrs,
            color=VARIANT_COLORS.get(variant, "#4C78A8"),
            linestyle=VARIANT_LINESTYLES.get(variant, "-"),
            linewidth=1.8,
            alpha=0.85,
        )
        for runtime, psnr, point in zip(runtimes, psnrs, points):
            ax.annotate(
                f"Q{point['quality']}",
                (runtime, psnr),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8.5,
                color="#333333",
            )

    ax.set_xlabel("Codec total time per image (ms)", fontsize=13, color="#222222")
    ax.set_ylabel("PSNR (dB)", fontsize=13, color="#222222")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=10)
    ax.margins(x=0.1, y=0.08)
    fig.subplots_adjust(top=0.82, right=0.94)

    output_dir = Path(output_dir)
    stem = f"runtime_tradeoff_psnr{suffix}"
    fig.savefig(output_dir / f"{stem}.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / f'{stem}.pdf'}")


def plot_runtime_param_scatter(data, output_dir, variants=None, suffix=""):
    curve_data = timed_curve_data(data)
    if variants is not None:
        curve_data = subset_curve_data(curve_data, variants)

    if not curve_data:
        return

    records = []
    for variant in ordered_variants(curve_data):
        points = curve_data[variant]
        mean_params = points[0].get("total_params")
        if mean_params is None:
            continue
        records.append({
            "variant": variant,
            "label": VARIANT_LABELS.get(variant, variant),
            "x": mean_params / 1_000_000.0,
            "y": float(np.mean([point["avg_codec_total_ms"] for point in points])),
            "mean_psnr": float(np.mean([point["avg_psnr"] for point in points])),
            "mean_bpp": float(np.mean([plot_bpp(point) for point in points])),
        })

    records = apply_horizontal_jitter(records)
    y_limits = broken_runtime_limits([record["y"] for record in records])

    if y_limits is None:
        fig, axes = plt.subplots(1, 1, figsize=(10.5, 7.0))
        axes = [axes]
    else:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(10.5, 7.8),
            sharex=True,
            gridspec_kw={"height_ratios": [1.1, 2.4], "hspace": 0.08},
        )

    for ax in axes:
        style_axes(ax)

    legend_handles = []
    legend_labels = []

    for record in records:
        target_axes = axes if y_limits is None else ([axes[0]] if record["y"] >= y_limits[1][0] else [axes[1]])
        for ax in target_axes:
            handle = ax.scatter(
                [record["x_plot"]],
                [record["y"]],
                s=max(120, 140 + 260 * record["mean_bpp"]),
                marker=VARIANT_MARKERS.get(record["variant"], "o"),
                color=VARIANT_COLORS.get(record["variant"], "#4C78A8"),
                edgecolors="white",
                linewidths=1.0,
                alpha=0.97,
                label=record["label"],
                zorder=3,
            )
            if record["label"] not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(record["label"])

    if y_limits is None:
        ax = axes[0]
        ax.set_ylabel("Mean codec total time (ms/image)", fontsize=13, color="#222222")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.margins(x=0.14, y=0.16)
    else:
        top_ax, bottom_ax = axes
        top_ax.set_ylim(*y_limits[1])
        bottom_ax.set_ylim(*y_limits[0])
        top_ax.spines["bottom"].set_visible(False)
        bottom_ax.spines["top"].set_visible(False)
        top_ax.tick_params(labeltop=False, bottom=False)
        bottom_ax.tick_params(top=False)
        top_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        bottom_ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        bottom_ax.margins(x=0.14)
        top_ax.margins(x=0.14)
        bottom_ax.set_ylabel("Mean codec total time (ms/image)", fontsize=13, color="#222222")

        d = 0.012
        kwargs = dict(transform=top_ax.transAxes, color="#555555", clip_on=False, linewidth=1.0)
        top_ax.plot((-d, +d), (-d, +d), **kwargs)
        top_ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=bottom_ax.transAxes)
        bottom_ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        bottom_ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    axes[-1].set_xlabel("Parameters (millions)", fontsize=13, color="#222222")
    axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=max(1, len(legend_labels)),
        frameon=False,
        fontsize=10,
    )
    fig.subplots_adjust(top=0.84, right=0.97)

    output_dir = Path(output_dir)
    stem = f"runtime_tradeoff_params{suffix}"
    fig.savefig(output_dir / f"{stem}.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / f'{stem}.pdf'}")


def bd_reference_variant(data):
    if "A_ft" in data:
        return "A_ft"
    if "A" in data:
        return "A"
    return None


def plot_rd_curves(data, output_dir, metric="psnr", variants=None, suffix=""):
    curve_data = select_curve_data(data, metric)
    if variants is not None:
        curve_data = subset_curve_data(curve_data, variants)

    if not curve_data:
        return {}, None

    presentation_zoom = suffix == "_project_zoom"
    fig, ax = plt.subplots(1, 1, figsize=(11.5, 7.6) if presentation_zoom else (10.5, 7.0))
    style_axes(ax, tick_labelsize=13 if presentation_zoom else 11)

    reference_variant = bd_reference_variant(curve_data)
    bd_rates = {}
    for variant in ordered_variants(curve_data):
        points = curve_data[variant]
        bpps = [plot_bpp(point) for point in points]
        vals = [point["avg_psnr"] for point in points] if metric == "psnr" else [point["avg_ms_ssim_db"] for point in points]
        ax.plot(
            bpps,
            vals,
            marker=VARIANT_MARKERS.get(variant, "o"),
            color=VARIANT_COLORS.get(variant, "#4C78A8"),
            label=VARIANT_LABELS.get(variant, variant),
            linewidth=2.4,
            linestyle=VARIANT_LINESTYLES.get(variant, "-"),
            markersize=8.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
        )

        if variant != reference_variant and reference_variant and len(points) >= 4 and len(curve_data[reference_variant]) >= 4:
            ref_bpps = [plot_bpp(point) for point in curve_data[reference_variant]]
            ref_vals = (
                [point["avg_psnr"] for point in curve_data[reference_variant]]
                if metric == "psnr"
                else [point["avg_ms_ssim_db"] for point in curve_data[reference_variant]]
            )
            bd_rates[variant] = compute_bd_rate(ref_bpps, ref_vals, bpps, vals)

    axis_fontsize = 16 if presentation_zoom else 13
    legend_fontsize = 13 if presentation_zoom else 10
    ax.set_xlabel("Bits per pixel (bpp)", fontsize=axis_fontsize, color="#222222")
    ax.set_ylabel("PSNR (dB)" if metric == "psnr" else "MS-SSIM (dB)", fontsize=axis_fontsize, color="#222222")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    if variants is None:
        ax.margins(x=0.08, y=0.08)
    else:
        xlim, ylim = axis_bounds(curve_data, metric)
        if xlim and ylim:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, frameon=False, fontsize=legend_fontsize)
    fig.subplots_adjust(top=0.80 if presentation_zoom else 0.82, right=0.92)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"rd_curve_{metric}{suffix}"
    fig.savefig(output_dir / f"{stem}.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / f'{stem}.pdf'}")
    return bd_rates, reference_variant


def plot_bd_rate_bars(bd_rates, output_dir, metric, reference_variant):
    if not bd_rates:
        return

    items = sorted(bd_rates.items(), key=lambda item: item[1])
    labels = [VARIANT_LABELS.get(variant, variant) for variant, _ in items]
    values = [value for _, value in items]
    colors = ["#0072B2" if value <= 0 else "#D55E00" for value in values]

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.5))
    style_axes(ax)
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.axvline(0.0, color="#444444", linewidth=1.2)
    ax.set_xlabel(f"BD-rate vs {reference_variant} (%)", fontsize=12, color="#222222")
    ax.set_title(f"BD-rate Summary on Kodak ({metric.upper()})", fontsize=15, color="#111111", pad=12)

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

    output_dir = Path(output_dir)
    fig.tight_layout()
    fig.savefig(output_dir / f"bd_rate_{metric}.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / f"bd_rate_{metric}.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / f"bd_rate_{metric}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / f'bd_rate_{metric}.pdf'}")


def write_bd_rate_summary(bd_psnr, bd_msssim, output_dir, reference_variant):
    summary_path = Path(output_dir) / "bd_rate_summary.csv"
    rows = []
    all_variants = ordered_variants({**{k: [] for k in bd_psnr}, **{k: [] for k in bd_msssim}})
    for variant in all_variants:
        if variant == reference_variant:
            continue
        rows.append({
            "variant": variant,
            "label": VARIANT_LABELS.get(variant, variant),
            "reference_variant": reference_variant,
            "bd_rate_psnr": bd_psnr.get(variant),
            "bd_rate_msssim": bd_msssim.get(variant),
        })

    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant", "label", "reference_variant", "bd_rate_psnr", "bd_rate_msssim"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {summary_path}")


def main():
    args = parse_args()
    data = load_summaries(args.results_dir)
    if not data:
        print(f"No summary files found in {args.results_dir}")
        return

    print(f"Loaded results for variants: {list(data.keys())}")
    for variant in data:
        print(f"  {variant}: {len(data[variant])} operating points")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary_table(data, output_dir)
    write_parameter_table(data, output_dir)
    write_runtime_tradeoff_tables(data, output_dir)
    plot_runtime_param_scatter(data, output_dir)
    plot_runtime_param_scatter(data, output_dir, variants=PROJECT_VARIANTS, suffix="_project_zoom")
    bd_psnr, reference_variant = plot_rd_curves(data, output_dir, metric="psnr")
    bd_msssim, _ = plot_rd_curves(data, output_dir, metric="msssim")
    plot_rd_curves(data, output_dir, metric="psnr", variants=PROJECT_VARIANTS, suffix="_project_zoom")
    plot_rd_curves(data, output_dir, metric="msssim", variants=PROJECT_VARIANTS, suffix="_project_zoom")
    plot_bd_rate_bars(bd_psnr, output_dir, metric="psnr", reference_variant=reference_variant)
    plot_bd_rate_bars(bd_msssim, output_dir, metric="msssim", reference_variant=reference_variant)
    write_bd_rate_summary(bd_psnr, bd_msssim, output_dir, reference_variant)

    print(f"\n--- BD-Rate Summary (vs. {reference_variant}) ---")
    for variant in ordered_variants(data):
        if variant == reference_variant:
            continue
        if variant in bd_psnr:
            print(f"  {VARIANT_LABELS.get(variant, variant)}:")
            print(f"    PSNR BD-rate:    {bd_psnr[variant]:+.2f}%")
        if variant in bd_msssim:
            print(f"    MS-SSIM BD-rate: {bd_msssim[variant]:+.2f}%")


if __name__ == "__main__":
    main()
