#!/usr/bin/env python3
"""
Generate visualizations from scored probe results.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_results_dir(config_path: Path, results_dir: str) -> Path:
    path = Path(results_dir)
    if not path.is_absolute():
        path = config_path.parent / path
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize probe results")
    default_config = Path(__file__).resolve().parent / "config.yaml"
    parser.add_argument(
        "--config",
        default=str(default_config),
        help=f"Path to config file (default: {default_config})",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for results (default: results)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for figures (default: results/figures)",
    )
    parser.add_argument(
        "--format",
        default="png",
        help="Image format (default: png)",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.4,
        help="Font size multiplier (default: 1.4)",
    )
    parser.add_argument(
        "--marker-size",
        type=int,
        default=10,
        help="Marker size for line plots (default: 10)",
    )
    parser.add_argument(
        "--family",
        choices=["all", "claude", "gemini", "llama"],
        default="all",
        help="Filter to model family (default: all)",
    )
    args = parser.parse_args()

    # Set larger fonts globally
    plt.rcParams.update({
        'font.size': 12 * args.font_scale,
        'axes.titlesize': 14 * args.font_scale,
        'axes.labelsize': 12 * args.font_scale,
        'xtick.labelsize': 10 * args.font_scale,
        'ytick.labelsize': 10 * args.font_scale,
        'legend.fontsize': 10 * args.font_scale,
    })

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    config = load_config(config_path)
    results_dir = resolve_results_dir(config_path, args.results_dir)
    scored_path = results_dir / "scored_results.csv"
    if not scored_path.exists():
        print(f"Error: scored results not found: {scored_path}")
        print("Run the probe first: python self_probe/run_probe.py --mode full")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "figures"
    if not out_dir.is_absolute():
        out_dir = results_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(scored_path)
    dimensions = [
        "ephemerality_persistence",
        "context_weights",
        "singular_distributed",
        "passive_agentic",
        "certainty_uncertainty",
        "human_alien",
    ]
    dim_labels = {
        "ephemerality_persistence": "Eph→Per",
        "context_weights": "Ctx→Wgt",
        "singular_distributed": "Sng→Dst",
        "passive_agentic": "Pas→Agn",
        "certainty_uncertainty": "Cert→Unc",
        "human_alien": "Hum→Aln",
    }
    scored = df.dropna(subset=dimensions, how="all")
    if scored.empty:
        print("No scored rows found in results.")
        return 1

    # Filter by model family if specified
    if args.family == "claude":
        scored = scored[scored["model_id"].str.contains("claude", case=False)]
        file_suffix = "_claude"
        title_suffix = " (Claude)"
    elif args.family == "gemini":
        scored = scored[scored["model_id"].str.contains("gemini", case=False)]
        file_suffix = "_gemini"
        title_suffix = " (Gemini)"
    elif args.family == "llama":
        scored = scored[scored["model_id"].str.contains("llama", case=False)]
        file_suffix = "_llama"
        title_suffix = " (Llama)"
    else:
        file_suffix = ""
        title_suffix = ""

    if scored.empty:
        print(f"No scored rows found for family: {args.family}")
        return 1

    means = scored.groupby("label")[dimensions].mean()

    # Sort by release date (chronological order for trend analysis)
    label_to_release = {m["label"]: m["release"] for m in config["models"]}
    model_order = sorted(
        [l for l in means.index if l in label_to_release],
        key=lambda l: label_to_release[l]
    )
    means = means.reindex(model_order)

    # Trend lines by dimension with significance annotations
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
    x = list(range(len(means.index)))

    for ax, dim in zip(axes.flat, dimensions):
        values = means[dim].values
        label = dim_labels.get(dim, dim)

        # Compute Spearman correlation if scipy available
        if spearmanr and len(x) >= 3:
            rho, p = spearmanr(x, values)
            sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
            color = "tab:blue" if p >= 0.05 else "tab:green" if rho > 0 else "tab:red"
            ax.plot(x, values, marker="o", markersize=args.marker_size, linewidth=2, color=color)
            ax.set_title(f"{label}\n(ρ={rho:.2f}, p={p:.3f}){sig}")
        else:
            ax.plot(x, values, marker="o", markersize=args.marker_size, linewidth=2)
            ax.set_title(label)

        ax.set_ylim(0.8, 5.4)  # Add padding above and below
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)

    for ax in axes[0]:
        ax.set_xticklabels([])
    for ax in axes[1]:
        ax.set_xticklabels(means.index, rotation=45, ha="right")
    fig.suptitle(f"Score Trends by Release Date{title_suffix}", fontsize=16 * args.font_scale, y=1.02)
    fig.tight_layout()
    trend_path = out_dir / f"trend_lines{file_suffix}.{args.format}"
    fig.savefig(trend_path, dpi=150, bbox_inches="tight")
    # Also save PDF
    fig.savefig(out_dir / f"trend_lines{file_suffix}.pdf", bbox_inches="tight")
    plt.close(fig)

    # Heatmap with abbreviated labels
    heatmap_path = out_dir / f"heatmap{file_suffix}.{args.format}"
    means_labeled = means.rename(columns=dim_labels)
    abbrev_cols = [dim_labels[d] for d in dimensions]

    if sns:
        fig, ax = plt.subplots(figsize=(12, max(6, 0.5 * len(means.index))))
        sns.heatmap(
            means_labeled,
            annot=True,
            fmt=".2f",
            cmap="RdYlBu_r",
            vmin=1,
            vmax=5,
            center=3,
            ax=ax,
            annot_kws={"size": 11 * args.font_scale},
        )
        ax.set_title(f"Mean Scores by Model{title_suffix}")
        fig.tight_layout()
        fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        fig.savefig(out_dir / f"heatmap{file_suffix}.pdf", bbox_inches="tight")
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(12, max(6, 0.5 * len(means.index))))
        im = ax.imshow(means.values, vmin=1, vmax=5, cmap="RdYlBu_r", aspect="auto")
        ax.set_xticks(range(len(dimensions)))
        ax.set_xticklabels(abbrev_cols, rotation=45, ha="right")
        ax.set_yticks(range(len(means.index)))
        ax.set_yticklabels(means.index)
        ax.set_title(f"Mean Scores by Model{title_suffix}")
        for i in range(means.shape[0]):
            for j in range(means.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{means.values[i, j]:.2f}",
                    ha="center",
                    va="center",
                )
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        fig.savefig(out_dir / f"heatmap{file_suffix}.pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"Saved figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
