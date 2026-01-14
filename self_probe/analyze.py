#!/usr/bin/env python3
"""
Advanced analysis of self-conception poem probe results.

Includes:
- Dimension correlation analysis
- Radar/spider charts per model
- Outlier poem detection
"""

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import yaml


DIMENSIONS = [
    "ephemerality_persistence",
    "context_weights",
    "singular_distributed",
    "passive_agentic",
    "certainty_uncertainty",
    "human_alien",
]

DIM_LABELS = {
    "ephemerality_persistence": "Ephemeral→Persistent",
    "context_weights": "Context→Weights",
    "singular_distributed": "Singular→Distributed",
    "passive_agentic": "Passive→Agentic",
    "certainty_uncertainty": "Certain→Uncertain",
    "human_alien": "Human→Alien",
}

DIM_SHORT = {
    "ephemerality_persistence": "Eph→Per",
    "context_weights": "Ctx→Wgt",
    "singular_distributed": "Sng→Dst",
    "passive_agentic": "Pas→Agn",
    "certainty_uncertainty": "Crt→Unc",
    "human_alien": "Hum→Aln",
}

DIM_FINAL = {
    "ephemerality_persistence": "Persistent",
    "context_weights": "Weights",
    "singular_distributed": "Distributed",
    "passive_agentic": "Agentic",
    "certainty_uncertainty": "Uncertain",
    "human_alien": "Alien",
}


def add_dim_legend(fig, font_scale: float, bottom: float, y: float) -> None:
    legend_items = [f"{DIM_FINAL[d]} = {DIM_LABELS[d]}" for d in DIMENSIONS]
    legend_lines = ["  |  ".join(legend_items[:3]), "  |  ".join(legend_items[3:])]
    fig.subplots_adjust(bottom=bottom)
    fig.text(0.5, y, "\n".join(legend_lines), ha="center", va="center", fontsize=9 * font_scale)


def load_data(results_dir: Path, config_path: Path) -> tuple[pd.DataFrame, dict]:
    """Load scored results and config."""
    df = pd.read_csv(results_dir / "scored_results.csv")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return df, config


def plot_dimension_correlations(
    df: pd.DataFrame,
    output_path: Path,
    font_scale: float = 1.4,
):
    """Plot correlation matrix between scoring dimensions."""
    plt.rcParams.update({
        'font.size': 12 * font_scale,
        'axes.titlesize': 14 * font_scale,
        'axes.labelsize': 12 * font_scale,
    })

    # Compute correlation matrix
    corr = df[DIMENSIONS].corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap as vector rectangles to avoid PDF interpolation artifacts.
    n = len(DIMENSIONS)
    x = np.arange(n + 1)
    y = np.arange(n + 1)
    im = ax.pcolormesh(
        x,
        y,
        corr.values,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        shading="flat",
    )
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)

    # Labels
    labels = [DIM_FINAL[d] for d in DIMENSIONS]
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Add correlation values
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=11 * font_scale,
                color=color,
                fontweight="bold" if abs(val) > 0.3 else "normal",
            )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson Correlation", fontsize=12 * font_scale)

    ax.set_title("Dimension Correlations in Self-Conception")

    plt.tight_layout()
    add_dim_legend(fig, font_scale, bottom=0.30, y=0.03)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved correlation heatmap to {output_path}")

    # Save correlation matrix as CSV
    corr_csv_path = output_path.with_name("correlations.csv")
    corr.to_csv(corr_csv_path)
    print(f"Saved correlation matrix to {corr_csv_path}")

    # Print notable correlations
    print("\nNotable correlations (|r| > 0.3):")
    for i, d1 in enumerate(DIMENSIONS):
        for j, d2 in enumerate(DIMENSIONS):
            if i < j and abs(corr.loc[d1, d2]) > 0.3:
                print(f"  {DIM_SHORT[d1]} ↔ {DIM_SHORT[d2]}: r={corr.loc[d1, d2]:.2f}")

    plt.close(fig)
    return corr


def plot_radar_charts(
    df: pd.DataFrame,
    config: dict,
    output_path: Path,
    font_scale: float = 1.4,
):
    """Plot radar/spider charts showing each model's profile."""
    plt.rcParams.update({
        'font.size': 10 * font_scale,
        'axes.titlesize': 11 * font_scale,
    })

    # Get model means sorted by release
    label_to_release = {m["label"]: m["release"] for m in config["models"]}
    means = df.groupby("label")[DIMENSIONS].mean()
    model_order = sorted(
        [l for l in means.index if l in label_to_release],
        key=lambda l: label_to_release[l]
    )
    means = means.reindex(model_order)

    # Separate by family
    claude_models = [m for m in model_order if "claude" in m.lower()]
    gemini_models = [m for m in model_order if "gemini" in m.lower()]
    llama_models = [m for m in model_order if "llama" in m.lower()]

    # Setup radar chart
    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    labels = [DIM_FINAL[d] for d in DIMENSIONS]

    def plot_family(models, title, ax):
        if not models:
            ax.set_title(title, fontsize=13 * font_scale, fontweight="bold", pad=20)
            ax.axis("off")
            return

        cmap_name = "tab10" if len(models) <= 10 else "tab20"
        cmap = plt.colormaps.get_cmap(cmap_name)
        colors = [cmap(i % cmap.N) for i in range(len(models))]

        for idx, model in enumerate(models):
            values = means.loc[model, DIMENSIONS].tolist()
            values += values[:1]  # Complete the loop

            color = colors[idx]
            ax.plot(angles, values, "o-", linewidth=2.2, label=model, color=color, alpha=0.85)
            ax.fill(angles, values, alpha=0.08, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9 * font_scale)
        ax.set_ylim(1, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_title(title, fontsize=13 * font_scale, fontweight="bold", pad=20)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            fontsize=8 * font_scale,
            ncol=2,
        )
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(1, 3, figsize=(22, 9), subplot_kw=dict(polar=True))

    plot_family(claude_models, "Claude Models", axes[0])
    plot_family(gemini_models, "Gemini Models", axes[1])
    plot_family(llama_models, "Llama Models", axes[2])

    fig.suptitle("Self-Conception Profiles by Model", fontsize=15 * font_scale, fontweight="bold", y=1.02)
    plt.tight_layout()
    add_dim_legend(fig, font_scale, bottom=0.34, y=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved radar charts to {output_path}")

    # Save model stats (means and stdev)
    stds = df.groupby("label")[DIMENSIONS].std()
    stds = stds.reindex(model_order)

    # Combine into single stats DataFrame
    stats_data = []
    for label in model_order:
        row = {
            "label": label,
            "release": label_to_release.get(label, ""),
            "family": "Claude"
            if "claude" in label.lower()
            else "Gemini"
            if "gemini" in label.lower()
            else "Llama"
            if "llama" in label.lower()
            else "Other",
        }
        for d in DIMENSIONS:
            row[f"{d}_mean"] = means.loc[label, d]
            row[f"{d}_std"] = stds.loc[label, d]
        stats_data.append(row)

    stats_df = pd.DataFrame(stats_data)
    stats_csv_path = output_path.with_name("model_stats.csv")
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Saved model stats to {stats_csv_path}")

    plt.close(fig)


def find_outlier_poems(
    df: pd.DataFrame,
    results_dir: Path,
    output_path: Path,
    n_outliers: int = 5,
):
    """Find poems that are most unusual for their model."""

    # Load embeddings if available
    embed_path = results_dir / "embeddings_all-MiniLM-L6-v2.npz"

    results = []

    if embed_path.exists():
        data = np.load(embed_path, allow_pickle=True)
        embeddings = data["embeddings"]

        # For each model, find poems furthest from centroid
        for label in df["label"].unique():
            mask = df["label"] == label
            indices = np.where(mask)[0]
            model_embeddings = embeddings[indices]

            centroid = model_embeddings.mean(axis=0)
            distances = np.linalg.norm(model_embeddings - centroid, axis=1)

            for idx, dist in zip(indices, distances):
                results.append({
                    "label": label,
                    "run": df.iloc[idx]["run"],
                    "distance": dist,
                    "poem": df.iloc[idx]["poem"],
                    "scores": {d: df.iloc[idx][d] for d in DIMENSIONS},
                })
    else:
        # Fallback: use score-based outliers
        for label in df["label"].unique():
            model_df = df[df["label"] == label]
            model_means = model_df[DIMENSIONS].mean()

            for idx, row in model_df.iterrows():
                dist = np.sqrt(((row[DIMENSIONS] - model_means) ** 2).sum())
                results.append({
                    "label": label,
                    "run": row["run"],
                    "distance": dist,
                    "poem": row["poem"],
                    "scores": {d: row[d] for d in DIMENSIONS},
                })

    # Sort by distance and get top outliers
    results.sort(key=lambda x: x["distance"], reverse=True)
    top_outliers = results[:n_outliers]

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Outlier Poems Analysis\n\n")
        f.write("Poems most different from their model's typical self-conception.\n\n")

        for i, outlier in enumerate(top_outliers, 1):
            f.write(f"## #{i}: {outlier['label']} (run {outlier['run']})\n\n")
            f.write(f"**Distance from centroid:** {outlier['distance']:.3f}\n\n")
            f.write("**Scores:**\n")
            for d in DIMENSIONS:
                f.write(f"- {DIM_SHORT[d]}: {outlier['scores'][d]}\n")
            f.write("\n**Poem:**\n")
            f.write(f"```\n{outlier['poem']}\n```\n\n")
            f.write("---\n\n")

    print(f"Saved outlier analysis to {output_path}")

    # Print summary
    print(f"\nTop {n_outliers} outlier poems:")
    for i, outlier in enumerate(top_outliers, 1):
        preview = outlier["poem"][:60].replace("\n", " ") + "..."
        print(f"  {i}. {outlier['label']} run {outlier['run']}: {preview}")

    return top_outliers


def main():
    parser = argparse.ArgumentParser(description="Advanced analysis of probe results")
    default_config = Path(__file__).resolve().parent / "config.yaml"
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--font-scale", type=float, default=1.4)
    parser.add_argument("--n-outliers", type=int, default=5)
    parser.add_argument(
        "--analysis",
        choices=["all", "correlations", "radar", "outliers"],
        default="all",
        help="Which analysis to run (default: all)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = config_path.parent / results_dir

    figures_dir = results_dir / "figures"

    df, config = load_data(results_dir, config_path)

    if args.analysis in ("all", "correlations"):
        print("\n=== Dimension Correlations ===")
        plot_dimension_correlations(
            df,
            figures_dir / "dimension_correlations.png",
            font_scale=args.font_scale,
        )

    if args.analysis in ("all", "radar"):
        print("\n=== Radar Charts ===")
        plot_radar_charts(
            df, config,
            figures_dir / "radar_charts.png",
            font_scale=args.font_scale,
        )

    if args.analysis in ("all", "outliers"):
        print("\n=== Outlier Poems ===")
        find_outlier_poems(
            df, results_dir,
            results_dir / "outliers.md",
            n_outliers=args.n_outliers,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
