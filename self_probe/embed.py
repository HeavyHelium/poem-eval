#!/usr/bin/env python3
"""
Semantic Embedding Visualization for Self-Conception Poems

Embeds poems in semantic space and visualizes with t-SNE/UMAP,
colored by model family, release date, or score dimensions.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None


def load_poems(results_dir: Path) -> pd.DataFrame:
    """Load poems from raw_responses.jsonl."""
    raw_path = results_dir / "raw_responses.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"No results found at {raw_path}")

    records = []
    with open(raw_path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data.get("poem") and not data.get("error"):
                    records.append(data)

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} poems")
    return df


def get_embedding_cache_path(results_dir: Path, model_name: str) -> Path:
    """Get path for cached embeddings."""
    safe_name = model_name.replace("/", "_")
    return results_dir / f"embeddings_{safe_name}.npz"


def load_cached_embeddings(
    cache_path: Path,
    poem_hashes: list[str],
) -> np.ndarray | None:
    """Load embeddings from cache if valid."""
    if not cache_path.exists():
        return None

    try:
        data = np.load(cache_path, allow_pickle=True)
        cached_hashes = list(data["poem_hashes"])
        if cached_hashes == poem_hashes:
            print(f"Loaded cached embeddings from {cache_path}")
            return data["embeddings"]
        else:
            print("Cache invalidated (poems changed)")
            return None
    except Exception as e:
        print(f"Failed to load cache: {e}")
        return None


def save_embeddings(
    cache_path: Path,
    embeddings: np.ndarray,
    poem_hashes: list[str],
):
    """Save embeddings to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, embeddings=embeddings, poem_hashes=np.array(poem_hashes))
    print(f"Saved embeddings to {cache_path}")


def get_coords_cache_path(results_dir: Path, method: str, perplexity: float) -> Path:
    """Get path for cached reduced coordinates."""
    return results_dir / f"coords_{method}_perp{perplexity}.npz"


def load_cached_coords(
    cache_path: Path,
    embedding_hashes: list[str],
) -> np.ndarray | None:
    """Load coordinates from cache if valid."""
    if not cache_path.exists():
        return None

    try:
        data = np.load(cache_path, allow_pickle=True)
        cached_hashes = list(data["embedding_hashes"])
        if cached_hashes == embedding_hashes:
            print(f"Loaded cached coordinates from {cache_path}")
            return data["coords"]
        else:
            print("Coords cache invalidated (embeddings changed)")
            return None
    except Exception as e:
        print(f"Failed to load coords cache: {e}")
        return None


def save_coords(
    cache_path: Path,
    coords: np.ndarray,
    embedding_hashes: list[str],
):
    """Save coordinates to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, coords=coords, embedding_hashes=np.array(embedding_hashes))
    print(f"Saved coordinates to {cache_path}")


def compute_embeddings(
    poems: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Compute semantic embeddings for poems."""
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers required: pip install sentence-transformers")

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Computing embeddings for {len(poems)} poems...")
    embeddings = model.encode(poems, show_progress_bar=True)

    return np.array(embeddings)


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "tsne",
    perplexity: float = 5.0,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce embeddings to 2D using t-SNE."""
    n_samples = embeddings.shape[0]

    if method == "tsne":
        if TSNE is None:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        # Perplexity must be less than n_samples
        perp = min(perplexity, n_samples - 1)
        print(f"Running t-SNE (perplexity={perp})...")
        reducer = TSNE(
            n_components=2,
            perplexity=perp,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
        return reducer.fit_transform(embeddings)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne'.")


def get_model_family(model_id: str) -> str:
    """Extract model family from model_id."""
    if "claude" in model_id.lower():
        return "Claude"
    elif "gemini" in model_id.lower():
        return "Gemini"
    elif "llama" in model_id.lower():
        return "Llama"
    elif "gpt" in model_id.lower():
        return "GPT"
    else:
        return "Other"


def release_to_numeric(release: str) -> float:
    """Convert release date string (YYYY-MM) to numeric for coloring."""
    try:
        year, month = release.split("-")
        return int(year) + int(month) / 12
    except (ValueError, AttributeError):
        return 0.0


def plot_embeddings(
    coords: np.ndarray,
    df: pd.DataFrame,
    color_by: str = "family",
    method: str = "tsne",
    output_path: Path | None = None,
    dot_size: int = 150,
    font_scale: float = 1.4,
):
    """Plot 2D embeddings with various coloring schemes."""
    # Set larger fonts
    plt.rcParams.update({
        'font.size': 12 * font_scale,
        'axes.titlesize': 14 * font_scale,
        'axes.labelsize': 12 * font_scale,
        'xtick.labelsize': 10 * font_scale,
        'ytick.labelsize': 10 * font_scale,
        'legend.fontsize': 10 * font_scale,
        'legend.title_fontsize': 11 * font_scale,
    })

    fig, ax = plt.subplots(figsize=(12, 10))

    families = df["model_id"].apply(get_model_family)
    is_haiku = df["label"].str.contains("haiku", case=False, na=False).to_numpy()
    is_opus = df["label"].str.contains("opus", case=False, na=False).to_numpy()
    family_markers = {
        "Claude": "s",
        "Gemini": "o",
        "Llama": "^",
        "GPT": "o",
        "Other": "o",
    }
    family_colors = {
        "Claude": "#6B5B95",
        "Gemini": "#88B04B",
        "Llama": "#F28E2B",
        "GPT": "#F7CAC9",
        "Other": "#92A8D1",
    }
    stroke_color = "#111111"

    class HandlerOverlay(HandlerBase):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            center_x = xdescent + width / 2
            center_y = ydescent + height / 2
            artists = []
            for handle in orig_handle:
                if not isinstance(handle, Line2D):
                    continue
                artists.append(
                    Line2D(
                        [center_x],
                        [center_y],
                        marker=handle.get_marker(),
                        markersize=handle.get_markersize(),
                        markerfacecolor=handle.get_markerfacecolor(),
                        markeredgecolor=handle.get_markeredgecolor(),
                        markeredgewidth=handle.get_markeredgewidth(),
                        linestyle="None",
                        color=handle.get_color(),
                        alpha=handle.get_alpha(),
                        transform=trans,
                    )
                )
            return artists

    def scatter_styled(mask, marker, facecolors, edgecolors, linewidths, label=None):
        if not mask.any():
            return None
        x = coords[mask, 0]
        y = coords[mask, 1]
        return ax.scatter(
            x,
            y,
            marker=marker,
            s=dot_size,
            facecolors=facecolors,
            edgecolors=edgecolors,
            alpha=0.7,
            linewidths=linewidths,
            label=label,
        )

    def overlay_strokes(mask, color):
        if not mask.any():
            return None
        x = coords[mask, 0]
        y = coords[mask, 1]
        return ax.scatter(
            x,
            y,
            marker="x",
            c=color,
            s=dot_size * 0.35,
            alpha=0.7,
            linewidths=0.9,
        )

    if color_by == "family":
        for family, color in family_colors.items():
            mask = (families == family).to_numpy()
            if not mask.any():
                continue
            marker = family_markers.get(family, "o")
            scatter_styled(mask, marker, color, "white", 0.5, label=family)

        ax.legend(
            title="Model Family",
            loc="best",
            fontsize=14 * font_scale,
            title_fontsize=15 * font_scale,
            markerscale=1.5,
        )
        title = f"Poem Embeddings by Model Family ({method.upper()})"

    elif color_by == "release":
        releases = df["release"].apply(release_to_numeric)
        cmap = plt.colormaps.get_cmap("viridis")
        norm = Normalize(vmin=float(releases.min()), vmax=float(releases.max()))

        for family in families.unique():
            mask = (families == family).to_numpy()
            if not mask.any():
                continue
            marker = family_markers.get(family, "o")
            colors = cmap(norm(releases[mask]))
            scatter_styled(mask, marker, colors, "white", 0.5)

        scatter = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scatter.set_array([])
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Release Date (year)")
        title = f"Poem Embeddings by Release Date ({method.upper()})"

    elif color_by == "model":
        labels = df["label"].unique()
        label_families = {}
        label_releases = {}
        for label in labels:
            family = get_model_family(df.loc[df["label"] == label, "model_id"].iloc[0])
            label_families[label] = family
            label_releases[label] = release_to_numeric(
                df.loc[df["label"] == label, "release"].iloc[0]
            )

        label_to_color = {}
        fixed_label_colors = {
            "Gemini-2.0-Flash": "#1a7a3d",
            "Gemini-2.5-Flash": "#2eb867",
            "Gemini-2.5-Pro": "#1a9e5c",
            "Gemini-3-Flash": "#3dcca3",
            "Gemini-3-Pro": "#20b8a8",
            "Llama-3-8B-Instruct": "#cc3d1a",
            "Llama-3-70B-Instruct": "#b32d00",
            "Llama-3.1-8B-Instruct": "#e86428",
            "Llama-3.1-70B-Instruct": "#d94e00",
            "Llama-3.1-405B-Instruct": "#c44000",
            "Llama-3.3-70B-Instruct": "#f59833",
            "Llama-4-Scout": "#e6b830",
            "Llama-4-Maverick": "#cca300",
            "Claude-3-Haiku": "#7c5ce6",
            "Claude-3.5-Haiku": "#9966ff",
            "Claude-3.5-Sonnet": "#a347d9",
            "Claude-3.7-Sonnet": "#b84dcc",
            "Claude-4-Sonnet": "#cc59bf",
            "Claude-4.5-Sonnet": "#d966a8",
            "Claude-Haiku-4.5": "#b088e0",
            "Claude-Opus-4": "#8033cc",
            "Claude-Opus-4.1": "#9933b3",
            "Claude-Opus-4.5": "#cc3399",
        }
        for label, color in fixed_label_colors.items():
            if label in label_families:
                label_to_color[label] = color
        family_palettes = {
            "Claude": ["#312E81", "#1D4ED8", "#0284C7", "#06B6D4"],
            "Gemini": ["#064E3B", "#047857", "#10B981", "#84CC16"],
            "Llama": ["#7F1D1D", "#DC2626", "#F97316", "#F59E0B"],
            "GPT": ["#4C1D95", "#6D28D9", "#8B5CF6", "#C084FC"],
            "Other": ["#374151", "#6B7280", "#9CA3AF", "#D1D5DB"],
        }
        for family in sorted(set(label_families.values())):
            family_labels = [l for l in labels if label_families[l] == family]
            releases = [label_releases[l] for l in family_labels if label_releases[l] > 0]
            unique_releases = sorted(set(releases))
            release_rank = {r: i for i, r in enumerate(unique_releases)}
            denom = max(1, len(unique_releases) - 1)
            palette = family_palettes.get(family, family_palettes["Other"])
            palette_len = len(palette)

            for label in family_labels:
                if label in label_to_color:
                    continue
                release_val = label_releases[label]
                if release_val > 0 and len(unique_releases) > 1:
                    norm_val = release_rank[release_val] / denom
                    idx = int(round(norm_val * (palette_len - 1)))
                else:
                    idx = (palette_len - 1) // 2
                label_to_color[label] = palette[idx]

        for label in labels:
            mask = df["label"] == label
            family = get_model_family(df.loc[mask, "model_id"].iloc[0])
            marker = family_markers.get(family, "o")
            color = label_to_color[label]
            mask_array = mask.to_numpy()
            if is_haiku[mask_array].any():
                scatter_styled(mask_array, marker, color, "black", 1.5, label=label)
            else:
                scatter_styled(mask_array, marker, color, "white", 0.5, label=label)
                if is_opus[mask_array].any():
                    overlay_strokes(mask_array, stroke_color)

        legend_handles = []
        legend_labels = []
        for label in labels:
            lower = label.lower()
            family = get_model_family(df.loc[df["label"] == label, "model_id"].iloc[0])
            marker = family_markers.get(family, "o")
            color = label_to_color[label]
            if "haiku" in lower:
                handle = Line2D(
                    [0],
                    [0],
                    marker=marker,
                    linestyle="None",
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    markersize=9,
                )
            elif "opus" in lower:
                base = Line2D(
                    [0],
                    [0],
                    marker=marker,
                    linestyle="None",
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    markersize=9,
                )
                stroke = Line2D(
                    [0],
                    [0],
                    marker="x",
                    linestyle="None",
                    color=stroke_color,
                    markersize=7,
                    markeredgewidth=1.0,
                )
                handle = (base, stroke)
            else:
                handle = Line2D(
                    [0],
                    [0],
                    marker=marker,
                    linestyle="None",
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    markersize=9,
                )
            legend_handles.append(handle)
            legend_labels.append(label)

        ax.legend(
            legend_handles,
            legend_labels,
            title="Model",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=11 * font_scale,
            title_fontsize=12 * font_scale,
            markerscale=1.3,
            ncol=1,
            handler_map={tuple: HandlerOverlay()},
        )
        title = (
            f"Poem Embeddings by Model ({method.upper()})"
            " — family palettes encode release tier (older->newer)"
        )

    elif color_by.startswith("score:"):
        # Color by a specific score dimension
        dim = color_by.split(":", 1)[1]
        if dim not in df.columns:
            print(f"Warning: Score dimension '{dim}' not found. Using family coloring.")
            return plot_embeddings(coords, df, "family", method, output_path)

        scores = df[dim].fillna(3)  # neutral for missing
        cmap = plt.colormaps.get_cmap("RdYlBu_r")
        norm = Normalize(vmin=1, vmax=5)

        for family in families.unique():
            mask = (families == family).to_numpy()
            if not mask.any():
                continue
            marker = family_markers.get(family, "o")
            colors = cmap(norm(scores[mask]))
            haiku_mask = mask & is_haiku
            opus_mask = mask & is_opus
            other_mask = mask & ~(is_haiku | is_opus)
            haiku_sub = is_haiku[mask]
            opus_sub = is_opus[mask]
            other_sub = ~(haiku_sub | opus_sub)
            if other_mask.any():
                scatter_styled(other_mask, marker, colors[other_sub], "white", 0.5)
            if opus_mask.any():
                scatter_styled(opus_mask, marker, colors[opus_sub], "white", 0.5)
                overlay_strokes(opus_mask, stroke_color)
            if haiku_mask.any():
                scatter_styled(haiku_mask, marker, colors[haiku_sub], "black", 1.5)

        scatter = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scatter.set_array([])
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(dim.replace("_", " ").title())
        title = f"Poem Embeddings by {dim.replace('_', ' ').title()} ({method.upper()})"

    else:
        raise ValueError(f"Unknown color_by: {color_by}")

    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
        # Also save PDF
        pdf_path = output_path.with_suffix(".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved plot to {pdf_path}")

    return fig, ax


def compute_centroids(
    embeddings: np.ndarray,
    df: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Compute centroid embedding for each model label."""
    centroids = {}
    releases = {}
    for label in df["label"].unique():
        mask = df["label"] == label
        centroids[label] = embeddings[mask].mean(axis=0)
        releases[label] = df.loc[mask, "release"].iloc[0]
    return centroids, releases


def format_perplexity(perplexity: float) -> str:
    """Format perplexity for filenames (e.g., 10.5 -> 10p5)."""
    text = f"{perplexity}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def plot_trajectory(
    coords: np.ndarray,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    method: str = "tsne",
    output_path: Path | None = None,
    font_scale: float = 1.4,
):
    """Plot model centroids connected by release date, showing temporal trajectory."""
    plt.rcParams.update({
        'font.size': 12 * font_scale,
        'axes.titlesize': 14 * font_scale,
        'axes.labelsize': 12 * font_scale,
        'xtick.labelsize': 10 * font_scale,
        'ytick.labelsize': 10 * font_scale,
        'legend.fontsize': 10 * font_scale,
        'legend.title_fontsize': 11 * font_scale,
    })

    fig, ax = plt.subplots(figsize=(14, 10))

    # Compute centroids in original embedding space, then project
    labels = df["label"].unique()
    label_to_release = {label: df.loc[df["label"] == label, "release"].iloc[0] for label in labels}

    # Compute centroid coordinates in the reduced space
    centroid_coords = {}
    for label in labels:
        mask = df["label"] == label
        centroid_coords[label] = coords[mask].mean(axis=0)

    # Separate by family
    families = {"Claude": [], "Gemini": [], "Llama": [], "GPT": [], "Other": []}
    for label in labels:
        families.get(get_model_family(label), families["Other"]).append(label)

    family_colors = {
        "Claude": "#6B5B95",
        "Gemini": "#88B04B",
        "Llama": "#F28E2B",
        "GPT": "#F7CAC9",
        "Other": "#92A8D1",
    }

    # Plot each family's trajectory
    for family, family_labels in families.items():
        if not family_labels:
            continue

        # Sort by release date
        sorted_labels = sorted(
            family_labels,
            key=lambda l: (release_to_numeric(label_to_release.get(l, "")), l),
        )

        xs = [centroid_coords[l][0] for l in sorted_labels]
        ys = [centroid_coords[l][1] for l in sorted_labels]

        # Draw trajectory line with arrows
        color = family_colors[family]
        ax.plot(xs, ys, '-', color=color, linewidth=2, alpha=0.6, zorder=1)

        # Draw arrows between consecutive points
        for i in range(len(xs) - 1):
            ax.annotate(
                '',
                xy=(xs[i+1], ys[i+1]),
                xytext=(xs[i], ys[i]),
                arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.7),
                zorder=2,
            )

        # Plot centroids as larger points with labels
        ax.scatter(xs, ys, c=color, s=200, alpha=0.9, edgecolors="white", linewidth=2, zorder=3, label=family)

        # Add model labels
        for label, x, y in zip(sorted_labels, xs, ys):
            # Shorten label for readability
            short_label = (
                label.replace("Claude-", "C")
                .replace("Gemini-", "G")
                .replace("Llama-", "L")
                .replace("-Preview", "")
            )
            ax.annotate(
                short_label,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9 * font_scale,
                alpha=0.8,
            )

    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(f"Self-Conception Trajectory by Release Date ({method.upper()})")
    ax.legend(title="Model Family", fontsize=12 * font_scale, title_fontsize=13 * font_scale, markerscale=0.8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
        pdf_path = output_path.with_suffix(".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved plot to {pdf_path}")

    return fig, ax


def plot_similarity_heatmap(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    output_path: Path | None = None,
    font_scale: float = 1.4,
):
    """Plot cosine similarity heatmap between model centroids."""
    plt.rcParams.update({
        'font.size': 12 * font_scale,
        'axes.titlesize': 14 * font_scale,
        'axes.labelsize': 12 * font_scale,
        'xtick.labelsize': 10 * font_scale,
        'ytick.labelsize': 10 * font_scale,
    })

    # Compute centroids
    family_order = {"Claude": 0, "Gemini": 1, "Llama": 2, "GPT": 3, "Other": 4}
    def label_sort_key(label: str):
        family = get_model_family(label)
        release = df.loc[df["label"] == label, "release"].iloc[0]
        return (family_order.get(family, 99), release or "")

    labels = sorted(df["label"].unique(), key=label_sort_key)
    centroids = []
    for label in labels:
        mask = df["label"] == label
        centroids.append(embeddings[mask].mean(axis=0))
    centroids = np.array(centroids)

    # Compute cosine similarity
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    normalized = centroids / norms
    similarity = normalized @ normalized.T

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    vmin = max(0.6, float(similarity.min()))
    vmax = 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(similarity, cmap="viridis", norm=norm, aspect="auto")

    # Add labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    short_labels = [
        l.replace("Claude-", "C-").replace("Gemini-", "G-").replace("Llama-", "L-")
        for l in labels
    ]
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_yticklabels(short_labels)

    family_colors = {
        "Claude": "#6B5B95",
        "Gemini": "#88B04B",
        "Llama": "#F28E2B",
        "GPT": "#F7CAC9",
        "Other": "#92A8D1",
    }
    for tick, label in zip(ax.get_xticklabels(), labels):
        tick.set_color(family_colors.get(get_model_family(label), "#333333"))
    for tick, label in zip(ax.get_yticklabels(), labels):
        tick.set_color(family_colors.get(get_model_family(label), "#333333"))

    # Add values
    for i in range(len(labels)):
        for j in range(len(labels)):
            rgba = im.cmap(im.norm(similarity[i, j]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "black" if luminance > 0.5 else "white"
            ax.text(j, i, f"{similarity[i, j]:.2f}", ha="center", va="center",
                   fontsize=9 * font_scale * 0.7, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine Similarity", fontsize=12 * font_scale)

    ax.set_title("Semantic Similarity Between Model Self-Conceptions")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
        pdf_path = output_path.with_suffix(".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved plot to {pdf_path}")

    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Embed poems in semantic space and visualize"
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory with raw_responses.jsonl (default: results)",
    )
    parser.add_argument(
        "--method",
        choices=["tsne"],
        default="tsne",
        help="Dimensionality reduction method (default: tsne)",
    )
    parser.add_argument(
        "--color-by",
        default="family",
        help="Coloring: family, release, model, or score:<dimension> (default: family)",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=5.0,
        help="t-SNE perplexity (default: 5.0)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force recompute embeddings (ignore cache)",
    )
    parser.add_argument(
        "--dot-size",
        type=int,
        default=150,
        help="Size of scatter plot dots (default: 150)",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.4,
        help="Font size multiplier (default: 1.4)",
    )
    parser.add_argument(
        "--trajectory",
        action="store_true",
        help="Generate trajectory plot (centroids connected by release date)",
    )
    parser.add_argument(
        "--similarity",
        action="store_true",
        help="Generate similarity heatmap (cosine similarity between model centroids)",
    )
    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).resolve().parent
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = script_dir / results_dir

    figures_dir = results_dir / "figures"

    # Load data
    df = load_poems(results_dir)

    # Compute or load cached embeddings
    poems = df["poem"].tolist()
    poem_hashes = [hashlib.md5(p.encode()).hexdigest() for p in poems]
    cache_path = get_embedding_cache_path(results_dir, args.embedding_model)

    embeddings = None
    if not args.no_cache:
        embeddings = load_cached_embeddings(cache_path, poem_hashes)
    if embeddings is None:
        embeddings = compute_embeddings(poems, args.embedding_model)
        save_embeddings(cache_path, embeddings, poem_hashes)

    # Create embedding hash for coordinate caching
    embedding_hash = hashlib.md5(embeddings.tobytes()).hexdigest()
    embedding_hashes = [embedding_hash]  # Single hash for whole array

    # Run dimensionality reduction and plot
    methods = [args.method]

    for method in methods:
        try:
            # Try to load cached coordinates
            coords_cache_path = get_coords_cache_path(
                results_dir, method, args.perplexity
            )
            coords = None
            if not args.no_cache:
                coords = load_cached_coords(coords_cache_path, embedding_hashes)

            if coords is None:
                coords = reduce_dimensions(
                    embeddings,
                    method=method,
                    perplexity=args.perplexity,
                )
                save_coords(coords_cache_path, coords, embedding_hashes)

            suffix = f"_perp{format_perplexity(args.perplexity)}" if method == "tsne" else ""
            color_bys = [args.color_by]
            for extra in ("release", "model"):
                if extra not in color_bys:
                    color_bys.append(extra)

            for color_by in color_bys:
                output_path = figures_dir / f"embed_{method}{suffix}_{color_by.replace(':', '_')}.png"
                fig, _ = plot_embeddings(
                    coords, df, color_by, method, output_path,
                    dot_size=args.dot_size,
                    font_scale=args.font_scale,
                )
                if not args.show:
                    plt.close(fig)

        except ImportError as e:
            print(f"Skipping {method}: {e}")

        # Generate trajectory plot if requested
        if args.trajectory:
            try:
                suffix = f"_perp{format_perplexity(args.perplexity)}" if method == "tsne" else ""
                trajectory_path = figures_dir / f"trajectory_{method}{suffix}.png"
                plot_trajectory(
                    coords, df, embeddings, method, trajectory_path,
                    font_scale=args.font_scale,
                )
            except Exception as e:
                print(f"Error generating trajectory plot: {e}")

    # Generate similarity heatmap if requested (doesn't need dim reduction)
    if args.similarity:
        try:
            similarity_path = figures_dir / "similarity_heatmap.png"
            plot_similarity_heatmap(
                embeddings, df, similarity_path,
                font_scale=args.font_scale,
            )
        except Exception as e:
            print(f"Error generating similarity heatmap: {e}")

    if args.show:
        plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
