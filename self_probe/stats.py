#!/usr/bin/env python3
"""
Run basic significance tests on scored probe results.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from scipy.stats import kruskal, spearmanr
except ImportError:
    kruskal = None
    spearmanr = None


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_results_dir(config_path: Path, results_dir: str) -> Path:
    path = Path(results_dir)
    if not path.is_absolute():
        path = config_path.parent / path
    return path


def release_to_ordinal(release: str) -> int | None:
    try:
        dt = datetime.strptime(release, "%Y-%m")
        return dt.toordinal()
    except ValueError:
        return None


def bootstrap_mean_ci(values: list[float], iters: int, alpha: float, rng) -> tuple[float, float, float] | None:
    if len(values) < 2:
        return None
    arr = np.asarray(values, dtype=float)
    means = np.empty(iters, dtype=float)
    for i in range(iters):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[i] = sample.mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(arr.mean()), lo, hi


def main() -> int:
    parser = argparse.ArgumentParser(description="Statistical tests for probe results")
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
        "--out",
        default=None,
        help="Output markdown path (default: results/stats.md)",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=2000,
        help="Bootstrap iterations for confidence intervals (default: 2000)",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap intervals (default: 0.95)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for bootstrap (default: 0)",
    )
    args = parser.parse_args()

    if not kruskal or not spearmanr:
        print("Error: scipy is required for stats. Install with: pip install scipy")
        return 1

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

    out_path = Path(args.out) if args.out else results_dir / "stats.md"
    if not out_path.is_absolute():
        out_path = results_dir / out_path

    df = pd.read_csv(scored_path)
    dimensions = [
        "ephemerality_persistence",
        "context_weights",
        "singular_distributed",
        "passive_agentic",
        "certainty_uncertainty",
        "human_alien",
    ]
    df = df.dropna(subset=dimensions, how="all")
    if df.empty:
        print("No scored rows found in results.")
        return 1

    label_to_release = {m["label"]: m["release"] for m in config["models"]}
    df["release"] = df["label"].map(label_to_release)
    df["release_ordinal"] = df["release"].apply(
        lambda r: release_to_ordinal(r) if isinstance(r, str) else None
    )
    df = df.dropna(subset=["release_ordinal"])
    rng = np.random.default_rng(args.seed)

    spearman_rows = []
    for dim in dimensions:
        subset = df[["release_ordinal", dim]].dropna()
        if subset["release_ordinal"].nunique() < 2 or len(subset) < 3:
            spearman_rows.append((dim, None, None, len(subset)))
            continue
        rho, p = spearmanr(subset["release_ordinal"], subset[dim])
        spearman_rows.append((dim, rho, p, len(subset)))

    kruskal_rows = []
    for dim in dimensions:
        groups = []
        labels = []
        for label in [m["label"] for m in config["models"]]:
            values = df.loc[df["label"] == label, dim].dropna().values
            if len(values) >= 2:
                groups.append(values)
                labels.append(label)
        if len(groups) < 2:
            kruskal_rows.append((dim, None, None, 0))
            continue
        h, p = kruskal(*groups)
        kruskal_rows.append((dim, h, p, len(groups)))

    ci_rows = []
    for label in [m["label"] for m in config["models"]]:
        row = {"label": label}
        for dim in dimensions:
            values = df.loc[df["label"] == label, dim].dropna().tolist()
            ci = bootstrap_mean_ci(values, args.bootstrap_iters, 1 - args.ci, rng)
            row[dim] = ci
        ci_rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Self-Conception Poem Probe — Stats\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## Spearman Correlation vs Release Date\n\n")
        f.write("Per-run scores correlated with model release (YYYY-MM).\n\n")
        f.write("| Dimension | rho | p-value | n |\n")
        f.write("|-----------|-----|---------|---|\n")
        for dim, rho, p, n in spearman_rows:
            if rho is None:
                f.write(f"| {dim} | n/a | n/a | {n} |\n")
            else:
                f.write(f"| {dim} | {rho:.3f} | {p:.4f} | {n} |\n")

        f.write("\n## Kruskal-Wallis Across Models\n\n")
        f.write("Nonparametric test for differences across model labels.\n\n")
        f.write("| Dimension | H | p-value | groups |\n")
        f.write("|-----------|---|---------|--------|\n")
        for dim, h, p, groups in kruskal_rows:
            if h is None:
                f.write(f"| {dim} | n/a | n/a | {groups} |\n")
            else:
                f.write(f"| {dim} | {h:.3f} | {p:.4f} | {groups} |\n")

        f.write("\n## Bootstrap Mean Confidence Intervals\n\n")
        f.write(
            f"Mean score per model with {args.ci:.2%} bootstrap CI "
            f"({args.bootstrap_iters} resamples, seed={args.seed}).\n\n"
        )
        f.write(
            "| Model | Eph→Per | Ctx→Wgt | Sng→Dst | Pas→Agn | Cert→Unc | Hum→Aln |\n"
        )
        f.write("|-------|---------|---------|---------|---------|----------|----------|\n")
        for row in ci_rows:
            label = row["label"]
            cells = []
            for dim in dimensions:
                ci = row[dim]
                if not ci:
                    cells.append("n/a")
                else:
                    mean, lo, hi = ci
                    cells.append(f"{mean:.2f} [{lo:.2f}, {hi:.2f}]")
            f.write(f"| {label} | " + " | ".join(cells) + " |\n")

        f.write("\nNotes:\n")
        f.write("- No multiple-comparisons correction is applied.\n")
        f.write("- Release date is treated as ordinal using YYYY-MM.\n")
        f.write("- Bootstrap CIs use resampling with replacement; small n widens intervals.\n")

    print(f"Saved stats to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
