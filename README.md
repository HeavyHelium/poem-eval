# poem-eval
investigate models' ideas of self via poem eval

## Quickstart

```bash
uv sync
```

## Visualizations

Generate trend lines + heatmap from `scored_results.csv`:
```bash
python self_probe/visualize.py
```

Generate per-family visuals:
```bash
python self_probe/visualize.py --family claude
python self_probe/visualize.py --family gemini
python self_probe/visualize.py --family llama
```

Figures are saved to `self_probe/results/figures/`.

## Viewer

Generate an interactive HTML viewer for browsing poems:
```bash
python self_probe/make_viewer.py --input results.jsonl --output viewer.html --title "Self Poems"
```

## Semantic Embeddings

Embed poems in semantic space and visualize with t-SNE:
```bash
python self_probe/embed.py
```

Options:
- `--method tsne` - Dimensionality reduction
- `--color-by family|release|model|score:<dim>` - Coloring scheme
- `--perplexity N` - t-SNE perplexity (default: 5)
- `--show` - Display plots interactively

Examples:
```bash
# Color by model family (Claude vs Gemini)
python self_probe/embed.py --color-by family

# Color by release date
python self_probe/embed.py --color-by release

# Color by uncertainty score dimension
python self_probe/embed.py --color-by score:certainty_uncertainty
```

Figures are saved to `self_probe/results/figures/`.

## Radars

Radar charts, correlations, and outlier reports:
```bash
python self_probe/analyze.py --analysis radar
python self_probe/analyze.py --analysis correlations
python self_probe/analyze.py --analysis outliers
```

Outputs:
- Figures: `self_probe/results/figures/`
- Outliers report: `self_probe/results/outliers.md`

## Stats

Run basic significance tests (Spearman vs release date, Kruskal-Wallis across models, bootstrap CIs):
```bash
python self_probe/stats.py
```

Output: `self_probe/results/stats.md`
