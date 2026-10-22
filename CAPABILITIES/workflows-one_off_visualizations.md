# src/workflows/one_off_visualizations

Scratch plots for talks/figures. Currently a hardcoded start-vs-final deepCRE bar chart and a single-TF "optimized track only" deepCIS binding plot for one gene's window-scan CSV, written to `one_off_visualizations/plots/`.

## vis.py  [one-off]
Bar chart of start vs final deepCRE prediction (hardcoded values) and `optimized_track()`, a reduced `analysis.motives.deepcis_visualize` plot showing only the optimized deepCIS track for one TF (reuses `extract_plot_data` and the private `_get_padding_edge_positions` from there).
