# Kepler exoplanet candidate classification

This course project asks whether a compact convolutional model using phase-folded flux curves improves screening over models using seven engineered Kepler features.

| File or directory | Contents |
|---|---|
| `01_exoplanet_eda.ipynb` | Data exploration, logistic regression, class weighting, threshold selection, and a feature-based MLP |
| `02_exoplanet_cnn.ipynb` | CNN experiments and a paired held-out comparison with the refitted MLP |
| `script/` | Reusable input audit, metrics, model training, and the original curve processing script |
| `execution/run.sh` | Entry point for either neural network experiment |
| `data/` | [Instructions for placing the local input files](data/README.md) |
| `figures/`, `mlp_results/`, `cnn_results/` | Local plots and experiment artifacts |

Use the existing course environment or install the direct dependencies in `requirements.txt` in a Python 3.10 environment. Open the notebooks with the project root as their working directory.

```bash
bash execution/run.sh mlp
bash execution/run.sh cnn
```

The launcher uses `python3` from the active environment or the interpreter supplied in `PYTHON`. Each command accepts `--data-dir` and `--output-dir`. The MLP command reproduces Notebook 01’s validation run in `mlp_results/`. The CNN command writes a fresh reproduction to `cnn_results/reproduction/`. Notebook 02 writes its run to `cnn_results/notebook_run/`. Rerunning a command replaces files in its output directory. Use a different output directory to retain separate runs.

Notebook 02 filters the original stellar split using the curve and label audit. It trains the feature models and CNNs on the same eligible training stars, selects thresholds on the same validation stars, and scores the final CNN alongside the feature baselines on the same held-out stars. This comparison uses a freshly fitted MLP with the architecture from Notebook 01 and the validation F2 rule used for every model in Notebook 02.

Git excludes the supplied datasets, exported figures, model outputs, and local `cnn_results/` reports. The notebooks include aggregate results for reading on GitHub. Run the two notebook kernels with the local inputs to regenerate their outputs.
