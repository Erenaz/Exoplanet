"""Fit the seven-feature MLP from Notebook 01 and score its validation split."""

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from script.benchmark_data import FEATURES, ROOT, SEED, load_summary_split
from script.metrics import grid_f2_threshold, metrics


def build_mlp():
    return nn.Sequential(
        nn.Linear(7, 16), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1),
    )


def fit_mlp(X_train, y_train, X_val, y_val):
    """Use the original Notebook 01 RNG stream, weighting and stopping rule."""
    torch.manual_seed(SEED)
    scaler = StandardScaler()
    Xt = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    Xv = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    yt = torch.tensor(np.asarray(y_train), dtype=torch.float32)
    yv = torch.tensor(np.asarray(y_val), dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=64, shuffle=True)

    model = build_mlp()
    weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    history = {'train': [], 'val': []}
    best_loss, best_state, bad_epochs = float('inf'), None, 0
    for epoch in range(150):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(1), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv).squeeze(1), yv).item()
        history['train'].append(np.mean(batch_losses))
        history['val'].append(val_loss)
        if val_loss < best_loss - 1e-4:
            best_loss, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
        if bad_epochs >= 20:
            print(f'Early stopping at epoch {epoch + 1}')
            break
    assert best_state is not None, 'No finite MLP validation checkpoint.'
    model.load_state_dict(best_state)
    model.eval()
    return model, scaler, history, best_loss


@torch.no_grad()
def mlp_probabilities(model, scaler, features):
    tensor = torch.tensor(scaler.transform(features), dtype=torch.float32)
    model.eval()
    return torch.sigmoid(model(tensor).squeeze(1)).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'mlp_results')
    args = parser.parse_args()
    torch.set_num_threads(2)
    _, _, split = load_summary_split(args.data_dir / 'kepler_candidates.csv')
    train, val = split['train'], split['validation']
    model, scaler, history, best_loss = fit_mlp(train[FEATURES], train.label, val[FEATURES], val.label)
    scores = mlp_probabilities(model, scaler, val[FEATURES])
    cutoff = grid_f2_threshold(val.label.to_numpy(), scores)
    results = pd.DataFrame([metrics('MLP', val.label.to_numpy(), scores, cutoff)])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / 'validation_results.csv', index=False)
    pd.DataFrame({'epoch': np.arange(1, len(history['train']) + 1),
                  'train_loss': history['train'], 'validation_loss': history['val']}).to_csv(
                      args.output_dir / 'training_history.csv', index=False)
    val[['kepid', 'label']].assign(score=scores).to_csv(args.output_dir / 'validation_predictions.csv', index=False)
    print(f'Best validation loss {best_loss:.6f}')
    print(results[['Model', 'N', 'Threshold', 'ROC-AUC', 'AP', 'Precision', 'Recall', 'F2', 'FP', 'FN']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
