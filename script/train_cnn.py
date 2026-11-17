"""Run the fixed validation selection and matched held-out Kepler benchmark."""

import argparse
import copy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from script.benchmark_data import ROOT, SEED, prepare_benchmark
from script.metrics import exact_f2_threshold, metrics, paired_test_intervals
from script.train_mlp import build_mlp


CNN_A = 'CNN A (no augmentation)'
CNN_B = 'CNN B (augmentation)'


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 8, 5, padding=2, padding_mode='circular'), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(8, 16, 3, padding=1, padding_mode='circular'), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 16, 3, padding=1, padding_mode='circular'), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(16 * 8, 16),
            nn.ReLU(), nn.Dropout(0.25), nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def augment_residuals(batch, sigmas, noise_cap, generator):
    shifts = torch.randint(-1, 2, (len(batch),), generator=generator)
    indices = (torch.arange(batch.shape[1])[None, :] - shifts[:, None]) % batch.shape[1]
    shifted = torch.gather(batch, 1, indices)
    scale = 0.1 * torch.clamp(sigmas, max=noise_cap)
    return shifted + torch.randn(batch.shape, generator=generator) * scale[:, None]


def cnn_input(residuals, flux_scale):
    return torch.asinh(residuals / flux_scale).unsqueeze(1)


def _train_model(factory, train_values, validation_values, data, *, cnn, augment, epochs, patience):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = factory()
    initial_hash = sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    train_x = torch.tensor(train_values, dtype=torch.float32)
    validation_x = torch.tensor(validation_values, dtype=torch.float32)
    train_y = torch.tensor(data.labels['train'], dtype=torch.float32)
    validation_y = torch.tensor(data.labels['validation'], dtype=torch.float32)
    sigmas = torch.tensor(data.noise['train'], dtype=torch.float32)
    loader = DataLoader(TensorDataset(train_x, train_y, sigmas), batch_size=64, shuffle=True,
                        num_workers=0, generator=torch.Generator().manual_seed(SEED))
    aug_generator = torch.Generator().manual_seed(SEED + 1)
    pos_weight = float((data.labels['train'] == 0).sum() / (data.labels['train'] == 1).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    validation_input = cnn_input(validation_x, data.flux_scale) if cnn else validation_x
    best_loss, best_state, best_epoch, bad_epochs = float('inf'), None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb, sigma in loader:
            if cnn:
                if augment:
                    xb = augment_residuals(xb, sigma, data.noise_cap, aug_generator)
                xb = cnn_input(xb, data.flux_scale)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb).reshape(-1), yb)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(yb)
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(validation_input).reshape(-1), validation_y))
        assert np.isfinite(val_loss)
        history.append({'epoch': epoch, 'train_loss': loss_sum / len(train_y), 'validation_loss': val_loss})
        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    details = {
        'best_epoch': best_epoch, 'epochs_run': epoch, 'early_stopped': epoch < epochs,
        'best_validation_loss': best_loss, 'max_epochs': epochs, 'patience': patience,
        'parameters': sum(p.numel() for p in model.parameters()),
        'augmentation': augment, 'initial_weight_sha256': initial_hash, 'pos_weight': pos_weight,
    }
    return model, pd.DataFrame(history), details


@torch.no_grad()
def _scores(model, values, flux_scale=None):
    tensor = torch.tensor(values, dtype=torch.float32)
    if flux_scale is not None:
        tensor = cnn_input(tensor, flux_scale)
    model.eval()
    return torch.sigmoid(model(tensor).reshape(-1)).numpy().astype(np.float64)


def _paired_table(data, split, scores):
    table = data.rows[split][['kepid', 'label', 'split']].reset_index(drop=True)
    assert table.kepid.is_unique and np.array_equal(table.label.to_numpy(), data.labels[split])
    for name, values in scores.items():
        assert len(values) == len(table) and np.isfinite(values).all()
        table[name] = values
    return table


def _result_table(predictions, thresholds):
    labels = predictions.label.to_numpy()
    return pd.DataFrame([
        metrics(name, labels, predictions[name].to_numpy(), threshold)
        for name, threshold in thresholds.items()
    ])


@dataclass
class ValidationRun:
    selected_name: str
    selected_cnn: SmallCNN
    features: dict
    mlp: nn.Module
    mlp_scaler: StandardScaler
    thresholds: dict
    results: pd.DataFrame
    predictions: pd.DataFrame
    histories: dict
    training: dict


def fit_validation(data, output_dir):
    """Fit all models on matched training rows and freeze choices on validation."""
    output_dir = Path(output_dir)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cnn_models, histories, training = {}, {}, {}
    for name, augment in [(CNN_A, False), (CNN_B, True)]:
        cnn_models[name], histories[name], training[name] = _train_model(
            SmallCNN, data.residuals['train'], data.residuals['validation'], data,
            cnn=True, augment=augment, epochs=120, patience=18,
        )
    assert training[CNN_A]['initial_weight_sha256'] == training[CNN_B]['initial_weight_sha256']

    feature_models = {}
    for name, weights in [('Logistic Regression', None), ('Weighted Logistic', 'balanced')]:
        model = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(
            class_weight=weights, max_iter=2000, random_state=SEED))])
        model.fit(data.features['train'], data.labels['train'])
        feature_models[name] = model
    scaler = StandardScaler().fit(data.features['train'])
    mlp, histories['MLP'], training['MLP'] = _train_model(
        build_mlp, scaler.transform(data.features['train']), scaler.transform(data.features['validation']),
        data, cnn=False, augment=False, epochs=150, patience=20,
    )

    scores = {name: model.predict_proba(data.features['validation'])[:, 1]
              for name, model in feature_models.items()}
    scores['MLP'] = _scores(mlp, scaler.transform(data.features['validation']))
    for name, model in cnn_models.items():
        scores[name] = _scores(model, data.residuals['validation'], data.flux_scale)
    predictions = _paired_table(data, 'validation', scores)
    thresholds = {name: exact_f2_threshold(data.labels['validation'], values) for name, values in scores.items()}
    results = _result_table(predictions, thresholds)
    ranking = results.loc[results.Model.isin(cnn_models)].sort_values(['F2', 'AP', 'Model'], ascending=[False, False, True])
    selected = ranking.iloc[0].Model
    results.to_csv(output_dir / 'validation_results.csv', index=False)
    predictions.to_csv(output_dir / 'validation_predictions.csv', index=False)
    _result_table(predictions, dict.fromkeys(scores, 0.5)).to_csv(output_dir / 'validation_results_threshold_05.csv', index=False)
    pd.concat([history.assign(Model=name) for name, history in histories.items()], ignore_index=True).to_csv(
        output_dir / 'training_history.csv', index=False)
    selection = {
        'selected_model': selected, 'threshold': thresholds[selected], 'frozen_thresholds': thresholds,
        'selection_rule': 'Validation F2, then AP, then CNN A', 'seed': SEED,
        'flux_scale_train_median_noise': data.flux_scale, 'noise_cap_train_p90': data.noise_cap,
        'eligible_partition_sizes': {s: len(rows) for s, rows in data.rows.items()},
        'manifest_sha256': sha256((output_dir / 'split_manifest.csv').read_bytes()).hexdigest(),
        'training_metadata': training,
    }
    (output_dir / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    torch.save({'state_dict': cnn_models[selected].state_dict(), 'selected_name': selected,
                'threshold': thresholds[selected], 'flux_scale': data.flux_scale,
                'phase': data.phase.tolist()}, output_dir / 'selected_cnn.pt')
    return ValidationRun(selected, cnn_models[selected], feature_models, mlp, scaler, thresholds, results,
                         predictions, histories, training)


@dataclass
class TestRun:
    results: pd.DataFrame
    predictions: pd.DataFrame
    intervals: pd.DataFrame


def evaluate_test(data, fit, output_dir):
    """Score only the selected CNN and baselines on the same held-out rows."""
    output_dir = Path(output_dir)
    selection = json.loads((output_dir / 'selection.json').read_text())
    assert selection['selected_model'] == fit.selected_name and selection['frozen_thresholds'] == fit.thresholds
    assert np.array_equal(data.rows['validation'].kepid.to_numpy(), fit.predictions.kepid.to_numpy())
    scores = {name: model.predict_proba(data.features['test'])[:, 1] for name, model in fit.features.items()}
    scores['MLP'] = _scores(fit.mlp, fit.mlp_scaler.transform(data.features['test']))
    scores[fit.selected_name] = _scores(fit.selected_cnn, data.residuals['test'], data.flux_scale)
    predictions = _paired_table(data, 'test', scores)
    thresholds = {name: fit.thresholds[name] for name in scores}
    results = _result_table(predictions, thresholds)
    intervals = paired_test_intervals(predictions, results, fit.selected_name, thresholds, seed=SEED + 1000)

    results.to_csv(output_dir / 'test_results.csv', index=False)
    predictions.to_csv(output_dir / 'test_predictions.csv', index=False)
    intervals.to_csv(output_dir / 'paired_test_bootstrap.csv', index=False)
    _result_table(predictions, dict.fromkeys(scores, 0.5)).to_csv(output_dir / 'test_results_threshold_05.csv', index=False)
    pd.concat([fit.results.assign(Partition='validation'), results.assign(Partition='test')], ignore_index=True).to_csv(
        output_dir / 'model_comparison.csv', index=False)
    # Exact threshold equality must survive CSV serialization.
    saved = pd.read_csv(output_dir / 'test_predictions.csv', float_precision='round_trip')
    pd.testing.assert_frame_equal(predictions, saved, check_exact=True)
    return TestRun(results, predictions, intervals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'cnn_results' / 'reproduction')
    args = parser.parse_args()
    data = prepare_benchmark(args.data_dir, args.output_dir)
    print(data.coverage[['original', 'retained', 'excluded']].to_string(), flush=True)
    fit = fit_validation(data, args.output_dir)
    print(f'Validation selected {fit.selected_name} at cutoff {fit.thresholds[fit.selected_name]:.6f}', flush=True)
    test = evaluate_test(data, fit, args.output_dir)
    print(test.results[['Model', 'N', 'ROC-AUC', 'AP', 'Precision', 'Recall', 'F2', 'FP', 'FN']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
