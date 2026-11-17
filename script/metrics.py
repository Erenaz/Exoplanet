"""Metrics and validation-only threshold rules for the two notebooks."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, fbeta_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


def grid_f2_threshold(labels, scores):
    """Notebook 01's 0.005 grid, taking the first cutoff in a tie."""
    grid = np.linspace(0.05, 0.95, 181)
    values = [fbeta_score(labels, scores >= cutoff, beta=2, zero_division=0) for cutoff in grid]
    return float(grid[np.argmax(values)])


def exact_f2_threshold(labels, scores):
    """Notebook 02's attainable cutoffs, taking the highest cutoff in a tie."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    p, r = precision[:-1], recall[:-1]
    denominator = 4 * p + r
    values = np.divide(5 * p * r, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    best = np.flatnonzero(np.isclose(values, values.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[best])


def metrics(name, labels, scores, threshold):
    assert len(labels) == len(scores) and np.isfinite(scores).all()
    prediction = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        'Model': name, 'N': len(labels), 'Threshold': float(threshold),
        'ROC-AUC': roc_auc_score(labels, scores), 'AP': average_precision_score(labels, scores),
        'Accuracy': accuracy_score(labels, prediction),
        'Balanced Accuracy': balanced_accuracy_score(labels, prediction),
        'Precision': precision_score(labels, prediction, zero_division=0),
        'Recall': recall_score(labels, prediction, zero_division=0),
        'F1': f1_score(labels, prediction, zero_division=0),
        'F2': fbeta_score(labels, prediction, beta=2, zero_division=0),
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
    }


def _interval_scores(labels, scores, threshold):
    pred = scores >= threshold
    tp = np.sum(pred & (labels == 1))
    fp = np.sum(pred & (labels == 0))
    fn = np.sum(~pred & (labels == 1))
    denominator = 5 * tp + 4 * fn + fp
    f2 = 5 * tp / denominator if denominator else 0.0
    return np.array([roc_auc_score(labels, scores), average_precision_score(labels, scores), f2])


def paired_test_intervals(predictions, results, selected, thresholds, seed, replicates=1000):
    """Compare models on the same resampled stars at their frozen decisions."""
    rng = np.random.default_rng(seed)
    y = predictions['label'].to_numpy()
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    samples = [np.concatenate([rng.choice(negative, len(negative), replace=True),
                               rng.choice(positive, len(positive), replace=True)]) for _ in range(replicates)]
    score_arrays = {name: predictions[name].to_numpy() for name in thresholds if name in predictions}
    differences = []
    indexed = results.set_index('Model')
    selected_scores = np.stack([_interval_scores(y[ix], score_arrays[selected][ix], thresholds[selected]) for ix in samples])
    for name in indexed.index.drop(selected):
        baseline_scores = np.stack([_interval_scores(y[ix], score_arrays[name][ix], thresholds[name]) for ix in samples])
        low, high = np.quantile(selected_scores - baseline_scores, [0.025, 0.975], axis=0)
        for j, metric in enumerate(['ROC-AUC', 'AP', 'F2']):
            differences.append({
                'Comparison': f'{selected} minus {name}', 'Metric': metric,
                'Difference': indexed.loc[selected, metric] - indexed.loc[name, metric],
                'CI_low': low[j], 'CI_high': high[j], 'Bootstrap_replicates': replicates,
            })
    return pd.DataFrame(differences)
