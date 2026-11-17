"""Preserve the original stellar split and align its eligible light curves."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
FEATURES = ['period', 'depth', 'snr', 'evenodd_ratio', 'folded_std', 'folded_min', 'folded_ptp']
SPLITS = ('train', 'validation', 'test')


def load_summary_split(path):
    """Apply Notebook 01's complete-case filter and split in original row order."""
    raw = pd.read_csv(path)
    summary = raw.dropna().copy().drop_duplicates(subset='kepid', keep='first')
    assert set(summary.label.unique()) == {0, 1}
    assert np.isfinite(summary[FEATURES].to_numpy()).all()
    dev, test = train_test_split(summary, test_size=0.20, stratify=summary.label, random_state=SEED)
    train, validation = train_test_split(dev, test_size=0.20, stratify=dev.label, random_state=SEED)
    return raw, summary, dict(zip(SPLITS, (train, validation, test)))


@dataclass
class BenchmarkData:
    manifest: pd.DataFrame
    coverage: pd.DataFrame
    rows: dict
    phase: np.ndarray
    curves: dict
    residuals: dict
    noise: dict
    features: dict
    labels: dict
    flux_scale: float
    noise_cap: float


def audit_catalog(raw, catalog, output_dir):
    disposition = catalog.koi_disposition.fillna('UNKNOWN').str.strip().str.upper()
    groups = disposition.groupby(catalog.kepid)
    dispositions = groups.agg(lambda values: tuple(sorted(set(values))))
    first = raw.kepid.map(groups.first())
    expected = raw.label.map({0: 'FALSE POSITIVE', 1: 'CONFIRMED'})
    audit = raw[['kepid', 'label']].assign(
        catalog_rows=raw.kepid.map(groups.size()).fillna(0).astype(int),
        dispositions=raw.kepid.map(dispositions).map(
            lambda values: '|'.join(values) if isinstance(values, tuple) else 'MISSING'),
        first_disposition=first.fillna('MISSING'),
        agrees_with_first=first.eq(expected),
        mixed_disposition=raw.kepid.map(dispositions.map(len)).gt(1),
    )
    audit['unambiguous_contradiction'] = (
        audit.dispositions.isin(['CONFIRMED', 'FALSE POSITIVE']) & ~audit.agrees_with_first
    )
    audit.to_csv(output_dir / 'label_audit.csv', index=False)
    if audit.unambiguous_contradiction.any():
        raise ValueError('A summary label contradicts an unambiguous catalog label. See label_audit.csv.')
    return dispositions


def audit_curves(data_dir, summary_by_id, output_dir):
    curve_paths = sorted((data_dir / 'candidates').glob('candidatev2_*.npz'))
    if not curve_paths:
        raise FileNotFoundError('Expected extracted data/candidates/candidatev2_*.npz files.')
    checked = ['period', 'depth', 'folded_std', 'folded_min', 'folded_ptp']
    required = {'kepid', 'phase', 'flux', 'period', 't0', 'duration', 'depth'}
    curves, rows, hashes = {}, [], {}
    phase = None
    for path in curve_paths:
        name = f'data/{path.relative_to(data_dir).as_posix()}'
        hashes[name] = sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as archive:
            assert required.issubset(archive.files), f'Missing NPZ keys in {name}'
            kid = int(archive['kepid'])
            assert kid == int(path.stem.split('_')[-1]) and kid not in curves, f'Invalid identifier in {name}'
            grid = archive['phase'].astype(np.float64)
            flux = archive['flux'].astype(np.float64)
            assert grid.shape == flux.shape == (201,), f'Unexpected curve shape in {name}'
            assert np.isfinite(grid).all() and np.all(np.diff(grid) > 0), f'Invalid phase grid in {name}'
            if phase is None:
                phase = grid.copy()
            assert np.allclose(grid, phase, atol=1e-7, rtol=0), f'Incompatible phase grid in {name}'
            curves[kid] = flux
            finite = np.isfinite(flux).all()
            row = {
                'kepid': kid, 'path': name, 'keys': '|'.join(sorted(archive.files)),
                'length': len(flux), 'same_grid': True,
                'nonfinite_flux': int((~np.isfinite(flux)).sum()),
                'flux_sha256': sha256(flux.tobytes()).hexdigest(),
                'period': float(archive['period']), 'depth': float(archive['depth']),
                'folded_std': float(flux.std()) if finite else np.nan,
                'folded_min': float(flux.min()) if finite else np.nan,
                'folded_ptp': float(np.ptp(flux)) if finite else np.nan,
                'median_flux': float(np.median(flux)) if finite else np.nan,
            }
        if kid not in summary_by_id.index:
            discrepancy = 'no_summary_record'
        elif not finite:
            discrepancy = 'nonfinite_flux'
        else:
            summary = summary_by_id.loc[kid]
            discrepancy = '|'.join(c for c in checked if not np.isfinite(summary[c])
                                   or not np.isclose(row[c], summary[c], rtol=0.01, atol=1e-7))
        rows.append(dict(row, pair_discrepancies=discrepancy))
    audit = pd.DataFrame(rows).set_index('kepid', drop=False)
    audit.to_csv(output_dir / 'curve_audit.csv', index=False)
    for name in ['kepler_candidates.csv', 'koi_catalog.csv', 'candidates.tgz']:
        path = data_dir / name
        if path.is_file():
            hashes[f'data/{name}'] = sha256(path.read_bytes()).hexdigest()
    (output_dir / 'input_fingerprints.json').write_text(json.dumps(hashes, indent=2) + '\n')
    return audit, phase, curves


def exclusion_reason(kid, dispositions, audit):
    reasons = []
    labels = dispositions.get(kid, ())
    if len(labels) > 1:
        reasons.append('mixed_KOI_dispositions')
    elif labels not in [('CONFIRMED',), ('FALSE POSITIVE',)]:
        reasons.append('missing_or_unknown_disposition')
    if kid not in audit.index:
        reasons.append('missing_curve')
    else:
        row = audit.loc[kid]
        if row.nonfinite_flux:
            reasons.append('nonfinite_flux')
        elif not np.isfinite(row.median_flux) or row.median_flux <= 0:
            reasons.append('invalid_flux_baseline')
        if row.pair_discrepancies:
            reasons.append('curve_summary_mismatch:' + row.pair_discrepancies)
    return ';'.join(reasons)


def prepare_benchmark(data_dir, output_dir):
    """Build one ordered manifest shared by the CNN and all feature baselines."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    raw, summary, partitions = load_summary_split(data_dir / 'kepler_candidates.csv')
    catalog = pd.read_csv(data_dir / 'koi_catalog.csv')
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_by_id = raw.set_index('kepid')
    assert summary_by_id.index.is_unique, 'Raw duplicate stellar identifiers require audit.'
    dispositions = audit_catalog(raw, catalog, output_dir)
    audit, phase, curves_by_id = audit_curves(data_dir, summary_by_id, output_dir)

    manifest = summary[['kepid', 'label']].copy()
    manifest.insert(0, 'summary_row', manifest.index)
    manifest['split'] = ''
    manifest['split_order'] = -1
    for split, partition in partitions.items():
        manifest.loc[partition.index, 'split'] = split
        manifest.loc[partition.index, 'split_order'] = np.arange(len(partition))
    manifest['exclusion_reason'] = manifest.kepid.map(lambda kid: exclusion_reason(kid, dispositions, audit))
    manifest['eligible'] = manifest.exclusion_reason.eq('')
    manifest['curve_path'] = manifest.kepid.map(audit.path).fillna('')
    manifest.to_csv(output_dir / 'split_manifest.csv', index=False)

    exclusions = raw[['kepid', 'label']].copy()
    exclusions['reason'] = exclusions.kepid.map(lambda kid: exclusion_reason(kid, dispositions, audit))
    incomplete = ~exclusions.kepid.isin(manifest.kepid)
    exclusions.loc[incomplete, 'reason'] = 'excluded_by_original_complete_case_filter;' + exclusions.loc[incomplete, 'reason']
    exclusions.loc[exclusions.reason.ne('')].to_csv(output_dir / 'exclusions.csv', index=False)

    coverage = manifest.groupby('split').agg(original=('kepid', 'size'), retained=('eligible', 'sum')).reindex(SPLITS)
    coverage['excluded'] = coverage.original - coverage.retained
    coverage['coverage_fraction'] = coverage.retained / coverage.original
    coverage.to_csv(output_dir / 'coverage.csv')

    eligible = manifest.loc[manifest.eligible]
    assert eligible.kepid.is_unique
    assert audit.loc[eligible.kepid, 'flux_sha256'].is_unique, 'Identical curves in eligible observations.'
    rows = {split: eligible.loc[eligible.split == split].sort_values('split_order') for split in SPLITS}
    curves, residuals, noise, features, labels = {}, {}, {}, {}, {}
    off_transit = np.abs(phase - 0.5) > 0.1
    adjacent = off_transit[:-1] & off_transit[1:]
    for split, partition in rows.items():
        assert set(partition.label.unique()) == {0, 1}
        curves[split] = np.stack([curves_by_id[kid] for kid in partition.kepid])
        residuals[split] = curves[split] / np.median(curves[split], axis=1, keepdims=True) - 1.0
        differences = np.diff(residuals[split], axis=1)[:, adjacent]
        mad = np.median(np.abs(differences - np.median(differences, axis=1, keepdims=True)), axis=1)
        noise[split] = 1.4826 * mad / np.sqrt(2)
        labels[split] = partition.label.to_numpy(dtype=np.int64)
        features[split] = summary_by_id.loc[partition.kepid, FEATURES].to_numpy(dtype=np.float64)
        assert np.isfinite(residuals[split]).all()
    flux_scale = float(np.median(noise['train']))
    noise_cap = float(np.quantile(noise['train'], 0.90))
    assert np.isfinite(flux_scale) and flux_scale > 0 and noise_cap > 0
    return BenchmarkData(manifest, coverage, rows, phase, curves, residuals, noise, features, labels,
                         flux_scale, noise_cap)
