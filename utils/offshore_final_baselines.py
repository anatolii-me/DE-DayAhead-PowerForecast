"""Notebook-4 baseline definitions and cached historical predictions."""
import hashlib
import json
from pathlib import Path

import pandas as pd


def baseline_selection(selection):
    """Return the fixed 365/1 L2 controls introduced in notebook 4."""
    params = {
        'LightGBM': dict(n_estimators=200, learning_rate=0.05, max_depth=4,
            num_leaves=15, min_child_samples=50, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=5.0),
        'XGBoost': dict(n_estimators=200, learning_rate=0.05, max_depth=4,
            min_child_weight=50.0, subsample=0.8, colsample_bytree=0.8,
            gamma=0.0, reg_alpha=0.1, reg_lambda=5.0),
        'CatBoost': dict(iterations=200, learning_rate=0.05, depth=4,
            l2_leaf_reg=5.0, random_strength=1.0, subsample=0.8, rsm=0.8),
    }
    configs = {
        name: dict(model=name, loss='l2', model_params=model_params,
            huber_delta=None, weighting='uniform', weight_alpha=0.0,
            weight_power=1.0)
        for name, model_params in params.items()
    }
    return dict(fit_protocol=selection['fit_protocol'],
                deployment_seed=selection['deployment_seed'],
                best_by_model=configs)


def _object_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def load_previous_predictions(selection_dir, selection):
    """Load notebook-4 controls replayed in notebook 5 on matched periods."""
    baseline = baseline_selection(selection)
    frames = []
    for name, config in baseline['best_by_model'].items():
        development_path = Path(selection_dir) / f'{_object_hash(config)}.parquet'
        holdout_path = Path(selection_dir) / 'holdout' / f'{name}_baseline.parquet'
        development = pd.read_parquet(development_path)
        development = development.loc[
            development.seed.eq(selection['deployment_seed'])
        ].copy()
        development['period'] = 'development'
        first_holdout = pd.read_parquet(holdout_path).copy()
        first_holdout['period'] = 'first_holdout'
        for frame in (development, first_holdout):
            frame['model'] = name
            frame['loss'] = 'l2'
            frame['row'] = pd.to_datetime(frame.target_time, utc=True).astype(str)
            if frame.empty or frame.duplicated('row').any():
                raise ValueError(f'Invalid notebook-4 baseline predictions for {name}.')
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if set(result.period.unique()) != {'development', 'first_holdout'}:
        raise ValueError('Notebook-4 baseline periods are incomplete.')
    return result
