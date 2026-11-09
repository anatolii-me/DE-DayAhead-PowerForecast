"""Fixed model evaluation for notebook 6; no hyperparameter search."""
import hashlib
import json
import os
import time
from pathlib import Path
import importlib.metadata

import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from utils.offshore_final_data import grid, frame_hash, save_parquet, TZ, FIRST_NEW_DAY

# Use available CPUs on a hosted runner as well as on the local machine.
N_THREADS = min(4, os.cpu_count() or 1)


def prepare_day(dataset, day, max_missing_train_days=3):
    """Keep the 365-day calendar window, dropping whole incomplete history days."""
    if not 0 <= max_missing_train_days < 365:
        raise ValueError('max_missing_train_days must be between 0 and 364.')
    day = pd.Timestamp(day)
    cutoff = (day - pd.DateOffset(days=1)).tz_localize(TZ)
    index = pd.date_range(cutoff - pd.DateOffset(days=365), cutoff,
                          freq='15min', inclusive='left').tz_convert('UTC')
    train = dataset.reindex(index)
    valid = dataset.reindex(grid(day))
    local_days = index.tz_convert(TZ).tz_localize(None).normalize()
    complete = train.row_complete.fillna(False).groupby(local_days).all()
    missing = complete.index[~complete]
    keep = complete.reindex(local_days).to_numpy(dtype=bool)
    audit = dict(day=str(day.date()), model='ALL', status='ready', reason='',
        train_days=int(complete.sum()), dropped_train_days=len(missing),
        dropped_train_dates=', '.join(str(d.date()) for d in missing),
        train_rows=int(keep.sum()), expected_train_rows=len(index))
    if not valid.row_complete.fillna(False).all():
        reasons = valid.get('coverage_reason', pd.Series(dtype=str)).dropna()
        reasons = [s for s in reasons.unique() if s]
        audit.update(status='skipped_target', reason='; '.join(reasons) or 'Incomplete target data')
    elif len(missing) > max_missing_train_days:
        audit.update(status='skipped_training', reason=(
            f'Training history has {len(missing)} incomplete days; allowed {max_missing_train_days}. '
            f'Missing dates: {audit["dropped_train_dates"]}'))
    return train.loc[keep], valid, audit


def evaluate_new_days(dataset, selection, cache_dir, target_days, max_days=None, max_missing_train_days=3):
    """Refit daily within 365 calendar days ending before D-1 00:00 Berlin.

    A small number of incomplete history days can be excluded for all models.
    The cache key includes historical labels and prediction inputs. Corrections
    therefore invalidate affected fits without erasing unrelated predictions.
    """
    features = selection['fit_protocol']['features']
    seed = selection['deployment_seed']
    versions = {name: importlib.metadata.version(name) for name in ['catboost', 'lightgbm', 'xgboost', 'numpy', 'pandas']}
    implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    days = list(target_days)
    if max_days is not None:
        days = days[:max_days]
    frames, audits = [], []
    for day in tqdm(days, desc='New holdout days'):
        day = pd.Timestamp(day)
        train, valid, day_audit = prepare_day(dataset, day, max_missing_train_days)
        if day_audit['status'] != 'ready':
            audits.append(day_audit)
            continue
        deadline = (day - pd.DateOffset(days=1) + pd.Timedelta(hours=12)).tz_localize(TZ).tz_convert('UTC')
        if not pd.to_datetime(valid.dt_run_utc, utc=True).lt(deadline).all():
            raise ValueError('Weather run crosses the day-ahead deadline.')
        X_train = train[features].astype('float32')
        X_valid = valid[features].astype('float32')
        y_train = train.capacity_factor.to_numpy()
        for name, config in selection['best_by_model'].items():
            identity = dict(config=config, seed=seed, versions=versions, threads=N_THREADS,
                implementation=implementation, max_missing_train_days=max_missing_train_days,
                train=frame_hash(X_train.assign(capacity_factor=y_train)),
                inputs=frame_hash(X_valid.assign(dt_run_utc=valid.dt_run_utc)))
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            path = Path(cache_dir) / name / f'{day.date()}_{key}.parquet'
            hit = path.exists()
            if hit:
                pred = pd.read_parquet(path)
                if not pred.target_time.equals(pd.Series(valid.index.astype(str), name='target_time')):
                    raise ValueError('Cached target times do not match the requested day.')
            else:
                started = time.perf_counter()
                model = build_model(config, seed)
                model.fit(X_train, y_train, sample_weight=make_training_weights(y_train, config))
                raw = np.asarray(model.predict(X_valid))
                if not np.isfinite(raw).all():
                    raise ValueError('Non-finite predictions.')
                train_mae = np.abs(np.clip(model.predict(X_train), 0, 1) - y_train).mean()
                pred = pd.DataFrame(dict(target_time=valid.index.astype(str),
                    raw_prediction=raw, prediction=np.clip(raw, 0, 1),
                    fit_seconds=time.perf_counter() - started, train_mae=train_mae))
                save_parquet(pred, path)
            # Current labels are attached after loading the fit cache.
            pred['actual'] = valid.capacity_factor.to_numpy()
            pred['day'] = str(day.date())
            pred['seed'] = seed
            pred['row'] = valid.index.astype(str)
            pred['model'] = name
            pred['loss'] = config['loss']
            pred['period'] = 'latest_holdout'
            frames.append(pred)
            audits.append(dict(day_audit, model=name, status='evaluated',
                               cache_hit=hit, fit_seconds=float(pred.fit_seconds.iloc[0])))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), pd.DataFrame(audits)


def load_previous_predictions(selection_dir, selection):
    """Use the same deployment seed for all periods, avoiding a seed mismatch."""
    frames = []
    for name, config in selection['best_by_model'].items():
        path = Path(selection_dir) / 'holdout' / f'{name}_full_history.parquet'
        pred = pd.read_parquet(path)
        pred = pred.loc[pred.seed.eq(selection['deployment_seed'])].copy()
        if pred.empty or pred.duplicated('target_time').any():
            raise ValueError(f'Invalid historical predictions for {name}.')
        pred['model'] = name
        pred['loss'] = config['loss']
        pred['period'] = np.where(pred.day <= selection['development_end'], 'development', 'first_holdout')
        frames.append(pred)
    result = pd.concat(frames, ignore_index=True)
    # Standardize timestamp spelling across old and new files.
    result['row'] = pd.to_datetime(result.target_time, utc=True).astype(str)
    return result


def comparison_table(previous, latest):
    """Report comparable metrics on shared dates; do not average period MAEs."""
    all_predictions = pd.concat([previous, latest], ignore_index=True)
    records = []
    for period in ['development', 'first_holdout', 'latest_holdout', 'combined_holdout']:
        frame = all_predictions.loc[all_predictions.period.ne('development') if period == 'combined_holdout'
                                    else all_predictions.period.eq(period)]
        if frame.empty:
            continue
        expected_models = set(previous.model.unique())
        if set(frame.model.unique()) != expected_models:
            raise ValueError('A model is missing from the comparison period.')
        shared = set.intersection(*(set(g.row) for _, g in frame.groupby('model')))
        if any(set(g.row) != shared for _, g in frame.groupby('model')):
            raise ValueError('Candidates were evaluated on different target rows.')
        for name, pred in frame.groupby('model'):
            metrics = summarize_predictions(pred)
            daily_fit = pred.drop_duplicates('day')
            # Historical timing is unavailable in saved prediction files.
            fit_time = daily_fit.fit_seconds.mean() if 'fit_seconds' in daily_fit else np.nan
            train_mae = daily_fit.train_mae.mean() if 'train_mae' in daily_fit else np.nan
            records.append(dict(model=name, loss=pred.loss.iloc[0], period=period,
                start=pred.day.min(), end=pred.day.max(), **metrics,
                mean_fit_seconds=fit_time, train_mae=train_mae,
                validation_minus_train_mae=(metrics['mae_mean'] - train_mae
                    if 'train_mae' in daily_fit and daily_fit.train_mae.notna().all() else np.nan),
                timing_days=int(daily_fit.fit_seconds.notna().sum()) if 'fit_seconds' in daily_fit else 0))
    return pd.DataFrame(records), all_predictions

def build_model(config, seed):
    """Create one estimator from a fully resolved search configuration."""
    model_name = config["model"]
    loss_name = config["loss"]
    params = dict(config["model_params"])

    if model_name == "LightGBM":
        params.update(
            objective={
                "l2": "regression",
                "l1": "regression_l1",
                "huber": "huber",
            }[loss_name],
            subsample_freq=1,
            random_state=seed,
            n_jobs=N_THREADS,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            device_type="cpu",
        )

        if loss_name == "huber":
            params["alpha"] = config["huber_delta"]

        return lgb.LGBMRegressor(**params)

    if model_name == "XGBoost":
        params.update(
            objective={
                "l2": "reg:squarederror",
                "l1": "reg:absoluteerror",
                "huber": "reg:pseudohubererror",
            }[loss_name],
            tree_method="hist",
            device="cpu",
            random_state=seed,
            n_jobs=N_THREADS,
            verbosity=0,
            base_score=0.3,
        )

        if loss_name == "huber":
            params["huber_slope"] = config["huber_delta"]

        return xgb.XGBRegressor(**params)

    if model_name == "CatBoost":
        loss_function = {
            "l2": "RMSE",
            "l1": "MAE",
            "huber": f"Huber:delta={config['huber_delta']}",
        }[loss_name]

        params.update(
            loss_function=loss_function,
            bootstrap_type="Bernoulli",
            grow_policy="SymmetricTree",
            task_type="CPU",
            random_seed=seed,
            thread_count=N_THREADS,
            allow_writing_files=False,
            verbose=False,
        )

        # Newton leaf updates were unstable for Huber on some dates/seeds.
        # Gradient updates avoid division by a nearly empty Huber Hessian.
        if loss_name == "huber":
            params["leaf_estimation_method"] = "Gradient"
            params["leaf_estimation_iterations"] = 1

        return cb.CatBoostRegressor(**params)

    raise ValueError(f"Unknown model: {model_name}")

def make_training_weights(target, config):
    """Build normalized training weights without reading validation targets."""
    weights = (
        1.0
        + config["weight_alpha"]
        * np.clip(target, 0, 1) ** config["weight_power"]
    )

    return weights / weights.mean()

def summarize_predictions(predictions):
    """Separate variation across days, variation across seeds, and row counts."""
    if predictions.empty or predictions.duplicated(["seed", "row"]).any():
        raise ValueError("Predictions must contain one observation per seed and target row.")

    error = predictions["prediction"] - predictions["actual"]
    daily_mae = (
        predictions.assign(absolute_error=error.abs())
        .groupby(["seed", "day"])["absolute_error"].mean()
    )
    day_scores = daily_mae.groupby("day").mean()
    seed_scores = daily_mae.groupby("seed").mean()
    unique_targets = predictions.drop_duplicates("row")
    high_cf = predictions["actual"] >= 0.6
    unique_high_cf = unique_targets["actual"] >= 0.6
    target_variance = float(np.mean((predictions["actual"] - predictions["actual"].mean()) ** 2))
    mse = float(np.mean(error ** 2))

    return {
        "mae_mean": float(daily_mae.mean()),
        "mae_std": float(day_scores.std(ddof=1)) if len(day_scores) > 1 else None,
        "seed_mae_std": float(seed_scores.std(ddof=1)) if len(seed_scores) > 1 else None,
        "seed_mae_min": float(seed_scores.min()),
        "seed_mae_max": float(seed_scores.max()),
        "rmse_pooled": float(np.sqrt(mse)),
        "r2_pooled": 1.0 - mse / target_variance if target_variance > 0 else None,
        "bias": float(error.mean()),
        "p95_absolute_error": float(error.abs().quantile(0.95)),
        "high_cf_mae": float(error[high_cf].abs().mean()) if high_cf.any() else None,
        "high_cf_rows": int(unique_high_cf.sum()),
        "high_cf_days": int(unique_targets.loc[unique_high_cf, "day"].nunique()),
        "target_rows": len(unique_targets),
        "target_days": int(unique_targets["day"].nunique()),
        "n_seeds": len(seed_scores),
        "raw_outside_fraction": float((~predictions["raw_prediction"].between(0, 1)).mean()),
        "raw_prediction_min": float(predictions["raw_prediction"].min()),
        "raw_prediction_max": float(predictions["raw_prediction"].max()),
    }
