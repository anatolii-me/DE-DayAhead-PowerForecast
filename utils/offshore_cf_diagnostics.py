"""Descriptive target-mix checks using already saved, paired predictions."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def paired_errors(tuned, baseline):
    """Reject missing pairs or changed labels instead of silently dropping rows."""
    keys = ['model', 'period', 'row', 'seed']
    keep = keys + ['day', 'actual', 'prediction']
    left, right = tuned[keep].copy(), baseline[keep].copy()
    for frame in (left, right):
        frame['row'] = pd.to_datetime(frame.row, utc=True)
        frame['day'] = pd.to_datetime(frame.day).dt.normalize()
    pair = left.merge(right, on=keys, suffixes=('_hpo', '_baseline'),
                      how='outer', validate='one_to_one', indicator=True)
    if not pair['_merge'].eq('both').all():
        raise ValueError('HPO and baseline must cover exactly the same rows.')
    np.testing.assert_allclose(pair.actual_hpo, pair.actual_baseline, rtol=0, atol=1e-12)
    if not pair.day_hpo.equals(pair.day_baseline) or pair.seed.nunique() != 1:
        raise ValueError('Use matching target days and one deployment seed.')
    if not np.isfinite(pair[['actual_hpo', 'prediction_hpo', 'prediction_baseline']]).all().all():
        raise ValueError('Non-finite labels or predictions.')
    pair = pair.rename(columns={'day_hpo': 'day', 'actual_hpo': 'actual'})
    if not pair.actual.between(0, 1).all():
        raise ValueError('Capacity factor must be inside [0, 1].')
    # Fixed 0.1-wide bins; CF=1 belongs to the final bin.
    pair['bin'] = np.minimum(np.floor(pair.actual * 10 + 1e-12).astype(int), 9)
    for name in ['hpo', 'baseline']:
        pair[f'error_{name}'] = pair[f'prediction_{name}'] - pair.actual
        pair[f'ae_{name}'] = pair[f'error_{name}'].abs()
        pair[f'se_{name}'] = pair[f'error_{name}'] ** 2
    pair['gain'] = pair.ae_baseline - pair.ae_hpo
    # Each family must describe the same population, not just have paired rows.
    reference = None
    for _, frame in pair.groupby('model'):
        population = frame.set_index(['period', 'row']).actual.sort_index()
        if reference is not None:
            pd.testing.assert_series_equal(reference, population)
        reference = population
    return pair


def bin_metrics(pair):
    table = pair.groupby(['period', 'model', 'bin'], observed=True).agg(
        rows=('row', 'size'), days=('day', 'nunique'),
        mae_hpo=('ae_hpo', 'mean'), mae_baseline=('ae_baseline', 'mean'),
        mse_hpo=('se_hpo', 'mean'), mse_baseline=('se_baseline', 'mean'),
        bias_hpo=('error_hpo', 'mean'), bias_baseline=('error_baseline', 'mean'),
    ).reset_index()
    for name in ['hpo', 'baseline']:
        table[f'rmse_{name}'] = np.sqrt(table.pop(f'mse_{name}'))
    table['gain'] = table.mae_baseline - table.mae_hpo
    table['gain_pct'] = 100 * table.gain / table.mae_baseline
    table['cf_bin'] = table.bin.map(lambda b: f'{b / 10:.1f}–{(b + 1) / 10:.1f}')
    return table


def mix_comparison(pair, bins, min_rows=20, min_days=3):
    """Compare each holdout with development on explicitly shared CF support.

    Weighted scores are pooled interval MAE, not the main mean daily MAE.
    Positive gain always means HPO has lower MAE than baseline.
    """
    records = []
    for model in sorted(pair.model.unique()):
        dev = bins[(bins.model == model) & (bins.period == 'development')].set_index('bin')
        for period in ['first_holdout', 'latest_holdout']:
            hold = bins[(bins.model == model) & (bins.period == period)].set_index('bin')
            support = dev.index.intersection(hold.index)
            support = support[(dev.loc[support, 'rows'] >= min_rows)
                & (hold.loc[support, 'rows'] >= min_rows)
                & (dev.loc[support, 'days'] >= min_days)
                & (hold.loc[support, 'days'] >= min_days)]
            if not len(support):
                continue
            d, h = dev.loc[support], hold.loc[support]
            wd, wh = d.rows / d.rows.sum(), h.rows / h.rows.sum()
            dev_gain, hold_gain = float(wd @ d.gain), float(wh @ h.gain)
            standardized_gain = float(wd @ h.gain)
            raw = pair[(pair.model == model) & (pair.period == period)]
            record = dict(model=model, period=period,
                bins_used=', '.join(d.cf_bin),
                development_mass=float(d.rows.sum() / dev.rows.sum()),
                holdout_mass=float(h.rows.sum() / hold.rows.sum()),
                raw_daily_gain=float(raw.groupby('day').gain.mean().mean()),
                raw_pooled_gain=float(raw.gain.mean()),
                development_gain_shared=dev_gain, holdout_gain_shared=hold_gain,
                holdout_gain_development_mix=standardized_gain,
                development_gain_holdout_mix=float(wh @ d.gain),
                # Exact decomposition on shared support, in CF units.
                mix_effect=hold_gain - standardized_gain,
                within_bin_effect=standardized_gain - dev_gain,
                holdout_mae_hpo_development_mix=float(wd @ h.mae_hpo),
                holdout_mae_baseline_development_mix=float(wd @ h.mae_baseline))
            np.testing.assert_allclose(hold_gain - dev_gain,
                record['mix_effect'] + record['within_bin_effect'], atol=1e-12)
            records.append(record)
    return pd.DataFrame(records)


def matched_windows(pair, window_days=28, min_days=21):
    """Match each fixed holdout block to a development window by targets only.

    Total variation distance uses all ten bins. Zero means identical histograms.
    Windows are chosen once for all models without reading prediction errors.
    """
    population = pair.drop_duplicates(['period', 'row']).copy()
    population['day'] = pd.to_datetime(population.day)
    dev = population[population.period == 'development']
    if dev.empty:
        return pd.DataFrame()
    def window(frame, start):
        return frame[(frame.day >= start) & (frame.day < start + pd.Timedelta(days=window_days))]
    def histogram(frame):
        return frame.bin.value_counts(normalize=True).reindex(range(10), fill_value=0).to_numpy()
    candidates = []
    for start in pd.date_range(dev.day.min(), dev.day.max() - pd.Timedelta(days=window_days - 1)):
        frame = window(dev, start)
        if frame.day.nunique() == window_days:
            candidates.append((start, histogram(frame), frame.actual.mean()))
    records = []
    for period in ['first_holdout', 'latest_holdout']:
        hold = population[population.period == period]
        if hold.empty or not candidates:
            continue
        # Include only full calendar spans; the final short tail is omitted.
        for start in pd.date_range(hold.day.min(), hold.day.max() - pd.Timedelta(days=window_days - 1), freq=f'{window_days}D'):
            target = window(hold, start)
            if target.day.nunique() < min_days:
                continue
            histogram_h = histogram(target)
            best = min(candidates, key=lambda c: (np.abs(c[1] - histogram_h).sum(), c[0]))
            dev_start, histogram_d, dev_mean = best
            for model, frame in pair.groupby('model'):
                d = window(frame[frame.period == 'development'], dev_start)
                h = window(frame[frame.period == period], start)
                records.append(dict(model=model, period=period,
                    holdout_start=start.date(), holdout_end=(start + pd.Timedelta(days=window_days - 1)).date(),
                    development_start=dev_start.date(), development_end=(dev_start + pd.Timedelta(days=window_days - 1)).date(),
                    histogram_distance=float(np.abs(histogram_d - histogram_h).sum() / 2),
                    development_mean_cf=dev_mean, holdout_mean_cf=float(target.actual.mean()),
                    development_high_cf_share=float(histogram_d[6:].sum()),
                    holdout_high_cf_share=float(histogram_h[6:].sum()),
                    holdout_days=h.day.nunique(),
                    development_daily_gain=float(d.groupby('day').gain.mean().mean()),
                    holdout_daily_gain=float(h.groupby('day').gain.mean().mean()),
                    holdout_mae_baseline=float(h.groupby('day').ae_baseline.mean().mean()),
                    holdout_mae_hpo=float(h.groupby('day').ae_hpo.mean().mean())))
    return pd.DataFrame(records)


def error_figure(bins, period):
    """MAE, RMSE and support counts, keeping empty bins visibly empty."""
    models = sorted(bins.model.unique())
    fig, axes = plt.subplots(3, len(models), figsize=(15, 9), sharex=True, sharey='row', squeeze=False)
    for col, model in enumerate(models):
        data = bins[(bins.model == model) & (bins.period == period)].set_index('bin').reindex(range(10))
        x = np.arange(10) / 10 + .05
        for row, metric in enumerate(['mae', 'rmse']):
            ax = axes[row, col]
            ax.plot(x, data[f'{metric}_baseline'], 'o-', label='Baseline L2')
            ax.plot(x, data[f'{metric}_hpo'], 'o-', label='HPO selected')
            ax.set_ylabel(metric.upper() + ' (CF)')
            ax.grid(alpha=.25)
            ax.axvline(.6, color='gray', linestyle=':', linewidth=1)
            if row == 0:
                ax.set_title(model)
                ax.legend()
        axes[2, col].bar(x, data.rows.fillna(0), width=.08, color='gray', alpha=.6)
        for xi, count, days in zip(x, data.rows, data.days):
            if pd.notna(count):
                axes[2, col].text(xi, count, f'{int(days)}d', ha='center', va='bottom', fontsize=8)
        axes[2, col].set(xlabel='Actual CF (bin midpoint)', ylabel='Intervals; labels = days', xlim=(0, 1))
    fig.suptitle(f'{period}: baseline and HPO errors on identical intervals')
    fig.tight_layout()
    return fig
