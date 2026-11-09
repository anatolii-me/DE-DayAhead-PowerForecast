"""Checks for matched populations and the target-mix decomposition."""
import unittest
import numpy as np
import pandas as pd
from utils.offshore_cf_diagnostics import paired_errors, bin_metrics, mix_comparison


class TargetMixTests(unittest.TestCase):
    def frames(self):
        records = []
        # Conditional errors stay fixed; only the target frequencies change.
        for period, start, counts in [('development', '2025-01-01', (2, 8)),
                                      ('latest_holdout', '2026-01-01', (8, 2))]:
            for day in pd.date_range(start, periods=4):
                for i, actual in enumerate([.15] * counts[0] + [.65] * counts[1]):
                    records.append(dict(model='test', period=period,
                        row=str(day.tz_localize('UTC') + pd.Timedelta(minutes=15*i)),
                        seed=1337, day=str(day.date()), actual=actual))
        baseline = pd.DataFrame(records)
        tuned = baseline.copy()
        baseline['prediction'] = baseline.actual + .1
        tuned['prediction'] = tuned.actual + np.where(tuned.actual < .6, .15, .02)
        return tuned, baseline

    def test_mix_only_reversal(self):
        h, b = self.frames()
        p = paired_errors(h, b)
        result = mix_comparison(p, bin_metrics(p), min_rows=1, min_days=1).iloc[0]
        self.assertGreater(result.development_gain_shared, 0)
        self.assertLess(result.holdout_gain_shared, 0)
        self.assertAlmostEqual(result.within_bin_effect, 0)
        self.assertAlmostEqual(result.holdout_gain_development_mix, result.development_gain_shared)

    def test_missing_pair_and_changed_label_fail(self):
        h, b = self.frames()
        with self.assertRaises(ValueError):
            paired_errors(h.iloc[:-1], b)
        b.loc[0, 'actual'] += .01
        with self.assertRaises(AssertionError):
            paired_errors(h, b)

    def test_sparse_support_is_not_extrapolated(self):
        h, b = self.frames()
        p = paired_errors(h, b)
        result = mix_comparison(p, bin_metrics(p), min_rows=10, min_days=1)
        self.assertTrue(result.empty)  # No bin has 10 rows in both periods.


if __name__ == '__main__':
    unittest.main()
