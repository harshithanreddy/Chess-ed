"""Needs duckdb, pandas, lightgbm, scikit-learn, shap (pip install -e .[dev,ml])."""
import unittest

try:
    import duckdb  # noqa: F401
    import lightgbm  # noqa: F401
    import shap  # noqa: F401
    import sklearn  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "duckdb/lightgbm/scikit-learn/shap not installed")
class RetentionShapTest(unittest.TestCase):
    def test_shap_summary_has_expected_shape_and_direction_for_known_signal(self):
        import numpy as np
        import pandas as pd

        from metalens.ml.retention import FEATURE_COLUMNS, _compute_shap

        rng = np.random.default_rng(0)
        n = 3000
        X = pd.DataFrame(
            {c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS}
        )

        class FakeBooster:  # LGBMClassifier.booster_ stand-in; shap.TreeExplainer
            pass           # is monkeypatched below so this is never introspected

        class FakeModel:
            booster_ = FakeBooster()

        # Train a REAL tiny LightGBM model so shap.TreeExplainer gets a genuine
        # booster, with net_rating_diff deliberately the dominant driver of y.
        import lightgbm as lgb

        y = (X["net_rating_diff"] + 0.1 * rng.normal(0, 1, n) > 0).astype(int)
        model = lgb.LGBMClassifier(n_estimators=50, num_leaves=7, min_child_samples=20)
        model.fit(X, y)

        summary, sample_df = _compute_shap(model, X, sample_size=1000)
        self.assertIsNotNone(summary)
        self.assertEqual(set(summary.keys()), set(FEATURE_COLUMNS))
        # the feature that actually drives y should rank #1 by mean_abs_shap
        top_feature = next(iter(summary))
        self.assertEqual(top_feature, "net_rating_diff")
        self.assertEqual(summary["net_rating_diff"]["direction"], "higher -> more likely to return")
        self.assertIn("shap_net_rating_diff", sample_df.columns)


if __name__ == "__main__":
    unittest.main()
