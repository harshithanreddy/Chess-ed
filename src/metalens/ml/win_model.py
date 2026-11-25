"""Win/draw/loss probability model: LightGBM (native categorical + multiclass),
benchmarked against a one-hot logistic regression baseline, as proposed.

Outputs (all under data/aggregated/models/, committed -- small, dashboard-ready):
  win_model.txt            LightGBM model (text format)
  win_model_metrics.json   log loss / accuracy for both models + the feature list
  win_model_calibration.parquet   predicted vs actual P(white wins), for a reliability chart
"""
from __future__ import annotations

import json
from pathlib import Path

from ..paths import AGG_DIR

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, LABEL_COLUMN, load_win_dataset, time_based_split


def _baseline_logloss_acc(train: pd.DataFrame, test: pd.DataFrame):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    numeric = ["white_elo", "black_elo", "rating_diff", "avg_elo", "base_seconds", "increment_seconds"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("eco", OneHotEncoder(handle_unknown="ignore"), ["eco"]),
    ])
    # multinomial is the automatic behaviour for a 3-class target on recent
    # scikit-learn; the explicit multi_class= kwarg was removed in 1.5+.
    clf = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=500))])
    clf.fit(train[FEATURE_COLUMNS], train[LABEL_COLUMN])
    proba = clf.predict_proba(test[FEATURE_COLUMNS])
    pred = clf.predict(test[FEATURE_COLUMNS])
    return {
        "log_loss": round(log_loss(test[LABEL_COLUMN], proba, labels=[0, 1, 2]), 4),
        "accuracy": round(accuracy_score(test[LABEL_COLUMN], pred), 4),
    }


def train_and_evaluate(con, speed: str = "blitz", month: str | None = None,
                       out_dir: Path | None = None) -> dict:
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, log_loss

    df = load_win_dataset(con, speed=speed, month=month)
    if len(df) < 1000:
        raise ValueError(f"Only {len(df)} rows after filtering -- too few to train on.")
    train, test = time_based_split(df)
    print(f"[win_model] train={len(train):,} test={len(test):,} (time-ordered split)")

    lgb_train = lgb.Dataset(train[FEATURE_COLUMNS], label=train[LABEL_COLUMN],
                            categorical_feature=["eco"], free_raw_data=False)
    lgb_test = lgb.Dataset(test[FEATURE_COLUMNS], label=test[LABEL_COLUMN],
                           categorical_feature=["eco"], reference=lgb_train, free_raw_data=False)
    params = {
        "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
        "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50, "verbose": -1,
    }
    model = lgb.train(
        params, lgb_train, num_boost_round=500, valid_sets=[lgb_test],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    proba = model.predict(test[FEATURE_COLUMNS], num_iteration=model.best_iteration)
    pred = proba.argmax(axis=1)
    lgb_metrics = {
        "log_loss": round(log_loss(test[LABEL_COLUMN], proba, labels=[0, 1, 2]), 4),
        "accuracy": round(accuracy_score(test[LABEL_COLUMN], pred), 4),
        "best_iteration": model.best_iteration,
    }
    baseline_metrics = _baseline_logloss_acc(train, test)
    always_draw_baseline = {
        "log_loss": round(log_loss(test[LABEL_COLUMN], np.tile(
            train[LABEL_COLUMN].value_counts(normalize=True).reindex([0, 1, 2], fill_value=1e-6),
            (len(test), 1)), labels=[0, 1, 2]), 4),
    }

    # Calibration: predicted vs actual P(white wins), in deciles of predicted probability
    cal = pd.DataFrame({"p_white_win": proba[:, 2], "actual_white_win": (test[LABEL_COLUMN] == 2).astype(int)})
    cal["bucket"] = pd.qcut(cal["p_white_win"], 10, duplicates="drop")
    calibration = cal.groupby("bucket", observed=True).agg(
        predicted=("p_white_win", "mean"), actual=("actual_white_win", "mean"), n=("actual_white_win", "size")
    ).reset_index(drop=True)

    out_dir = out_dir or (AGG_DIR / "models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / "win_model.txt"))
    calibration.to_parquet(out_dir / "win_model_calibration.parquet", index=False)
    metrics = {
        "speed": speed, "month": month or "all_ingested", "n_train": len(train), "n_test": len(test),
        "features": FEATURE_COLUMNS, "lightgbm": lgb_metrics,
        "baseline_logreg_onehot": baseline_metrics, "majority_class_baseline": always_draw_baseline,
        "outcome_rates_train": train[LABEL_COLUMN].value_counts(normalize=True).round(4).to_dict(),
    }
    (out_dir / "win_model_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps(metrics, indent=2, default=str))
    return metrics
