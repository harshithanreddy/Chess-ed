"""Player retention (churn) model: will a player who played in month M play
again in month M+1?

Requires two CONSECUTIVE calendar months to be ingested (e.g. 2024-01 and
2024-02): "returned" is measured as "appears at all in the very next month",
which only means something if there is no gap between the feature month and
the label month. Jumping from 2016-01 to 2019-01, for instance, cannot
produce a meaningful churn label -- a 3-year gap tells you nothing about
30-day retention.

Player identity caveat: a Lichess username is the only identifier available
in the public PGN dumps. This undercounts true "returning" players who
create a new account, and overcounts if a username were ever reused (Lichess
usernames are unique and not reused, so this second risk does not apply).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..paths import AGG_DIR

import numpy as np
import pandas as pd


def _available_months(con) -> list[str]:
    return [r[0] for r in con.execute("SELECT * FROM months").fetchall()]


def _next_month(month: str) -> str:
    y, m = map(int, month.split("-"))
    return f"{y+1}-01" if m == 12 else f"{y}-{m+1:02d}"


def build_player_month_features(con, feature_month: str, label_month: str | None = None,
                                 speed: str = "blitz") -> pd.DataFrame:
    """One row per player active in feature_month, with behavioural features
    and (if label_month is ingested) a 'returned' label for the next month.
    """
    months = _available_months(con)
    if feature_month not in months:
        raise ValueError(f"{feature_month} is not ingested. Ingested months: {months}")
    label_month = label_month or _next_month(feature_month)
    have_label = label_month in months
    if not have_label:
        print(f"[retention] WARNING: {label_month} not ingested -- building FEATURES ONLY, "
              f"no label. Run: python -m metalens run {label_month} --delete-raw")

    # Union white/black appearances into one player-centric game log for feature_month.
    per_player = con.execute(f"""
        WITH appearances AS (
            SELECT white AS player, month, played_at, white_elo AS elo,
                   coalesce(white_rating_diff, 0) AS rating_diff, white_score AS score,
                   base_seconds, increment_seconds, rated
            FROM games WHERE month = '{feature_month}' AND speed = '{speed}'
            UNION ALL
            SELECT black AS player, month, played_at, black_elo AS elo,
                   coalesce(black_rating_diff, 0) AS rating_diff, 1 - white_score AS score,
                   base_seconds, increment_seconds, rated
            FROM games WHERE month = '{feature_month}' AND speed = '{speed}'
        ),
        -- Longest consecutive-loss streak per player via "gaps and islands":
        -- row_number() over all games minus row_number() over games of the same
        -- is_loss flag is constant within one unbroken run of losses (or wins),
        -- so grouping by that difference isolates each run; verified against a
        -- hand-checked sequence in tests/test_retention.py before trusting it here.
        flagged AS (
            SELECT *, (score = 0.0) AS is_loss FROM appearances
        ),
        islands AS (
            SELECT *,
                   row_number() OVER (PARTITION BY player ORDER BY played_at)
                   - row_number() OVER (PARTITION BY player, is_loss ORDER BY played_at) AS island
            FROM flagged
        ),
        loss_streaks AS (
            SELECT player, island, count(*) AS streak_len
            FROM islands WHERE is_loss GROUP BY player, island
        ),
        max_streak AS (
            SELECT player, max(streak_len) AS max_loss_streak FROM loss_streaks GROUP BY player
        ),
        agg AS (
            SELECT
                player,
                count(*) AS n_games,
                count(DISTINCT date_trunc('day', played_at)) AS n_active_days,
                avg(elo) AS avg_elo,
                (max_by(elo, played_at) - min_by(elo, played_at)) AS rating_change,
                sum(rating_diff) AS net_rating_diff,
                avg(score) AS win_rate,
                avg(rated::int) AS rated_share
            FROM appearances
            GROUP BY player
        )
        SELECT agg.*, coalesce(max_streak.max_loss_streak, 0) AS max_loss_streak
        FROM agg LEFT JOIN max_streak USING (player)
    """).df()

    per_player["feature_month"] = feature_month
    if have_label:
        active_next = con.execute(f"""
            SELECT DISTINCT player FROM (
                SELECT white AS player FROM games WHERE month = '{label_month}'
                UNION ALL SELECT black AS player FROM games WHERE month = '{label_month}'
            )
        """).df()
        returning = set(active_next["player"])
        per_player["returned_next_month"] = per_player["player"].isin(returning).astype(int)
    return per_player


FEATURE_COLUMNS = ["n_games", "n_active_days", "avg_elo", "rating_change",
                   "net_rating_diff", "win_rate", "rated_share", "max_loss_streak"]


def train_and_evaluate(con, feature_month: str, label_month: str | None = None,
                       speed: str = "blitz", out_dir: Path | None = None,
                       shap_sample_size: int = 5000) -> dict:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.model_selection import train_test_split

    df = build_player_month_features(con, feature_month, label_month, speed)
    if "returned_next_month" not in df.columns:
        raise ValueError("Label month not ingested yet; see the warning printed above.")

    df = df.dropna(subset=FEATURE_COLUMNS)
    X, y = df[FEATURE_COLUMNS], df["returned_next_month"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "feature_month": feature_month, "label_month": label_month or _next_month(feature_month),
        "n_players": len(df), "base_rate_returned": round(y.mean(), 4),
        "auc": round(roc_auc_score(y_test, proba), 4),
        "average_precision": round(average_precision_score(y_test, proba), 4),
        "feature_importance": dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist())),
    }

    shap_summary, shap_sample_df = _compute_shap(model, X_test, sample_size=shap_sample_size)
    metrics["shap_importance"] = shap_summary

    out_dir = out_dir or (AGG_DIR / "models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out_dir / "retention_model.txt"))
    (out_dir / "retention_model_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    if shap_sample_df is not None:
        shap_sample_df.to_parquet(out_dir / "retention_shap_sample.parquet", index=False)
    print(json.dumps(metrics, indent=2, default=str))
    return metrics


def _compute_shap(model, X_test: pd.DataFrame, sample_size: int = 5000):
    """Global SHAP importance + direction, plus a small raw sample for a future
    beeswarm-style chart in the dashboard.

    Split-count feature_importances_ only says which features the model uses a
    lot; it says nothing about whether a feature pushes predictions up or down.
    SHAP values do: for each player, how much did feature X push THEIR predicted
    return-probability away from the average. Direction here is summarized as the
    correlation between a feature's value and its own SHAP value across players
    (positive = higher feature value -> more likely to return), which is a cheap
    and honest substitute for eyeballing a full beeswarm plot.

    Returns (summary_dict_or_None, sample_dataframe_or_None). Returns (None, None)
    if the `shap` package is not installed, so this is always optional.
    """
    try:
        import shap
    except ImportError:
        print("[retention] 'shap' not installed -- skipping SHAP (pip install -e .[ml])")
        return None, None

    n = min(sample_size, len(X_test))
    X_sample = X_test.sample(n, random_state=0).reset_index(drop=True)

    explainer = shap.TreeExplainer(model.booster_)
    raw = explainer.shap_values(X_sample)
    # LightGBM binary classifiers return shape (n, n_features) directly via
    # TreeExplainer in most shap versions; older/sklearn-wrapped paths can
    # instead return a 2-element list [class0_shap, class1_shap] or a 3-D
    # array (n, n_features, n_classes). Normalise to "SHAP for the positive
    # (returned) class" in all three cases.
    if isinstance(raw, list):
        shap_vals = raw[1]
    elif hasattr(raw, "ndim") and raw.ndim == 3:
        shap_vals = raw[:, :, 1]
    else:
        shap_vals = raw

    summary = {}
    for i, feat in enumerate(FEATURE_COLUMNS):
        vals = shap_vals[:, i]
        mean_abs = float(np.abs(vals).mean())
        has_variance = X_sample[feat].std() > 0 and vals.std() > 0
        corr = float(np.corrcoef(X_sample[feat], vals)[0, 1]) if has_variance else 0.0
        direction = "higher -> more likely to return" if corr >= 0 else "higher -> less likely to return"
        summary[feat] = {"mean_abs_shap": round(mean_abs, 5), "correlation": round(corr, 3), "direction": direction}
    summary = dict(sorted(summary.items(), key=lambda kv: -kv[1]["mean_abs_shap"]))

    sample_df = X_sample.copy()
    for i, feat in enumerate(FEATURE_COLUMNS):
        sample_df[f"shap_{feat}"] = shap_vals[:, i]
    return summary, sample_df
