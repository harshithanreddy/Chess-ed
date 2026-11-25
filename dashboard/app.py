"""MetaLens dashboard: reads ONLY the small, committed files in data/aggregated/.
Never touches data/raw or data/processed, so it runs after those are deleted,
and deploys to Streamlit Community Cloud straight from the repo.

Run: streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import charts
from theme import banner, divider, inject as inject_theme

AGG_DIR = Path(__file__).resolve().parents[1] / "data" / "aggregated"

st.set_page_config(page_title="MetaLens", layout="wide", page_icon="♞")
inject_theme()
banner(
    "MetaLens",
    "Game analytics on 100M+ Lichess games -- opening meta shift, matchmaking "
    "fairness, and player retention, framed for a game-studio audience.",
)


@st.cache_data
def load(name: str) -> pd.DataFrame | None:
    path = AGG_DIR / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data
def load_json(name: str) -> dict | None:
    path = AGG_DIR / "models" / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


tab_meta, tab_match, tab_time, tab_models = st.tabs(
    ["Opening Meta", "Matchmaking Fairness", "Time Pressure", "ML Models"]
)

with tab_meta:
    df = load("opening_meta_by_month")
    if df is None:
        st.info("Run `python -m metalens analyze` to generate this chart.")
    else:
        bands = sorted(df["rating_band"].unique())
        band = st.selectbox("Rating band", bands, index=min(1, len(bands) - 1), key="meta_band")
        sub = df[df["rating_band"] == band]
        top_openings = sub.groupby("opening")["n_games"].sum().nlargest(8).index

        # Lichess sometimes gives the SAME display name to several ECO codes
        # (e.g. multiple B1x sub-variations can all show as plain "Caro-Kann
        # Defense"). Grouping by name alone, without this step, leaves two
        # rows at the same (month, name) -- Altair then draws two points at
        # the same x and connects them, which looks like the line plunging
        # to near-zero and snapping back. pick_rate is already n_games /
        # band_games with a shared denominator within a month, so colliding
        # rows combine by SUMMING pick_rate (verified against hand-computed
        # numbers before shipping this).
        filtered = sub[sub["opening"].isin(top_openings)]
        long_df = filtered.groupby(["month", "opening"], as_index=False).agg(
            pick_rate=("pick_rate", "sum"), n_games=("n_games", "sum")
        )
        st.subheader(f"Opening pick rate over time -- {band}")
        charts.line(long_df, x="month", y="pick_rate", color="opening",
                   x_title="Month", y_title="Pick rate")
        divider()
        st.subheader("Win rate by opening (latest month)")
        # Same name-collision issue applies here, but win rate is an AVERAGE,
        # not a sum -- colliding rows must be combined as a games-weighted
        # average, not summed (summing two win rates is not a meaningful
        # number). Also verified against hand-computed numbers before shipping.
        latest_month = sub[sub["month"] == sub["month"].max()]
        n_games_sum = latest_month.groupby("opening")["n_games"].sum()
        weighted_sum = (latest_month["white_win_rate"] * latest_month["n_games"]).groupby(
            latest_month["opening"]
        ).sum()
        latest_agg = pd.concat(
            [n_games_sum.rename("n_games"), (weighted_sum / n_games_sum).rename("white_win_rate")], axis=1
        ).reset_index()
        latest = latest_agg.nlargest(10, "n_games")
        charts.bar(latest, x="opening", y="white_win_rate", x_title="Opening", y_title="White win rate")
        st.caption(
            "Conditioned on rating band, so this is not just 'strong players play X more'."
        )

with tab_match:
    df = load("matchmaking_fairness")
    if df is None:
        st.info("Run `python -m metalens analyze` to generate this chart.")
    else:
        bands = sorted(df["rating_band"].unique())
        band = st.selectbox("Rating band", bands, index=min(1, len(bands) - 1), key="mm_band")
        sub = df[df["rating_band"] == band].sort_values("avg_gap")
        st.subheader(f"White win rate vs. rating gap -- {band}")
        charts.line(sub, x="avg_gap", y="white_win_rate", y_domain=(0, 1),
                   x_title="Rating gap (White Elo - Black Elo)", y_title="White win rate")
        st.caption(
            "x axis: white Elo minus black Elo. A well-functioning matchmaker keeps most "
            "games near gap=0; the curve shows how much edge a given gap actually buys."
        )
        st.dataframe(sub[["avg_gap", "n_games", "white_win_rate"]], use_container_width=True)

with tab_time:
    cov = load("clock_coverage_by_month")
    tp = load("time_pressure")
    if cov is not None:
        st.subheader("Clock-data coverage by month")
        st.caption("Clock annotations are sparse before ~2017 -- check this before trusting the chart below.")
        cov_display = cov.copy()
        for c in ("pct_has_clock", "pct_has_eval"):
            cov_display[c] = (cov_display[c] * 100).round(1).astype(str) + "%"
        st.dataframe(cov_display, use_container_width=True)
        divider()
    if tp is None or tp.empty:
        st.info("No time-pressure rows yet -- ingest a month from 2017+ and re-run `metalens analyze`.")
    else:
        bands = sorted(tp["rating_band"].unique())
        band = st.selectbox("Rating band", bands, index=min(1, len(bands) - 1), key="tp_band")
        sub = tp[tp["rating_band"] == band].sort_values("avg_seconds_left")
        st.subheader(f"White loss rate vs. clock time remaining -- {band}")
        charts.line(sub, x="avg_seconds_left", y="white_loss_rate", y_domain=(0, 1),
                   x_title="Seconds remaining on clock", y_title="White loss rate")

with tab_models:
    win_metrics = load_json("win_model_metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Win / draw / loss model")
        if win_metrics is None:
            st.info("Run `python -m metalens train-win` to populate this.")
        else:
            st.metric("LightGBM log loss", win_metrics["lightgbm"]["log_loss"])
            st.metric("vs. majority-class baseline", win_metrics["majority_class_baseline"]["log_loss"])
            st.metric("vs. one-hot logistic regression", win_metrics["baseline_logreg_onehot"]["log_loss"])
            cal = load("models/win_model_calibration")
            if cal is not None:
                st.caption("Calibration: predicted vs. actual P(white wins), by decile "
                          "(dashed line = perfectly calibrated)")
                charts.calibration(cal)
    with col2:
        st.subheader("Player retention model")
        ret_metrics = load_json("retention_model_metrics")
        if ret_metrics is None:
            st.info(
                "Run `python -m metalens train-retention <month>` after ingesting "
                "two consecutive months."
            )
        else:
            st.metric("AUC", ret_metrics["auc"])
            st.metric("Average precision", ret_metrics["average_precision"])
            st.metric("Base return rate", ret_metrics["base_rate_returned"])
            imp_df = (
                pd.Series(ret_metrics["feature_importance"])
                .rename("importance").rename_axis("feature").reset_index()
            )
            st.caption("Feature importance (split count -- which features the model uses most)")
            charts.bar(imp_df, x="feature", y="importance", x_title="Feature", y_title="Split count")
            shap_imp = ret_metrics.get("shap_importance")
            if shap_imp:
                st.caption("SHAP importance + direction (requires `pip install -e .[ml]`)")
                shap_df = pd.DataFrame(shap_imp).T
                shap_df["mean_abs_shap"] = shap_df["mean_abs_shap"].astype(float)
                st.dataframe(
                    shap_df[["mean_abs_shap", "correlation", "direction"]].sort_values(
                        "mean_abs_shap", ascending=False
                    ),
                    use_container_width=True,
                )

divider()
st.html(
    '<p class="ml-footer">Data: Lichess open database (CC0). This dashboard reads only '
    "data/aggregated/, which is committed to the repo -- raw downloads and full Parquet "
    "output are not required to run or deploy it.</p>"
)
