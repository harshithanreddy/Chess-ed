"""Themed Altair charts, replacing st.line_chart/st.bar_chart's default
rainbow palette with the project's rust/navy/cream identity.

Why Altair instead of the one-line st.line_chart: Streamlit's built-in
chart helpers don't expose a custom color sequence, axis tick density, or
explicit y-domain -- all three were visible problems in the first themed
screenshot (clashing default colors, overlapping x-axis labels on the
16-bucket matchmaking chart, and a y-axis that auto-padded up to 1.6 for
data that is always in [0, 1]). Altair (bundled with Streamlit, no separate
install) gives direct control over all three.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from theme import CHART_BG, CHART_COLORS, CHART_GRID, CHART_TEXT


def _base(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure(background=CHART_BG)
        .configure_view(strokeWidth=0)
        .configure_axis(
            gridColor=CHART_GRID, domainColor=CHART_GRID, tickColor=CHART_GRID,
            labelColor=CHART_TEXT, titleColor=CHART_TEXT, labelFontSize=11,
        )
        .configure_legend(labelColor=CHART_TEXT, titleColor=CHART_TEXT)
    )


def line(df: pd.DataFrame, x: str, y: str, color: str | None = None,
         y_domain: tuple[float, float] | None = None, x_title: str | None = None,
         y_title: str | None = None, height: int = 360) -> None:
    """A themed multi- or single-series line chart.

    y_domain pins the axis (e.g. (0, 1) for a win rate) instead of letting
    Vega-Lite auto-pad it, which was making small win-rate differences look
    exaggerated or, conversely, making the real range look flatter than it is.
    """
    y_scale = alt.Scale(domain=list(y_domain)) if y_domain else alt.Undefined
    enc = dict(
        x=alt.X(f"{x}:Q" if pd.api.types.is_numeric_dtype(df[x]) else f"{x}:N",
                title=x_title or x),
        y=alt.Y(f"{y}:Q", title=y_title or y, scale=y_scale),
    )
    if color:
        enc["color"] = alt.Color(f"{color}:N", scale=alt.Scale(range=CHART_COLORS), title=color)
    chart = alt.Chart(df).mark_line(strokeWidth=2.4).encode(**enc).properties(height=height)
    st.altair_chart(_base(chart), use_container_width=True)


def bar(df: pd.DataFrame, x: str, y: str, color: str | None = None,
        height: int = 320, x_title: str | None = None, y_title: str | None = None) -> None:
    enc = dict(
        x=alt.X(f"{x}:N", title=x_title or x, sort="-y"),
        y=alt.Y(f"{y}:Q", title=y_title or y),
    )
    if color:
        enc["color"] = alt.Color(f"{color}:N", scale=alt.Scale(range=CHART_COLORS), legend=None)
    else:
        enc["color"] = alt.value(CHART_COLORS[0])
    chart = alt.Chart(df).mark_bar().encode(**enc).properties(height=height)
    st.altair_chart(_base(chart), use_container_width=True)


def calibration(df: pd.DataFrame, x: str = "predicted", y: str = "actual", height: int = 300) -> None:
    """Predicted vs. actual rate, with a dashed y=x reference line for
    'perfectly calibrated'. A bare line (what the dashboard showed before)
    doesn't tell you whether the model is over- or under-confident; the
    reference line does.
    """
    points = alt.Chart(df).mark_line(strokeWidth=2.4, point=True, color=CHART_COLORS[0]).encode(
        x=alt.X(f"{x}:Q", title="Predicted P(white wins)", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y(f"{y}:Q", title="Actual P(white wins)", scale=alt.Scale(domain=[0, 1])),
    )
    ref = alt.Chart(pd.DataFrame({"a": [0, 1], "b": [0, 1]})).mark_line(
        strokeDash=[4, 4], color=CHART_TEXT, opacity=0.5
    ).encode(x="a:Q", y="b:Q")
    st.altair_chart(_base((ref + points).properties(height=height)), use_container_width=True)
