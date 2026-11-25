"""Visual theme for the MetaLens dashboard.

Palette and motifs are drawn from three chess-art references: a muted
tile-grid oil painting (rust/ochre/navy, stamped piece silhouettes), a
Folio-style illustration (deep maroon as an accent, used sparingly here --
the brighter candy tones were deliberately left out), and an art-deco
two-player illustration (dark navy/teal, geometric checkerboard ground).

Streamlit doesn't expose CSS variables the way a hand-written HTML page
does, so colors are hardcoded here and in .streamlit/config.toml instead.
The data-testid selectors below are Streamlit's documented (if not
contractually stable) styling hooks; if a Streamlit upgrade changes its
internal markup, these may need updating -- check Streamlit's release notes
for "data-testid" changes if the custom styling stops applying.
"""
from __future__ import annotations

import streamlit as st

# Chess-piece glyphs used as decorative watermarks, not gameplay UI.
PIECES = "♟ ♞ ♝ ♜ ♛ ♚"

_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
    --ml-bg:        #11151c;
    --ml-bg-alt:    #1c2430;
    --ml-rust:      #c97c3d;
    --ml-maroon:    #8c2f39;
    --ml-cream:     #eee6da;
    --ml-muted:     #7d8ba0;
}

h1, h2, h3 { font-family: "Playfair Display", Georgia, serif !important; }
[data-testid="stMetricValue"], code, .ml-mono {
    font-family: "IBM Plex Mono", monospace !important;
}

/* Subtle checkerboard strip behind the title banner, echoing the grid
   motif in the reference art -- low-contrast so it reads as texture,
   not as an actual board. */
.ml-banner {
    position: relative;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    border-radius: 10px;
    overflow: hidden;
    background:
        repeating-conic-gradient(
            var(--ml-bg-alt) 0% 25%, var(--ml-bg) 0% 50%
        ) 0 0 / 28px 28px,
        linear-gradient(135deg, rgba(201,124,61,0.10), rgba(140,47,57,0.08));
    border: 1px solid rgba(201,124,61,0.25);
    animation: ml-fade-in 0.6s ease-out;
}
.ml-banner h1 {
    margin: 0 0 0.2rem 0; color: var(--ml-cream); font-size: 2.1rem;
}
.ml-banner p { margin: 0; color: var(--ml-muted); }
.ml-pieces {
    letter-spacing: 0.4rem; color: var(--ml-rust); opacity: 0.55;
    font-size: 1.1rem; margin-top: 0.6rem;
}

/* Metric cards get a tile-like border and a soft rust glow on hover,
   nodding to the individually painted tiles in the grid reference. */
[data-testid="stMetric"] {
    background: var(--ml-bg-alt);
    border: 1px solid rgba(201,124,61,0.2);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
[data-testid="stMetric"]:hover {
    border-color: var(--ml-rust);
    transform: translateY(-2px);
}

/* Tabs: rust underline instead of Streamlit's default accent. */
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--ml-rust) !important;
    border-bottom-color: var(--ml-rust) !important;
}

@keyframes ml-fade-in {
    from { opacity: 0; transform: translateY(-6px); }
    to   { opacity: 1; transform: translateY(0); }
}
.ml-fade-in { animation: ml-fade-in 0.5s ease-out; }

/* Thin decorative divider using the two accent colors, standing in for
   a checkerboard rule without drawing an actual board. */
.ml-divider {
    height: 3px; margin: 1.4rem 0;
    background: linear-gradient(90deg, var(--ml-rust), var(--ml-maroon));
    border-radius: 2px; opacity: 0.6;
}

.ml-footer {
    color: var(--ml-muted); font-size: 0.85rem; opacity: 0.8;
}
</style>
"""


def inject() -> None:
    """Call once near the top of the app, before any other st.* calls that
    render visible content, so the CSS is in the page before it's needed.

    Uses st.html() (Streamlit 1.33+), not st.markdown(unsafe_allow_html=True):
    the latter runs the string through Streamlit's Markdown parser first, and
    a <style> block not separated by a blank line can get partially treated
    as literal text (it still gets applied by the browser regardless, since
    browsers parse <style> tags wherever they appear -- which is why the
    colors worked AND the raw CSS printed as a visible paragraph). st.html()
    skips Markdown parsing entirely, so neither issue occurs.
    """
    st.html(_CSS)


def banner(title: str, subtitle: str) -> None:
    st.html(
        f"""
        <div class="ml-banner">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <div class="ml-pieces">{PIECES}</div>
        </div>
        """
    )


def divider() -> None:
    st.html('<div class="ml-divider"></div>')


# Palette for data visualization, kept harmonious with the CSS above but
# distinct from Streamlit's default rainbow (which clashed with the rust/
# navy/cream identity). Used for Altair charts via charts.py.
CHART_COLORS = [
    "#c97c3d",  # rust (primary)
    "#8c2f39",  # maroon
    "#7d8ba0",  # dusty blue
    "#d4b483",  # sand
    "#5b7065",  # muted olive-green
    "#a85751",  # terracotta
    "#6b8f99",  # slate teal
    "#e4c88c",  # pale gold
]
CHART_BG = "#11151c"
CHART_GRID = "#2a3340"
CHART_TEXT = "#c9c2b4"
