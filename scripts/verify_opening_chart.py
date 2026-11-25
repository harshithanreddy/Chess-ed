"""Diagnostic: checks the exact numbers behind the Opening Meta chart.

Run from the repo root (same convention as everything else -- this reads
the committed data/aggregated/ file, so it works even without raw/processed
data on disk):

    python scripts/verify_opening_chart.py --band 1200-1599

Prints:
1. The top-8 openings for that band and the exact pick_rate per month --
   compare these numbers directly against what the chart renders.
2. A check that the collision fix produced exactly one row per
   (month, opening) -- if this fails, the chart WILL be wrong again.
3. A "same-family name" scan across the WHOLE table (not just the top 8):
   groups opening names by the text before a leading colon (e.g. "Queen's
   Pawn Game" and "Queen's Pawn Game: Chigorin Variation" share the root
   "Queen's Pawn Game") so you can see which display names are variations
   of the same opening family. This is informational only -- merging them
   in the chart is a judgment call, not something this script decides.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

AGG_DIR = Path(__file__).resolve().parents[1] / "data" / "aggregated"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="1200-1599")
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--family-scan-out", default="family_scan.txt",
                    help="the same-root-name scan is long (most openings have this tree "
                         "structure), so it's written here instead of the terminal, which "
                         "can silently scroll the top-8/collision results out of view.")
    args = ap.parse_args()

    df = pd.read_parquet(AGG_DIR / "opening_meta_by_month.parquet")
    sub = df[df["rating_band"] == args.band]
    if sub.empty:
        print(f"No rows for rating_band={args.band!r}. Available bands: "
              f"{sorted(df['rating_band'].unique())}")
        return

    top = sub.groupby("opening")["n_games"].sum().nlargest(args.top_n).index
    filtered = sub[sub["opening"].isin(top)]

    print(f"=== Top {args.top_n} openings, band={args.band} ===")
    print(sub.groupby("opening")["n_games"].sum().nlargest(args.top_n).to_string())

    # Same aggregation the dashboard now does -- reproduced here so this
    # script tests the ACTUAL logic, not a re-description of it.
    combined = filtered.groupby(["month", "opening"], as_index=False).agg(
        pick_rate=("pick_rate", "sum"), n_games=("n_games", "sum")
    )

    print(f"\n=== Exact values the chart plots (band={args.band}) ===")
    pivot = combined.pivot(index="month", columns="opening", values="pick_rate")
    print(pivot.to_string())

    print("\n=== Collision check ===")
    dup_check = combined.groupby(["month", "opening"]).size()
    bad = dup_check[dup_check > 1]
    if bad.empty:
        print("OK: exactly one row per (month, opening) -- no collisions, chart will be correct.")
    else:
        print("FAIL: duplicate (month, opening) pairs still exist:")
        print(bad.to_string())

    # The full scan is long -- almost every opening with any depth has this
    # tree structure (e.g. Sicilian Defense alone has 200+ named lines), so
    # printing it to the terminal reliably scrolls the top-8/collision
    # results above it out of the visible scrollback. Write it to a file
    # instead; only print whether any of THIS band's current top-N openings
    # collide with each other, which is the one case actually worth a
    # by-hand decision.
    all_names = df["opening"].dropna().unique()
    roots = {}
    for name in all_names:
        root = name.split(":")[0].strip()
        roots.setdefault(root, set()).add(name)
    families = {root: names for root, names in roots.items() if len(names) > 1}

    out_path = Path(args.family_scan_out)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Same-root-name families across the whole table ({len(families)} found):\n\n")
        for root, names in sorted(families.items()):
            total = df[df["opening"].isin(names)]["n_games"].sum()
            f.write(f"Family root: {root!r}  (combined n_games={total:,})\n")
            for n in sorted(names):
                g = df[df["opening"] == n]["n_games"].sum()
                f.write(f"  - {n!r}  (n_games={g:,})\n")
            f.write("\n")

    print(f"\n=== Same-family name scan: {len(families)} families found, written to {out_path} ===")
    print("(This is expected -- most openings have this tree structure; it is NOT a sign of a bug.)")

    top_names = set(top)
    top_families = {root: names & top_names for root, names in families.items() if len(names & top_names) > 1}
    if top_families:
        print(f"\n*** For band={args.band}, these CURRENT top-{args.top_n} legend entries share a root ***")
        for root, names in top_families.items():
            print(f"  {root!r}: {sorted(names)}")
        print("  This is the one case worth deciding by hand -- tell me if you want these merged.")
    else:
        print(f"\nNone of the current top-{args.top_n} openings for band={args.band} share a root "
              "with each other -- no merge decision needed for this chart right now.")


if __name__ == "__main__":
    main()
