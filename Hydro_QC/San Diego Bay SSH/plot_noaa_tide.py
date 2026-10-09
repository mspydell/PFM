#!/usr/bin/env python3
"""NOAA only: measured vs predicted water level at the tide gauge, and the non-tidal residual.

No model and no inverse-barometer correction: this shows NOAA's own data.
  panel 1  measured water level (verified where NOAA has verified it, preliminary otherwise;
           the sample on the hour, MSL datum) and NOAA's harmonic tide prediction (hourly, MSL)
  panel 2  residual = measured - predicted, hourly and PL64 low-passed (33 h; plot_ssh_lowpass.lowpass).
           This is the sea level the tide prediction leaves out: weather, coastal-trapped and
           Kelvin waves, and seasonal/interannual anomalies beyond NOAA's mean annual cycle (Sa).
Saved to plots/<name>_noaa_tide_<start>_<end>.png. Uses the cache; add --update to download first.
The same file is used in ../SanDiegoBay and ../LaJolla.

  python plot_noaa_tide.py                                 # 2025-01-01 to today
  python plot_noaa_tide.py --start 2026-06-01 --end 2026-09-30 --update
  python plot_noaa_tide.py --end today --duration 90        # the last 90 days
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from download_obs import load_noaa, update
from plot_ssh import DIFF_YLIM, GAUGE, NAME, OUT_DIR, TITLE
from plot_ssh_lowpass import NW, lowpass

PLOT_DIR = os.path.join(OUT_DIR, "plots")
START = "2025-01-01"
VERIFIED, PRELIM, PRED = "#2a78d6", "#1baf7a", "#eb6834"  # validated categorical slots 1, 3, 2


def window(p, args):
    """(start, end) dates from --start/--end/--duration; --end may be 'today' (UTC)."""
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() if args.end == "today" else pd.Timestamp(args.end)
    if args.duration is not None:
        if args.start:
            p.error("use --start or --duration, not both")
        if args.duration < 1:
            p.error("--duration must be at least 1 day")
        return end - pd.Timedelta(days=args.duration - 1), end
    start = pd.Timestamp(args.start or START)
    if end < start:
        p.error("--end must be on or after --start")
    return start, end


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", help=f"Start date YYYY-MM-DD (UTC, default {START}); not with --duration")
    p.add_argument("--end", default="today", help="End date YYYY-MM-DD or 'today', inclusive (UTC, default today)")
    p.add_argument("--duration", type=int, help="Number of days to plot, ending on --end (both days included)")
    p.add_argument("--update", action="store_true", help="Download new NOAA data first")
    a = p.parse_args()
    start, end = window(p, a)
    a.start, a.end = f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}"
    if a.update:
        update(a.end)
    idx = pd.date_range(pd.Timestamp(a.start, tz="UTC"), pd.Timestamp(a.end, tz="UTC") + pd.Timedelta(hours=23), freq="h")
    now = pd.Timestamp.now(tz="UTC")
    pad = idx[0] - pd.Timedelta(days=10)  # filter run-in
    df = load_noaa(f"{pad:%Y-%m-%d}", a.end).reindex(pd.date_range(pad, idx[-1], freq="h"))
    df.loc[df.index > now, "predicted"] = np.nan  # predictions exist for the future; measurements don't
    resid = df["measured"] - df["predicted"]
    resid_lp = lowpass(resid)
    df, resid, resid_lp = df.reindex(idx), resid.reindex(idx), resid_lp.reindex(idx)
    ver = df["measured"].where(df["quality"] == "v")
    pre = df["measured"].where(df["quality"] == "p")

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    ax = axes[0]
    ax.plot(idx, df["predicted"], color=PRED, lw=0.6, label="NOAA tide prediction (harmonic)")
    ax.plot(idx, ver, color=VERIFIED, lw=0.6, alpha=0.8, label="NOAA measured, verified")
    ax.plot(idx, pre, color=PRELIM, lw=0.6, alpha=0.8, label="NOAA measured, preliminary")
    ax.set_title(f"Water level, {GAUGE} (MSL datum)", loc="left", fontsize=10.5)
    ax.set_ylabel("Water level (m)")
    if pre.notna().any():
        ax.set_title(f"preliminary from {pre.first_valid_index():%Y-%m-%d}", loc="right", fontsize=9, color="0.25")

    ax = axes[1]
    ax.plot(idx, resid, color="0.7", lw=0.5, label="hourly")
    ax.plot(idx, resid_lp, color="0.15", lw=1.2, label="PL64 low-pass (33 h)")
    last = resid.last_valid_index()
    if last is not None:
        ax.axvspan(max(last - pd.Timedelta(hours=NW), idx[0]), idx[-1], color="0.92", zorder=0,
                   label=f"last {NW} h: filter edge, will change")
    ax.set_title("Residual = measured - predicted (non-tidal sea level)", loc="left", fontsize=10.5)
    if resid.notna().any():
        ax.set_title(f"mean = {resid.mean():+.3f} m, std = {resid.std():.3f} m (hourly)", loc="right",
                     fontsize=9, color="0.25")
    ax.set_ylabel("Residual (m)")
    ax.set_ylim(DIFF_YLIM)

    for ax in axes:
        ax.axhline(0, color="0.6", lw=0.8)
        ax.grid(axis="y", color="0.9")
        ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
    axes[-1].set_xlim(idx[0], idx[-1])
    axes[-1].set_xlabel("Date (UTC)")
    fig.suptitle(f"{TITLE}: NOAA measured vs predicted water level, {a.start} to {a.end}", fontsize=13)
    fig.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, f"{NAME}_noaa_tide_{pd.Timestamp(a.start):%Y%m%d}_{pd.Timestamp(a.end):%Y%m%d}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
