#!/usr/bin/env python3
"""NOAA only: measured vs predicted water level at a tide gauge, and the non-tidal residual.
Shared code for plot_sdbay_noaa_tide.py and plot_lajolla_noaa_tide.py.

No model and no inverse-barometer correction: this shows NOAA's own data.
  panel 1  measured water level (verified where NOAA has verified it, preliminary otherwise;
           the sample on the hour, MSL datum) and NOAA's harmonic tide prediction (hourly, MSL)
  panel 2  residual = measured - predicted, hourly and PL64 low-passed (33 h; plot_ssh_lowpass.lowpass).
           This is the sea level the tide prediction leaves out: weather, coastal-trapped and
           Kelvin waves, and seasonal/interannual anomalies beyond NOAA's mean annual cycle (Sa).
Saved to plots/<station>_noaa_tide_<start>_<end>.png. Uses the cache; add --update to download first.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from download_obs import load_noaa, update
from plot_ssh import DIFF_YLIM, OUT_DIR, date_window, hours
from plot_ssh_lowpass import NW, PAD_DAYS, lowpass
from stations import STATIONS

PLOT_DIR = os.path.join(OUT_DIR, "plots")
START = "2025-01-01"
VERIFIED, PRELIM, PRED = "#2a78d6", "#1baf7a", "#eb6834"  # validated categorical slots 1, 3, 2


def plot_station(station, start, end):
    st = STATIONS[station]
    idx = hours(start, end)
    full = pd.date_range(idx[0] - pd.Timedelta(days=PAD_DAYS), idx[-1], freq="h")  # filter run-in
    df = load_noaa(station, f"{full[0]:%Y-%m-%d}", f"{end:%Y-%m-%d}").reindex(full)
    df.loc[df.index > pd.Timestamp.now(tz="UTC"), "predicted"] = np.nan  # predictions exist for the future
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
    ax.set_title(f"Water level, NOAA {st['noaa']} (MSL datum)", loc="left", fontsize=10.5)
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
    fig.suptitle(f"{st['title']}: NOAA measured vs predicted water level, {start:%Y-%m-%d} to {end:%Y-%m-%d}",
                 fontsize=13)
    fig.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, f"{station}_noaa_tide_{start:%Y%m%d}_{end:%Y%m%d}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


def run(station):
    """Command-line entry point of plot_<station>_noaa_tide.py."""
    p = argparse.ArgumentParser(description=f"NOAA measured vs predicted water level at {STATIONS[station]['title']}")
    p.add_argument("--start", help=f"Start date YYYY-MM-DD (UTC, default {START}); not with --duration")
    p.add_argument("--end", default="today", help="End date YYYY-MM-DD or 'today', inclusive (UTC, default today)")
    p.add_argument("--duration", type=int, help="Number of days to plot, ending on --end (both days included)")
    p.add_argument("--update", action="store_true", help="Download new NOAA data first")
    args = p.parse_args()
    start, end = date_window(p, args, START)
    if args.update:
        update(station, f"{end:%Y-%m-%d}")
    plot_station(station, start, end)
