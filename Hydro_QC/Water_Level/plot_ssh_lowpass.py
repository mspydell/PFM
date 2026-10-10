#!/usr/bin/env python3
"""Tides-removed SSH figure and the date-range plots (not for the QC website; used by
plot_<station>_range.py and noaa_tide.py).

Figure <station>_ssh_lowpass: the plot_ssh.py figure after removing the tides with the PL64
low-pass filter (pl64ta.m, Rosenfeld 1983; half amplitude at 33 h), applied separately to the
observed and model hourly series.
  - Each gap-free stretch is filtered on its own (pl64ta.m would join the pieces across a gap) and
    its first/last 66 h (the filter half-width, folded and tapered) are blanked next to gaps.
  - The last 66 h before the end of the data are kept but shaded: they change as new data arrive.
  - PL64 removes M2/S2 completely but passes ~3% of K1 and ~9% of O1, so a 2-3 cm ~25 h wiggle
    remains (it largely cancels in Delta SSH).
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.signal import detrend, lfilter

from plot_ssh import OUT_DIR, date_window, hours, labels, load_series, plot_window, two_panel, update_data
from stations import STATIONS

CUTOFF_H = 33            # PL64 half-amplitude period (h)
NW = int(np.floor(2 * CUTOFF_H))  # filter half-width (h)
PAD_DAYS = 10            # extra data before the window so its start is not a filter edge


def pl64ta(x, cutoff=CUTOFF_H):
    """Port of pl64ta.m (Steve Lentz 1992; PL64 filter, Rosenfeld 1983) for one gap-free hourly series:
    detrend, fold and cosine-taper each end, filter, add the trend back. Matches MATLAB to 1e-10."""
    x = np.asarray(x, float)
    fq = 1.0 / cutoff
    nw = int(np.floor(2 * cutoff))
    nw2 = 2 * nw
    j = np.arange(1, nw + 1)
    t = np.pi * j
    wts = (2 * np.sin(2 * fq * t) - np.sin(fq * t) - np.sin(3 * fq * t)) / (fq * fq * t ** 3)
    wts = np.concatenate([wts[::-1], [2 * fq], wts])
    wts /= wts.sum()
    n = len(x)
    if n <= nw2:
        return np.full(n, np.nan)  # too short to filter
    xdt = detrend(x)
    trnd = x - xdt
    cs = np.cos(t / nw2)
    y = np.concatenate([cs[::-1] * xdt[nw - 1::-1], xdt, cs * xdt[n - j]])
    return lfilter(wts, 1.0, y)[nw2:n + nw2] + trnd


def lowpass(s):
    """PL64 low-pass of an hourly series, each gap-free stretch on its own. The first and last NW hours
    of each stretch are folded/tapered by pl64ta and unreliable, so they are blanked, except the end
    of the newest stretch (the most recent data; shaded in the figure instead)."""
    s = s.asfreq("h")
    out = pd.Series(np.nan, index=s.index)
    good = s.notna().values
    run = np.cumsum(np.r_[True, good[1:] != good[:-1]])  # label of each run of good / missing hours
    runs = np.unique(run[good])
    for r in runs:
        pos = np.flatnonzero(run == r)
        f = pl64ta(s.values[pos])
        f[:NW] = np.nan                    # after a gap or the start of the record
        if r != runs[-1]:
            f[-NW:] = np.nan               # before a gap
        out.iloc[pos] = f
    return out


def plot_lowpass_window(station, start, end, out, period):
    """Tides-removed figure for UTC dates start..end (both inclusive); out(name) gives the PNG path."""
    idx = hours(start, end)
    full = pd.date_range(idx[0] - pd.Timedelta(days=PAD_DAYS), idx[-1], freq="h")  # filter run-in
    obs, model = load_series(station, full)
    last = min(s.last_valid_index() or full[0] for s in (obs, model))  # end of the shorter record
    obs_label, model_label = labels(station)
    two_panel(idx, lowpass(obs).reindex(idx), lowpass(model).reindex(idx),
              f"{STATIONS[station]['title']} sea surface height, tides removed (PL64 low-pass, {CUTOFF_H} h): "
              f"PFM vs tide gauge, {period}",
              out(f"{station}_ssh_lowpass"), obs_label + ", low-pass", model_label + ", low-pass",
              shade=(last - pd.Timedelta(hours=NW), f"last {NW} h of data: filter edge, will change"))


def run_range(station, default_start="2026-01-01"):
    """Command-line entry point of plot_<station>_range.py: the hourly figure (plot_ssh.py) and the
    tides-removed figure for any dates, into plots/<station>_ssh[_lowpass]_<start>_<end>.png."""
    p = argparse.ArgumentParser(description=f"PFM vs observed SSH at {STATIONS[station]['title']} for any start/end dates")
    p.add_argument("--start", help=f"Start date YYYY-MM-DD (UTC, default {default_start}); not with --duration")
    p.add_argument("--end", default="today", help="End date YYYY-MM-DD or 'today', inclusive (UTC, default today)")
    p.add_argument("--duration", type=int, help="Number of days to plot, ending on --end (both days included)")
    p.add_argument("--update", action="store_true", help="Download new obs and extract new model days first")
    args = p.parse_args()
    start, end = date_window(p, args, default_start)
    if args.update:
        update_data(station, f"{end:%Y-%m-%d}")
    plot_dir = os.path.join(OUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    out = lambda name: os.path.join(plot_dir, f"{name}_{start:%Y%m%d}_{end:%Y%m%d}.png")
    period = f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"
    plot_window(station, f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}", out, period)          # hourly SSH
    plot_lowpass_window(station, f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}", out, period)  # tides removed
