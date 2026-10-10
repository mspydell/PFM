#!/usr/bin/env python3
"""Tides-removed SSH figure (not for the QC website; used by the *_range.py script and plot_noaa_tide.py).

Figure <name>_ssh_lowpass: the plot_ssh.py figure after removing the tides with the PL64 low-pass
filter (pl64ta.m, Rosenfeld 1983; half amplitude at 33 h), applied separately to the observed and
model hourly series.
  - Each gap-free stretch is filtered on its own (pl64ta.m would join the pieces across a gap) and
    its first/last 66 h (the filter half-width, folded and tapered) are blanked next to gaps.
  - The last 66 h before the end of the data are kept but shaded: they change as new data arrive.
  - PL64 removes M2/S2 completely but passes ~3% of K1 and ~9% of O1, so a 2-3 cm ~25 h wiggle
    remains (it largely cancels in Delta SSH).
The same file is used in ../SanDiegoBay and ../LaJolla.
"""
import numpy as np
import pandas as pd
from scipy.signal import detrend, lfilter

from plot_ssh import MODEL_LABEL, NAME, OBS_LABEL, TITLE, hours, load_series, two_panel

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


def plot_lowpass_window(start, end, out, period):
    """Tides-removed figure for UTC dates start..end (both inclusive); out(name) gives the PNG path."""
    idx = hours(start, end)
    full = pd.date_range(idx[0] - pd.Timedelta(days=PAD_DAYS), idx[-1], freq="h")  # filter run-in
    obs, model = load_series(full)
    last = min(s.last_valid_index() or full[0] for s in (obs, model))  # end of the shorter record
    two_panel(idx, lowpass(obs).reindex(idx), lowpass(model).reindex(idx),
              f"{TITLE} sea surface height, tides removed (PL64 low-pass, {CUTOFF_H} h): PFM vs tide gauge, {period}",
              out(f"{NAME}_ssh_lowpass"), OBS_LABEL + ", low-pass", MODEL_LABEL + ", low-pass",
              shade=(last - pd.Timedelta(hours=NW), f"last {NW} h of data: filter edge, will change"))
