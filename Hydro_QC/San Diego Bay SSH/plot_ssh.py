#!/usr/bin/env python3
"""Shared plotting code: PFM (day-1 forecast) vs observed sea surface height at a NOAA tide gauge.
Used by the *_3weeks.py (QC website) and *_range.py scripts in this folder. The same file is used
in ../SanDiegoBay and ../LaJolla; only the STATION settings below differ.

Figures (both laid out like Plot_SSH.m, PFM_Master_Code_For_Plots.m, PFM branch):
  <name>_ssh          hourly SSH: observed (inverse-barometer corrected, MSL datum) and model zeta;
                      Delta SSH = observed - model (y axis fixed at -0.5 to 0.5 m)
  <name>_ssh_lowpass  the same after removing the tides with the PL64 low-pass filter
                      (pl64ta.m, Rosenfeld 1983; half amplitude at 33 h), applied separately to the
                      observed and model hourly series. Each gap-free stretch is filtered on its
                      own (pl64ta.m would join the pieces across a gap) and its first/last 66 h
                      (the filter half-width, folded and tapered) are blanked next to gaps. The
                      last 66 h before the end of the data are kept but shaded: they change as
                      new data arrive. PL64 removes M2/S2 completely but passes ~3% of K1 and
                      ~9% of O1, so a 2-3 cm ~25 h wiggle remains (it largely cancels in Delta SSH).
Times UTC.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend, lfilter

from download_obs import load_obs, update as update_obs
from extract_model import LEVEL, extract, load_model

# --- station settings (the only lines that differ between SanDiegoBay and LaJolla) ---
NAME, TITLE, GAUGE = "SDBay", "San Diego Bay", "NOAA 9410170"
# --------------------------------------------------------------------------------------

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OBS_COLOR, MODEL_COLOR, DIFF_COLOR = "#2a78d6", "#eb6834", "0.25"
DIFF_YLIM = (-0.5, 0.5)  # m, Delta SSH panels
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


def two_panel(idx, obs, model, title, out_path, obs_label, model_label, shade_from=None):
    diff = obs - model
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    ax = axes[0]
    ax.plot(idx, obs, color=OBS_COLOR, lw=0.8, label=obs_label)
    ax.plot(idx, model, color=MODEL_COLOR, lw=0.8, label=model_label)
    both = pd.concat([obs, model], axis=1).dropna()
    if len(both) > 48:
        d = both.iloc[:, 1] - both.iloc[:, 0]
        ax.set_title(f"r = {both.corr().iloc[0, 1]:.3f}, RMSE = {np.sqrt((d ** 2).mean()):.3f} m, "
                     f"bias (model - obs) = {d.mean():+.3f} m, n = {len(both)} h", loc="right", fontsize=9, color="0.25")
    ax.set_title("Sea surface height", loc="left", fontsize=10.5)

    ax = axes[1]
    ax.plot(idx, diff, color=DIFF_COLOR, lw=0.8)
    if diff.notna().any():
        ax.set_title(f"mean = {diff.mean():+.3f} m, std = {diff.std():.3f} m", loc="right", fontsize=9, color="0.25")
    ax.set_title("ΔSSH = observed - model", loc="left", fontsize=10.5)
    ax.set_ylim(DIFF_YLIM)

    for ax, label in zip(axes, ("SSH (m)", "ΔSSH (m)")):
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_ylabel(label)
        ax.grid(axis="y", color="0.9")
        if shade_from is not None and shade_from < idx[-1]:
            ax.axvspan(max(shade_from, idx[0]), idx[-1], color="0.92", zorder=0,
                       label=f"last {NW} h of data: filter edge, will change" if ax is axes[0] else None)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False, ncol=3)

    span = (idx[-1] - idx[0]).days
    if span > 200:
        loc, fmt = mdates.MonthLocator(), "%b\n%Y"
    elif span > 60:
        loc, fmt = mdates.MonthLocator(), "%b %Y"
    elif span > 21:
        loc, fmt = mdates.WeekdayLocator(byweekday=mdates.MO), "%b %d"
    else:
        loc, fmt = mdates.DayLocator(interval=max(1, span // 10)), "%b %d"
    axes[-1].xaxis.set_major_locator(loc)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    axes[-1].set_xlim(idx[0], idx[-1])
    axes[-1].set_xlabel("Date (UTC)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


def plot_window(start, end, out, period):
    """Both figures for UTC dates start..end (both inclusive, YYYY-MM-DD); out(name) gives the PNG path."""
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23), freq="h")
    t0 = idx[0] - pd.Timedelta(days=PAD_DAYS)
    full = pd.date_range(t0, idx[-1], freq="h")
    obs = load_obs(f"{t0:%Y-%m-%d}", end)["wl_corr"].reindex(full)
    model = load_model()
    model = model.reindex(full) if model is not None else pd.Series(np.nan, index=full)
    obs_label = f"Observed, {GAUGE} (inverse-barometer corrected, MSL)"
    model_label = f"PFM {LEVEL} day 1 (zeta at nearest grid point)"

    two_panel(idx, obs.reindex(idx), model.reindex(idx),
              f"{TITLE} sea surface height: PFM vs tide gauge (hourly), {period}",
              out(f"{NAME}_ssh"), obs_label, model_label)

    obs_lp, model_lp = lowpass(obs), lowpass(model)
    last = min(s.last_valid_index() or full[0] for s in (obs, model))  # end of the shorter record
    two_panel(idx, obs_lp.reindex(idx), model_lp.reindex(idx),
              f"{TITLE} sea surface height, tides removed (PL64 low-pass, {CUTOFF_H} h): PFM vs tide gauge, {period}",
              out(f"{NAME}_ssh_lowpass"), obs_label + ", low-pass", model_label + ", low-pass",
              shade_from=last - pd.Timedelta(hours=NW))


def update_data(end):
    """Refresh the NOAA observations and extract any new model days up to end (UTC date)."""
    update_obs(end)
    extract(end=end)
