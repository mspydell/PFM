#!/usr/bin/env python3
"""Shared plotting code: PFM (day-1 forecast) vs observed sea surface height at a NOAA tide gauge.
Used by the *_3weeks.py (QC website) and *_range.py scripts in this folder. The same file is used
in ../SanDiegoBay and ../LaJolla; only the STATION settings below differ.

Figure <name>_ssh, laid out like Plot_SSH.m (PFM_Master_Code_For_Plots.m, PFM branch):
  panel 1  hourly observed (inverse-barometer corrected, MSL datum) and model zeta
  panel 2  Delta SSH = observed - model (y axis fixed at -0.5 to 0.5 m)
Times UTC. The tides-removed version (not for the website) is in plot_ssh_lowpass.py.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from download_obs import load_obs, update as update_obs
from extract_model import LEVEL, extract, load_model

# --- station settings (the only lines that differ between SanDiegoBay and LaJolla) ---
NAME, TITLE, GAUGE = "SDBay", "San Diego Bay", "NOAA 9410170"
# --------------------------------------------------------------------------------------

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OBS_COLOR, MODEL_COLOR, DIFF_COLOR = "#2a78d6", "#eb6834", "0.25"
DIFF_YLIM = (-0.5, 0.5)  # m, Delta SSH panels
OBS_LABEL = f"Observed, {GAUGE} (inverse-barometer corrected, MSL)"
MODEL_LABEL = f"PFM {LEVEL} day 1 (zeta at nearest grid point)"


def hours(start, end):
    """Hourly UTC index covering the dates start..end (both inclusive)."""
    return pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23), freq="h")


def load_series(idx):
    """Hourly observed (corrected) and model SSH on idx."""
    obs = load_obs(f"{idx[0]:%Y-%m-%d}", f"{idx[-1]:%Y-%m-%d}")["wl_corr"].reindex(idx)
    model = load_model()
    model = model.reindex(idx) if model is not None else pd.Series(np.nan, index=idx)
    return obs, model


def two_panel(idx, obs, model, title, out_path, obs_label=OBS_LABEL, model_label=MODEL_LABEL, shade=None):
    """SSH panel and Delta SSH panel. shade = (start time, legend text) greys out the end of the window."""
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
        if shade is not None and shade[0] < idx[-1]:
            ax.axvspan(max(shade[0], idx[0]), idx[-1], color="0.92", zorder=0,
                       label=shade[1] if ax is axes[0] else None)
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
    """Hourly SSH figure for UTC dates start..end (both inclusive, YYYY-MM-DD); out(name) gives the PNG path."""
    idx = hours(start, end)
    obs, model = load_series(idx)
    two_panel(idx, obs, model, f"{TITLE} sea surface height: PFM vs tide gauge (hourly), {period}", out(f"{NAME}_ssh"))


def update_data(end):
    """Refresh the NOAA observations and extract any new model days up to end (UTC date)."""
    update_obs(end)
    extract(end=end)
