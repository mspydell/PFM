#!/usr/bin/env python3
"""Shared plotting code: PFM (day-1 forecast) vs observed sea surface height at a NOAA tide gauge.
Used by plot_<station>_3weeks.py (QC website; run_3weeks below) and plot_<station>_range.py.

Figure <station>_ssh, laid out like Plot_SSH.m (PFM_Master_Code_For_Plots.m, PFM branch):
  panel 1  hourly observed (inverse-barometer corrected, MSL datum) and model zeta
  panel 2  Delta SSH = observed - model (y axis fixed at -0.5 to 0.5 m)
Times on the plots and the date windows are Pacific (PDT/PST, America/Los_Angeles); data and caches
stay UTC and are converted for plotting. The tides-removed version (not for the website) is in plot_ssh_lowpass.py.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from download_obs import load_obs, update as update_obs
from extract_model import extract, latest_forecast, load_model
from stations import STATIONS

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OBS_COLOR, MODEL_COLOR, DIFF_COLOR = "#2a78d6", "#eb6834", "0.25"
DIFF_YLIM = (-0.5, 0.5)  # m, Delta SSH panels
TZ = "America/Los_Angeles"
matplotlib.rcParams["timezone"] = TZ  # date axes in Pacific time (the data stay tz-aware UTC)


def labels(station):
    st = STATIONS[station]
    return (f"Observed, NOAA {st['noaa']} (inverse-barometer corrected, MSL)",
            f"PFM {st['level']} day 1 (zeta at nearest grid point)")


def hours(start, end):
    """Hourly UTC index covering the Pacific dates start..end (both inclusive)."""
    first = pd.Timestamp(start).tz_localize(TZ)
    last = (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize(TZ) - pd.Timedelta(hours=1)
    return pd.date_range(first.tz_convert("UTC"), last.tz_convert("UTC"), freq="h")


def today():
    """Today's Pacific date (naive)."""
    return pd.Timestamp.now(tz=TZ).tz_localize(None).normalize()


def load_series(station, idx):
    """Hourly observed (corrected) and model SSH on idx."""
    obs = load_obs(station, f"{idx[0]:%Y-%m-%d}", f"{idx[-1]:%Y-%m-%d}")["wl_corr"].reindex(idx)
    model = load_model(station)
    model = model.reindex(idx) if model is not None else pd.Series(np.nan, index=idx)
    return obs, model


def date_window(p, args, default_start):
    """(start, end) dates from --start/--end/--duration; --end may be 'today' (Pacific dates)."""
    end = today() if args.end == "today" else pd.Timestamp(args.end)
    if args.duration is not None:
        if args.start:
            p.error("use --start or --duration, not both")
        if args.duration < 1:
            p.error("--duration must be at least 1 day")
        return end - pd.Timedelta(days=args.duration - 1), end
    start = pd.Timestamp(args.start or default_start)
    if end < start:
        p.error("--end must be on or after --start")
    return start, end


def date_ticks(ax, span):
    """A labelled tick each (Pacific) day up to 45 days: day numbers, the month name where it starts;
    longer windows weekly or monthly labels, with a small tick each day up to 60 days."""
    if span <= 45:
        loc = mdates.DayLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc, show_offset=False))
        return
    loc, fmt = ((mdates.MonthLocator(), "%b\n%Y") if span > 200 else (mdates.MonthLocator(), "%b %Y") if span > 60
                else (mdates.WeekdayLocator(byweekday=mdates.MO), "%b %d"))
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    if span <= 60:
        ax.xaxis.set_minor_locator(mdates.DayLocator())


def two_panel(idx, obs, model, title, out_path, obs_label, model_label, shade=None, split=None):
    """SSH panel and Delta SSH panel. shade = (start time, legend text) greys out the end of the window.
    split: the model is solid up to this time and dashed after it (forecast), with a dotted line there."""
    diff = obs - model
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    ax = axes[0]
    ax.plot(idx, obs, color=OBS_COLOR, lw=0.8, label=obs_label)
    if split is None:
        ax.plot(idx, model, color=MODEL_COLOR, lw=0.8, label=model_label)
    else:
        ax.plot(model[:split].index, model[:split], color=MODEL_COLOR, lw=0.8, label=model_label)
        ax.plot(model[split:].index, model[split:], color=MODEL_COLOR, lw=0.8, ls="--")
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
        if split is not None:
            ax.axvline(split, color="0.4", lw=0.8, ls=":")          # now: forecast to the right
    handles = axes[0].get_legend_handles_labels()[0]
    if split is not None:
        handles.append(Line2D([], [], color=MODEL_COLOR, lw=0.8, ls="--", label="PFM forecast, next days (dashed)"))
    axes[0].legend(handles=handles, loc="upper left", fontsize=8, frameon=False, ncol=3)

    date_ticks(axes[-1], (idx[-1] - idx[0]).days)
    axes[-1].set_xlim(idx[0], idx[-1])
    axes[-1].set_xlabel("Date (Pacific time, PDT/PST)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


def plot_window(station, start, end, out, period):
    """Hourly SSH figure for Pacific dates start..end (both inclusive, YYYY-MM-DD); out(name) gives the PNG path.
    The model runs on past the window to the end of the latest forecast issued by then (its next ~5 days),
    dashed after now (or after the window's end, for an older window); observations stop there too."""
    idx = hours(start, end)
    obs, model = load_series(station, idx)
    title = f"{STATIONS[station]['title']} sea surface height: PFM vs tide gauge (hourly), {period}"
    fc = latest_forecast(station, idx[-1].tz_localize(None).normalize())   # UTC date of the last hour
    split = None
    if fc is not None and fc.index[-1] > idx[-1]:
        split = min(pd.Timestamp.now(tz="UTC").floor("h"), idx[-1] + pd.Timedelta(hours=1))
        idx = pd.date_range(idx[0], fc.index[-1], freq="h")
        # day 1 up to split, then the latest forecast; split hour in both so the lines join
        model = model.reindex(idx)
        model[model.index > split] = np.nan
        model = model.fillna(fc[fc.index >= split].reindex(idx))
        obs = obs.reindex(idx)
        title += f" + forecast to {idx[-1].tz_convert(TZ):%Y-%m-%d %H:%M %Z}"
    two_panel(idx, obs, model, title, out(f"{station}_ssh"), *labels(station), split=split)


def update_data(station, end):
    """Refresh the NOAA observations and extract any new model days up to the Pacific date end
    (its evening is the next UTC day, so one UTC day further)."""
    end = f"{pd.Timestamp(end) + pd.Timedelta(days=1):%Y-%m-%d}"
    update_obs(station, end)
    extract(station, end=end)


def run_3weeks(station, days=21, label="3weeks"):
    """Command-line entry point of plot_<station>_3weeks.py: refresh the data, then draw the past 3 weeks
    up to the end of today (Pacific) into plots_web/<station>_ssh_<YYYYMMDD>_3weeks.png and delete the
    station's older *_3weeks.png (unless --date is given)."""
    p = argparse.ArgumentParser(description=f"QC-website plot: PFM vs observed SSH at {STATIONS[station]['title']}, past 3 weeks")
    p.add_argument("--date", help="Last day to plot, YYYY-MM-DD Pacific (default: today)")
    p.add_argument("--no-update", action="store_true", help="Skip downloading obs and extracting model")
    args = p.parse_args()
    end = pd.Timestamp(args.date) if args.date else today()
    start = end - pd.Timedelta(days=days - 1)
    if not args.no_update:
        update_data(station, f"{end:%Y-%m-%d}")
    web_dir = os.path.join(OUT_DIR, "plots_web")
    os.makedirs(web_dir, exist_ok=True)
    out = os.path.join(web_dir, f"{station}_ssh_{end:%Y%m%d}_{label}.png")
    plot_window(station, f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}", lambda name: out,
                f"past 3 weeks ({start:%Y-%m-%d} to {end:%Y-%m-%d})")
    if not args.date:  # keep only the newest plot
        for old in glob.glob(os.path.join(web_dir, f"{station}_ssh_{'[0-9]' * 8}_{label}.png")):
            if old != out:
                os.remove(old)
                print("removed", os.path.basename(old))
