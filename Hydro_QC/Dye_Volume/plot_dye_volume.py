#!/usr/bin/env python3
"""QC-website plot: total PTB, TJRE and combined raw-sewage (dye) volume in the LV4 domain.

Volume = sum over wet cells of dye * Hz * dx * dy (m^3), dye_01 = PTB, dye_02 = TJRE, as in
PTB_TJRE_DyeFactor/Dye_Volume.py (whose file_volume, forecast_files and gap_plan are used here).
Each hour comes from the latest LV4 forecast covering it (LV4_ocean_his_<YYYYMMDDHHMM>.nc), so
past hours are day-1 values, days without a forecast are filled from earlier forecasts, and hours
after the newest forecast start are its forecast. Volumes are cached per forecast file and record
in volume_cache.csv, so the daily run only computes the hours of new forecasts.

Times on the plot are Pacific (PDT/PST, America/Los_Angeles), as on the other QC plots; the model
times are UTC and converted.

Daily run (default): past 3 weeks up to the end of today (Pacific), the same window as the SBOO / IB
and PTB discharge QC plots, into plots_web/Dye_volume_<YYYYMMDD>_3weeks.png; older *_3weeks.png are
removed so plots_web/ only holds the newest plot. For any start and end dates use
plot_dye_volume_range.py.

  python plot_dye_volume.py                                     # past 3 weeks, for the website
  python plot_dye_volume.py --date 2026-09-30                   # 3 weeks ending on another day (old plots kept)
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/akg004/Python_files/PTB_TJRE_DyeFactor")
from Dye_Volume import file_volume, forecast_files, gap_plan  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "plots_web")
PLOT_DIR = os.path.join(HERE, "plots")
CACHE = os.path.join(HERE, "volume_cache.csv")
DAYS, LABEL = 21, "3weeks"
TZ = "America/Los_Angeles"
WORKERS = 16

# colours of PTB_TJRE_DyeFactor/plot_style.py
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_PTB, C_TJRE, C_ALL = "#2a78d6", "#eb6834", INK


def to_utc(t):
    """Naive Pacific time -> naive UTC."""
    return pd.Timestamp(t).tz_localize(TZ, ambiguous=False, nonexistent="shift_forward").tz_convert("UTC").tz_localize(None)


def to_pacific(idx):
    """Naive UTC index -> naive Pacific index."""
    return idx.tz_localize("UTC").tz_convert(TZ).tz_localize(None)


def _volume(job):
    path, recs = job
    return file_volume(path, recs)


def volumes(start, end):
    """Hourly PTB and TJRE volume (m^3) from start to end (naive UTC), latest forecast per hour."""
    files = forecast_files()
    times = pd.date_range(start, end, freq="h")
    plan, uncovered = gap_plan(times.to_numpy(), np.ones(len(times), bool), files)
    if uncovered:
        print(f"{len(uncovered)} hours not covered by any forecast")
    # cache: one row per (forecast file, record); a file that changed (size, mtime) is recomputed
    cache = pd.read_csv(CACHE, dtype={"key": str}) if os.path.isfile(CACHE) else \
        pd.DataFrame(columns=["file", "key", "record", "ptb", "tjre"])
    keys = {}
    for path in plan:
        st = os.stat(path)
        keys[path] = f"{st.st_size}:{int(st.st_mtime)}"
    current = {os.path.basename(p): k for p, k in keys.items()}
    cache = cache.loc[np.array([current.get(f, k) == k for f, k in zip(cache["file"], cache["key"])], bool)]
    have = {(f, int(r)): (a, b) for f, r, a, b in zip(cache["file"], cache["record"], cache["ptb"], cache["tjre"])}
    jobs = []
    for path, recs in plan.items():
        todo = sorted({k for k, _ in recs if (os.path.basename(path), k) not in have})
        if todo:
            jobs.append((path, todo))
    if jobs:
        print(f"computing {sum(len(r) for _, r in jobs)} hours from {len(jobs)} forecasts", flush=True)
        new = []
        with ProcessPoolExecutor(min(len(jobs), WORKERS)) as pool:
            for (path, todo), V in zip(jobs, pool.map(_volume, jobs)):
                for k, (a, b) in zip(todo, V):
                    have[os.path.basename(path), k] = (a, b)
                    new.append([os.path.basename(path), keys[path], k, a, b])
        new = pd.DataFrame(new, columns=cache.columns)
        cache = pd.concat([cache, new], ignore_index=True) if len(cache) else new
        cache.sort_values(["file", "record"]).to_csv(CACHE, index=False)
    ptb, tjre = pd.Series(np.nan, index=times), pd.Series(np.nan, index=times)
    for path, recs in plan.items():
        for k, t in recs:
            ptb[pd.Timestamp(t)], tjre[pd.Timestamp(t)] = have[os.path.basename(path), k]
    newest = max(f[1] for f in files)
    return ptb, tjre, pd.Timestamp(newest)


def plot(start, end, path):
    """Plot from start to end, naive Pacific times."""
    s_utc, e_utc = to_utc(start), to_utc(end)
    ptb, tjre, newest = volumes(s_utc, e_utc)
    total = ptb + tjre
    t = to_pacific(ptb.index)
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "axes.formatter.limits": (-3, 4), "axes.formatter.use_mathtext": True})
    days = (end - start).days
    lw = 1.5 if days <= 60 else 0.8
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(t, total, color=C_ALL, lw=lw, label="PTB + TJRE")
    ax.plot(t, ptb, color=C_PTB, lw=lw, label="PTB (dye_01)")
    ax.plot(t, tjre, color=C_TJRE, lw=lw, label="TJRE (dye_02)")
    # hours after the start of the newest forecast are forecast, not day-1 values
    t_fc = to_pacific(pd.DatetimeIndex([newest]))[0]
    if start < t_fc < end:
        ax.axvline(t_fc, color=INK2, lw=0.8, ls=":")
        ax.text(t_fc, 1, " forecast →", transform=ax.get_xaxis_transform(), va="top", ha="left",
                color=INK2, fontsize=8)
    ax.set_ylabel("Volume (m$^3$)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(start, end)
    ax.set_title(f"Raw-sewage dye volume in the LV4 domain, {start:%Y-%m-%d} to "
                 f"{(end - pd.Timedelta(seconds=1)):%Y-%m-%d}", pad=30)
    ax.set_xlabel("Date (Pacific time, PDT/PST)")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=3, fontsize=9)
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.xaxis.set_minor_locator(mdates.DayLocator() if days <= 60 else mdates.MonthLocator())
    ax.tick_params(axis="x", which="minor", length=3, color=INK2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    print(f"median volume (m3): PTB {ptb.median():.4g}, TJRE {tjre.median():.4g}, total {total.median():.4g}; "
          f"newest forecast {newest} UTC")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date", help="Last day of the 3-week plot, YYYY-MM-DD Pacific (default: today)")
    args = p.parse_args()
    today = pd.Timestamp.now(tz=TZ).tz_localize(None).normalize()
    last = pd.Timestamp(args.date) if args.date else today
    start, end = last - pd.Timedelta(days=DAYS - 1), last + pd.Timedelta(days=1)
    path = os.path.join(WEB_DIR, f"Dye_volume_{last:%Y%m%d}_{LABEL}.png")
    plot(start, end, path)
    # daily run: keep only the newest plot for the website
    if not args.date:
        for old in glob.glob(os.path.join(WEB_DIR, f"Dye_volume_{'[0-9]' * 8}_{LABEL}.png")):
            if old != path:
                os.remove(old)
                print("removed", os.path.basename(old))


if __name__ == "__main__":
    main()
