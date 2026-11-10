#!/usr/bin/env python3
"""Command-line runners shared by the mooring folders (SBOO, IB, PLOO, DelMar).

Each folder has two small scripts that call these with their group:
  plot_<group>_3weeks.py -> run_3weeks: past 3 weeks for the QC website, into <folder>/plots_web/
  plot_<group>_range.py  -> run_range:  any start/end dates, into <folder>/plots/ (not for the website)

3 weeks = the 2-week dye vs ddPCR window plus the week before it. The 3-week run first refreshes the
group's observations and extracts any new model days (add --no-update to skip), and a normal run
(without --date) deletes the folder's older *_3weeks.png so plots_web/ only holds the newest set.
The range run uses the cached data unless --update is given. Dates are Pacific days (PDT/PST), both inclusive.
"""
import argparse
import glob
import os

import pandas as pd

from plot_model_vs_obs import TZ, plot_window, update_data

DAYS = 21
LABEL = "3weeks"
RANGE_START = "2025-01-01"  # default start of the range plots


def today():
    """Today's Pacific date (naive)."""
    return pd.Timestamp.now(tz=TZ).tz_localize(None).normalize()


def run_3weeks(group, folder):
    p = argparse.ArgumentParser(description=f"{group}: PFM vs observed currents, T and S, past 3 weeks (QC website)")
    p.add_argument("--date", help="Last day to plot, YYYY-MM-DD Pacific (default: today)")
    p.add_argument("--no-update", action="store_true", help="Skip downloading obs and extracting model")
    args = p.parse_args()
    end = pd.Timestamp(args.date) if args.date else today()
    start = end - pd.Timedelta(days=DAYS - 1)
    if not args.no_update:
        update_data(f"{end:%Y-%m-%d}", group)
    web_dir = os.path.join(folder, "plots_web")
    os.makedirs(web_dir, exist_ok=True)
    tag = f"{end:%Y%m%d}_{LABEL}"
    written = []
    out = lambda name: written.append(os.path.join(web_dir, f"{name}_{tag}.png")) or written[-1]
    plot_window(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}", out, f"past 3 weeks ({start:%Y-%m-%d} to {end:%Y-%m-%d})", group)
    # daily run: keep only the newest set for the website
    if not args.date:
        for old in glob.glob(os.path.join(web_dir, f"*_{'[0-9]' * 8}_{LABEL}.png")):
            if old not in written:
                os.remove(old)
                print("removed", os.path.basename(old))


def run_range(group, folder):
    p = argparse.ArgumentParser(description=f"{group}: PFM vs observed currents, T and S for any dates")
    p.add_argument("--start", help=f"Start date YYYY-MM-DD (Pacific, default {RANGE_START}); not with --duration")
    p.add_argument("--end", default="today", help="End date YYYY-MM-DD or 'today', inclusive (Pacific, default today)")
    p.add_argument("--duration", type=int, help="Number of days to plot, ending on --end (both days included)")
    p.add_argument("--update", action="store_true", help="Download new obs and extract new model days first")
    args = p.parse_args()
    end = today() if args.end == "today" else pd.Timestamp(args.end)
    if args.duration is not None:
        if args.start:
            p.error("use --start or --duration, not both")
        if args.duration < 1:
            p.error("--duration must be at least 1 day")
        start = end - pd.Timedelta(days=args.duration - 1)
    else:
        start = pd.Timestamp(args.start or RANGE_START)
        if end < start:
            p.error("--end must be on or after --start")
    if args.update:
        update_data(f"{end:%Y-%m-%d}", group)
    plot_dir = os.path.join(folder, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    tag = f"{start:%Y%m%d}_{end:%Y%m%d}"
    plot_window(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}", lambda name: os.path.join(plot_dir, f"{name}_{tag}.png"),
                f"{start:%Y-%m-%d} to {end:%Y-%m-%d}", group)
