#!/usr/bin/env python3
"""Daily QC-website plots: PFM vs observed sea surface height in San Diego Bay, past 3 weeks.

Each run refreshes the NOAA observations and extracts any new model days, then draws the past
3 weeks up to the end of today (UTC) into plots_web/<name>_ssh_<YYYYMMDD>_3weeks.png
(hourly SSH and Delta SSH, plot_ssh.py). Older *_3weeks.png files are removed, so plots_web/
only holds the newest plot. No low-pass filtering here (that is in the range script only).

  python plot_sdbay_3weeks.py                    # update data, plot the past 3 weeks
  python plot_sdbay_3weeks.py --no-update        # plot from the cached data only
  python plot_sdbay_3weeks.py --date 2026-09-30  # end on another day (old plots are kept)
"""
import argparse
import glob
import os

import pandas as pd

from plot_ssh import NAME, OUT_DIR, plot_window, update_data

WEB_DIR = os.path.join(OUT_DIR, "plots_web")
DAYS = 21
LABEL = "3weeks"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date", help="Last day to plot, YYYY-MM-DD UTC (default: today)")
    p.add_argument("--no-update", action="store_true", help="Skip downloading obs and extracting model")
    args = p.parse_args()
    end = pd.Timestamp(args.date) if args.date else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    start = end - pd.Timedelta(days=DAYS - 1)
    if not args.no_update:
        update_data(f"{end:%Y-%m-%d}")
    os.makedirs(WEB_DIR, exist_ok=True)
    written = []
    out = lambda name: written.append(os.path.join(WEB_DIR, f"{name}_{end:%Y%m%d}_{LABEL}.png")) or written[-1]
    plot_window(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}", out, f"past 3 weeks ({start:%Y-%m-%d} to {end:%Y-%m-%d})")
    if not args.date:  # daily run: keep only the newest set
        for old in glob.glob(os.path.join(WEB_DIR, f"{NAME}_ssh*_{'[0-9]' * 8}_{LABEL}.png")):
            if old not in written:
                os.remove(old)
                print("removed", os.path.basename(old))


if __name__ == "__main__":
    main()
