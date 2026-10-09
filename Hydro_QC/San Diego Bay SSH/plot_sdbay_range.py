#!/usr/bin/env python3
"""PFM vs observed sea surface height in San Diego Bay for any start/end dates.

Same figures as the 3-week script (listed in plot_ssh.py), saved to
plots/<name>_ssh_<start>_<end>.png and plots/<name>_ssh_lowpass_<start>_<end>.png.
Dates are UTC days, both inclusive. Not for the website. Uses the cached data; add --update to
download new observations and extract new model days first.

  python plot_sdbay_range.py --start 2026-06-01 --end 2026-08-31
  python plot_sdbay_range.py --start 2024-12-05 --end 2026-10-09 --update
"""
import argparse
import os

import pandas as pd

from plot_ssh import OUT_DIR, plot_window, update_data

PLOT_DIR = os.path.join(OUT_DIR, "plots")

# default window (UTC dates, both inclusive); --start/--end override these
START = "2026-01-01"
END = f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d}"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", default=START, help=f"Start date YYYY-MM-DD (UTC, default {START})")
    p.add_argument("--end", default=END, help="End date YYYY-MM-DD, inclusive (UTC, default today)")
    p.add_argument("--update", action="store_true", help="Download new obs and extract new model days first")
    args = p.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if end < start:
        p.error("--end must be on or after --start")
    if args.update:
        update_data(f"{end:%Y-%m-%d}")
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_window(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}",
                lambda name: os.path.join(PLOT_DIR, f"{name}_{start:%Y%m%d}_{end:%Y%m%d}.png"),
                f"{start:%Y-%m-%d} to {end:%Y-%m-%d}")


if __name__ == "__main__":
    main()
