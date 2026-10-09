#!/usr/bin/env python3
"""PFM vs observed sea surface height in San Diego Bay for any start/end dates.

The 3-week script's figure (plot_ssh.py) plus the tides-removed one (plot_ssh_lowpass.py), saved to
plots/<name>_ssh_<start>_<end>.png and plots/<name>_ssh_lowpass_<start>_<end>.png.
Dates are UTC days, both inclusive. Not for the website. Uses the cached data; add --update to
download new observations and extract new model days first.

  python plot_sdbay_range.py --start 2026-06-01 --end 2026-08-31
  python plot_sdbay_range.py --end today --duration 45      # the last 45 days, today included
  python plot_sdbay_range.py --start 2024-12-05 --end 2026-10-09 --update
"""
import argparse
import os

import pandas as pd

from plot_ssh import OUT_DIR, plot_window, update_data
from plot_ssh_lowpass import plot_lowpass_window

PLOT_DIR = os.path.join(OUT_DIR, "plots")

# default start (UTC date); --start, or --duration with --end, override it
START = "2026-01-01"


def window(p, args):
    """(start, end) dates from --start/--end/--duration; --end may be 'today' (UTC)."""
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() if args.end == "today" else pd.Timestamp(args.end)
    if args.duration is not None:
        if args.start:
            p.error("use --start or --duration, not both")
        if args.duration < 1:
            p.error("--duration must be at least 1 day")
        return end - pd.Timedelta(days=args.duration - 1), end
    start = pd.Timestamp(args.start or START)
    if end < start:
        p.error("--end must be on or after --start")
    return start, end


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", help=f"Start date YYYY-MM-DD (UTC, default {START}); not with --duration")
    p.add_argument("--end", default="today", help="End date YYYY-MM-DD or 'today', inclusive (UTC, default today)")
    p.add_argument("--duration", type=int, help="Number of days to plot, ending on --end (both days included)")
    p.add_argument("--update", action="store_true", help="Download new obs and extract new model days first")
    args = p.parse_args()
    start, end = window(p, args)
    if args.update:
        update_data(f"{end:%Y-%m-%d}")
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = lambda name: os.path.join(PLOT_DIR, f"{name}_{start:%Y%m%d}_{end:%Y%m%d}.png")
    period = f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"
    plot_window(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}", out, period)          # hourly SSH
    plot_lowpass_window(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}", out, period)  # tides removed


if __name__ == "__main__":
    main()
