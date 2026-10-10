#!/usr/bin/env python3
"""3-panel PFM dye (day1/day5) vs ddPCR Enterococcus plot for any start/end date.

Same layout as plot_dye_ddpcr.py. Dates are Pacific local days, both inclusive.
Edit START/END below and run, or override on the command line:
  python plot_dye_ddpcr_range.py --start 2026-06-01 --end 2026-08-31
  python plot_dye_ddpcr_range.py --end today --duration 14
Output: plots/dye_ddpcr_<start>_<end>.png (or --out PATH).
Dye comes from the newest sites_dye_tot_day1_day5_*.csv (run make_forecast_csv.py first
for current data); ddPCR is downloaded for the window.
"""
import argparse
import os

import pandas as pd

from plot_dye_ddpcr import PLOT_DIR, TZ, plot_window

# default window (Pacific dates, both inclusive); --start/--end override these
START = "2026-07-01"
END = "2026-07-15"


def parse_date(date_str, field_name):
    """Parse a date string as YYYY-MM-DD or 'today' in Pacific timezone."""
    if date_str.strip().lower() == "today":
        return pd.Timestamp.now(TZ).normalize()
    try:
        return pd.Timestamp(date_str).tz_localize(TZ)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid date for {field_name} '{date_str}': {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", help=f"Start date YYYY-MM-DD or 'today' (Pacific, default {START})")
    p.add_argument("--end", help=f"End date YYYY-MM-DD or 'today', inclusive (Pacific, default {END})")
    p.add_argument("--duration", type=int, help="Number of days back from --end (cannot be used with --start)")
    p.add_argument("--out", help="Output PNG path (default: plots/dye_ddpcr_<start>_<end>.png)")
    args = p.parse_args()

    if args.start and args.duration:
        p.error("Specify either --start or --duration, not both")

    if args.duration is not None and args.duration <= 0:
        p.error("--duration must be a positive integer")

    # Determine end date
    if args.end is not None:
        try:
            end = parse_date(args.end, "--end")
        except argparse.ArgumentTypeError as e:
            p.error(str(e))
    elif args.duration is not None:
        # Default to today if --duration is given without an explicit --end
        end = pd.Timestamp.now(TZ).normalize()
    else:
        end = parse_date(END, "--end")

    # Determine start date
    if args.duration is not None:
        start = end - pd.Timedelta(days=args.duration)
    elif args.start is not None:
        try:
            start = parse_date(args.start, "--start")
        except argparse.ArgumentTypeError as e:
            p.error(str(e))
    else:
        start = parse_date(START, "--start")

    if end < start:
        p.error("--end must be on or after --start")
    t1 = end + pd.Timedelta(days=5)  # include the whole end day

    out_path = args.out or os.path.join(PLOT_DIR, f"dye_ddpcr_{start:%Y%m%d}_{end:%Y%m%d}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    title = f"PFM dye forecast vs ddPCR Enterococcus — {start:%Y-%m-%d} to {end:%Y-%m-%d}"
    plot_window(start, t1, out_path, title)


if __name__ == "__main__":
    main()
