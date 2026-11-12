#!/usr/bin/env python3
"""3-panel PFM dye (day1/day5) vs ddPCR Enterococcus plot for any start/end date.

Same layout as plot_dye_ddpcr.py. Dates are Pacific local days, both inclusive.
Edit START/END below and run, or override on the command line:
  python plot_dye_ddpcr_range.py --start 2026-06-01 --end 2026-08-31
  python plot_dye_ddpcr_range.py --end today --duration 14     # the last 14 days, today included
--end takes a date or 'today'; --duration N plots the N days ending on --end (not with --start).
With --end today the plot also shows the next 5 days of forecast, as the daily plot does.
Output: plots/dye_ddpcr_<start>_<end>.png (or --out PATH), and the dye vs ddPCR scatter of the same window
next to it, <name>_scatter.png.
Dye and ddPCR come from the shared cache (see plot_dye_ddpcr.py). Run make_forecast_csv.py first for
the newest forecast; --update downloads new ddPCR samples first.
"""
import argparse
import os

import pandas as pd

from plot_dye_ddpcr import FUTURE_DAYS, PLOT_DIR, TZ, plot_scatter, plot_window
from update_cache import update_ddpcr

# default window (Pacific dates, both inclusive); --start/--end override these
START = "2026-07-01"
END = "2026-07-15"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", help=f"Start date YYYY-MM-DD (Pacific, default {START}); not with --duration")
    p.add_argument("--end", default=END, help=f"End date YYYY-MM-DD or 'today', inclusive (Pacific, default {END})")
    p.add_argument("--duration", type=int, help="Number of days to plot, ending on --end (both days included)")
    p.add_argument("--out", help="Output PNG path (default: plots/dye_ddpcr_<start>_<end>.png)")
    p.add_argument("--update", action="store_true", help="Download new ddPCR samples into the cache first")
    args = p.parse_args()
    if args.update:
        os.umask(0o002)  # the cache is shared with the group
        update_ddpcr()

    # calendar-day arithmetic on plain dates, then Pacific time (so DST changes don't shift midnight)
    end = pd.Timestamp.now(TZ).tz_localize(None).normalize() if args.end == "today" else pd.Timestamp(args.end)
    if args.duration is not None:
        if args.start:
            p.error("use --start or --duration, not both")
        if args.duration < 1:
            p.error("--duration must be at least 1 day")
        start = end - pd.Timedelta(days=args.duration - 1)
    else:
        start = pd.Timestamp(args.start or START)
    start, end = start.tz_localize(TZ), end.tz_localize(TZ)
    if end < start:
        p.error("--end must be on or after --start")
    t1 = end + pd.Timedelta(days=1)  # include the whole end day
    # --end today: also the forecast days, as in the daily plot (the scatter uses past samples only)
    ahead = f", and the {FUTURE_DAYS}-day forecast" if args.end == "today" else ""
    if ahead:
        t1 += pd.Timedelta(days=FUTURE_DAYS)

    out_path = args.out or os.path.join(PLOT_DIR, f"dye_ddpcr_{start:%Y%m%d}_{end:%Y%m%d}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    title = f"PFM dye forecast vs ddPCR Enterococcus — {start:%Y-%m-%d} to {end:%Y-%m-%d}{ahead}"
    plot_window(start, t1, out_path, title)
    plot_scatter(start, t1, os.path.splitext(out_path)[0] + "_scatter.png",
                 f"PFM dye forecast vs ddPCR Enterococcus, scatter — {start:%Y-%m-%d} to {end:%Y-%m-%d}")


if __name__ == "__main__":
    main()
