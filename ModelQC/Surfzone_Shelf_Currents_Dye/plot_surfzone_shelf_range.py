#!/usr/bin/env python3
"""Surfzone and shelf currents of the PFM LV4 model for any start and end dates.

Same figures as plot_surfzone_shelf.py (the daily QC-website plot, WEB_VARIANTS), for a chosen range, into
plots/ (not for the website). The first long range computes the history files not yet in the cache
(about 1 min per file, --workers in parallel); later runs read the cache.

  python plot_surfzone_shelf_range.py --start 2026-01-01 --end 2026-10-08
  python plot_surfzone_shelf_range.py --start 2024-12-05 --workers 12     # whole archive to today
  python plot_surfzone_shelf_range.py --end today --duration 45          # the last 45 days, today included
  python plot_surfzone_shelf_range.py --start 2026-07-01 --end 2026-07-31 --out july.png

Both dates are Pacific days (PDT/PST) and are included; times on the plot are Pacific.
"""
import argparse
import os

import pandas as pd

from plot_surfzone_shelf import HERE, TZ, WEB_VARIANTS, plot


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", help="First day to plot, YYYY-MM-DD (Pacific); or use --duration")
    p.add_argument("--end", default="today", help="Last day to plot, included, YYYY-MM-DD or 'today' (Pacific; default: today)")
    p.add_argument("--duration", type=int, help="Number of days to plot, ending on --end (both days included)")
    p.add_argument("--workers", type=int, default=4, help="History files read in parallel (default 4)")
    p.add_argument("--out", help="Output PNG (default: plots/SurfzoneShelf_<start>_<end>.png)")
    args = p.parse_args()
    last = pd.Timestamp.now(tz=TZ).tz_localize(None).normalize() if args.end == "today" else pd.Timestamp(args.end)
    if args.duration is not None:
        if args.start:
            p.error("use --start or --duration, not both")
        if args.duration < 1:
            p.error("--duration must be at least 1 day")
        start = last - pd.Timedelta(days=args.duration - 1)
    elif args.start:
        start = pd.Timestamp(args.start)
    else:
        p.error("give --start or --duration")
    if last < start:
        p.error("--end is before --start")
    out = args.out or os.path.join(HERE, "plots", f"SurfzoneShelf_{start:%Y%m%d}_{last:%Y%m%d}.png")
    plot(start, last + pd.Timedelta(days=1), out, args.workers, only=WEB_VARIANTS)


if __name__ == "__main__":
    main()
