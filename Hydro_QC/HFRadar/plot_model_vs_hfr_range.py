#!/usr/bin/env python3
"""PFM surface currents vs HF radar 6 km and moorings for any start and end dates (not for the website).

Same figure as plot_model_vs_hfr.py (3 days x 4 panels, every 6 h, Pacific time), one page per 3 days
from --start on, into plots/Model_vs_HFR6km_<LV>_<first day>_<last day>.png. A day's row is 06:00,
12:00, 18:00 and the next day's 00:00. The last page always has 3 rows, so it can run up to 2 days past
--end.

  python plot_model_vs_hfr_range.py --start 2026-10-01 --end 2026-10-20
  python plot_model_vs_hfr_range.py --end today --duration 9          # the last 9 days, today included
  python plot_model_vs_hfr_range.py --start 2026-10-10 --level LV3 --update-moorings

Both dates are Pacific days (PDT/PST) and are included.
"""
import argparse
import datetime as DT
import os

from plot_model_vs_hfr import HERE, LEVELS, LOCAL_TZ, NROW, plot_level, update_moorings


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", help="First day to plot, YYYY-MM-DD (Pacific); or use --duration")
    p.add_argument("--end", default="today", help="Last day to plot, included, YYYY-MM-DD or 'today' (Pacific; default: today)")
    p.add_argument("--duration", type=int, help="Number of days to plot, ending on --end (both days included)")
    p.add_argument("--level", choices=LEVELS, nargs="+", default=list(LEVELS))
    p.add_argument("--update-moorings", action="store_true", help="download new mooring obs from SIO first")
    args = p.parse_args()
    last = (DT.datetime.now(LOCAL_TZ).date() if args.end == "today"
            else DT.datetime.strptime(args.end, "%Y-%m-%d").date())
    if args.duration is not None:
        if args.start:
            p.error("use --start or --duration, not both")
        if args.duration < 1:
            p.error("--duration must be at least 1 day")
        start = last - DT.timedelta(days=args.duration - 1)
    elif args.start:
        start = DT.datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        p.error("give --start or --duration")
    if last < start:
        p.error("--end is before --start")
    if args.update_moorings:
        update_moorings(args.level)
    out_dir = os.path.join(HERE, "plots")
    for lv in args.level:
        first = start
        while first <= last:                       # rows first .. first + NROW - 1
            end_day = first + DT.timedelta(days=NROW)   # its 00:00 is the last panel
            plot_level(lv, end_day, out_dir=out_dir,
                       name=f"Model_vs_HFR6km_{lv}_{first:%Y%m%d}_{end_day - DT.timedelta(days=1):%Y%m%d}")
            first = end_day


if __name__ == "__main__":
    main()
