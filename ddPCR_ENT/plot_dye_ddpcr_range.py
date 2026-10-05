#!/usr/bin/env python3
"""3-panel PFM dye (day1/day5) vs ddPCR Enterococcus plot for any start/end date.

Same layout as plot_dye_ddpcr.py. Dates are Pacific local days, both inclusive.
Edit START/END below and run, or override on the command line:
  python plot_dye_ddpcr_range.py --start 2026-06-01 --end 2026-08-31
Output: plots/dye_ddpcr_<start>_<end>.png (or --out PATH).
Dye comes from the newest sites_dye_tot_day1_day5_*.csv (run make_forecast_csv.py first
for current data); ddPCR is downloaded for the window
"""
import argparse
import os

import pandas as pd

from plot_dye_ddpcr import PLOT_DIR, TZ, plot_window

# default window (Pacific dates, both inclusive); --start/--end override these
START = "2026-07-01"
END = "2026-07-15"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", default=START, help=f"Start date YYYY-MM-DD (Pacific, default {START})")
    p.add_argument("--end", default=END, help=f"End date YYYY-MM-DD, inclusive (Pacific, default {END})")
    p.add_argument("--out", help="Output PNG path (default: plots/dye_ddpcr_<start>_<end>.png)")
    args = p.parse_args()

    start = pd.Timestamp(args.start).tz_localize(TZ)
    end = pd.Timestamp(args.end).tz_localize(TZ)
    if end < start:
        p.error("--end must be on or after --start")
    t1 = end + pd.Timedelta(days=1)  # include the whole end day

    out_path = args.out or os.path.join(PLOT_DIR, f"dye_ddpcr_{start:%Y%m%d}_{end:%Y%m%d}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    title = f"PFM dye forecast vs ddPCR Enterococcus — {start:%Y-%m-%d} to {end:%Y-%m-%d}"
    plot_window(start, t1, out_path, title)


if __name__ == "__main__":
    main()
