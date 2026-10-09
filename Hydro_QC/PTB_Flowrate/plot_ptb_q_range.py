#!/usr/bin/env python3
"""PTB discharge, observed Punta Bandera flow vs the PFM model forcing, for any start and end dates.

Same plot as plot_ptb_q.py (the daily QC-website plot), for a chosen range, into plots/
(not for the website).

  python plot_ptb_q_range.py --start 2026-01-01 --end 2026-10-08
  python plot_ptb_q_range.py --start 2025-01-01                  # to today
  python plot_ptb_q_range.py --start 2026-07-01 --end 2026-07-31 --out july.png

Both dates are UTC days and are included.
"""
import argparse
import os

import pandas as pd

from plot_ptb_q import HERE, plot


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", required=True, help="First day to plot, YYYY-MM-DD (UTC)")
    p.add_argument("--end", help="Last day to plot, included, YYYY-MM-DD (UTC; default: today)")
    p.add_argument("--out", help="Output PNG (default: plots/PTB_discharge_<start>_<end>.png)")
    args = p.parse_args()
    start = pd.Timestamp(args.start)
    last = pd.Timestamp(args.end) if args.end else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    if last < start:
        p.error("--end is before --start")
    out = args.out or os.path.join(HERE, "plots", f"PTB_discharge_{start:%Y%m%d}_{last:%Y%m%d}.png")
    plot(start, last + pd.Timedelta(days=1), out)


if __name__ == "__main__":
    main()
