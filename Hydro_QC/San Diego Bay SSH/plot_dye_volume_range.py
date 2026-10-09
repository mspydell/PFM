#!/usr/bin/env python3
"""PTB, TJRE and combined raw-sewage (dye) volume in the LV4 domain, for any start and end dates.

Same plot as plot_dye_volume.py (the daily QC-website plot), for a chosen range, into plots/
(not for the website). Hours already in volume_cache.csv are not recomputed.

  python plot_dye_volume_range.py --start 2026-01-01 --end 2026-10-08
  python plot_dye_volume_range.py --start 2025-01-01                  # to today
  python plot_dye_volume_range.py --start 2026-07-01 --end 2026-07-31 --out july.png

Both dates are Pacific days (PDT/PST) and are included; times on the plot are Pacific.
"""
import argparse
import os

import pandas as pd

from plot_dye_volume import PLOT_DIR, TZ, plot


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", required=True, help="First day to plot, YYYY-MM-DD (Pacific)")
    p.add_argument("--end", help="Last day to plot, included, YYYY-MM-DD (Pacific; default: today)")
    p.add_argument("--out", help="Output PNG (default: plots/Dye_volume_<start>_<end>.png)")
    args = p.parse_args()
    start = pd.Timestamp(args.start)
    last = pd.Timestamp(args.end) if args.end else pd.Timestamp.now(tz=TZ).tz_localize(None).normalize()
    if last < start:
        p.error("--end is before --start")
    out = args.out or os.path.join(PLOT_DIR, f"Dye_volume_{start:%Y%m%d}_{last:%Y%m%d}.png")
    plot(start, last + pd.Timedelta(days=1), out)


if __name__ == "__main__":
    main()
