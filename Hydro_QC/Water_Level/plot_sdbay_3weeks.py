#!/usr/bin/env python3
"""QC-website plot: PFM vs observed sea surface height at San Diego Bay (NOAA 9410170, LV4), past 3 weeks.

Refreshes the NOAA observations and extracts any new model days, then draws the past 3 weeks up to
the end of today (UTC) into plots_web/SDBay_ssh_<YYYYMMDD>_3weeks.png and deletes this station's
older *_3weeks.png. No low-pass filtering. The plotting code is in plot_ssh.py.

  python plot_sdbay_3weeks.py                    # update data, plot the past 3 weeks
  python plot_sdbay_3weeks.py --no-update        # plot from the cached data only
  python plot_sdbay_3weeks.py --date 2026-09-30  # end on another day (older plots are kept)
"""
from plot_ssh import run_3weeks

if __name__ == "__main__":
    run_3weeks("SDBay")
