#!/usr/bin/env python3
"""PFM vs observed sea surface height at San Diego Bay (NOAA 9410170, LV4) for any start/end dates (not for the website).

Writes plots/SDBay_ssh_<start>_<end>.png (hourly, as on the website) and
plots/SDBay_ssh_lowpass_<start>_<end>.png (tides removed, PL64). Dates are UTC days, both included.
Uses the cached data; add --update to download new observations and extract new model days first.
The plotting code is in plot_ssh.py and plot_ssh_lowpass.py.

  python plot_sdbay_range.py                                   # START below to today
  python plot_sdbay_range.py --start 2026-06-01 --end 2026-08-31
  python plot_sdbay_range.py --end today --duration 45         # the last 45 days, today included
  python plot_sdbay_range.py --start 2024-12-05 --update
"""
from plot_ssh_lowpass import run_range

# default start (UTC date); --start, or --duration with --end, override it
START = "2026-01-01"

if __name__ == "__main__":
    run_range("SDBay", START)
