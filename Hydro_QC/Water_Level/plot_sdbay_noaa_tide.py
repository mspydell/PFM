#!/usr/bin/env python3
"""NOAA measured vs predicted water level at San Diego Bay (NOAA 9410170, LV4), and their difference (no model).

Writes plots/SDBay_noaa_tide_<start>_<end>.png. Dates are UTC days, both included; default start
2025-01-01. Uses the cache; add --update to download first. The plotting code is in noaa_tide.py.

  python plot_sdbay_noaa_tide.py                              # 2025-01-01 to today
  python plot_sdbay_noaa_tide.py --end today --duration 90
  python plot_sdbay_noaa_tide.py --start 2026-06-01 --end 2026-09-30 --update
"""
from noaa_tide import run

if __name__ == "__main__":
    run("SDBay")
