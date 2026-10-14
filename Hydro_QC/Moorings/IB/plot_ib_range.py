#!/usr/bin/env python3
"""IB: PFM vs observed currents, temperature and salinity, any start/end dates, into plots/ (not for the website)

The code is shared with the other moorings in ../common (runner.py, plot_model_vs_obs.py).

  python plot_ib_range.py --start 2026-06-01 --end 2026-08-31
  python plot_ib_range.py --end today --duration 45
  python plot_ib_range.py --start 2025-01-01 --update
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

from runner import run_range  # noqa: E402

if __name__ == "__main__":
    run_range("IB", HERE)
