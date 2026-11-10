#!/usr/bin/env python3
"""DelMar: PFM vs observed currents, temperature and salinity, past 3 weeks for the QC website, into plots_web/

The code is shared with the other moorings in ../common (runner.py, plot_model_vs_obs.py).

  python plot_delmar_3weeks.py                    # update data, plot the past 3 weeks
  python plot_delmar_3weeks.py --no-update        # cached data only
  python plot_delmar_3weeks.py --date 2026-09-30  # end on another day (old plots kept)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

from runner import run_3weeks  # noqa: E402

if __name__ == "__main__":
    run_3weeks("DelMar", HERE)
