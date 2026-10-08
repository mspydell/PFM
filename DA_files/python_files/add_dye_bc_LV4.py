#!/usr/bin/env python3
"""
add_dye_bc_LV4.py

Add zero-filled dye boundary variables (dye_XX_{south,north,west}) to an
LV4 BC file.  LV4's ROMS runtime is compiled with DYE_01 + DYE_02 tracers
and expects these BC vars even under RadNud (which pulls interior values
toward the boundary value — zero here means no dye flux at the boundary).

We generate BC files from LV3 (which has no dyes), so this patcher runs
after parent_history_to_BC.py to add the missing slots.

CLI:
    add_dye_bc_LV4.py <path-to-BC-nc>
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import netCDF4 as nc


BOUNDARIES = ("south", "north", "west")  # LV4 east boundary is closed
DYE_TRACERS = ("01", "02")


def add_dye_vars(bc_fn: Path):
    """PFM convention: dye_time = [start_day, end_day] (2 records), fields
    all zeros.  Sufficient for ROMS's multifile time-coverage check and
    keeps the file small (2 vs 25 time records)."""
    with nc.Dataset(bc_fn, "a") as d:
        # bracket dye_time to the temp_time range (2 records)
        src_time = "temp_time" if "temp_time" in d.variables else "time"
        t = np.asarray(d.variables[src_time][:])
        t0, t1 = float(t.min()), float(t.max())
        if "dye_time" not in d.dimensions:
            d.createDimension("dye_time", 2)
        if "dye_time" not in d.variables:
            tv = d.createVariable("dye_time", "f8", ("dye_time",))
            tv.units = "days"
            tv[:] = np.array([t0, t1], dtype="f8")
        added = 0
        for tag in DYE_TRACERS:
            for bnd in BOUNDARIES:
                vname = f"dye_{bnd}_{tag}"
                if vname in d.variables:
                    continue
                # match shape of temp_<bnd>: (temp_time, s_rho, xr|er)
                temp_var = d.variables[f"temp_{bnd}"]
                new_dims = tuple("dye_time" if x == "temp_time" else x
                                 for x in temp_var.dimensions)
                v = d.createVariable(vname, "f8", new_dims,
                                     fill_value=np.float64(0.0))
                v[:] = 0.0
                added += 1
        return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bc_file", type=Path)
    a = ap.parse_args()
    n = add_dye_vars(a.bc_file)
    print(f"  {a.bc_file.name}: added {n} dye BC vars (zeros)")


if __name__ == "__main__":
    main()
