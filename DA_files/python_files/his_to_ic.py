#!/usr/bin/env python3
"""
his_to_ic.py

Extract one time slice from a ROMS `his` file and write it as an initial
condition NetCDF suitable for I4D-Var cold start.

Notes:
  - WET_DRY is banned under I4DVAR (memory: fourdvar_wet_dry_mismatch.md).
    If the source his file has wetdry_mask_rho, use --sanitize to strip the
    dried-cell bookkeeping (-h+Dcrit) that would otherwise leak into ROMS
    prsgrd32 and blow up the run.
  - The DA driver writes AKv/AKt/AKs slots back into ININAME between outer
    loops.  This utility does NOT add those slots, so IC files it produces
    are ONLY suitable for a day-1 cold start.  Chained cycles must use the
    prior day's ocean_dai.nc, which already contains those slots.

CLI:
    python his_to_ic.py --his LV1_ocean_his_202603100000.nc \\
                        --time 2026-03-10T00:00:00 \\
                        --out LV1_IC_20260310.nc \\
                        [--sanitize]
"""
from __future__ import annotations
import argparse
import datetime as DT
import re
from pathlib import Path

import numpy as np
import netCDF4 as nc


IC_VARS_2D = ["zeta", "ubar", "vbar"]
IC_VARS_3D = ["u", "v", "temp", "salt"]
# Passive tracers are carried through when the source has them.  LV4 runs
# coawstM with ANA_PASSIVE UNDEF so that IC-file dye is honoured rather than
# zeroed at t=0 -- an IC without these fails there.  LV1/LV2/LV3 have no dye, so
# this is a no-op for them.
IC_VARS_3D_OPT_RE = r"^dye_\d+$"
COPY_VARS  = ["h", "lat_rho", "lon_rho", "lat_u", "lon_u", "lat_v", "lon_v",
              "mask_rho", "mask_u", "mask_v",
              "Vtransform", "Vstretching", "theta_s", "theta_b", "Tcline",
              "hc", "Cs_r", "Cs_w", "s_rho", "s_w"]


def _time_index(dh, t_target: DT.datetime) -> int:
    t = nc.num2date(dh.variables["ocean_time"][:], dh.variables["ocean_time"].units,
                    only_use_python_datetimes=True, only_use_cftime_datetimes=False)
    t = np.array([DT.datetime(x.year, x.month, x.day, x.hour, x.minute, x.second)
                  for x in t])
    d = np.abs(t - t_target)
    k = int(np.argmin(d))
    if d[k] > DT.timedelta(minutes=30):
        raise SystemExit(f"nearest his time {t[k]} is {d[k]} from requested {t_target}")
    return k


def build_ic(his_path: str, t_target: DT.datetime, out_path: str, sanitize: bool = False):
    src = nc.Dataset(his_path)
    k = _time_index(src, t_target)

    dst = nc.Dataset(out_path, "w", format="NETCDF4")
    # spatial dims (copied as-is)
    for name in ("xi_rho", "xi_u", "xi_v", "eta_rho", "eta_u", "eta_v",
                 "s_rho", "s_w"):
        if name in src.dimensions:
            dst.createDimension(name, src.dimensions[name].size)
    # ocean_time must be UNLIMITED — ROMS DA appends adjusted-IC records here
    dst.createDimension("ocean_time", None)
    # boundary + IorJ dims for the *_obc slots that DA writes back on
    # adjust-boundary cycles.  Both are required by ROMS's def_ini.F even for
    # IC-only runs when Lobc(:,is*,ng) is TRUE anywhere.
    if "boundary" not in dst.dimensions:
        dst.createDimension("boundary", 4)
    if "IorJ" not in dst.dimensions:
        IorJ = max(src.dimensions["xi_rho"].size, src.dimensions["eta_rho"].size)
        dst.createDimension("IorJ", IorJ)

    # ocean_time
    tv = dst.createVariable("ocean_time", "f8", ("ocean_time",))
    tv.units = src.variables["ocean_time"].units
    tv[:] = src.variables["ocean_time"][k:k+1]

    # copy invariants
    for v in COPY_VARS:
        if v in src.variables:
            sv = src.variables[v]
            dv = dst.createVariable(v, sv.dtype, sv.dimensions)
            dv[:] = sv[:]

    # state 2D
    for v in IC_VARS_2D:
        sv = src.variables[v]
        dv = dst.createVariable(v, sv.dtype, ("ocean_time",) + sv.dimensions[1:])
        arr = np.array(sv[k, ...])
        dv[0, ...] = arr
    # state 3D
    extra3d = [v for v in src.variables if re.match(IC_VARS_3D_OPT_RE, v)]
    if extra3d:
        print(f"  carrying passive tracers through: {', '.join(sorted(extra3d))}")
    for v in IC_VARS_3D + sorted(extra3d):
        sv = src.variables[v]
        dv = dst.createVariable(v, sv.dtype, ("ocean_time",) + sv.dimensions[1:])
        arr = np.array(sv[k, ...])
        dv[0, ...] = arr

    # Boundary-adjust slots the DA writes back on IC+BC cycles (see memory
    # obc_dim_order).  Packed CDL order: 2D obc (ocean_time, boundary, IorJ);
    # 3D obc (ocean_time, boundary, s_rho, IorJ).
    for name in ("zeta_obc", "ubar_obc", "vbar_obc"):
        dv = dst.createVariable(name, "f8", ("ocean_time", "boundary", "IorJ"))
        dv[0, ...] = 0.0
        dv.long_name = f"{name} boundary values"
    for name in ("u_obc", "v_obc", "temp_obc", "salt_obc"):
        dv = dst.createVariable(name, "f8", ("ocean_time", "boundary", "s_rho", "IorJ"))
        dv[0, ...] = 0.0
        dv.long_name = f"{name} boundary values"

    # Vertical-mixing slots that wrt_ini writes back on every cycle (see memory
    # wrt_ini_format_bug — missing vars here trigger Intel severe 61).
    # Dims: (ocean_time, s_w, eta_rho, xi_rho); values overwritten by ROMS.
    for name in ("AKv", "AKt", "AKs"):
        dv = dst.createVariable(name, "f8", ("ocean_time", "s_w", "eta_rho", "xi_rho"))
        dv[0, ...] = 0.0
        dv.long_name = f"{name} vertical mixing coefficient (placeholder)"
        dv.units = "meter2 second-1"

    # optional dried-cell sanitize (see memory fourdvar_wet_dry_mismatch)
    if sanitize and "wetdry_mask_rho" in src.variables:
        wd = np.array(src.variables["wetdry_mask_rho"][k, ...], dtype=bool)
        mask = np.array(src.variables["mask_rho"][:], dtype=bool)
        h    = np.array(src.variables["h"][:])
        z = np.array(dst.variables["zeta"][0, ...])
        dried = mask & (~wd)
        z[dried] = 0.0
        dst.variables["zeta"][0, ...] = z
        print(f"  sanitized {int(dried.sum())} dried cells (zeta -> 0)")

    src.close()
    dst.close()
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--his",  required=True)
    ap.add_argument("--time", required=True, help="YYYY-MM-DDTHH:MM:SS (UTC)")
    ap.add_argument("--out",  required=True)
    ap.add_argument("--sanitize", action="store_true",
                    help="strip WET_DRY dried-cell bookkeeping (needed under I4DVAR)")
    args = ap.parse_args()
    t = DT.datetime.strptime(args.time, "%Y-%m-%dT%H:%M:%S")
    build_ic(args.his, t, args.out, args.sanitize)


if __name__ == "__main__":
    main()
