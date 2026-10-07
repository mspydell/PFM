#!/usr/bin/env python3
"""
check_vs_operational.py — the acceptance test for stages 2 and 3.

Stages 2 and 3 do not reimplement PFM's forcing builders, they call them over a
different window and into different directories.  The claim that therefore
follows is: given the SAME window, the adapter must reproduce the operational
file to round-off.  If it does, the adapter demonstrably changed the window and
the paths and nothing about the science.  This script is that comparison.

    python check_vs_operational.py atm --da <DA file> --op <operational file>
    python check_vs_operational.py bc  --da <DA file> --op <operational file>

MATCH THE INIT, NOT THE VALID TIME.  The first attempt at this compared a DA
file built from the 2026-10-05 init against the operational file built from the
2026-10-06 init, on the single record where they overlap.  They disagreed by up
to 5.9 m/s in Uwind and 212 W/m2 in swrad -- which is ECMWF's 24 h forecast error
at that valid time, not a defect.  A DA file is only comparable to an operational
file built from the same initialisation, so run the DA stage with
t_fore = (operational fetch_time + 24 h) and compare the overlapping leads.

A non-zero exit means the fields disagree by more than round-off.
"""
from __future__ import annotations

import argparse
import datetime as DT
import sys
from pathlib import Path

import numpy as np

try:
    import netCDF4 as nc
except ImportError:
    sys.exit("netCDF4 is required")

ROMS_EPOCH = DT.datetime(1999, 1, 1)
FILL = 1e30

# float32 storage, then an interpolation onto the same grid, so bit-equality is
# not the bar.  Anything above this is a real difference in the computation.
RTOL = 1e-5

ATM_VARS = ("Uwind", "Vwind", "Tair", "Pair", "Qair", "rain",
            "swrad", "lwrad", "lwrad_down")
BC_VARS = ("zeta_south", "zeta_north", "zeta_west",
           "temp_south", "temp_north", "temp_west",
           "salt_south", "salt_north", "salt_west",
           "u_south", "v_south", "ubar_south", "vbar_south")


def _axis(d: nc.Dataset, tv: str) -> np.ndarray:
    """Decode one time variable to datetimes."""
    raw = np.asarray(d[tv][:]).ravel().astype(float)
    units = getattr(d[tv], "units", "")
    scale = 86400.0 if units.strip().lower().startswith("day") else 1.0
    return np.array([ROMS_EPOCH + DT.timedelta(seconds=x * scale)
                     for x in raw])


def _times(d: nc.Dataset, prefer: tuple[str, ...]) -> tuple[np.ndarray, str]:
    for tv in prefer:
        if tv in d.variables:
            break
    else:
        cands = [v for v in d.variables if v.endswith("_time")]
        if not cands:
            raise SystemExit("no time variable found")
        tv = cands[0]
    return _axis(d, tv), tv


# ---------------------------------------------------------------------------
# land masking.  This is not a refinement; without it the test reports a false
# failure.
#
# A ROMS BC file stores a value in every boundary cell, land included, and what
# lands in a land cell is whatever the gap-filling produced -- ROMS masks it out
# and never reads it.  Comparing those cells measures the filling, not the
# science.  Measured 2026-10-06 on LV1: the southern boundary has exactly 6 land
# cells (xi 247-252) and those were exactly the 6 columns that differed; the
# western boundary has none and was bit-identical over all 390; the northern
# boundary has 124 and showed the largest difference.  Every WATER cell matched
# to 0.000e+00.
_MASK_FOR = {"u": "mask_u", "ubar": "mask_u", "v": "mask_v", "vbar": "mask_v"}


def boundary_mask(grid: Path, var: str, n: int) -> np.ndarray | None:
    """Water mask (True = compare) for the boundary cells of `var`, or None."""
    if "_" not in var:
        return None
    stem, _, side = var.rpartition("_")
    if side not in ("south", "north", "west", "east"):
        return None
    mname = _MASK_FOR.get(stem, "mask_rho")
    with nc.Dataset(grid) as g:
        if mname not in g.variables:
            mname = "mask_rho"
        if mname not in g.variables:
            return None
        m = np.asarray(g[mname][:])
    sl = {"south": m[0, :], "north": m[-1, :],
          "west": m[:, 0], "east": m[:, -1]}[side]
    if sl.size != n:
        # u/v points are one shorter in their staggered direction; fall back to
        # whichever rho slice matches rather than guessing wrong.
        return None
    return sl > 0


def _time_var_of(d: nc.Dataset, v: str) -> str | None:
    """The time coordinate THIS variable is on.

    A ROMS BC file carries more than one time axis -- zeta_* are hourly on
    zeta_time while temp_*/salt_*/u_*/v_* are 3-hourly on temp_time -- and a
    variable's first dimension names its own axis.  Indexing every variable by
    one shared axis walks off the end of the 3-hourly ones.
    """
    dims = d[v].dimensions
    if not dims:
        return None
    return dims[0] if dims[0] in d.variables else None


def compare(da_path: Path, op_path: Path, kind: str,
            grid: Path | None = None) -> int:
    prefer = ("wind_time", "ocean_time") if kind == "atm" else ("zeta_time",
                                                                "ocean_time")
    variables = ATM_VARS if kind == "atm" else BC_VARS
    A = nc.Dataset(op_path)
    B = nc.Dataset(da_path)
    ta, tva = _times(A, prefer)
    tb, tvb = _times(B, prefer)

    print(f"  operational {op_path.name}")
    print(f"              {ta[0]} .. {ta[-1]}  (n={ta.size}, {tva})")
    print(f"  DA          {da_path.name}")
    print(f"              {tb[0]} .. {tb[-1]}  (n={tb.size}, {tvb})")

    common = sorted(set(ta) & set(tb))
    if not common:
        print("\n  NO SHARED RECORDS -- nothing to compare.  Build the DA file "
              "for the window that starts at the operational fetch_time.")
        return 2
    if len(common) == 1:
        print(f"\n  WARNING: only one shared record ({common[0]}).  If the two "
              f"files came from different inits this compares forecast skill, "
              f"not the adapter -- see the module docstring.")
    print(f"\n  {len(common)} shared record(s): "
          f"{common[0]} .. {common[-1]}\n")

    worst: dict[str, tuple[float, float, str]] = {}
    axis_cache: dict[tuple[int, str], np.ndarray] = {}

    def idx_map(d: nc.Dataset, which: int, tv: str) -> dict:
        key = (which, tv)
        if key not in axis_cache:
            axis_cache[key] = _axis(d, tv)
        return {t: i for i, t in enumerate(axis_cache[key])}

    for v in variables:
        if v not in A.variables or v not in B.variables:
            continue
        # resolve each variable onto ITS OWN time axis, in both files
        tva, tvb = _time_var_of(A, v), _time_var_of(B, v)
        if tva is None or tvb is None:
            continue
        ia, ib = idx_map(A, 0, tva), idx_map(B, 1, tvb)
        wet = boundary_mask(grid, v, A[v].shape[-1]) if grid else None
        shared = sorted(set(ia) & set(ib))
        if not shared:
            worst[v] = (np.nan, np.nan, f"no shared record on {tva}/{tvb}")
            continue
        wmax = wrel = 0.0
        wat = ""
        for t in shared:
            a = np.asarray(A[v][ia[t]], dtype=float)
            b = np.asarray(B[v][ib[t]], dtype=float)
            if a.shape != b.shape:
                worst[v] = (np.nan, np.nan, f"SHAPE {a.shape} vs {b.shape}")
                break
            a = np.where(np.abs(a) > FILL, np.nan, a)
            b = np.where(np.abs(b) > FILL, np.nan, b)
            dif = np.abs(a - b)
            if wet is not None:
                # broadcast the boundary mask over any leading (s_rho) axes
                dif = np.where(np.broadcast_to(wet, dif.shape), dif, np.nan)
            if not np.isfinite(dif).any():
                continue
            m = float(np.nanmax(dif))
            span = float(np.nanmax(a) - np.nanmin(a))
            rel = m / span if span > 0 else m
            if rel > wrel:
                wmax, wrel, wat = m, rel, str(t)
        else:
            worst[v] = (wmax, wrel, wat)

    bad = []
    for v, (m, rel, at) in worst.items():
        if not np.isfinite(m):
            print(f"    {v:14s} {at}")
            bad.append(v)
            continue
        flag = "ok" if rel <= RTOL else "DIFFERS"
        if rel > RTOL:
            bad.append(v)
        print(f"    {v:14s} max|diff|={m:.3e}  rel={rel:.2e}  {flag}"
              f"{'  @ ' + at if rel > RTOL else ''}")

    A.close()
    B.close()
    print()
    if bad:
        print(f"  FAIL: {len(bad)} field(s) differ by more than rtol={RTOL}: "
              f"{', '.join(bad)}")
        print(f"  If the two files came from different ECMWF/HYCOM inits this is "
              f"expected -- check the init, not the code.")
        return 1
    print(f"  PASS: all {len(worst)} compared fields agree within rtol={RTOL} "
          f"over {len(common)} record(s)"
          f"{', water cells only' if grid else ''}.")
    print(f"  The adapter reproduces PFM's builder; it changed the window and "
          f"the paths only.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=("atm", "bc"))
    ap.add_argument("--da", required=True, type=Path)
    ap.add_argument("--op", required=True, type=Path)
    ap.add_argument("--grid", type=Path,
                    default=Path("/scratch/PFM_Simulations/Grids/"
                                 "GRID_SDTJRE_LV1_rx020_hmask.nc"),
                    help="LV1 grid, for the land mask (bc only). "
                         "Pass --grid '' to compare land cells too.")
    a = ap.parse_args()
    for p in (a.da, a.op):
        if not p.is_file():
            return print(f"missing: {p}") or 2
    grid = a.grid if (a.kind == "bc" and a.grid and str(a.grid)) else None
    if grid and not grid.is_file():
        print(f"  warning: grid {grid} not found; comparing land cells too")
        grid = None
    return compare(a.da, a.op, a.kind, grid)


if __name__ == "__main__":
    raise SystemExit(main())
