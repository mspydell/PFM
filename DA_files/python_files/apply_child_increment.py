#!/usr/bin/env python3
"""
apply_child_increment.py

Add a PRE-BUILT child-grid increment to a child-grid base state, producing the
initial condition for a child "analysis" forward run.

    IC = base + increment

`base` is a single-record IC on the child grid -- for LV2 3/7 that is the PFM
operational history slice at 00Z, which gives LV2 the same provenance LV1's DA
cold start had; for later days it is the previous day's analysis history at
t = 24 h.  Build either with his_to_ic.py, then hand it here.

`incr` is the output of make_child_increment.py: the LV1 4D-Var increment
already objectively mapped onto this child grid and vertically remapped, with
--blend-space D.  It is ALREADY on the child grid, which is why this is a plain
addition and not an interpolation -- unlike
interp_LV1_increment_to_LV2.add_increment_to_ic(), which takes an increment on
the LV1 grid and interpolates it on the way in.

Nothing is rotated or remapped here.  make_child_increment.py already rotated
velocity onto the child's XI/ETA axes and remapped onto the child's s-levels, so
doing anything further would double-apply it.  The vertical parameters of base
and incr are therefore checked for an exact match and the run aborts if they
disagree -- that mismatch is silent and ruinous otherwise.

ubar/vbar are added for completeness, but note ROMS discards them: every .in in
this project runs NRREC=0, so ini_fields overwrites ubar with the Hz-weighted
depth average of u at the first time step.  What reaches the model is
depth_avg(u_base + du).  zeta IS kept (ini_zeta only copies time levels).

CLI:
    python apply_child_increment.py \\
        --base /dataSIO/.../IC_files_oi/LV2_prior_20260307.nc \\
        --incr /dataSIO/.../INC_files/INC_LV2_20260307.nc \\
        --out  /dataSIO/.../IC_files_oi/IC_LV2_oi_20260307.nc
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import numpy as np
import netCDF4 as nc

STATE_2D = ("zeta", "ubar", "vbar")
STATE_3D = ("u", "v", "temp", "salt")
VPARAMS = ("Vtransform", "hc")
VPROFILE = ("Cs_r", "s_rho")

FILL = 1e30


def _san(a):
    """ROMS writes ~1e37 on land and unused u/v points."""
    return np.where(np.abs(a) > FILL, 0.0, a)


def check_compatible(b: nc.Dataset, i: nc.Dataset) -> None:
    """Abort on any grid or vertical-coordinate mismatch.

    A silent mismatch here would add the increment onto the wrong levels or the
    wrong cells and still produce a file that runs, so this is deliberately
    strict rather than forgiving.
    """
    bad = []
    for v in STATE_2D + STATE_3D:
        for d in (b, i):
            if v not in d.variables:
                bad.append(f"{v} missing from {d.filepath()}")
        if bad:
            continue
        sb = b[v].shape[-2:]
        si = i[v].shape[-2:]
        if sb != si:
            bad.append(f"{v}: base horizontal {sb} != incr {si}")
        if v in STATE_3D and b[v].shape[-3] != i[v].shape[-3]:
            bad.append(f"{v}: base N={b[v].shape[-3]} != incr N={i[v].shape[-3]}")
    for v in VPARAMS:
        if v in b.variables and v in i.variables:
            vb, vi = float(np.asarray(b[v][...])), float(np.asarray(i[v][...]))
            if not np.isclose(vb, vi):
                bad.append(f"{v}: base {vb} != incr {vi}")
    for v in VPROFILE:
        if v in b.variables and v in i.variables:
            ab, ai = np.asarray(b[v][:]), np.asarray(i[v][:])
            if ab.shape != ai.shape or not np.allclose(ab, ai, atol=1e-12):
                bad.append(f"{v}: base and incr s-coordinate profiles differ")
    if bad:
        raise SystemExit("  INCOMPATIBLE base / increment:\n    "
                         + "\n    ".join(bad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="single-record child IC (from his_to_ic.py)")
    ap.add_argument("--incr", required=True,
                    help="child-grid increment (from make_child_increment.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rec", type=int, default=0,
                    help="record in --base to use (default 0)")
    ap.add_argument("--grid", default=None,
                    help="the grid file the RUN will use (GRDNAME).  Mask to "
                         "this rather than to the base file's own mask.  The PFM "
                         "LV3 history carries the rx020 mask, which has 153 more "
                         "wet cells than the maskCoast grid the run uses, and "
                         "some hold salinity up to 49.6 PSU; those cells are "
                         "coastal, so they sit in pressure-gradient stencils "
                         "next to real water.")
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.base, out)

    with nc.Dataset(a.incr) as i, nc.Dataset(out, "a") as o:
        check_compatible(o, i)
        bs = getattr(i, "blend_space", "(absent)")
        print(f"  base = {Path(a.base).name}  rec {a.rec}")
        print(f"  incr = {Path(a.incr).name}  blend_space={bs}")
        print(f"         {getattr(i, 'source', '(no source attr)')}")
        if bs != "D":
            print(f"  WARNING: increment blend_space is {bs!r}, not 'D'.  The "
                  f"h-space blend needs a Dcrit clip that kinks h_p and, at a "
                  f"deep low tide, can invert the parent column.")

        # Land mask: prefer the RUN's grid when given, else the base file's own.
        if a.grid:
            with nc.Dataset(a.grid) as g:
                mr = np.asarray(g["mask_rho"][:]).astype(bool)
                mu = (np.asarray(g["mask_u"][:]).astype(bool)
                      if "mask_u" in g.variables else mr[:, :-1] & mr[:, 1:])
                mv = (np.asarray(g["mask_v"][:]).astype(bool)
                      if "mask_v" in g.variables else mr[:-1, :] & mr[1:, :])
            nb = (np.asarray(o["mask_rho"][:]).astype(bool)
                  if "mask_rho" in o.variables else mr)
            extra = int((nb & ~mr).sum())
            print(f"  mask from run grid {Path(a.grid).name}"
                  + (f"  ({extra} cell(s) wet in the base but land in the run "
                     f"grid will be zeroed)" if extra else ""))
            if "mask_rho" in o.variables:
                o["mask_rho"][:] = mr.astype(o["mask_rho"].dtype)
            if "mask_u" in o.variables:
                o["mask_u"][:] = mu.astype(o["mask_u"].dtype)
            if "mask_v" in o.variables:
                o["mask_v"][:] = mv.astype(o["mask_v"].dtype)
        else:
            mr = (np.asarray(o["mask_rho"][:]).astype(bool)
                  if "mask_rho" in o.variables else None)
            mu = (np.asarray(o["mask_u"][:]).astype(bool)
                  if "mask_u" in o.variables else None)
            mv = (np.asarray(o["mask_v"][:]).astype(bool)
                  if "mask_v" in o.variables else None)
        MASK = {"zeta": mr, "temp": mr, "salt": mr,
                "ubar": mu, "u": mu, "vbar": mv, "v": mv}

        print(f"\n  {'var':5s} {'|base|max':>10s} {'|incr|max':>10s} "
              f"{'|sum|max':>10s}   rms(incr)")
        for v in STATE_2D + STATE_3D:
            base = _san(np.asarray(o[v][a.rec]))
            dinc = _san(np.asarray(i[v][0]))
            tot = base + dinc
            m = MASK.get(v)
            if m is not None:
                tot = np.where(m, tot, 0.0) if tot.ndim == 2 else \
                      np.where(m[None, ...], tot, 0.0)
            o[v][a.rec] = tot
            sel = (slice(None),) if m is None else \
                  ((m,) if base.ndim == 2 else (slice(None), m))
            print(f"  {v:5s} {np.abs(base[sel]).max():10.5f} "
                  f"{np.abs(dinc[sel]).max():10.5f} "
                  f"{np.abs(tot[sel]).max():10.5f}   "
                  f"{np.sqrt((dinc[sel]**2).mean()):.6f}")

        o.setncattr("increment_applied", Path(a.incr).name)
        o.setncattr("increment_source", getattr(i, "source", "unknown"))
        o.setncattr("increment_blend_space", bs)
        o.setncattr("increment_base", Path(a.base).name)
        o.setncattr("increment_note",
                    "IC = base + child-grid increment, added on the child grid "
                    "with no further rotation or vertical remap (both were done "
                    "by make_child_increment.py).  ubar/vbar are written but "
                    "ROMS recomputes them from u,v at NRREC=0.")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
