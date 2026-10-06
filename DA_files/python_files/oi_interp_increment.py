#!/usr/bin/env python3
"""Objective-mapping (kriging) interpolation of the LV1 DA increment onto the
child grids, replacing LinearNDInterpolator + a land-fill rule.

The idea: LV1's wet cells ARE the data.  Dry cells are simply absent -- there
is nothing to fill, no zero to invent, no plateau to extrapolate.  Every child
point is estimated from the wet LV1 points around it under the same covariance
the DA itself assumed:

    Matern 5/2,  L = 15 km  (= HdecayI),  nugget = 1e-2

Matern rather than squared-exponential because L/dx = 15/1.92 ~ 7.8 makes a
Gaussian kernel pathologically ill-conditioned (adjacent cells correlate at
0.992), and because ROMS's own correlation operator is diffusion-based, which
is Matern-like rather than infinitely smooth.

    f(t) = C_tw (C_ww + nugget I)^-1 f_w

Made tractable by two things:
  * a moving neighbourhood -- each target uses only wet LV1 points within
    R_FACTOR*L, capped at K_MAX nearest;
  * blocking -- targets are grouped into tiles of BLOCK x BLOCK LV1 cells and
    share one Cholesky factorisation.  Since f_w is fixed per field we solve
    A y = f_w ONCE per block, after which every target in it is a dot product.

Velocity is mapped as its EARTH components (uE, vN), each as an independent
scalar, then rotated onto the child's XI/ETA axes -- same convention as the
pipeline's `_interp_arrays_to_child`.

Usage:
  python3 oi_interp_increment.py --field v --date 20260401 --tag "" \\
      --root /dataSIO/DA_Simulations/output_files
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import netCDF4 as nc
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree

import plot_increment_smoothing as P

L_KM = 15.0          # e-folding scale, matches HdecayI
NUGGET = 1e-2        # relative to sigma^2 = 1
R_FACTOR = 2.5       # neighbourhood radius in units of L
K_MAX = 900          # cap on neighbours per block
BLOCK = 4            # tile size, in LV1 cells
KM = 111.0
OUTDIR = P.OUTDIR


def matern52(r, L=None):
    """L defaults to the module-level L_KM, read at CALL time so callers can
    override it (e.g. a shorter scale for bathymetry than for the increment)."""
    L = L_KM if L is None else L
    a = np.sqrt(5.0) * r / L
    return (1.0 + a + a * a / 3.0) * np.exp(-a)


def to_km(lon, lat, lat0):
    return lon * np.cos(np.deg2rad(lat0)) * KM, lat * KM


def oi_map(fields, lon_w, lat_w, lon_t, lat_t, lat0, label=""):
    """Map several co-located scalar fields from wet points onto targets.

    Weights are built at the NODES of a regular lattice (spacing BLOCK parent
    cells) and each target bilinearly combines the estimates from the four
    nodes around it, so the result is continuous by construction.

    The previous version assigned every target in a tile the single
    tile-centred neighbourhood.  Adjacent tiles draw on slightly different
    data, so their estimates disagreed at the shared edge and the field came
    out with seams on the tile lattice -- invisible on LV2/LV3 but a 0.5 degC
    jump across one 32 m cell on LV4, i.e. 29 degC/km, 470x LV1's own
    gradient.  The cost is 4x the C_tw evaluations (each target is now served
    by four nodes); the Cholesky count, which dominates, is unchanged.

    `fields` is a list of 1-D arrays over the data points; returns a list of
    1-D arrays over the targets.  They share the neighbourhoods and the
    factorisations, so mapping many fields costs barely more than one.
    """
    xw, yw = to_km(lon_w, lat_w, lat0)
    xt, yt = to_km(lon_t, lat_t, lat0)
    tree = cKDTree(np.column_stack([xw, yw]))
    M = xt.size
    # float32, and ONE array rather than a list of float64 vectors plus a
    # float32 copy in oi_many.  With every field in a single call (see
    # oi_many's `chunk`) the float64 accumulator became the peak allocation:
    # 83 fields x 554k LV4 targets was 0.37 GB of f8 plus 0.18 GB of f4 copy.
    # The stored output is f4 regardless, and each target accumulates only the
    # 4 bilinear node terms, so f8 bought ~1 ulp of nothing.
    out = np.zeros((len(fields), M), dtype="f4")

    step = BLOCK * 1.92
    x0, y0 = xw.min(), yw.min()
    fx = (xt - x0) / step
    fy = (yt - y0) / step
    i0 = np.floor(fx).astype(np.int64)
    j0 = np.floor(fy).astype(np.int64)
    tx, ty = fx - i0, fy - j0

    OFF, BIG = 10_000, 1_000_003          # keep the node key positive
    nid, tid, wts = [], [], []
    for di, dj, w in ((0, 0, (1-tx)*(1-ty)), (1, 0, tx*(1-ty)),
                      (0, 1, (1-tx)*ty),     (1, 1, tx*ty)):
        nid.append((i0 + di + OFF) * BIG + (j0 + dj + OFF))
        tid.append(np.arange(M, dtype=np.int64))
        wts.append(w)
    nid = np.concatenate(nid); tid = np.concatenate(tid); wts = np.concatenate(wts)
    keep = wts > 1e-12
    nid, tid, wts = nid[keep], tid[keep], wts[keep]
    order = np.argsort(nid, kind="stable")
    nid, tid, wts = nid[order], tid[order], wts[order]
    groups = np.split(np.arange(nid.size), np.flatnonzero(np.diff(nid)) + 1)

    R = R_FACTOR * L_KM
    t0 = time.perf_counter()
    nfall = 0
    for n, grp in enumerate(groups):
        key = nid[grp[0]]
        ci = key // BIG - OFF
        cj = key - (key // BIG) * BIG - OFF
        cx, cy = x0 + ci * step, y0 + cj * step

        rr = R
        for _ in range(5):
            idx = np.asarray(tree.query_ball_point([cx, cy], rr), dtype=np.int64)
            if idx.size >= 8:
                break
            rr *= 2.0
        tt, ww = tid[grp], wts[grp]
        if idx.size < 8:                      # genuinely nothing: nearest data
            _, nn = tree.query(np.column_stack([xt[tt], yt[tt]]))
            for fi, f in enumerate(fields):
                out[fi, tt] += (ww * f[nn]).astype("f4")
            nfall += tt.size
            continue
        if idx.size > K_MAX:
            d = np.hypot(xw[idx] - cx, yw[idx] - cy)
            idx = idx[np.argpartition(d, K_MAX)[:K_MAX]]

        d_ww = np.hypot(xw[idx][:, None] - xw[idx][None, :],
                        yw[idx][:, None] - yw[idx][None, :])
        A = matern52(d_ww)
        A[np.diag_indices_from(A)] += NUGGET
        try:
            cf = cho_factor(A, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            A[np.diag_indices_from(A)] += 10 * NUGGET
            cf = cho_factor(A, lower=True, check_finite=False)
        ys = [cho_solve(cf, f[idx], check_finite=False) for f in fields]

        for s in range(0, tt.size, 4000):
            sl = slice(s, s + 4000)
            ts, wsl = tt[sl], ww[sl]
            d_tw = np.hypot(xt[ts][:, None] - xw[idx][None, :],
                            yt[ts][:, None] - yw[idx][None, :])
            Ctw = matern52(d_tw)
            for fi, y in enumerate(ys):
                out[fi, ts] += (wsl * (Ctw @ y)).astype("f4")
        if n % 400 == 0 and n:
            print(f"    {label} node {n}/{len(groups)}  "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"    {label}: {len(groups)} nodes, {time.perf_counter()-t0:.1f}s"
          + (f", {nfall} target-weights fell back to nearest" if nfall else ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="v",
                    choices=("u", "v", "zeta", "temp", "salt"))
    ap.add_argument("--date", default="20260401")
    ap.add_argument("--tag", default="")
    ap.add_argument("--root", default=None)
    ap.add_argument("--mask", action="append", default=[],
                    help="LEVEL=file[,file2] -- multiply the increment on that "
                         "level by these mask files (product if several). "
                         "Repeatable. zeta is never masked.")
    ap.add_argument("--out-tag", default="", help="suffix for output filenames")
    args = ap.parse_args()

    masks = {}
    for spec in args.mask:
        lvl, files = spec.split("=", 1)
        masks[lvl.strip()] = [f.strip() for f in files.split(",")]

    P.FIELD = args.field
    M = P.FIELD_META[args.field]
    if args.root:
        P.LV1_V2 = Path(args.root)
        print(f"  LV1 source tree: {P.LV1_V2}")
    print(f"  Matern 5/2, L = {L_KM} km, nugget = {NUGGET}, "
          f"R = {R_FACTOR}L, K_max = {K_MAX}, block = {BLOCK} LV1 cells\n")

    grids = {k: P.load_grid(k) for k in P.LEVELS}
    G1 = grids["LV1"]
    d1, d2 = P.lv1_surface_uv(args.date, args.tag)
    lat0 = float(G1["lat_rho"].mean())
    mw = G1["mask_rho"]

    if args.field in P.SCALARS:
        data = [np.where(np.abs(d1) > 1e30, 0.0, d1)[mw]]
    else:
        du = np.where(np.abs(d1) > 1e30, 0.0, d1)
        dv = np.where(np.abs(d2) > 1e30, 0.0, d2)
        u_r = np.zeros_like(G1["lat_rho"]); v_r = np.zeros_like(G1["lat_rho"])
        u_r[:, 1:-1] = 0.5 * (du[:, :-1] + du[:, 1:])
        v_r[1:-1, :] = 0.5 * (dv[:-1, :] + dv[1:, :])
        cA, sA = np.cos(G1["angle"]), np.sin(G1["angle"])
        data = [(u_r * cA - v_r * sA)[mw], (u_r * sA + v_r * cA)[mw]]

    lon_w, lat_w = G1["lon_rho"][mw], G1["lat_rho"][mw]
    print(f"  data: {lon_w.size:,} wet LV1 points\n")

    prim = d2 if args.field == "v" else d1
    oi = {"LV1": np.where(P.geom(G1)[2], prim, np.nan)}
    for k in ("LV2", "LV3", "LV4"):
        C = grids[k]
        lo, la, msk, _, _ = P.geom(C)
        res = oi_map(data, lon_w, lat_w, lo.ravel(), la.ravel(), lat0, label=k)
        if args.field in P.SCALARS:
            fld = res[0].reshape(lo.shape)
        else:
            if args.field == "u":
                ang = 0.5 * (C["angle"][:, :-1] + C["angle"][:, 1:])
                fld = (res[0] * np.cos(ang).ravel()
                       + res[1] * np.sin(ang).ravel()).reshape(lo.shape)
            else:
                ang = 0.5 * (C["angle"][:-1, :] + C["angle"][1:, :])
                fld = (-res[0] * np.sin(ang).ravel()
                       + res[1] * np.cos(ang).ravel()).reshape(lo.shape)
        oi[k] = np.where(msk, fld, np.nan)
        if k in masks and args.field != "zeta":
            mm = load_mask(masks[k], C, args.field)
            before = np.nanmax(np.abs(oi[k]))
            oi[k] = oi[k] * mm
            print(f"  {k}: applied {len(masks[k])} mask(s)  "
                  f"{[Path(x).name for x in masks[k]]}")
            print(f"       max|incr| {before:.5f} -> {np.nanmax(np.abs(oi[k])):.5f}")
        elif k in masks:
            print(f"  {k}: mask NOT applied (field is zeta)")
        print(f"  {k}: max|incr| = {np.nanmax(np.abs(oi[k])):.5f}\n")

    g = {k: P.grad_mag(oi[k], grids[k]) for k in oi}
    print(f"{'lvl':5s} {'max|grad|':>10s}  (OI)")
    for k in ("LV2", "LV3", "LV4"):
        print(f"{k:5s} {np.nanmax(g[k][np.isfinite(g[k])]):10.4f}")

    vlim = float(np.nanpercentile(np.abs(oi["LV1"]), 99.5))
    gmax = float(np.nanpercentile(np.concatenate(
        [g[k][np.isfinite(g[k])].ravel() for k in ("LV2", "LV3", "LV4")]), 99.0))
    d = f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"
    st = (f"objective mapping direct LV1$\\to$child, Mat\\'ern 5/2, "
          f"L = {L_KM:g} km, nugget {NUGGET:g}")
    for k in grids:
        grids[k]["sigma"] = None            # nothing was smoothed

    T = f"_{args.out_tag}" if args.out_tag else ""
    if masks and args.field != "zeta":
        st += ", bay/TJRE masked"
    P.make_figure(OUTDIR / f"fig1_raw_oi{T}_full_grids.png", oi, grids, vlim,
                  suptitle=f"{M['sym']} increment — {st} — {d}")
    P.make_figure(OUTDIR / f"fig5_grad_raw_oi{T}_full_grids.png", g, grids, gmax,
                  cmap="Oranges", vmin=0.0,
                  cblabel=f"${M['gsym']}$   ({M['gunit']})",
                  suptitle=f"Gradient magnitude — {st} — {d}")


if __name__ == "__main__":
    main()
