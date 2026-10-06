#!/usr/bin/env python3
"""Build a child-grid DA increment file from the LV1 4D-Var increment.

Replaces the LinearNDInterpolator + land_fill path with objective mapping, and
does the vertical remap properly for LV4 (N = 40 -> 10, hc = 50 -> 3.5).

    increment = outer1(t=0) - outer0(t=0)      on LV1, 40 s-levels

HORIZONTAL   objective mapping (Matern 5/2, L = 15 km = HdecayI, nugget 1e-2)
             straight from LV1's WET rho points onto the child's rho/u/v
             points.  Dry LV1 cells are simply not data -- no land fill, no
             zero to invent, no plateau to extrapolate.  Measured against the
             old path this reproduces LV1's own gradient instead of inflating
             it 4-6x, and removes the triangulation facets entirely.

VERTICAL     per-column linear interpolation from the parent's z to the
             child's z, both built with a COMMON reference zeta so the choice
             of zeta is second order.

    zeta_ref = LV1 outer0 zeta(t=0), OI-mapped        (base-independent)
    h_p      = exp(OI(log h_LV1)),  L = 10 km, NO clamp
    h_c      = max(h_child, DCRIT - zeta_ref)

  * log h because h decreases shoreward, so any extrapolation past the coast
    would cross zero; log space is positive by construction.  L = 10 km was
    dialled in against L = 15 / 5 km -- see figures/bathy/.
  * no clamp: where the child outruns the parent the remap flat-extrapolates
    the parent's bottom value, exactly as parent_history_to_BC does.  Measured
    on LV4: median 0% of the column, 90th pct 8%, and no column over 50%.
    Clamping instead stretched the whole parent profile on 40% of the grid.
  * DCRIT floors the TOTAL depth, matching how ROMS masks a cell dry.  4% of
    LV4's "water" cells have NEGATIVE h (wetting/drying), which would give a
    degenerate z column.

MASKS        u, v, temp, salt, ubar, vbar are multiplied by the bay/TJRE mask.
             zeta is NOT: the bay fills and drains in phase with the coastal
             ocean, so suppressing its zeta increment would manufacture a head
             difference across the mouth.

BAROTROPIC   ubar/vbar are re-derived as the Hz-weighted depth average of the
             masked, remapped 3-D u/v.  The outer-diff source carries a real
             barotropic signal, but once u and v are masked in the bay the
             mapped ubar no longer equals depth_avg(u) there, and that
             mismatch is exactly the fast-transient generator we are trying to
             avoid.  --keep-ubar restores the old behaviour.

Usage:
  python3 make_child_increment.py --child LV4 --date 20260401 \\
      --root /dataSIO/DA_Simulations/output_files \\
      --child-vparams-from /path/to/LV4_IC_or_his.nc \\
      --mask masks/LV4_incr_mask_sigma1km.nc \\
      --mask masks/LV4_incr_mask_sigma1km_tjre.nc \\
      --out /path/to/INC_LV4_20260401.nc
"""
from __future__ import annotations
import argparse
import datetime as DT
import sys
import time
from pathlib import Path

import numpy as np
import netCDF4 as nc
from scipy.ndimage import distance_transform_edt, gaussian_filter

import plot_increment_smoothing as P
import oi_interp_increment as OI

GRIDS = {
    "LV1": "/scratch/PFM_Simulations/Grids/GRID_SDTJRE_LV1_rx020_hmask.nc",
    "LV2": "/scratch/PFM_Simulations/Grids/GRID_SDTJRE_LV2_rx020.nc",
    "LV3": "/scratch/PFM_Simulations/Grids/GRID_SDTJRE_LV3_rx020_maskCoast.nc",
    "LV4": "/scratch/PFM_Simulations/Grids/GRID_SDTJRE_LV4_mss_oct2024.nc",
}
L_INCR = 15.0      # km, matches HdecayI -- the covariance the DA itself used
L_BATHY = 10.0     # km, dialled in against 15 / 5 km
DCRIT = 0.35       # m, from LV4's .in file
DX_PARENT = 1920.0 # m, LV1 mean grid spacing
OI_CHUNK = 30      # fields per OI call; bounds peak memory on LV4
HP_CACHE_DIR = Path("/dataSIO/DA_Simulations/input_files/hp_oi_cache")
NLEV_P = 40        # LV1 s-levels


def _san(a):
    """ROMS writes ~1e37 on land and unused u/v points."""
    return np.where(np.abs(a) > 1e30, 0.0, a)


def read_h(level):
    with nc.Dataset(GRIDS[level]) as d:
        return np.asarray(d["h"][:])


def read_vparams(fn):
    """Vtransform/hc/Cs_r/s_rho AND Cs_w/s_w -- the w-levels are needed for Hz."""
    with nc.Dataset(fn) as d:
        out = dict(Vt=int(np.asarray(d["Vtransform"][...])),
                   hc=float(np.asarray(d["hc"][...])),
                   Cs_r=np.asarray(d["Cs_r"][:]),
                   s_rho=np.asarray(d["s_rho"][:]))
        for k, v in (("Cs_w", "Cs_w"), ("s_w", "s_w")):
            out[k] = np.asarray(d[v][:]) if v in d.variables else None
    return out


def z_levels(h, zeta, Vt, hc, Cs, s):
    """z at rho (or w) levels; h, zeta are (M,) flat.  Returns (N, M),
    ascending in z along axis 0 (deepest first)."""
    N = len(s)
    Sk = (hc * s[:, None] + h[None, :] * Cs[:, None]) / (hc + h[None, :])
    if Vt == 2:
        return zeta[None, :] + (zeta[None, :] + h[None, :]) * Sk
    return h[None, :] * Sk + zeta[None, :] * (1.0 + Sk / h[None, :])


def remap_z(f_p, z_p, z_c, chunk=50_000):
    """Per-column linear interpolation from parent z to child z.

    Vectorised: the reference implementation loops over every column, which is
    554,526 python iterations on LV4.  Flat extrapolation at BOTH ends falls
    out of clipping the interpolation weight to [0, 1], which reproduces
    np.interp's end behaviour (and parent_history_to_BC's).
    """
    Np, M = f_p.shape
    Nc = z_c.shape[0]
    out = np.empty((Nc, M), dtype="f4")
    for s0 in range(0, M, chunk):
        s1 = min(s0 + chunk, M)
        zp, zc, fp = z_p[:, s0:s1], z_c[:, s0:s1], f_p[:, s0:s1]
        idx = (zc[:, None, :] > zp[None, :, :]).sum(axis=1) - 1
        np.clip(idx, 0, Np - 2, out=idx)
        z0 = np.take_along_axis(zp, idx, 0); z1 = np.take_along_axis(zp, idx + 1, 0)
        f0 = np.take_along_axis(fp, idx, 0); f1 = np.take_along_axis(fp, idx + 1, 0)
        w = np.clip((zc - z0) / np.where(z1 - z0 == 0, 1e-9, z1 - z0), 0.0, 1.0)
        out[:, s0:s1] = f0 + w * (f1 - f0)
    return out


def oi_many(fields, lon_w, lat_w, lon_t, lat_t, lat0, L, label="", chunk=None):
    """OI a list of co-located scalars, chunked to bound memory.

    `chunk` overrides OI_CHUNK.  This matters a great deal: nearly all the cost
    of an oi_map call is the GEOMETRY -- the KD-tree queries, the C_ww distance
    matrices and their Cholesky factorisations -- and that is rebuilt on every
    call.  Measured on LV3 (124 nodes): 1 field costs 12.0 s and 30 fields cost
    13.6 s, i.e. ~12 s fixed and ~0.05 s per field.  So passing every field in
    ONE call instead of chunking is close to a free 3-4x, and batching several
    DAYS into one call amortises the geometry further still, because the
    operator depends only on the grids and not on the data.
    """
    OI.L_KM = L
    step = chunk or OI_CHUNK
    if step >= len(fields):            # one call: hand oi_map's own array back
        return OI.oi_map(fields, lon_w, lat_w, lon_t, lat_t, lat0,
                         label=f"{label}[0:{len(fields)}]")
    out = np.empty((len(fields), lon_t.size), dtype="f4")
    for i in range(0, len(fields), step):
        j = min(i + step, len(fields))
        res = OI.oi_map(fields[i:j], lon_w, lat_w, lon_t, lat_t, lat0,
                        label=f"{label}[{i}:{j}]")
        for k, r in enumerate(res):
            out[i + k] = r.astype("f4")
        del res
    return out


_HP_CACHE = {}


def hp_cached(child, kind, logh, lon_w, lat_w, lon_t, lat_t, lat0):
    """h_p = exp(OI(log h_LV1)) depends ONLY on the grids, so it is identical
    for every date.  Cache it per (child, stagger) in-process and on disk --
    recomputing it once per day was pure waste over a 35-day batch."""
    key = (child, kind)
    if key in _HP_CACHE:
        return _HP_CACHE[key]
    f = HP_CACHE_DIR / f"hp_{child}_{kind}_L{L_BATHY:g}km.npy"
    if f.exists():
        hp = np.load(f)
        if hp.size == lon_t.size:
            _HP_CACHE[key] = hp
            print(f"  h_p {kind}: from cache {f.name}", flush=True)
            return hp
    hp = np.exp(oi_many([logh], lon_w, lat_w, lon_t, lat_t, lat0, L_BATHY,
                        label=f"{kind}-h", chunk=1)[0])
    try:
        HP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(f, hp)
        print(f"  h_p {kind}: cached -> {f.name}", flush=True)
    except OSError as e:
        print(f"  h_p {kind}: not cached ({e})", flush=True)
    _HP_CACHE[key] = hp
    return hp


def load_masks(paths, C, kind):
    """Product of rho-point masks, averaged onto the rho / u / v grid."""
    if not paths:
        return None
    m = None
    for fp in paths:
        with nc.Dataset(fp) as d:
            a = np.asarray(d["mask_incr"][:])
        if a.shape != C["lon_rho"].shape:
            raise SystemExit(f"{fp}: shape {a.shape} != {C['lon_rho'].shape}")
        m = a if m is None else m * a
    if kind == "rho":
        return m
    if kind == "u":
        return 0.5 * (m[:, :-1] + m[:, 1:])
    return 0.5 * (m[:-1, :] + m[1:, :])


# ---------------------------------------------------------------------------

def blend_hp(hp, h_child, q0, q1):
    """Pull h_parent toward the child's bathymetry where they disagree badly.

    LV1's shelf break is one or two 1.9 km cells wide, so ANY smoothing kernel
    blurs deep water inshore and h_p comes out 2-4x too deep in a coastal
    fringe.  There the child's column is a small fraction of the parent's, the
    surface-intensified increment is up-weighted, and the depth-averaged ubar
    inflates (measured on LV2: |vbar| 0.132 -> 0.253 against an LV1 source of
    0.134).

        q = h_child / h_p
        w = smoothstep(q; q0, q1)
        h_p <- exp( w log h_p + (1-w) log h_child )

    q ~ 1  keep h_p -- they agree, which is 97% of the domain
    q << 1 fall back to the child's own bathymetry; the remap is then
           sigma-to-sigma there and the depth average is preserved
    q > 1  keep h_p; child-deeper is the safe direction, handled by the
           flat extrapolation at the bottom
    """
    if q0 is None:
        return hp
    w = blend_weight(hp, h_child, q0, q1)
    return np.exp(w * np.log(np.maximum(hp, 1e-6))
                  + (1.0 - w) * np.log(np.maximum(h_child, 0.5)))


def blend_weight(hp, h_child, q0, q1):
    """smoothstep(h_child/h_p; q0, q1): 1 = keep the parent, 0 = take the child."""
    q = h_child / np.maximum(hp, 1e-6)
    x = np.clip((q - q0) / (q1 - q0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def blend_columns(Dp, Dc, w):
    """Same geometric blend, but on the WATER COLUMN D = h + zeta, not on h.

    Blending h and then flooring the result is not safe at a deep low tide: the
    blend target is the child's (smoothed) h, which at zeta = -0.94 m can sit
    below -zeta, so h_p + zeta goes to zero or negative and z_p collapses or
    inverts.  Flooring h_p at Dcrit - zeta repairs the sign but CLIPS, and on
    LV4 2/3 the clip was active on more than 10% of wet points -- a kink in h_p
    right where the bathymetry matters most.

    Blending D instead removes the need for any clip: D_c = h_eff + zeta is
    >= Dcrit by construction and D_p from the OI'd LV1 depth is >= ~10 m, so
    the geometric mean of two positive numbers is positive and smooth.
    """
    return np.exp(w * np.log(np.maximum(Dp, 1e-6))
                  + (1.0 - w) * np.log(np.maximum(Dc, 1e-6)))


def auto_blend_smooth(dx_child):
    """Smoothing window for the blended h_p, from the two grid scales.

    The structure to REMOVE is h_child's own grid-scale detail (dx_child); the
    structure to KEEP is the blend's depth correction, whose width is set by
    where h_p is wrong, i.e. the PARENT's scale.  So the window must sit
    between them: a few child cells, but well below the parent's.

        sigma = min(3 * dx_child, dx_parent / 4)

    Applied to h_child BEFORE it is used as the blend target, NOT to the
    blended h_p.  Smoothing the blend RESULT averages blended cells with their
    unblended neighbours: LV2's corrected fringe is 1-2 cells wide, so even a
    0.8-cell window leaves the centre only ~46% weight and mixes in h_p ~ 60 m
    from next door, undoing the correction entirely (LV2 3/7 ubar 0.097
    unsmoothed -> 0.217 at 480 m -> 0.218 at 1 km).  Smoothing the TARGET
    removes the grid-scale detail at source and leaves the blend intact.
    """
    return min(3.0 * dx_child, DX_PARENT / 4.0)


def smooth_hp(hp_flat, wet_2d, sigma_m, dx):
    """Smooth the BLENDED parent depth before it is used to build z_p.

    The blend mixes h_child into h_p, and h_child carries the child's own
    grid-scale bathymetry (32 m on LV4).  That detail propagates into z_p, so
    the remap samples the parent profile at depths that wobble cell to cell,
    printing the child's bathymetry into the increment as small-scale noise --
    measured on LV4 as a 10-23x inflation of max|grad| versus LV1's native.

    Smoothing in log space (positive by construction) with land filled from
    the nearest wet cell keeps the large-scale depth correction the blend is
    there for, while removing structure the parent could never resolve.
    """
    if not sigma_m:
        return hp_flat
    lf = np.log(np.maximum(hp_flat, 1e-6)).reshape(wet_2d.shape)
    _, (jj, ii) = distance_transform_edt(~wet_2d, return_indices=True)
    filled = np.where(wet_2d, lf, lf[jj, ii])
    return np.exp(gaussian_filter(filled, sigma_m / dx, mode="nearest")).ravel()



def lv1_increment(root, date, tag, G1, m1):
    """One day's LV1 increment, packed into per-stagger field lists.

    Velocity is carried as EARTH components at rho points and rotated onto the
    child's axes after mapping, matching the pipeline's other interpolators.
    """
    o1 = Path(root) / "analysis_files" / f"LV1_fwd_outer1_{date}{tag}.nc"
    o0 = Path(root) / "prior_files" / f"LV1_fwd_outer0_{date}{tag}.nc"
    with nc.Dataset(o1) as a, nc.Dataset(o0) as b:
        dz  = _san(np.asarray(a["zeta"][0])) - _san(np.asarray(b["zeta"][0]))
        dub = _san(np.asarray(a["ubar"][0])) - _san(np.asarray(b["ubar"][0]))
        dvb = _san(np.asarray(a["vbar"][0])) - _san(np.asarray(b["vbar"][0]))
        du  = _san(np.asarray(a["u"][0]))    - _san(np.asarray(b["u"][0]))
        dv  = _san(np.asarray(a["v"][0]))    - _san(np.asarray(b["v"][0]))
        dT  = _san(np.asarray(a["temp"][0])) - _san(np.asarray(b["temp"][0]))
        dS  = _san(np.asarray(a["salt"][0])) - _san(np.asarray(b["salt"][0]))
        z_ref = _san(np.asarray(b["zeta"][0]))        # outer0 -> option (b)
        pvp = dict(Vt=int(np.asarray(a["Vtransform"][...])),
                   hc=float(np.asarray(a["hc"][...])),
                   Cs_r=np.asarray(a["Cs_r"][:]), s_rho=np.asarray(a["s_rho"][:]))
    summary = (f"  LV1 increment {date}: |dzeta|max {np.abs(dz[m1]).max():.4f} m, "
               f"|du|max {np.abs(du[:, m1[:, :-1] & m1[:, 1:]]).max():.4f} m/s, "
               f"|dT|max {np.abs(dT[:, m1]).max():.4f} degC")

    cA, sA = np.cos(G1["angle"]), np.sin(G1["angle"])

    def to_earth(u_lv, v_lv):
        ur = np.zeros_like(G1["lat_rho"]); vr = np.zeros_like(G1["lat_rho"])
        ur[:, 1:-1] = 0.5 * (u_lv[..., :, :-1] + u_lv[..., :, 1:])
        vr[1:-1, :] = 0.5 * (v_lv[..., :-1, :] + v_lv[..., 1:, :])
        return ur * cA - vr * sA, ur * sA + vr * cA

    uEb, vNb = to_earth(dub, dvb)
    uE = np.empty((NLEV_P,) + G1["lat_rho"].shape); vN = np.empty_like(uE)
    for k in range(NLEV_P):
        uE[k], vN[k] = to_earth(du[k], dv[k])

    fields = {
        "rho": [dz[m1]] + [dT[k][m1] for k in range(NLEV_P)]
                        + [dS[k][m1] for k in range(NLEV_P)],
        "u":   [uEb[m1], vNb[m1]] + [uE[k][m1] for k in range(NLEV_P)]
                                  + [vN[k][m1] for k in range(NLEV_P)],
    }
    fields["v"] = fields["u"]          # same source list; different targets
    return fields, z_ref[m1], pvp, summary


def oi_for_dates(child, dates, root, tag, C, G1, m1, h1, t0):
    """Horizontal OI for one or many dates with ONE geometry build per stagger.

    The OI operator depends only on the grids, the correlation length and the
    nugget -- never on the data -- so every date in `dates` can ride on the same
    KD-tree queries, C_ww distance matrices and Cholesky factorisations.  Since
    those dominate the cost (12 s fixed vs 0.05 s per field on LV3), batching
    dates is close to a pure win; see oi_many's docstring.

    zeta_ref rides along in the same call because it uses the same L_INCR, and
    h_p is pulled from hp_cached() because it is identical for every date.
    """
    lon_w, lat_w = G1["lon_rho"][m1], G1["lat_rho"][m1]
    lat0 = float(G1["lat_rho"].mean())
    logh = np.log(h1[m1])
    per_day = [lv1_increment(root, d, tag, G1, m1) for d in dates]
    pvp = per_day[0][2]
    gots = [{} for _ in dates]
    for kind in ("rho", "u", "v"):
        lo, la = (C["lon_rho"], C["lat_rho"]) if kind == "rho" else \
                 (C["lon_u"], C["lat_u"]) if kind == "u" else \
                 (C["lon_v"], C["lat_v"])
        fl, nper = [], 0
        for fields, zr, _, _ in per_day:
            blk = list(fields[kind]) + [zr]      # zeta_ref shares L_INCR
            nper = len(blk)
            fl += blk
        res = oi_many(fl, lon_w, lat_w, lo.ravel(), la.ravel(), lat0, L_INCR,
                      label=f"{kind}x{len(dates)}d", chunk=len(fl))
        hp = hp_cached(child, kind, logh, lon_w, lat_w,
                       lo.ravel(), la.ravel(), lat0)
        for i in range(len(dates)):
            b = res[i*nper:(i+1)*nper]
            gots[i][kind] = dict(fld=b[:-1], zeta=b[-1], hp=hp, shape=lo.shape)
        print(f"  OI {kind}: {nper-1} fields x {len(dates)} date(s) "
              f"-> {lo.size:,} points ({time.perf_counter()-t0:.0f}s)", flush=True)
    return gots, pvp, [p[3] for p in per_day]


def build(child, date, root, vparams_fn, mask_paths, keep_ubar=False,
          dcrit=DCRIT, tag="", blend=(0.5, 1.0), blend_smooth=1000.0,
          blend_space="D", got=None, pvp=None):
    t0 = time.perf_counter()
    root = Path(root)
    G1 = P.load_grid("LV1")
    C = P.load_grid(child)
    h1, hc_grid = read_h("LV1"), read_h(child)
    m1 = G1["mask_rho"]
    vp = read_vparams(vparams_fn)
    if vp["Cs_w"] is None and not keep_ubar:
        raise SystemExit(f"{vparams_fn} has no Cs_w/s_w; needed for the Hz-"
                         f"weighted ubar recompute (use --keep-ubar to skip)")

    # ---- 1-4. LV1 increment + horizontal OI -------------------------------
    # Done by oi_for_dates(), which can share ONE geometry build across several
    # dates; `got`/`pvp` are passed in by the batch driver in that case.
    if got is None:
        gots, pvp, summaries = oi_for_dates(child, [date], root, tag,
                                            C, G1, m1, h1, t0)
        got = gots[0]
        print(summaries[0], flush=True)
    # ---- 5. rotate onto the child's axes ----------------------------------
    ang_u = 0.5 * (C["angle"][:, :-1] + C["angle"][:, 1:]).ravel()
    ang_v = 0.5 * (C["angle"][:-1, :] + C["angle"][1:, :]).ravel()
    U = got["u"]["fld"]; V = got["v"]["fld"]
    ubar_c = U[0] * np.cos(ang_u) + U[1] * np.sin(ang_u)
    vbar_c = -V[0] * np.sin(ang_v) + V[1] * np.cos(ang_v)
    u_c = U[2:2+NLEV_P] * np.cos(ang_u) + U[2+NLEV_P:] * np.sin(ang_u)
    v_c = -V[2:2+NLEV_P] * np.sin(ang_v) + V[2+NLEV_P:] * np.cos(ang_v)
    zeta_c = got["rho"]["fld"][0].reshape(C["lon_rho"].shape)
    T_c = got["rho"]["fld"][1:1+NLEV_P]
    S_c = got["rho"]["fld"][1+NLEV_P:]

    # ---- 6-8. vertical remap on each point set ----------------------------
    hu = 0.5 * (hc_grid[:, :-1] + hc_grid[:, 1:])
    hv = 0.5 * (hc_grid[:-1, :] + hc_grid[1:, :])
    out3d = {}
    diag = {}
    for kind, hch, flds, names in (
            ("rho", hc_grid, (T_c, S_c), ("temp", "salt")),
            ("u", hu, (u_c,), ("u",)),
            ("v", hv, (v_c,), ("v",))):
        g = got[kind]
        ze = g["zeta"]
        h_eff = np.maximum(hch.ravel(), dcrit - ze)
        wet = (C["mask_rho"] if kind == "rho" else
               C["mask_u"] if kind == "u" else
               C["mask_rho"][:-1, :] & C["mask_rho"][1:, :])
        hch_s = (smooth_hp(hch.ravel(), wet, blend_smooth, C["dx"])
                 if blend[0] is not None else hch.ravel())
        if blend_space == "D" and blend[0] is not None:
            # identical smoothstep weight, so the blend engages in exactly the
            # places the h-space version did -- only the blended quantity changes
            w_bl = blend_weight(g["hp"], hch_s, *blend)
            Dc_s = smooth_hp(h_eff + ze, wet, blend_smooth, C["dx"])
            hp_b = blend_columns(g["hp"] + ze, Dc_s, w_bl) - ze
        else:
            hp_b = blend_hp(g["hp"], hch_s, *blend)
        # The child's depth is floored at Dcrit so its column stays physical;
        # the PARENT's must be too.  Without this, blend + a deep low tide can
        # give h_p + zeta <= 0 -- a collapsed or INVERTED parent column, on
        # which np.interp returns arbitrary values.  Found at (-117.109,
        # 32.438) on 2/3: zeta -0.944 m, h_p blended to 0.90-1.14 m, so D_p
        # ran +0.19 .. -0.05 m across three cells and the remapped increment
        # jumped 0.5 degC over 32 m (50 degC/km on the metric in gradchk,
        # 1492 cells above 5 degC/km).
        #
        # In D space this line is a no-op safety net, not load-bearing: the
        # geometric mean of D_p (>= ~10 m, LV1's shallowest wet depth) and
        # D_c (>= Dcrit by construction) cannot fall below Dcrit.  Measured on
        # LV4 2/3, D_parent p01 = 0.358 m, just clear of the 0.350 floor.
        # In h space it IS load-bearing, and that is the problem with h space:
        # it bound on >10% of wet LV4 points (p01 AND p10 both pinned at
        # exactly 0.350 m), kinking h_p where the bathymetry matters most.
        hp_b = np.maximum(hp_b, dcrit - ze)
        g["hp_blend"] = hp_b
        bad = int(((hp_b + ze) <= 0).sum())
        if bad:
            print(f"    WARNING {kind}: {bad} points still have a non-positive "
                  f"parent column after the floor")
        z_p = z_levels(hp_b, ze, pvp["Vt"], pvp["hc"], pvp["Cs_r"], pvp["s_rho"])
        z_c = z_levels(h_eff, ze, vp["Vt"], vp["hc"], vp["Cs_r"], vp["s_rho"])
        wet = wet.ravel()
        fl = np.clip((hch.ravel() - g["hp"]) / np.maximum(hch.ravel(), 0.5),
                     0, 1)[wet & (hch.ravel() > 0.5)] * 100
        diag[kind] = dict(
            floor=100 * (h_eff > hch.ravel() + 1e-9)[wet].mean(),
            flat_med=np.median(fl), flat_p90=np.percentile(fl, 90),
            flat_p99=np.percentile(fl, 99))
        for f, nm in zip(flds, names):
            out3d[nm] = remap_z(f, z_p, z_c).reshape((len(vp["s_rho"]),) + g["shape"])
        if kind == "rho":
            zw_p = None
        got[kind]["z_c"] = z_c
        got[kind]["h_eff"] = h_eff
        # How thin is the parent water column the child column is mapped onto?
        # A very thin D_p squeezes the parent's whole vertical structure into a
        # few cm, so a small cell-to-cell change in D_p shifts which parent
        # level each child level samples -- the mechanism behind the residual
        # gradient spikes.  Watch the low percentiles here.
        Dp = (hp_b + ze)[wet]
        Dc = (h_eff + ze)[wet]
        diag[kind].update(
            Dp_p01=np.percentile(Dp, 1), Dp_p10=np.percentile(Dp, 10),
            Dp_med=np.median(Dp),
            thin=100 * (Dp < 0.5).mean(), squeeze=100 * (Dp < 0.25 * Dc).mean())
        print(f"    D_parent: p01 {diag[kind]['Dp_p01']:.3f} "
              f"p10 {diag[kind]['Dp_p10']:.3f} med {diag[kind]['Dp_med']:.2f} m | "
              f"<0.5 m on {diag[kind]['thin']:.2f}% | "
              f"<1/4 of D_child on {diag[kind]['squeeze']:.2f}%", flush=True)
        print(f"  remap {kind}: {NLEV_P} -> {len(vp['s_rho'])} levels | "
              f"Dcrit floor on {diag[kind]['floor']:.1f}% of WET points | "
              f"blend moved h_p on {100*(np.abs(hp_b-g['hp'])>0.01*g['hp'])[wet].mean():.1f}% | "
              f"% of column flat-extrapolated: med {diag[kind]['flat_med']:.1f}, "
              f"90th {diag[kind]['flat_p90']:.1f}, 99th {diag[kind]['flat_p99']:.1f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    inc = dict(zeta=zeta_c,
               ubar=ubar_c.reshape(C["lon_u"].shape),
               vbar=vbar_c.reshape(C["lon_v"].shape),
               u=out3d["u"], v=out3d["v"], temp=out3d["temp"], salt=out3d["salt"])

    # ---- 9. masks (zeta deliberately excluded) ----------------------------
    if mask_paths:
        mr = load_masks(mask_paths, C, "rho")
        mu = load_masks(mask_paths, C, "u")
        mv = load_masks(mask_paths, C, "v")
        for nm, mm in (("temp", mr), ("salt", mr), ("u", mu), ("v", mv),
                       ("ubar", mu), ("vbar", mv)):
            inc[nm] = inc[nm] * (mm if inc[nm].ndim == 2 else mm[None, ...])
        print(f"  masked u,v,temp,salt,ubar,vbar with {len(mask_paths)} mask(s); "
              f"zeta left unmasked")

    # ---- 10. Hz-weighted barotropic recompute -----------------------------
    if not keep_ubar:
        for kind, nm3, nm2, hch in (("u", "u", "ubar", hu), ("v", "v", "vbar", hv)):
            g = got[kind]
            zw = z_levels(g["h_eff"], g["zeta"], vp["Vt"], vp["hc"],
                          vp["Cs_w"], vp["s_w"])
            Hz = np.diff(zw, axis=0)
            D = Hz.sum(axis=0)
            f = inc[nm3].reshape(inc[nm3].shape[0], -1)
            wet2 = (C["mask_u"] if kind == "u" else
                    C["mask_rho"][:-1, :] & C["mask_rho"][1:, :])
            old = np.abs(inc[nm2][wet2]).max()
            inc[nm2] = ((f * Hz).sum(axis=0) / np.maximum(D, 1e-6)
                        ).reshape(g["shape"])
            # over WATER only: the land values are zeroed later, and reporting
            # them here previously showed 0.217 where the real figure was 0.097
            print(f"  {nm2} re-derived as depth_avg({nm3}) [water only]: "
                  f"|max| {old:.5f} -> {np.abs(inc[nm2][wet2]).max():.5f} m/s")

    # ---- mask out child land ---------------------------------------------
    mk = dict(zeta=C["mask_rho"], temp=C["mask_rho"], salt=C["mask_rho"],
              u=C["mask_u"], ubar=C["mask_u"],
              v=C["mask_rho"][:-1, :] & C["mask_rho"][1:, :],
              vbar=C["mask_rho"][:-1, :] & C["mask_rho"][1:, :])
    for nm, mm in mk.items():
        inc[nm] = np.where(mm if inc[nm].ndim == 2 else mm[None, ...],
                           inc[nm], 0.0)
    return inc, C, vp, diag


def write_nc(out_fn, inc, C, vp, child, date, meta):
    Path(out_fn).parent.mkdir(parents=True, exist_ok=True)
    ny, nx = C["lon_rho"].shape
    N = len(vp["s_rho"])
    with nc.Dataset(out_fn, "w", format="NETCDF4") as d:
        d.createDimension("ocean_time", None)
        d.createDimension("s_rho", N)
        d.createDimension("eta_rho", ny); d.createDimension("xi_rho", nx)
        d.createDimension("eta_u", ny);   d.createDimension("xi_u", nx - 1)
        d.createDimension("eta_v", ny - 1); d.createDimension("xi_v", nx)
        dims = dict(zeta=("ocean_time", "eta_rho", "xi_rho"),
                    ubar=("ocean_time", "eta_u", "xi_u"),
                    vbar=("ocean_time", "eta_v", "xi_v"),
                    u=("ocean_time", "s_rho", "eta_u", "xi_u"),
                    v=("ocean_time", "s_rho", "eta_v", "xi_v"),
                    temp=("ocean_time", "s_rho", "eta_rho", "xi_rho"),
                    salt=("ocean_time", "s_rho", "eta_rho", "xi_rho"))
        units = dict(zeta="m", ubar="m s-1", vbar="m s-1", u="m s-1",
                     v="m s-1", temp="Celsius", salt="1")
        for nm, dd in dims.items():
            v = d.createVariable(nm, "f4", dd, zlib=True, complevel=4)
            v.units = units[nm]
            v.long_name = f"{nm} DA increment, outer1(t=0) - outer0(t=0)"
            v[0] = inc[nm]
        for nm in ("s_rho", "Cs_r"):
            v = d.createVariable(nm, "f8", ("s_rho",)); v[:] = vp[nm.replace("s_rho", "s_rho")] \
                if nm == "s_rho" else vp["Cs_r"]
        for nm, val in (("Vtransform", vp["Vt"]), ("hc", vp["hc"])):
            v = d.createVariable(nm, "f8", ()); v[...] = val
        d.setncatts(meta)
    print(f"\n  wrote {out_fn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", required=True, choices=("LV2", "LV3", "LV4"))
    ap.add_argument("--date", required=True, nargs="+",
                    help="one or more YYYYMMDD, or START..END for an inclusive "
                         "range.  Several dates share ONE OI geometry build per "
                         "stagger, which is where almost all the cost is.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave dates whose --out file already exists.  Off by "
                         "default: a rebuild is the only way to pick up a change "
                         "to the pipeline, and silently keeping stale files is "
                         "the more dangerous failure.  Use it when EXTENDING a "
                         "range, e.g. 3/7..4/20 already built and LV1 has since "
                         "reached 5/7 -- then only the new dates cost anything.")
    ap.add_argument("--days-per-batch", type=int, default=0,
                    help="dates per shared-geometry group (0 = choose from the "
                         "target count to keep peak memory near 2 GB)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--root", default="/dataSIO/DA_Simulations/LV1_v2")
    ap.add_argument("--child-vparams-from", required=True,
                    help="a child IC/history file carrying Vtransform, hc, "
                         "Cs_r, s_rho, Cs_w, s_w")
    ap.add_argument("--mask", action="append", default=[],
                    help="bay/TJRE mask netCDF; repeatable, multiplied together")
    ap.add_argument("--keep-ubar", action="store_true",
                    help="keep the OI-mapped ubar/vbar instead of re-deriving "
                         "them from the masked 3-D fields")
    ap.add_argument("--dcrit", type=float, default=DCRIT)
    ap.add_argument("--blend-space", choices=("h", "D"), default="D",
                    help="blend the water column D=h+zeta (default, positive by "
                         "construction) or the depth h (legacy, needs a "
                         "Dcrit-zeta clip that kinks h_p on LV4)")
    ap.add_argument("--blend-smooth", default="1000",
                    help="smooth the blended h_p over this many METRES before "
                         "building z_p; 'auto' = min(3*dx_child, dx_parent/4), "
                         "0 disables")
    ap.add_argument("--blend", default="0.5,1.0",
                    help="q0,q1 for blending h_p toward the child bathymetry "
                         "where h_child/h_p is small; 'none' disables")
    ap.add_argument("--out", required=True,
                    help="output path; must contain {date} when several dates "
                         "are requested")
    a = ap.parse_args()

    # expand START..END and sort
    dates = []
    for tok in a.date:
        if ".." in tok:
            s0, s1 = tok.split("..")
            d = DT.datetime.strptime(s0, "%Y%m%d")
            end = DT.datetime.strptime(s1, "%Y%m%d")
            while d <= end:
                dates.append(d.strftime("%Y%m%d")); d += DT.timedelta(days=1)
        else:
            dates.append(tok)
    dates = sorted(set(dates))
    if len(dates) > 1 and "{date}" not in a.out:
        raise SystemExit("--out must contain {date} when several dates are given")
    if a.skip_existing:
        keep, done = [], []
        for d in dates:
            (done if Path(a.out.replace("{date}", d)).exists() else keep).append(d)
        if done:
            print(f"  --skip-existing: {len(done)} of {len(dates)} already built "
                  f"({done[0]}..{done[-1]}), {len(keep)} to do", flush=True)
        if not keep:
            print("  nothing to do"); return
        dates = keep

    print(f"  {a.child} <- LV1   {len(dates)} date(s) {dates[0]}..{dates[-1]}"
          f"   L_incr = {L_INCR} km, L_bathy = {L_BATHY} km, "
          f"Dcrit = {a.dcrit} m", flush=True)
    bl = (None, None) if a.blend.lower() == "none" else \
         tuple(float(x) for x in a.blend.split(","))
    dxc = P.load_grid(a.child)["dx"]
    a.blend_smooth = (auto_blend_smooth(dxc) if str(a.blend_smooth) == "auto"
                      else float(a.blend_smooth))
    print(f"  h_p blend q0,q1 = {bl}, smooth = {a.blend_smooth:.0f} m "
          f"({a.blend_smooth/dxc:.1f} child cells, dx_child = {dxc:.0f} m)")
    # Group size.  Peak memory is dominated by the OI output: ~83 fields per
    # date times the target count, held once as float64 inside oi_map and once
    # as float32 in oi_many, plus each date's masked LV1 source fields.
    # Measured on LV3 (103,663 targets): 1 date ~0.9 GB, 5 dates ~1.7 GB, so
    # roughly 0.7 GB fixed + 83*M*25 bytes per date.
    #
    # Size the group against the cgroup headroom actually available, because a
    # SIGKILL loses the whole group.  Two runs were OOM-killed on 2026-10-02
    # this way: once by 3 concurrent LV4 jobs, once by a session already holding
    # 7 of its 8 GB in editor and notebook-kernel processes.
    Ctmp = P.load_grid(a.child)
    M = max(Ctmp["lon_rho"].size, Ctmp["lon_u"].size, Ctmp["lon_v"].size)
    budget = 2.0e9
    try:
        cg = Path("/proc/self/cgroup").read_text().split("::")[1].split("\n")[0]
        base = Path("/sys/fs/cgroup") / cg.lstrip("/")
        for d in (base, base.parent, base.parent.parent):
            mx, st = d / "memory.max", d / "memory.stat"
            if mx.exists() and st.exists():
                lim = mx.read_text().strip()
                if lim == "max":
                    continue
                # Use UNRECLAIMABLE memory, not memory.current.  memory.current
                # counts page cache, and reading LV1's 6.8 GB outer0/outer1
                # files pins the cache right up against memory.max -- which made
                # this report "0.6 GB free" on a cgroup holding only 0.13 GB of
                # anon, and silently dropped the batch to 1 date.
                kv = dict(l.split() for l in st.read_text().splitlines()
                          if " " in l)
                used = sum(int(kv.get(k, 0))
                           for k in ("anon", "slab", "kernel_stack", "sock",
                                     "shmem"))
                free = int(lim) - used
                budget = max(0.5e9, 0.6 * free)
                print(f"  cgroup limit {int(lim)/1e9:.1f} GB, unreclaimable "
                      f"{used/1e9:.2f} GB -> OI budget {budget/1e9:.1f} GB",
                      flush=True)
                break
    except Exception:
        pass
    G = a.days_per_batch or max(1, int((budget - 0.7e9) / (83 * M * 25)))
    groups = [dates[i:i+G] for i in range(0, len(dates), G)]
    print(f"  {len(groups)} shared-geometry group(s) of up to {G} date(s) "
          f"({M:,} target points per stagger)", flush=True)

    t_all = time.perf_counter()
    for gi, grp in enumerate(groups, 1):
        G1 = P.load_grid("LV1"); Cg = P.load_grid(a.child)
        gots, pvp_g, summaries = oi_for_dates(
            a.child, grp, a.root, a.tag, Cg, G1, G1["mask_rho"],
            read_h("LV1"), time.perf_counter())
        for di, d in enumerate(grp):
            print(f"\n  --- group {gi}/{len(groups)}, {d} "
                  f"({(gi-1)*G+di+1}/{len(dates)})", flush=True)
            print(summaries[di], flush=True)
            build_and_write(a, d, bl, got=gots[di], pvp=pvp_g)
        del gots
    print(f"\n  ALL {len(dates)} date(s) in "
          f"{(time.perf_counter()-t_all)/60:.1f} min "
          f"({(time.perf_counter()-t_all)/len(dates):.0f}s per date)")


def build_and_write(a, date, bl, got=None, pvp=None):
    inc, C, vp, diag = build(a.child, date, a.root, a.child_vparams_from,
                             a.mask, a.keep_ubar, a.dcrit, a.tag, bl,
                             a.blend_smooth, a.blend_space, got=got, pvp=pvp)
    meta = dict(
        source=f"LV1 outer1(t=0) - outer0(t=0), {date}",
        horizontal="objective mapping, Matern 5/2, L=%g km, nugget=%g, "
                   "from LV1 WET rho points only (no land fill)"
                   % (L_INCR, OI.NUGGET),
        bathymetry=("h_parent = exp(OI(log h_LV1)), L=%g km, no clamp; "
                    "blended toward h_child (pre-smoothed %g m) over "
                    "q=h_child/h_p in %s"
                    % (L_BATHY, a.blend_smooth, a.blend)),
        zeta_reference="LV1 outer0 zeta(t=0), OI-mapped; common to both "
                       "parent and child depth cubes",
        vertical="per-column linear in z, flat-extrapolated at both ends",
        dcrit="child depth floored at max(h, Dcrit - zeta), Dcrit=%g m" % a.dcrit,
        blend_space=a.blend_space,
        masks=("; ".join(Path(m).name for m in a.mask) or "none")
              + "  (applied to u,v,temp,salt,ubar,vbar; NOT zeta)",
        barotropic=("kept from OI" if a.keep_ubar else
                    "re-derived as Hz-weighted depth average of masked 3-D u,v"),
    )
    out = a.out.replace("{date}", date)
    write_nc(out, inc, C, vp, a.child, date, meta)

    print("  per-variable |max| of the child increment:")
    for nm in ("zeta", "ubar", "vbar", "u", "v", "temp", "salt"):
        print(f"    {nm:5s} {np.abs(inc[nm]).max():.5f}")


if __name__ == "__main__":
    main()
