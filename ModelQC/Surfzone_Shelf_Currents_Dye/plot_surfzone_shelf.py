#!/usr/bin/env python3
"""QC-website plot: surfzone and shelf (10, 12 and 15 m isobaths) currents of the PFM LV4 model at Coronado
Avenida Lunar, the Silver Strand guard shack and the IB Pier.

Python port of Matlab_files/SurfZone_transect_forOneLocation.m (PFM only, no PHM runs), at three stations:
  - cross-shore transect normal to the coastline fitted within FIT_RADIUS of the station (the mask_rho
    = 0.5 contour, as ShorelineOrientation.m); fitted as a total-least-squares line, since the coast runs
    nearly north-south here, and "offshore" is the side with deeper water, so the transect direction
    and the current signs are the same at every station;
  - points every DS m from LAND m onshore of the fitted coastline to past the 15 m isobath;
  - hourly depth-averaged currents (ROMS ubar, vbar) with the wet/dry masks, averaged from the u/v faces
    to rho points, rotated to east/north with the grid angle and then to alongshore (+ up-coast,
    northward) and cross-shore (+ onshore); values on the transect are the inverse-distance-squared
    weighted mean of the K nearest wet rho cells, as in the MATLAB code;
  - shoreline: the last wet transect point going shoreward from offshore (moves with the tide);
  - surfzone: from the shoreline to the first local Hs (Hwave) maximum going offshore (a point higher
    than the points 2 before and 2 after it; else the largest Hs), instead of 1.25 x its distance
    in the MATLAB code. Surfzone currents are the mean over those points;
  - shelf: the points of the same transect where the grid depth h reaches 10, 12 and 15 m (SHELF_DEPTHS);
  - near-surface: at the 12 m point, the 3-D u, v averaged over the top 3 m (TOP_DEPTH, TOP).

Each model hour comes from the latest history file covering it (files run 2.5-5 days ahead); hours no
file covers are left empty. The model lines run past the plot window to the end of the latest file,
dashed after now (or after the window's end, for an older window), with a dotted line at that split.

Cache (/dataSIO/PFM_Simulations/cache/Surfzone_ShelfCurrents/, shared by swell and backwash):
cache/model/<station>_LV4_<YYYYMMDDHHMM>_<hash>.csv holds the hourly results of each history file
(time UTC); <hash> changes with the station position and the transect settings, so changing them
rebuilds the cache. A file is recomputed if the history file is newer. Delete a file to rebuild it.

Times on the plots are Pacific (PDT/PST, America/Los_Angeles); the data are read in UTC and converted.

Daily run (default): past 3 weeks up to the end of today (Pacific), into
plots_web/SurfzoneShelf_<YYYYMMDD>_3weeks.png; older *_3weeks.png are removed so plots_web/ only
holds the newest plot. For any start and end dates use plot_surfzone_shelf_range.py.

  python plot_surfzone_shelf.py                         # past 3 weeks, for the website
  python plot_surfzone_shelf.py --date 2026-09-30       # 3 weeks ending on another day (old plots kept)
  python plot_surfzone_shelf.py --map                   # map of the transects -> plots/transects_map.png
"""
import argparse
import glob
import hashlib
import json
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
import cmocean
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.spatial import cKDTree
import gsw

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "plots_web")
# shared cache on /dataSIO, seen from swell and backwash; the HYDRO_QC_DATA environment variable overrides the root
CACHE_DIR = os.path.join(os.environ.get("HYDRO_QC_DATA", "/dataSIO/PFM_Simulations/cache"), "Surfzone_ShelfCurrents")
MODEL_DIR = os.path.join(CACHE_DIR, "model")  # hourly results of each history file, per station
HIS_DIR = "/dataSIO/PFM_Simulations/Archive/LV4_His"
GRID = "/dataSIO/PFM_Simulations/Grid/GRID_SDTJRE_LV4_mss_oct2024.nc"
EPOCH = pd.Timestamp("1999-01-01")            # ocean_time is seconds since this
DAYS, LABEL = 21, "3weeks"
TZ = "America/Los_Angeles"

# positions of SurfZone_transect_forOneLocation.m
STATIONS = {                                       # plot order, south to north
    "IBPierSouth": dict(name="Imperial Beach Pier (south)", lon=-117.133, lat=32.5787),
    "SSGuardShack": dict(name="Silver Strand, guard shack", lon=-117.1398, lat=32.6249),
    "COAvenidaLunar": dict(name="Coronado, Avenida Lunar", lon=-117.1723, lat=32.674066),
}
FIT_RADIUS = 220.0     # m, coastline cells used for the shoreline orientation (0.002 deg in ShorelineOrientation.m)
DS = 7.0               # m, transect spacing (the MATLAB transect: 80 points over 550 m)
LAND = 100.0           # m, the transect starts this far onshore of the fitted coastline
SHELF_DEPTHS = (10, 12, 15)   # m, grid depth h of the shelf points
ALONG_DEPTH = 12              # m, the shelf point of the alongshore-only figure
WAVE_DEPTH = 15               # m, the shelf point of the waves figure (Hs, Tp, mean direction)
COLUMN_DEPTHS = (2, 5, 8)     # m, extra points of the water-column dye figures only (1 m often dries)
COLUMN_VARS = ("dye", "temp", "salt")   # cached fields of the water-column figures
COLUMN_FIGS = COLUMN_VARS + ("rho",)    # water-column figures; rho (sigma0) is computed from temp, salt, z
COLUMN_PANELS = ("sz",) + tuple(f"{d:g}" for d in COLUMN_DEPTHS + SHELF_DEPTHS)   # surfzone edge, then onshore to offshore
TOP_DEPTH, TOP = 12, 3.0     # near-surface current: mean of the 3-D u, v over the top TOP m at this shelf point
PROFILE_PAST = 100.0          # m, the cached Hs profiles end this far past the deepest shelf point
HS_NEAR = 700.0               # m, offshore end of the nearshore Hs figure
MAX_LENGTH = 4000.0    # m, longest transect offshore of the coastline
K = 4                  # nearest rho cells per transect point (Kinterp)
M_LAT, M_LON = 110540.0, 111320.0             # m per degree latitude, longitude at the equator
COLUMNS = ["sz_along", "sz_cross", "sz_hs", "sz_width", "sz_depth"] + [
    f"shelf{d}_{c}" for d in SHELF_DEPTHS for c in ("along", "cross", "hs", "tp", "dp")] + [
    f"top{TOP_DEPTH}_{c}" for c in ("along", "cross")]


def to_utc(t):
    """Naive Pacific time -> naive UTC."""
    return pd.Timestamp(t).tz_localize(TZ, ambiguous=False, nonexistent="shift_forward").tz_convert("UTC").tz_localize(None)


def to_pacific(s):
    """Series or frame with a naive UTC index -> the same with a naive Pacific index."""
    return s.set_axis(s.index.tz_localize("UTC").tz_convert(TZ).tz_localize(None))

# colours of PTB_TJRE_DyeFactor/plot_style.py
INK, INK2, GRID_C, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_SZ = "#eb6834"
C_SHELF = {10: "#8fbbea", 12: "#2a78d6", 15: "#0d3f80"}            # light to dark blue, shallow to deep
C_TOP = "#c2398a"                                                   # 12 m point, top 3 m
TOP_LABEL = f"Shelf {TOP_DEPTH:g} m, top {TOP:g} m"                  # the other lines are depth-averaged
C_STATION = {"COAvenidaLunar": "#2a78d6", "SSGuardShack": "#2a9d5c", "IBPierSouth": "#7b4fb0"}


# ----------------------------------------------------------------------------- transects
def station_hash(station):
    """Short hash of the station position and the transect settings, part of the cache file names."""
    key = [STATIONS[station]["lon"], STATIONS[station]["lat"], FIT_RADIUS, DS, LAND, SHELF_DEPTHS, MAX_LENGTH, K,
           TOP_DEPTH, TOP, 4]
    return hashlib.sha1(json.dumps(key).encode()).hexdigest()[:10]


def transects():
    """Transect of each station: {station: dict} with the K nearest rho cells (j, i) and weights of
    each point (offshore distance s from the fitted coastline), of each shelf point, the nearest cell of
    each point (for wet/dry), the alongshore and offshore unit vectors (east, north) and the grid box."""
    with Dataset(GRID) as g:
        g.set_auto_mask(False)
        lon, lat, h, mask = (np.asarray(g[k][:], float) for k in ("lon_rho", "lat_rho", "h", "mask_rho"))
    clat = np.cos(np.deg2rad(lat.mean()))
    land = mask < 0.5
    coast = np.zeros_like(land)                                # wet cells next to land: the mask = 0.5 contour
    coast[1:] |= land[:-1]
    coast[:-1] |= land[1:]
    coast[:, 1:] |= land[:, :-1]
    coast[:, :-1] |= land[:, 1:]
    coast &= ~land
    out = {}
    for st, p in STATIONS.items():
        x, y = (lon - p["lon"]) * M_LON * clat, (lat - p["lat"]) * M_LAT    # m from the station
        pts = np.column_stack([x[coast & (np.hypot(x, y) <= FIT_RADIUS)], y[coast & (np.hypot(x, y) <= FIT_RADIUS)]])
        c = pts.mean(axis=0)
        along = np.linalg.svd(pts - c)[2][0]                   # total-least-squares coastline direction
        if along[1] < 0:                                       # alongshore + up-coast (northward)
            along = -along
        off = np.array([along[1], -along[0]])
        p0 = c - np.dot(c, along) * along                      # the station projected onto the coastline
        tree = cKDTree(np.column_stack([x.ravel(), y.ravel()]))
        def side_depth(sign):                                  # summed wet depth 50-500 m to one side
            i = tree.query(p0 + np.arange(50, 501, 50)[:, None] * sign * off)[1]
            return np.where(mask.flat[i] == 1, h.flat[i], 0).sum()
        if side_depth(-1) > side_depth(1):                     # offshore: the side with deeper water
            off = -off
        s = np.arange(-LAND, MAX_LENGTH + DS / 2, DS)
        xy = p0 + s[:, None] * off
        d, idx = tree.query(xy, k=K)
        w = 1 / np.maximum(d, 1) ** 2
        wet = mask.ravel()[idx] == 1
        with np.errstate(invalid="ignore"):
            h_tr = np.where(wet, w * h.ravel()[idx], 0).sum(1) / np.where(wet, w, 0).sum(1)   # nan on land
        shelf = {}
        for dep in SHELF_DEPTHS + COLUMN_DEPTHS:               # first point offshore where h reaches dep
            deep = np.flatnonzero((s > 0) & (h_tr >= dep))
            if not len(deep):
                raise ValueError(f"{st}: no {dep:g} m depth within {MAX_LENGTH:g} m offshore")
            k = deep[0]
            s_shelf = np.interp(dep, h_tr[k - 1:k + 1], s[k - 1:k + 1]) if np.isfinite(h_tr[k - 1]) else s[k]
            ds_, idx_shelf = tree.query(p0 + s_shelf * off, k=K)
            shelf[dep] = dict(s=s_shelf, idx=np.unravel_index(idx_shelf, lon.shape), w=1 / np.maximum(ds_, 1) ** 2)
        column = {d: shelf.pop(d) for d in COLUMN_DEPTHS}      # not shelf points: only for the water-column dye
        cells = np.concatenate([idx.ravel()] + [np.ravel_multi_index(v["idx"], lon.shape) for v in shelf.values()])
        jj, ii = np.unravel_index(cells, lon.shape)
        # grid box, kept one cell off the edges for the u/v faces around each rho cell
        box = (max(jj.min() - 1, 1), min(jj.max() + 2, lon.shape[0] - 1), max(ii.min() - 1, 1), min(ii.max() + 2, lon.shape[1] - 1))
        out[st] = dict(s=s, idx=np.unravel_index(idx, lon.shape), w=w, near=np.unravel_index(idx[:, 0], lon.shape),
                       shelf=shelf, column=column, along=along, off=off, box=box, h=h_tr,
                       lon=xy[:, 0] / (M_LON * clat) + p["lon"], lat=xy[:, 1] / M_LAT + p["lat"])
    return out


# ----------------------------------------------------------------------------- model
def history_files():
    """{first hour (UTC): path} of the LV4 history files, only the standard LV4_ocean_his_<YYYYMMDDHHMM>.nc
    names: failed runs such as LV4_ocean_his_202610110000_dt2_BLOWUP.nc share the date."""
    files = {}
    for f in sorted(glob.glob(os.path.join(HIS_DIR, "LV4_ocean_his_*.nc"))):
        m = re.search(r"LV4_ocean_his_(\d{12})\.nc$", f)
        if m:
            files[pd.Timestamp(m.group(1)[:8] + "T" + m.group(1)[8:])] = f
    return files


def cache_path(station, path, kind="csv"):
    """Cache file of one station and history file: kind "csv" (hourly results), "npz" (Hs and surface dye
    along the transect) or "dyez" (dye, temperature and salinity over the water column)."""
    stamp = re.search(r"(\d{12})\.nc$", path).group(1)
    name = f"{station}_LV4_{stamp}_{station_hash(station)}"
    return os.path.join(MODEL_DIR, name + {"csv": ".csv", "npz": "_hs.npz", "dyez": "_column4.npz"}[kind])


def face_mean(a, b):
    """Mean of the two faces of each rho cell, of the wet faces only (nan if both are dry)."""
    n = np.isfinite(a).astype(float) + np.isfinite(b)
    with np.errstate(invalid="ignore"):
        return np.where(n > 0, (np.nan_to_num(a) + np.nan_to_num(b)) / n, np.nan)


def idw(f, jj, ii, w):
    """Inverse-distance weighted mean over the K cells of each point, of the finite values only.
    f: time x box rows x box columns; jj, ii: points x K (box indices); -> time x points."""
    v = f[:, jj, ii]
    wt = np.where(np.isfinite(v), w, 0)
    with np.errstate(invalid="ignore"):
        return np.where(wt.sum(-1) > 0, (np.nan_to_num(v) * wt).sum(-1) / wt.sum(-1), np.nan)


def compute(path, tr):
    """Hourly surfzone and shelf results of one history file at one station (frame, index UTC), and
    the Hs and surface dye (dye_01 + dye_02) profiles along the transect: dict(time, hs, dye (time x points up to PROFILE_END(tr), nan onshore
    of the shoreline), s_sl and s_edge (offshore distance of the shoreline and the surfzone edge))."""
    j0, j1, i0, i1 = tr["box"]
    with Dataset(path) as nc:
        nc.set_auto_mask(False)
        t = (EPOCH + pd.to_timedelta(np.asarray(nc["ocean_time"][:], float), unit="s")).round("h")
        wu = nc["wetdry_mask_u"][:, j0:j1, i0 - 1:i1] > 0
        u = np.where(wu, nc["ubar"][:, j0:j1, i0 - 1:i1], np.nan)
        wv = nc["wetdry_mask_v"][:, j0 - 1:j1, i0:i1] > 0
        v = np.where(wv, nc["vbar"][:, j0 - 1:j1, i0:i1], np.nan)
        wet = nc["wetdry_mask_rho"][:, j0:j1, i0:i1] > 0
        hs = np.where(wet, nc["Hwave"][:, j0:j1, i0:i1], np.nan)
        # peak period and mean wave direction (Dwave; cached as "dp") at the nearest cell of each shelf
        # point (periods come in discrete bins, directions do not average linearly)
        near = {dep: (sh["idx"][0][0] - j0, sh["idx"][1][0] - i0) for dep, sh in tr["shelf"].items()}
        tp, dp = (nc[k][:, j0:j1, i0:i1] for k in ("Pwave_top", "Dwave"))
        # total dye (PTB dye_01 + TJRE dye_02) in the surface layer
        dye = np.where(wet, nc["dye_01"][:, -1, j0:j1, i0:i1] + nc["dye_02"][:, -1, j0:j1, i0:i1], np.nan)
        zeta = nc["zeta"][:, j0:j1, i0:i1]
        top = top_mean(nc, tr)
    with Dataset(GRID) as g:
        g.set_auto_mask(False)
        a = g["angle"][j0:j1, i0:i1]
        wet &= g["mask_rho"][j0:j1, i0:i1] == 1
        depth = np.where(wet, g["h"][j0:j1, i0:i1] + zeta, np.nan)
    ur, vr = face_mean(u[..., :-1], u[..., 1:]), face_mean(v[:, :-1], v[:, 1:])
    ue, vn = ur * np.cos(a) - vr * np.sin(a), ur * np.sin(a) + vr * np.cos(a)       # east, north
    along = ue * tr["along"][0] + vn * tr["along"][1]
    cross = -(ue * tr["off"][0] + vn * tr["off"][1])                               # + onshore
    jj, ii = tr["idx"][0] - j0, tr["idx"][1] - i0
    al, cr, h_s, d_s, dy = (idw(f, jj, ii, tr["w"]) for f in (along, cross, hs, depth, dye))
    wet_tr = wet[:, tr["near"][0] - j0, tr["near"][1] - i0]
    # shoreline: the last wet point of the run of wet points from the offshore end
    from_off = np.flip(np.cumprod(np.flip(wet_tr, 1), 1), 1).astype(bool)
    out = pd.DataFrame(np.nan, index=pd.DatetimeIndex(t, name="time"), columns=COLUMNS)
    npro = profile_points(tr)
    prof = dict(time=t.values, hs=np.full((len(t), npro), np.nan, np.float32), dye=np.full((len(t), npro), np.nan, np.float32),
                s_sl=np.full(len(t), np.nan, np.float32), s_edge=np.full(len(t), np.nan, np.float32))
    for n in range(len(t)):
        if not from_off[n].any():
            continue
        k0 = np.argmax(from_off[n])
        prof["hs"][n, k0:] = h_s[n, k0:npro]
        prof["dye"][n, k0:] = dy[n, k0:npro]
        prof["s_sl"][n] = tr["s"][k0]
        hsn = h_s[n, k0:]
        # first local Hs maximum going offshore (SurfZone_transect_forOneLocation.m), else the largest
        km = next((k for k in range(2, len(hsn) - 2) if hsn[k] > hsn[k - 2] and hsn[k] > hsn[k + 2]), None)
        if km is None:
            if not np.isfinite(hsn).any():
                continue
            km = int(np.nanargmax(hsn))
        seg = slice(k0, k0 + km + 1)
        prof["s_edge"][n] = tr["s"][k0 + km]
        with warnings.catch_warnings():                       # all-nan segments give nan
            warnings.simplefilter("ignore", RuntimeWarning)
            out.iloc[n, :5] = [np.nanmean(al[n, seg]), np.nanmean(cr[n, seg]), hsn[km],
                               tr["s"][k0 + km] - tr["s"][k0], d_s[n, k0 + km]]
    for dep, sh in tr["shelf"].items():
        jj, ii = sh["idx"][0][None] - j0, sh["idx"][1][None] - i0
        for c, f in (("along", along), ("cross", cross), ("hs", hs)):
            out[f"shelf{dep}_{c}"] = idw(f, jj, ii, sh["w"][None])[:, 0]
        out[f"shelf{dep}_tp"], out[f"shelf{dep}_dp"] = tp[:, near[dep][0], near[dep][1]], dp[:, near[dep][0], near[dep][1]]
    out[f"top{TOP_DEPTH}_along"], out[f"top{TOP_DEPTH}_cross"] = top
    return out, prof


def top_mean(nc, tr):
    """Alongshore and cross-shore (+ onshore) mean of the 3-D u, v over the top TOP m of the water
    column at the TOP_DEPTH shelf point (layer thicknesses from the ROMS Vtransform 2 levels; the
    point is always wet, so no wet/dry masks)."""
    jj, ii = tr["shelf"][TOP_DEPTH]["idx"]
    j0, j1, i0, i1 = jj.min(), jj.max() + 1, ii.min(), ii.max() + 1
    nk = len(nc.dimensions["s_rho"]) // 2          # the top TOP m is always in the upper half of the layers
    u = nc["u"][:, nk:, j0:j1, i0 - 1:i1]           # (one stored chunk of the 10 levels: about twice as fast)
    v = nc["v"][:, nk:, j0 - 1:j1, i0:i1]
    zeta = nc["zeta"][:, j0:j1, i0:i1][:, None]
    s_w, cs_w, hc = (np.asarray(nc[k][:], float) for k in ("s_w", "Cs_w", "hc"))
    with Dataset(GRID) as g:
        g.set_auto_mask(False)
        h, a = (g[k][j0:j1, i0:i1] for k in ("h", "angle"))
    s_w, cs_w = s_w[nk:], cs_w[nk:]
    S = (hc * s_w[:, None, None] + h * cs_w[:, None, None]) / (hc + h)                       # N+1 x j x i
    z_w = zeta + (zeta + h) * S                                                               # time x N+1 x j x i
    dz = np.clip(np.minimum(z_w[:, 1:], zeta) - np.maximum(z_w[:, :-1], zeta - TOP), 0, None)  # layer part in the top TOP m
    ur = ((0.5 * (u[..., :-1] + u[..., 1:])) * dz).sum(1) / dz.sum(1)
    vr = ((0.5 * (v[:, :, :-1] + v[:, :, 1:])) * dz).sum(1) / dz.sum(1)
    ue, vn = ur * np.cos(a) - vr * np.sin(a), ur * np.sin(a) + vr * np.cos(a)
    along = ue * tr["along"][0] + vn * tr["along"][1]
    cross = -(ue * tr["off"][0] + vn * tr["off"][1])
    w = tr["shelf"][TOP_DEPTH]["w"][None]
    return (idw(f, (jj - j0)[None], (ii - i0)[None], w)[:, 0] for f in (along, cross))


def compute_dyez(path, tr, s_edge):
    """Total dye (dye_01 + dye_02), temperature and salinity in every layer, and the depth z of each layer
    (m, + up, moving with zeta; ROMS Vtransform 2): at the nearest cell of each COLUMN_DEPTHS and shelf
    point (dye<d>, temp<d>, salt<d>, z<d>), and at the nearest cell of the surfzone edge (first Hs
    maximum) of each hour, s_edge (dyesz, ..., zsz). dict(time, ... (time x layers))."""
    cells = {dep: (sh["idx"][0][0], sh["idx"][1][0]) for dep, sh in tr["shelf"].items()}
    cells.update({dep: (pt["idx"][0][0], pt["idx"][1][0]) for dep, pt in tr["column"].items()})
    j0, j1, i0, i1 = tr["box"]
    with Dataset(path) as nc:
        nc.set_auto_mask(False)
        t = (EPOCH + pd.to_timedelta(np.asarray(nc["ocean_time"][:], float), unit="s")).round("h")
        fields = {"dye": nc["dye_01"][:, :, j0:j1, i0:i1] + nc["dye_02"][:, :, j0:j1, i0:i1],
                  "temp": nc["temp"][:, :, j0:j1, i0:i1], "salt": nc["salt"][:, :, j0:j1, i0:i1]}
        zeta = nc["zeta"][:, j0:j1, i0:i1]
        s_r, cs_r, hc = (np.asarray(nc[k][:], float) for k in ("s_rho", "Cs_r", "hc"))
    with Dataset(GRID) as g:
        g.set_auto_mask(False)
        h = g["h"][j0:j1, i0:i1]
    out = dict(time=t.values)
    for dep, (j, i) in cells.items():
        hh, z0 = h[j - j0, i - i0], zeta[:, j - j0, i - i0][:, None]
        out[f"z{dep:g}"] = (z0 + (z0 + hh) * (hc * s_r + hh * cs_r) / (hc + hh)).astype(np.float32)
        for v, f in fields.items():
            out[f"{v}{dep:g}"] = f[:, :, j - j0, i - i0].astype(np.float32)
    # surfzone edge: the nearest cell of the edge point of each hour
    for v in list(fields) + ["z"]:
        out[f"{v}sz"] = np.full((len(t), len(s_r)), np.nan, np.float32)
    for n in np.flatnonzero(np.isfinite(s_edge)):
        k = np.argmin(abs(tr["s"] - s_edge[n]))
        j, i = tr["near"][0][k] - j0, tr["near"][1][k] - i0
        for v, f in fields.items():
            out[f"{v}sz"][n] = f[n, :, j, i]
        out["zsz"][n] = zeta[n, j, i] + (zeta[n, j, i] + h[j, i]) * (hc * s_r + h[j, i] * cs_r) / (hc + h[j, i])
    return out


def profile_points(tr):
    """Number of transect points kept in the Hs profiles: up to PROFILE_PAST m past the deepest shelf point."""
    return int(np.searchsorted(tr["s"], max(sh["s"] for sh in tr["shelf"].values()) + PROFILE_PAST)) + 1


def fresh(c, path):
    return os.path.exists(c) and os.path.getmtime(c) >= os.path.getmtime(path)


def process(path, trs):
    """Compute and cache the stations of one history file that have no up-to-date cache files."""
    todo = {st: tr for st, tr in trs.items()
            if not (fresh(cache_path(st, path), path) and fresh(cache_path(st, path, "npz"), path))}
    for st, tr in todo.items():
        df, prof = compute(path, tr)
        c = cache_path(st, path)
        df.to_csv(c + ".part", float_format="%.5g")
        os.replace(c + ".part", c)
        c = cache_path(st, path, "npz")
        with open(c + ".part", "wb") as f:                     # Hs to 1 mm is plenty: float16
            np.savez_compressed(f, **{**prof, "hs": prof["hs"].astype(np.float16)})
        os.replace(c + ".part", c)
    done = list(todo)
    for st, tr in trs.items():          # water-column dye, its own cache file: added without rebuilding the rest
        c = cache_path(st, path, "dyez")
        if not fresh(c, path):
            s_edge = np.load(cache_path(st, path, "npz"))["s_edge"]
            with open(c + ".part", "wb") as f:
                np.savez_compressed(f, **compute_dyez(path, tr, s_edge))
            os.replace(c + ".part", c)
            done += [] if st in done else [st]
    return os.path.basename(path), done


def merge_npz(npz, keys, hours):
    """The keys of a run of cached npz files on the given hours, each hour from the latest file (nan if none)."""
    t = pd.DatetimeIndex(np.concatenate([z["time"] for z in npz]))
    last = ~t.duplicated(keep="last")                                   # later files overwrite earlier ones
    pos = pd.Series(np.flatnonzero(last), index=t[last]).reindex(hours)
    ok = pos.notna().to_numpy()
    out = {}
    for k in keys:
        a = np.concatenate([z[k] for z in npz]).astype(np.float32)
        out[k] = np.full((len(hours),) + a.shape[1:], np.nan, np.float32)
        out[k][ok] = a[pos[ok].astype(int)]
    return out


def model_currents(start, end, workers=4):
    """{station: hourly frame} from start to the end of the latest history file issued by end (UTC),
    each hour from the latest file covering it; hours no file covers are nan. Also the Hs profiles
    the same way: {station: dict(time, s, hs (time x points), s_sl, s_edge)}."""
    files = history_files()
    use = sorted(t for t in files if start - pd.Timedelta(days=6) <= t <= end)
    if not use:
        raise SystemExit(f"no LV4 history files between {start - pd.Timedelta(days=6)} and {end}")
    trs = transects_cache()
    os.makedirs(MODEL_DIR, exist_ok=True)
    paths = [files[t] for t in use]
    with ProcessPoolExecutor(max(1, min(workers, len(paths)))) as ex:
        for name, done in ex.map(process, paths, [trs] * len(paths)):
            if done:
                print("computed", name, ", ".join(done))
    out, profs = {}, {}
    for st in STATIONS:
        df = pd.concat([pd.read_csv(cache_path(st, p), index_col="time", parse_dates=["time"]) for p in paths])
        df = df[~df.index.duplicated(keep="last")].sort_index()        # later files overwrite earlier ones
        hours = pd.date_range(start, max(end, df.index.max()), freq="h")
        out[st] = df.reindex(hours)
        prof = dict(time=hours, s=trs[st]["s"][:profile_points(trs[st])])
        prof.update(merge_npz([np.load(cache_path(st, p, "npz")) for p in paths], ("hs", "dye", "s_sl", "s_edge"), hours))
        prof["column"] = merge_npz([np.load(cache_path(st, p, "dyez")) for p in paths],
                                   [f"{v}{k}" for k in COLUMN_PANELS for v in COLUMN_VARS + ("z",)], hours)
        add_sigma0(prof["column"], STATIONS[st]["lon"], STATIONS[st]["lat"])
        profs[st] = prof
    return out, profs


def add_sigma0(col, lon, lat):
    """Add rho<panel> to the water-column fields: potential density anomaly sigma0 (kg m-3, TEOS-10, gsw)
    from the model's potential temperature, practical salinity and layer depth z."""
    for k in COLUMN_PANELS:
        p = gsw.p_from_z(col[f"z{k}"], lat)
        sa = gsw.SA_from_SP(col[f"salt{k}"], p, lon, lat)
        col[f"rho{k}"] = gsw.sigma0(sa, gsw.CT_from_pt(sa, col[f"temp{k}"]))


# ----------------------------------------------------------------------------- plots
STYLE = {"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
         "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK,
         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
         "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.6,
         "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
         "font.size": 13, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
         "legend.frameon": False}
AHEAD = Line2D([], [], color=INK2, lw=1.5, ls="--", label="Future dates (dashed)")


def model_line(ax, s, split, **kw):
    """Model series solid up to split, dashed after it (hours still ahead)."""
    ax.plot(s[:split].index, s[:split], **kw)
    kw.pop("label", None)
    ax.plot(s[split:].index, s[split:], ls="--", **kw)


def time_axis(axs, start, xend, split, days):
    """Shared time axis of a column of panels, with the dotted line at now."""
    for ax in axs:
        ax.axvline(split, color=INK2, lw=0.8, ls=":")                 # now: hours still ahead to the right
        ax.tick_params(axis="x", which="minor", length=3, color=INK2)
    axs[0].set_xlim(start, xend)
    axs[-1].set_xlabel("Date (Pacific time, PDT/PST)")
    loc = mdates.DayLocator() if days <= 45 else mdates.AutoDateLocator()   # a labelled tick each day
    axs[-1].xaxis.set_major_locator(loc)
    axs[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    axs[-1].xaxis.set_minor_locator(mdates.DayLocator() if days <= 60 else mdates.MonthLocator())


def fit_legends(fig):
    """Fewer legend columns (more rows) until each panel legend is no wider than its panel, and the panel
    titles moved up clear of a legend that sits above the panel."""
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        width = ax.get_window_extent(r).width
        handles, labels = list(leg.legend_handles), [t.get_text() for t in leg.texts]
        while leg._ncols > 1 and leg.get_window_extent(r).width > width:   # a new legend, same settings
            anchor = leg._bbox_to_anchor._bbox.bounds if leg._bbox_to_anchor is not None else None   # axes fraction
            leg = ax.legend(handles, labels, loc=leg._loc, bbox_to_anchor=anchor,
                            ncol=leg._ncols - 1, fontsize=leg.prop.get_size(), frameon=leg.get_frame_on(),
                            facecolor=leg.get_frame().get_facecolor(), edgecolor=leg.get_frame().get_edgecolor())
        top, box = leg.get_window_extent(r).y1, ax.get_window_extent(r)
        if leg.get_window_extent(r).y0 >= box.y1 - 1:              # legend above the panel, below the title
            for t in (ax.title, ax._left_title, ax._right_title):
                if t.get_text():
                    gap = top + 4 * fig.dpi / 72 - t.get_window_extent(r).y0
                    if gap > 0:
                        t.set_y(t.get_position()[1] + gap / box.height)
                        ax._autotitlepos = False                   # keep it there when the figure is drawn


def fit_titles(fig):
    """Re-wrap each panel title (word by word) so that it is no wider than its panel."""
    fig.canvas.draw()                        # final panel positions (fixed-aspect maps shrink when drawn)
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        if hasattr(ax, "_colorbar"):
            continue
        # the title may also run over a colour bar to the right of the panel (with its labels), or other
        # axes listed in ax.title_span
        box = ax.get_window_extent(r)
        span = getattr(ax, "title_span", []) + [a for a in fig.axes if hasattr(a, "_colorbar")
                                                and a.get_window_extent(r).x0 >= box.x1]
        width = max([box.x1] + [a.get_tightbbox(r).x1 for a in span]) - box.x0
        for t in (ax.title, ax._left_title, ax._right_title):
            text = t.get_text()
            if not text or t.get_window_extent(r).width <= width:
                continue
            out = []
            for para in text.split("\n"):
                line = ""
                for word in para.split(" "):
                    t.set_text(f"{line} {word}" if line else word)
                    if line and t.get_window_extent(r).width > width:
                        out.append(line)
                        line = word
                    else:
                        line = f"{line} {word}" if line else word
                out.append(line)
            t.set_text("\n".join(out))


def save(fig, path):
    fit_titles(fig)
    fit_legends(fig)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def variant_path(path, suffix):
    """Path of another figure next to the full one: <name>_<suffix>.png."""
    root, ext = os.path.splitext(path)
    return f"{root}_{suffix}{ext}"


WEB_VARIANTS = ("alongshore", "dye_to15m", "dyerho_10-15m", "waves",
                "surfzone")   # the figures the QC website keeps in plots_web (+ the DyeAlongshore ones, main())
VARIANTS = ("alongshore", "hs_0-700m", "hs_to15m", "dye_0-700m", "dye_to15m", "surfzone", "waves",
            "dyerho_10-15m", "salttemp_10-15m") + tuple(
    f"{v}column_{st}" for v in COLUMN_FIGS for st in STATIONS)   # the figures saved next to the full one


def plot_surfzone(data, profs, start, end, xend, split, days, lw, path):
    """Surfzone edge (first Hs maximum from shore), one line per station: Hs there (sz_hs), surfzone
    width from the shoreline (sz_width), water depth h + zeta there (sz_depth), and the position of the
    moving shoreline relative to the fixed (fitted) coastline (s_sl, + offshore)."""
    panels = (("sz_hs", "H$_s$ (m)", "H$_s$ at the surfzone edge (sz_hs)"),
              ("sz_width", "m", "Surfzone width, shoreline to edge (sz_width)"),
              ("sz_depth", "m", "Water depth h + ζ at the edge (sz_depth)"),
              ("s_sl", "m", "Shoreline (last wet point) from the fixed coastline, + offshore (s_sl)"))
    fig, axs = plt.subplots(len(panels), 1, figsize=(12, 11), sharex=True, gridspec_kw={"hspace": 0.4})
    for n, (ax, (col, unit, desc)) in enumerate(zip(axs, panels)):
        for st, df in data.items():
            s = df[col] if col in df else pd.Series(profs[st][col], index=profs[st]["time"])
            model_line(ax, s, split, color=C_STATION[st], lw=lw, label=STATIONS[st]["name"])
        if col == "s_sl":                                       # can be onshore of the fitted coastline
            ax.axhline(0, color=INK2, lw=0.6)
        else:
            ax.set_ylim(bottom=0)
        ax.set_ylabel(unit)
        ax.set_title(desc if n else
                     f"PFM LV4 surfzone edge (first H$_s$ maximum from shore), {start:%Y-%m-%d} to "
                     f"{(end - pd.Timedelta(seconds=1)):%Y-%m-%d}, latest run to {xend:%Y-%m-%d %H:%M}\n\n{desc}",
                     pad=32 if n == 0 else 6)
    axs[0].legend(handles=axs[0].get_legend_handles_labels()[0] + [AHEAD], loc="lower right", bbox_to_anchor=(1, 1),
                  ncol=4, fontsize=11)
    time_axis(axs, start, xend, split, days)
    save(fig, path)


def plot_waves(data, start, end, xend, split, days, lw, path):
    """Waves at the WAVE_DEPTH shelf point, one line per station: Hs, peak period Tp and mean direction (Dwave)."""
    panels = (("hs", "H$_s$ (m)", "Significant wave height H$_s$ (Hwave)"),
              ("tp", "s", "Peak period T$_p$ (Pwave_top)"),
              ("dp", "° from N", "Mean wave direction D$_m$, coming from (Dwave)"))
    fig, axs = plt.subplots(len(panels), 1, figsize=(12, 9), sharex=True, gridspec_kw={"hspace": 0.4})
    for n, (ax, (col, unit, desc)) in enumerate(zip(axs, panels)):
        for st, df in data.items():
            model_line(ax, df[f"shelf{WAVE_DEPTH}_{col}"], split, color=C_STATION[st], lw=lw, label=STATIONS[st]["name"])
        ax.set_ylabel(unit)
        if col != "dp":
            ax.set_ylim(bottom=0)
        ax.set_title(desc if n else
                     f"PFM LV4 waves at the {WAVE_DEPTH:g} m isobath, {start:%Y-%m-%d} to "
                     f"{(end - pd.Timedelta(seconds=1)):%Y-%m-%d}, latest run to {xend:%Y-%m-%d %H:%M}\n\n{desc}",
                     pad=32 if n == 0 else 6)
    axs[0].legend(handles=axs[0].get_legend_handles_labels()[0] + [AHEAD], loc="lower right", bbox_to_anchor=(1, 1),
                  ncol=4, fontsize=11)
    time_axis(axs, start, xend, split, days)
    save(fig, path)


# fields of the time x offshore-distance figures: title, colour bar label, colormap, colour of the lines on top
PROFILE_FIELDS = {"hs": ("H$_s$", "H$_s$ (m)", "viridis", "white"),
                  "dye": ("surface dye (dye_01 PTB + dye_02 TJRE)", "Dye (unitless fraction), log scale", cmocean.cm.matter, INK)}
DYE_RANGE = (10 ** -5.5, 10 ** -1.5)  # fixed dye colour scale; colormap as Matlab_files/Hovmoller_plot_dye.m
DYE_STEPS = 4                   # colour blocks per power of ten (no smooth transition)


def dye_scale():
    """Stepped log colour scale of the dye figures: (colormap, norm)."""
    bounds = 10 ** np.linspace(*np.log10(DYE_RANGE), round(np.log10(DYE_RANGE[1] / DYE_RANGE[0]) * DYE_STEPS) + 1)
    cmap = ListedColormap(cmocean.cm.matter(np.linspace(0, 1, len(bounds) - 1)))
    return cmap, BoundaryNorm(bounds, cmap.N)


def dye_colorbar(fig, m, axs, label, **kw):
    """Colour bar of a dye figure, with ticks at the powers of ten only; kw: fig.colorbar options."""
    cb = fig.colorbar(m, label=label, **(kw or dict(ax=axs, shrink=0.6, pad=0.06)))
    exps = np.arange(np.ceil(np.log10(DYE_RANGE[0])), np.log10(DYE_RANGE[1]) + 0.5).astype(int)
    cb.set_ticks(10.0 ** exps, labels=[f"$10^{{{e}}}$" for e in exps])
    cb.minorticks_off()


# water-column figures: title, colour bar label, colormap (None: the stepped dye scale)
COLUMN_FIELDS = {"dye": ("dye (dye_01 PTB + dye_02 TJRE)", PROFILE_FIELDS["dye"][1], None),
                 "temp": ("temperature", "Temperature (°C)", cmocean.cm.thermal),
                 "salt": ("salinity", "Salinity", cmocean.cm.haline),
                 "rho": ("potential density", r"$\sigma_0$ (kg m$^{-3}$)", cmocean.cm.dense)}
# temperature colour scale: width and colour step (the scale widens if the data needs more);
# variables not listed (salinity, density) use the 1st-99th percentile
COLUMN_SCALE = {"temp": (5.0, 0.25)}


def nice_step(x):
    """The round number (1, 2, 2.5 or 5 x 10^n) nearest above x."""
    e = 10 ** np.floor(np.log10(x))
    return next(m * e for m in (1, 2, 2.5, 5, 10) if m * e >= x)


def column_norm(profs, var, cmap, panels=COLUMN_PANELS, fixed=True):
    """Temperature, salinity or density colour scale shared by all stations (over the panels given): COLUMN_SCALE
    width centred on the median (rounded), widened to the 1st-99th percentile if that is wider (keeping about
    20 round steps); without a COLUMN_SCALE entry, or fixed False, the 1st-99th percentile."""
    allv = np.concatenate([p["column"][f"{var}{k}"].ravel() for p in profs.values() for k in panels])
    p1, med, p99 = np.nanpercentile(allv, [1, 50, 99])
    if fixed and var in COLUMN_SCALE:
        width, step = COLUMN_SCALE[var]
        lo = np.round((med - width / 2) / (2 * step)) * 2 * step
        hi = lo + width
    else:
        lo, hi, step = med, med, 0
    if p1 < lo or p99 > hi:
        lo, hi = min(lo, p1), max(hi, p99)
        step = max(step, nice_step((hi - lo) / 20))         # about 20 round steps
        lo, hi = np.floor(lo / step) * step, np.ceil(hi / step) * step
    return BoundaryNorm(np.arange(lo, hi + step / 2, step), cmap.N, extend="both")


def column_panel(ax, p, var, key, cmap, norm):
    """One time x depth panel of a water-column figure: var at panel key ("sz" or an isobath); the mappable."""
    val, z = p["column"][f"{var}{key}"], p["column"][f"z{key}"]
    if key == "sz":
        bottom = np.nanmin(z[:, 0]) if np.isfinite(z).any() else -3
    else:
        bottom = -float(key)
    rest = np.nanmedian(z, axis=0)                     # hours without data: still-water levels, nothing drawn
    z = np.where(np.isfinite(z), z, rest)
    f = np.where(np.isfinite(val), np.maximum(val, norm.vmin), np.nan) if var == "dye" else val
    x = np.broadcast_to(p["time"].values[:, None], z.shape)
    m = ax.pcolormesh(x, z, f, cmap=cmap, norm=norm, shading="nearest", rasterized=True)
    ax.set_ylim(bottom - 0.5, 1.5)
    ax.set_ylabel("z (m)")
    ax.grid(False)
    return m


GRID_DEPTHS = ("10", "15")   # isobaths of the QC-website 3 x 4 figures
# QC-website 3 x 4 figures: suffix, the two fields (columns: field 1 and 2 at 10 m, then at 15 m)
GRID_FIGS = {"dyerho_10-15m": ("dye", "rho"), "salttemp_10-15m": ("salt", "temp")}
GRID_NAMES = {"dye": "Dye", "rho": r"Density $\sigma_0$", "salt": "Salinity", "temp": "Temperature"}


def plot_column_grid(profs, pair, start, end, xend, split, days, path):
    """QC-website figure: the two fields of pair over the water column at the GRID_DEPTHS isobaths
    (columns), one row per station; one colour scale per field, shared by the stations."""
    scales = {}
    for var in pair:
        cmap = COLUMN_FIELDS[var][2]
        scales[var] = dye_scale() if cmap is None else (cmap, column_norm(profs, var, cmap, GRID_DEPTHS, fixed=False))
    panels = [(var, key) for key in GRID_DEPTHS for var in pair]
    fig, axs = plt.subplots(len(STATIONS), len(panels), figsize=(22, 10), sharex=True,
                            gridspec_kw={"hspace": 0.35, "wspace": 0.18})
    fig.subplots_adjust(bottom=0.17)
    mappable = {}
    for i, (st, p) in enumerate(profs.items()):
        for j, (var, key) in enumerate(panels):
            ax = axs[i, j]
            mappable[var] = column_panel(ax, p, var, key, *scales[var])
            if j:
                ax.set_ylabel("")
            ax.set_title(f"{STATIONS[st]['name']}\n{GRID_NAMES[var]}, {key} m isobath", fontsize=12, pad=4)
    fig.suptitle(f"PFM LV4 {COLUMN_FIELDS[pair[0]][0].split(' (')[0]} and {COLUMN_FIELDS[pair[1]][0]} over the "
                 f"water column, {' and '.join(GRID_DEPTHS)} m isobaths\n"
                 f"{start:%Y-%m-%d} to {(end - pd.Timedelta(seconds=1)):%Y-%m-%d}, latest run to "
                 f"{xend:%Y-%m-%d %H:%M}", fontweight="bold", y=0.97)
    for var, x0 in zip(pair, (0.14, 0.56)):
        cax = fig.add_axes([x0, 0.07, 0.32, 0.018])
        if var == "dye":
            dye_colorbar(fig, mappable[var], None, COLUMN_FIELDS[var][1], cax=cax, orientation="horizontal")
        else:
            fig.colorbar(mappable[var], cax=cax, orientation="horizontal", label=COLUMN_FIELDS[var][1])
    for j in range(len(panels)):
        time_axis(axs[:, j], start, xend, split, days)
        if j:
            axs[-1, j].set_xlabel("")
    axs[-1, 0].xaxis.labelpad = 20                     # below the "2026-Nov" offset text of the narrow panels
    if days <= 45:                                     # four panels across: a labelled tick every few days
        axs[-1, 0].xaxis.set_major_locator(mdates.DayLocator(interval=4))
    save(fig, path)


def plot_column(p, st, var, start, end, xend, split, days, path, norm=None):
    """Dye, temperature, salinity or density (var) over the water column (time x depth), one panel each, onshore to
    offshore: at the surfzone edge (first Hs maximum, moving each hour), and at the station's
    COLUMN_DEPTHS and 10, 12, 15 m points. Temperature and salinity use norm (from column_norm)."""
    title, label, cmap = COLUMN_FIELDS[var]
    if cmap is None:
        cmap, norm = dye_scale()
    tr = transects_cache()[st]
    fig, axs = plt.subplots(len(COLUMN_PANELS), 1, figsize=(12, 2.7 * len(COLUMN_PANELS)), sharex=True,
                            gridspec_kw={"hspace": 0.5})
    for n, (ax, key) in enumerate(zip(axs, COLUMN_PANELS)):
        if key == "sz":
            desc = "Surfzone edge (first H$_s$ maximum from shore; moves with the waves and tide)"
        else:
            s_pt = {**tr["column"], **tr["shelf"]}[next(d for d in COLUMN_DEPTHS + SHELF_DEPTHS if d == float(key))]["s"]
            desc = f"{key} m isobath ({s_pt:.0f} m offshore)"
        m = column_panel(ax, p, var, key, cmap, norm)
        ax.set_title(desc if n else
                     f"PFM LV4 {title} over the water column, {STATIONS[st]['name']}\n"
                     f"{start:%Y-%m-%d} to {(end - pd.Timedelta(seconds=1)):%Y-%m-%d}, latest run to "
                     f"{xend:%Y-%m-%d %H:%M}\n\n{desc}", pad=6)
    if var == "dye":
        dye_colorbar(fig, m, axs, label)
    else:
        fig.colorbar(m, ax=axs, label=label, shrink=0.6, pad=0.06)
    time_axis(axs, start, xend, split, days)
    save(fig, path)


def plot_hs(profs, start, end, xend, split, days, ymax, path, var="hs"):
    """Hs or surface dye (var) along the transect (time x offshore distance), one panel per station, with
    the shoreline, the surfzone edge and the shelf points; ymax: offshore end (m), None = past the
    deepest shelf point."""
    fig, axs = plt.subplots(len(STATIONS), 1, figsize=(12, 10), sharex=True, gridspec_kw={"hspace": 0.35})
    title, label, cmap, line = PROFILE_FIELDS[var]
    if var == "dye":
        cmap, norm = dye_scale()
    else:
        norm = Normalize(0, max(np.nanpercentile(p[var], 99.5) for p in profs.values()))   # one scale for all stations
    lw = 0.8 if days <= 60 else 0.25
    for n, (ax, (st, p)) in enumerate(zip(axs, profs.items())):
        top = ymax if ymax else p["s"][-1]
        k = p["s"] <= top
        f = p[var][:, k].T
        if var == "dye":                                  # dye below the scale shows as its lowest colour
            f = np.where(np.isfinite(f), np.maximum(f, norm.vmin), np.nan)
        m = ax.pcolormesh(p["time"], p["s"][k], f, cmap=cmap, norm=norm, shading="nearest", rasterized=True)
        ax.plot(p["time"], p["s_sl"], color=line, lw=lw, label="Shoreline (last wet point)")
        ax.plot(p["time"], p["s_edge"], color=C_SZ if var == "hs" else "#2a78d6", lw=lw, label="Surfzone edge (first H$_s$ maximum)")
        for dep, sh in transects_cache()[st]["shelf"].items():
            if sh["s"] <= top:
                ax.axhline(sh["s"], color=line, lw=0.8, ls="--")
                ax.text(1.003, sh["s"], f"{dep:g} m", transform=ax.get_yaxis_transform(), va="center", fontsize=10)
        ax.set_ylim(max(-LAND, np.nanmin(p["s_sl"]) - 20), top)
        ax.set_ylabel("Offshore (m)")
        ax.grid(False)
        ax.set_title(STATIONS[st]["name"] if n else
                     f"PFM LV4 {title} along the cross-shore transect, 0-{top if ymax else SHELF_DEPTHS[-1]:g}"
                     f"{' m offshore' if ymax else ' m isobath'}\n{start:%Y-%m-%d} to "
                     f"{(end - pd.Timedelta(seconds=1)):%Y-%m-%d}, latest run to {xend:%Y-%m-%d %H:%M}\n\n"
                     f"{STATIONS[st]['name']}", pad=32 if n == 0 else 6)
    dark = line == "white"                                # white lines need a dark legend box
    axs[0].legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2, fontsize=11, facecolor=INK2 if dark else SURFACE,
                  labelcolor="white" if dark else INK, frameon=dark)
    if var == "dye":
        dye_colorbar(fig, m, axs, label)
    else:
        fig.colorbar(m, ax=axs, label=label, shrink=0.6, pad=0.06)
    time_axis(axs, start, xend, split, days)
    for ax in axs:
        ax.axvline(split, color=line, lw=0.8, ls=":")
    save(fig, path)


_TRANSECTS = {}


def transects_cache():
    """transects(), computed once per run."""
    if not _TRANSECTS:
        _TRANSECTS.update(transects())
    return _TRANSECTS


def plot(start, end, path, workers=4, only=None):
    """Plot from start to end, naive Pacific times: the full figure to path, and the alongshore
    surfzone and ALONG_DEPTH shelf currents (3 panels) to <name>_alongshore.png, and Hs along the
    transects to <name>_hs_0-700m.png (0 to HS_NEAR m offshore) and <name>_hs_to15m.png (to the 15 m point),
    the surfzone edge Hs, width, depth and shoreline to <name>_surfzone.png, and Hs, Tp, mean direction at the
    WAVE_DEPTH point to <name>_waves.png. only: the VARIANTS to draw ("" = the full figure), None = all."""
    def want(v):
        return only is None or v in only

    data, profs = model_currents(to_utc(start), to_utc(end), workers)
    data = {st: to_pacific(df) for st, df in data.items()}
    for p in profs.values():
        p["time"] = p["time"].tz_localize("UTC").tz_convert(TZ).tz_localize(None)
    # model hours after now (or after the window, for an older window): dashed
    split = min(pd.Timestamp.now(tz=TZ).tz_localize(None).floor("h"), end)
    xend = max(end, max(df.index[-1] for df in data.values()))
    days = (end - start).days
    lw = 1.2 if days <= 60 else 0.5
    plt.rcParams.update(STYLE)
    if want(""):
        plot_full(data, start, end, xend, split, days, lw, path)
    if want("alongshore"):
        plot_alongshore(data, start, end, xend, split, days, lw, variant_path(path, "alongshore"))
    for v, ymax, var in (("hs_0-700m", HS_NEAR, "hs"), ("hs_to15m", None, "hs"),
                         ("dye_0-700m", HS_NEAR, "dye"), ("dye_to15m", None, "dye")):
        if want(v):
            plot_hs(profs, start, end, xend, split, days, ymax, variant_path(path, v), var)
    if want("surfzone"):
        plot_surfzone(data, profs, start, end, xend, split, days, lw, variant_path(path, "surfzone"))
    if want("waves"):
        plot_waves(data, start, end, xend, split, days, lw, variant_path(path, "waves"))
    for suffix, pair in GRID_FIGS.items():
        if want(suffix):
            plot_column_grid(profs, pair, start, end, xend, split, days, variant_path(path, suffix))
    for v in COLUMN_FIGS:
        sts = [st for st in profs if want(f"{v}column_{st}")]
        if not sts:
            continue
        norm = None if COLUMN_FIELDS[v][2] is None else column_norm(profs, v, COLUMN_FIELDS[v][2])
        for st in sts:
            plot_column(profs[st], st, v, start, end, xend, split, days, variant_path(path, f"{v}column_{st}"), norm)
    for st, df in data.items():
        past = df[:split]
        print(f"{st}: to {df['sz_along'].last_valid_index()}, median |surfzone along| {past['sz_along'].abs().median():.3f} m/s, "
              f"surfzone width {past['sz_width'].median():.0f} m, |shelf along| "
              + ", ".join(f"{dep:g} m {past[f'shelf{dep}_along'].abs().median():.3f}" for dep in SHELF_DEPTHS) + " m/s")


def plot_full(data, start, end, xend, split, days, lw, path):
    """Full figure: surfzone-edge Hs, then the alongshore and cross-shore currents of each station."""
    fig, axs = plt.subplots(1 + 2 * len(STATIONS), 1, figsize=(12, 17), sharex=True, gridspec_kw={"hspace": 0.55})
    ax = axs[0]
    for st, df in data.items():
        model_line(ax, df["sz_hs"], split, color=C_STATION[st], lw=lw, label=STATIONS[st]["name"])
    ax.set_ylabel("H$_s$ (m)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"PFM LV4 surfzone and shelf currents, {start:%Y-%m-%d} to {(end - pd.Timedelta(seconds=1)):%Y-%m-%d}"
                 f", latest run to {xend:%Y-%m-%d %H:%M}\nH$_s$ at the surfzone edge (first H$_s$ maximum from shore)",
                 pad=30)
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [AHEAD], loc="lower right", bbox_to_anchor=(1, 1), ncol=4, fontsize=11)
    for n, (st, df) in enumerate(data.items()):
        for ax, comp, desc in ((axs[1 + 2 * n], "along", "alongshore, + up-coast (northward)"),
                               (axs[2 + 2 * n], "cross", "cross-shore, + onshore")):
            model_line(ax, df[f"sz_{comp}"], split, color=C_SZ, lw=lw, label="Surfzone mean (shoreline to H$_s$ max)")
            for dep in SHELF_DEPTHS:
                model_line(ax, df[f"shelf{dep}_{comp}"], split, color=C_SHELF[dep], lw=lw, label=f"Shelf {dep:g} m")
            model_line(ax, df[f"top{TOP_DEPTH}_{comp}"], split, color=C_TOP, lw=lw, label=TOP_LABEL)
            ax.axhline(0, color=INK2, lw=0.6)
            ax.set_ylabel("m/s")
            ax.set_title(f"{STATIONS[st]['name']}: {desc}; depth-averaged, except {TOP_LABEL}", pad=22 if comp == "along" else 6)
            if comp == "along":
                ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2 + len(SHELF_DEPTHS), fontsize=11)
    time_axis(axs, start, xend, split, days)
    save(fig, path)


def plot_alongshore(data, start, end, xend, split, days, lw, path):
    """Alongshore only: surfzone and the ALONG_DEPTH shelf point (depth-averaged and top TOP m), one panel per
    station, same y axis."""
    fig, axs = plt.subplots(len(STATIONS), 1, figsize=(12, 9), sharex=True, sharey=True, gridspec_kw={"hspace": 0.4})
    for n, (ax, (st, df)) in enumerate(zip(axs, data.items())):
        model_line(ax, df["sz_along"], split, color=C_SZ, lw=lw, label="Surfzone mean (shoreline to H$_s$ max), depth-averaged")
        model_line(ax, df[f"shelf{ALONG_DEPTH}_along"], split, color=C_SHELF[ALONG_DEPTH], lw=lw,
                   label=f"Shelf {ALONG_DEPTH:g} m, depth-averaged")
        if ALONG_DEPTH == TOP_DEPTH:
            model_line(ax, df[f"top{TOP_DEPTH}_along"], split, color=C_TOP, lw=lw, label=TOP_LABEL)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_ylabel("m/s")
        ax.set_title(STATIONS[st]["name"] if n else
                     f"PFM LV4 alongshore current, + up-coast (northward)\n{start:%Y-%m-%d} to "
                     f"{(end - pd.Timedelta(seconds=1)):%Y-%m-%d}, latest run to {xend:%Y-%m-%d %H:%M}\n\n"
                     f"{STATIONS[st]['name']}", pad=32 if n == 0 else 6)
    axs[0].legend(handles=axs[0].get_legend_handles_labels()[0] + [AHEAD], loc="lower right", bbox_to_anchor=(1, 1),
                  ncol=4, fontsize=11)
    time_axis(axs, start, xend, split, days)
    save(fig, path)


def plot_map(path):
    """Map of the grid depth with each station's transect, surfzone start and shelf point."""
    trs = transects()
    with Dataset(GRID) as g:
        g.set_auto_mask(False)
        lon, lat, h, mask = (np.asarray(g[k][:], float) for k in ("lon_rho", "lat_rho", "h", "mask_rho"))
    plt.rcParams.update(STYLE)
    fig, axs = plt.subplots(1, len(trs), figsize=(5.5 * len(trs), 9))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)
    for ax, (st, tr) in zip(axs, trs.items()):
        p = STATIONS[st]
        ax.pcolormesh(lon, lat, np.where(mask == 1, h, np.nan), cmap="Blues", vmin=0, vmax=25, shading="nearest")
        ax.contour(lon, lat, mask, [0.5], colors=INK, linewidths=1)
        ax.contour(lon, lat, h, list(SHELF_DEPTHS), colors=INK2, linewidths=0.6, linestyles=":")
        ax.plot(tr["lon"], tr["lat"], color=C_SZ, lw=2, label="Transect")
        for dep, sh in tr["shelf"].items():
            k = np.argmin(np.abs(tr["s"] - sh["s"]))
            ax.plot(tr["lon"][k], tr["lat"][k], "o", color=C_SHELF[dep], mec=INK, ms=8, label=f"Shelf point, h = {dep:g} m")
        ax.plot(p["lon"], p["lat"], "*", color="#d03b3b", ms=12, label="Station")
        s_deep = max(sh["s"] for sh in tr["shelf"].values())
        k_sh = tr["s"] <= s_deep + 300                 # the transect up to just past the shelf point
        x0, x1 = tr["lon"][k_sh].min() - 0.006, tr["lon"][k_sh].max() + 0.006
        y0, y1 = tr["lat"][k_sh].min() - 0.006, tr["lat"][k_sh].max() + 0.006
        half = max((x1 - x0) * np.cos(np.deg2rad(p["lat"])), y1 - y0) / 2      # square map around them
        ax.set_xlim((x0 + x1) / 2 - half / np.cos(np.deg2rad(p["lat"])), (x0 + x1) / 2 + half / np.cos(np.deg2rad(p["lat"])))
        ax.set_ylim((y0 + y1) / 2 - half, (y0 + y1) / 2 + half)
        ax.set_aspect(1 / np.cos(np.deg2rad(p["lat"])))
        # directions counter-clockwise from east, as ShorelineOrientation.m (0 = east, 90 = north)
        th_off, th_al = (np.degrees(np.arctan2(v[1], v[0])) for v in (tr["off"], tr["along"]))
        ax.set_title(p["name"], fontsize=12)
        ax.text(0.5, -0.16, f"Directions from east (0° = E, 90° = N):\ntransect, offshore: {th_off:.1f}°\n"
                f"onshore (+ cross-shore): {th_off + 180 if th_off < 0 else th_off - 180:.1f}°\n"
                f"coastline, + alongshore: {th_al:.1f}°\nShelf points, offshore of the coastline:\n"
                + "\n".join(f"{d:g} m isobath: {sh['s']:.0f} m" for d, sh in tr["shelf"].items()), transform=ax.transAxes, ha="center", va="top", fontsize=11,
                bbox=dict(facecolor=SURFACE, edgecolor=INK2, alpha=0.9))
        ax.annotate("", xy=(0.12, 0.94), xytext=(0.04, 0.94), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=INK))
        ax.text(0.13, 0.94, "E", transform=ax.transAxes, va="center", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=10)
        ax.ticklabel_format(useOffset=False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_xlabel("Longitude (°E)", fontsize=11)
    axs[0].set_ylabel("Latitude (°N)", fontsize=11)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=12)
    save(fig, path)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date", help="Last day of the 3-week plot, YYYY-MM-DD Pacific (default: today)")
    p.add_argument("--workers", type=int, default=4, help="History files read in parallel (default 4)")
    p.add_argument("--map", action="store_true", help="Only plot the transects map, plots/transects_map.png")
    args = p.parse_args()
    if args.map:
        return plot_map(os.path.join(HERE, "plots", "transects_map.png"))
    today = pd.Timestamp.now(tz=TZ).tz_localize(None).normalize()
    last = pd.Timestamp(args.date) if args.date else today
    start, end = last - pd.Timedelta(days=DAYS - 1), last + pd.Timedelta(days=1)
    path = os.path.join(WEB_DIR, f"SurfzoneShelf_{last:%Y%m%d}_{LABEL}.png")
    plot(start, end, path, args.workers, only=WEB_VARIANTS)
    # the website keeps only the WEB_VARIANTS figures, and on the daily run only the newest ones
    keep = [variant_path(path, v) for v in WEB_VARIANTS]
    for old in glob.glob(os.path.join(WEB_DIR, f"SurfzoneShelf_{'[0-9]' * 8}_{LABEL}*.png")):
        if old not in keep and (not args.date or os.path.basename(old).startswith(f"SurfzoneShelf_{last:%Y%m%d}_")):
            os.remove(old)
            print("removed", os.path.basename(old))
    # alongshore dye Hovmoller (plot_dye_alongshore.py) and its map, same 3 weeks
    import plot_dye_alongshore as da
    lines = da.isobath_lines()
    along = os.path.join(WEB_DIR, f"DyeAlongshore_surf_{last:%Y%m%d}_{LABEL}.png")
    da.plot_alongshore(start, last, along, args.workers, lines=lines)
    for old in glob.glob(os.path.join(WEB_DIR, f"DyeAlongshore_surf_{'[0-9]' * 8}_{LABEL}.png")):
        if old != along and not args.date:
            os.remove(old)
            print("removed", os.path.basename(old))
    amap = os.path.join(WEB_DIR, "DyeAlongshore_map.png")
    if not os.path.exists(amap):        # static: the lines only change with the grid
        shore, _ = da.shoreline_dye(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"))
        da.plot_map(lines, shore, amap)
    tmap = os.path.join(WEB_DIR, "transects_map.png")
    if not os.path.exists(tmap):        # static as well
        plot_map(tmap)


if __name__ == "__main__":
    main()
