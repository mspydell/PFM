#!/usr/bin/env python3
"""Alongshore Hovmoller of total dye at the shoreline and the 10 and 15 m isobaths, PFM LV4 (the QC-website copy,
plots_web/DyeAlongshore_surf_<YYYYMMDD>_3weeks.png, is made by plot_surfzone_shelf.py with plot_alongshore()).

Time on x, distance along the shoreline from the south end of the LV4 grid on y (km, south to north),
dye_01 (PTB) + dye_02 (TJRE) in colour, one panel per line. Shoreline: shoreline_dye_tot of the
QC-website files (Archive/web/web_data_<run>.nc, from 2025-04-12; 1060 points, 32.42-32.69 N, 32.6 km),
each hour from the latest run. Isobaths: put on the shoreline distance (match_shore) by pinning the
isobath to the shoreline at the marked places (each at its nearest point on both lines) and the two ends, and
spacing it evenly in between, so all panels share the shoreline distance and no part of the isobath is
skipped (the 15 m line bulges about 3 km offshore round the Tijuana River delta). The isobath is the longest h contour of the
grid (10 m: to about 32.69 N, where it breaks up near the bay entrance; 15 m: to about 32.73 N), resampled
every DS m, each point at its nearest wet rho cell. The stations, the Tijuana River mouth, the border and Punta Bandera are
marked. --mean: the depth-averaged isobath dye instead of the surface layer. --map: the map of the lines,
plots/DyeAlongshore_map.png.

Cache: one npz per history file in CACHE_DIR/alongshore (surface and depth-averaged dye on the points),
each model hour from the latest file covering it, as plot_surfzone_shelf.py.

  python plot_dye_alongshore.py                                  # the past 3 weeks, surface dye
  python plot_dye_alongshore.py --mean
  python plot_dye_alongshore.py --map
  python plot_dye_alongshore.py --start 2025-01-01 --end today --workers 10
"""
import argparse
import glob
import hashlib
import os
import re
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.spatial import cKDTree

from plot_surfzone_shelf import (CACHE_DIR, DAYS, EPOCH, GRID, HERE, INK, M_LAT, M_LON, STATIONS, STYLE, TZ,
                                 dye_colorbar, dye_scale, fresh, history_files, merge_npz, save, time_axis, to_utc)

ISOBATHS = (10, 15)    # m, one panel each
DS = 100.0             # m, spacing of the points along the isobath
SMOOTH = 5             # points, running mean of the contour before measuring distance (grid-cell wiggles)
ALONG_DIR = os.path.join(CACHE_DIR, "alongshore")
WEB_DIR = "/dataSIO/PFM_Simulations/Archive/web"   # QC-website files with the shoreline dye
# marked on the y axis besides the stations: name, lon, lat
MARKS = (("Tijuana River mouth", -117.1270, 32.5519), ("US-Mexico border", -117.1240, 32.5343),
         ("Punta Bandera (PTB)", -117.1078, 32.4456))   # PTB: the dye_01 source cell of river_LV4 (river 6)


def isobath_lines():
    """{depth: dict(lon, lat, dist (km from the south end), j, i (nearest wet cell))} for each ISOBATHS."""
    with Dataset(GRID) as g:
        g.set_auto_mask(False)
        h, mask, lon, lat = (g[k][:] for k in ("h", "mask_rho", "lon_rho", "lat_rho"))
    clat = np.cos(np.deg2rad(lat.mean()))
    wet = np.flatnonzero(mask.ravel() > 0)
    tree = cKDTree(np.c_[lon.ravel()[wet] * M_LON * clat, lat.ravel()[wet] * M_LAT])
    cs = plt.contour(lon, lat, np.where(mask > 0, h, np.nan), list(ISOBATHS))
    plt.close("all")
    lines = {}
    for dep, segs in zip(ISOBATHS, cs.allsegs):
        xy = max(segs, key=len)                                   # the long contour along the coast
        if xy[0, 1] > xy[-1, 1]:
            xy = xy[::-1]                                         # south to north
        x, y = xy[:, 0] * M_LON * clat, xy[:, 1] * M_LAT
        s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        sn = np.arange(0, s[-1], DS)
        x, y = np.interp(sn, s, x), np.interp(sn, s, y)
        k = np.ones(SMOOTH) / SMOOTH
        x = np.r_[x[:SMOOTH // 2], np.convolve(x, k, "valid"), x[len(x) - SMOOTH // 2:]]
        y = np.r_[y[:SMOOTH // 2], np.convolve(y, k, "valid"), y[len(y) - SMOOTH // 2:]]
        cell = wet[tree.query(np.c_[x, y])[1]]
        j, i = np.unravel_index(cell, h.shape)
        lines[dep] = dict(lon=x / (M_LON * clat), lat=y / M_LAT, j=j, i=i,
                          dist=np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))] / 1000)
    return lines


def line_hash(lines):
    key = repr([ISOBATHS, DS, SMOOTH] + [(lines[d]["j"].tolist(), lines[d]["i"].tolist()) for d in ISOBATHS])
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def cache_path(path, lines):
    stamp = os.path.basename(path)[len("LV4_ocean_his_"):-3]
    return os.path.join(ALONG_DIR, f"Alongshore_LV4_{stamp}_{line_hash(lines)}.npz")


def process(path, lines):
    """Surface and depth-averaged total dye on the isobath points of one history file, cached."""
    c = cache_path(path, lines)
    if fresh(c, path):
        return None
    j = np.concatenate([lines[d]["j"] for d in ISOBATHS])
    i = np.concatenate([lines[d]["i"] for d in ISOBATHS])
    j0, j1, i0, i1 = j.min(), j.max() + 1, i.min(), i.max() + 1
    with Dataset(path) as nc:
        nc.set_auto_mask(False)
        t = (EPOCH + pd.to_timedelta(np.asarray(nc["ocean_time"][:], float), unit="s")).round("h")
        dye = nc["dye_01"][:, :, j0:j1, i0:i1][..., j - j0, i - i0] + nc["dye_02"][:, :, j0:j1, i0:i1][..., j - j0, i - i0]
        s_w, cs_w, hc = (np.asarray(nc[k][:], float) for k in ("s_w", "Cs_w", "hc"))
    with Dataset(GRID) as g:
        h = np.asarray(g["h"][:])[j, i]
    # Vtransform 2: layer thickness / water depth = diff((hc s_w + h Cs_w) / (hc + h)), the same at every time
    frac = np.diff((hc * s_w[:, None] + h * cs_w[:, None]) / (hc + h), axis=0)        # layers x points
    out, n = dict(time=t.values), 0
    for d in ISOBATHS:
        m = len(lines[d]["j"])
        out[f"surf{d}"] = dye[:, -1, n:n + m].astype(np.float32)
        out[f"mean{d}"] = (dye[:, :, n:n + m] * frac[:, n:n + m]).sum(axis=1).astype(np.float32)
        n += m
    os.makedirs(ALONG_DIR, exist_ok=True)
    with open(c + ".part", "wb") as f:
        np.savez_compressed(f, **out)
    os.replace(c + ".part", c)
    return os.path.basename(path)


def match_shore(line, shore):
    """Index of the line point at each shoreline point: the two lines pinned together at the ref_points and
    the ends of the shoreline (each at its nearest point on both lines), linear in distance in between."""
    pins = [(shore["lon"][0], shore["lat"][0]), (shore["lon"][-1], shore["lat"][-1])] + \
        [(lon, lat) for _, lon, lat, _ in ref_points()]
    s_pin, l_pin = (np.array([along_dist(ln, lon, lat) for lon, lat in pins]) for ln in (shore, line))
    order = np.argsort(s_pin)
    s_pin, l_pin = s_pin[order], l_pin[order]
    if np.any(np.diff(l_pin) <= 0):
        raise SystemExit(f"pins out of order along the isobath: {l_pin.round(2)}")
    d = np.interp(shore["dist"], s_pin, l_pin)
    return np.clip(np.searchsorted(line["dist"], d), 0, len(line["dist"]) - 1)


def along_dist(line, lon, lat):
    """Distance (km) along the line of the point nearest to lon, lat."""
    clat = np.cos(np.deg2rad(lat))
    return line["dist"][np.argmin(np.hypot((line["lon"] - lon) * M_LON * clat, (line["lat"] - lat) * M_LAT))]


def shoreline_dye(s_utc, e_utc):
    """Shoreline dye of the QC-website runs from s_utc to e_utc: (line dict(lon, lat, dist), list of
    dict(time, shore (time x points)) per run, oldest first)."""
    runs = []
    for f in sorted(glob.glob(os.path.join(WEB_DIR, "web_data_*.nc"))):
        m = re.search(r"web_data_(\d{10})\.nc$", f)
        if m and s_utc - pd.Timedelta(days=6) <= pd.Timestamp(m.group(1)[:8] + "T" + m.group(1)[8:]) <= e_utc:
            with Dataset(f) as nc:
                t = (pd.Timestamp("1999-01-01") + pd.to_timedelta(np.asarray(nc["time"][:], float), unit="D")).round("h")
                runs.append(dict(time=t.values, shore=np.ma.filled(nc["shoreline_dye_tot"][:].astype(float), np.nan)))
                lon, lat = (np.asarray(nc[k][:], float) for k in ("shoreline_lon", "shoreline_lat"))
    if not runs:
        raise SystemExit(f"no QC-website files in {WEB_DIR} for these dates (they start on 2025-04-12)")
    clat = np.cos(np.deg2rad(lat.mean()))
    dist = np.r_[0, np.cumsum(np.hypot(np.diff(lon) * M_LON * clat, np.diff(lat) * M_LAT))] / 1000
    return dict(lon=lon, lat=lat, dist=dist), runs


def ref_points():
    """The stations and MARKS: (name, lon, lat, dashed)."""
    return [(STATIONS[st]["name"], STATIONS[st]["lon"], STATIONS[st]["lat"], False) for st in STATIONS] \
        + [(name, lon, lat, True) for name, lon, lat in MARKS]


def plot_map(lines, shore, path):
    """LV4 grid depth with the shoreline and isobath lines: every 5 km along the shoreline, the isobath points
    that take that distance; the isobath parts matched to the shoreline solid, the rest dotted."""
    with Dataset(GRID) as g:
        g.set_auto_mask(False)
        h, mask, lon, lat = (g[k][:] for k in ("h", "mask_rho", "lon_rho", "lat_rho"))
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8, 11))
    pc = ax.pcolormesh(lon, lat, np.where(mask > 0, h, np.nan), cmap="Blues", vmin=0, vmax=60, shading="auto",
                       rasterized=True)
    ax.pcolormesh(lon, lat, np.where(mask > 0, np.nan, 1), cmap="Greys", vmin=0, vmax=3, shading="auto",
                  rasterized=True)
    ks = [np.argmin(abs(shore["dist"] - km)) for km in np.arange(0, shore["dist"][-1], 5)]
    for name, ln, color in (("Shoreline (QC-website points)", shore, "#d03b3b"),
                            ("10 m isobath", lines[10], "#e08b1a"), ("15 m isobath", lines[15], "#7b3fa0")):
        idx = np.arange(len(ln["lon"])) if ln is shore else match_shore(ln, shore)
        used = np.zeros(len(ln["lon"]), bool)
        used[idx.min():idx.max() + 1] = True
        ax.plot(np.where(used, ln["lon"], np.nan), ln["lat"], color=color, lw=2, label=name)
        ax.plot(np.where(used, np.nan, ln["lon"]), ln["lat"], color=color, lw=1, ls=":")
        ax.plot(ln["lon"][idx[ks]], ln["lat"][idx[ks]], "o", color=color, mec=INK, ms=4, zorder=3)
    for n, k in enumerate(ks):
        k15 = match_shore(lines[15], shore)[k]
        ax.plot([shore["lon"][k], lines[15]["lon"][k15]], [shore["lat"][k], lines[15]["lat"][k15]], color=INK, lw=0.6)
        ax.annotate(f"{5 * n:g} km", (lines[15]["lon"][k15], lines[15]["lat"][k15]), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=10)
    for name, ln_, la_, dashed in ref_points():
        ax.plot(ln_, la_, "*" if not dashed else "D", color=INK, ms=10 if not dashed else 5)
        ax.annotate(name, (ln_, la_), xytext=(6, 0), textcoords="offset points", va="center", fontsize=10)
    ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_xlabel("Longitude")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_ylabel("Latitude")
    ax.grid(False)
    ax.legend(loc="lower left", fontsize=11)
    cb = fig.colorbar(pc, ax=ax, label="Water depth h (m)", shrink=0.5, extend="max", pad=0.2)
    ax.title_span = [cb.ax]                  # save() wraps the title to the map + colour bar width
    ax.set_title("PFM LV4 grid: shoreline and 10, 15 m isobath lines of the alongshore dye Hovmoller; "
                 "every 5 km along the shoreline, the isobath points at that distance (pinned at the marks)",
                 loc="left")
    save(fig, path)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", help="First day, YYYY-MM-DD Pacific (default: 3 weeks before --end)")
    p.add_argument("--end", default="today", help="Last day, included, YYYY-MM-DD or 'today' (Pacific)")
    p.add_argument("--mean", action="store_true", help="Depth-averaged isobath dye (default: surface layer)")
    p.add_argument("--map", action="store_true", help="Only the map of the lines, plots/DyeAlongshore_map.png")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    now = pd.Timestamp.now(tz=TZ).tz_localize(None)
    last = now.normalize() if args.end == "today" else pd.Timestamp(args.end)
    start = pd.Timestamp(args.start) if args.start else last - pd.Timedelta(days=DAYS - 1)
    if args.map:
        lines = isobath_lines()
        shore, _ = shoreline_dye(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"))
        return plot_map(lines, shore, os.path.join(HERE, "plots", "DyeAlongshore_map.png"))

    plot_alongshore(start, last, os.path.join(HERE, "plots", f"DyeAlongshore_{'mean' if args.mean else 'surf'}_"
                                                      f"{start:%Y%m%d}_{last:%Y%m%d}.png"), args.workers, args.mean)


def plot_alongshore(start, last, path, workers=4, mean=False, lines=None):
    """The alongshore dye Hovmoller from Pacific day start to last (included), into path."""
    now = pd.Timestamp.now(tz=TZ).tz_localize(None)
    end = last + pd.Timedelta(days=1)
    s_utc, e_utc = to_utc(start), to_utc(end)
    lines = lines or isobath_lines()
    files = history_files()
    paths = [files[t] for t in sorted(files) if s_utc - pd.Timedelta(days=6) <= t <= e_utc]
    with ProcessPoolExecutor(max(1, min(workers, len(paths)))) as ex:
        for name in ex.map(process, paths, [lines] * len(paths)):
            if name:
                print("computed", name)
    npz = [np.load(cache_path(f, lines)) for f in paths]
    shore, runs = shoreline_dye(s_utc, e_utc)
    tmax = max(pd.Timestamp(z["time"][-1]) for z in npz + runs)
    hours = pd.date_range(s_utc, max(e_utc, tmax), freq="h")
    kind = "mean" if mean else "surf"
    data = merge_npz(npz, [f"{kind}{d}" for d in ISOBATHS], hours)
    data["shore"] = merge_npz(runs, ["shore"], hours)["shore"]
    t = hours.tz_localize("UTC").tz_convert(TZ).tz_localize(None)
    split = min(now.floor("h"), end)
    xend, days = t[-1], (end - start).days

    for d in ISOBATHS:                                 # on the shoreline distance, pinned at the marked places
        data[f"{kind}{d}"] = data[f"{kind}{d}"][:, match_shore(lines[d], shore)]
    panels = [("shore", "Shoreline (surfzone; QC-website shoreline points)")] + [
        (f"{kind}{d}", f"{d} m isobath, {'depth-averaged' if mean else 'surface'}; stretched to the shoreline "
                       f"distance, pinned at the marked places") for d in ISOBATHS]
    plt.rcParams.update(STYLE)
    cmap, norm = dye_scale()
    fig, axs = plt.subplots(len(panels), 1, figsize=(12, 4.2 * len(panels)), sharex=True,
                            gridspec_kw={"hspace": 0.3})
    fig.subplots_adjust(bottom=0.12, right=0.82)
    for ax, (key, desc) in zip(axs, panels):
        f = data[key]
        m = ax.pcolormesh(t, shore["dist"], np.where(np.isfinite(f), np.maximum(f, norm.vmin), np.nan).T,
                          cmap=cmap, norm=norm, shading="nearest", rasterized=True)
        ax.set_ylim(0, shore["dist"][-1])
        ax.set_ylabel("Distance along the shoreline\nfrom the south end (km)")
        ax.grid(False)
        ax.set_title(desc, loc="left")
        for name, lon, lat, dashed in ref_points():
            y = along_dist(shore, lon, lat)
            ax.axhline(y, color=INK, lw=0.8, ls="--" if dashed else "-")
            ax.text(1.005, y, name, transform=ax.get_yaxis_transform(), va="center", fontsize=10)
    axs[0].set_title(f"PFM LV4 dye (dye_01 PTB + dye_02 TJRE) along the coast, south to north\n"
                     f"{start:%Y-%m-%d} to {last:%Y-%m-%d}, latest run to {xend:%Y-%m-%d %H:%M}\n\n"
                     f"{panels[0][1]}", loc="left")
    dye_colorbar(fig, m, axs, "Dye (unitless fraction), log scale", cax=fig.add_axes([0.2, 0.04, 0.5, 0.012]),
                 orientation="horizontal")
    time_axis(axs, start, xend, split, days)
    save(fig, path)


if __name__ == "__main__":
    main()
