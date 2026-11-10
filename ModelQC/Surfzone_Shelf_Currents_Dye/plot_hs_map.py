#!/usr/bin/env python3
"""Maps of the PFM LV4 wave field (Hs contours and mean wave direction arrows), daily mean or one hour,
one panel per date, with the three surfzone/shelf stations and their transects, and Hs at each
station's 15 m point and surfzone edge (first Hs maximum from shore, read from the surfzone/shelf cache). To see why Hs differs
between the IB Pier, the guard shack and Avenida Lunar.

  python plot_hs_map.py --dates 2026-06-17 2026-04-14              # daily means (Pacific days)
  python plot_hs_map.py --dates "2026-10-11 09:00" "2026-10-24 00:00"   # single hours (Pacific)
  python plot_hs_map.py --dates 2026-06-17 2026-04-14 --ratio       # Hs / its value at the IB 15 m point

Each hour comes from the latest history file covering it, as in plot_surfzone_shelf.py.
Output: plots/HsMap_<date>_<date>....png (or --out).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset

from plot_surfzone_shelf import (C_STATION, EPOCH, GRID, HERE, INK2, STATIONS, STYLE, history_files, model_currents,
                                 to_utc, transects_cache)

LAT_RANGE = (32.53, 32.70)     # map window: TJRE mouth to Coronado
LON_RANGE = (-117.27, -117.10)
ARROW_EVERY = 22               # grid cells between direction arrows (~0.6 km)


def hours_fields(hours):
    """Mean Hwave and mean direction (unit vectors of Dwave, coming from) over the given UTC hours,
    each hour from the latest history file covering it."""
    files = history_files()
    todo = set(hours)
    hs_sum = dx = dy = None
    n = 0
    for start in sorted((t for t in files if t <= max(hours) and t >= min(hours) - pd.Timedelta(days=6)), reverse=True):
        if not todo:
            break
        with Dataset(files[start]) as nc:
            nc.set_auto_mask(False)
            t = (EPOCH + pd.to_timedelta(np.asarray(nc["ocean_time"][:], float), unit="s")).round("h")
            recs = [k for k, tk in enumerate(t) if tk in todo]
            for k in recs:
                hs = nc["Hwave"][k]
                d = np.deg2rad(nc["Dwave"][k])
                if hs_sum is None:
                    hs_sum, dx, dy = np.zeros_like(hs, float), np.zeros_like(hs, float), np.zeros_like(hs, float)
                hs_sum += hs
                dx += np.sin(d)
                dy += np.cos(d)
                n += 1
                todo.discard(t[k])
    if not n:
        raise SystemExit(f"no model hours for {min(hours)} to {max(hours)} UTC")
    if todo:
        print(f"  {len(todo)} of {len(hours)} hours not in any history file")
    return hs_sum / n, np.rad2deg(np.arctan2(dx, dy)) % 360, n


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dates", nargs="+", required=True,
                   help="Pacific days (YYYY-MM-DD, daily mean) or hours ('YYYY-MM-DD HH:MM')")
    p.add_argument("--ratio", action="store_true", help="plot Hs divided by Hs at the IB Pier 15 m point")
    p.add_argument("--out", help="output PNG (default plots/HsMap_<dates>.png)")
    args = p.parse_args()

    with Dataset(GRID) as g:
        g.set_auto_mask(False)
        lon, lat, h, mask = (np.asarray(g[k][:], float) for k in ("lon_rho", "lat_rho", "h", "mask_rho"))
    rows = np.flatnonzero((lat.min(1) <= LAT_RANGE[1]) & (lat.max(1) >= LAT_RANGE[0]))
    sl = slice(rows[0], rows[-1] + 1)
    lon, lat, h, mask = lon[sl], lat[sl], h[sl], mask[sl]
    trs = transects_cache()
    ib = trs["IBPierSouth"]["shelf"][15]["idx"]
    ib = (ib[0][0] - rows[0], ib[1][0])                 # nearest cell of the IB 15 m point, in the map window

    plt.rcParams.update(STYLE)
    fig, axs = plt.subplots(1, len(args.dates), figsize=(5.2 * len(args.dates), 7.2), squeeze=False)
    clat = np.cos(np.deg2rad(lat.mean()))
    for ax, d in zip(axs[0], args.dates):
        daily = len(d) <= 10
        t0 = pd.Timestamp(d)
        hours = list(pd.date_range(to_utc(t0), to_utc(t0 + pd.Timedelta(days=1)), freq="h", inclusive="left")) if daily \
            else [to_utc(t0)]
        hs, dp, n = hours_fields(hours)
        # surfzone edge (first Hs maximum from shore) of the same hours, from the surfzone/shelf cache
        data, profs = model_currents(hours[0], hours[-1], workers=4)
        edge = {st: (data[st]["sz_hs"].reindex(hours).mean(),
                     pd.Series(profs[st]["s_edge"], index=profs[st]["time"]).reindex(hours).mean()) for st in STATIONS}
        hs, dp = hs[sl], dp[sl]
        hs = np.where(mask == 1, hs, np.nan)
        if args.ratio:
            f, levels, label = hs / hs[ib], np.arange(0, 1.41, 0.05), "H$_s$ / H$_s$ at IB Pier 15 m"
        else:
            top = np.ceil(np.nanpercentile(hs, 99.5) * 10) / 10
            f, levels, label = hs, np.arange(0, top + 0.05, 0.1), "H$_s$ (m)"
        cf = ax.contourf(lon, lat, f, levels=levels, cmap="viridis", extend="max")
        cs = ax.contour(lon, lat, f, levels=levels[::2], colors="white", linewidths=0.5, alpha=0.7)
        ax.clabel(cs, fontsize=7, fmt="%.1f")
        ax.contourf(lon, lat, np.where(mask == 1, np.nan, 1), levels=[0.5, 1.5], colors=["#d9d6cf"])
        ax.contour(lon, lat, np.where(mask == 1, h, np.nan), levels=[10, 15, 20, 30], colors=INK2,
                   linewidths=0.5, linestyles=":")
        # mean wave direction: arrows the way the waves travel (Dwave is where they come from)
        s = (slice(None, None, ARROW_EVERY), slice(None, None, ARROW_EVERY))
        go = np.deg2rad(dp[s] + 180)
        ok = (mask[s] == 1) & (hs[s] > 0.2)              # no arrows where there are hardly any waves (the bay)
        ax.quiver(lon[s][ok], lat[s][ok], np.sin(go[ok]) / clat, np.cos(go[ok]), color="white", scale=28,
                  width=0.004, headwidth=4, alpha=0.9)
        for st, tr in trs.items():
            ax.plot(tr["lon"][(tr["s"] >= 0) & (tr["s"] <= tr["shelf"][15]["s"])],
                    tr["lat"][(tr["s"] >= 0) & (tr["s"] <= tr["shelf"][15]["s"])], color=C_STATION[st], lw=2)
            ax.plot(STATIONS[st]["lon"], STATIONS[st]["lat"], "o", ms=6, mfc=C_STATION[st], mec="white")
            j, i = tr["shelf"][15]["idx"][0][0] - rows[0], tr["shelf"][15]["idx"][1][0]
            sz_hs, s_edge = edge[st]
            ax.plot(np.interp(s_edge, tr["s"], tr["lon"]), np.interp(s_edge, tr["s"], tr["lat"]), "D", ms=6,
                    mfc="#eb6834", mec="white", mew=1, zorder=5)
            ax.text(lon[j, i] - 0.004, lat[j, i], f"15 m: {hs[j, i]:.2f} m\nedge: {sz_hs:.2f} m", ha="right",
                    va="center", fontsize=7.5, color="white", linespacing=1.1,
                    bbox=dict(boxstyle="round,pad=0.25", fc=C_STATION[st], ec="none"))
        ax.set_xlim(LON_RANGE)
        ax.set_ylim(LAT_RANGE)
        ax.set_aspect(1 / clat)
        ax.grid(False)
        ax.tick_params(labelsize=8)
        ax.xaxis.get_major_formatter().set_useOffset(False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        when = f"{t0:%Y-%m-%d}, daily mean ({n} h)" if daily else f"{t0:%Y-%m-%d %H:%M} PDT/PST"
        ib_dp = dp[ib]
        ax.set_title(f"{when}\nIB 15 m: mean direction D$_m$ {ib_dp:.0f}° (from)", fontsize=10)
        fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.07, shrink=0.9, label=label)
    fig.suptitle("PFM LV4 significant wave height, mean wave direction (arrows, towards), 10/15/20/30 m isobaths (dotted)\n"
                 "Labels: H$_s$ at each station's 15 m point and at the surfzone edge (first H$_s$ maximum from shore, "
                 "orange diamond: its mean position). Stations: " +
                 ", ".join(f"{STATIONS[st]['name']}" for st in STATIONS), fontsize=10, y=1.0)
    out = args.out or os.path.join(HERE, "plots", "HsMap_" + "_".join(d.replace(" ", "T").replace(":", "")
                                                                       for d in args.dates)
                                   + ("_ratio" if args.ratio else "") + ".png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
