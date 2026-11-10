#!/usr/bin/env python3
"""Dye vs density (sigma0), salinity and temperature scatter at the 3 stations, PFM LV4 (exploratory, not for
the website).

One figure per field (SCATTER_FIELDS): a row per isobath (SCATTER_DEPTHS: 5, 10, 15 m), a column per station;
every layer, every past hour, dye_01 + dye_02 (log) against the field, coloured by depth z. Pearson
correlation of log10(dye) and the field per panel, printed and written on the panels, and the
least-squares line of log10(dye) on the field (dashed). Reads the
water-column cache of plot_surfzone_shelf.py.

  python plot_dye_scatter.py                                # the past 3 weeks
  python plot_dye_scatter.py --start 2025-01-01 --end today
"""
import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from plot_surfzone_shelf import (DAYS, INK, DYE_RANGE, HERE, STATIONS, STYLE, TZ, model_currents, save,
                                 to_utc)

MAX_POINTS = 150_000   # points drawn per panel (random subset; the correlations use all)
SCATTER_DEPTHS = ("5", "10", "15")   # isobaths (m), one row each (water-column cache points)
# depth colours: plasma without its pale yellow end, orange at the surface, dark blue at the bottom; each
# isobath row has its own colour bar, 0 to the isobath depth
DEPTH_CMAP = ListedColormap(plt.cm.plasma(np.linspace(0, 0.85, 256))[::-1])
# field: file name, x axis label, title name
SCATTER_FIELDS = {"rho": ("DyeDensity", r"$\sigma_0$ (kg m$^{-3}$)", "potential density"),
                  "salt": ("DyeSalinity", "Salinity", "salinity"),
                  "temp": ("DyeTemperature", "Temperature (°C)", "temperature")}


def field_dye(col, var, keys, past):
    """var, log10(dye) and z of the panels keys, past hours only, flattened; finite points only."""
    s, d, z = (np.concatenate([col[f"{v}{k}"][past].ravel() for k in keys]) for v in (var, "dye", "z"))
    ok = np.isfinite(s) & np.isfinite(d) & np.isfinite(z) & (d > 0)
    return s[ok], np.log10(d[ok]), z[ok]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", help="First day, YYYY-MM-DD Pacific (default: 3 weeks before --end)")
    p.add_argument("--end", default="today", help="Last day, included, YYYY-MM-DD or 'today' (Pacific)")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    now = pd.Timestamp.now(tz=TZ).tz_localize(None)
    last = now.normalize() if args.end == "today" else pd.Timestamp(args.end)
    start = pd.Timestamp(args.start) if args.start else last - pd.Timedelta(days=DAYS - 1)
    end = last + pd.Timedelta(days=1)
    _, profs = model_currents(to_utc(start), to_utc(end), args.workers)

    plt.rcParams.update(STYLE)
    for var, (name, xlabel, title) in SCATTER_FIELDS.items():
        plot_field(profs, var, name, xlabel, title, start, end, last, now)


def plot_field(profs, var, name, xlabel, title, start, end, last, now):
    """One scatter figure: dye against var, one row per SCATTER_DEPTHS isobath, one column per station."""
    fig, axs = plt.subplots(len(SCATTER_DEPTHS), len(STATIONS), figsize=(16, 4.3 * len(SCATTER_DEPTHS)),
                            sharex=True, sharey=True, gridspec_kw={"wspace": 0.06, "hspace": 0.25})
    rng = np.random.default_rng(0)
    xs = []
    for col, (st, prof) in enumerate(profs.items()):
        t = prof["time"].tz_localize("UTC").tz_convert(TZ).tz_localize(None)
        past = np.asarray((t >= start) & (t < min(end, now)))      # model hours already past, in the window
        for row, key in enumerate(SCATTER_DEPTHS):
            ax = axs[row, col]
            s, d, z = field_dye(prof["column"], var, [key], past)
            r = pearsonr(s, d)[0]               # Pearson, of log10(dye), as the Matlab dye scripts
            print(f"{st} {key} m: n = {s.size}, Pearson r({var}, log10 dye) {r:+.2f}")
            xs.append(np.nanpercentile(s, [0.1, 99.9]))
            i = rng.permutation(s.size)[:MAX_POINTS]
            i = i[np.argsort(z[i])]                                 # shallow points drawn last, on top
            m = ax.scatter(s[i], 10 ** d[i], c=-z[i], s=1.5, cmap=DEPTH_CMAP, vmin=0, vmax=float(key),
                           linewidths=0, alpha=0.5, rasterized=True)
            ax.set_yscale("log")
            ax.set_ylim(*DYE_RANGE)
            slope, icpt = np.polyfit(s, d, 1)                 # least-squares fit of log10(dye) on the field
            xf = np.array(xs[-1])
            ax.plot(xf, 10 ** (slope * xf + icpt), color=INK, lw=1.5, ls="--")
            ax.set_title(f"{STATIONS[st]['name']}, {key} m isobath", loc="left", fontsize=10)
            ax.text(0.03, 0.04, f"Pearson r = {r:+.2f}\nfit: log10(dye) = {slope:+.2f} x {icpt:+.1f}", transform=ax.transAxes, fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
            if col == 0:
                ax.set_ylabel("Dye (unitless), log scale")
            if row == len(SCATTER_DEPTHS) - 1:
                ax.set_xlabel(xlabel)
            if col == len(STATIONS) - 1:
                fig.colorbar(m, ax=axs[row, :], label="Depth, -z (m)", pad=0.015, aspect=12).ax.invert_yaxis()
    pad = 0.03 * (max(x[1] for x in xs) - min(x[0] for x in xs))
    axs[0, 0].set_xlim(min(x[0] for x in xs) - pad, max(x[1] for x in xs) + pad)   # outliers past 0.1-99.9 % off
    fig.suptitle(f"PFM LV4 dye vs {title} over the water column at the "
                 f"{', '.join(SCATTER_DEPTHS)} m isobaths\n{start:%Y-%m-%d} to {last:%Y-%m-%d}, past hours",
                 fontweight="bold", y=0.95)
    save(fig, os.path.join(HERE, "plots", f"{name}_{start:%Y%m%d}_{last:%Y%m%d}.png"))

if __name__ == "__main__":
    main()
