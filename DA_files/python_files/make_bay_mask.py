#!/usr/bin/env python3
"""Build a smoothly-tapered 'bay water' mask for damping DA increments.

The increment that objective mapping delivers is defined everywhere there is
water, including inside San Diego Bay.  For zeta that is right -- the bay fills
and drains in phase with the coastal ocean, so zeroing it would manufacture a
head difference across the mouth.  For u, v, temp and salt it is not: bay
currents are tidal and topographically steered, the HFR observations carry no
information about them, and the balance operator that produced the T/S
increment does not hold in there.

So: a mask that is 0 inside the bay, 1 on the shelf, with a smooth transition,
to multiply into the u/v/T/S increments as a final step.

    mask = 1  open ocean
    mask = 0  inside the bay
    transition over SIGMA km, measured THROUGH WATER

The last point is the one that matters.  A Euclidean taper leaks across the
Silver Strand -- a bay cell is a few hundred metres of LAND from the open
ocean, so a 2 km Euclidean blur would lift the mask in the middle of the bay
while dropping it on the ocean side of the spit.  Here the transition is grown
by an explicit diffusion restricted to water cells with no-flux at land, so it
travels the ~20 km around through the mouth, not the 300 m across the sand.

The sharp 0/1 split is set by a line across the bay mouth plus a flood fill,
so the "inside" region is whatever is connected to the bay seed once that line
is a barrier -- no polygon needed.

Usage (all on one line -- the coordinates start with '-', which is handled):
  python3 make_bay_mask.py --level LV3 --p1 -117.2380,32.6690 --p2 -117.1800,32.6780 --seed -117.1700,32.7000 --sigma-km 2.0
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import label

GRIDS = {
    "LV2": "/scratch/PFM_Simulations/Grids/GRID_SDTJRE_LV2_rx020.nc",
    "LV3": "/scratch/PFM_Simulations/Grids/GRID_SDTJRE_LV3_rx020_maskCoast.nc",
    "LV4": "/scratch/PFM_Simulations/Grids/GRID_SDTJRE_LV4_mss_oct2024.nc",
}
# PFM copy: figures go to the DA scratch log dir, not back into roms_fourdvar.
OUTDIR = Path("/scratch/PFM_Simulations/DA_Sim/Log/figures/bay_mask")
KM = 111.0
ALPHA = 0.2          # explicit diffusion coefficient, stable below 0.25


def seg_distance_km(lon, lat, p1, p2, lat0):
    """Distance from each cell to the line SEGMENT p1-p2, in km."""
    c = np.cos(np.deg2rad(lat0))
    x, y = (lon - p1[0]) * c * KM, (lat - p1[1]) * KM
    bx, by = (p2[0] - p1[0]) * c * KM, (p2[1] - p1[1]) * KM
    t = np.clip((x * bx + y * by) / (bx * bx + by * by), 0.0, 1.0)
    return np.hypot(x - t * bx, y - t * by), (x * by - y * bx)


def diffuse_in_water(f, water, nsteps, alpha=ALPHA):
    """Explicit diffusion with no-flux at land: fluxes only between wet pairs."""
    f = f.astype("f8", copy=True)
    wx = water[:, :-1] & water[:, 1:]
    wy = water[:-1, :] & water[1:, :]
    for _ in range(nsteps):
        lap = np.zeros_like(f)
        fx = np.where(wx, f[:, 1:] - f[:, :-1], 0.0)
        lap[:, :-1] += fx
        lap[:, 1:] -= fx
        fy = np.where(wy, f[1:, :] - f[:-1, :], 0.0)
        lap[:-1, :] += fy
        lap[1:, :] -= fy
        f += alpha * lap
    return f


def _join_negative_values(argv, opts=("--p1", "--p2", "--seed")):
    """Rewrite '--p1 VALUE' as '--p1=VALUE'.

    argparse refuses '-117.2380,32.6690' as a value because it starts with '-'
    and its negative-number matcher only accepts a bare number, not 'lon,lat'.
    Joining with '=' means the value is never inspected as a flag.
    """
    out, i = [], 0
    while i < len(argv):
        if argv[i] in opts and i + 1 < len(argv):
            out.append(f"{argv[i]}={argv[i + 1]}")
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="LV3", choices=tuple(GRIDS))
    ap.add_argument("--p1", default="-117.2380,32.6690",
                    help="lon,lat -- SE corner of Point Loma")
    ap.add_argument("--p2", default="-117.1800,32.6780",
                    help="lon,lat -- SW corner of Coronado")
    ap.add_argument("--seed", default="-117.1700,32.7000",
                    help="lon,lat of a water point INSIDE the bay")
    ap.add_argument("--sigma-km", type=float, default=2.0)
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    args = ap.parse_args(_join_negative_values(sys.argv[1:]))

    p1 = tuple(float(v) for v in args.p1.split(","))
    p2 = tuple(float(v) for v in args.p2.split(","))
    sd = tuple(float(v) for v in args.seed.split(","))

    with nc.Dataset(GRIDS[args.level]) as d:
        lon = np.asarray(d["lon_rho"][:]); lat = np.asarray(d["lat_rho"][:])
        water = np.asarray(d["mask_rho"][:]).astype(bool)
        dx = float((1.0 / np.asarray(d["pm"][:]))[water].mean())
    lat0 = float(lat.mean())
    print(f"  {args.level}: dx = {dx:.1f} m, {water.sum():,} wet cells")
    print(f"  line  {p1} -> {p2}")

    # --- 1. sharp 0/1 split: line as a barrier, then flood fill from the bay
    dseg, side = seg_distance_km(lon, lat, p1, p2, lat0)
    barrier = dseg < (dx / 1000.0)              # ~one cell thick
    open_w = water & ~barrier
    lab, n = label(open_w, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    j, i = np.unravel_index(
        np.argmin((lon - sd[0]) ** 2 + (lat - sd[1]) ** 2), lon.shape)
    if not open_w[j, i]:
        raise SystemExit(f"seed {sd} is not open water on {args.level}")
    bay_lab = lab[j, i]
    bay = (lab == bay_lab)
    seed_side = side[j, i]
    bay |= barrier & water & (np.sign(side) == np.sign(seed_side))
    frac = 100 * bay.sum() / water.sum()
    print(f"  flood fill: {n} water regions, bay region = {bay.sum():,} cells "
          f"({frac:.1f}% of {args.level} water)")
    if frac > 15.0:
        print("  *** WARNING: the 'bay' region is implausibly large.  The line")
        print("      probably does not span the mouth from shore to shore, so")
        print("      the flood fill leaked around its end into the open ocean.")
        print("      Extend p1/p2 until both endpoints sit ON land.")

    m0 = np.where(water, np.where(bay, 0.0, 1.0), np.nan)

    # --- 2. diffuse the step, through water only
    #  variance after n steps = 2*alpha*n*dx^2  ->  sigma = dx*sqrt(2*alpha*n)
    nst = int(round((args.sigma_km * 1000.0 / dx) ** 2 / (2 * ALPHA)))
    print(f"  diffusing {nst} steps for sigma = {args.sigma_km} km "
          f"({args.sigma_km*1000/dx:.1f} cells)")
    m1 = diffuse_in_water(np.nan_to_num(m0), water, nst)
    m1 = np.where(water, np.clip(m1, 0.0, 1.0), np.nan)

    # --- 3. report the realised transition
    inb = water & bay
    out = water & ~bay
    print(f"  after diffusion:  bay-side mean {np.nanmean(m1[inb]):.3f}, "
          f"ocean-side mean {np.nanmean(m1[out]):.3f}")
    print(f"  cells with 0.05 < mask < 0.95 : {int(((m1>0.05)&(m1<0.95)&water).sum()):,}"
          f"  ({100*((m1>0.05)&(m1<0.95)&water).sum()/water.sum():.1f}% of water)")

    # --- 3b. save, so the increment pipeline can just multiply it in
    # PFM copy: write beside the masks the pipeline reads, i.e. da_config.MASKDIR.
    MASKDIR = Path(__file__).resolve().parent.parent / "masks"
    MASKDIR.mkdir(parents=True, exist_ok=True)
    mf = MASKDIR / (f"{args.level}_incr_mask_sigma{args.sigma_km:g}km"
                    f"{'_'+args.tag if args.tag else ''}.nc")
    with nc.Dataset(mf, "w", format="NETCDF4") as o:
        o.createDimension("eta_rho", lon.shape[0])
        o.createDimension("xi_rho", lon.shape[1])
        v = o.createVariable("mask_incr", "f4", ("eta_rho", "xi_rho"),
                             zlib=True, complevel=4)
        v.long_name = ("multiplier for u/v/temp/salt DA increments; 0 in "
                       "enclosed water, 1 on the shelf")
        v.units = "1"
        v[:] = np.where(water, m1, 0.0).astype("f4")
        for nm, arr in (("lon_rho", lon), ("lat_rho", lat)):
            vv = o.createVariable(nm, "f8", ("eta_rho", "xi_rho"))
            vv[:] = arr
        o.setncatts(dict(
            level=args.level, p1=str(p1), p2=str(p2), seed=str(sd),
            sigma_km=args.sigma_km, tag=args.tag,
            method=("sharp 0/1 split across the line, then explicit diffusion "
                    "restricted to water with no-flux at land"),
            note="zeta is NOT masked; this is for u, v, temp, salt only"))
    print(f"  wrote {mf}")

    # --- 4. plot
    OUTDIR.mkdir(parents=True, exist_ok=True)
    LAND = ListedColormap(["#d9d9d9"])
    asp = 1.0 / np.cos(np.deg2rad(lat0))
    # zoom on whatever region the flood fill found, plus a margin, so this
    # works for the TJRE on LV4 as well as San Diego Bay on LV3
    bl, bt = lon[bay], lat[bay]
    mlon = 0.35 * max(np.ptp(bl), 0.02) + 0.02
    mlat = 0.35 * max(np.ptp(bt), 0.02) + 0.02
    zoom = (bl.min() - mlon, bl.max() + mlon, bt.min() - mlat, bt.max() + mlat)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 11.0))
    for row, (fld, ttl) in enumerate(((m0, "before  —  sharp 0 / 1 split"),
                                      (m1, f"after  —  diffused through water, "
                                           f"$\\sigma$ = {args.sigma_km:g} km"))):
        for col, ext in enumerate((None, zoom)):
            ax = axes[row, col]
            ax.pcolormesh(lon, lat, np.where(water, np.nan, 1.0), cmap=LAND,
                          shading="auto", zorder=1)
            im = ax.pcolormesh(lon, lat, np.ma.masked_invalid(fld),
                               cmap="viridis", vmin=0, vmax=1,
                               shading="auto", zorder=2, rasterized=True)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#d62728", lw=2.0,
                    zorder=5)
            ax.plot(*sd, marker="*", ms=11, mfc="#d62728", mec="white",
                    mew=1.0, zorder=6)
            if ext:
                ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.set_aspect(asp)
            ax.set_title(f"{ttl}{'  (bay mouth)' if ext else ''}",
                         fontsize=10.5, pad=5)
            ax.tick_params(labelsize=8, length=2.5, color="#999999")
            for s in ax.spines.values():
                s.set_edgecolor("#bbbbbb"); s.set_linewidth(0.8)
    fig.suptitle(f"{args.level} enclosed-water mask for damping u, v, temp, salt "
                 f"increments  (zeta left unmasked)", fontsize=13, y=0.965)
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.040,
                      pad=0.055, aspect=46)
    cb.set_label("mask   (0 = bay, increment suppressed    1 = shelf, "
                 "increment kept)", fontsize=10)
    cb.ax.tick_params(labelsize=8.5); cb.outline.set_edgecolor("#bbbbbb")
    fp = OUTDIR / (f"{args.level}_bay_mask_sigma{args.sigma_km:g}km"
                   f"{'_'+args.tag if args.tag else ''}.png")
    fig.savefig(fp, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  wrote {fp}")


if __name__ == "__main__":
    main()
