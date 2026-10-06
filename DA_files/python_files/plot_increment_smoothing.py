#!/usr/bin/env python3
"""Four figures comparing the raw vs smoothed surface-u DA increment on LV1-LV4.

  fig1  raw increment, each level on its own full grid
  fig2  raw increment, every level zoomed to the LV3 footprint
        (LV4 panel: LV3 field behind, LV4 field on top, LV4 outline drawn)
  fig3  as fig1, with the smoothing we are about to implement
  fig4  as fig2, with that smoothing

The child fields are produced by the same horizontal path the real pipeline
uses (`_interp_arrays_to_child`): LV1 u,v -> rho points -> rotate by the LV1
angle to earth (E,N) -> LinearND interpolation with `land_fill="zero"` ->
rotate by the child angle onto the child XI axis -> mask.

The vertical s-level remap the pipeline also does is skipped: it is a small
correction at the surface, where parent and child top levels nearly coincide,
and it costs a per-column python loop (~550k columns on LV4).

Smoothing is `smooth_increment_ic._smooth_layer`: zero on land, gaussian_filter
with sigma = sigma_km*1000/dx_mean cells, zero on land again.  LV1 is the DA
grid itself -- its increment is native ROMS output, never interpolated, so
nothing is smoothed there and its panel is identical in fig1/fig3.

Usage:  python3 plot_increment_smoothing.py [--date 20260307] [--tag _bcstale]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay

GRIDS = Path("/scratch/PFM_Simulations/Grids")
LV1_V2 = Path("/dataSIO/DA_Simulations/LV1_v2")
OUTDIR = Path("/scratch/PFM_Simulations/DA_Sim/Log/figures")

# grid file, mean dx (m), smoothing sigma (m).  sigma=None -> no smoothing.
LEVELS = {
    "LV1": dict(grid="GRID_SDTJRE_LV1_rx020_hmask.nc",       dx=1920.5, sigma=None),
    "LV2": dict(grid="GRID_SDTJRE_LV2_rx020.nc",              dx=625.8, sigma=2000.0),
    "LV3": dict(grid="GRID_SDTJRE_LV3_rx020_maskCoast.nc",    dx=206.9, sigma=2000.0),
    "LV4": dict(grid="GRID_SDTJRE_LV4_mss_oct2024.nc",         dx=32.0, sigma=100.0),
}
SBOO = (32.531952, -117.18613)

# Which state variable the figures show.  Set from --field in main().
#   "u"    surface u increment, lives on the u-grid, needs vector rotation
#   "zeta" free-surface increment, lives on the rho-grid, a plain scalar
FIELD = "u"
FIELD_META = {
    "u":    dict(sym="surface $u$", unit="m s$^{-1}$",
                 gunit="(m s$^{-1}$) km$^{-1}$", gsym=r"|\nabla(\Delta u)|"),
    "v":    dict(sym="surface $v$", unit="m s$^{-1}$",
                 gunit="(m s$^{-1}$) km$^{-1}$", gsym=r"|\nabla(\Delta v)|"),
    "zeta": dict(sym=r"$\zeta$", unit="m",
                 gunit="m km$^{-1}$", gsym=r"|\nabla(\Delta\zeta)|"),
    "temp": dict(sym="surface temp", unit=r"$^\circ$C",
                 gunit=r"$^\circ$C km$^{-1}$", gsym=r"|\nabla(\Delta T)|"),
    "salt": dict(sym="surface salt", unit="psu",
                 gunit="psu km$^{-1}$", gsym=r"|\nabla(\Delta S)|"),
}

# fields that live on rho points and need no vector rotation
SCALARS = ("zeta", "temp", "salt")


def geom(G):
    """(lon, lat, mask, pm, pn) for the grid the current FIELD lives on."""
    if FIELD in SCALARS:
        return G["lon_rho"], G["lat_rho"], G["mask_rho"], G["pm"], G["pn"]
    if FIELD == "v":
        mr = G["mask_rho"]
        return (G["lon_v"], G["lat_v"], mr[:-1, :] & mr[1:, :],
                0.5 * (G["pm"][:-1, :] + G["pm"][1:, :]),
                0.5 * (G["pn"][:-1, :] + G["pn"][1:, :]))
    return (G["lon_u"], G["lat_u"], G["mask_u"],
            0.5 * (G["pm"][:, :-1] + G["pm"][:, 1:]),
            0.5 * (G["pn"][:, :-1] + G["pn"][:, 1:]))


def _san(a):
    """ROMS writes ~1e37 on land / unused points."""
    return np.where(np.abs(a) > 1e30, 0.0, a)


def load_grid(name):
    g = LEVELS[name]
    with nc.Dataset(GRIDS / g["grid"]) as d:
        mr = np.asarray(d["mask_rho"][:]).astype(bool)
        out = dict(
            lon_rho=np.asarray(d["lon_rho"][:]), lat_rho=np.asarray(d["lat_rho"][:]),
            lon_u=np.asarray(d["lon_u"][:]),     lat_u=np.asarray(d["lat_u"][:]),
            lon_v=np.asarray(d["lon_v"][:]),     lat_v=np.asarray(d["lat_v"][:]),
            angle=np.asarray(d["angle"][:]), mask_rho=mr,
            mask_u=mr[:, :-1] & mr[:, 1:],
            pm=np.asarray(d["pm"][:]), pn=np.asarray(d["pn"][:]),
        )
    out.update(dx=g["dx"], sigma=g["sigma"])
    return out


def lv1_surface_uv(date, tag):
    """LV1 surface u,v increment = outer1(t=0) - outer0(t=0), top s_rho level."""
    o1 = LV1_V2 / "analysis_files" / f"LV1_fwd_outer1_{date}{tag}.nc"
    o0 = LV1_V2 / "prior_files" / f"LV1_fwd_outer0_{date}{tag}.nc"
    for f in (o0, o1):
        if not f.exists():
            raise SystemExit(f"missing {f}")
    with nc.Dataset(o1) as a, nc.Dataset(o0) as b:
        if FIELD in SCALARS:
            if FIELD == "zeta":
                d = _san(np.asarray(a["zeta"][0])) - _san(np.asarray(b["zeta"][0]))
            else:                                   # temp / salt, surface level
                d = (_san(np.asarray(a[FIELD][0, -1]))
                     - _san(np.asarray(b[FIELD][0, -1])))
            return d, None
        du = _san(np.asarray(a["u"][0, -1])) - _san(np.asarray(b["u"][0, -1]))
        dv = _san(np.asarray(a["v"][0, -1])) - _san(np.asarray(b["v"][0, -1]))
    return du, dv


class ZeroFillInterp:
    """The pipeline's `_CurviInterp`, inlined so this script stands alone.

    land_fill="zero"    parent land forced to 0 before the linear interp --
                        what the pipeline does today, so child coastal cells
                        taper toward *no perturbation*.
    land_fill="nearest" parent land takes the value of the nearest wet cell,
                        so triangles spanning a coast stay continuous and the
                        child inherits an ocean value instead of a zero that
                        belongs to the PARENT's coastline, not its own.
    """

    def __init__(self, lon, lat, mask, land_fill="zero"):
        self.pts = np.column_stack([lon.ravel(), lat.ravel()])
        self.tri = Delaunay(self.pts)
        self.land = ~mask.ravel().astype(bool)
        self.land_fill = land_fill
        if land_fill == "nearest" and self.land.any():
            wet = np.where(~self.land)[0]
            _, nn = cKDTree(self.pts[wet]).query(self.pts[self.land])
            self.nn_from = wet[nn]

    def __call__(self, field, xy):
        v = field.ravel().astype(float, copy=True)
        if self.land_fill == "nearest" and self.land.any():
            v[self.land] = v[self.nn_from]
        else:
            v[self.land] = 0.0
        out = LinearNDInterpolator(self.tri, v)(xy)
        if np.isnan(out).any():
            bad = np.isnan(out)
            out[bad] = NearestNDInterpolator(self.pts, v)(xy[bad])
        return out


def lv1_to_child_u(du, dv, P, C, ip):
    """LV1 -> child, on the grid the current FIELD lives on.

    zeta is a scalar: one interpolation, no rotation.  u needs the full
    vector path, since the child XI axis points a different way than LV1's.
    """
    if FIELD in SCALARS:
        xy = np.column_stack([C["lon_rho"].ravel(), C["lat_rho"].ravel()])
        out = ip(du, xy).reshape(C["lon_rho"].shape)
        return np.where(C["mask_rho"], out, np.nan)

    u_r = np.zeros_like(P["lat_rho"]); v_r = np.zeros_like(P["lat_rho"])
    u_r[:, 1:-1] = 0.5 * (du[:, :-1] + du[:, 1:])
    v_r[1:-1, :] = 0.5 * (dv[:-1, :] + dv[1:, :])
    cosA, sinA = np.cos(P["angle"]), np.sin(P["angle"])
    uE = u_r * cosA - v_r * sinA
    vN = u_r * sinA + v_r * cosA

    if FIELD == "v":
        xy = np.column_stack([C["lon_v"].ravel(), C["lat_v"].ravel()])
        ang_v = 0.5 * (C["angle"][:-1, :] + C["angle"][1:, :])
        out = (-ip(uE, xy) * np.sin(ang_v).ravel()
               + ip(vN, xy) * np.cos(ang_v).ravel()).reshape(C["lon_v"].shape)
        return np.where(geom(C)[2], out, np.nan)

    xy = np.column_stack([C["lon_u"].ravel(), C["lat_u"].ravel()])
    ang_u = 0.5 * (C["angle"][:, :-1] + C["angle"][:, 1:])
    out = (ip(uE, xy) * np.cos(ang_u).ravel()
           + ip(vN, xy) * np.sin(ang_u).ravel()).reshape(C["lon_u"].shape)
    return np.where(C["mask_u"], out, np.nan)


def smooth(field, mask, sigma_m, dx, shepard=False):
    """smooth_increment_ic._smooth_layer: zero land, blur, zero land.

    With `shepard=True`, divide by the identically-blurred mask.  The plain
    form convolves water values against land zeros, so within ~sigma of the
    coast the result is a weighted SUM over a kernel that is only partly over
    water -- an amplitude loss, and on a grid whose mask is finer than the
    parent's, a manufactured ramp from the ambient increment down to zero.
    Normalizing by A = G_sigma(mask) turns that sum into a weighted MEAN over
    the water cells actually in the kernel, which preserves amplitude to the
    shoreline while still removing the interpolation creases.

    The quotient is a convex combination of nearby water values, so it is
    bounded by the field's own range no matter how small A gets; the floor
    below only guards 0/0 in fully enclosed cells.
    """
    if sigma_m is None:
        return field
    sc = sigma_m / dx
    f = np.where(mask, np.nan_to_num(field), 0.0).astype("f4")
    s = gaussian_filter(f, sigma=sc, mode="nearest")
    if shepard:
        A = gaussian_filter(mask.astype("f4"), sigma=sc, mode="nearest")
        s = np.where(A > 1e-3, s / np.maximum(A, 1e-3), 0.0)
    return np.where(mask, s, np.nan)


def grad_mag(field, G):
    """|grad(du)| on the u-grid, in (m/s) per km.

    Central differences scaled by the grid metrics averaged onto u points.
    The field is NaN on land, so any stencil touching the coast propagates
    NaN -- a difference between a water cell and an undefined land cell is
    not a gradient, and including it would paint a bright rim along every
    coastline that has nothing to do with the increment.

    Note what this figure is really showing on the child grids: LinearND
    interpolation is piecewise LINEAR, so its gradient is piecewise CONSTANT
    and jumps across every triangle edge.  The raw child panels are therefore
    a mosaic of flat facets -- that mosaic IS the "step" artifact.
    """
    _, _, mask, pm_u, pn_u = geom(G)
    f = np.where(mask, field, np.nan)
    gx = np.full_like(f, np.nan, dtype="f8")
    gy = np.full_like(f, np.nan, dtype="f8")
    gx[:, 1:-1] = 0.5 * (f[:, 2:] - f[:, :-2]) * pm_u[:, 1:-1]
    gy[1:-1, :] = 0.5 * (f[2:, :] - f[:-2, :]) * pn_u[1:-1, :]
    return np.hypot(gx, gy) * 1000.0          # (m/s) / km


# ---------------------------------------------------------------- plotting --
LAND = ListedColormap(["#d9d9d9"])          # light gray land
ASPECT = 1.0 / np.cos(np.deg2rad(32.6))


def draw_panel(ax, G, field, vlim, *, title, extent=None,
               behind=None, outline=False, cmap="RdBu_r", vmin=None):
    """Land first, optional 'behind' field, then this level's field on top."""
    if behind is not None:
        Gb, fb = behind
        ax.pcolormesh(Gb["lon_rho"], Gb["lat_rho"],
                      np.where(Gb["mask_rho"], np.nan, 1.0),
                      cmap=LAND, shading="auto", zorder=1)
        lob, lab_, _, _, _ = geom(Gb)
        ax.pcolormesh(lob, lab_, np.ma.masked_invalid(fb),
                      cmap=cmap, vmin=(-vlim if vmin is None else vmin),
                      vmax=vlim, shading="auto", zorder=2, rasterized=True)

    ax.pcolormesh(G["lon_rho"], G["lat_rho"],
                  np.where(G["mask_rho"], np.nan, 1.0),
                  cmap=LAND, shading="auto", zorder=3)
    lo_f, la_f, _, _, _ = geom(G)
    im = ax.pcolormesh(lo_f, la_f, np.ma.masked_invalid(field),
                       cmap=cmap, vmin=(-vlim if vmin is None else vmin),
                       vmax=vlim, shading="auto", zorder=4, rasterized=True)

    if outline:
        lo, la = G["lon_rho"], G["lat_rho"]
        bx = np.concatenate([lo[0, :], lo[:, -1], lo[-1, ::-1], lo[::-1, 0]])
        by = np.concatenate([la[0, :], la[:, -1], la[-1, ::-1], la[::-1, 0]])
        ax.plot(bx, by, color="#111111", lw=1.4, zorder=6)

    ax.plot(*SBOO[::-1], marker="o", ms=5, mfc="none", mec="#111111",
            mew=1.3, zorder=7)

    if extent is not None:
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(ASPECT)
    ax.set_title(title, fontsize=10.5, pad=5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(labelsize=8, length=2.5, color="#999999")
    for s in ax.spines.values():
        s.set_edgecolor("#bbbbbb"); s.set_linewidth(0.8)
    return im


def make_figure(path, fields, grids, vlim, *, suptitle, extent=None,
                lv4_behind=None, smoothed=False, shepard=False,
                cmap="RdBu_r", vmin=None,
                cblabel=None):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 10.4))
    for ax, name in zip(axes.ravel(), ("LV1", "LV2", "LV3", "LV4")):
        G = grids[name]
        sig = G["sigma"]
        if name == "LV1":
            tag = "  |  native (not interpolated, not smoothed)"
        elif smoothed and shepard:
            tag = f"  |  Shepard, $\\sigma$ = {sig/1000:g} km"
        elif smoothed:
            tag = f"  |  smoothed, $\\sigma$ = {sig/1000:g} km"
        else:
            tag = "  |  raw interpolation from LV1"
        behind = lv4_behind if (name == "LV4" and lv4_behind is not None) else None
        im = draw_panel(ax, G, fields[name], vlim,
                        title=f"{name}   dx $\\approx$ {G['dx']:.0f} m{tag}",
                        extent=extent, behind=behind, cmap=cmap, vmin=vmin,
                        outline=(name == "LV4" and extent is not None))
    for ax in axes[:, 0]:
        ax.set_ylabel("latitude", fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel("longitude", fontsize=9)

    fig.suptitle(suptitle, fontsize=13, y=0.975)
    cb = fig.colorbar(im, ax=axes, orientation="horizontal",
                      fraction=0.040, pad=0.055, aspect=46,
                      extend=("both" if vmin is None else "max"))
    M = FIELD_META[FIELD]
    if cblabel is None:
        cblabel = (f"{M['sym']} increment,  outer1 $-$ outer0   ({M['unit']})")
    cb.set_label(cblabel, fontsize=10)
    cb.ax.tick_params(labelsize=8.5)
    cb.outline.set_edgecolor("#bbbbbb")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20260307")
    ap.add_argument("--tag", default="_bcstale")
    ap.add_argument("--field", default="u", choices=("u", "v", "zeta"),
                    help="state variable to plot (default: u)")
    ap.add_argument("--fill", default="zero", choices=("zero", "nearest"),
                    help="how LV1 land is filled before interpolating to a child")
    ap.add_argument("--root", default=None,
                    help="tree holding prior_files/ and analysis_files/ "
                         "(default: the LV1_v2 chain)")
    ap.add_argument("--sigma", default=None,
                    help="override smoothing lengths, km, e.g. 'LV4=2' or "
                         "'LV2=2,LV3=2,LV4=2' (default: 2/2/0.1)")
    args = ap.parse_args()

    if args.sigma:
        for item in args.sigma.split(","):
            lvl, val = item.split("=")
            lvl = lvl.strip().upper()
            LEVELS[lvl]["sigma"] = float(val) * 1000.0
            print(f"  sigma override: {lvl} -> {float(val):g} km")

    global FIELD, LV1_V2
    if args.root:
        LV1_V2 = Path(args.root)
        print(f"  LV1 source tree: {LV1_V2}")
    FIELD = args.field
    M = FIELD_META[FIELD]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    grids = {k: load_grid(k) for k in LEVELS}
    du, dv = lv1_surface_uv(args.date, args.tag)

    print("interpolating LV1 -> children (pipeline horizontal path) ...")
    ip = ZeroFillInterp(grids["LV1"]["lon_rho"], grids["LV1"]["lat_rho"],
                        grids["LV1"]["mask_rho"], land_fill=args.fill)
    print(f"  LV1 land_fill = {args.fill!r}")
    prim = dv if FIELD == "v" else du
    raw = {"LV1": np.where(geom(grids["LV1"])[2], prim, np.nan)}
    for k in ("LV2", "LV3", "LV4"):
        raw[k] = lv1_to_child_u(du, dv, grids["LV1"], grids[k], ip)
        print(f"  {k}: max|incr| = {np.nanmax(np.abs(raw[k])):.5f} "
              f"{'m' if FIELD=='zeta' else 'm/s'}")

    sm = {k: smooth(raw[k], geom(grids[k])[2], grids[k]["sigma"], grids[k]["dx"])
          for k in raw}

    vlim = float(np.nanpercentile(np.abs(raw["LV1"]), 99.5))
    print(f"\ncolor limit (99.5th pct of |LV1 increment|) = +/- {vlim:.4f}")

    g3 = grids["LV3"]
    ext3 = (g3["lon_rho"].min(), g3["lon_rho"].max(),
            g3["lat_rho"].min(), g3["lat_rho"].max())

    d = f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"
    FS = "" if args.fill == "zero" else "_nnfill"
    FL = "" if args.fill == "zero" else ", LV1 land = nearest-water"
    print()
    make_figure(OUTDIR / f"fig1_raw{FS}_full_grids.png", raw, grids, vlim,
                suptitle=f"RAW {M['sym']} increment, direct LV1$\\to$child{FL} — {d}")
    make_figure(OUTDIR / f"fig2_raw{FS}_zoom_LV3.png", raw, grids, vlim,
                suptitle=f"Raw DA {M['sym']} increment, zoomed to the LV3 footprint — {d}",
                extent=ext3, lv4_behind=(grids["LV3"], raw["LV3"]))
    make_figure(OUTDIR / f"fig3_smoothed{FS}_full_grids.png", sm, grids, vlim, smoothed=True,
                suptitle=f"Smoothed DA {M['sym']} increment — {d}")
    make_figure(OUTDIR / f"fig4_smoothed{FS}_zoom_LV3.png", sm, grids, vlim, smoothed=True,
                suptitle=f"Smoothed DA {M['sym']} increment, zoomed to the LV3 footprint — {d}",
                extent=ext3, lv4_behind=(grids["LV3"], sm["LV3"]))

    # ---------------- gradient magnitude ----------------------------------
    graw = {k: grad_mag(raw[k], grids[k]) for k in raw}
    gsm  = {k: grad_mag(sm[k],  grids[k]) for k in sm}
    gmax = float(np.nanpercentile(np.concatenate(
        [graw[k][np.isfinite(graw[k])].ravel() for k in ("LV2", "LV3", "LV4")]), 99.0))
    gu = "m/km" if FIELD == "zeta" else "(m/s)/km"
    print(f"\ngradient color limit (99th pct of raw child |grad|) = {gmax:.5f} {gu}")
    for k in raw:
        a = graw[k][np.isfinite(graw[k])]; b = gsm[k][np.isfinite(gsm[k])]
        print(f"  {k}: median |grad| raw {np.median(a):.5f} -> smoothed "
              f"{np.median(b):.5f} {gu}   max {a.max():.4f} -> {b.max():.4f}")
    # ---------------- Shepard-normalized smoothing ------------------------
    shp = {k: smooth(raw[k], geom(grids[k])[2], grids[k]["sigma"],
                     grids[k]["dx"], shepard=True) for k in raw}
    gshp = {k: grad_mag(shp[k], grids[k]) for k in shp}

    GL = f"${M['gsym']}$   ({M['gunit']})"
    print()
    make_figure(OUTDIR / f"fig5_grad_raw{FS}_full_grids.png", graw, grids, gmax,
                cmap="Oranges", vmin=0.0, cblabel=GL,
                suptitle=f"Gradient magnitude, RAW {M['sym']} increment, direct LV1$\\to$child{FL} — {d}")
    make_figure(OUTDIR / f"fig6_grad_raw{FS}_zoom_LV3.png", graw, grids, gmax,
                cmap="Oranges", vmin=0.0, cblabel=GL, extent=ext3,
                lv4_behind=(grids["LV3"], graw["LV3"]),
                suptitle=f"Gradient magnitude, RAW increment, zoomed to the LV3 footprint — {d}")
    make_figure(OUTDIR / f"fig7_grad_smoothed{FS}_full_grids.png", gsm, grids, gmax,
                cmap="Oranges", vmin=0.0, cblabel=GL, smoothed=True,
                suptitle=f"Gradient magnitude of the SMOOTHED {M['sym']} increment — {d}")
    make_figure(OUTDIR / f"fig8_grad_smoothed{FS}_zoom_LV3.png", gsm, grids, gmax,
                cmap="Oranges", vmin=0.0, cblabel=GL, smoothed=True, extent=ext3,
                lv4_behind=(grids["LV3"], gsm["LV3"]),
                suptitle=f"Gradient magnitude, SMOOTHED increment, zoomed to the LV3 footprint — {d}")

    print()
    make_figure(OUTDIR / f"fig9_shepard{FS}_full_grids.png", shp, grids, vlim,
                smoothed=True, shepard=True,
                suptitle=f"Shepard-normalized smoothed {M['sym']} increment — {d}")
    make_figure(OUTDIR / f"fig10_shepard{FS}_zoom_LV3.png", shp, grids, vlim,
                smoothed=True, shepard=True, extent=ext3,
                lv4_behind=(grids["LV3"], shp["LV3"]),
                suptitle=f"Shepard-normalized smoothed {M['sym']} increment, "
                         f"zoomed to the LV3 footprint — {d}")
    make_figure(OUTDIR / f"fig11_grad_shepard{FS}_full_grids.png", gshp, grids, gmax,
                cmap="Oranges", vmin=0.0, cblabel=GL, smoothed=True, shepard=True,
                suptitle=f"Gradient magnitude, SHEPARD-normalized smoothing — {d}")
    make_figure(OUTDIR / f"fig12_grad_shepard{FS}_zoom_LV3.png", gshp, grids, gmax,
                cmap="Oranges", vmin=0.0, cblabel=GL, smoothed=True, shepard=True,
                extent=ext3, lv4_behind=(grids["LV3"], gshp["LV3"]),
                suptitle="Gradient magnitude, SHEPARD-normalized smoothing, "
                         f"zoomed to the LV3 footprint — {d}")


if __name__ == "__main__":
    main()
