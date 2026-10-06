#!/usr/bin/env python3
"""
parent_history_to_BC.py

Build a ROMS boundary-condition (BC) file for a child grid by horizontally
interpolating a parent-grid ROMS history file. Parent and child grids may
have different curvilinear geometry, different rotations, and different
bathymetry. Vertical stretching parameters are assumed identical (asserted
at runtime); with identical Cs_r the vertical remap reduces to a linear-in-z
correction driven only by the h and zeta differences between the two grids.

Primary API:
    history_file_to_BC_file(parent_grid, child_grid, history_file, bc_file)

Rotation:
    Parent (u, v) live on the parent xi/eta axes. We rotate them to true
    (E, N) using the parent's `angle`, horizontally interpolate to the child
    boundary rho points, then rotate (E, N) -> child xi/eta axes using the
    child's `angle`. Rotated components are then averaged onto the child's u
    and v boundary strips.
"""
from __future__ import annotations
import argparse
import datetime as DT
from pathlib import Path

import numpy as np
import netCDF4 as nc
from scipy.spatial import Delaunay, cKDTree
from scipy.interpolate import LinearNDInterpolator


T_REF = DT.datetime(1999, 1, 1, 0, 0, 0)
BOUNDARIES = ("south", "north", "west", "east")


# ---------------------------------------------------------------------------
# Grid + boundary helpers
# ---------------------------------------------------------------------------

def _load_grid(fn: str) -> dict:
    with nc.Dataset(fn) as g:
        out = dict(
            lat_rho=np.asarray(g.variables["lat_rho"][:]),
            lon_rho=np.asarray(g.variables["lon_rho"][:]),
            mask_rho=np.asarray(g.variables["mask_rho"][:], dtype=bool),
            angle=np.asarray(g.variables["angle"][:]),
            h=np.asarray(g.variables["h"][:]),
        )
        for k in ("lat_u", "lon_u", "lat_v", "lon_v", "mask_u", "mask_v"):
            if k in g.variables:
                out[k] = np.asarray(g.variables[k][:])
    # angle at u/v points: bilinear from rho if not present
    if "lat_u" not in out:
        out["lat_u"] = 0.5 * (out["lat_rho"][:, :-1] + out["lat_rho"][:, 1:])
        out["lon_u"] = 0.5 * (out["lon_rho"][:, :-1] + out["lon_rho"][:, 1:])
        out["lat_v"] = 0.5 * (out["lat_rho"][:-1, :] + out["lat_rho"][1:, :])
        out["lon_v"] = 0.5 * (out["lon_rho"][:-1, :] + out["lon_rho"][1:, :])
    out["angle_u"] = 0.5 * (out["angle"][:, :-1] + out["angle"][:, 1:])
    out["angle_v"] = 0.5 * (out["angle"][:-1, :] + out["angle"][1:, :])
    out["h_u"] = 0.5 * (out["h"][:, :-1] + out["h"][:, 1:])
    out["h_v"] = 0.5 * (out["h"][:-1, :] + out["h"][1:, :])
    # u/v masks — a cell is water iff both adjacent rho cells are water
    if "mask_u" not in out:
        out["mask_u"] = (out["mask_rho"][:, :-1] & out["mask_rho"][:, 1:])
    else:
        out["mask_u"] = out["mask_u"].astype(bool)
    if "mask_v" not in out:
        out["mask_v"] = (out["mask_rho"][:-1, :] & out["mask_rho"][1:, :])
    else:
        out["mask_v"] = out["mask_v"].astype(bool)
    return out


def _child_boundary_points(cg: dict) -> dict:
    """For each boundary (south/north/west/east), return dict of rho/u/v
    strips containing (lon, lat, h, angle, mask)."""
    def _slice(bnd, arr, kind):
        # kind in {'rho','u','v'} — chooses which grid's strip index to use
        if bnd == "south":  return arr[0, :]
        if bnd == "north":  return arr[-1, :]
        if bnd == "west":   return arr[:, 0]
        return arr[:, -1]  # east

    B = {}
    for bnd in BOUNDARIES:
        b = {}
        b["rho"] = dict(lon=_slice(bnd, cg["lon_rho"], "rho"),
                        lat=_slice(bnd, cg["lat_rho"], "rho"),
                        h=_slice(bnd, cg["h"], "rho"),
                        angle=_slice(bnd, cg["angle"], "rho"),
                        mask=_slice(bnd, cg["mask_rho"], "rho").astype(float))
        b["u"]   = dict(lon=_slice(bnd, cg["lon_u"], "u"),
                        lat=_slice(bnd, cg["lat_u"], "u"),
                        h=_slice(bnd, cg["h_u"], "u"),
                        angle=_slice(bnd, cg["angle_u"], "u"),
                        mask=_slice(bnd, cg["mask_u"], "u").astype(float))
        b["v"]   = dict(lon=_slice(bnd, cg["lon_v"], "v"),
                        lat=_slice(bnd, cg["lat_v"], "v"),
                        h=_slice(bnd, cg["h_v"], "v"),
                        angle=_slice(bnd, cg["angle_v"], "v"),
                        mask=_slice(bnd, cg["mask_v"], "v").astype(float))
        B[bnd] = b
    return B


# ---------------------------------------------------------------------------
# Vertical
# ---------------------------------------------------------------------------

def _z_rho(h: np.ndarray, zeta: np.ndarray,
           Vtransform: int, hc: float,
           Cs_r: np.ndarray, s_r: np.ndarray) -> np.ndarray:
    """Return z_rho with leading dim = N.  Accepts h/zeta as scalar, 1D
    (boundary strip), or 2D (full field).  Broadcasts:
        h.shape == zeta.shape  =>  z_rho.shape == (N,) + h.shape"""
    h    = np.asarray(h,    dtype=float)
    zeta = np.asarray(zeta, dtype=float)
    N    = len(s_r)
    tail = (1,) * h.ndim
    sN = s_r.reshape((N,) + tail)     # broadcasts against h[None,...]
    cN = Cs_r.reshape((N,) + tail)
    Sk = (hc * sN + h[None, ...] * cN) / (hc + h[None, ...])
    if Vtransform == 2:
        z = zeta[None, ...] + (zeta[None, ...] + h[None, ...]) * Sk
    else:  # Vtransform == 1
        z = h[None, ...] * Sk + zeta[None, ...] * (1 + Sk / h[None, ...])
    return z


def _linear_interp_z(src_z: np.ndarray, src_f: np.ndarray, tgt_z: np.ndarray) -> np.ndarray:
    """Linear interpolation along axis 0.
    src_z, src_f: (Ns,) monotonic (either direction). tgt_z: (Nt,)."""
    # ensure src is ascending in z
    if src_z[0] > src_z[-1]:
        src_z = src_z[::-1]
        src_f = src_f[::-1]
    # numpy.interp requires ascending xp; extrapolate flat at ends
    return np.interp(tgt_z, src_z, src_f)


# ---------------------------------------------------------------------------
# Horizontal interp on curvilinear parent
# ---------------------------------------------------------------------------

class _CurviInterp:
    """Precomputes a Delaunay of a parent grid once, then interpolates any
    number of scalar fields with LinearNDInterpolator (which reuses the tri).

    Parent land cells are KEPT in the triangulation; how their values are
    filled before interp is chosen by `land_fill`:
      * 'nearest' (default) — copy from nearest water cell.  Right for full
        state fields (temp, salt, u, v, zeta, ubar, vbar): keeps triangles
        that span coasts physically continuous, no zero drag on LV2 coastal
        cells.  Standard oceanographic regridding choice.
      * 'zero' — force land values to 0.  Right for DA increment fields:
        we don't know what the increment should be over land, so the safest
        assumption is 0 (no perturbation), and the LV2 coastal water cells
        will be pulled toward 0 as they should be near the LV1 coast."""

    def __init__(self, lon: np.ndarray, lat: np.ndarray,
                 mask: np.ndarray | None, land_fill: str = "nearest"):
        assert land_fill in ("nearest", "zero"), f"unknown land_fill={land_fill!r}"
        self._land_fill = land_fill
        self._pts = np.column_stack([lon.ravel(), lat.ravel()])
        self._tri = Delaunay(self._pts)
        if mask is not None:
            water = mask.ravel().astype(bool)
            self._land = ~water
            if land_fill == "nearest":
                water_idx = np.where(water)[0]
                if water_idx.size == 0 or self._land.sum() == 0:
                    self._nn_from = None
                else:
                    tree = cKDTree(self._pts[water_idx])
                    _, nn = tree.query(self._pts[self._land])
                    self._nn_from = water_idx[nn]
            else:  # 'zero'
                self._nn_from = None
        else:
            self._land = np.zeros(self._pts.shape[0], dtype=bool)
            self._nn_from = None

    def interp(self, field: np.ndarray, xy: np.ndarray) -> np.ndarray:
        """field: same shape as lon/lat used to build. xy: (M, 2)."""
        vals = field.ravel().astype(float, copy=True)
        if self._land_fill == "nearest" and self._nn_from is not None:
            vals[self._land] = vals[self._nn_from]
        elif self._land_fill == "zero":
            vals[self._land] = 0.0
        f = LinearNDInterpolator(self._tri, vals)
        out = f(xy)
        # nearest-neighbor fill for query points outside the parent hull
        if np.isnan(out).any():
            from scipy.interpolate import NearestNDInterpolator
            nf = NearestNDInterpolator(self._pts, vals)
            bad = np.isnan(out)
            out[bad] = nf(xy[bad])
        return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def read_vparams(fn: str) -> dict:
    """Extract (Vt, hc, Cs_r, s_rho) from any ROMS file that carries them
    (IC, rst, his, mod, ...).  Returned dict is the shape expected by the
    `child_vparams=` kwargs of history_file_to_BC_file and
    interp_increment_to_child."""
    with nc.Dataset(fn) as d:
        return dict(
            Vt=int(np.asarray(d.variables["Vtransform"][...])),
            hc=float(np.asarray(d.variables["hc"][...])),
            Cs_r=np.asarray(d.variables["Cs_r"][:]),
            s_rho=np.asarray(d.variables["s_rho"][:]),
        )


def history_file_to_BC_file(parent_grid_fn: str, child_grid_fn: str,
                            history_fn: str, bc_out_fn: str,
                            child_vparams: dict | None = None) -> None:
    """Full pipeline: build a child-grid BC file from a parent-grid history file.

    child_vparams: if the child grid uses a DIFFERENT vertical stretching
    from the parent (e.g. LV3->LV4 with N=40->10 and different theta_s/Tcline),
    pass dict(Vt=..., hc=..., Cs_r=array, s_rho=array).  Otherwise the child's
    depths default to the parent's Cs_r/s_rho (i.e. identity in s).
    """
    pg = _load_grid(parent_grid_fn)
    cg = _load_grid(child_grid_fn)
    B  = _child_boundary_points(cg)

    with nc.Dataset(history_fn) as p:
        ot   = np.asarray(p.variables["ocean_time"][:], dtype=float)
        ot_units = p.variables["ocean_time"].units
        # parent (source) vertical params
        Vt_p   = int(np.asarray(p.variables["Vtransform"][...]))
        hc_p   = float(np.asarray(p.variables["hc"][...]))
        Cs_r_p = np.asarray(p.variables["Cs_r"][:])
        s_r_p  = np.asarray(p.variables["s_rho"][:])
        N_p    = len(s_r_p)
        # child (target) vertical params: override if provided
        if child_vparams is not None:
            Vt_c   = int(child_vparams["Vt"])
            hc_c   = float(child_vparams["hc"])
            Cs_r_c = np.asarray(child_vparams["Cs_r"])
            s_r_c  = np.asarray(child_vparams["s_rho"])
        else:
            Vt_c, hc_c, Cs_r_c, s_r_c = Vt_p, hc_p, Cs_r_p, s_r_p
        N    = len(s_r_c)
        # legacy locals still used below
        Vt, hc, Cs_r, s_r = Vt_c, hc_c, Cs_r_c, s_r_c

        # build one interpolator per parent grid (rho/u/v)
        rho_ip = _CurviInterp(pg["lon_rho"], pg["lat_rho"], pg["mask_rho"])
        u_ip   = _CurviInterp(pg["lon_u"],   pg["lat_u"],
                              0.5 * (pg["mask_rho"][:, :-1] + pg["mask_rho"][:, 1:]) >= 0.5)
        v_ip   = _CurviInterp(pg["lon_v"],   pg["lat_v"],
                              0.5 * (pg["mask_rho"][:-1, :] + pg["mask_rho"][1:, :]) >= 0.5)

        n_t = len(ot)
        # allocate output arrays per boundary
        out = {b: {} for b in BOUNDARIES}
        for b in BOUNDARIES:
            n_r = len(B[b]["rho"]["lon"])
            n_u = len(B[b]["u"]["lon"])
            n_v = len(B[b]["v"]["lon"])
            out[b]["zeta"] = np.zeros((n_t, n_r))
            out[b]["ubar"] = np.zeros((n_t, n_u))
            out[b]["vbar"] = np.zeros((n_t, n_v))
            out[b]["u"]    = np.zeros((n_t, N, n_u))
            out[b]["v"]    = np.zeros((n_t, N, n_v))
            out[b]["temp"] = np.zeros((n_t, N, n_r))
            out[b]["salt"] = np.zeros((n_t, N, n_r))

        # per-boundary XY arrays
        xy_rho = {b: np.column_stack([B[b]["rho"]["lon"], B[b]["rho"]["lat"]]) for b in BOUNDARIES}
        xy_u   = {b: np.column_stack([B[b]["u"]["lon"],   B[b]["u"]["lat"]])   for b in BOUNDARIES}
        xy_v   = {b: np.column_stack([B[b]["v"]["lon"],   B[b]["v"]["lat"]])   for b in BOUNDARIES}
        # child depths at boundary rho/u/v (zeta filled in per-time)
        h_r    = {b: B[b]["rho"]["h"] for b in BOUNDARIES}
        h_u    = {b: B[b]["u"]["h"]   for b in BOUNDARIES}
        h_v    = {b: B[b]["v"]["h"]   for b in BOUNDARIES}
        ang_r  = {b: B[b]["rho"]["angle"] for b in BOUNDARIES}
        ang_u  = {b: B[b]["u"]["angle"]   for b in BOUNDARIES}
        ang_v  = {b: B[b]["v"]["angle"]   for b in BOUNDARIES}
        # parent-h interpolated to child boundary rho/u/v points (for z_parent_at_child)
        hp_at_r = {b: rho_ip.interp(pg["h"], xy_rho[b])                for b in BOUNDARIES}
        hp_at_u = {b: u_ip.interp(pg["h_u"], xy_u[b])                  for b in BOUNDARIES}
        hp_at_v = {b: v_ip.interp(pg["h_v"], xy_v[b])                  for b in BOUNDARIES}

        # Fill sentinel: ROMS his files write ~1e37 on land / unused points
        # (u/v are undefined at u/v-points adjacent to two land rho cells,
        # etc.).  Replace those with 0 so the subsequent u->rho, v->rho
        # averaging doesn't poison water rho cells adjacent to land.  Land
        # values used to be nn-filled later inside _CurviInterp, but that
        # only fixes purely-land rho cells; the "half land" u/v averages
        # need explicit sanitizing first.
        _FILL_THRESH = 1e30
        def _san(arr):
            return np.where(np.abs(arr) > _FILL_THRESH, 0.0, arr)

        for k in range(n_t):
            zeta_p = _san(np.asarray(p.variables["zeta"][k, :, :]))
            ubar_p = _san(np.asarray(p.variables["ubar"][k, :, :]))
            vbar_p = _san(np.asarray(p.variables["vbar"][k, :, :]))
            u_p    = _san(np.asarray(p.variables["u"][k, :, :, :]))   # (N, eta_u, xi_u)
            v_p    = _san(np.asarray(p.variables["v"][k, :, :, :]))   # (N, eta_v, xi_v)
            temp_p = _san(np.asarray(p.variables["temp"][k, :, :, :]))
            salt_p = _san(np.asarray(p.variables["salt"][k, :, :, :]))

            # rotate parent barotropic (u,v) to (E,N) on parent rho grid
            ubar_rho = np.zeros_like(zeta_p)
            vbar_rho = np.zeros_like(zeta_p)
            ubar_rho[:, 1:-1] = 0.5 * (ubar_p[:, :-1] + ubar_p[:, 1:])
            vbar_rho[1:-1, :] = 0.5 * (vbar_p[:-1, :] + vbar_p[1:, :])
            cosA = np.cos(pg["angle"])
            sinA = np.sin(pg["angle"])
            uE_bar = ubar_rho * cosA - vbar_rho * sinA
            vN_bar = ubar_rho * sinA + vbar_rho * cosA

            for b in BOUNDARIES:
                # zeta at child rho / u / v strip points
                zc   = rho_ip.interp(zeta_p, xy_rho[b])
                zc_u = rho_ip.interp(zeta_p, xy_u[b])
                zc_v = rho_ip.interp(zeta_p, xy_v[b])
                out[b]["zeta"][k, :] = zc

                # child depths use child vparams; parent depths (at the same
                # child (lon,lat)) use parent vparams — they can differ
                z_c_r = _z_rho(h_r[b],    zc,   Vt_c, hc_c, Cs_r_c, s_r_c)
                z_p_r = _z_rho(hp_at_r[b], zc,  Vt_p, hc_p, Cs_r_p, s_r_p)
                z_c_u = _z_rho(h_u[b],    zc_u, Vt_c, hc_c, Cs_r_c, s_r_c)
                z_p_u = _z_rho(hp_at_u[b], zc_u, Vt_p, hc_p, Cs_r_p, s_r_p)
                z_c_v = _z_rho(h_v[b],    zc_v, Vt_c, hc_c, Cs_r_c, s_r_c)
                z_p_v = _z_rho(hp_at_v[b], zc_v, Vt_p, hc_p, Cs_r_p, s_r_p)

                # barotropic: interp rotated components to child u/v strips, then rotate to child axes
                uE_u = rho_ip.interp(uE_bar, xy_u[b])
                vN_u = rho_ip.interp(vN_bar, xy_u[b])
                uE_v = rho_ip.interp(uE_bar, xy_v[b])
                vN_v = rho_ip.interp(vN_bar, xy_v[b])
                out[b]["ubar"][k, :] =  uE_u * np.cos(ang_u[b]) + vN_u * np.sin(ang_u[b])
                out[b]["vbar"][k, :] = -uE_v * np.sin(ang_v[b]) + vN_v * np.cos(ang_v[b])

                # 3D fields: per-layer horizontal at PARENT s-levels, then
                # per-column vertical remap parent-z -> child-z.  Scratch
                # arrays sized to N_p (may differ from N_c).
                n_r_b = len(zc); n_u_b = len(zc_u); n_v_b = len(zc_v)
                T_scratch = np.zeros((N_p, n_r_b))
                S_scratch = np.zeros((N_p, n_r_b))
                U_scratch = np.zeros((N_p, n_u_b))
                V_scratch = np.zeros((N_p, n_v_b))
                for s in range(N_p):
                    u_r_s = np.zeros_like(zeta_p)
                    v_r_s = np.zeros_like(zeta_p)
                    u_r_s[:, 1:-1] = 0.5 * (u_p[s, :, :-1] + u_p[s, :, 1:])
                    v_r_s[1:-1, :] = 0.5 * (v_p[s, :-1, :] + v_p[s, 1:, :])
                    uE_s = u_r_s * cosA - v_r_s * sinA
                    vN_s = u_r_s * sinA + v_r_s * cosA

                    T_scratch[s, :] = rho_ip.interp(temp_p[s], xy_rho[b])
                    S_scratch[s, :] = rho_ip.interp(salt_p[s], xy_rho[b])

                    uE_u_s = rho_ip.interp(uE_s, xy_u[b])
                    vN_u_s = rho_ip.interp(vN_s, xy_u[b])
                    uE_v_s = rho_ip.interp(uE_s, xy_v[b])
                    vN_v_s = rho_ip.interp(vN_s, xy_v[b])
                    U_scratch[s, :] =  uE_u_s * np.cos(ang_u[b]) + vN_u_s * np.sin(ang_u[b])
                    V_scratch[s, :] = -uE_v_s * np.sin(ang_v[b]) + vN_v_s * np.cos(ang_v[b])

                # per-column vertical remap (parent depths z_p -> child depths z_c)
                for i in range(n_r_b):
                    out[b]["temp"][k, :, i] = _linear_interp_z(z_p_r[:, i], T_scratch[:, i], z_c_r[:, i])
                    out[b]["salt"][k, :, i] = _linear_interp_z(z_p_r[:, i], S_scratch[:, i], z_c_r[:, i])
                for i in range(n_u_b):
                    out[b]["u"][k, :, i] = _linear_interp_z(z_p_u[:, i], U_scratch[:, i], z_c_u[:, i])
                for i in range(n_v_b):
                    out[b]["v"][k, :, i] = _linear_interp_z(z_p_v[:, i], V_scratch[:, i], z_c_v[:, i])

        # --- post-mask by child boundary strip masks ----------------------
        # Zero out any values on child land cells so nothing weird sneaks
        # into slots ROMS won't dynamically use.
        for b in BOUNDARIES:
            mr = B[b]["rho"]["mask"]; mu = B[b]["u"]["mask"]; mv = B[b]["v"]["mask"]
            out[b]["zeta"] *= mr[None, :]
            out[b]["ubar"] *= mu[None, :]
            out[b]["vbar"] *= mv[None, :]
            out[b]["u"]    *= mu[None, None, :]
            out[b]["v"]    *= mv[None, None, :]
            out[b]["temp"] *= mr[None, None, :]
            out[b]["salt"] *= mr[None, None, :]

        # --- write BC file ------------------------------------------------
        _write_bc_file(bc_out_fn, ot, ot_units, Vt, hc, Cs_r, s_r,
                       cg, out, BOUNDARIES)


def _write_bc_file(out_fn: str, ot: np.ndarray, ot_units: str,
                   Vt: int, hc: float, Cs_r: np.ndarray, s_r: np.ndarray,
                   cg: dict, out: dict, boundaries=BOUNDARIES) -> None:
    """Write a single-time-dim BC file. All variables share ocean_time
    (parent his cadence -- typically hourly)."""
    Path(out_fn).parent.mkdir(parents=True, exist_ok=True)
    n_t = len(ot)
    N   = len(s_r)

    # ROMS reads BC time as *days since TIME_REF* (raw numeric — no CF units
    # parsing).  Source parent his uses "seconds since 1999-01-01"; convert
    # to plain days.
    if ot_units.startswith("seconds"):
        ot_days = ot / 86400.0
    elif ot_units.startswith("days"):
        ot_days = ot
    else:
        raise RuntimeError(f"unhandled parent time units: {ot_units!r}")
    ot_out_units = "days"

    with nc.Dataset(out_fn, "w", format="NETCDF4") as d:
        d.createDimension("s_rho", N)
        # child strip lengths
        n_r = cg["lat_rho"].shape[1]
        n_u = cg["lat_u"].shape[1]
        n_v = cg["lat_v"].shape[1]
        m_r = cg["lat_rho"].shape[0]
        m_u = cg["lat_u"].shape[0]
        m_v = cg["lat_v"].shape[0]
        d.createDimension("xr", n_r); d.createDimension("xu", n_u); d.createDimension("xv", n_v)
        d.createDimension("er", m_r); d.createDimension("eu", m_u); d.createDimension("ev", m_v)
        d.createDimension("time", n_t)

        for name in ("time", "temp_time", "salt_time", "v3d_time", "v2d_time", "zeta_time"):
            if name != "time":
                d.createDimension(name, n_t)
            v = d.createVariable(name, "f8", (name,) if name != "time" else ("time",))
            v.units = ot_out_units
            v[:] = ot_days
        oc = d.createVariable("ocean_time", "f8", ("time",))
        oc.units = ot_out_units
        oc[:] = ot_days

        for k, val in (("Vtransform", Vt), ("hc", hc)):
            v = d.createVariable(k, "f8", ())
            v[...] = val
        v = d.createVariable("Cs_r", "f8", ("s_rho",)); v[:] = Cs_r
        v = d.createVariable("s_rho", "f8", ("s_rho",)); v[:] = s_r

        # boundary vars: shape depends on which side
        dim_r = {"south": ("xr",), "north": ("xr",), "west": ("er",), "east": ("er",)}
        dim_u = {"south": ("xu",), "north": ("xu",), "west": ("eu",), "east": ("eu",)}
        dim_v = {"south": ("xv",), "north": ("xv",), "west": ("ev",), "east": ("ev",)}
        for bnd in boundaries:
            d.createVariable(f"zeta_{bnd}", "f8", ("zeta_time",) + dim_r[bnd])[:, :] = out[bnd]["zeta"]
            d.createVariable(f"ubar_{bnd}", "f8", ("v2d_time",) + dim_u[bnd])[:, :] = out[bnd]["ubar"]
            d.createVariable(f"vbar_{bnd}", "f8", ("v2d_time",) + dim_v[bnd])[:, :] = out[bnd]["vbar"]
            d.createVariable(f"u_{bnd}",    "f8", ("v3d_time", "s_rho") + dim_u[bnd])[:, :, :] = out[bnd]["u"]
            d.createVariable(f"v_{bnd}",    "f8", ("v3d_time", "s_rho") + dim_v[bnd])[:, :, :] = out[bnd]["v"]
            d.createVariable(f"temp_{bnd}", "f8", ("temp_time", "s_rho") + dim_r[bnd])[:, :, :] = out[bnd]["temp"]
            d.createVariable(f"salt_{bnd}", "f8", ("salt_time", "s_rho") + dim_r[bnd])[:, :, :] = out[bnd]["salt"]

    print(f"  wrote {out_fn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-grid", required=True)
    ap.add_argument("--child-grid",  required=True)
    ap.add_argument("--history",     required=True, help="parent his file (LV1_fwd_outer[01]_*.nc)")
    ap.add_argument("--out",         required=True, help="BC output file")
    ap.add_argument("--child-vparams-from", default=None,
                    help="path to any ROMS file (IC/rst/his) that carries the CHILD grid's "
                         "vertical params (Vtransform, hc, Cs_r, s_rho).  Only needed when "
                         "child vertical stretching differs from parent — e.g. LV3->LV4 (40->10 levels).")
    a = ap.parse_args()
    cv = read_vparams(a.child_vparams_from) if a.child_vparams_from else None
    history_file_to_BC_file(a.parent_grid, a.child_grid, a.history, a.out,
                            child_vparams=cv)


if __name__ == "__main__":
    main()
