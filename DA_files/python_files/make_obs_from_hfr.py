#!/usr/bin/env python3
"""
make_obs_from_hfr.py

Build ONE ROMS 4D-Var observation NetCDF per DA cycle, covering the 25 hourly
HFR surveys in a 1-day window (start 00Z .. next-day 00Z inclusive).

Pipeline per hourly HFR file (steps a..h below map to the user's spec):

  a. Subset to cells whose (lon, lat) lie inside LV1's rho polygon, extended
     outward by a 12 km buffer.
  b. 3x3 weighted stencil average with per-cell NaN-adjusted weights:
        w = [[.25, 1, .25],
             [ 1 , 1,  1 ],
             [.25, 1, .25]] / 6
     Missing neighbors drop from the sum and the denominator is re-normalised
     to the sum of the *present* weights.
  c. Helmholtz decompose (uE, vN); subtract the divergent part; keep the
     rotational (non-divergent) component.  Uses the surviving
     DA_setup_python_funs._helmholtz_div solver (Poisson w/ Dirichlet on
     open boundary + Neumann at coast).
  d. Drop obs within 2.5 km of any shoreline (any LV1 land cell — includes
     islands inside LV1).
  e. Keep every obs inside LV3 (highest resolution: weighted "most").  Outside
     LV3, thin to 1-in-2 walking from coast offshore (Keep, Reject, Keep, ...).
  f. Drop obs within 12 km of LV1 outer boundary.
  g. Apply the elliptic obs_error weighting from set_up_DA_simulation.sh:
        d_ell   = sqrt((dx_rot/Lx_km)^2 + (dy_rot/Ly_km)^2)
        obs_err = obs_error_var * exp(2 * d_ell)
     Center (32.5, -117.6), Lx=150 km, Ly=600 km, theta=25 deg CCW, with
     obs_error_var = 0.015 (m/s)^2 at the centre.  (The ellipse was retuned on
     2026-08-19 for the 2/1-6/1/2026 LV1_v2 rerun; this docstring previously
     still quoted the pre-retune centre (32.9, -118.2) with Lx=120, Ly=190.
     The constants in force are ELL_* below -- they are what the output file's
     ellipse_* attributes record.)
  h. Aggregate all 25 surveys into one ROMS obs NetCDF (schema per
     fourdvar_notebook.ipynb cell 5 -> write_roms_obs_file).

CLI:
    python make_obs_from_hfr.py 2026-03-07 [--outdir DIR]
"""
from __future__ import annotations
import argparse
import datetime as DT
import re
import sys
from pathlib import Path

import numpy as np
import netCDF4 as nc
from matplotlib.path import Path as MplPath
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# reuse the surviving Helmholtz solver
# ---------------------------------------------------------------------------
# DA_setup_python_funs.py is vendored alongside this file (PFM must not
# reach back into the roms_fourdvar tree at run time).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from DA_setup_python_funs import _helmholtz_div  # noqa: E402

# ---------------------------------------------------------------------------
# paths / grids
# ---------------------------------------------------------------------------
HFR_CACHE = Path("/dataSIO/DA_Simulations/input_files/hfr_cache")
LV1_GRID  = "/scratch/PFM_Simulations/Grids/GRID_SDTJRE_LV1_rx020_hmask.nc"
LV2_GRID  = "/scratch/PFM_Simulations/Grids/GRID_SDTJRE_LV2_rx020.nc"
LV3_GRID  = "/scratch/PFM_Simulations/Grids/GRID_SDTJRE_LV3_rx020.nc"

ROMS_TREF = DT.datetime(1999, 1, 1, 0, 0, 0)

# HFR ~6 km grid; DA window is 24 h at 1 h -> 25 surveys inclusive.
DA_WINDOW_HR = 24
STEP_HR      = 1

# ---------------------------------------------------------------------------
# knobs
# ---------------------------------------------------------------------------
BOUNDARY_BUFFER_INCLUDE_KM = 12.0   # step a: outward buffer around LV1 to include
BOUNDARY_BUFFER_EXCLUDE_KM = 12.0   # step f: inward buffer to exclude
LAND_BUFFER_KM             = 2.5    # step d
THIN_OUTSIDE_LV3           = 2      # step e: keep every 2nd (50%)
OBS_ERROR_VAR              = 0.015  # baseline (m/s)^2 at ellipse center
# ellipse (step g) — retuned 2026-08-19 for the 2/1-6/1/2026 LV1 DA rerun.
# Tuning was done interactively in ipynb_files/make_HF_current_plot.ipynb
# (Ellipse-weighting playground cell); see the log10(weight) map to reproduce.
ELL_LON   = -117.6
ELL_LAT   =   32.5
ELL_LX_KM =  150.0
ELL_LY_KM =  600.0
ELL_THETA_DEG = 25.0                # CCW from east

# ROMS obs_type flag: 1=zeta 2=ubar 3=vbar 4=u 5=v 6=temp 7=salt
OBS_TYPE_U = 4
OBS_TYPE_V = 5


# ---------------------------------------------------------------------------
# geodesic + polygon helpers
# ---------------------------------------------------------------------------
KM_PER_DEG_LAT = 111.32


def km_per_deg_lon(lat_deg: float) -> float:
    return KM_PER_DEG_LAT * np.cos(np.deg2rad(lat_deg))


def _grid_boundary_polygon(lat_rho: np.ndarray, lon_rho: np.ndarray) -> np.ndarray:
    """CCW walk around the outer rho border of a curvilinear grid.
    Returns (M, 2) array of (lon, lat) points."""
    ring = np.concatenate([
        np.column_stack([lon_rho[0, :],      lat_rho[0, :]]),        # south
        np.column_stack([lon_rho[1:-1, -1],  lat_rho[1:-1, -1]]),    # east
        np.column_stack([lon_rho[-1, ::-1],  lat_rho[-1, ::-1]]),    # north (reversed)
        np.column_stack([lon_rho[-2:0:-1, 0],lat_rho[-2:0:-1, 0]]),  # west (reversed)
    ])
    return ring


def _dist_to_polygon_km(lon: np.ndarray, lat: np.ndarray,
                        poly_lon: np.ndarray, poly_lat: np.ndarray) -> np.ndarray:
    """KDTree-nearest distance in km from (lon, lat) points to a polygon's
    vertices.  Good enough at ~1-2 km resolution — the vertex spacing of the
    LV1 boundary is ~2 km so a nearest-vertex approximation is within grid pitch."""
    lat_ref = float(np.mean(poly_lat))
    kx = km_per_deg_lon(lat_ref)
    ky = KM_PER_DEG_LAT
    poly_xy = np.column_stack([poly_lon * kx, poly_lat * ky])
    obs_xy  = np.column_stack([lon      * kx, lat      * ky])
    tree = cKDTree(poly_xy)
    dist, _ = tree.query(obs_xy)
    return dist


def _dist_to_points_km(lon: np.ndarray, lat: np.ndarray,
                       tgt_lon: np.ndarray, tgt_lat: np.ndarray) -> np.ndarray:
    """Same idea but for scattered target points (e.g. land cells)."""
    lat_ref = float(np.mean(tgt_lat)) if tgt_lat.size else float(np.mean(lat))
    kx = km_per_deg_lon(lat_ref)
    ky = KM_PER_DEG_LAT
    tgt_xy = np.column_stack([tgt_lon * kx, tgt_lat * ky])
    obs_xy = np.column_stack([lon     * kx, lat     * ky])
    tree = cKDTree(tgt_xy)
    dist, _ = tree.query(obs_xy)
    return dist


def _inside_polygon(lon: np.ndarray, lat: np.ndarray,
                    poly_lon: np.ndarray, poly_lat: np.ndarray) -> np.ndarray:
    path = MplPath(np.column_stack([poly_lon, poly_lat]))
    return path.contains_points(np.column_stack([lon, lat]))


# ---------------------------------------------------------------------------
# HFR cache lookup + loader
# ---------------------------------------------------------------------------
_HFR_STAMP_RE = re.compile(r"_s(\d{15})_")


def hfr_file_for(t: DT.datetime) -> Path | None:
    """Find the cached HFR file whose start-stamp matches target hour t.
    Returns None if not in cache."""
    stamp = t.strftime("%Y%m%d%H%M%S") + "0"
    matches = sorted(HFR_CACHE.glob(f"*_s{stamp}_*.nc"))
    return matches[0] if matches else None


def load_hfr(t: DT.datetime, lon_lo: float, lon_hi: float, lat_lo: float, lat_hi: float):
    """Return (lat_1d, lon_1d, uE_2d, vN_2d) subset to a lon/lat bounding box.
    Returns None if the file is not cached (permanent NCEI catalog gap)."""
    fp = hfr_file_for(t)
    if fp is None:
        return None
    with nc.Dataset(fp) as d:
        lat = np.asarray(d.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(d.variables["lon"][:], dtype=np.float64)
        # subset indices for LV1 + buffer
        ilo = int(np.searchsorted(lon, lon_lo, side="left"))
        ihi = int(np.searchsorted(lon, lon_hi, side="right"))
        jlo = int(np.searchsorted(lat, lat_lo, side="left"))
        jhi = int(np.searchsorted(lat, lat_hi, side="right"))
        # HFR is packed int16 with scale_factor=0.01 and _FillValue=-32767.
        # netCDF4 returns a masked array — use np.ma.filled to convert to NaN.
        u = np.ma.filled(d.variables["u"][0, jlo:jhi, ilo:ihi], np.nan).astype(np.float64)
        v = np.ma.filled(d.variables["v"][0, jlo:jhi, ilo:ihi], np.nan).astype(np.float64)
    return lat[jlo:jhi], lon[ilo:ihi], u, v


# ---------------------------------------------------------------------------
# step b: 3x3 weighted stencil with NaN adjustment
# ---------------------------------------------------------------------------
_STENCIL33 = np.array([[0.25, 1.0, 0.25],
                       [1.00, 1.0, 1.00],
                       [0.25, 1.0, 0.25]], dtype=np.float64)


def _stencil33_avg(arr: np.ndarray) -> np.ndarray:
    """3x3 weighted stencil per user spec, NaN-adjusted denominator."""
    pad_arr = np.pad(arr, 1, mode="constant", constant_values=np.nan)
    ny, nx  = arr.shape
    num = np.zeros_like(arr)
    den = np.zeros_like(arr)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            w = _STENCIL33[di + 1, dj + 1]
            slab = pad_arr[1 + di:1 + di + ny, 1 + dj:1 + dj + nx]
            good = np.isfinite(slab)
            num += np.where(good, w * slab, 0.0)
            den += np.where(good, w,        0.0)
    with np.errstate(invalid="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return out


# ---------------------------------------------------------------------------
# ellipse (step g)
# ---------------------------------------------------------------------------
def ellipse_scaled_error(obs_lon: np.ndarray, obs_lat: np.ndarray) -> np.ndarray:
    """obs_error inflated by exp(2 * d_ellipse). Sec 5b of set_up_DA_simulation.sh."""
    lat_to_km = KM_PER_DEG_LAT
    lon_to_km = km_per_deg_lon(ELL_LAT)
    th = np.deg2rad(ELL_THETA_DEG)
    c, s = np.cos(th), np.sin(th)
    dx_km =  (obs_lon - ELL_LON) * lon_to_km
    dy_km =  (obs_lat - ELL_LAT) * lat_to_km
    dx_r  =  c * dx_km + s * dy_km
    dy_r  = -s * dx_km + c * dy_km
    d_ell = np.sqrt((dx_r / ELL_LX_KM) ** 2 + (dy_r / ELL_LY_KM) ** 2)
    return OBS_ERROR_VAR * np.exp(2.0 * d_ell)


# ---------------------------------------------------------------------------
# lon/lat -> LV1 fractional (Xgrid, Ygrid)
# ---------------------------------------------------------------------------
def build_grid_index_lookup(lon_rho: np.ndarray, lat_rho: np.ndarray):
    """KDTree-based nearest-neighbor fractional index look-up on a curvilinear
    grid (sufficient for HFR obs at ~6 km on a 2 km grid)."""
    eta_size, xi_size = lon_rho.shape
    pts = np.column_stack([lon_rho.ravel(), lat_rho.ravel()])
    tree = cKDTree(pts)
    def _lookup(lons, lats):
        d, idx = tree.query(np.column_stack([lons, lats]))
        return (idx % xi_size).astype(float), (idx // xi_size).astype(float)
    return _lookup


# ---------------------------------------------------------------------------
# rotate earth (uE, vN) -> LV1 grid frame (u_xi, v_eta) using LV1 'angle'
# ---------------------------------------------------------------------------
def rotate_earth_to_grid(uE: np.ndarray, vN: np.ndarray, angle_rad: np.ndarray):
    c = np.cos(angle_rad); s = np.sin(angle_rad)
    u_xi  = uE * c + vN * s
    v_eta = -uE * s + vN * c
    return u_xi, v_eta


# ---------------------------------------------------------------------------
# core per-hour processor
# ---------------------------------------------------------------------------
class GridBundle:
    """LV1/LV2/LV3 polygon/mask/land info + HFR grid-cached derived arrays."""
    def __init__(self):
        with nc.Dataset(LV1_GRID) as g:
            self.lv1_lat  = np.asarray(g.variables["lat_rho"][:])
            self.lv1_lon  = np.asarray(g.variables["lon_rho"][:])
            self.lv1_mask = np.asarray(g.variables["mask_rho"][:], dtype=int)  # 1 water / 0 land
            self.lv1_angle= np.asarray(g.variables["angle"][:])
        with nc.Dataset(LV2_GRID) as g:
            self.lv2_lat  = np.asarray(g.variables["lat_rho"][:])
            self.lv2_lon  = np.asarray(g.variables["lon_rho"][:])
        with nc.Dataset(LV3_GRID) as g:
            self.lv3_lat  = np.asarray(g.variables["lat_rho"][:])
            self.lv3_lon  = np.asarray(g.variables["lon_rho"][:])

        # LV1 outer boundary polygon (for buffer distances)
        self.lv1_ring = _grid_boundary_polygon(self.lv1_lat, self.lv1_lon)
        self.lv2_ring = _grid_boundary_polygon(self.lv2_lat, self.lv2_lon)
        self.lv3_ring = _grid_boundary_polygon(self.lv3_lat, self.lv3_lon)

        # LV1 land points (mask==0) in lon/lat
        land_j, land_i = np.where(self.lv1_mask == 0)
        self.land_lon = self.lv1_lon[land_j, land_i]
        self.land_lat = self.lv1_lat[land_j, land_i]

        # angle interpolator for HFR grid (nearest neighbour)
        self.angle_interp = NearestNDInterpolator(
            np.column_stack([self.lv1_lon.ravel(), self.lv1_lat.ravel()]),
            self.lv1_angle.ravel())

        # (grid index lookup — built once)
        self.lookup_LV1 = build_grid_index_lookup(self.lv1_lon, self.lv1_lat)

    def lon_bbox(self, buffer_deg: float) -> tuple[float, float]:
        return self.lv1_lon.min() - buffer_deg, self.lv1_lon.max() + buffer_deg

    def lat_bbox(self, buffer_deg: float) -> tuple[float, float]:
        return self.lv1_lat.min() - buffer_deg, self.lv1_lat.max() + buffer_deg


def process_one_hour(t: DT.datetime, GB: GridBundle):
    """Return arrays for one hourly survey (or None if HFR file missing).
    Yields per-obs (lon, lat, u_xi, v_eta, obs_error) after all filtering."""
    # bounding-box subset with a generous buffer (~0.3 deg > 12 km outward)
    lon_lo, lon_hi = GB.lon_bbox(0.5)
    lat_lo, lat_hi = GB.lat_bbox(0.5)
    hfr = load_hfr(t, lon_lo, lon_hi, lat_lo, lat_hi)
    if hfr is None:
        return None
    hfr_lat_1d, hfr_lon_1d, uE, vN = hfr
    if uE.size == 0 or vN.size == 0:
        return None
    LAT2, LON2 = np.meshgrid(hfr_lat_1d, hfr_lon_1d, indexing="ij")   # (ny, nx)

    # ---- rotate to LV1 grid frame (using LV1 angle nearest to each HFR cell) ----
    ang2d = GB.angle_interp(np.column_stack([LON2.ravel(), LAT2.ravel()])
                             ).reshape(LAT2.shape)
    u_gr, v_gr = rotate_earth_to_grid(uE, vN, ang2d)

    # ---- step (a): subset to LV1 + 12 km outward buffer ----------------------
    # inside LV1 polygon OR within 12 km of LV1 boundary
    inside_lv1 = _inside_polygon(LON2.ravel(), LAT2.ravel(),
                                 GB.lv1_ring[:, 0], GB.lv1_ring[:, 1]
                                 ).reshape(LAT2.shape)
    d2_lv1_bnd = _dist_to_polygon_km(LON2.ravel(), LAT2.ravel(),
                                     GB.lv1_ring[:, 0], GB.lv1_ring[:, 1]
                                     ).reshape(LAT2.shape)
    keep_a = inside_lv1 | (d2_lv1_bnd <= BOUNDARY_BUFFER_INCLUDE_KM)
    # apply as NaN mask (retain shape for stencil + helmholtz)
    u_gr = np.where(keep_a, u_gr, np.nan)
    v_gr = np.where(keep_a, v_gr, np.nan)
    if not np.isfinite(u_gr).any():
        return None

    # ---- step (b): 3x3 weighted stencil, NaN-adjusted ------------------------
    u_avg = _stencil33_avg(u_gr)
    v_avg = _stencil33_avg(v_gr)

    # ---- step (c): Helmholtz, keep rotational part ---------------------------
    # Poisson solve mask = cells that have valid u AND v after step b.
    solve_mask = np.isfinite(u_avg) & np.isfinite(v_avg)
    if not solve_mask.any():
        return None
    # land mask on HFR grid (for Neumann BC at coast/islands)
    land_interp = NearestNDInterpolator(
        np.column_stack([GB.lv1_lon.ravel(), GB.lv1_lat.ravel()]),
        GB.lv1_mask.ravel())
    land_hfr = land_interp(np.column_stack([LON2.ravel(), LAT2.ravel()])
                            ).reshape(LAT2.shape) < 0.5
    land_not_in_mask = land_hfr & ~solve_mask
    # regular grid: dx and dy in metres from lat/lon 1D
    lat_mean = float(np.mean(hfr_lat_1d))
    dx = float(hfr_lon_1d[1] - hfr_lon_1d[0]) * km_per_deg_lon(lat_mean) * 1000.0
    dy = float(hfr_lat_1d[1] - hfr_lat_1d[0]) * KM_PER_DEG_LAT       * 1000.0
    u_div, v_div = _helmholtz_div(u_avg, v_avg, solve_mask, land_not_in_mask, dx, dy)
    u_rot = u_avg - u_div
    v_rot = v_avg - v_div

    # ---- collapse to scattered arrays ---------------------------------------
    good = np.isfinite(u_rot) & np.isfinite(v_rot)
    LON = LON2[good]; LAT = LAT2[good]
    U   = u_rot[good]; V = v_rot[good]

    # ---- step (d): remove within 2.5 km of any LV1 land cell -----------------
    d_land = _dist_to_points_km(LON, LAT, GB.land_lon, GB.land_lat)
    keep = d_land >= LAND_BUFFER_KM
    LON, LAT, U, V = LON[keep], LAT[keep], U[keep], V[keep]
    if LON.size == 0:
        return None

    # ---- step (e): keep all in LV3, thin every-other outside LV3 ------------
    in_lv3 = _inside_polygon(LON, LAT, GB.lv3_ring[:, 0], GB.lv3_ring[:, 1])
    # outside-LV3 subset: thin walking from coast offshore (nearest land ascending)
    out3_idx = np.where(~in_lv3)[0]
    if out3_idx.size:
        d_land_out = _dist_to_points_km(LON[out3_idx], LAT[out3_idx],
                                         GB.land_lon, GB.land_lat)
        # sort out3 by distance-to-land ascending; keep every 2nd (positions 0,2,4,...)
        order = np.argsort(d_land_out)
        keep_pos = order[::THIN_OUTSIDE_LV3]
        thin_keep = np.zeros(LON.size, dtype=bool)
        thin_keep[in_lv3] = True
        thin_keep[out3_idx[keep_pos]] = True
    else:
        thin_keep = np.ones(LON.size, dtype=bool)
    LON, LAT, U, V = LON[thin_keep], LAT[thin_keep], U[thin_keep], V[thin_keep]
    if LON.size == 0:
        return None

    # ---- step (f): remove within 12 km INWARD of LV1 outer boundary ---------
    d_bnd = _dist_to_polygon_km(LON, LAT, GB.lv1_ring[:, 0], GB.lv1_ring[:, 1])
    inside_lv1_flag = _inside_polygon(LON, LAT, GB.lv1_ring[:, 0], GB.lv1_ring[:, 1])
    keep = inside_lv1_flag & (d_bnd > BOUNDARY_BUFFER_EXCLUDE_KM)
    LON, LAT, U, V = LON[keep], LAT[keep], U[keep], V[keep]
    if LON.size == 0:
        return None

    # ---- step (g): ellipse-inflated per-obs error ---------------------------
    err = ellipse_scaled_error(LON, LAT)   # (M,)

    return dict(lon=LON, lat=LAT, u=U, v=V, err=err)


# ---------------------------------------------------------------------------
# NetCDF writer (schema per fourdvar_notebook.ipynb cell 5)
# ---------------------------------------------------------------------------
def write_roms_obs(out_path: Path,
                   survey_times: list[float],
                   nobs_per_survey: list[int],
                   obs_time: np.ndarray,
                   obs_lon: np.ndarray,
                   obs_lat: np.ndarray,
                   obs_xgrid: np.ndarray,
                   obs_ygrid: np.ndarray,
                   obs_value: np.ndarray,
                   obs_type: np.ndarray,
                   obs_error: np.ndarray,
                   obs_provenance: np.ndarray):
    n_total   = obs_value.size
    n_surveys = len(survey_times)
    with nc.Dataset(out_path, "w", format="NETCDF4") as d:
        d.createDimension("datum",          n_total)
        d.createDimension("survey",         n_surveys)
        d.createDimension("state_variable", 7)

        sph = d.createVariable("spherical", "i4")
        sph.long_name = "grid type logical switch"
        sph.flag_values = "0, 1"; sph.flag_meanings = "Cartesian spherical"
        sph[:] = 1

        v = d.createVariable("Nobs", "i4", ("survey",))
        v.long_name = "number of observations with the same survey time"
        v[:] = np.asarray(nobs_per_survey, dtype=np.int32)

        v = d.createVariable("survey_time", "f8", ("survey",))
        v.long_name = "survey time"; v.units = "day"
        v[:] = np.asarray(survey_times, dtype=np.float64)

        v = d.createVariable("obs_type", "i4", ("datum",))
        v.long_name = "model state variable associated with observation"
        v.flag_values = "1, 2, 3, 4, 5, 6, 7"
        v.flag_meanings = "zeta ubar vbar u v temperature salinity"
        v[:] = obs_type.astype(np.int32)

        v = d.createVariable("obs_provenance", "i4", ("datum",))
        v.long_name = "observation origin"
        v[:] = obs_provenance.astype(np.int32)

        v = d.createVariable("obs_time", "f8", ("datum",))
        v.long_name = "time of observation"; v.units = "day"
        v[:] = obs_time

        v = d.createVariable("obs_depth", "f8", ("datum",))
        v.long_name = "depth of observation"; v.units = "meter"
        v.negative = "downwards"
        v[:] = np.zeros(n_total, dtype=np.float64)

        v = d.createVariable("obs_Xgrid", "f8", ("datum",))
        v.long_name = "x-grid observation location"
        v[:] = obs_xgrid

        v = d.createVariable("obs_Ygrid", "f8", ("datum",))
        v.long_name = "y-grid observation location"
        v[:] = obs_ygrid

        v = d.createVariable("obs_Zgrid", "f8", ("datum",))
        v.long_name = "z-grid observation location"
        v[:] = np.zeros(n_total, dtype=np.float64)

        v = d.createVariable("obs_error", "f8", ("datum",))
        v.long_name = "observation error covariance"
        v.units = "squared state variable units"
        v[:] = obs_error

        v = d.createVariable("obs_value", "f8", ("datum",))
        v.long_name = "observation value"; v.units = "state variable units"
        v[:] = obs_value

        v = d.createVariable("obs_lon", "f8", ("datum",))
        v.long_name = "longitude of observation"; v.units = "degrees_east"
        v[:] = obs_lon

        v = d.createVariable("obs_lat", "f8", ("datum",))
        v.long_name = "latitude of observation"; v.units = "degrees_north"
        v[:] = obs_lat

        d.type = "ROMS Observations"
        d.title = "HF Radar Observations for LV1 4D-Var (rotational, ellipse-weighted)"
        d.Conventions = "CF-1.4"
        d.state_variables = (
            "1: free-surface (m), "
            "2: vertically integrated u-momentum (m/s), "
            "3: vertically integrated v-momentum (m/s), "
            "4: u-momentum (m/s), "
            "5: v-momentum (m/s), "
            "6: potential temperature (Celsius), "
            "7: salinity (nondimensional)")
        d.setncattr("ellipse_center", f"({ELL_LON}, {ELL_LAT})")
        d.setncattr("ellipse_Lx_Ly_theta",
                    f"Lx={ELL_LX_KM} km, Ly={ELL_LY_KM} km, theta={ELL_THETA_DEG} deg CCW")
        d.setncattr("baseline_obs_error_var", OBS_ERROR_VAR)


# ---------------------------------------------------------------------------
# top-level driver
# ---------------------------------------------------------------------------
def roms_days(t: DT.datetime) -> float:
    return (t - ROMS_TREF).total_seconds() / 86400.0


def build_obs_for_start(start_date: str, out_dir: Path):
    t0 = DT.datetime.strptime(start_date, "%Y-%m-%d")
    times = [t0 + DT.timedelta(hours=h) for h in range(0, DA_WINDOW_HR + 1, STEP_HR)]
    print(f"  {start_date}: {len(times)} hourly HFR surveys")

    GB = GridBundle()

    survey_times: list[float] = []
    nobs_per_survey: list[int] = []
    all_time: list[float] = []
    all_lon:  list[np.ndarray] = []
    all_lat:  list[np.ndarray] = []
    all_xg:   list[np.ndarray] = []
    all_yg:   list[np.ndarray] = []
    all_val:  list[np.ndarray] = []
    all_type: list[np.ndarray] = []
    all_err:  list[np.ndarray] = []
    all_prov: list[np.ndarray] = []

    n_missing = 0
    for t in times:
        got = process_one_hour(t, GB)
        if got is None:
            n_missing += 1
            continue
        # fractional grid indices for the retained obs
        Xg, Yg = GB.lookup_LV1(got["lon"], got["lat"])
        M = got["lon"].size
        tdays = roms_days(t)

        survey_times.append(tdays)
        nobs_per_survey.append(2 * M)      # u + v per point

        # u block
        all_time.append(np.full(M, tdays));           all_lon.append(got["lon"])
        all_lat.append(got["lat"]);                   all_xg.append(Xg); all_yg.append(Yg)
        all_val.append(got["u"]);                     all_type.append(np.full(M, OBS_TYPE_U, dtype=np.int32))
        all_err.append(got["err"]);                   all_prov.append(np.full(M, 1, dtype=np.int32))
        # v block (same lon/lat, same grid indices, obs_type=5)
        all_time.append(np.full(M, tdays));           all_lon.append(got["lon"])
        all_lat.append(got["lat"]);                   all_xg.append(Xg); all_yg.append(Yg)
        all_val.append(got["v"]);                     all_type.append(np.full(M, OBS_TYPE_V, dtype=np.int32))
        all_err.append(got["err"]);                   all_prov.append(np.full(M, 1, dtype=np.int32))

        if len(survey_times) <= 3 or len(survey_times) % 6 == 0:
            print(f"    {t:%Y-%m-%d %H:%MZ}  ->  {M} paired (u,v) obs  "
                  f"({2*M} datums; err ranges: {got['err'].min():.4f}..{got['err'].max():.2e})")

    if not survey_times:
        raise RuntimeError(f"no HFR obs survived filtering for {start_date}")

    obs_time  = np.concatenate(all_time)
    obs_lon   = np.concatenate(all_lon)
    obs_lat   = np.concatenate(all_lat)
    obs_xgrid = np.concatenate(all_xg)
    obs_ygrid = np.concatenate(all_yg)
    obs_value = np.concatenate(all_val)
    obs_type  = np.concatenate(all_type)
    obs_error = np.concatenate(all_err)
    obs_prov  = np.concatenate(all_prov)

    out_path = out_dir / f"obs_HF_LV1_{t0:%Y%m%d}.nc"
    write_roms_obs(out_path, survey_times, nobs_per_survey,
                   obs_time, obs_lon, obs_lat, obs_xgrid, obs_ygrid,
                   obs_value, obs_type, obs_error, obs_prov)
    n_kept_pts = sum(nobs_per_survey) // 2
    print(f"  {start_date}: wrote {out_path.name}  "
          f"{len(survey_times)} surveys, {n_kept_pts} (u,v) pairs, "
          f"{sum(nobs_per_survey)} datums total, "
          f"{n_missing} surveys skipped (missing HFR); size="
          f"{out_path.stat().st_size/1_048_576:.2f} MB")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start_date", help="YYYY-MM-DD (start 00Z of 1-day DA window)")
    ap.add_argument("--outdir",
                    default="/dataSIO/DA_Simulations/input_files/obs_files",
                    help="output directory (default: %(default)s)")
    args = ap.parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_obs_for_start(args.start_date, out_dir)


if __name__ == "__main__":
    main()
