"""DA_setup_python_funs.py
==========================
Single-file Python helpers used by set_up_DA_simulation.sh to build the ROMS
4D-Var auxiliary files: std files (IC / BC / forcing) and the obs file.

Each "build_*" function is idempotent in spirit (writes a `.config.json`
sidecar; the shell orchestrator checks sidecar params and rebuilds only when
they change), but the functions themselves are dumb and always (re)write
when called.

Functions imported here were extracted from
  ipynb_files/fourdvar_notebook.ipynb
  ipynb_files/hf_radar_load_qc.ipynb
and unified with consistent argument names.
"""

import json
import os
import re
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import netCDF4 as nc
import numpy as np


# =============================================================================
# Std file builders
# =============================================================================

def build_std_i(grid_file, n_levels, t_str,
                std_u, std_v, std_temp, std_salt, std_zeta,
                output_name):
    """Create roms_std_i.nc (IC error standard deviations).

    Std values are zeroed on land cells using grid masks. n_levels = N (s_rho).
    t_str is 'YYYYMMDDHH'; written into ocean_time as days since 1999-01-01.
    """
    from scipy.ndimage import distance_transform_edt

    with nc.Dataset(grid_file, 'r') as g:
        lm = g.dimensions['xi_rho'].size
        mm = g.dimensions['eta_rho'].size
        mask_rho = np.asarray(g.variables['mask_rho'][:])
        mask_u   = np.asarray(g.variables['mask_u'][:])
        mask_v   = np.asarray(g.variables['mask_v'][:])
        pm       = np.asarray(g.variables['pm'][:])

    dx_avg = 1.0 / pm.mean()
    dist_rho = distance_transform_edt(mask_rho) * dx_avg
    L0 = 10.0e3
    min_dist_water = dist_rho[mask_rho == 1].min()
    recommended_Hdecay = min(L0, min_dist_water / 2.0)
    print(f"  std_i build: min dist to land = {min_dist_water:.0f} m, "
          f"recommended HdecayI = {recommended_Hdecay:.0f} m")

    z2d = np.where(mask_rho == 1, std_zeta, 0.0)
    t3d = np.where(mask_rho[None, :, :] == 1, std_temp, 0.0)
    t3d = np.broadcast_to(t3d, (n_levels, mm, lm)).copy()
    s3d = np.where(mask_rho[None, :, :] == 1, std_salt, 0.0)
    s3d = np.broadcast_to(s3d, (n_levels, mm, lm)).copy()
    u3d = np.where(mask_u[None, :, :] == 1, std_u, 0.0)
    u3d = np.broadcast_to(u3d, (n_levels, mm, lm - 1)).copy()
    v3d = np.where(mask_v[None, :, :] == 1, std_v, 0.0)
    v3d = np.broadcast_to(v3d, (n_levels, mm - 1, lm)).copy()
    ub2d = np.where(mask_u == 1, std_u, 0.0)
    vb2d = np.where(mask_v == 1, std_v, 0.0)

    with nc.Dataset(output_name, 'w', format='NETCDF4') as ds:
        ds.createDimension('xi_rho',  lm)
        ds.createDimension('eta_rho', mm)
        ds.createDimension('xi_u',    lm - 1)
        ds.createDimension('eta_u',   mm)
        ds.createDimension('xi_v',    lm)
        ds.createDimension('eta_v',   mm - 1)
        ds.createDimension('s_rho',   n_levels)
        ds.createDimension('ocean_time', None)

        ot = ds.createVariable('ocean_time', 'f8', ('ocean_time',))
        ot.long_name = "days since 1999-01-01 00:00:00"
        ot.units = "days"
        t_dt = datetime.strptime(t_str, '%Y%m%d%H')
        ot[0] = (t_dt - datetime(1999, 1, 1)).total_seconds() / 86400.0

        for vname, arr, dims, units, lname in [
                ('zeta', z2d,  ('ocean_time', 'eta_rho', 'xi_rho'),         'meter',          'free-surface IC std'),
                ('temp', t3d,  ('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),'Celsius',        'temperature IC std'),
                ('salt', s3d,  ('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),'PSU',            'salinity IC std'),
                ('u',    u3d,  ('ocean_time', 's_rho', 'eta_u',   'xi_u'),  'meter second-1', '3D u-momentum IC std'),
                ('v',    v3d,  ('ocean_time', 's_rho', 'eta_v',   'xi_v'),  'meter second-1', '3D v-momentum IC std'),
                ('ubar', ub2d, ('ocean_time', 'eta_u',  'xi_u'),            'meter second-1', '2D u-momentum IC std'),
                ('vbar', vb2d, ('ocean_time', 'eta_v',  'xi_v'),            'meter second-1', '2D v-momentum IC std')]:
            var = ds.createVariable(vname, 'f8', dims)
            var.units = units
            var.long_name = lname
            var[(0,) + (slice(None),) * (len(dims) - 1)] = arr

    print(f"  wrote {output_name}")


def build_std_b(grid_file, n_levels,
                std_u, std_v, std_temp, std_salt, std_zeta,
                output_name):
    """Create roms_std_b.nc (BC error standard deviations).

    Packed CDL order: 2D obc (ocean_time, boundary, IorJ); 3D obc
    (ocean_time, boundary, s_rho, IorJ). Boundary slot order is W=0, S=1,
    E=2, N=3 — matches the s4dvar.in Lobc diagram (W=1,S=2,E=3,N=4).
    """
    with nc.Dataset(grid_file, 'r') as g:
        xi_rho  = g.dimensions['xi_rho'].size
        eta_rho = g.dimensions['eta_rho'].size
        mask_rho = np.asarray(g.variables['mask_rho'][:])
        mask_u   = np.asarray(g.variables['mask_u'][:])
        mask_v   = np.asarray(g.variables['mask_v'][:])

    IorJ  = max(xi_rho, eta_rho)
    edges = ['west', 'south', 'east', 'north']

    def edge_slice(a, e):
        return {'west':  a[:,  0], 'east':  a[:, -1],
                'south': a[0, :], 'north': a[-1, :]}[e]

    def pad(a):
        out = np.zeros(IorJ)
        out[:len(a)] = a
        return out

    def bry_mask(m2d):
        out = np.zeros((4, IorJ))
        for ib, e in enumerate(edges):
            out[ib, :] = pad((edge_slice(m2d, e) == 1).astype(float))
        return out

    bmask_rho = bry_mask(mask_rho)
    bmask_u   = bry_mask(mask_u)
    bmask_v   = bry_mask(mask_v)

    with nc.Dataset(output_name, 'w', format='NETCDF4') as ds:
        ds.createDimension('IorJ',     IorJ)
        ds.createDimension('s_rho',    n_levels)
        ds.createDimension('boundary', 4)
        ds.createDimension('ocean_time', None)
        ds.boundary_index = "West=1, South=2, East=3, North=4"

        ot = ds.createVariable('ocean_time', 'f8', ('ocean_time',))
        ot.units = "days since 1999-01-01 00:00:00"
        ot[0] = 0.0

        for vname, bmask, scalar, units, lname in [
                ('zeta_obc', bmask_rho, std_zeta, 'meter',          'free-surface OBC std'),
                ('ubar_obc', bmask_u,   std_u,    'meter second-1', '2D u OBC std'),
                ('vbar_obc', bmask_v,   std_v,    'meter second-1', '2D v OBC std')]:
            var = ds.createVariable(vname, 'f8', ('ocean_time', 'boundary', 'IorJ'))
            var.units = units; var.long_name = lname
            var[0, :, :] = bmask * scalar

        for vname, bmask, scalar, units, lname in [
                ('u_obc',    bmask_u,   std_u,    'meter second-1', '3D u OBC std'),
                ('v_obc',    bmask_v,   std_v,    'meter second-1', '3D v OBC std'),
                ('temp_obc', bmask_rho, std_temp, 'Celsius',        'temperature OBC std'),
                ('salt_obc', bmask_rho, std_salt, 'PSU',            'salinity OBC std')]:
            var = ds.createVariable(vname, 'f8', ('ocean_time', 'boundary', 's_rho', 'IorJ'))
            var.units = units; var.long_name = lname
            var[0, :, :, :] = np.broadcast_to((bmask * scalar)[:, None, :], (4, n_levels, IorJ))

    print(f"  wrote {output_name}")


def build_std_f(grid_file, std_sustr, std_svstr, std_shflux, std_swflux,
                output_name, std_ssflux=1.0e-8):
    """Create roms_std_f.nc (surface forcing error standard deviations)."""
    with nc.Dataset(grid_file, 'r') as g:
        xi_rho = g.dimensions['xi_rho'].size
        eta_rho = g.dimensions['eta_rho'].size
        xi_u  = g.dimensions['xi_u'].size
        eta_u = g.dimensions['eta_u'].size
        xi_v  = g.dimensions['xi_v'].size
        eta_v = g.dimensions['eta_v'].size

    with nc.Dataset(output_name, 'w', format='NETCDF4') as ds:
        ds.createDimension('xi_rho', xi_rho); ds.createDimension('eta_rho', eta_rho)
        ds.createDimension('xi_u',   xi_u);   ds.createDimension('eta_u',   eta_u)
        ds.createDimension('xi_v',   xi_v);   ds.createDimension('eta_v',   eta_v)
        ds.createDimension('ocean_time', None)

        ot = ds.createVariable('ocean_time', 'f8', ('ocean_time',))
        ot.units = "days since 1999-01-01 00:00:00"
        ot[0] = 0.0

        for vname, dims, val, units, lname in [
                ('sustr',  ('ocean_time','eta_u','xi_u'),     std_sustr,  'Newton meter-2',         'surface u-stress std'),
                ('svstr',  ('ocean_time','eta_v','xi_v'),     std_svstr,  'Newton meter-2',         'surface v-stress std'),
                ('shflux', ('ocean_time','eta_rho','xi_rho'), std_shflux, 'Watts meter-2',          'surface heat flux std'),
                ('swflux', ('ocean_time','eta_rho','xi_rho'), std_swflux, 'meter second-1',         'surface freshwater flux std'),
                ('ssflux', ('ocean_time','eta_rho','xi_rho'), std_ssflux, 'salt_units meter second-1', 'surface salt flux std')]:
            v = ds.createVariable(vname, 'f8', dims)
            v.units = units; v.long_name = lname
            v[0, :, :] = val

    print(f"  wrote {output_name}")


# =============================================================================
# Obs file builders
# =============================================================================

def _cross5_avg(arr):
    """5-point cross stencil average, NaN-skipping.

        [0 a 0;
         a a a;     where a = 1 if not-NaN, 0 if NaN, divided by sum(a)
         0 a 0]

    For each cell, the result is the mean of (center, up, down, left, right)
    over the non-NaN subset.  Returns NaN where all 5 are NaN.  Used as a
    spatial pre-filter for HFR obs so that subsequent thinning (default
    stride=3 -> 18 km) consumes every native ~6 km cell rather than
    discarding ~8 of 9.
    """
    pad = np.pad(arr, 1, mode='constant', constant_values=np.nan)
    stack = np.stack([pad[1:-1, 1:-1],   # center
                      pad[ :-2, 1:-1],   # up
                      pad[2:  , 1:-1],   # down
                      pad[1:-1,  :-2],   # left
                      pad[1:-1, 2:  ]],  # right
                     axis=0)
    valid = ~np.isnan(stack)
    count = valid.sum(axis=0)
    total = np.where(valid, stack, 0.0).sum(axis=0)
    with np.errstate(invalid='ignore'):
        return np.where(count > 0, total / count, np.nan)


def _divergence_2d(u, v, dx, dy):
    """Centered-difference divergence on a regular 2D grid (m/s per m).

    Returns NaN at the array edges and wherever either neighbor is NaN.
    """
    du_dx = np.full_like(u, np.nan)
    dv_dy = np.full_like(v, np.nan)
    du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    dv_dy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dy)
    return du_dx + dv_dy


def _helmholtz_div(u, v, mask, land_not_in_mask, dx, dy):
    """Return (u_div, v_div) -- the divergent (irrotational) part of (u, v).

    Solves ∇²X = div(u) on the data domain with:
      - Dirichlet X = 0 on the open-ocean / data-edge boundary
      - Neumann dX/dn = 0 on land / coast cells  (no flow into land)
    Then u_div = ∂X/∂x, v_div = ∂X/∂y.

    The caller can subtract these from (u, v) to get a non-divergent field:
        u_rot = u - u_div,  v_rot = v - v_div    (with div(u_rot) ≈ 0)
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from scipy.ndimage import distance_transform_edt

    N = int(mask.sum())
    if N == 0:
        nan = np.full(mask.shape, np.nan)
        return nan, nan

    div_HF = _divergence_2d(u, v, dx, dy)

    # Fill NaN div values INSIDE the mask via nearest-valid neighbor
    valid_div = np.isfinite(div_HF) & mask
    fill_needed = mask & ~valid_div
    if fill_needed.any():
        ind = distance_transform_edt(~valid_div, return_distances=False,
                                     return_indices=True)
        div_filled = div_HF[tuple(ind)]
    else:
        div_filled = div_HF.copy()
    div_rhs = np.where(mask, div_filled, 0.0)

    # Sparse 5-point Laplacian over the mask (vectorized assembly)
    idx_map = -np.ones(mask.shape, dtype=int); idx_map[mask] = np.arange(N)
    inv_dx2 = 1.0 / dx**2
    inv_dy2 = 1.0 / dy**2
    rows = []; cols = []; vals = []

    both = mask[:, :-1] & mask[:, 1:]
    ii, jj = np.where(both)
    ki = idx_map[ii, jj]; kr = idx_map[ii, jj+1]
    rows.extend(np.concatenate([ki, kr])); cols.extend(np.concatenate([kr, ki]))
    vals.extend([inv_dx2] * (2*len(ii)))

    both = mask[:-1, :] & mask[1:, :]
    ii, jj = np.where(both)
    ki = idx_map[ii, jj]; kr = idx_map[ii+1, jj]
    rows.extend(np.concatenate([ki, kr])); cols.extend(np.concatenate([kr, ki]))
    vals.extend([inv_dy2] * (2*len(ii)))

    # Diagonal: base -2/dx^2 - 2/dy^2 + Neumann mirror for each land neighbor
    n_land_x = np.zeros(mask.shape, dtype=int)
    n_land_y = np.zeros(mask.shape, dtype=int)
    n_land_x[:, :-1] += land_not_in_mask[:, 1:].astype(int)
    n_land_x[:, 1:]  += land_not_in_mask[:, :-1].astype(int)
    n_land_y[:-1, :] += land_not_in_mask[1:, :].astype(int)
    n_land_y[1:, :]  += land_not_in_mask[:-1, :].astype(int)
    diag = (-2.0*inv_dx2 - 2.0*inv_dy2
            + n_land_x.astype(float)*inv_dx2
            + n_land_y.astype(float)*inv_dy2)
    rows.extend(np.arange(N)); cols.extend(np.arange(N))
    vals.extend(diag[mask].tolist())

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    X_flat = spsolve(A.tocsc(), div_rhs[mask])
    X = np.full(mask.shape, np.nan); X[mask] = X_flat

    u_div = np.full(mask.shape, np.nan); v_div = np.full(mask.shape, np.nan)
    X_safe = np.where(np.isfinite(X), X, 0.0)
    ok_x = np.isfinite(X[:, :-2]) & np.isfinite(X[:, 2:])
    u_div[:, 1:-1] = np.where(ok_x, (X_safe[:, 2:] - X_safe[:, :-2]) / (2*dx),
                              np.nan)
    ok_y = np.isfinite(X[:-2, :]) & np.isfinite(X[2:, :])
    v_div[1:-1, :] = np.where(ok_y, (X_safe[2:, :] - X_safe[:-2, :]) / (2*dy),
                              np.nan)
    return u_div, v_div


def _circular_mask(lat, lon, lat0, lon0, r_km):
    """Boolean array of cells within r_km of (lat0, lon0). Haversine."""
    R = 6371.0
    phi1, phi2 = np.radians(lat0), np.radians(lat)
    dphi = np.radians(lat - lat0)
    dlam = np.radians(lon - lon0)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)) <= r_km


def build_fake_obs(fwd_his_file, grid_file, output_name,
                   start_date_str, window_hours,
                   n_obs=75, min_sep_grid=2, lat0=32.0, lon0=-120.0,
                   radius_km=120.0, v_bias=-0.02, seed=42,
                   obs_error_var=0.01):
    """Build synthetic HF-radar obs from a forward-only NLM trajectory.

    Picks `n_obs` random wet rho-cells (≥ min_sep_grid apart) inside a
    radius_km disk centered at (lat0, lon0); samples the NLM u, v at the
    surface for each hour from start_date+3h to start_date+(window_hours-3)h
    (matching the Test 6 convention of obs at hours 3..N-3).
    Writes ROMS obs file at output_name and a sidecar `.config.json` with
    the parameters.
    """
    t0 = datetime.strptime(start_date_str, '%Y-%m-%d')
    t_start = t0 + timedelta(hours=3)
    t_end   = t0 + timedelta(hours=window_hours - 3)
    times = []
    cur = t_start
    while cur <= t_end:
        times.append(cur); cur += timedelta(hours=1)
    time_HF = np.array(times)

    with nc.Dataset(fwd_his_file) as fh:
        time_his = fh.variables['ocean_time'][:]
        time_dt = datetime(1999, 1, 1) + time_his * timedelta(seconds=1)
        lat = np.asarray(fh.variables['lat_rho'][1:, 1:])
        lon = np.asarray(fh.variables['lon_rho'][1:, 1:])
        mask_full = np.asarray(fh.variables['mask_rho'][:])
        mask_crop = mask_full[1:, 1:].astype(bool)
        u_all = np.squeeze(fh.variables['u'][:, -1, :, :])
        v_all = np.squeeze(fh.variables['v'][:, -1, :, :])

    mask0 = _circular_mask(lat, lon, lat0, lon0, radius_km)
    interior_wet = mask_crop.copy()
    interior_wet[1:,  :] &= mask_crop[:-1, :]
    interior_wet[:-1, :] &= mask_crop[1:,  :]
    interior_wet[:,  1:] &= mask_crop[:, :-1]
    interior_wet[:, :-1] &= mask_crop[:,  1:]
    interior_wet[0, :]   = False
    interior_wet[-1, :]  = False
    interior_wet[:, 0]   = False
    interior_wet[:, -1]  = False
    candidates = np.argwhere(mask0 & interior_wet)

    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    selected = []
    for c in candidates:
        if len(selected) >= n_obs:
            break
        if all(np.hypot(c[0] - s[0], c[1] - s[1]) >= min_sep_grid for s in selected):
            selected.append(tuple(c))
    selected = [(j, i) for (j, i) in selected if bool(mask_crop[j, i])]

    sel_mask = np.zeros_like(mask0, dtype=bool)
    for j, i in selected:
        sel_mask[j, i] = True

    HF = {}
    for k, t_ob in enumerate(time_HF):
        idx = int(np.argmin(np.abs(np.array([(td - t_ob).total_seconds()
                                              for td in time_dt]))))
        u1 = 0.5 * (u_all[idx, 0:-1, :] + u_all[idx, 1:, :])
        v1 = 0.5 * (v_all[idx, :, 0:-1] + v_all[idx, :, 1:])
        t_str = t_ob.strftime('%Y%m%d%H')
        HF[(t_str, 'lat')] = lat[sel_mask]
        HF[(t_str, 'lon')] = lon[sel_mask]
        HF[(t_str, 'u')]   = u1[sel_mask]
        HF[(t_str, 'v')]   = v1[sel_mask] + v_bias

    write_roms_obs_file(HF, output_name, grid_file, obs_err_var=obs_error_var)

    cfg = dict(kind='fake', start_date=start_date_str, window_hours=window_hours,
               n_obs=n_obs, min_sep_grid=min_sep_grid, lat0=lat0, lon0=lon0,
               radius_km=radius_km, v_bias=v_bias, seed=seed,
               obs_error_var=obs_error_var, fwd_his_file=fwd_his_file,
               selected_count=len(selected))
    with open(output_name + '.config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"  wrote fake obs file with {len(selected)} obs at {len(time_HF)} times -> {output_name}")


def write_roms_obs_file(HF, nc_file_out, grid_file, obs_err_var=0.01):
    """Convert HF dictionary -> ROMS obs.nc."""
    from scipy.spatial import cKDTree
    with nc.Dataset(grid_file) as g:
        lon_rho = g.variables['lon_rho'][:]
        lat_rho = g.variables['lat_rho'][:]

    eta_size, xi_size = lon_rho.shape
    pts = np.column_stack([lon_rho.ravel(), lat_rho.ravel()])
    tree = cKDTree(pts)

    def lonlat_to_grid(lons, lats):
        _, idx = tree.query(np.column_stack([lons, lats]))
        return (idx %  xi_size).astype(float), (idx // xi_size).astype(float)

    all_dates = sorted({k[0] for k in HF})
    time_ref = datetime(1999, 1, 1)
    obs_time, obs_lat, obs_lon, obs_xg, obs_yg = [], [], [], [], []
    obs_val, obs_type, obs_prov = [], [], []
    survey_t, n_per_survey = [], []
    for d in all_dates:
        dt = datetime(int(d[:4]), int(d[4:6]), int(d[6:8]), int(d[8:10]))
        days = (dt - time_ref).total_seconds() / 86400.0
        u = HF[(d, 'u')];   v = HF[(d, 'v')]
        lon_ = HF[(d, 'lon')]; lat_ = HF[(d, 'lat')]
        xg, yg = lonlat_to_grid(lon_, lat_)
        n = len(u)
        for vec, otype in ((u, 4), (v, 5)):
            obs_time.extend([days] * n);  obs_lat.extend(lat_); obs_lon.extend(lon_)
            obs_xg.extend(xg); obs_yg.extend(yg); obs_val.extend(vec)
            obs_type.extend([otype] * n); obs_prov.extend([1] * n)
        survey_t.append(days); n_per_survey.append(n * 2)

    n_total = len(obs_val)
    with nc.Dataset(nc_file_out, 'w', format='NETCDF4') as ds:
        ds.createDimension('datum', n_total)
        ds.createDimension('survey', len(all_dates))
        ds.createDimension('state_variable', 7)

        sp = ds.createVariable('spherical', 'i4'); sp[:] = 1
        sp.flag_values = "0, 1"; sp.flag_meanings = "Cartesian spherical"

        ds.createVariable('Nobs', 'i4', ('survey',))[:] = np.array(n_per_survey)
        ds.createVariable('survey_time', 'f8', ('survey',))[:] = np.array(survey_t)
        ds.variables['survey_time'].units = "day"

        ot = ds.createVariable('obs_type', 'i4', ('datum',))
        ot.flag_values   = "1, 2, 3, 4, 5, 6, 7"
        ot.flag_meanings = "zeta ubar vbar u v temperature salinity"
        ot[:] = np.array(obs_type)

        ds.createVariable('obs_provenance', 'i4', ('datum',))[:] = np.array(obs_prov)

        otm = ds.createVariable('obs_time', 'f8', ('datum',))
        otm.units = "day"; otm[:] = np.array(obs_time)

        ds.createVariable('obs_depth', 'f8', ('datum',))[:] = np.zeros(n_total)
        ds.createVariable('obs_Xgrid', 'f8', ('datum',))[:] = np.array(obs_xg)
        ds.createVariable('obs_Ygrid', 'f8', ('datum',))[:] = np.array(obs_yg)
        ds.createVariable('obs_Zgrid', 'f8', ('datum',))[:] = np.zeros(n_total)
        ds.createVariable('obs_error', 'f8', ('datum',))[:] = np.full(n_total, obs_err_var)
        ds.createVariable('obs_value', 'f8', ('datum',))[:] = np.array(obs_val)
        ds.createVariable('obs_lon',   'f8', ('datum',))[:] = np.array(obs_lon)
        ds.createVariable('obs_lat',   'f8', ('datum',))[:] = np.array(obs_lat)
        ds.type = "ROMS Observations"
        ds.title = "HF Radar Observations for LV1 4D-Var"
        ds.Conventions = "CF-1.4"


def build_real_obs(grid_file, output_name, start_date_str, window_hours,
                   thin_stride=3, hdop_max=1.5, sites_min=2, speed_max=1.5,
                   boundary_buffer_km=12.0, land_buffer_km=4.0,
                   obs_error_var=0.04, rotate_to_grid=True,
                   remove_u_divergent=True,
                   protect_grid_files=None,
                   hfr_cache_dir='/scratch/matt/testing/LV1/files_to_run/HF_radar',
                   ndbc_catalog='https://dods.ndbc.noaa.gov/thredds/catalog/hfradar/catalog.xml',
                   ndbc_base='https://dods.ndbc.noaa.gov/thredds/dodsC/'):
    """Build a ROMS obs file from real NDBC USWC 6km HF radar archive.

    Mirrors the QC pipeline in ipynb_files/hf_radar_load_qc.ipynb:
      - load raw HFR from `hfr_cache_dir` if available, else fetch from NDBC
        (and save the cache for future runs).  NDBC's main catalog only
        keeps a rolling ~90-day window, so always have the cache for older
        dates.
      - subset to grid lon/lat box
      - inside-grid + on-water + boundary/land buffers
      - 5pt cross spatial pre-average (if thin_stride==3)
      - if remove_u_divergent: solve a Helmholtz decomposition per time step
        and subtract the divergent (irrotational) part of the velocity field
        -- removes most of the small-scale HFR divergence noise while
        preserving large-scale rotational currents
      - rotate to ROMS (xi, eta) frame using grid 'angle'
      - QC: hdop, sites, speed cap, spatial Hampel (per-time, 3x3, N_MAD=4)
      - thin to every Nth lat/lon cell
      - if `protect_grid_files` is given: cells whose (lon,lat) falls inside
        ANY protected grid's outline polygon bypass the thinning step (they
        are always kept if they pass the earlier QC).  Use this to guarantee
        HFR coverage inside a specific ROMS sub-grid (e.g. LV3, LV4).  Cell
        outlines are read from each protect grid's lon_rho/lat_rho corners.
    Writes obs file + .config.json sidecar.
    """
    import xarray as xr
    from matplotlib.path import Path as MplPath
    from scipy.interpolate import NearestNDInterpolator
    from scipy.spatial import cKDTree

    t_start = datetime.strptime(start_date_str, '%Y-%m-%d')
    t_end = t_start + timedelta(hours=window_hours)

    # ---- LV1 box from grid ----
    with nc.Dataset(grid_file) as g:
        lv1_lon  = np.asarray(g.variables['lon_rho'][:])
        lv1_lat  = np.asarray(g.variables['lat_rho'][:])
        lv1_mask = np.asarray(g.variables['mask_rho'][:])
        lv1_angle = np.asarray(g.variables['angle'][:])

    lon_min, lon_max = lv1_lon.min(), lv1_lon.max()
    lat_min, lat_max = lv1_lat.min(), lv1_lat.max()
    KEEP = ['u', 'v', 'dopx', 'dopy', 'hdop', 'number_of_sites', 'number_of_radials']

    # ---- cache file path (matches notebook hf_radar_load_qc.ipynb convention) ----
    cache_path = None
    if hfr_cache_dir:
        cache_name = (f"hfr_uswc_6km_LV1_"
                      f"{t_start:%Y%m%dT%H}_{t_end:%Y%m%dT%H}.nc")
        cache_path = os.path.join(hfr_cache_dir, cache_name)

    # ---- load: cache first, fall back to NDBC fetch + save cache ----
    if cache_path and os.path.exists(cache_path):
        print(f"  loading cached HFR file:  {cache_path}")
        hfr = xr.open_dataset(cache_path, engine='netcdf4',
                               decode_timedelta=True)
        # Subset (cache may have been written with a slightly larger box)
        hfr = hfr.sel(lat=slice(lat_min, lat_max),
                       lon=slice(lon_min, lon_max),
                       time=slice(t_start, t_end))
        if 'depth' in hfr.dims:
            hfr = hfr.squeeze('depth', drop=True)
        hfr = hfr.load()
        # Keep only the columns build_real_obs needs (cache might have extras)
        hfr = hfr[[v for v in KEEP if v in hfr.data_vars]]
        print(f"  cache has {hfr.sizes['time']} time records in window")
        n_records_for_cfg = int(hfr.sizes['time'])
    else:
        # ---- discover NDBC files ----
        # NDBC's main catalog only retains a rolling ~90-day window, so this
        # path won't find files older than that.  Save the result to cache
        # afterward so future runs of the same date are fast and robust.
        ns = {'t': 'http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0'}
        FILE_RE = re.compile(r'^rtv-uswc-6km-uwls_v1r0_hfr_s(?P<s>\d{14})\d_e\d{15}_c\d{15}\.nc$')
        with urllib.request.urlopen(ndbc_catalog, timeout=120) as r:
            cat = ET.parse(r).getroot()
        records = []
        for ds in cat.iter(f"{{{ns['t']}}}dataset"):
            name = ds.get('name'); url_path = ds.get('urlPath')
            if not name or not url_path: continue
            m = FILE_RE.match(name)
            if not m: continue
            t = datetime.strptime(m['s'], '%Y%m%d%H%M%S')
            if t_start <= t <= t_end:
                records.append((t, url_path))
        records.sort()
        print(f"  found {len(records)} HFR files in window {t_start} .. {t_end}")
        if not records:
            raise RuntimeError(
                f"No HFR files found in NDBC catalog for window "
                f"{t_start} .. {t_end} and no cache at "
                f"{cache_path!r}.  The date may be outside NDBC's rolling "
                f"~90-day catalog window."
            )

        def _load(url):
            ds = xr.open_dataset(url, engine='netcdf4', decode_timedelta=True)[KEEP]
            ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
            if 'depth' in ds.dims:
                ds = ds.squeeze('depth', drop=True)
            return ds.load()

        pieces = []
        for t, urlpath in records:
            print(f"    {t:%H:%MZ} loading {urlpath.split('/')[-1]}")
            pieces.append(_load(ndbc_base + urlpath))
        hfr = xr.concat(pieces, dim='time').sortby('time')
        n_records_for_cfg = len(records)

        # Save cache so future runs skip the NDBC fetch
        if cache_path:
            try:
                os.makedirs(hfr_cache_dir, exist_ok=True)
                enc = {v: {'zlib': True, 'complevel': 4} for v in hfr.data_vars}
                hfr.to_netcdf(cache_path, encoding=enc)
                print(f"  saved cache:  {cache_path}  "
                      f"({os.path.getsize(cache_path)/1e6:.1f} MB)")
            except Exception as e:
                print(f"  WARNING: failed to save cache {cache_path}: {e}")

    # ---- 5-point cross spatial average (per time step) ----
    # Smooths each cell using itself + 4 cardinal neighbors (NaN-skipping).
    # Calibrated for thin_stride=3: subsequent 3:1 thinning at 18 km then
    # uses every native ~6 km cell rather than discarding ~8 of 9.
    if thin_stride == 3:
        u_arr = hfr['u'].values
        v_arr = hfr['v'].values
        n_in  = np.isfinite(u_arr).sum()
        for it in range(u_arr.shape[0]):
            u_arr[it] = _cross5_avg(u_arr[it])
            v_arr[it] = _cross5_avg(v_arr[it])
        n_out = np.isfinite(u_arr).sum()
        hfr = hfr.assign(u=(('time','lat','lon'), u_arr),
                         v=(('time','lat','lon'), v_arr))
        print(f"  5pt-cross pre-average: valid u cells {n_in} -> {n_out} "
              f"(+{n_out - n_in} from filled neighbors)")
    else:
        print(f"  thin_stride={thin_stride}: skipping 5pt-cross pre-average "
              f"(stencil is calibrated for thin_stride=3 only)")

    # ---- Helmholtz decomposition: remove the divergent part of the velocity ----
    # Per time step: solve ∇²X = div(u, v) on the HFR data domain with
    # Dirichlet X=0 at the open-ocean / data edge and Neumann dX/dn=0 at land.
    # Then subtract u_div = ∂X/∂x, v_div = ∂X/∂y so the residual is
    # non-divergent (div(u_rot, v_rot) ≈ 0) -- mostly removes HFR's small-scale
    # measurement-noise divergence while preserving large-scale rotational
    # currents.  Done on (uE, vN) before rotation so the lat/lon dx, dy and
    # the centered-difference divergence are straightforward.
    if remove_u_divergent:
        LAT_h = hfr.lat.values
        LON_h = hfr.lon.values
        # km grid spacing for the HFR lat/lon grid (uniform for NDBC USWC 6km)
        KMLAT = 111.32
        KMLON = KMLAT * np.cos(np.deg2rad(0.5 * (lat_min + lat_max)))
        dx_hfr = float(LON_h[1] - LON_h[0]) * KMLON * 1000.0
        dy_hfr = float(LAT_h[1] - LAT_h[0]) * KMLAT * 1000.0

        # Land mask on HFR grid (for Neumann coast condition)
        lat2d_h, lon2d_h = np.meshgrid(LAT_h, LON_h, indexing='ij')
        _mask_interp = NearestNDInterpolator(
            np.column_stack([lv1_lon.ravel(), lv1_lat.ravel()]),
            lv1_mask.ravel())
        land_hfr = _mask_interp(np.column_stack([lon2d_h.ravel(), lat2d_h.ravel()])
                                ).reshape(lat2d_h.shape) < 0.5

        u_arr = hfr['u'].values
        v_arr = hfr['v'].values
        div_rms_before = []
        div_rms_after  = []
        for it in range(u_arr.shape[0]):
            u_t = u_arr[it]; v_t = v_arr[it]
            mask_t = np.isfinite(u_t) & np.isfinite(v_t)
            if mask_t.sum() == 0:
                continue
            land_not_in_mask_t = land_hfr & ~mask_t
            u_div_t, v_div_t = _helmholtz_div(u_t, v_t, mask_t,
                                              land_not_in_mask_t,
                                              dx_hfr, dy_hfr)
            # Divergence rms before/after (for the log line)
            div_b = _divergence_2d(u_t, v_t, dx_hfr, dy_hfr)
            div_rms_before.append(np.sqrt(np.nanmean(div_b**2)))
            u_arr[it] = u_t - u_div_t
            v_arr[it] = v_t - v_div_t
            div_a = _divergence_2d(u_arr[it], v_arr[it], dx_hfr, dy_hfr)
            div_rms_after.append(np.sqrt(np.nanmean(div_a**2)))

        hfr = hfr.assign(u=(('time', 'lat', 'lon'), u_arr),
                         v=(('time', 'lat', 'lon'), v_arr))
        rms_b = float(np.nanmean(div_rms_before)) if div_rms_before else 0.0
        rms_a = float(np.nanmean(div_rms_after))  if div_rms_after  else 0.0
        red = 100*(1 - rms_a/rms_b) if rms_b > 0 else 0.0
        print(f"  removed divergent part (Neumann at land):  "
              f"div rms {rms_b:.3e} -> {rms_a:.3e}  ({red:.0f}% reduction)")
    else:
        print("  remove_u_divergent=False: keeping full (rotational+divergent) field")

    # ---- subsample masks ----
    lat2d, lon2d = np.meshgrid(hfr.lat.values, hfr.lon.values, indexing='ij')
    pts2d = np.column_stack([lon2d.ravel(), lat2d.ravel()])

    # boundary polygon
    blon = np.concatenate([lv1_lon[0,:], lv1_lon[1:-1,-1], lv1_lon[-1,::-1], lv1_lon[-2:0:-1,0]])
    blat = np.concatenate([lv1_lat[0,:], lv1_lat[1:-1,-1], lv1_lat[-1,::-1], lv1_lat[-2:0:-1,0]])
    inside_lv1 = MplPath(np.column_stack([blon, blat])).contains_points(pts2d).reshape(lat2d.shape)
    nn_mask = NearestNDInterpolator(np.column_stack([lv1_lon.ravel(), lv1_lat.ravel()]),
                                    lv1_mask.ravel())
    on_water = nn_mask(pts2d).reshape(lat2d.shape) >= 0.5

    # km grid for distance tests
    lat0 = 0.5*(lat_min+lat_max); lon0 = 0.5*(lon_min+lon_max)
    KMLAT = 111.32; KMLON = 111.32*np.cos(np.deg2rad(lat0))
    def ll_to_xy(lon, lat):
        lon = np.asarray(lon).ravel(); lat = np.asarray(lat).ravel()
        return np.column_stack([(lon-lon0)*KMLON, (lat-lat0)*KMLAT])

    bdy_xy = ll_to_xy(blon, blat)
    hfr_xy = ll_to_xy(lon2d, lat2d)
    d_bdy, _ = cKDTree(bdy_xy).query(hfr_xy, k=1)
    far_from_bdy = d_bdy.reshape(lat2d.shape) > boundary_buffer_km

    land_pts = np.column_stack([lv1_lon[lv1_mask==0], lv1_lat[lv1_mask==0]])
    d_land, _ = cKDTree(ll_to_xy(land_pts[:,0], land_pts[:,1])).query(hfr_xy, k=1)
    far_from_land = d_land.reshape(lat2d.shape) > land_buffer_km

    keep_2d = inside_lv1 & on_water & far_from_bdy & far_from_land
    print(f"  subsample 2D mask: {keep_2d.sum()} of {keep_2d.size} cells kept")
    hfr_lv1 = hfr.where(xr.DataArray(keep_2d, dims=('lat','lon')))

    # rotate
    if rotate_to_grid:
        ang_interp = NearestNDInterpolator(np.column_stack([lv1_lon.ravel(), lv1_lat.ravel()]),
                                           lv1_angle.ravel())
        a2d = ang_interp(pts2d).reshape(lat2d.shape)
        ca = np.cos(a2d)[None,:,:]; sa = np.sin(a2d)[None,:,:]
        uE = hfr_lv1['u'].values; vN = hfr_lv1['v'].values
        u_xi  =  uE*ca + vN*sa
        v_eta = -uE*sa + vN*ca
        hfr_lv1 = hfr_lv1.assign(u=(('time','lat','lon'), u_xi),
                                 v=(('time','lat','lon'), v_eta))

    # QC: hdop, sites, speed, spatial Hampel (per time)
    u_data = hfr_lv1['u'].values; v_data = hfr_lv1['v'].values
    hdop = hfr_lv1['hdop'].values; nsites = hfr_lv1['number_of_sites'].values
    speed = np.hypot(u_data, v_data)
    keep_qc = ((hdop <= hdop_max) & ~np.isnan(hdop)
               & (nsites >= sites_min) & ~np.isnan(nsites)
               & (speed <= speed_max) & ~np.isnan(speed))

    # Hampel
    N_MAD = 4; MAD_FLOOR = 0.01
    nT, nlat, nlon = u_data.shape
    def nbhrs(a):
        pad = np.pad(a, 1, mode='constant', constant_values=np.nan)
        stk = np.empty((9,nlat,nlon))
        k=0
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                stk[k] = pad[1+dy:1+dy+nlat, 1+dx:1+dx+nlon]; k+=1
        return stk
    keep_hampel = np.zeros_like(u_data, dtype=bool)
    for t in range(nT):
        u_n = nbhrs(u_data[t]); v_n = nbhrs(v_data[t])
        # All-NaN slices at edges/land produce a benign nanmedian warning;
        # the resulting NaN cells are masked out by `ok[np.isnan(...)]=False` below.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    message='All-NaN slice encountered',
                                    category=RuntimeWarning)
            um = np.nanmedian(u_n, axis=0); vm = np.nanmedian(v_n, axis=0)
            umad = np.nanmedian(np.abs(u_n - um[None]), axis=0)
            vmad = np.nanmedian(np.abs(v_n - vm[None]), axis=0)
        ok = (np.abs(u_data[t]-um) <= N_MAD*np.fmax(umad, MAD_FLOOR)) & \
             (np.abs(v_data[t]-vm) <= N_MAD*np.fmax(vmad, MAD_FLOOR))
        ok[np.isnan(u_data[t]) | np.isnan(v_data[t])] = False
        keep_hampel[t] = ok
    qc_mask = keep_qc & keep_hampel
    print(f"  QC retained {qc_mask.sum()} of {(~np.isnan(u_data)).sum()} valid cells "
          f"({100*qc_mask.sum()/max((~np.isnan(u_data)).sum(),1):.1f}%)")

    # spatial thinning
    thin = np.zeros((nlat, nlon), dtype=bool)
    thin[::thin_stride, ::thin_stride] = True

    # Protected polygons -- cells inside these regions bypass thinning entirely.
    # Read each grid's four corners to build a closed quadrilateral polygon.
    if protect_grid_files:
        for gpath in protect_grid_files:
            with nc.Dataset(gpath) as pg:
                p_lon = np.asarray(pg.variables['lon_rho'][:])
                p_lat = np.asarray(pg.variables['lat_rho'][:])
            corners = np.array([
                [p_lon[0, 0],   p_lat[0, 0]],
                [p_lon[0, -1],  p_lat[0, -1]],
                [p_lon[-1, -1], p_lat[-1, -1]],
                [p_lon[-1, 0],  p_lat[-1, 0]],
                [p_lon[0, 0],   p_lat[0, 0]],
            ])
            in_protect = (MplPath(corners)
                          .contains_points(pts2d)
                          .reshape(lat2d.shape))
            n_new = int((in_protect & ~thin).sum())
            thin |= in_protect
            print(f"  protect_grid_file {os.path.basename(gpath)}: "
                  f"kept {int(in_protect.sum())} HFR cells inside polygon "
                  f"({n_new} newly protected from thinning)")

    qc_mask = qc_mask & thin[None]
    print(f"  after thin every {thin_stride} (+ protected polygons): {qc_mask.sum()} cells")

    hfr_lv1 = hfr_lv1.where(xr.DataArray(qc_mask, dims=('time','lat','lon'),
                                         coords=dict(time=hfr_lv1.time,
                                                     lat=hfr_lv1.lat,
                                                     lon=hfr_lv1.lon)))

    # build dict
    HF = {}
    lat2dh, lon2dh = np.meshgrid(hfr_lv1.lat.values, hfr_lv1.lon.values, indexing='ij')
    for it in range(hfr_lv1.sizes['time']):
        t_str = np.datetime64(hfr_lv1.time.values[it], 's').astype(object).strftime('%Y%m%d%H')
        u_t = hfr_lv1['u'].isel(time=it).values
        v_t = hfr_lv1['v'].isel(time=it).values
        valid = np.isfinite(u_t) & np.isfinite(v_t)
        HF[(t_str, 'lat')] = lat2dh[valid]
        HF[(t_str, 'lon')] = lon2dh[valid]
        HF[(t_str, 'u')]   = u_t[valid]
        HF[(t_str, 'v')]   = v_t[valid]

    write_roms_obs_file(HF, output_name, grid_file, obs_err_var=obs_error_var)

    protect_names = (sorted(os.path.basename(g) for g in protect_grid_files)
                     if protect_grid_files else [])
    # Config_matches from the shell compares scalars; store a joined string too
    # so a simple `protect_key='LV3+LV4'` kwarg matches on retrieval.
    protect_key = '+'.join(protect_names) if protect_names else ''
    cfg = dict(kind='real', start_date=start_date_str, window_hours=window_hours,
               thin_stride=thin_stride, hdop_max=hdop_max, sites_min=sites_min,
               speed_max=speed_max, boundary_buffer_km=boundary_buffer_km,
               land_buffer_km=land_buffer_km, obs_error_var=obs_error_var,
               rotate_to_grid=bool(rotate_to_grid),
               remove_u_divergent=bool(remove_u_divergent),
               protect_grids=protect_names,
               protect_key=protect_key,
               n_files=n_records_for_cfg)
    with open(output_name + '.config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"  wrote real obs file -> {output_name}")

    # Report final counts inside each protected grid (post-QC, post-thinning,
    # post-Hampel).  These are the numbers that actually reach the DA.
    if protect_grid_files:
        with nc.Dataset(output_name) as ods:
            obs_lon_f  = np.asarray(ods.variables['obs_lon'][:])
            obs_lat_f  = np.asarray(ods.variables['obs_lat'][:])
            obs_type_f = np.asarray(ods.variables['obs_type'][:])
        pts_f = np.column_stack([obs_lon_f, obs_lat_f])
        for gpath in protect_grid_files:
            with nc.Dataset(gpath) as pg:
                p_lon = np.asarray(pg.variables['lon_rho'][:])
                p_lat = np.asarray(pg.variables['lat_rho'][:])
            corners = np.array([
                [p_lon[0, 0],   p_lat[0, 0]],
                [p_lon[0, -1],  p_lat[0, -1]],
                [p_lon[-1, -1], p_lat[-1, -1]],
                [p_lon[-1, 0],  p_lat[-1, 0]],
                [p_lon[0, 0],   p_lat[0, 0]],
            ])
            inside = MplPath(corners).contains_points(pts_f)
            n_u = int((inside & (obs_type_f == 4)).sum())
            n_v = int((inside & (obs_type_f == 5)).sum())
            n_pairs = min(n_u, n_v)  # paired at same (lon,lat) by write_roms_obs_file
            print(f"  final obs inside {os.path.basename(gpath)}: "
                  f"{n_pairs} (u,v) pairs across all surveys "
                  f"= {n_u + n_v} datums ({n_u} u + {n_v} v)")


def build_real_obs_weighted(grid_file, output_name, start_date_str, window_hours,
                            target_lon, target_lat,
                            Lx_km, Ly_km, theta_deg=0.0,
                            thin_stride=3, hdop_max=1.5, sites_min=2, speed_max=1.5,
                            boundary_buffer_km=12.0, land_buffer_km=4.0,
                            obs_error_var=0.015, rotate_to_grid=True,
                            remove_u_divergent=True,
                            protect_grid_files=None,
                            hfr_cache_dir='/scratch/matt/testing/LV1/files_to_run/HF_radar',
                            ndbc_catalog='https://dods.ndbc.noaa.gov/thredds/catalog/hfradar/catalog.xml',
                            ndbc_base='https://dods.ndbc.noaa.gov/thredds/dodsC/'):
    """build_real_obs + per-obs elliptical inflation of obs_error.

    Equivalent to running build_real_obs with the same baseline obs_error_var,
    then overwriting `obs_error` per-obs so that

        sigma_obs(x) = sqrt(obs_error_var) * exp(d_ellipse(x))
        obs_error(x) = obs_error_var       * exp(2 * d_ellipse(x))

    where d_ellipse is the Mahalanobis-like distance in the frame rotated CCW
    by theta_deg from east, centered at (target_lon, target_lat):

        dx_km = (obs_lon - target_lon) * 111.32 * cos(target_lat)
        dy_km = (obs_lat - target_lat) * 111.32
        dx_rot =  cos(theta)*dx + sin(theta)*dy
        dy_rot = -sin(theta)*dx + cos(theta)*dy
        d_ellipse = sqrt((dx_rot/Lx_km)**2 + (dy_rot/Ly_km)**2)

    Obs at the target keep their full weight (d=0 -> obs_error unchanged); obs
    on the d=1 ellipse have weight ~exp(-2) ~ 0.135x the baseline; obs far out
    have effectively zero weight.

    Sidecar `kind` is set to 'real_weighted' so config_matches will distinguish
    a weighted file from a plain-real one even at the same baseline params.
    """
    # 1. Build the base obs file (NetCDF + sidecar) via build_real_obs
    build_real_obs(grid_file=grid_file, output_name=output_name,
                   start_date_str=start_date_str, window_hours=window_hours,
                   thin_stride=thin_stride, hdop_max=hdop_max,
                   sites_min=sites_min, speed_max=speed_max,
                   boundary_buffer_km=boundary_buffer_km,
                   land_buffer_km=land_buffer_km,
                   obs_error_var=obs_error_var,
                   rotate_to_grid=rotate_to_grid,
                   remove_u_divergent=remove_u_divergent,
                   protect_grid_files=protect_grid_files,
                   hfr_cache_dir=hfr_cache_dir,
                   ndbc_catalog=ndbc_catalog,
                   ndbc_base=ndbc_base)

    # 2. Inflate obs_error per-obs from elliptical distance to target
    lat_to_km = 111.32
    lon_to_km = 111.32 * np.cos(np.deg2rad(target_lat))
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)

    with nc.Dataset(output_name, 'a') as ds:
        obs_lon_arr = np.asarray(ds.variables['obs_lon'][:])
        obs_lat_arr = np.asarray(ds.variables['obs_lat'][:])
        dx_km =  (obs_lon_arr - target_lon) * lon_to_km
        dy_km =  (obs_lat_arr - target_lat) * lat_to_km
        dx_rot =  c * dx_km + s * dy_km
        dy_rot = -s * dx_km + c * dy_km
        d_ell  = np.sqrt((dx_rot / Lx_km) ** 2 + (dy_rot / Ly_km) ** 2)
        obs_error_new = obs_error_var * np.exp(2.0 * d_ell)
        ds.variables['obs_error'][:] = obs_error_new

    # 3. Update sidecar: add ellipse params + bump 'kind' so the orchestrator
    #    won't reuse a weighted file when the user later flips back to plain
    #    HFreal (or vice versa).
    sidecar = output_name + '.config.json'
    with open(sidecar) as fh:
        cfg = json.load(fh)
    cfg['kind']       = 'real_weighted'
    cfg['target_lon'] = float(target_lon)
    cfg['target_lat'] = float(target_lat)
    cfg['Lx_km']      = float(Lx_km)
    cfg['Ly_km']      = float(Ly_km)
    cfg['theta_deg']  = float(theta_deg)
    with open(sidecar, 'w') as fh:
        json.dump(cfg, fh, indent=2)

    # 4. Report (mirrors the notebook 5b text box)
    n_total = int(d_ell.size)
    n_in1   = int((d_ell <= 1).sum())
    n_in2   = int((d_ell <= 2).sum())
    sigma_min = float(np.sqrt(obs_error_new.min()))
    sigma_max = float(np.sqrt(obs_error_new.max()))
    print(f"  elliptical weighting applied:")
    print(f"    center = ({target_lon}, {target_lat}); "
          f"Lx={Lx_km} km, Ly={Ly_km} km, theta={theta_deg} deg")
    print(f"    obs inside d<=1: {n_in1}/{n_total} ({100*n_in1/n_total:.1f}%)")
    print(f"    obs inside d<=2: {n_in2}/{n_total} ({100*n_in2/n_total:.1f}%)")
    print(f"    sigma_obs range: {sigma_min:.4f} ... {sigma_max:.4f} m/s")


# =============================================================================
# Sidecar config helpers (used by the shell orchestrator)
# =============================================================================

def write_config(path, **kwargs):
    """Write a .config.json sidecar describing what was used to build `path`."""
    with open(path + '.config.json', 'w') as f:
        json.dump(kwargs, f, indent=2, sort_keys=True)


def config_matches(path, **kwargs):
    """True iff path exists AND its sidecar matches kwargs.

    Used by set_up_DA_simulation.sh to decide whether to rebuild std/obs/nrm.
    """
    if not os.path.exists(path):
        return False
    sidecar = path + '.config.json'
    if not os.path.exists(sidecar):
        return False
    try:
        with open(sidecar) as f:
            saved = json.load(f)
    except Exception:
        return False
    for k, v in kwargs.items():
        if k not in saved:
            return False
        a, b = saved[k], v
        try:
            if isinstance(a, float) or isinstance(b, float):
                if not np.isclose(float(a), float(b), rtol=1e-9, atol=0):
                    return False
            elif a != b:
                return False
        except Exception:
            if a != b:
                return False
    return True


if __name__ == '__main__':
    # Tiny CLI: `python DA_setup_python_funs.py config_matches <path> k=v k=v ...`
    # returns exit 0 if match, 1 otherwise. Used by the shell script.
    import sys
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    cmd = sys.argv[1]
    if cmd == 'config_matches':
        path = sys.argv[2]
        kvs = {}
        for arg in sys.argv[3:]:
            k, v = arg.split('=', 1)
            try:
                kvs[k] = float(v) if ('.' in v or 'e' in v.lower()) else int(v)
            except ValueError:
                kvs[k] = v
        sys.exit(0 if config_matches(path, **kvs) else 1)
    else:
        print(f"unknown command: {cmd}", file=sys.stderr); sys.exit(2)
