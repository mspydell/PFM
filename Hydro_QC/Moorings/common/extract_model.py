#!/usr/bin/env python3
"""Extract PFM model time series at the moorings (SBOO and the IB mini moorings from LV4, PLOO and
Del Mar from LV3), and export hourly CSVs.

Python port of the MATLAB chain
  PFM_Master_Code_For_TimeSeriesExtraction.m   (registry, resume from the last saved date)
  LV1to4_model_data_serial_PHM_PFM_V2.m        (nearest grid point, forecast-day window, S-coordinates)
  append_to_matfile.m                          (merge with the saved file, keep last duplicate)
  SBOO_Model_DataExtraction.m                  (ADCP-equivalent, layer-thickness weighted depth average)
  export_model_timeseries.m                    (T/S at fixed depths, top-5 m currents)

Step 1, extract (slow, reads /dataSIO): all sites are read from each history file in one pass, so
  one file costs ~35 s for all five sites, and files run in parallel (--workers).
  Output: cache/model/<site>_LV4_day<N>_profiles.nc with hourly u, v (on the nearest u/v points,
  grid-relative), temp, salt (nearest rho point) for all s-levels, and zeta.
  Reruns only read files from the last saved day on (like resolve_date_range in MATLAB).
Step 2, export (fast): cache/model/<site>_LV4_day<N>_hourly.csv with
  u_davg, v_davg  east/north averaged over DAVG_RANGE (the depths the site's ADCP measures),
                  weighted by how much of each model layer lies inside the range
  u_top, v_top    same over TOP_RANGE (the depths the obs "top" panel uses)
  temp_<d>m, sal_<d>m  interpolated to d m below mean sea level (fixed sensor depth)
  Velocities are rotated from grid to east/north with the grid angle, as in the MATLAB code:
    u_east = u cos(angle) - v sin(angle),  v_north = u sin(angle) + v cos(angle)

  python extract_model.py                         # extract new days for all sites, then export
  python extract_model.py --sites SBOO IB-C --start 2024-12-05 --end 2026-10-07
  python extract_model.py --export-only           # just rebuild the CSVs
"""
import argparse
import glob
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor

import netCDF4
import numpy as np
import pandas as pd

# shared cache on /dataSIO, seen from swell and backwash; the HYDRO_QC_DATA environment variable overrides the root
MODEL_DIR = os.path.join(os.environ.get("HYDRO_QC_DATA", "/dataSIO/Hydro_QC"), "Moorings", "cache", "model")
BASE_DATA = "/dataSIO/PFM_Simulations/Archive"
GRIDS = {"LV1": "/dataSIO/PFM_Simulations/Grid/GRID_SDTJRE_LV1_rx020_hmask.nc",
         "LV2": "/dataSIO/PFM_Simulations/Grid/GRID_SDTJRE_LV2_rx020.nc",
         "LV3": "/dataSIO/PFM_Simulations/Grid/GRID_SDTJRE_LV3_rx020.nc",
         "LV4": "/dataSIO/PFM_Simulations/Grid/GRID_SDTJRE_LV4_mss_oct2024.nc"}
TIME0 = pd.Timestamp("1999-01-01", tz="UTC")  # ocean_time is seconds since this
VARS = ["u", "v", "temp", "salt", "zeta"]

SBOO_DEPTH = 31.48  # m, water depth at the SBOO mooring
# Positions are the MATLAB registry ones (IDs 2, 10, 101, 102, 103). The grid is ~28 m, so the
# small differences from Jeff's surveyed positions in download_obs.SITES can change the grid cell.
# Depth ranges are (top, bottom) in m, positive down, below mean sea level ("msl") or the moving
# free surface ("surface"); bottom None = the sea floor.
SITES = {
    # Downward-looking ADCP with bins at fixed depths below the mean surface (SBOO_Model_DataExtraction):
    # blanking distance 3.43 m at the top, side-lobe cutoff at 92% of the water depth.
    # "top" = the 3.33, 4.33 and 5.33 m ADCP bins (1 m bins), i.e. 2.83-5.83 m.
    "SBOO": dict(lon=-117.186170, lat=32.532040, z_ref="msl",
                 davg=(3.43, 0.92 * SBOO_DEPTH), top=(2.83, 5.83)),
    # IB-C ADCP: good bins (CUR_MASK) run from ~1.75 m to ~8.74 m below the moving surface (median
    # over Dec 2024 - Oct 2026), 0.5 m bins; the obs "top" uses the bins shallower than 3 m (top 3 m).
    "IB-C": dict(lon=-117.141586, lat=32.578995, z_ref="surface", davg=(1.5, 9.0), top=(1.5, 3.0)),
    # No ADCP at these three: whole water column and the top 5 m.
    "IB-N": dict(lon=-117.140397, lat=32.615455, z_ref="surface", davg=(0.0, None), top=(0.0, 5.0)),
    "IB-S": dict(lon=-117.151040, lat=32.539694, z_ref="surface", davg=(0.0, None), top=(0.0, 5.0)),
    "IB-W": dict(lon=-117.162993, lat=32.581562, z_ref="surface", davg=(0.0, None), top=(0.0, 5.0)),
    # LV3 sites, registry IDs 15 (Point Loma) and 14 (Del Mar LV3). Downward-looking ADCPs at fixed bin
    # depths below the mean surface (PLOO 1 m bins from 3.43 m, Del Mar 2 m bins from 5.42 m). The
    # model is shallower than the moorings (LV3 h = 95.6 m at PLOO, 102 m at Del Mar), so obs and model
    # are both averaged from the first bin down to 92% of the model depth (as DelMar_Model_DataExtraction:
    # R_max = 0.92 h, blanking 5.42 m). "top": PLOO the 3.43, 4.43, 5.43 m bins; Del Mar the 5.42, 7.42,
    # 9.42 m bins (its first bin is below 5 m).
    "PLOO": dict(lon=-117.32421, lat=32.67005, z_ref="msl", level="LV3", davg=(3.43, 0.92 * 95.6),
                 top=(2.93, 5.93), ts_depths=[1, 10, 20, 30, 45, 60, 75, 87, 89, 90]),
    "DelMar": dict(lon=-117.31661, lat=32.92947, z_ref="msl", level="LV3", davg=(5.42, 0.92 * 102.0),
                   top=(4.42, 10.42), ts_depths=[1, 4, 7, 15, 24, 35, 47, 60, 74, 89, 90]),
}
TS_DEPTHS = [1, 8, 10, 18, 26]  # m below mean sea level: IB sensors at 1, 8, 10 m; SBOO at 1, 10, 18, 26 m


def level_of(site):
    return SITES[site].get("level", "LV4")


# ----------------------------------------------------------------------------- extraction
def nearest_points(level, sites):
    """Nearest rho, u and v grid indices (j, i) for each site, as knnsearch on (lon, lat) in MATLAB."""
    out = {}
    with netCDF4.Dataset(GRIDS[level]) as g:
        grids = {p: (g[f"lon_{p}"][:], g[f"lat_{p}"][:]) for p in ("rho", "u", "v")}
        angle = g["angle"][:]
        for s in sites:
            lon, lat = SITES[s]["lon"], SITES[s]["lat"]
            ij = {p: np.unravel_index(np.argmin((x - lon) ** 2 + (y - lat) ** 2), x.shape)
                  for p, (x, y) in grids.items()}
            jr, ir = ij["rho"]
            out[s] = dict(ij=ij, angle=float(angle[jr, ir]),
                          lon_model=float(grids["rho"][0][jr, ir]), lat_model=float(grids["rho"][1][jr, ir]))
    return out


def history_files(level):
    """History files, only the standard LV4_ocean_his_<YYYYMMDDHHMM>.nc names: failed runs such as
    LV4_ocean_his_202610110000_dt2_BLOWUP.nc share the date and would overwrite the good run."""
    pat = re.compile(rf"{level}_ocean_his_\d{{12}}\.nc$")
    return sorted(f for f in glob.glob(os.path.join(BASE_DATA, f"{level}_His", f"{level}_ocean_his_*.nc"))
                  if pat.search(os.path.basename(f)))


def file_date(name):
    return pd.Timestamp(re.findall(r"\d{8}", name)[-1], tz="UTC")


def read_file(path, points, forecast_day, last_day=None):
    """Forecast days forecast_day..last_day (default: just forecast_day) of one history file at every
    site: {site: {var: array}}, time."""
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        nt = len(ds["ocean_time"])
        t1, t2 = 24 * (forecast_day - 1), min(24 * (last_day or forecast_day), nt)
        if t1 >= nt:
            return None
        t = (TIME0 + pd.to_timedelta(ds["ocean_time"][t1:t2], unit="s")).round("h")
        out = {}
        for s, p in points.items():
            jr, ir = p["ij"]["rho"]
            ju, iu = p["ij"]["u"]
            jv, iv = p["ij"]["v"]
            out[s] = {"u": ds["u"][t1:t2, :, ju, iu], "v": ds["v"][t1:t2, :, jv, iv],
                      "temp": ds["temp"][t1:t2, :, jr, ir], "salt": ds["salt"][t1:t2, :, jr, ir],
                      "zeta": ds["zeta"][t1:t2, jr, ir]}
    return t, out


def static_fields(path, points):
    """S-coordinate parameters and depth at each site (zeta-independent)."""
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        vt = int(ds["Vtransform"][...]) if "Vtransform" in ds.variables else 2
        if vt != 2:
            raise ValueError(f"Vtransform = {vt}; only 2 is implemented (as in the MATLAB code)")
        common = {k: np.asarray(ds[k][:], float) for k in ("s_rho", "s_w", "Cs_r", "Cs_w")}
        common["hc"] = float(ds["hc"][...])
        return {s: dict(common, h=float(ds["h"][p["ij"]["rho"]])) for s, p in points.items()}


def tag(forecast_day):
    """'day1' ... 'day5', or 'forecast' for days 2-5 of the latest history file."""
    return forecast_day if forecast_day == "forecast" else f"day{forecast_day}"


def profiles_path(site, level, forecast_day):
    return os.path.join(MODEL_DIR, f"{site}_{level}_{tag(forecast_day)}_profiles.nc")


def load_profiles(path):
    """Saved extraction as (time index, {var: array}, attrs), or None."""
    if not os.path.exists(path):
        return None
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        t = TIME0 + pd.to_timedelta(ds["time"][:], unit="s")
        data = {v: ds[v][:] for v in VARS}
        attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
        attrs.update({k: ds[k][:] for k in ("s_rho", "s_w", "Cs_r", "Cs_w")})
    return pd.DatetimeIndex(t), data, attrs


def save_profiles(path, t, data, attrs):
    tmp = path + ".part"
    with netCDF4.Dataset(tmp, "w") as ds:
        ds.createDimension("time", len(t))
        ds.createDimension("s_rho", data["temp"].shape[1])
        ds.createDimension("s_w", data["temp"].shape[1] + 1)
        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = "seconds since 1999-01-01 00:00:00"
        tv[:] = (t - TIME0).total_seconds().values
        for v, units in (("u", "m/s, grid xi direction"), ("v", "m/s, grid eta direction"),
                         ("temp", "degC"), ("salt", "psu")):
            x = ds.createVariable(v, "f4", ("time", "s_rho"), zlib=True, fill_value=np.nan)
            x.units = units
            x[:] = data[v]
        x = ds.createVariable("zeta", "f4", ("time",), zlib=True, fill_value=np.nan)
        x.units = "m"
        x[:] = data["zeta"]
        for k in ("s_rho", "s_w"):
            ds.createVariable(k, "f8", (k,))[:] = attrs[k]
        ds.createVariable("Cs_r", "f8", ("s_rho",))[:] = attrs["Cs_r"]
        ds.createVariable("Cs_w", "f8", ("s_w",))[:] = attrs["Cs_w"]
        for k, v in attrs.items():
            if k not in ("s_rho", "s_w", "Cs_r", "Cs_w"):
                ds.setncattr(k, v)
    os.replace(tmp, path)


def merge(old, t_new, new):
    """Append new hours to old, sort, keep the last of any duplicate times (append_to_matfile)."""
    if old is not None:
        t_old, d_old = old
        t_new = t_old.append(t_new)
        new = {v: np.concatenate([d_old[v], new[v]]) for v in VARS}
    order = np.argsort(t_new.values, kind="stable")
    t_new = t_new[order]
    new = {v: a[order] for v, a in new.items()}
    keep = ~t_new.duplicated(keep="last")
    return t_new[keep], {v: a[keep] for v, a in new.items()}


def extract(sites, level, forecast_day, start, end, workers, batch=48):
    data_folder = os.path.join(BASE_DATA, f"{level}_His")
    files = history_files(level)
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    points = nearest_points(level, sites)

    # Resume from the last saved day (re-read that day), per site; read from the earliest of them
    saved, resume = {}, {}
    for s in sites:
        p = load_profiles(profiles_path(s, level, forecast_day))
        saved[s] = p
        resume[s] = p[0].max().floor("D") if p is not None and len(p[0]) else start
    first = min(resume.values())
    todo = [f for f in files if first <= file_date(f) <= end]
    have = {file_date(f) for f in files}
    missing = [d for d in pd.date_range(first, end, freq="D") if d not in have]
    print(f"{len(todo)} history files to read ({first:%Y-%m-%d} to {end:%Y-%m-%d}), {len(missing)} days missing"
          + (f": {', '.join(f'{d:%Y-%m-%d}' for d in missing[:20])}{' ...' if len(missing) > 20 else ''}" if missing else ""))
    if not todo:
        return
    static = static_fields(todo[0], points)
    acc = {s: (saved[s][0], saved[s][1]) if saved[s] is not None else None for s in sites}

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for b in range(0, len(todo), batch):
            chunk = todo[b:b + batch]
            results = list(pool.map(read_file, chunk, [points] * len(chunk), [forecast_day] * len(chunk)))
            for s in sites:
                parts = [(t, out[s]) for f, r in zip(chunk, results) if r is not None
                         for t, out in [r] if file_date(f) >= resume[s]]
                if not parts:
                    continue
                t_new = pd.DatetimeIndex(np.concatenate([t.values for t, _ in parts])).tz_localize("UTC")
                new = {v: np.concatenate([o[v] for _, o in parts]).astype(np.float32) for v in VARS}
                for v in VARS:  # ROMS fill values (land/dry) to NaN
                    new[v][np.abs(new[v]) > 1e30] = np.nan
                t_all, d_all = merge(acc[s], t_new, new)
                acc[s] = (t_all, d_all)
                attrs = dict(static[s], site=s, level=level, forecast_day=forecast_day,
                             angle=points[s]["angle"], lon=SITES[s]["lon"], lat=SITES[s]["lat"],
                             lon_model=points[s]["lon_model"], lat_model=points[s]["lat_model"],
                             source=data_folder)
                save_profiles(profiles_path(s, level, forecast_day), t_all, d_all, attrs)
            done = b + len(chunk)
            rate = (time.time() - t0) / done
            print(f"  {done}/{len(todo)} files, last {os.path.basename(chunk[-1])}, "
                  f"{rate:.1f} s/file, ~{rate * (len(todo) - done) / 60:.0f} min left", flush=True)


def extract_forecast(sites, level, end):
    """Days 2-5 of the latest history file up to end (UTC date): the model forecast beyond the day-1
    series. Saved to cache/model/<site>_<level>_forecast_profiles.nc, replaced on every run."""
    files = [f for f in history_files(level) if file_date(f) <= pd.Timestamp(end, tz="UTC")]
    if not files:
        return
    points = nearest_points(level, sites)
    r = read_file(files[-1], points, 2, 5)
    if r is None:
        print(f"forecast: {os.path.basename(files[-1])} has only day 1")
        return
    t, out = r
    static = static_fields(files[-1], points)
    t = pd.DatetimeIndex(t)
    t = t.tz_localize("UTC") if t.tz is None else t
    for s in sites:
        d = {v: np.asarray(out[s][v], np.float32) for v in VARS}
        for v in VARS:  # ROMS fill values (land/dry) to NaN
            d[v][np.abs(d[v]) > 1e30] = np.nan
        attrs = dict(static[s], site=s, level=level, forecast_day="2-5", issued=f"{file_date(files[-1]):%Y-%m-%d}",
                     angle=points[s]["angle"], lon=SITES[s]["lon"], lat=SITES[s]["lat"],
                     lon_model=points[s]["lon_model"], lat_model=points[s]["lat_model"], source=files[-1])
        save_profiles(profiles_path(s, level, "forecast"), t, d, attrs)
    print(f"forecast: {os.path.basename(files[-1])}, {t[0]:%Y-%m-%d %H:%M} to {t[-1]:%Y-%m-%d %H:%M}")


# ----------------------------------------------------------------------------- export
def z_levels(a, zeta):
    """ROMS Vtransform 2 depths (m, negative down, relative to mean sea level): z_rho [nt, N], z_w [nt, N+1]."""
    h, hc = a["h"], a["hc"]
    z0r = (hc * a["s_rho"] + h * a["Cs_r"]) / (hc + h)
    z0w = (hc * a["s_w"] + h * a["Cs_w"]) / (hc + h)
    zeta = zeta[:, None]
    return zeta + (zeta + h) * z0r[None, :], zeta + (zeta + h) * z0w[None, :]


def layer_mean(x, zw, ref, top, bottom):
    """Mean of x [nt, N] over depths top..bottom below ref [nt], each layer weighted by how much of it
    lies in the range. Equivalent to SBOO_Model_DataExtraction's sidelobe/blanking mask + boundary
    interpolation + dz-weighted mean, but exact for ROMS layers (piecewise constant between w levels)."""
    d_top, d_bot = ref[:, None] - zw[:, 1:], ref[:, None] - zw[:, :-1]  # layer upper/lower depth, + down
    lo = top
    hi = d_bot[:, :1] if bottom is None else np.minimum(bottom, d_bot[:, :1])  # never below the sea floor
    w = np.clip(np.minimum(d_bot, hi) - np.maximum(d_top, lo), 0, None)
    w = np.where(np.isnan(x), 0, w)
    with np.errstate(invalid="ignore"):
        return np.nansum(x * w, axis=1) / np.where(w.sum(axis=1) > 0, w.sum(axis=1), np.nan)


def at_depth(x, zr, depth):
    """x interpolated to depth m below mean sea level, as in Model_vs_Observation_comparison_temp_sal.m
    (interp1(zrho, temp, -depth)): the mooring T/S sensors hang at fixed depths and do not follow
    the tide. The top level is used if depth is less than 0.5 m above it (at low tide the top
    level can sit below 1 m), NaN if deeper than the deepest level."""
    d = -zr  # depth below mean sea level, decreasing with level index
    out = np.full(len(x), np.nan)
    for k in range(len(x)):
        dk = d[k]
        if not np.all(np.isfinite(dk)) or depth < dk.min() - 0.5 or depth > dk.max():
            continue
        out[k] = np.interp(max(depth, dk.min()), dk[::-1], x[k, ::-1])
    return out


def export(site, level, forecast_day):
    p = load_profiles(profiles_path(site, level, forecast_day))
    if p is None:
        print(f"{site}: no extraction yet, skipping export")
        return
    t, d, a = p
    cfg = SITES[site]
    zr, zw = z_levels(a, d["zeta"])
    ang = float(a["angle"])
    ue = d["u"] * np.cos(ang) - d["v"] * np.sin(ang)
    vn = d["u"] * np.sin(ang) + d["v"] * np.cos(ang)
    ref = np.zeros(len(t)) if cfg["z_ref"] == "msl" else d["zeta"]
    out = pd.DataFrame({"time": t.tz_localize(None).strftime("%Y-%m-%dT%H:%M:%S")})
    for name, rng in (("davg", cfg["davg"]), ("top", cfg["top"])):
        out[f"u_{name}"] = layer_mean(ue, zw, ref, *rng)
        out[f"v_{name}"] = layer_mean(vn, zw, ref, *rng)
    for dd in cfg.get("ts_depths", TS_DEPTHS):
        out[f"temp_{dd}m"] = at_depth(d["temp"], zr, dd)
        out[f"sal_{dd}m"] = at_depth(d["salt"], zr, dd)
    path = os.path.join(MODEL_DIR, f"{site}_{level}_{tag(forecast_day)}_hourly.csv")
    out.to_csv(path, index=False, float_format="%.5g")
    depth = float(np.nanmedian(zw[:, -1] - zw[:, 0]))
    print(f"{site}: {t[0]:%Y-%m-%d %H:%M} to {t[-1]:%Y-%m-%d %H:%M}, water depth {depth:.1f} m, "
          f"{100 * np.isfinite(out['u_davg']).mean():.0f}% hours valid -> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sites", nargs="+", choices=SITES, default=list(SITES))
    p.add_argument("--start", default="2024-12-05", help="UTC start date (first run only; later runs resume)")
    p.add_argument("--end", default=f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d}", help="UTC end date, inclusive")
    p.add_argument("--level", default=None, choices=GRIDS, help="default: each site's own level")
    p.add_argument("--forecast-day", type=int, default=1, choices=range(1, 6))
    p.add_argument("--workers", type=int, default=6, help="history files read in parallel")
    p.add_argument("--export-only", action="store_true")
    args = p.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    levels = {}
    for s in args.sites:  # one pass over the history files per model level
        levels.setdefault(args.level or level_of(s), []).append(s)
    for level, sites in levels.items():
        if not args.export_only:
            extract(sites, level, args.forecast_day, args.start, args.end, args.workers)
        for s in sites:
            export(s, level, args.forecast_day)


if __name__ == "__main__":
    main()
