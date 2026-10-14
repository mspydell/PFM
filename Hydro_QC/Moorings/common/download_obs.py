#!/usr/bin/env python3
"""Download and cache mooring observations (SBOO, IB mini moorings, PLOO, Del Mar) from the SIO Ocean
Time-Series Group.

Source: https://mooring.ucsd.edu/sboo/, /mini_moorings/, /ploo/ and /delmar1/ (Jeff Sevadjian).
PLOO and Del Mar (delmar1): only deployments from FIRST_DEP on (the model starts 2024-12-05; each
ADCP file is ~400 MB). Their T/S are temp.csv/sal.csv at 9 depths (which vary a little between
deployments); ADCP <DEP>_R_ADCP.nc at fixed bin depths below the mean surface, downward looking
(PLOO 1 m bins from 3.43 m, 10 min; Del Mar 2 m bins from 5.42 m, 15 min, with UCUR_QC/VCUR_QC: only
flag 1 "good" is kept, as Jeff does; no other filtering. SIO reprocessed delmar1_22 on 2026-10-14 so its
flags remove the corrupt even-hour :15/:30 pings; delmar1_20/21 were not, and their flags miss them).
Every deployment folder (sboo_01 ... sboo_08, ib-c_01 ... ib-c_06, ...) is mirrored to
cache/obs/<site>/<deployment>/. Files are only re-downloaded when the server copy is newer
(If-Modified-Since), so daily runs just refresh the active deployments.

Notes from Jeff (Oct 2026):
  T/S: calibrated against bottle samples but not yet QC'd for outliers (QC flags planned);
    load_ts drops values outside TS_RANGE in the meantime.
    Only available per deployment: <deployment>/csv/temp.csv and sal.csv, used here for all
    sites. (IB <dep>_data_merged.csv has the same data but mislabels depths, e.g. 9 m instead
    of 8 m in ib-c_02; ib-c_01 has a single sensor with no depth, in TS.csv, and is skipped.)
    mini_moorings/ib/csv/{temp,sal}_{01m,10m}.csv merge all four IB sites, but only from
    Feb 2026; their "10m" is really 8 m at IB-C and IB-N, so they are not used here.
  ADCP: east/north are corrected for magnetic declination and QC'd inside the instrument.
    IB: <deployment>/nc/*.nc with CUR_MASK flagging bins contaminated by the sea surface
    (masked bins are dropped). SBOO: <deployment>/nc/SBOO_<nn>_R_ADCP.nc, 10-min samples, as in
    SBOO_Observed_DataExtraction.m. No sea-floor flags yet, so bins deeper than SBOO_SIDELOBE
    (bottom 8% of the water column) are dropped and the first of them is replaced by the value
    interpolated to SBOO_SIDELOBE, as in the MATLAB code. (SIO's adcpEast-01hAvg.csv is centred on
    the hour too but has extra processing: r = 0.98 against the nc averaged this way, so it is not used.)

  python download_obs.py                 # refresh all sites
  python download_obs.py --sites SBOO IB-C

From other scripts:
  from download_obs import load_ts, load_adcp
  ts = load_ts("IB-N", "2026-09-15", "2026-10-05")   # time, depth_m, var ('temp'|'salt'), value
  uv = load_adcp("SBOO", "2026-09-15", "2026-10-05") # time, z_m, z_ref, u_east, v_north
"""
import argparse
import email.utils
import glob
import os
import re
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from scipy.io import netcdf_file

BASE = "https://mooring.ucsd.edu"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUT_DIR, "cache", "obs")

# site -> (index URL holding the deployment folders, deployment prefix, lat, lon)
SITES = {
    "SBOO": (f"{BASE}/sboo/",               "sboo", 32.53187,  -117.18619),  # from sboo_08 ADCP nc
    "IB-N": (f"{BASE}/mini_moorings/ib-n/", "ib-n", 32.615222, -117.140231),
    "IB-C": (f"{BASE}/mini_moorings/ib-c/", "ib-c", 32.579008, -117.141815),
    "IB-W": (f"{BASE}/mini_moorings/ib-w/", "ib-w", 32.581562, -117.162993),
    "IB-S": (f"{BASE}/mini_moorings/ib-s/", "ib-s", 32.539694, -117.151040),
    "PLOO": (f"{BASE}/ploo/",               "ploo", 32.67042,  -117.325015),  # from ploo_07 ADCP nc
    "DelMar": (f"{BASE}/delmar1/",          "delmar1", 32.9295, -117.31665),  # from delmar1_22 ADCP nc
}
FIRST_DEP = {"PLOO": "ploo_05", "DelMar": "delmar1_20"}  # earlier deployments end before the model starts
# ADCP netCDF per deployment; SBOO: skip the 6-h averages. IB: any .nc in the folder
ADCP_NC = {"SBOO": r"SBOO_\d+_R_ADCP\.nc", "PLOO": r"PLOO_\d+_R_ADCP\.nc", "DelMar": r"DELMAR1_\d+_R_ADCP\.nc"}
SBOO_DEPTH = 31.48                 # m, water depth at SBOO
SBOO_SIDELOBE = 0.92 * SBOO_DEPTH  # deeper bins are sea-floor contaminated
TS_RANGE = pd.DataFrame({"min": {"temp": 5.0, "salt": 20.0}, "max": {"temp": 30.0, "salt": 36.0}})  # plausible values


TS_FILES = ["temp.csv", "sal.csv"]  # same names at every site and deployment


def list_index(url, pattern):
    """Names in an Apache directory listing that match pattern."""
    with urllib.request.urlopen(url, timeout=60) as r:
        html = r.read().decode("utf-8", "replace")
    return sorted(set(re.findall(rf'href="(?:\./)?({pattern})/?"', html)))  # delmar1 links as ./delmar1_22


def fetch(url, path):
    """Download url to path unless the local copy is current. Returns 'new', 'same' or 'missing'."""
    req = urllib.request.Request(url)
    if os.path.exists(path):
        req.add_header("If-Modified-Since", email.utils.formatdate(os.path.getmtime(path), usegmt=True))
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
            modified = r.headers.get("Last-Modified")
    except urllib.error.HTTPError as e:
        if e.code == 304:
            return "same"
        if e.code == 404:
            return "missing"
        raise
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    if modified:  # stamp with the server time so If-Modified-Since works next run
        t = email.utils.parsedate_to_datetime(modified).timestamp()
        os.utime(path, (t, t))
    return "new"


def update_site(site):
    index, prefix, _, _ = SITES[site]
    deps = [d for d in list_index(index, rf"{re.escape(prefix)}_\d+") if d >= FIRST_DEP.get(site, "")]
    for dep in deps:
        urls = {f: f"{index}{dep}/csv/{f}" for f in TS_FILES}
        # ADCP netCDF names carry the deployment number or dates, so list the folder
        pattern = ADCP_NC.get(site, r"[^\"/?]+\.nc")
        try:
            ncs = list_index(f"{index}{dep}/nc/", pattern)
        except urllib.error.HTTPError:
            ncs = []  # deployment without an ADCP
        urls.update({f: f"{index}{dep}/nc/{f}" for f in ncs})
        status = {f: fetch(u, os.path.join(CACHE_DIR, site, dep, f)) for f, u in urls.items()}
        got = [f for f, s in status.items() if s == "new"]
        # the server renames ADCP files as a deployment grows (..._END<date>_RT.nc) or is reprocessed;
        # drop local copies it no longer lists, else the loader would read old and new data together
        gone = [f for f in os.listdir(os.path.join(CACHE_DIR, site, dep))
                if f.endswith(".nc") and f not in ncs] if os.path.isdir(os.path.join(CACHE_DIR, site, dep)) else []
        for f in gone:
            os.remove(os.path.join(CACHE_DIR, site, dep, f))
        print(f"{site:5s} {dep:8s} " + (f"updated {', '.join(got)}" if got else "up to date")
              + (f"; removed {', '.join(gone)}" if gone else ""))


def _parse_time(df):
    """UTC times from either a UnixTime (ms) column or GMT_Year..Sec columns."""
    if "GMT_Year" in df.columns:
        parts = df[["GMT_Year", "Month", "Day", "Hour", "Min", "Sec"]].astype(int)
        parts.columns = ["year", "month", "day", "hour", "minute", "second"]
        return pd.to_datetime(parts, utc=True)
    t = df.iloc[:, 0].astype("int64")
    return pd.to_datetime(t, unit="ms" if t.max() > 1e11 else "s", utc=True)


def _depth(col):
    m = re.search(r"_([\d.]+)m$", col)
    return float(m.group(1)) if m else None


def _deployment_files(site, name):
    """name may contain {dep} for the deployment name and glob wildcards."""
    for dep_dir in sorted(glob.glob(os.path.join(CACHE_DIR, site, "*"))):
        for path in sorted(glob.glob(os.path.join(dep_dir, name.format(dep=os.path.basename(dep_dir))))):
            if os.path.getsize(path) > 0:
                yield path


def _window(df, start, end):
    if start is not None:
        df = df[df["time"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df["time"] < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]  # end day inclusive
    return df


def load_ts(site, start=None, end=None):
    """Temperature and salinity in long form: time (UTC), depth_m, var ('temp'|'salt'), value.

    start/end are UTC dates (YYYY-MM-DD), end inclusive. Uses the cache only; run update first.
    """
    frames = []
    for name in TS_FILES:
        for path in _deployment_files(site, name):
            df = pd.read_csv(path)
            time = _parse_time(df)
            for col in df.columns:
                var = "temp" if col.startswith("T_") else "salt" if col.startswith("S_") else None
                depth = _depth(col)
                if var is None or depth is None:  # e.g. ib-c_01 has no sensor depth in its names
                    continue
                frames.append(pd.DataFrame({"time": time, "depth_m": depth, "var": var,
                                            "value": pd.to_numeric(df[col], errors="coerce")}))
    if not frames:
        return pd.DataFrame(columns=["time", "depth_m", "var", "value"])
    out = pd.concat(frames).dropna(subset=["value"])
    # Gross range check until the SIO QC flags exist (raw files hold e.g. 127466 degC, S = 0.04 or 68)
    lo, hi = out["var"].map(TS_RANGE["min"]), out["var"].map(TS_RANGE["max"])
    out = out[out["value"].between(lo, hi)]
    return _window(out, start, end).sort_values(["var", "depth_m", "time"]).reset_index(drop=True)


def _load_ib_adcp_nc(site):
    """IB ADCP netCDF: native-interval currents with surface-contaminated bins (CUR_MASK != 1) dropped.
    z_m is the bin depth below the moving sea surface."""
    frames = []
    for path in _deployment_files(site, "*.nc"):
        with netcdf_file(path, "r", mmap=False, maskandscale=True) as ds:
            v = {k: np.asarray(ds.variables[k][:], dtype=float) for k in ("TIME", "DEPTH", "UCUR", "VCUR", "CUR_MASK")}
        nz = v["UCUR"].shape[1]
        good = (v["CUR_MASK"] == 1) & np.isfinite(v["UCUR"]) & np.isfinite(v["VCUR"])
        time = pd.to_datetime("1950-01-01", utc=True) + pd.to_timedelta(v["TIME"], unit="D")
        frames.append(pd.DataFrame({"time": np.repeat(time.round("s"), nz)[good.ravel()],
                                    "z_m": v["DEPTH"].ravel()[good.ravel()],
                                    "u_east": v["UCUR"].ravel()[good.ravel()],
                                    "v_north": v["VCUR"].ravel()[good.ravel()]}))
    if not frames:
        return pd.DataFrame(columns=["time", "z_m", "z_ref", "u_east", "v_north"])
    out = pd.concat(frames).drop_duplicates(["time", "z_m"])
    out["z_ref"] = "depth"
    return out


def _load_sboo_adcp_nc():
    """SBOO ADCP netCDF: 10-min currents at fixed bin depths below the mean surface, with the
    side-lobe cut and boundary interpolation of SBOO_Observed_DataExtraction.m."""
    frames = []
    for path in _deployment_files("SBOO", "SBOO_*_R_ADCP.nc"):
        with netcdf_file(path, "r", mmap=False, maskandscale=True) as ds:
            v = {k: np.asarray(ds.variables[k][:], dtype=float) for k in ("TIME", "DEPTH", "UCUR", "VCUR")}
        time = (pd.to_datetime("1950-01-01", utc=True) + pd.to_timedelta(v["TIME"], unit="D")).round("s")
        keep = ~time.duplicated(keep="last")  # the nc repeats some timestamps
        z, u, w = v["DEPTH"], v["UCUR"][keep], v["VCUR"][keep]
        time = time[keep]
        below = np.flatnonzero(z > SBOO_SIDELOBE)
        if len(below) and below[0] > 0:  # first contaminated bin -> value interpolated to the cutoff
            k = below[0]
            f = (SBOO_SIDELOBE - z[k - 1]) / (z[k] - z[k - 1])
            u[:, k] = u[:, k - 1] + f * (u[:, k] - u[:, k - 1])
            w[:, k] = w[:, k - 1] + f * (w[:, k] - w[:, k - 1])
            z = z.copy()
            z[k] = SBOO_SIDELOBE
            below = below[1:]
        good = np.ones(len(z), bool)
        good[below] = False
        nz = good.sum()
        frames.append(pd.DataFrame({"time": np.repeat(time, nz), "z_m": np.tile(z[good], len(time)),
                                    "u_east": u[:, good].ravel(), "v_north": w[:, good].ravel()}))
    if not frames:
        return pd.DataFrame(columns=["time", "z_m", "z_ref", "u_east", "v_north"])
    out = pd.concat(frames).dropna(subset=["u_east", "v_north"], how="all")
    out["z_ref"] = "depth"
    return out


def _load_fixed_bin_adcp_nc(site):
    """PLOO / Del Mar ADCP netCDF: native-interval currents at fixed bin depths below the mean surface.
    Samples with fill times are dropped; where UCUR_QC exists only flag 1 (good) is kept. Bins are not
    cut here: plot_model_vs_obs keeps those in the depth range the model is averaged over."""
    frames = []
    for path in _deployment_files(site, "*_R_ADCP.nc"):
        with netcdf_file(path, "r", mmap=False, maskandscale=True) as ds:
            names = ("TIME", "DEPTH", "UCUR", "VCUR") + tuple(k for k in ("UCUR_QC", "VCUR_QC") if k in ds.variables)
            v = {k: np.asarray(ds.variables[k][:], dtype=float) for k in names}
        ok = np.isfinite(v["TIME"]) & (np.abs(v["TIME"]) < 1e6)
        time = (pd.to_datetime("1950-01-01", utc=True) + pd.to_timedelta(v["TIME"][ok], unit="D")).round("s")
        u, w = v["UCUR"][ok], v["VCUR"][ok]
        if "UCUR_QC" in v:  # Jeff's QC: ucur(ucur_qc ~= 1) = NaN; vcur(vcur_qc ~= 1) = NaN
            u = np.where(v["UCUR_QC"][ok] != 1, np.nan, u)
        if "VCUR_QC" in v:
            w = np.where(v["VCUR_QC"][ok] != 1, np.nan, w)
        keep = ~time.duplicated(keep="last")
        time, u, w = time[keep], u[keep], w[keep]
        nz = len(v["DEPTH"])
        frames.append(pd.DataFrame({"time": np.repeat(time, nz), "z_m": np.tile(v["DEPTH"], len(time)),
                                    "u_east": u.ravel(), "v_north": w.ravel()}))
    if not frames:
        return pd.DataFrame(columns=["time", "z_m", "z_ref", "u_east", "v_north"])
    out = pd.concat(frames).dropna(subset=["u_east", "v_north"], how="all")
    out["z_ref"] = "depth"
    return out


def load_adcp(site, start=None, end=None):
    """ADCP currents at native sampling: time (UTC), z_m, z_ref, u_east, v_north (m/s), suspect bins removed.

    SBOO: 10-min samples, z_m = fixed bin depth below the mean surface.
    IB sites: 30-min (older deployments hourly) samples, z_m = bin depth below the moving surface.
    z_ref is always 'depth'.
    """
    out = (_load_sboo_adcp_nc() if site == "SBOO" else _load_fixed_bin_adcp_nc(site) if site in ("PLOO", "DelMar")
           else _load_ib_adcp_nc(site))
    return _window(out, start, end).sort_values(["time", "z_m"]).reset_index(drop=True)[
        ["time", "z_m", "z_ref", "u_east", "v_north"]]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sites", nargs="+", choices=SITES, default=list(SITES), help="Sites to update (default: all)")
    args = p.parse_args()
    for site in args.sites:
        update_site(site)
    for site in args.sites:
        ts, uv = load_ts(site), load_adcp(site)
        span = lambda d: f"{d['time'].min():%Y-%m-%d} to {d['time'].max():%Y-%m-%d %H:%M}" if len(d) else "none"
        depths = sorted(float(d) for d in ts["depth_m"].unique())
        print(f"{site:5s} T/S {span(ts)} depths {depths} | ADCP {span(uv)}")


if __name__ == "__main__":
    main()
