#!/usr/bin/env python3
"""
fetch_hfr_nrt.py — near-real-time HF radar for one DA window.

HFR is the ONE input PFM_DA has to fetch itself.  Everything else (atm, ocean BC,
the initial condition) is produced by the forecast cycle and is already on disk;
PFM does not use HF radar at all, so nothing fetches it for us.  That makes
downloading here the normal path, not the exception it is for ECMWF and HYCOM.

It still only fetches the GAP: any hour already cached is skipped, so a re-run
after a partial failure costs only the missing hours.

OUTPUT SCHEMA -- deliberately NCEI's, not the NRT feed's.
`make_obs_from_hfr.py` already reads the NCEI retrospective cache: one file per
hour, variables `lat`/`lon`/`u`/`v`, `u` and `v` shaped (1, lat, lon), int16
packed with scale_factor=0.01 and _FillValue=-32767.  NDBC's aggregation uses the
same variable names and the same packing (verified 2026-10-06), differing only in
that its `u`/`v` span many times instead of one.  So this writes per-hour files in
NCEI's exact layout and the whole downstream obs pipeline -- the 3x3 stencil, the
Helmholtz decomposition, the coast and boundary culls, the 1-in-2 thinning, the
error ellipse -- runs unchanged.  That pipeline is the carefully tuned part; it
would be a poor trade to perturb it for the sake of a nicer file format.

SOURCES.  NDBC OPeNDAP is primary: all resolutions sit at the same newest hour,
the names and packing already match, and subsetting one day to the LV1 box takes
a couple of seconds.  CoastWatch ERDDAP serves the same product and is the
fallback -- it is flakier (individual resolutions 404 for minutes while ERDDAP
reloads) and its resolutions often sit at different hours.  Both of those
behaviours, and the retry/backoff they need, were established in
roms_fourdvar/LV3_direct/ipynb_files/explore_hfr_nearrealtime.ipynb.

    python fetch_hfr_nrt.py 2026-10-06 --res 2km --outdir DIR
"""
from __future__ import annotations

import argparse
import datetime as DT
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import netCDF4 as nc

EPOCH = DT.datetime(1970, 1, 1)

NDBC_URL = "https://dods.ndbc.noaa.gov/thredds/dodsC/hfradar_uswc_{res}"
ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
ERDDAP_ID = {"1km": "ucsdHfrW1", "2km": "ucsdHfrW2", "6km": "ucsdHfrW6"}

# LV1 plus a margin.  make_obs_from_hfr culls to LV1's polygon + a 12 km buffer
# itself, so this only has to be generous enough not to clip that buffer.
LAT_LO, LAT_HI = 28.0, 37.0
LON_LO, LON_HI = -124.5, -115.0

# 25 hourly surveys: start 00Z through next-day 00Z inclusive, matching the DA
# window and make_obs_from_hfr's expectation.
WINDOW_HOURS = 24

FILL_I2 = -32767
SCALE = 0.01

# An hour can EXIST upstream with nothing in it: NDBC publishes the timestamp
# before the radials are assembled.  Measured 2026-10-07, 6 km's newest hour was
# 0.0% covered and the one before it 8.8% against a ~21% steady state.  Writing
# such an hour would make it count as cached, pass the survey-count test, and
# contribute no observations -- so it is reported MISSING instead, and a later run
# picks it up once it has filled.  Well below both products' steady state so a
# genuinely sparse hour still counts.
MIN_COVERAGE = 0.005


def hour_filename(t: DT.datetime, res: str) -> str:
    """NCEI-style name, so make_obs_from_hfr's `*_s{stamp}_*.nc` glob finds it.

    'nrt' replaces NCEI's 'uwls' processing tag so the provenance of a file is
    visible from its name -- near-real-time and retrospective data for the same
    hour are not identical, and when something looks odd it matters which was
    used.
    """
    stamp = t.strftime("%Y%m%d%H%M%S") + "0"
    return f"rtv-uswc-{res}-nrt_v1r0_hfr_s{stamp}_e{stamp}_c{stamp}.nc"


def already_cached(t: DT.datetime, caches: list[Path], res: str) -> Path | None:
    """Any cache already holding this hour AT THIS RESOLUTION.

    Globs the same way make_obs_from_hfr does -- including the resolution token --
    so 'cached' here means exactly 'the obs builder will find this hour'.  Without
    the token a 2 km file satisfied a 6 km fetch: the fetch reported 25 hours
    already cached, downloaded nothing, and the obs builder then quietly used the
    2 km data.
    """
    stamp = t.strftime("%Y%m%d%H%M%S") + "0"
    for c in caches:
        hits = sorted(c.glob(f"*-{res}-*_s{stamp}_*.nc")) if c.is_dir() else []
        if hits:
            return hits[0]
    return None


def _write_hour(out: Path, t: DT.datetime, lat: np.ndarray, lon: np.ndarray,
                u: np.ndarray, v: np.ndarray, source: str) -> None:
    """One hour, in NCEI's layout: u/v int16 (1, lat, lon), scale 0.01.

    Written to a .tmp and renamed, so an interrupted fetch cannot leave a
    half-written file that the obs builder would happily read.
    """
    tmp = out.with_suffix(".tmp")
    with nc.Dataset(tmp, "w", format="NETCDF4") as d:
        d.createDimension("time", 1)
        d.createDimension("lat", lat.size)
        d.createDimension("lon", lon.size)
        tv = d.createVariable("time", "f8", ("time",))
        tv.units = "seconds since 1970-01-01"
        tv[:] = [(t - EPOCH).total_seconds()]
        d.createVariable("lat", "f4", ("lat",))[:] = lat
        d.createVariable("lon", "f4", ("lon",))[:] = lon
        for nm, arr in (("u", u), ("v", v)):
            var = d.createVariable(nm, "i2", ("time", "lat", "lon"),
                                   fill_value=np.int16(FILL_I2), zlib=True,
                                   complevel=4)
            var.scale_factor = SCALE
            var.units = "m s-1"
            # Assign PHYSICAL values as a masked array and let netCDF4 pack them.
            #
            # Do NOT hand-pack to int16 here.  netCDF4-python's
            # auto-maskandscale applies scale_factor on WRITE as well as read, so
            # pre-packed integers get divided by 0.01 a SECOND time -- and the
            # -32767 sentinel is rescaled with them, destroying the missing-data
            # mask.  That bug produced files reading -81..56 m/s at 100% finite,
            # against a real NCEI file's -0.88..0.74 m/s at 8.9% finite.  Both
            # symptoms were visible only by looking at the values; the file count
            # and the schema were perfect.
            var[0] = np.ma.masked_invalid(arr)
        d.source = source
        d.note = ("near-real-time HFR written in the NCEI retrospective layout "
                  "so make_obs_from_hfr.py reads it unchanged")
    tmp.replace(out)


# ---------------------------------------------------------------------------
# NDBC (primary)
# ---------------------------------------------------------------------------
_NDBC: dict = {}


def ndbc_meta(res: str) -> dict:
    """Open the NDBC aggregation once; keep its axes and the LV1 slice.

    The full grid is large (2 km is 1099 x 700) but OPeNDAP ships only the box we
    ask for, so the cost is this one-off metadata read.
    """
    if res not in _NDBC:
        d = nc.Dataset(NDBC_URL.format(res=res))
        lat = np.asarray(d.variables["lat"][:], float)
        lon = np.asarray(d.variables["lon"][:], float)
        tv = d.variables["time"]
        times = np.array([x.replace(microsecond=0) for x in nc.num2date(
            np.asarray(tv[:], float), tv.units,
            only_use_cftime_datetimes=False)])
        jj = np.where((lat >= LAT_LO) & (lat <= LAT_HI))[0]
        ii = np.where((lon >= LON_LO) & (lon <= LON_HI))[0]
        if not jj.size or not ii.size:
            raise RuntimeError(
                f"the LV1 box ({LAT_LO}..{LAT_HI} N, {LON_LO}..{LON_HI} E) does "
                f"not intersect the NDBC {res} grid")
        _NDBC[res] = dict(ds=d, times=times, lat=lat[jj], lon=lon[ii],
                          jsl=slice(int(jj[0]), int(jj[-1]) + 1),
                          isl=slice(int(ii[0]), int(ii[-1]) + 1))
    return _NDBC[res]


def fetch_hour_ndbc(t: DT.datetime, res: str) -> tuple | None:
    M = ndbc_meta(res)
    k = np.where(M["times"] == t)[0]
    if not k.size:
        return None
    i = int(k[0])
    ds = M["ds"]
    u = np.ma.filled(ds.variables["u"][i, M["jsl"], M["isl"]].astype(float),
                     np.nan)
    v = np.ma.filled(ds.variables["v"][i, M["jsl"], M["isl"]].astype(float),
                     np.nan)
    return M["lat"], M["lon"], u, v


# ---------------------------------------------------------------------------
# CoastWatch ERDDAP (fallback)
# ---------------------------------------------------------------------------
def _get(url: str, tries: int = 6, pause: int = 20, timeout: int = 300) -> bytes:
    """GET with growing backoff.

    ERDDAP reloads these near-real-time datasets constantly, and a request
    landing mid-reload returns 404 ('Currently unknown datasetID') or 500
    ('partialResults[0] was not as expected').  Both are transient, but a single
    dataset can also stay down for minutes.
    """
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            if k < tries - 1:
                time.sleep(pause * (1 + k // 3))
    raise RuntimeError(f"{type(last).__name__}: {last}  <- {url[:110]}")


def fetch_hour_erddap(t: DT.datetime, res: str) -> tuple | None:
    # ERDDAP does NOT clamp time constraints: asking for an hour it does not have
    # yet returns a 404 that reads exactly like a mid-reload failure.
    sel = (f"%5B({t:%Y-%m-%d}T{t:%H}:00:00Z):1:({t:%Y-%m-%d}T{t:%H}:00:00Z)%5D"
           f"%5B({LAT_LO:.4f}):1:({LAT_HI:.4f})%5D"
           f"%5B({LON_LO:.4f}):1:({LON_HI:.4f})%5D")
    blob = _get(f"{ERDDAP}/{ERDDAP_ID[res]}.nc?water_u{sel},water_v{sel}")
    tmp = Path("/tmp") / f".hfr_erddap_{res}_{t:%Y%m%d%H}.nc"
    tmp.write_bytes(blob)
    try:
        with nc.Dataset(tmp) as d:
            lat = np.asarray(d.variables["latitude"][:], float)
            lon = np.asarray(d.variables["longitude"][:], float)
            u = np.ma.filled(d.variables["water_u"][0].astype(float), np.nan)
            v = np.ma.filled(d.variables["water_v"][0].astype(float), np.nan)
    finally:
        tmp.unlink(missing_ok=True)
    return lat, lon, u, v


# ---------------------------------------------------------------------------
def fetch_window(start_date: str, res: str, outdir: Path,
                 caches: list[Path] | None = None,
                 source: str = "ndbc") -> dict:
    """Cache the 25 hourly surveys for [start 00Z, next-day 00Z].

    Returns {'present', 'fetched', 'missing'} -- counts, plus the missing hours.
    A missing hour is NOT fatal here: make_obs_from_hfr tolerates gaps (the NCEI
    catalog has permanent ones), and a DA cycle with 23 of 25 surveys is still a
    good cycle.  The caller decides how many gaps are too many.
    """
    t0 = DT.datetime.strptime(start_date, "%Y-%m-%d")
    hours = [t0 + DT.timedelta(hours=h) for h in range(WINDOW_HOURS + 1)]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    caches = [outdir] + list(caches or [])

    present, fetched, missing, coverage = 0, 0, [], []
    for t in hours:
        hit = already_cached(t, caches, res)
        if hit is not None:
            present += 1
            continue
        try:
            got = (fetch_hour_ndbc(t, res) if source == "ndbc"
                   else fetch_hour_erddap(t, res))
        except Exception as e:  # noqa: BLE001
            print(f"    {t:%Y-%m-%d %H}Z  {source} FAILED: "
                  f"{type(e).__name__}: {str(e)[:70]}")
            _NDBC.pop(res, None)        # drop a stale handle so the next hour reopens
            got = None
        if got is None and source == "ndbc":
            # fall back to ERDDAP for this hour only
            try:
                got = fetch_hour_erddap(t, res)
                if got is not None:
                    print(f"    {t:%Y-%m-%d %H}Z  via ERDDAP fallback")
            except Exception as e:  # noqa: BLE001
                print(f"    {t:%Y-%m-%d %H}Z  ERDDAP fallback FAILED: "
                      f"{str(e)[:70]}")
                got = None
        if got is None:
            missing.append(t)
            continue
        lat, lon, u, v = got
        frac = float(np.isfinite(u).mean())
        if frac < MIN_COVERAGE:
            print(f"    {t:%Y-%m-%d %H}Z  only {100*frac:.2f}% covered "
                  f"(< {100*MIN_COVERAGE:.1f}%) -- treating as NOT YET AVAILABLE "
                  f"rather than caching an empty hour")
            missing.append(t)
            continue
        _write_hour(outdir / hour_filename(t, res), t, lat, lon, u, v, source)
        fetched += 1
        coverage.append((t, frac))

    if coverage:
        fr = [f for _, f in coverage]
        print(f"    coverage of the fetched hours: "
              f"{100*min(fr):.1f}%..{100*max(fr):.1f}% of the LV1 box "
              f"(mean {100*sum(fr)/len(fr):.1f}%)")
    return {"present": present, "fetched": fetched, "missing": missing,
            "n_hours": len(hours), "coverage": coverage}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("start_date", help="YYYY-MM-DD, the 00Z start of the window")
    ap.add_argument("--res", default="2km", choices=sorted(ERDDAP_ID))
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--also-cache", action="append", default=[], type=Path,
                    help="another cache to treat as already-having an hour "
                         "(repeatable); the retrospective NCEI cache belongs here")
    ap.add_argument("--source", default="ndbc", choices=("ndbc", "coastwatch"))
    a = ap.parse_args()
    r = fetch_window(a.start_date, a.res, a.outdir, a.also_cache, a.source)
    print(f"  HFR {a.res} {a.start_date}: {r['present']} already cached, "
          f"{r['fetched']} fetched, {len(r['missing'])} missing "
          f"of {r['n_hours']} hours")
    for t in r["missing"]:
        print(f"    missing: {t:%Y-%m-%d %H}Z")
    return 0 if not r["missing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
