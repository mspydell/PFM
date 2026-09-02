#!/usr/bin/env python3
"""
make_obs_qc_figures.py

Daily observation-QC figures for the LV3 domain.  Takes no arguments: it pulls
the current HF radar, mooring and wave data and writes three PNGs.

  1. HF radar over the whole LV3 domain, 3 x 3 panels ending at the newest hour
  2. the same panels zoomed to the shelf
  3. HF radar coverage over the last COV_DAYS days

Each panel also carries the SBOO and PLOO depth-integrated and surface currents
(hour-centred means) and the CDIP significant wave height and direction.

This is a direct lift of
  research/sdtjreL3/v_intel/roms_fourdvar/LV3_direct/ipynb_files/
  explore_hfr_nearrealtime.ipynb
with the notebook's settings unchanged.  See that notebook for why the data
comes from where it does -- in short, HFRNet's own THREDDS
(hfrnet-tds.ucsd.edu) is unreachable from this network and NCEI's archive runs
about a month behind, so the near-real-time HF radar comes from NDBC.

Run it:
    python3 make_obs_qc_figures.py

Cron (07:10 local, daily):
    10 7 * * *  /home/mspydell/anaconda3/envs/PHM-env/bin/python3 \
        /home/mspydell/models/PFM_root/PFM/qc_obs_py_files/make_obs_qc_figures.py \
        >> /scratch/PFM_Simulations/obs_qc_figures/make_obs_qc_figures.log 2>&1

Exit status is 0 on success and 1 on failure, so cron will mail on failure.
"""
from __future__ import annotations

import os
import sys
import time
import shutil
import datetime as DT
import traceback
from pathlib import Path

import matplotlib
matplotlib.use('Agg')          # must precede the pyplot import: no display here

# ======================= paths this script owns =======================
# Where the three figures land.
FIG_DIR = Path('/scratch/PFM_Simulations/obs_qc_figures')

# Downloaded data.  All three feeds share this directory:
#   HFR   hfr_nrt_{source}_{res}_{YYYYMMDD}.nc   ~4 MB/day, one per resolution
#   SBOO  SBOO_{nn}_R_ADCP.nc                    ~20 MB, whole file per refresh
#   PLOO  PLOO_{nn}_R_ADCP.nc                    ~305 MB, whole file per refresh
#   CDIP  cdip_{station}.npz                     ~1 MB
# mooring.ucsd.edu serves plain files with no OPeNDAP, so the mooring files can
# only be fetched whole -- PLOO is most of the daily transfer.
DATA_DIR = Path('/dataSIO/DA_Simulations/input_files/hfr_nrt')

# HFR day-files are never overwritten once written, so without this they grow
# without bound (~3 files, ~4 MB per day).
KEEP_DAYS = 60

# One run at a time.  A daily job that overruns must not race the next one:
# both would write the same day-file and the same figures.
LOCK_FILE = DATA_DIR / '.make_obs_qc_figures.lock'
LOCK_STALE_H = 6.0

# Besides the timestamped figures, keep a fixed-name copy of each so a
# dashboard or a symlink can point at "the current one".
WRITE_LATEST = True
# ======================================================================


def log(msg=''):
    if msg:
        print(f'[{DT.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}', flush=True)
    else:
        print(flush=True)


def acquire_lock():
    """Refuse to start if another run is in flight; break a stale lock."""
    if LOCK_FILE.exists():
        age_h = (time.time() - LOCK_FILE.stat().st_mtime) / 3600
        if age_h < LOCK_STALE_H:
            log(f'another run started {age_h:.1f} h ago '
                f'({LOCK_FILE}); exiting')
            return False
        log(f'breaking stale lock ({age_h:.1f} h old)')
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(f'{os.getpid()}\n{DT.datetime.now():%Y-%m-%d %H:%M:%S}\n')
    return True


def prune_cache(keep_days=KEEP_DAYS):
    """Drop HFR day-files older than keep_days.  Moorings/CDIP are single files."""
    cutoff = (DT.datetime.now(DT.timezone.utc).replace(tzinfo=None)
              - DT.timedelta(days=keep_days)).date()
    n, mb = 0, 0.0
    for fn in DATA_DIR.glob('hfr_nrt_*_*.nc'):
        stamp = fn.stem.split('_')[-1]
        try:
            d = DT.datetime.strptime(stamp, '%Y%m%d').date()
        except ValueError:
            continue                      # not a day-file; leave it alone
        if d < cutoff:
            mb += fn.stat().st_size / 1e6
            fn.unlink()
            n += 1
    if n:
        log(f'pruned {n} HFR day-files older than {cutoff} ({mb:.0f} MB)')


def copy_latest(paths):
    """Fixed-name copy of each figure, so something can point at the current one."""
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        # HFR_LV3_nrt_20260902_19Z_dt3h.png -> HFR_LV3_nrt_latest.png
        base = p.stem.split('_20')[0]     # everything before the date stamp
        shutil.copy2(p, p.with_name(f'{base}_latest{p.suffix}'))
        log(f'  latest -> {base}_latest{p.suffix}')


def run():
    # HFR_RES, PANEL_TIMES and T0 are all rebound part-way through the body
    # (resolutions that produced no grid are dropped; the panel times re-anchor
    # onto the newest hour that actually has data), so they must stay global
    # rather than becoming locals of run().
    global HFR_RES, PANEL_TIMES, T0

    # ------------------------------- Setup --------------------------------
    import os, re, time, urllib.request, urllib.error
    import datetime as DT
    from pathlib import Path

    import numpy as np
    import netCDF4 as nc
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.dates import DateFormatter
    from scipy.ndimage import binary_fill_holes

    # ====================== WHAT TO PLOT (edit these) =====================
    PANEL_MODE = 'latest'      # 'latest' -> newest HFR hour is the LAST panel
                               # 'start'  -> T0_USER is the FIRST panel
    T0_USER    = DT.datetime(2026, 9, 1, 0, 0)   # used only when PANEL_MODE=='start'
    DT_HOURS   = 3
    NPANEL     = 9             # 3 across; fewer panels if the data runs out

    COV_DAYS   = 14            # days of coverage history to fetch for the last figure
    MAX_AGE_H  = 3.0           # re-download SBOO / CDIP when the cache is older
    # ======================================================================

    # --- LV3 grid ---
    LV3_G = Path('/scratch/PFM_Simulations/Grids/GRID_SDTJRE_LV3_rx020.nc')

    # --- near-real-time HFR ---
    # NDBC is the primary: all three resolutions land on the SAME newest hour, the
    # variables are named u/v like the static cache, and OPeNDAP subsetting to the
    # LV3 box takes ~2 s per day.  CoastWatch ERDDAP serves the same product and is
    # kept as a fallback -- it is noticeably flakier (individual resolutions 404 for
    # minutes at a time) and its three resolutions often sit at different hours.
    HFR_SOURCE = 'ndbc'                    # 'ndbc' or 'coastwatch'
    NDBC_URL   = 'https://dods.ndbc.noaa.gov/thredds/dodsC/hfradar_uswc_{res}'
    ERDDAP     = 'https://coastwatch.pfeg.noaa.gov/erddap/griddap'
    ERDDAP_ID  = {'1km': 'ucsdHfrW1', '2km': 'ucsdHfrW2', '6km': 'ucsdHfrW6'}
    HFR_RES    = ('1km', '2km', '6km')     # coarse last, so sparse arrows draw on top
    NRT_CACHE  = Path('/dataSIO/DA_Simulations/input_files/hfr_nrt')

    # --- moorings: SBOO and PLOO, both City of San Diego / SIO, updated daily ---
    # The deployment number is NOT shared between them (SBOO is on 08 while PLOO is
    # on 07) and it rolls over without warning, so it is discovered rather than
    # hard-coded -- otherwise this notebook silently goes stale the day a mooring is
    # re-deployed.
    MOORING_URL = ('https://mooring.ucsd.edu/{site}/{site}_{nn}/nc/'
                   '{SITE}_{nn}_R_ADCP.nc')
    MOORING_MAX_DEPLOY = 20        # highest deployment number to probe for

    # PLOO is a 127 m site and SBOO a 37 m one, so a full-column average at PLOO
    # would not be comparable with SBOO's.  Both are averaged over the SAME top
    # `Z_BAD` metres by default, and the full-column PLOO average is printed
    # alongside so the difference is visible rather than hidden.
    Z_BAD = -28.0        # bins at or below this are excluded from the depth average

    MOORINGS = {
        'SBOO': dict(color='k',       z_bad=Z_BAD),
        'PLOO': dict(color='#6A1B9A', z_bad=Z_BAD),
    }

    # --- CDIP: live SoCal buoys inside LV3 are 191, 201 and 100.
    # 155 (Imperial Beach Nearshore) and 093 would be closer to the TJ River plume
    # but neither is in realtime right now, so 191 is the nearest live buoy.
    # 191 (Point Loma South) is the only live buoy inside the shelf zoom, but it
    # goes stale for a day at a time; 201 (Scripps Nearshore) and 100 (Torrey Pines
    # Outer) are reliably current and inside the LV3 window, though north of the
    # zoom box.  All three are shown so the figure has live wave context whichever
    # one is reporting.  Trim this tuple to taste.
    CDIP_STATIONS = ('191', '201', '100')
    CDIP_MAX_GAP_H = 2.0     # fall back to the nearest record within this many hours
    CDIP_URL = 'https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/{st}p1_rt.nc'
    CDIP_LOCAL = NRT_CACHE / 'cdip_{st}.npz'

    OUT_DIR = FIG_DIR
    NRT_CACHE.mkdir(parents=True, exist_ok=True)

    # --------------------------- plot styling -----------------------------
    LAND_COLOR  = '#cdb79e'
    COAST_COLOR = '#7a5d33'
    BATHY_COLOR = '0.55'
    LV3_COLOR   = 'k'
    BATHY_INT   = 20.0
    BATHY_MAX   = 200.0
    HFR_COLOR = {'1km': 'tab:blue', '2km': 'tab:red', '6km': 'tab:green'}
    HFR_SUB   = {'1km': 1, '2km': 1, '6km': 1}

    QSCALE = 6.0        # m/s across the axes width (full-domain figure)
    QWIDTH = 0.0035
    QKEY   = 0.25
    SCALE_KM = 30.0
    SCALE_ANCHOR = (0.96, 0.93)
    PAD_DEG = 0.03
    ZOOM_BOX = (-117.45, -117.05, 32.35, 32.80)

    # thick = depth-integrated, thin = surface bin; colour distinguishes the site
    MOOR_KW_DAVG = dict(width=0.012, headwidth=4.0, headlength=4.5,
                        headaxislength=4.0, zorder=9)
    MOOR_KW_SURF = dict(width=0.0035, headwidth=4.5, headlength=5.0,
                        headaxislength=4.5, zorder=10)

    # CDIP: long and thick so it reads through a field of current vectors
    CDIP_COLOR = "#45951E"
    # Arrow length is proportional to Hs, normalised so the LARGEST Hs anywhere in
    # the figure set gets CDIP_FRAC of the axes width.  Panels are therefore
    # comparable with each other and between the two figures.
    CDIP_FRAC  = 0.13        # axes-width fraction at the largest Hs
    CDIP_TEXT_ANCHOR = (0.98, 0.79)   # Hs readout, axes fraction, right-aligned
    CDIP_TEXT_DY = 0.042              # line spacing, axes fraction
    CDIP_FONTSIZE = 10
    CDIP_KW    = dict(color=CDIP_COLOR, width=0.011, headwidth=4.0, headlength=4.5,
                      headaxislength=4.0, zorder=13)
    # CDIP waveDp is the direction waves come FROM.  Currents are drawn as
    # "toward" vectors, so the wave arrow is drawn toward as well (Dp + 180).
    CDIP_TOWARD = True

    # Short names for the wave labels.  A buoy pinned into the frame by
    # WAVE_ANCHOR is NOT at its real position, so it is always named -- an
    # unlabelled arrow somewhere it does not belong would read as a measurement
    # at that spot.
    CDIP_NAME = {'191': 'Pt Loma', '201': 'Scripps', '100': 'Torrey Pines'}

    # Buoys outside a figure's window can be pinned to a fixed spot in the frame
    # for reference, as (x, y) in axes fraction.  Torrey Pines (100) sits at
    # 32.93 N, well north of the shelf zoom, but it is the buoy that stays current
    # when 191 goes stale -- so it is pinned just inside the western edge, a
    # quarter of the way up.  201 (Scripps, 32.87 N) is off-map too; add it here
    # if you want it as well.
    ZOOM_WAVE_ANCHOR = {'100': (0.10, 0.25)}

    def now_utc():
        """Naive UTC 'now' (datetime.utcnow() is deprecated)."""
        return DT.datetime.now(DT.timezone.utc).replace(tzinfo=None)


    EPOCH = DT.datetime(1970, 1, 1)
    _to_secs = lambda t: np.array([(x - EPOCH).total_seconds() for x in t], dtype='f8')
    _from_secs = lambda s: np.array([EPOCH + DT.timedelta(seconds=float(x)) for x in s])

    print(f'now (UTC)  : {now_utc():%Y-%m-%d %H:%M}')
    print(f'HFR source : {HFR_SOURCE}')
    print(f'panel mode : {PANEL_MODE}')
    print(f'cache      : {NRT_CACHE}')
    print(f'figures    : {OUT_DIR}')

    # ------------------------- LV3 grid ----------------------------------
    with nc.Dataset(LV3_G) as g:
        lv3_lon  = np.asarray(g.variables['lon_rho'][:], float)
        lv3_lat  = np.asarray(g.variables['lat_rho'][:], float)
        lv3_h    = np.asarray(g.variables['h'][:], float)
        lv3_mask = np.asarray(g.variables['mask_rho'][:], float)

    lv3_land = binary_fill_holes(lv3_mask == 0)
    land_fld = np.where(lv3_land, 1.0, np.nan)
    h_water  = np.where(lv3_land, np.nan, lv3_h)
    _hmax = np.nanmax(h_water) if BATHY_MAX is None else min(BATHY_MAX, np.nanmax(h_water))
    BATHY_LEVELS = np.arange(BATHY_INT, _hmax + 0.5 * BATHY_INT, BATHY_INT)

    LV3_OUTLINE = (
        np.concatenate([lv3_lon[0, :], lv3_lon[:, -1], lv3_lon[-1, ::-1], lv3_lon[::-1, 0]]),
        np.concatenate([lv3_lat[0, :], lv3_lat[:, -1], lv3_lat[-1, ::-1], lv3_lat[::-1, 0]]),
    )

    XLIM = (lv3_lon.min() - PAD_DEG, lv3_lon.max() + PAD_DEG)
    YLIM = (lv3_lat.min() - PAD_DEG, lv3_lat.max() + PAD_DEG)
    LAT_MID = 0.5 * (YLIM[0] + YLIM[1])

    print(f'LV3 grid : {lv3_lon.shape}  lon {XLIM[0]:.3f}..{XLIM[1]:.3f}  '
          f'lat {YLIM[0]:.3f}..{YLIM[1]:.3f}')
    print(f'bathy    : {len(BATHY_LEVELS)} contours every {BATHY_INT:g} m to {_hmax:.0f} m')

    # ---------------- near-real-time HFR from CoastWatch ERDDAP -----------
    def _get(url, tries=8, pause=20, timeout=300):
        """GET with retries and growing backoff.

        ERDDAP reloads these near-real-time datasets constantly; a request landing
        mid-reload returns 404 (`Currently unknown datasetID`) or 500
        (`partialResults[0] was not as expected`).  Both are transient -- but an
        individual dataset can also stay down for many minutes, which is why
        nothing here treats a failure as fatal.
        """
        last = None
        for k in range(tries):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as r:
                    return r.read()
            except Exception as e:
                last = e
                if k < tries - 1:
                    time.sleep(pause * (1 + k // 3))     # 20s, 20s, 20s, 40s, ...
        raise RuntimeError(f'{type(last).__name__}: {last}  <- {url[:110]}')


    def erddap_last_time(res):
        """Newest hour ERDDAP holds for one resolution, or None if it is down."""
        try:
            txt = _get(f'{ERDDAP}/{ERDDAP_ID[res]}.csv?time%5Blast%5D').decode()
        except Exception as e:
            print(f'{res}: ERDDAP unavailable ({str(e)[:70]})')
            return None
        return DT.datetime.strptime(txt.strip().split('\n')[-1].strip(),
                                    '%Y-%m-%dT%H:%M:%SZ')


    _NDBC = {}


    def ndbc_meta(res):
        """Open the NDBC aggregation once and keep its axes and LV3 slice.

        The full grids are large (1 km is 2197 x 1399) but OPeNDAP only ships the
        box we ask for, so the cost is the one-off metadata read.
        """
        if res not in _NDBC:
            d = nc.Dataset(NDBC_URL.format(res=res))
            lat = np.asarray(d.variables['lat'][:], float)
            lon = np.asarray(d.variables['lon'][:], float)
            tv = d.variables['time']
            times = np.array([t.replace(microsecond=0) for t in
                              nc.num2date(np.asarray(tv[:], float), tv.units,
                                          only_use_cftime_datetimes=False)])
            jj = np.where((lat >= YLIM[0]) & (lat <= YLIM[1]))[0]
            ii = np.where((lon >= XLIM[0]) & (lon <= XLIM[1]))[0]
            _NDBC[res] = dict(ds=d, times=times, lat=lat[jj], lon=lon[ii],
                              jsl=slice(int(jj[0]), int(jj[-1]) + 1),
                              isl=slice(int(ii[0]), int(ii[-1]) + 1))
        return _NDBC[res]


    def hfr_last_time(res):
        """Newest hour the chosen source holds, or None if it is unreachable."""
        if HFR_SOURCE == 'coastwatch':
            return erddap_last_time(res)
        try:
            return ndbc_meta(res)['times'][-1]
        except Exception as e:
            print(f'{res}: NDBC unavailable ({type(e).__name__}: {str(e)[:70]})')
            return None


    def cached_last_time(res):
        """Newest hour already on disk for one resolution, or None."""
        best = None
        for fn in NRT_CACHE.glob(f'hfr_nrt_{HFR_SOURCE}_{res}_*.nc'):
            try:
                with nc.Dataset(fn) as d:
                    t = np.asarray(d.variables['time'][:], float)
                if t.size:
                    tt = EPOCH + DT.timedelta(seconds=float(t.max()))
                    best = tt if best is None else max(best, tt)
            except Exception:
                pass
        return best


    def _day_file(res, day, source=None):
        # The source is in the name on purpose: NDBC and ERDDAP return slightly
        # different box shapes for the same lon/lat request, so a cache shared
        # between them would hand back day-files that do not align.
        return NRT_CACHE / f'hfr_nrt_{source or HFR_SOURCE}_{res}_{day:%Y%m%d}.nc'


    def _write_day(fn, times, lat, lon, u, v):
        """One day-file, same schema whichever backend produced it."""
        tmp = fn.with_suffix('.tmp')
        with nc.Dataset(tmp, 'w', format='NETCDF4') as d:
            d.createDimension('time', len(times))
            d.createDimension('latitude', lat.size)
            d.createDimension('longitude', lon.size)
            tv = d.createVariable('time', 'f8', ('time',))
            tv.units = 'seconds since 1970-01-01'
            tv[:] = _to_secs(times)
            d.createVariable('latitude', 'f4', ('latitude',))[:] = lat
            d.createVariable('longitude', 'f4', ('longitude',))[:] = lon
            for nm, arr in (('water_u', u), ('water_v', v)):
                va = d.createVariable(nm, 'f4', ('time', 'latitude', 'longitude'),
                                      fill_value=np.float32(np.nan))
                va[:] = arr
            d.source = HFR_SOURCE
        tmp.replace(fn)                    # never leave a half-written day behind


    def fetch_day(res, day, force=False, last=None):
        """Cache one day of `res` over the LV3 box.  Returns the path or None.

        A day already on disk is trusted unless it is today -- today's file grows
        hour by hour, so it is always re-fetched.
        """
        fn = _day_file(res, day)
        if fn.exists() and not force and day.date() < now_utc().date():
            return fn
        if last is not None and day > last:
            return None                     # entirely in the future

        if HFR_SOURCE == 'ndbc':
            try:
                M = ndbc_meta(res)
                k = np.where(np.array([t.date() for t in M['times']]) == day.date())[0]
                if not k.size:
                    return None
                ds = M['ds']
                sl = slice(int(k[0]), int(k[-1]) + 1)
                u = np.ma.filled(ds.variables['u'][sl, M['jsl'], M['isl']].astype(float), np.nan)
                v = np.ma.filled(ds.variables['v'][sl, M['jsl'], M['isl']].astype(float), np.nan)
                _write_day(fn, list(M['times'][sl]), M['lat'], M['lon'], u, v)
                return fn
            except Exception as e:
                print(f'    {res} {day:%Y-%m-%d}: FAILED  {type(e).__name__} {str(e)[:80]}')
                _NDBC.pop(res, None)        # drop a stale handle so the next try reopens
                return None

        # --- CoastWatch ERDDAP fallback ---
        # ERDDAP does NOT clamp time constraints: asking for 23:00Z on a day whose
        # data stops at 15:00Z returns a 404 that reads exactly like a mid-reload
        # failure, so the stop time has to be clamped here.
        day_end = day + DT.timedelta(hours=23)
        if last is not None:
            day_end = min(day_end, last)
        sel = (f'%5B({day:%Y-%m-%d}T00:00:00Z):1:({day_end:%Y-%m-%d}T{day_end:%H}:00:00Z)%5D'
               f'%5B({YLIM[0]:.4f}):1:({YLIM[1]:.4f})%5D'
               f'%5B({XLIM[0]:.4f}):1:({XLIM[1]:.4f})%5D')
        try:
            blob = _get(f'{ERDDAP}/{ERDDAP_ID[res]}.nc?water_u{sel},water_v{sel}')
        except Exception as e:
            print(f'    {res} {day:%Y-%m-%d}: FAILED  {str(e)[:90]}')
            return None
        tmp = fn.with_suffix('.tmp')
        tmp.write_bytes(blob)
        tmp.replace(fn)
        return fn


    def load_day(res, day):
        """{datetime: (u, v)} for one cached day, plus its lon/lat axes."""
        fn = _day_file(res, day)
        if not fn.exists():
            return {}, None, None
        with nc.Dataset(fn) as d:
            t = _from_secs(np.asarray(d.variables['time'][:], float))
            lon = np.asarray(d.variables['longitude'][:], float)
            lat = np.asarray(d.variables['latitude'][:], float)
            u = np.ma.filled(d.variables['water_u'][:].astype(float), np.nan)
            v = np.ma.filled(d.variables['water_v'][:].astype(float), np.nan)
        return ({tt: (u[k], v[k]) for k, tt in enumerate(t)}, lon, lat)


    LAST_TIME, LIVE = {}, {}
    for res in HFR_RES:
        t_live = hfr_last_time(res)
        LIVE[res] = t_live is not None
        t_use = t_live if t_live is not None else cached_last_time(res)
        LAST_TIME[res] = t_use
        if t_use is None:
            print(f'{res}: no live feed and nothing cached')
        else:
            age = (now_utc() - t_use).total_seconds() / 3600
            print(f'{res}: newest {t_use:%Y-%m-%d %H:%M}Z ({age:.1f} h old)'
                  f'{"" if LIVE[res] else "   [from cache -- feed down]"}')

    _have = [t for t in LAST_TIME.values() if t is not None]
    if not _have:
        raise RuntimeError('no HFR available live or cached; nothing to plot')
    HFR_LATEST = max(_have)
    print(f'\nmost current HFR hour: {HFR_LATEST:%Y-%m-%d %H:%M}Z '
          f'(live: {", ".join(r for r in HFR_RES if LIVE[r]) or "none"})')

    # --------------------------- panel times ------------------------------
    if PANEL_MODE == 'latest':
        PANEL_TIMES = [HFR_LATEST - DT.timedelta(hours=DT_HOURS * k)
                       for k in range(NPANEL)][::-1]
    elif PANEL_MODE == 'start':
        PANEL_TIMES = [T0_USER + DT.timedelta(hours=DT_HOURS * k)
                       for k in range(NPANEL)]
        PANEL_TIMES = [t for t in PANEL_TIMES if t <= HFR_LATEST]
        if not PANEL_TIMES:
            raise ValueError(f'T0_USER {T0_USER} is later than the newest HFR hour '
                             f'{HFR_LATEST}; nothing to plot')
    else:
        raise ValueError(f"PANEL_MODE must be 'latest' or 'start', got {PANEL_MODE!r}")

    T0 = PANEL_TIMES[0]
    print(f'{len(PANEL_TIMES)} panels, {DT_HOURS} h apart '
          f'({"of " + str(NPANEL) + " requested" if len(PANEL_TIMES) < NPANEL else "full grid"})')
    for k, t in enumerate(PANEL_TIMES):
        print(f'  {k}: {t:%Y-%m-%d %H:%M}Z')

    # --- days we need, and the coverage history window ---
    _panel_days = sorted({t.date() for t in PANEL_TIMES})
    _cov_days = [(now_utc().date() - DT.timedelta(days=k))
                 for k in range(COV_DAYS)]
    NEED_DAYS = sorted({*_panel_days, *_cov_days})
    print(f'\nfetching {len(NEED_DAYS)} days x {len(HFR_RES)} resolutions '
          f'({NEED_DAYS[0]} .. {NEED_DAYS[-1]}) -- cached days are skipped')

    for res in HFR_RES:
        n_new, n_fail = 0, 0
        for d in NEED_DAYS:
            day = DT.datetime.combine(d, DT.time())
            existed = _day_file(res, day).exists()
            if not LIVE[res]:
                n_fail += not existed
                continue                       # feed is down; use what we have
            got = fetch_day(res, day, last=LAST_TIME[res])
            n_fail += got is None
            n_new += (not existed) or (d == now_utc().date())
        n_cached = len(list(NRT_CACHE.glob(f'hfr_nrt_{HFR_SOURCE}_{res}_*.nc')))
        print(f'  {res}: {n_new} fetched/refreshed, {n_fail} unavailable, '
              f'{n_cached} day-files cached'
              f'{"" if LIVE[res] else "   [feed down, cache only]"}')

    # --------------- assemble the HFR grid and panel snapshots ------------
    HFR_GRID, HFR = {}, {}
    DAY_CACHE = {}

    for res in HFR_RES:
        lon = lat = None
        for d in NEED_DAYS:
            day = DT.datetime.combine(d, DT.time())
            recs, lo, la = load_day(res, day)
            DAY_CACHE[(res, d)] = recs
            if lo is not None and lon is None:
                lon, lat = lo, la
        if lon is None:
            print(f'{res}: NO DATA (feed down and nothing cached) -- dropped from '
                  f'this run')
            continue
        LON, LAT = np.meshgrid(lon, lat)
        HFR_GRID[res] = dict(lon=LON, lat=LAT, lon1d=lon, lat1d=lat,
                             dx_km=float(np.diff(lon).mean()) * 111.0 * np.cos(np.radians(LAT_MID)),
                             dy_km=float(np.diff(lat).mean()) * 111.0)
        G = HFR_GRID[res]
        print(f'{res}: {LON.shape} cells over LV3, '
              f'dx {G["dx_km"]:.2f} km, dy {G["dy_km"]:.2f} km')

    # only the resolutions that actually produced a grid are plotted
    HFR_RES = tuple(r for r in HFR_RES if r in HFR_GRID)
    if not HFR_RES:
        raise RuntimeError('no HFR resolution has any data')
    print(f'\nplotting resolutions: {", ".join(HFR_RES)}')

    _nan = lambda res: (np.full(HFR_GRID[res]['lon'].shape, np.nan),) * 2

    # ERDDAP advertises the newest hour BEFORE we try to fetch it, and an individual
    # resolution can 404 for minutes at a time -- so the advertised hour is not
    # always an hour we hold.  In 'latest' mode, re-anchor on the newest hour that
    # actually has data, otherwise the most interesting panel comes out blank.
    HAVE = sorted({t for res in HFR_RES for d in NEED_DAYS
                   for t, (u, _) in DAY_CACHE.get((res, d), {}).items()
                   if np.isfinite(u).any()})
    if not HAVE:
        raise RuntimeError('no cached hour has any valid HFR data')
    if PANEL_MODE == 'latest' and HAVE[-1] != PANEL_TIMES[-1]:
        print(f're-anchoring: newest hour WITH DATA is {HAVE[-1]:%Y-%m-%d %H:%M}Z, '
              f'not the advertised {PANEL_TIMES[-1]:%Y-%m-%d %H:%M}Z')
        PANEL_TIMES = [HAVE[-1] - DT.timedelta(hours=DT_HOURS * k)
                       for k in range(NPANEL)][::-1]
        T0 = PANEL_TIMES[0]

    for res in HFR_RES:
        HFR[res] = {t: DAY_CACHE.get((res, t.date()), {}).get(t, _nan(res))
                    for t in PANEL_TIMES}

    print(f'\n{"time":>16} ' + ' '.join(f'{r:>16}' for r in HFR_RES))
    print(f'{"":>16} ' + ' '.join(f'{"cells":>8}{"avail":>8}' for _ in HFR_RES))
    print('-' * (17 + 17 * len(HFR_RES)))
    for t in PANEL_TIMES:
        row = f'{t:%Y-%m-%d %H:%M} '
        for res in HFR_RES:
            u, v = HFR[res][t]
            g = np.isfinite(u) & np.isfinite(v)
            row += f'{int(g.sum()):8d}{100*g.mean():7.1f}%'
        print(row)

    # ------------------- moorings: discover, download, hourly -------------
    def stale(path, max_age_h=MAX_AGE_H):
        if not path.exists():
            return True
        return (time.time() - path.stat().st_mtime) / 3600 > max_age_h


    def _mooring_url(site, n):
        return MOORING_URL.format(site=site.lower(), SITE=site.upper(), nn=f'{n:02d}')


    def latest_deployment(site, hi=MOORING_MAX_DEPLOY):
        """Highest deployment number whose file exists, probing downward."""
        for n in range(hi, 0, -1):
            try:
                req = urllib.request.Request(_mooring_url(site, n), method='HEAD')
                with urllib.request.urlopen(req, timeout=30) as r:
                    if r.status == 200:
                        return n
            except Exception:
                continue
        return None


    def load_mooring(site, z_bad):
        """Download (if stale) and reduce one mooring to hourly-ready series.

        Returns lat/lon, the raw time axis, depth-averaged and surface-bin u/v, and
        a full-column average kept only for reporting.
        """
        n = latest_deployment(site)
        if n is None:
            raise RuntimeError(f'no {site} deployment found')
        url = _mooring_url(site, n)
        local = NRT_CACHE / f'{site.upper()}_{n:02d}_R_ADCP.nc'
        if stale(local):
            print(f'  downloading {url} ...')
            local.write_bytes(_get(url, tries=3, pause=10))
        print(f'{site}: deployment {n:02d}, {local.name} '
              f'({(time.time() - local.stat().st_mtime)/60:.0f} min old, '
              f'{local.stat().st_size/1e6:.0f} MB)')

        with nc.Dataset(local) as d:
            t_raw = np.asarray(d.variables['TIME'][:], float)   # days since 1950
            depth = np.asarray(d.variables['DEPTH'][:], float)  # positive down
            u = np.ma.filled(d.variables['UCUR'][:].astype(float), np.nan)
            v = np.ma.filled(d.variables['VCUR'][:].astype(float), np.nan)
            lat = float(np.asarray(d.variables['LATITUDE'][:], float).ravel()[0])
            lon = float(np.asarray(d.variables['LONGITUDE'][:], float).ravel()[0])

        # the raw axis is neither sorted nor unique -- fix before anything reads it
        order = np.argsort(t_raw, kind='stable')
        t_raw, u, v = t_raw[order], u[order], v[order]
        _, first = np.unique(t_raw, return_index=True)
        first.sort()
        n_dup = len(t_raw) - len(first)
        t_raw, u, v = t_raw[first], u[first], v[first]
        t = np.array([DT.datetime(1950, 1, 1) + DT.timedelta(days=float(x))
                      for x in t_raw])

        keep = (-depth) > z_bad
        k_surf = int(np.argmin(depth))
        with np.errstate(invalid='ignore'):
            ud, vd = np.nanmean(u[:, keep], axis=1), np.nanmean(v[:, keep], axis=1)
            uf, vf = np.nanmean(u, axis=1), np.nanmean(v, axis=1)   # full column

        print(f'   {lat:.5f} N {lon:.5f} E   {len(t)} records '
              f'({n_dup} duplicate timestamps dropped)')
        print(f'   {t[0]:%Y-%m-%d %H:%M} -> {t[-1]:%Y-%m-%d %H:%M}  '
              f'({(now_utc()-t[-1]).total_seconds()/3600:.1f} h old)')
        print(f'   {len(depth)} bins {depth.min():.2f}..{depth.max():.2f} m; '
              f'{int(keep.sum())} kept for the depth average (z > {z_bad:g} m); '
              f'surface bin {depth[k_surf]:.2f} m')
        return dict(lat=lat, lon=lon, t=t, ts=_to_secs(t), depth=depth,
                    k_surf=k_surf, n_keep=int(keep.sum()),
                    u_davg=ud, v_davg=vd, u_surf=u[:, k_surf], v_surf=v[:, k_surf],
                    u_full=uf, v_full=vf, deploy=n)


    MOOR = {}
    for site, cfg in MOORINGS.items():
        try:
            MOOR[site] = load_mooring(site, cfg['z_bad'])
            MOOR[site].update(cfg)
        except Exception as e:
            print(f'{site}: unavailable ({type(e).__name__}: {str(e)[:80]})')


    def mooring_hour_mean(site, t_c, half_win_s=1800.0):
        """Mean (u, v) over the hour centred on t_c: [t_c-30min, t_c+30min)."""
        M = MOOR.get(site)
        if M is None:
            return dict(n=0)
        c = (t_c - EPOCH).total_seconds()
        m = (M['ts'] >= c - half_win_s) & (M['ts'] < c + half_win_s)
        if not m.any():
            return dict(n=0)
        with np.errstate(invalid='ignore'):
            out = {k: (float(np.nanmean(M[k][m])) if np.isfinite(M[k][m]).any()
                       else np.nan)
                   for k in ('u_davg', 'v_davg', 'u_surf', 'v_surf', 'u_full', 'v_full')}
        out['n'] = int(m.sum())
        return out


    SBOO = {}          # {site: {time: {...}}}, keyed by site
    for site in MOOR:
        SBOO[site] = {t: mooring_hour_mean(site, t) for t in PANEL_TIMES}

    print(f'\n{"time":>16} {"site":>5} {"n":>3} {"u_davg":>8} {"v_davg":>8} '
          f'{"u_surf":>8} {"v_surf":>8} {"u_full":>8} {"v_full":>8}')
    print('-' * 80)
    for t in PANEL_TIMES:
        for site in MOOR:
            w = SBOO[site][t]
            if not w['n']:
                print(f'{t:%Y-%m-%d %H:%M} {site:>5} {0:3d}' + '       -' * 6)
                continue
            print(f'{t:%Y-%m-%d %H:%M} {site:>5} {w["n"]:3d} '
                  + ' '.join(f'{w[k]:+8.4f}' for k in
                             ('u_davg', 'v_davg', 'u_surf', 'v_surf', 'u_full', 'v_full')))

    # how much the top-28 m choice matters at the deep site
    if 'PLOO' in MOOR:
        _d = [(SBOO['PLOO'][t]['u_davg'] - SBOO['PLOO'][t]['u_full'],
               SBOO['PLOO'][t]['v_davg'] - SBOO['PLOO'][t]['v_full'])
              for t in PANEL_TIMES if SBOO['PLOO'][t]['n']]
        if _d:
            _m = np.nanmean([np.hypot(a, b) for a, b in _d])
            print(f'\nPLOO: mean |top-{-Z_BAD:g} m avg - full-column avg| = {_m:.4f} m/s'
                  f'  -- the cost of making it comparable with SBOO')

    # -------------------------- CDIP waves --------------------------------
    def load_cdip(st):
        """(lat, lon, name, t, Hs, Tp, Dp) for one realtime station, cached locally."""
        fn = Path(str(CDIP_LOCAL).format(st=st))
        if stale(fn):
            with nc.Dataset(CDIP_URL.format(st=st)) as d:
                z = dict(lat=float(d.variables['metaDeployLatitude'][:]),
                         lon=float(d.variables['metaDeployLongitude'][:]),
                         name=str(d.variables['metaStationName'][:].tobytes()
                                  .decode('utf-8', 'ignore')).strip('\x00').strip(),
                         t=np.asarray(d.variables['waveTime'][:], float),
                         hs=np.asarray(d.variables['waveHs'][:], float),
                         tp=np.asarray(d.variables['waveTp'][:], float),
                         dp=np.asarray(d.variables['waveDp'][:], float))
            np.savez(fn, **z)
        z = np.load(fn, allow_pickle=True)
        return dict(lat=float(z['lat']), lon=float(z['lon']), name=str(z['name']),
                    t=z['t'], hs=z['hs'], tp=z['tp'], dp=z['dp'])


    CDIP = {}
    for st in CDIP_STATIONS:
        try:
            CDIP[st] = load_cdip(st)
        except Exception as e:
            print(f'CDIP {st}: unavailable ({type(e).__name__}: {str(e)[:70]})')
    for st, c in CDIP.items():
        last = EPOCH + DT.timedelta(seconds=float(c['t'][-1]))
        inside = (XLIM[0] <= c['lon'] <= XLIM[1]) and (YLIM[0] <= c['lat'] <= YLIM[1])
        print(f"CDIP {st}: {c['name']}")
        print(f"   {c['lat']:.4f} N {c['lon']:.4f} E   in LV3 window: {inside}")
        print(f"   {len(c['t'])} records, newest {last:%Y-%m-%d %H:%M}Z "
              f"({(now_utc()-last).total_seconds()/3600:.1f} h old)")


    def cdip_hour_mean(st, t_c, half_win_s=1800.0, max_gap_h=CDIP_MAX_GAP_H):
        """Hs and direction near t_c.  Returns (Hs, dir_deg, Tp, n, age_h).

        Preferred: the mean over the hour centred on t_c, matching SBOO and the HFR
        time_bnds.  These buoys drop out for hours at a time, so if that window is
        empty the nearest record within `max_gap_h` is used instead and its offset
        is returned as `age_h` -- non-zero age is surfaced on the plot, because a
        wave height carried in from hours away should not look like a measurement
        at the panel time.

        Direction is averaged as a unit vector, so 350 and 10 degrees give 0, not 180.
        """
        c = CDIP.get(st)
        if c is None:
            return np.nan, np.nan, np.nan, 0, np.nan
        cs = (t_c - EPOCH).total_seconds()
        ok = np.isfinite(c['hs']) & np.isfinite(c['dp'])
        g = ok & (c['t'] >= cs - half_win_s) & (c['t'] < cs + half_win_s)
        age = 0.0
        if not g.any():
            if not ok.any():
                return np.nan, np.nan, np.nan, 0, np.nan
            k = int(np.argmin(np.abs(c['t'][ok] - cs)))
            gap = abs(c['t'][ok][k] - cs) / 3600.0
            if gap > max_gap_h:
                return np.nan, np.nan, np.nan, 0, gap
            g = np.zeros(len(c['t']), bool)
            g[np.where(ok)[0][k]] = True
            age = (c['t'][g][0] - cs) / 3600.0
        th = np.radians(c['dp'][g])
        ang = np.degrees(np.arctan2(np.mean(np.sin(th)), np.mean(np.cos(th)))) % 360
        return (float(np.mean(c['hs'][g])), float(ang),
                float(np.nanmean(c['tp'][g])), int(g.sum()), float(age))


    WAVES = {}
    print(f'\n{"time":>16} {"st":>4} {"n":>3} {"Hs_m":>6} {"Tp_s":>6} {"Dp_deg":>7} {"offset_h":>9}')
    print('-' * 58)
    for t in PANEL_TIMES:
        for st in CDIP:
            hs, dp, tp, n, age = cdip_hour_mean(st, t)
            WAVES[(st, t)] = dict(hs=hs, dp=dp, tp=tp, n=n, age=age)
            print(f'{t:%Y-%m-%d %H:%M} {st:>4} {n:3d} {hs:6.2f} {tp:6.1f} {dp:7.1f} '
                  f'{age:9.1f}')

    # One normalisation for every panel of both figures, so arrow length is
    # comparable across panels and between the LV3 and shelf views.
    _hs_all = [w['hs'] for w in WAVES.values() if np.isfinite(w.get('hs', np.nan))]
    HS_MAX = float(max(_hs_all)) if _hs_all else 1.0
    print(f'\narrow scaling: largest Hs over all panels = {HS_MAX:.2f} m '
          f'-> {CDIP_FRAC:.2f} of the axes width')

    # ---------------------- basemap and panel grid ------------------------
    def draw_basemap(ax, xlim, ylim, outline=False):
        """LV3 bathymetry (gray, 20 m), land (light brown) and the coastline."""
        ax.contour(lv3_lon, lv3_lat, h_water, levels=BATHY_LEVELS,
                   colors=BATHY_COLOR, linewidths=0.5, zorder=1)
        ax.pcolormesh(lv3_lon, lv3_lat, land_fld, shading='auto',
                      cmap=ListedColormap([LAND_COLOR]), zorder=2)
        ax.contour(lv3_lon, lv3_lat, lv3_land.astype(float), levels=[0.5],
                   colors=COAST_COLOR, linewidths=0.7, zorder=3)
        if outline:
            ax.plot(*LV3_OUTLINE, color=LV3_COLOR, lw=1.2, ls='--', zorder=4)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect(1.0 / np.cos(np.radians(0.5 * (ylim[0] + ylim[1]))))
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))


    def draw_waves(ax, t, qscale, xlim, ylim, anchor=None, hs_max=None):
        """Wave arrows at the CDIP buoys, plus an Hs readout in the upper right.

        Arrow LENGTH is proportional to Hs, normalised so `hs_max` (the largest Hs
        anywhere in the figure set) draws at CDIP_FRAC of the axes width -- so the
        arrows carry height as well as direction, and panels are comparable.

        The numbers live in a corner block rather than beside each arrow: labels
        sitting on top of a dense vector field were unreadable.

        A buoy outside the window is drawn only if `anchor` pins it to a spot in
        the frame, in which case it gets a SQUARE marker, so it is not mistaken for
        a measurement at that location.
        """
        anchor = anchor or {}
        hs_max = hs_max or HS_MAX
        rows = []
        for st in CDIP:
            c, W = CDIP[st], WAVES.get((st, t), {})
            hs, dp = W.get('hs', np.nan), W.get('dp', np.nan)
            inside = (xlim[0] <= c['lon'] <= xlim[1] and
                      ylim[0] <= c['lat'] <= ylim[1])
            if inside:
                x, y, pinned = c['lon'], c['lat'], False
            elif st in anchor:
                fx, fy = anchor[st]
                x = xlim[0] + fx * (xlim[1] - xlim[0])
                y = ylim[0] + fy * (ylim[1] - ylim[0])
                pinned = True
            else:
                continue                     # not shown, so not in the readout
            ax.plot(x, y, marker='s' if pinned else 'o', mfc='none', mec=CDIP_COLOR,
                    mew=1.4, ms=7, ls='none', zorder=13)
            name = CDIP_NAME.get(st, st) + ('*' if pinned else '')
            if not (np.isfinite(hs) and np.isfinite(dp)):
                rows.append(f'{name}  --')
                continue
            th = np.radians(dp + (180.0 if CDIP_TOWARD else 0.0))
            mag = CDIP_FRAC * qscale * (hs / hs_max)
            ax.quiver([x], [y], [mag * np.sin(th)], [mag * np.cos(th)],
                      angles='uv', scale=qscale, scale_units='width', **CDIP_KW)
            age = W.get('age', 0.0)
            rows.append(f'{name}  {hs:.2f}' +
                        (f' ({age:+.0f}h)' if abs(age) >= 0.5 else ''))

        if rows:
            fx, fy = CDIP_TEXT_ANCHOR
            ax.text(fx, fy, '$H_s$ (m)', transform=ax.transAxes, ha='right',
                    va='top', color=CDIP_COLOR, fontsize=CDIP_FONTSIZE,
                    fontweight='bold', zorder=14,
                    bbox=dict(facecolor='white', alpha=0.72, edgecolor='none', pad=1.5))
            for k, r in enumerate(rows, start=1):
                ax.text(fx, fy - k * CDIP_TEXT_DY, r, transform=ax.transAxes,
                        ha='right', va='top', color=CDIP_COLOR,
                        fontsize=CDIP_FONTSIZE, fontweight='bold', zorder=14,
                        bbox=dict(facecolor='white', alpha=0.72, edgecolor='none',
                                  pad=1.5))


    def panel_grid(times, draw_panel, data_handles, sboo=SBOO, title='',
                   fname=None, xlim=None, ylim=None, qscale=QSCALE, qkey=QKEY,
                   ncol=3, fig_w=15.0, row_h=6.0, outline=False, scale_km=SCALE_KM,
                   wave_anchor=None):
        """Basemap, SBOO arrows, CDIP wave arrows, key, labels and legend.

        `draw_panel(ax, t, qscale)` draws the data for one time and returns the text
        appended to that panel's title.  Rows follow from how many times are passed
        and `row_h` fixes the height of one row, so panels keep their size.
        """
        xlim = XLIM if xlim is None else xlim
        ylim = YLIM if ylim is None else ylim

        def draw_key(ax):
            (x0, x1), (y0, y1) = xlim, ylim
            xr = x0 + SCALE_ANCHOR[0] * (x1 - x0)
            yb = y0 + SCALE_ANCHOR[1] * (y1 - y0)
            dlon = scale_km / (111.0 * np.cos(np.radians(yb)))
            tick = 0.010 * (y1 - y0)
            ax.plot([xr - dlon, xr], [yb, yb], color='k', lw=1.8,
                    solid_capstyle='butt', zorder=12)
            for xx in (xr - dlon, xr):
                ax.plot([xx, xx], [yb - tick, yb + tick], color='k', lw=1.8, zorder=12)
            ax.text(xr - 0.5 * dlon, yb + 1.6 * tick, f'{scale_km:g} km',
                    ha='center', va='bottom', fontsize=9, zorder=12)
            ya = yb - 5.5 * tick
            ax.quiver([xr - dlon], [ya], [qkey], [0.0], color='k', angles='uv',
                      scale=qscale, scale_units='width', width=QWIDTH,
                      headwidth=3.5, headlength=4.0, zorder=12)
            ax.text(xr - dlon, ya - 1.6 * tick, f'{qkey:g} m/s',
                    ha='left', va='top', fontsize=9, zorder=12)

        nrow = int(np.ceil(len(times) / ncol))
        fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True,
                                 figsize=(fig_w, row_h * nrow + 1.0), squeeze=False)
        axes = axes.ravel()

        for k, (ax, t) in enumerate(zip(axes, times)):
            draw_basemap(ax, xlim, ylim, outline=outline)
            suffix = draw_panel(ax, t, qscale)

            for site, M in MOOR.items():
                S = sboo.get(site, {}).get(t, {})
                col = M['color']
                for key_u, key_v, kw in (('u_davg', 'v_davg', MOOR_KW_DAVG),
                                         ('u_surf', 'v_surf', MOOR_KW_SURF)):
                    uu, vv = S.get(key_u, np.nan), S.get(key_v, np.nan)
                    if np.isfinite(uu) and np.isfinite(vv):
                        ax.quiver([M['lon']], [M['lat']], [uu], [vv], color=col,
                                  angles='uv', scale=qscale, scale_units='width', **kw)
                ax.plot(M['lon'], M['lat'], marker='o', mfc='none', mec=col,
                        mew=1.0, ms=6, ls='none', zorder=11)
            draw_waves(ax, t, qscale, xlim, ylim, anchor=wave_anchor)
            draw_key(ax)

            ax.set_title(f'({chr(97+k)}) {t:%d-%b %H:%M}Z   {suffix}', fontsize=9)
            if k + ncol >= len(times):
                ax.set_xlabel('longitude')
            if k % ncol == 0:
                ax.set_ylabel('latitude')

        for ax in axes[len(times):]:
            ax.set_visible(False)
        for k, ax in enumerate(axes[:len(times)]):
            if k + ncol < len(times):
                ax.tick_params(labelbottom=False)
            if k % ncol:
                ax.tick_params(labelleft=False)

        handles = list(data_handles)
        if outline:
            handles += [plt.Line2D([], [], color=LV3_COLOR, lw=1.2, ls='--',
                                   label='LV3 domain')]
        for site, M in MOOR.items():
            handles += [plt.Line2D([], [], color=M['color'], lw=3.5,
                                   label=f'{site} depth-int (z > {M["z_bad"]:g} m)'),
                        plt.Line2D([], [], color=M['color'], lw=1.0,
                                   label=f'{site} surface '
                                         f'({M["depth"][M["k_surf"]]:.2f} m)')]
        handles += [
                    plt.Line2D([], [], color=CDIP_COLOR, lw=3.0,
                               label='CDIP $H_s$ (arrow: direction, length $\\propto H_s$)')]
        fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=9,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))

        fig.suptitle(title, fontsize=13)
        _top = 1.0 - 0.55 / (row_h * nrow + 1.0)
        _bot = 0.40 / (row_h * nrow + 1.0)
        plt.tight_layout(rect=(0, _bot, 1, _top), h_pad=2.5)
        if fname:
            fig.savefig(fname, dpi=140, bbox_inches='tight')
            print(f'wrote {fname}')
            return fig


    def snapshot_panels(times, getter, **kw):
        """One quiver layer per HFR resolution.  getter(res, t) -> (u, v)."""
        def draw(ax, t, qscale):
            bits = []
            for res in HFR_RES:
                G, sub = HFR_GRID[res], HFR_SUB[res]
                u, v = getter(res, t)
                sl = (slice(None, None, sub), slice(None, None, sub))
                lo, la, uu, vv = G['lon'][sl], G['lat'][sl], u[sl], v[sl]
                g = np.isfinite(uu) & np.isfinite(vv)
                if g.any():
                    ax.quiver(lo[g], la[g], uu[g], vv[g], color=HFR_COLOR[res],
                              angles='uv', scale=qscale, scale_units='width',
                              width=QWIDTH, headwidth=3.5, headlength=4.0,
                              zorder=5 + HFR_RES.index(res))
                bits.append(f'{res[0]} km n={int(g.sum())}')
            return ', '.join(bits)

        handles = [plt.Line2D([], [], color=HFR_COLOR[r], lw=2, label=f'HFR {r[0]} km')
                   for r in HFR_RES]
        return panel_grid(times, draw, handles, **kw)

    # ------------------- figure 1: full LV3 domain ------------------------
    _stamp = f'{PANEL_TIMES[-1]:%Y%m%d_%H}Z'
    OUT_FN = OUT_DIR / f'HFR_LV3_nrt_{_stamp}_dt{DT_HOURS}h.png'
    _ = snapshot_panels(PANEL_TIMES, lambda res, t: HFR[res][t], fname=OUT_FN,
                        outline=True,
                        title=f'HF radar over LV3 — near real time   '
                              f'{PANEL_TIMES[0]:%Y-%m-%d %H:%M}Z .. '
                              f'{PANEL_TIMES[-1]:%Y-%m-%d %H:%M}Z, every {DT_HOURS} h')

    # ------------------- figure 2: shelf zoom -----------------------------
    OUT_ZOOM = OUT_DIR / f'HFR_LV3_nrt_zoom_{_stamp}_dt{DT_HOURS}h.png'
    _ = snapshot_panels(PANEL_TIMES, lambda res, t: HFR[res][t], fname=OUT_ZOOM,
                        xlim=ZOOM_BOX[:2], ylim=ZOOM_BOX[2:],
                        qscale=2.0, qkey=QKEY, scale_km=10.0, row_h=6.5,
                        wave_anchor=ZOOM_WAVE_ANCHOR,
                        title=f'HF radar, shelf zoom — near real time   '
                              f'{PANEL_TIMES[0]:%Y-%m-%d %H:%M}Z .. '
                              f'{PANEL_TIMES[-1]:%Y-%m-%d %H:%M}Z, every {DT_HOURS} h')

    # ------------------ coverage over the cached window -------------------
    COV_REGIONS = {'LV3 box': None, 'shelf box (ZOOM_BOX)': ZOOM_BOX}

    def region_mask(res, box):
        G = HFR_GRID[res]
        if box is None:
            return np.ones(G['lon'].shape, bool)
        return ((G['lon'] >= box[0]) & (G['lon'] <= box[1]) &
                (G['lat'] >= box[2]) & (G['lat'] <= box[3]))


    MASKS = {res: {n: region_mask(res, b) for n, b in COV_REGIONS.items()}
             for res in HFR_RES}
    print(f'{"res":>5} ' + ' '.join(f'{n:>22}' for n in COV_REGIONS) + '   (cells)')
    for res in HFR_RES:
        print(f'{res:>5} ' + ' '.join(f'{int(MASKS[res][n].sum()):22d}' for n in COV_REGIONS))

    COV = {}
    for res in HFR_RES:
        recs = {}
        for d in NEED_DAYS:
            recs.update(DAY_CACHE.get((res, d), {}))
        ts = sorted(recs)
        for name in COV_REGIONS:
            m = MASKS[res][name]
            n = int(m.sum())
            f = np.array([np.isfinite(recs[t][0][m]).sum() / n if n else np.nan
                          for t in ts])
            COV[(res, name)] = (np.array(ts, dtype='O'), f)
        print((f'{res}: {len(ts)} hours cached '
               f'({ts[0]:%d-%b %H}Z .. {ts[-1]:%d-%b %H}Z)') if ts else f'{res}: none')

    fig, axes = plt.subplots(len(COV_REGIONS), 1, figsize=(14, 7), sharex=True)
    for ax, name in zip(np.atleast_1d(axes), COV_REGIONS):
        for res in HFR_RES:
            t, f = COV[(res, name)]
            ax.plot(t, 100 * f, '-', color=HFR_COLOR[res], lw=1.0, label=f'HFR {res}')
        for t in PANEL_TIMES:
            ax.axvline(t, color='k', lw=0.6, alpha=0.5)
        ax.set_ylabel('valid cells [%]')
        ax.set_title(name, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    axes[-1].xaxis.set_major_formatter(DateFormatter('%d-%b %H'))
    fig.suptitle(f'HF radar coverage, last {COV_DAYS} days '
                 f'(black lines = panel times)', fontsize=12)
    fig.autofmt_xdate()
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    OUT_COV = OUT_DIR / f'HFR_nrt_coverage_{_stamp}.png'
    fig.savefig(OUT_COV, dpi=140, bbox_inches='tight')
    print(f'wrote {OUT_COV}')

    print(f'\n{"res":>5} ' + ' '.join(f'{n:>22}' for n in COV_REGIONS) + '   (mean %)')
    for res in HFR_RES:
        print(f'{res:>5} ' + ' '.join(
            f'{100*np.nanmean(COV[(res, n)][1]):22.1f}' for n in COV_REGIONS))
    return [OUT_FN, OUT_ZOOM, OUT_COV]


def main():
    t0 = time.time()
    log('=' * 68)
    log('make_obs_qc_figures starting')
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not acquire_lock():
        return 0
    try:
        written = run()
        prune_cache()
        if WRITE_LATEST and written:
            copy_latest(written)
        log(f'done in {time.time() - t0:.0f} s; {len(written or [])} figures '
            f'in {FIG_DIR}')
        return 0
    except Exception:
        log('FAILED')
        traceback.print_exc()
        return 1
    finally:
        LOCK_FILE.unlink(missing_ok=True)


if __name__ == '__main__':
    sys.exit(main())
