"""Calibrate the website shoreline dye to Enterococcus: alpha(shoreline) = Ent / dye_tot.

For each ddPCR station with enough recent samples,

    alpha_s = 10 ** mean( log10 Ent - log10 dye_tot )

over the pairs in the 30 days before the run (the geometric mean of Ent / dye_tot). Each
Ent sample is paired with the mean shoreline_dye_tot within +-3 h at the station's
shoreline point. The dye is the website file's own shoreline_dye_tot, so alpha is fitted
against the same dye it multiplies. Along the shoreline, log10 alpha is interpolated
linearly in shoreline index between stations, and held constant south of the southernmost
and north of the northernmost station with a fit.

Three caches, all CSV, in one directory (see cache_dir):
    ent_ddpcr.csv              station, time_utc, ent      (every ocean-side ddPCR sample)
    dye_shoreline_stations.csv time, <station>...          (hourly shoreline_dye_tot)
    alpha_history.csv          one row per run: status, alpha and n per station

The Ent cache is refreshed on every run from the shared QC cache
(/dataSIO/PFM_Simulations/cache/Dye_ddPCRENT, refreshed 07:01) and from a direct
download of the County portal; either may fail without harm. The dye cache is
appended from each new web file (forecast records 0-23, newest run wins); `backfill`
fills it from archived web files.

Nothing here may stop the web file being written: full_his_to_essential wraps the call,
and if no station has enough pairs the constant ALPHA0 is used, which reproduces the old
dye thresholds at the low end (1413 copies/100 mL <-> dye_tot = 1e-5).

Command line:
    python ent_alpha.py backfill [cache_dir] [web_dir]   # build the dye cache from archive
    python ent_alpha.py alpha    [cache_dir] [YYYYMMDDHH] # print alpha for a run time
"""
import glob
import os
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = "/dataSIO/PFM_Simulations/cache/Ent_alpha"      # operational track
DEFAULT_CACHE_DA = "/dataSIO/PFM_Simulations/cache/Ent_alpha_DA"
SHARED_ENT_CSV = "/dataSIO/PFM_Simulations/cache/Dye_ddPCRENT/ddpcr_ent_recent.csv"
WEB_ARCHIVE = "/dataSIO/PFM_Simulations/Archive/web"
DA_SCRATCH = "/scratch/PFM_Simulations/DA_Sim"                   # DA and sandbox runs

ENT_CSV = "ent_ddpcr.csv"
DYE_CSV = "dye_shoreline_stations.csv"
ALPHA_CSV = "alpha_history.csv"

# ocean-side ddPCR stations, south to north (bayside EH-07x/08x/09x/12x excluded).
# Stations that are no longer sampled simply have no pairs and drop out.
STATIONS = {
    "IB-010": (32.53595, -117.12430),   # Border Fence N side
    "IB-020": (32.54341, -117.12523),   # Monument Rd.
    "IB-030": (32.55514, -117.12932),   # Tijuana Estuary mouth
    "IB-050": (32.56658, -117.13321),   # End of Seacoast Dr
    "EH-010": (32.57267, -117.13320),   # Cortez Ave
    "EH-030": (32.57875, -117.13319),   # Imperial Beach Pier
    "IB-060": (32.58568, -117.13305),   # Carnation Ave.
    "IB-068": (32.62640, -117.13942),   # Silver Strand - Guard Shack
    "IB-069": (32.63177, -117.14189),   # Silver Strand - Lifeguard Tower
    "IB-070": (32.63580, -117.14317),   # Silver Strand N end (ocean)
    "IB-079": (32.67407, -117.17230),   # Avenida Lunar
    "EH-050": (32.68184, -117.18526),   # Loma Ave
    "EH-060": (32.68559, -117.19432),   # Navy Fence (A)
}

WINDOW_DAYS = 30          # trailing calibration window
PAIR_HOURS = 3            # Ent sample paired with mean dye within +- this
MIN_PAIRS = 5             # pairs needed in the window to fit a station
DYE_LEADS = range(0, 24)  # forecast records cached from each run
ENT_DOWNLOAD_DAYS = 45
THRESH_ENT = np.array([1413.0, 1.0e5])            # copies/100 mL: low|medium|high
ALPHA0 = THRESH_ENT[0] / 1.0e-5                   # fallback: 1413 <-> dye_tot 1e-5
TZ = ZoneInfo("America/Los_Angeles")


# ------------------------------------------------------------------------- helpers
def cache_dir(PFM=None):
    """ENT_ALPHA_CACHE env > PFM['ent_alpha_cache_dir'] > ops or DA default."""
    if os.environ.get("ENT_ALPHA_CACHE"):
        return os.environ["ENT_ALPHA_CACHE"]
    PFM = PFM or {}
    if PFM.get("ent_alpha_cache_dir"):
        return PFM["ent_alpha_cache_dir"]
    is_da = ("da_t_fore" in PFM
             or str(PFM.get("lv4_web_name_full", "")).startswith(DA_SCRATCH))
    return DEFAULT_CACHE_DA if is_da else DEFAULT_CACHE


def seed_dye_cache(cdir, log=print):
    """A new non-operational cache starts from the operational dye history.

    The DA track keeps its web files for 7 days only, so it has no archive to backfill
    from. Seeded with the ops dye, its 30-day window is all DA dye after 30 days of runs
    (each run overwrites its own hours).
    """
    dst = os.path.join(cdir, DYE_CSV)
    src = os.path.join(DEFAULT_CACHE, DYE_CSV)
    if os.path.abspath(cdir) != os.path.abspath(DEFAULT_CACHE) and not os.path.exists(dst) \
            and os.path.exists(src):
        import shutil
        shutil.copyfile(src, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        log(f"ent_alpha: seeded {dst} from {src}")


def station_rows(shore_lat, shore_lon):
    """Shoreline index nearest each station (distance in metres, flat-earth)."""
    lat = np.asarray(shore_lat, float).ravel()
    lon = np.asarray(shore_lon, float).ravel()
    k = np.cos(np.deg2rad(32.6))
    rows = {}
    for s, (la, lo) in STATIONS.items():
        d2 = ((lat - la) * 111e3) ** 2 + ((lon - lo) * 111e3 * k) ** 2
        rows[s] = int(np.nanargmin(d2))
    return rows


def _write_atomic(df, path, **kw):
    tmp = path + ".tmp"
    df.to_csv(tmp, **kw)
    os.replace(tmp, path)


# ------------------------------------------------------------------------- Ent cache
def _read_ent(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["station", "time_utc", "ent"])
    d = pd.read_csv(path)
    d["time_utc"] = pd.to_datetime(d["time_utc"])
    return d[["station", "time_utc", "ent"]]


def _download_ent(cdir):
    """Recent ddPCR from the County portal (BeachWatch fallback), as station/time_utc/ent."""
    sys.path.insert(0, os.path.join(HERE, "..", "ddPCR_ENT"))
    from download_water_quality import download_sd_water_quality
    end = pd.Timestamp.now(tz=TZ).normalize() + pd.Timedelta(days=1)
    df = download_sd_water_quality(
        output_csv=os.path.join(cdir, "ent_download_tmp.csv"),
        start_date=(end - pd.Timedelta(days=ENT_DOWNLOAD_DAYS)).strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        parameter="Enterococcus", ddpcr_only=True)
    if df is None or len(df) == 0:
        return None
    st = df["Station Name"].astype(str).str.strip().str.upper()
    local = pd.to_datetime(df["SampleDate"].dt.strftime("%Y-%m-%d") + " "
                           + df["SampleTime"].fillna("12:00:00"), errors="coerce")
    t = local.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
    return pd.DataFrame({"station": st, "time_utc": t.dt.tz_convert("UTC").dt.tz_localize(None),
                         "ent": pd.to_numeric(df["Result_Numeric"], errors="coerce")})


def update_ent_cache(cdir, download=True, log=print):
    """Merge the shared QC cache and a fresh download into cdir/ent_ddpcr.csv."""
    path = os.path.join(cdir, ENT_CSV)
    parts = [_read_ent(path)]
    try:
        parts.append(_read_ent(SHARED_ENT_CSV))
    except Exception as e:                                     # noqa: BLE001
        log(f"ent_alpha: shared Ent cache not read ({e})")
    if download:
        try:
            d = _download_ent(cdir)
            if d is not None:
                parts.append(d)
        except Exception as e:                                 # noqa: BLE001
            log(f"ent_alpha: Ent download failed ({e}); using cached values")
    d = pd.concat([p for p in parts if len(p)], ignore_index=True)
    d["ent"] = pd.to_numeric(d["ent"], errors="coerce").astype(float)
    d["time_utc"] = pd.to_datetime(d["time_utc"])
    d = d[d["station"].isin(STATIONS) & d["time_utc"].notna() & (d["ent"] > 0)]
    d["time_utc"] = d["time_utc"].dt.floor("min")
    d = d.drop_duplicates(["station", "time_utc"], keep="last").sort_values(["time_utc", "station"])
    _write_atomic(d, path, index=False, date_format="%Y-%m-%d %H:%M")
    return d


# ------------------------------------------------------------------------- dye cache
def _read_dye(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=list(STATIONS), index=pd.DatetimeIndex([], name="time"))
    return pd.read_csv(path, parse_dates=["time"], index_col="time").astype(float)


def update_dye_cache(cdir, times, shore_dye, shore_lat, shore_lon):
    """Add forecast records DYE_LEADS of one run (times: UTC, naive) to the dye cache.

    shore_dye is shoreline_dye_tot[ntime, nshore]; later calls overwrite earlier hours,
    so with runs added in time order the newest run wins.
    """
    rows = station_rows(shore_lat, shore_lon)
    lead = [k for k in DYE_LEADS if k < len(times)]
    add = pd.DataFrame({s: np.asarray(shore_dye)[lead, j] for s, j in rows.items()},
                       index=pd.DatetimeIndex(pd.to_datetime([times[k] for k in lead]), name="time"))
    path = os.path.join(cdir, DYE_CSV)
    old = _read_dye(path)
    keep = old[~old.index.isin(add.index)]
    new = (pd.concat([keep, add]) if len(keep) else add).sort_index().astype(float)
    _write_atomic(new, path, date_format="%Y-%m-%dT%H:%M:%S", float_format="%.5g")
    return new


def backfill(cdir=None, web_dir=WEB_ARCHIVE, log=print):
    """Build the dye cache from archived web_data_*.nc files, oldest first."""
    import netCDF4 as nc
    cdir = cdir or DEFAULT_CACHE
    os.makedirs(cdir, exist_ok=True)
    epoch = pd.Timestamp("1999-01-01")
    parts = []
    files = sorted(glob.glob(os.path.join(web_dir, "web_data_*.nc")))
    for f in files:
        try:
            with nc.Dataset(f) as d:
                t = epoch + pd.to_timedelta(np.asarray(d["time"][:], float), "D")
                dye = np.ma.filled(d["shoreline_dye_tot"][:].astype(float), np.nan)
                rows = station_rows(d["shoreline_lat"][:], d["shoreline_lon"][:])
        except Exception as e:                                 # noqa: BLE001
            log(f"skip {os.path.basename(f)}: {e}")
            continue
        lead = [k for k in DYE_LEADS if k < len(t)]
        parts.append(pd.DataFrame({s: dye[lead, j] for s, j in rows.items()},
                                  index=pd.DatetimeIndex(t[lead].round("h"), name="time")))
    if not parts:
        log("backfill: no web files read")
        return None
    new = pd.concat(parts)
    path = os.path.join(cdir, DYE_CSV)
    old = _read_dye(path)
    keep = old[~old.index.isin(new.index)]
    new = pd.concat([keep, new]) if len(keep) else new
    new = new[~new.index.duplicated(keep="last")].sort_index()
    _write_atomic(new, path, date_format="%Y-%m-%dT%H:%M:%S", float_format="%.5g")
    log(f"backfill: {len(files)} files -> {path}: {len(new)} hours, "
        f"{new.index.min()} to {new.index.max()}")
    return new


# ------------------------------------------------------------------------- alpha
def station_alpha(ent, dye, t_run, window_days=WINDOW_DAYS):
    """Per-station geometric-mean Ent/dye over pairs in [t_run - window, t_run)."""
    t_run = pd.Timestamp(t_run)
    t0 = t_run - pd.Timedelta(days=window_days)
    e = ent[(ent["time_utc"] >= t0) & (ent["time_utc"] < t_run)]
    hw = np.timedelta64(PAIR_HOURS, "h")
    out = {}
    for s in STATIONS:
        es = e[e["station"] == s]
        if s not in dye.columns or es.empty:
            out[s] = (np.nan, 0)
            continue
        ds = dye[s].dropna()
        ds = ds[ds > 0]
        td, vd = ds.index.to_numpy(), ds.to_numpy()
        ts = es["time_utc"].to_numpy()
        lo = np.searchsorted(td, ts - hw, "left")
        hi = np.searchsorted(td, ts + hw, "right")
        c = np.array([vd[a:b].mean() if b > a else np.nan for a, b in zip(lo, hi)])
        ok = np.isfinite(c)
        n = int(ok.sum())
        if n < MIN_PAIRS:
            out[s] = (np.nan, n)
            continue
        r = np.log10(es["ent"].to_numpy()[ok]) - np.log10(c[ok])
        out[s] = (10 ** r.mean(), n)
    return out


def shoreline_alpha(alpha_st, rows, nshore):
    """log10-linear interpolation in shoreline index; constant beyond the end stations."""
    pts = {}
    for s, (a, _) in alpha_st.items():
        if np.isfinite(a) and a > 0:
            pts.setdefault(rows[s], []).append(np.log10(a))
    if not pts:
        return None
    j = np.array(sorted(pts))
    la = np.array([np.mean(pts[k]) for k in j])
    return 10 ** np.interp(np.arange(nshore), j, la)


def compute(t_run, shore_lat, shore_lon, cdir, download=True, log=print):
    """Refresh the Ent cache and return (alpha_shore[nshore], info dict).

    info: status ('ok', 'last_good' or 'constant'), per-station alpha, n and shoreline row.
    Appends a row to alpha_history.csv.
    """
    os.makedirs(cdir, exist_ok=True)
    nshore = len(np.asarray(shore_lat).ravel())
    rows = station_rows(shore_lat, shore_lon)
    ent = update_ent_cache(cdir, download=download, log=log)
    dye = _read_dye(os.path.join(cdir, DYE_CSV))
    ast = station_alpha(ent, dye, t_run)
    status = "ok"
    alpha = shoreline_alpha(ast, rows, nshore)
    hist_path = os.path.join(cdir, ALPHA_CSV)
    if alpha is None:                         # no station fitted: last good run, <= 7 days old
        status = "constant"
        if os.path.exists(hist_path):
            h = pd.read_csv(hist_path, parse_dates=["run"])
            h = h[(h["status"] == "ok") & (h["run"] < pd.Timestamp(t_run))
                  & (h["run"] >= pd.Timestamp(t_run) - pd.Timedelta(days=7))]
            if len(h):
                last = h.iloc[-1]
                prev = {s: (float(last.get(f"alpha_{s}", np.nan)), 0) for s in STATIONS}
                alpha = shoreline_alpha(prev, rows, nshore)
                if alpha is not None:
                    status = "last_good"
                    ast = prev
        if alpha is None:
            alpha = np.full(nshore, ALPHA0)
    rec = {"run": pd.Timestamp(t_run), "status": status,
           "n_stations": int(sum(np.isfinite(a) for a, _ in ast.values()))}
    for s, (a, n) in ast.items():
        rec[f"alpha_{s}"] = a
        rec[f"n_{s}"] = n
    h = pd.read_csv(hist_path, parse_dates=["run"]) if os.path.exists(hist_path) else pd.DataFrame()
    h = pd.concat([h[h["run"] != rec["run"]] if len(h) else h, pd.DataFrame([rec])])
    _write_atomic(h.sort_values("run"), hist_path, index=False, float_format="%.4g",
                  date_format="%Y-%m-%d %H:%M")
    info = {"status": status, "alpha": ast, "rows": rows}
    log(f"ent_alpha: {status}, " + ", ".join(
        f"{s} {a:.2e} (n={n})" for s, (a, n) in ast.items() if np.isfinite(a)))
    return alpha, info


def risk_from_ent(ent):
    """0 below THRESH_ENT[0], 1 between, 2 at or above THRESH_ENT[1] (as the dye risk)."""
    r = np.zeros(np.shape(ent))
    r[(ent > THRESH_ENT[0]) & (ent < THRESH_ENT[1])] = 1
    r[ent >= THRESH_ENT[1]] = 2
    return r


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("backfill", "alpha"):
        print(__doc__)
        sys.exit(1)
    if a[0] == "backfill":
        backfill(a[1] if len(a) > 1 else None, a[2] if len(a) > 2 else WEB_ARCHIVE)
    else:
        import netCDF4 as nc
        cd = a[1] if len(a) > 1 else DEFAULT_CACHE
        f = sorted(glob.glob(os.path.join(WEB_ARCHIVE, "web_data_*.nc")))[-1]
        with nc.Dataset(f) as d:
            la, lo = d["shoreline_lat"][:], d["shoreline_lon"][:]
        t = pd.Timestamp.utcnow().tz_localize(None) if len(a) < 3 else pd.to_datetime(a[2], format="%Y%m%d%H")
        al, info = compute(t, la, lo, cd, download=False)
        for s, j in info["rows"].items():
            print(f"{s}  row {j:4d}  alpha {info['alpha'][s][0]:.3g}  n {info['alpha'][s][1]}"
                  f"  shoreline alpha {al[j]:.3g}")
