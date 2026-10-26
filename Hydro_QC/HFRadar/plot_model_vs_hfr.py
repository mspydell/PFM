#!/usr/bin/env python3
"""Model surface currents (PFM LV3 or LV4, top 2 m) vs HF radar 6 km, every 6 hours.

One figure per model level, 3 rows x 4 columns, every 6 h in local time (America/Los_Angeles,
PDT/PST): each row is one local day at 06:00, 12:00, 18:00 and the next day's 00:00. The rows are
the day before yesterday, yesterday and today (--date sets "today"). Made for the QC website run at
~06:00 local: HFR then reaches ~03:00 local, so today's row is mostly model forecast ("no HFR yet").
Model for each panel: day 1 of the PFM run for the panel's UTC date; if that run is not written yet
(it usually lands ~04:40 local, some days after noon), day 2 of the previous run, marked in the title.

Model currents
  u, v averaged over the top SURFACE_M = 2 m below the moving free surface (each s-layer weighted
  by how much of it lies in 0-2 m; whole column where shallower), roughly the depth the 6 km HFR
  senses, at rho points and rotated from grid to east/north with the grid angle, are linearly interpolated onto a regular ~100 m lon/lat grid. The grid
  spacing is exactly 1/60 of the HFR 6 km spacing and its origin is an HFR node, so every
  30th point (~3 km; LV4 every 20th, ~2 km) is drawn as an arrow and every 60th point is an HFR
  6 km cell centre.
  LV3 is a regular ~207 m grid; LV4 is irregular, 8 m near the coast to ~86 m offshore.
  Interpolation cells that touch land are left blank (no extrapolation onto the coast).

HFR 6 km
  hourly NDBC near-real-time product from the shared cache that make_obs_qc_figures.py
  fills (/dataSIO/DA_Simulations/input_files/hfr_nrt); hours the cache lacks are read
  live from NDBC.

Moorings (blue): observed ADCP current of the top layer, hourly (samples within +-30 min of the
  hour), from the ../Moorings cache (plot_model_vs_obs.obs_currents): SBOO on LV4; SBOO, PLOO and
  Del Mar on LV3. Top layer = bins shallower than 5.5 m (SBOO), 6 m (PLOO), 10 m (Del Mar).
  --update-moorings downloads new mooring data from SIO first.
  Model at the moorings (thick black): u, v at the nearest model u/v points, averaged over the same
  top layer as the ADCP (extract_model.SITES "top": SBOO 2.83-5.83 m, PLOO 2.93-5.93 m, Del Mar
  4.42-10.42 m below mean sea level, layer-thickness weighted) and rotated to east/north, read from
  the same run and record as the panel: from the Moorings model cache (u_top/v_top of
  <site>_<LV>_day1_hourly.csv) where it has the hour, else from the history file
  (extract_model.read_file / layer_mean).

Caches, in /dataSIO/Hydro_QC/HFRadar/cache (HYDRO_QC_DATA overrides the root):
  weights_<LV>_<key>.npz    interpolation weights, model grid -> ~100 m grid, built once per grid
  model_<LV>_<run>_rec<NNN>_top2m_every<N>_<key>.npz  model arrows of one panel (a run never changes)
  mooring_<LV>_<run>_<site>.npz  model at a mooring, days 1-2, where the Moorings cache lacks it

Stats in each panel title, at HFR 6 km cell centres where both exist: n cells, mean speed
(model / HFR) and RMS vector difference.

  python plot_model_vs_hfr.py                     # LV3 and LV4, rows: day before yesterday .. today
  python plot_model_vs_hfr.py --date 2026-10-18   # rows 16, 17 and 18 Oct (last panel 19 Oct 00:00)
  python plot_model_vs_hfr.py --level LV4
  python plot_model_vs_hfr.py --update-moorings   # refresh the mooring obs first
"""
import argparse
import datetime as DT
import glob
import hashlib
import os
import re
import sys
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4 as nc  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from scipy.spatial import Delaunay  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MOORINGS_COMMON = os.path.join(HERE, "..", "Moorings", "common")
OUT_DIR = os.path.join(HERE, "plots_web")   # website figures, as in the other Hydro_QC folders
# shared cache on /dataSIO (like the Moorings caches); HYDRO_QC_DATA overrides the root
CACHE_DIR = os.path.join(os.environ.get("HYDRO_QC_DATA", "/dataSIO/Hydro_QC"), "HFRadar", "cache")

HIS_DIR = "/dataSIO/PFM_Simulations/Archive/{lv}_His"
GRID = {"LV3": "/dataSIO/PFM_Simulations/Grid/GRID_SDTJRE_LV3_rx020.nc",
        "LV4": "/dataSIO/PFM_Simulations/Grid/GRID_SDTJRE_LV4_mss_oct2024.nc"}
HFR_CACHE = "/dataSIO/DA_Simulations/input_files/hfr_nrt/hfr_nrt_ndbc_6km_{day:%Y%m%d}.nc"
HFR_NDBC = "https://dods.ndbc.noaa.gov/thredds/dodsC/hfradar_uswc_6km"
# older HFR 6 km (hourly, same grid, lon 0-360): monthly Nov 2024 - Dec 2025, then 1 Jan - 18 Mar 2026
HFR_ARCHIVE = ["/dataSIO/HFRadar_files/socal_hfr_6km_*/*_6km.mat", "/dataSIO/HFRadar_files/hfradar_uswc_6km_*.mat"]

MOOR_ALPHA = 0.6                 # observed mooring arrows see-through: the model arrow shows beneath
QKEY = 0.25                      # reference (key) arrow, m/s
LOCAL_TZ = ZoneInfo("America/Los_Angeles")   # panel times in local time (PDT / PST)
LOCAL_HOURS = (6, 12, 18, 24)    # columns: 06:00, 12:00, 18:00 and the next day's 00:00 local
NROW = 3                         # local days (rows); the last panel is 00:00 local of the end day
SURFACE_M = 2.0      # model grid arrows: mean over the top 2 m below the free surface (~the HFR 6 km depth)
FINE = 60            # fine-grid points per HFR 6 km spacing (~100 m)
MOORINGS = {"LV3": ("SBOO", "PLOO", "DelMar"), "LV4": ("SBOO",)}
LEVELS = ("LV3", "LV4")
ARROW_EVERY = {"LV3": 30, "LV4": 20}   # fine-grid points between model arrows (~3 km, ~2 km); divides FINE
BATHY = {"LV3": [10, 20, 50], "LV4": [10, 20, 50]}   # isobaths (m): near the coast, then the 50 m

MODEL_COLOR = {"LV3": "0.45", "LV4": "0.3"}   # grey, thin: the HFR stands out on top (LV4 has fewer red arrows)
MODEL_LW = 0.8
HFR_COLOR = "#ff0000"
HFR_LW = {"LV3": 1.15, "LV4": 1.5}   # LV3 arrows are shorter on the page
MOOR_LW = 1.7
ARROW_LEN = 1.35         # the 95th percentile speed is drawn this many model arrow spacings long
HEAD = 0.3               # open arrowhead: two strokes, this fraction of the arrow long, 25 deg to the shaft
# observed mooring arrows, one colour per mooring (none red or black: those are HFR and the model)
MOOR_COLOR = {"SBOO": "#0072b2", "PLOO": "#009e73", "DelMar": "#8e44ad"}
MOOR_NAME = {"SBOO": "SBOO", "PLOO": "PLOO", "DelMar": "Del Mar"}
MOOR_MODEL_COLOR = "k"
LAND_COLOR = "#cdb79e"
COAST_COLOR = "#7a5d33"

KM_LAT = 110.57      # km per degree latitude


def now_utc():
    """Naive UTC now."""
    return DT.datetime.now(DT.timezone.utc).replace(tzinfo=None)


def km_lon(lat):
    return 111.32 * np.cos(np.radians(lat))


# ----------------------------------------------------------------------------- model
def history_files(lv, day):
    """[(start, path)] of the runs started on this UTC date: <lv>_ocean_his_<YYYYMMDDHH>00.nc, usually one
    at 00 UTC, some at 06 or 12 UTC (failed runs such as *_dt2_BLOWUP.nc are skipped)."""
    out = []
    for f in glob.glob(os.path.join(HIS_DIR.format(lv=lv), f"{lv}_ocean_his_{day:%Y%m%d}[0-9][0-9]00.nc")):
        out.append((DT.datetime.strptime(os.path.basename(f)[len(lv) + 11:len(lv) + 21], "%Y%m%d%H"), f))
    return out


def run_id(lv, f):
    """Run label and cache key: YYYYMMDD for a 00 UTC run, YYYYMMDDHH for one started later."""
    stamp = os.path.basename(f)[len(lv) + 11:len(lv) + 21]
    return stamp[:8] if stamp.endswith("00") else stamp


def load_grid(lv):
    with nc.Dataset(GRID[lv]) as g:
        G = {k: np.asarray(g[k][:], float) for k in ("lon_rho", "lat_rho", "angle", "mask_rho", "h")}
    G["wet"] = G["mask_rho"] == 1
    return G


def surface_uv(path, rec, G, top_m=None):
    """East/north velocity (m/s) at rho points for one record, averaged over the top `top_m` metres
    below the moving free surface (default SURFACE_M); NaN on land and the edge.

    Each s-layer is weighted by how much of it lies in 0..top_m (as extract_model.layer_mean), so
    where the water is shallower than top_m it is the whole-column mean. ROMS Vtransform 2."""
    top_m = SURFACE_M if top_m is None else top_m
    with nc.Dataset(path) as d:
        if int(d["Vtransform"][...]) != 2:
            raise ValueError("only Vtransform 2 is implemented")
        sw, csw, hc = (np.asarray(d[k][:], float) for k in ("s_w", "Cs_w", "hc"))
        zeta = np.ma.filled(d["zeta"][rec].astype(float), np.nan)
        h = G["h"]
        z0w = (hc * sw[:, None, None] + h * csw[:, None, None]) / (hc + h)     # [N+1, eta, xi], at zeta = 0
        # levels needed: every layer whose top is within top_m of the surface anywhere, +2 for low tide
        k0 = max(0, int(np.argmax(np.any((-z0w * h <= top_m) & G["wet"], axis=(1, 2)))) - 1 - 2)
        u = np.ma.filled(d["u"][rec, k0:].astype(float), np.nan)              # [K, eta, xi_u]
        v = np.ma.filled(d["v"][rec, k0:].astype(float), np.nan)
        t = nc.num2date(d["ocean_time"][rec], d["ocean_time"].units, only_use_cftime_datetimes=False)
    zw = zeta + (zeta + h) * z0w[k0:]                                          # w levels of the K layers
    dep_top = zeta - zw[1:]                                                    # layer top, below the surface
    dep_bot = zeta - zw[:-1]
    w = np.clip(np.minimum(dep_bot, top_m) - np.maximum(dep_top, 0.0), 0, None)
    shape = G["mask_rho"].shape
    ur = np.full((u.shape[0],) + shape, np.nan)
    vr = np.full((v.shape[0],) + shape, np.nan)
    ur[:, :, 1:-1] = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    vr[:, 1:-1, :] = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    with np.errstate(invalid="ignore"):
        wsum = np.where(np.isnan(ur), 0, w).sum(axis=0)
        ur = np.nansum(ur * w, axis=0) / np.where(wsum > 0, wsum, np.nan)
        vr = np.nansum(vr * w, axis=0) / np.where(wsum > 0, wsum, np.nan)
    ca, sa = np.cos(G["angle"]), np.sin(G["angle"])
    ue, vn = ur * ca - vr * sa, ur * sa + vr * ca
    ue[~G["wet"]] = np.nan
    vn[~G["wet"]] = np.nan
    # to the nearest minute: some runs store 00:59:59.9999 for 01:00
    return ue, vn, (DT.datetime(t.year, t.month, t.day, t.hour, t.minute)
                    + DT.timedelta(minutes=round((t.second + t.microsecond / 1e6) / 60)))


class FineGrid:
    """Regular lon/lat grid with spacing HFR/FINE, anchored on an HFR node, covering the model grid,
    plus linear-interpolation weights from the model rho points (Delaunay, computed once)."""

    def __init__(self, G, hfr_lon, hfr_lat, every, lv):
        dlon = float(np.diff(hfr_lon).mean()) / FINE
        dlat = float(np.diff(hfr_lat).mean()) / FINE
        lo0, la0 = hfr_lon[0], hfr_lat[0]
        i0 = np.floor((G["lon_rho"].min() - lo0) / dlon)
        i1 = np.ceil((G["lon_rho"].max() - lo0) / dlon)
        j0 = np.floor((G["lat_rho"].min() - la0) / dlat)
        j1 = np.ceil((G["lat_rho"].max() - la0) / dlat)
        self.i = np.arange(i0, i1 + 1).astype(int)          # index counted from the HFR origin
        self.j = np.arange(j0, j1 + 1).astype(int)
        self.lon = lo0 + self.i * dlon
        self.lat = la0 + self.j * dlat
        self.dx_m = dlon * km_lon(G["lat_rho"].mean()) * 1e3
        self.dy_m = dlat * KM_LAT * 1e3
        LON, LAT = np.meshgrid(self.lon, self.lat)
        self.shape = LON.shape
        # weights depend only on the model grid and the fine grid: cached, rebuilt if either changes
        sig = f"{GRID[lv]}|{os.path.getmtime(GRID[lv]):.0f}|{FINE}|{lo0:.6f}|{la0:.6f}|{dlon:.9f}|{dlat:.9f}"
        self.key = hashlib.sha1(sig.encode()).hexdigest()[:10]
        path = os.path.join(CACHE_DIR, f"weights_{lv}_{self.key}.npz")
        if os.path.exists(path):
            z = np.load(path)
            self.w, self.vert, self.outside = z["w"].astype(float), z["vert"], z["outside"]
        else:
            print(f"  building interpolation weights ({lv}, once) -> {path}")
            lat0 = G["lat_rho"].mean()
            src = np.column_stack([G["lon_rho"].ravel() * km_lon(lat0), G["lat_rho"].ravel() * KM_LAT])
            dst = np.column_stack([LON.ravel() * km_lon(lat0), LAT.ravel() * KM_LAT])
            tri = Delaunay(src)
            s = tri.find_simplex(dst)
            T = tri.transform[s]
            b = np.einsum("ijk,ik->ij", T[:, :2], dst - T[:, 2])
            self.w = np.column_stack([b, 1 - b.sum(axis=1)])
            self.vert = tri.simplices[s].astype(np.int32)
            self.outside = s < 0
            save_npz(path, w=self.w, vert=self.vert, outside=self.outside)
        # model arrows every `every` points, HFR cell centres every FINE points (both on HFR nodes);
        # every divides FINE, so the centres are a subset of the arrow points (node_in_arrow)
        self.every = every
        self.arrow = np.ix_(self.j % every == 0, self.i % every == 0)
        ja, ia = self.j[self.j % every == 0], self.i[self.i % every == 0]
        self.node_in_arrow = np.ix_(ja % FINE == 0, ia % FINE == 0)
        self.node_ij = (self.j[self.j % FINE == 0] // FINE, self.i[self.i % FINE == 0] // FINE)

    def interp(self, f):
        """Linear interpolation of a rho-point field; NaN where any triangle corner is NaN or outside."""
        out = (f.ravel()[self.vert] * self.w).sum(axis=1)
        out[self.outside] = np.nan
        return out.reshape(self.shape)


def save_npz(path, **arrays):
    """np.savez_compressed via a temp file, group-writable (the cache is shared)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.part.npz"     # per process: parallel runs may write the same file
    old = os.umask(0o002)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        os.umask(old)


def model_panel(lv, G, F, f, rec, t):
    """Model east/north (top SURFACE_M m mean) at the arrow points for one history record, cached per
    run, record, depth, arrow spacing and weights: a run never changes once written."""
    run = run_id(lv, f)
    path = os.path.join(CACHE_DIR, f"model_{lv}_{run}_rec{rec:03d}_top{SURFACE_M:g}m_every{F.every}_{F.key}.npz")
    if os.path.exists(path):
        z = np.load(path)
        if int(z["t"]) == int((t - DT.datetime(1970, 1, 1)).total_seconds()):
            return z["mu"].astype(float), z["mv"].astype(float)
    ue, vn, tm = surface_uv(f, rec, G)
    assert tm == t, f"{f} record {rec} is {tm}, expected {t}"
    mu, mv = F.interp(ue)[F.arrow], F.interp(vn)[F.arrow]
    save_npz(path, mu=mu.astype(np.float32), mv=mv.astype(np.float32),
             t=np.int64((t - DT.datetime(1970, 1, 1)).total_seconds()))
    return mu, mv


# ----------------------------------------------------------------------------- HFR
def hfr_axes():
    """lon, lat of the cached 6 km grid (LV3 box)."""
    f = sorted(glob.glob(HFR_CACHE.replace("{day:%Y%m%d}", "*")))[-1]
    with nc.Dataset(f) as d:
        return np.asarray(d["longitude"][:], float), np.asarray(d["latitude"][:], float)


_NDBC = {}


def hfr_uv(t, lon, lat):
    """HFR 6 km (u, v) [lat x lon] at hour t from the near-real-time cache, else the .mat archive, else
    live from NDBC; None if none has it."""
    f = HFR_CACHE.format(day=t)
    if os.path.exists(f):
        with nc.Dataset(f) as d:
            secs = np.asarray(d["time"][:], float)
            k = np.where(np.isclose(secs, (t - DT.datetime(1970, 1, 1)).total_seconds()))[0]
            if k.size:
                return (np.ma.filled(d["water_u"][k[0]].astype(float), np.nan),
                        np.ma.filled(d["water_v"][k[0]].astype(float), np.nan))
    A = hfr_archive(lon, lat)
    if t in A:
        z, k = A[t]
        return z["u"][k].astype(float), z["v"][k].astype(float)
    if t < DT.datetime(2026, 7, 1):     # NDBC keeps only the last ~3 months
        return None
    try:
        if "ds" not in _NDBC:
            d = nc.Dataset(HFR_NDBC)
            tv = d["time"]
            _NDBC.update(ds=d, t=[x.replace(microsecond=0) for x in
                                  nc.num2date(tv[:], tv.units, only_use_cftime_datetimes=False)],
                         jj=np.searchsorted(np.asarray(d["lat"][:], float), lat[0] - 1e-4),
                         ii=np.searchsorted(np.asarray(d["lon"][:], float), lon[0] - 1e-4))
        if t not in _NDBC["t"]:
            return None
        k = _NDBC["t"].index(t)
        js, is_ = slice(_NDBC["jj"], _NDBC["jj"] + lat.size), slice(_NDBC["ii"], _NDBC["ii"] + lon.size)
        d = _NDBC["ds"]
        return (np.ma.filled(d["u"][k, js, is_].astype(float), np.nan),
                np.ma.filled(d["v"][k, js, is_].astype(float), np.nan))
    except Exception as e:
        print(f"  HFR {t:%Y-%m-%d %H}Z: NDBC unavailable ({type(e).__name__})")
        return None


_ARCH = {}


def hfr_archive(lon, lat):
    """{naive UTC hour: (arrays, index)} of the HFR .mat archive on the lon/lat box. Each .mat file is
    cut to the box once into CACHE_DIR/hfr6km_<file>_<key>.npz (float32 [hours x lat x lon]); the .mat
    files are large (~0.5-1.4 GB per variable), the cut ones ~2 MB."""
    if _ARCH:
        return _ARCH
    key = hashlib.md5(np.r_[lon, lat].round(5).tobytes()).hexdigest()[:10]
    for src in sorted(f for pat in HFR_ARCHIVE for f in glob.glob(pat)):
        path = os.path.join(CACHE_DIR, f"hfr6km_{os.path.splitext(os.path.basename(src))[0]}_{key}.npz")
        if not os.path.exists(path):
            import scipy.io as sio
            print(f"  cutting {os.path.basename(src)} to the HFR box (once) -> {path}")
            m = sio.loadmat(src, squeeze_me=True, variable_names=["x", "y", "timeraw"])
            ii = [int(np.argmin(abs(m["x"] - 360 - v))) for v in lon]
            jj = [int(np.argmin(abs(m["y"] - v))) for v in lat]
            assert max(abs(m["x"][ii] - 360 - lon)) < 1e-4 and max(abs(m["y"][jj] - lat)) < 1e-4, src
            secs = np.round((m["timeraw"] - 719529) * 86400 / 3600).astype(np.int64) * 3600   # datenum -> hour
            out = {"t": secs}
            for var, name in (("uraw", "u"), ("vraw", "v")):       # [x, y, time] -> [time, lat, lon]
                a = sio.loadmat(src, squeeze_me=True, variable_names=[var])[var]
                out[name] = a[np.ix_(ii, jj)].transpose(2, 1, 0).astype(np.float32)
                del a
            save_npz(path, **out)
        z = dict(np.load(path))
        for k, ts in enumerate(z["t"]):
            _ARCH.setdefault(DT.datetime(1970, 1, 1) + DT.timedelta(seconds=int(ts)), (z, k))
    return _ARCH


# ----------------------------------------------------------------------------- figure
def panels(lv, end_day):
    """The 3 x 4 panels, local time: each row is one local day at 06, 12, 18 and the next day's 00,
    the last panel is end_day 00:00 local. Per panel: t (naive UTC), tl (local), history file, record
    and fc_day: day 1 of the run for t's UTC date, else day 2 of the previous run (that run not yet
    written), else no run."""
    return [panel_at(lv, DT.datetime.combine(end_day - DT.timedelta(days=r), DT.time()) + DT.timedelta(hours=h))
            for r in range(NROW, 0, -1) for h in LOCAL_HOURS]


_NREC = {}


def n_records(f):
    """Number of records in a history file."""
    if f not in _NREC:
        with nc.Dataset(f) as d:
            _NREC[f] = len(d["ocean_time"])
    return _NREC[f]


def panel_at(lv, wall):
    """One panel at local wall-clock time `wall` (naive): t (naive UTC), tl (local), history file, record,
    fc_day and run."""
    tl = wall.replace(tzinfo=LOCAL_TZ)
    t = tl.astimezone(DT.timezone.utc).replace(tzinfo=None)
    # the newest run started at or before t that reaches it within 48 h (forecast day 1 or 2); a run cut
    # short (e.g. LV4 2025-08-31, one record) is passed over
    day = DT.datetime.combine(t.date(), DT.time())
    runs = [r for k in range(3) for r in history_files(lv, day - DT.timedelta(days=k))]   # a 06 UTC run reaches 2 dates on
    for start, f in sorted(runs, reverse=True):
        rec, rem = divmod((t - start).total_seconds(), 3600)
        if 0 <= rec < 48 and rem == 0 and rec < n_records(f):
            return dict(t=t, tl=tl, f=f, rec=int(rec), fc_day=1 + int(rec) // 24, run=run_id(lv, f))
    return dict(t=t, tl=tl, f=None, rec=None, fc_day=None, run=None)


# ----------------------------------------------------------------------------- moorings
def mooring_currents(lv, PAN):
    """{site: (lon, lat, hourly top-layer east/north DataFrame or None)} for the moorings on this level."""
    sys.path.insert(0, MOORINGS_COMMON)
    import plot_model_vs_obs as PMO
    start = f"{PAN[0]['t']:%Y-%m-%d}"
    end = f"{PAN[-1]['t']:%Y-%m-%d}"
    out = {}
    for s in MOORINGS[lv]:
        _, top = PMO.obs_currents(s, start, end)
        if top is not None:
            top.index = top.index.tz_convert("UTC").tz_localize(None)
            print(f"  {s}: mooring obs {top.index.min():%Y-%m-%d %H}Z .. {top.index.max():%Y-%m-%d %H}Z")
        else:
            print(f"  {s}: no mooring obs in the window")
        out[s] = (PMO.MODEL_SITES[s]["lon"], PMO.MODEL_SITES[s]["lat"], top)
    return out


def model_at_moorings(lv, PAN):
    """{(history file, time): {site: (u_east, v_north)}}: model current at each mooring over the ADCP
    top layer. Taken from the Moorings model cache (<site>_<lv>_day1_hourly.csv, u_top/v_top: the same
    nearest points and layer mean) where it has the panel; read from the history file otherwise: sites
    the Moorings code does not extract on this level (SBOO on LV3), day-2 fallback panels, and hours
    newer than the cache."""
    sys.path.insert(0, MOORINGS_COMMON)
    import extract_model as EM
    out, todo = {}, {}                                   # todo: history file -> sites still needed
    for p in PAN:
        if p["f"] is None:
            continue
        for s in MOORINGS[lv]:
            uv = _moorings_csv(EM, s, lv).get(p["t"]) if p["fc_day"] == 1 and len(p["run"]) == 8 else None
            if uv is not None and np.all(np.isfinite(uv)):
                out.setdefault((p["f"], p["t"]), {})[s] = uv
            else:
                todo.setdefault(p["f"], set()).add(s)
    for f, sites in sorted(todo.items()):
        run = run_id(lv, f)
        cached = {s: os.path.join(CACHE_DIR, f"mooring_{lv}_{run}_{s}.npz") for s in sites}
        for s in sorted(sites):                          # our own cache of these (days 1-2 of the run)
            if os.path.exists(cached[s]):
                z = np.load(cached[s])
                for k, ts in enumerate(z["t"]):
                    tt = DT.datetime(1970, 1, 1) + DT.timedelta(seconds=int(ts))
                    out.setdefault((f, tt), {}).setdefault(s, (float(z["u"][k]), float(z["v"][k])))
                sites = sites - {s}
        if not sites:
            continue
        sites = sorted(sites)
        print(f"  model at {', '.join(sites)}: from {os.path.basename(f)} (not in the Moorings cache)")
        pts = EM.nearest_points(lv, sites)
        t, data = EM.read_file(f, pts, 1, 2)
        static = EM.static_fields(f, pts)
        for s in sites:
            d, ang = data[s], pts[s]["angle"]
            _, zw = EM.z_levels(static[s], d["zeta"])
            ue = d["u"] * np.cos(ang) - d["v"] * np.sin(ang)
            vn = d["u"] * np.sin(ang) + d["v"] * np.cos(ang)
            ref = np.zeros(len(t)) if EM.SITES[s]["z_ref"] == "msl" else d["zeta"]
            u = EM.layer_mean(ue, zw, ref, *EM.SITES[s]["top"])
            v = EM.layer_mean(vn, zw, ref, *EM.SITES[s]["top"])
            for k, tt in enumerate(t):
                out.setdefault((f, tt.tz_convert(None).round("min").to_pydatetime()), {}).setdefault(s, (float(u[k]), float(v[k])))
            secs = np.array([(tt.tz_convert(None).round("min") - DT.datetime(1970, 1, 1)).total_seconds() for tt in t], np.int64)
            save_npz(cached[s], t=secs, u=np.asarray(u, np.float32), v=np.asarray(v, np.float32))
    return out


_CSV = {}


def _moorings_csv(EM, site, lv):
    """{naive UTC time: (u_top, v_top)} from the Moorings day-1 model cache, {} if it has no such file."""
    if (site, lv) not in _CSV:
        path = os.path.join(EM.MODEL_DIR, f"{site}_{lv}_day1_hourly.csv")
        d = {}
        if os.path.exists(path):
            import pandas as pd
            df = pd.read_csv(path, usecols=["time", "u_top", "v_top"], parse_dates=["time"])
            d = {t.to_pydatetime(): (u, v) for t, u, v in zip(df["time"], df["u_top"], df["v_top"])}
        _CSV[(site, lv)] = d
    return _CSV[(site, lv)]


def mooring_at(MO, t):
    """{site: (lon, lat, u, v)} at hour t; u, v NaN where the mooring has no data."""
    out = {}
    for s, (lon, lat, top) in MO.items():
        u = v = np.nan
        if top is not None and t in top.index:
            u, v = top.loc[t, "u_east"], top.loc[t, "v_north"]
        out[s] = (lon, lat, float(u), float(v))
    return out


def open_arrows(ax, C, x, y, u, v, qscale, **kw):
    """Arrows with an open head (shaft and two head strokes, no fill) as one LineCollection. Lengths as
    quiver(scale=qscale, scale_units="width"): speed qscale spans the axes width. Drawn in lon and
    lat/cos(lat0) so they look right on the cos(lat0)-scaled map."""
    from matplotlib.collections import LineCollection
    x, y, u, v = (np.ravel(np.asarray(a, float)) for a in (x, y, u, v))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(u) & np.isfinite(v)
    x, y, u, v = x[ok], y[ok], u[ok], v[ok]
    c = np.cos(np.radians(np.mean(C["ylim"])))
    k = (C["xlim"][1] - C["xlim"][0]) / qscale                     # degrees of longitude per m/s
    X, Y, dX, dY = x, y / c, u * k, v * k                          # isotropic map units
    tX, tY = X + dX, Y + dY
    segs = [np.stack([np.c_[X, Y * c], np.c_[tX, tY * c]], axis=1)]
    for a in (np.radians(155), np.radians(-155)):                  # head strokes, back from the tip
        hX = tX + HEAD * (dX * np.cos(a) - dY * np.sin(a))
        hY = tY + HEAD * (dX * np.sin(a) + dY * np.cos(a))
        segs.append(np.stack([np.c_[tX, tY * c], np.c_[hX, hY * c]], axis=1))
    ax.add_collection(LineCollection(np.concatenate(segs), capstyle="round", joinstyle="round", **kw))


def draw_moorings(ax, moor, moor_model, qscale, C):
    for s, (lon, lat, u, v) in moor.items():
        mu, mv = moor_model.get(s, (np.nan, np.nan))
        if np.isfinite(mu) and np.isfinite(mv):
            open_arrows(ax, C, [lon], [lat], [mu], [mv], qscale, colors=MOOR_MODEL_COLOR, linewidths=MOOR_LW, zorder=9)
        if np.isfinite(u) and np.isfinite(v):
            open_arrows(ax, C, [lon], [lat], [u], [v], qscale, colors=MOOR_COLOR[s], linewidths=MOOR_LW,
                        alpha=MOOR_ALPHA, zorder=10)
        # hollow and under the arrow, so short arrows are not hidden by the marker
        ax.plot(lon, lat, "o", mfc="none", mec=MOOR_COLOR[s], mew=1.2, ms=4, zorder=8)


def prepare(lv, PAN, every=None):
    """Grid, ~100 m interpolation and the data of each panel in PAN (dicts from panels()): model arrows,
    HFR, moorings and the stats. Returns (C, P): C the shared plotting context, P the panels."""
    G = load_grid(lv)
    hlon, hlat = hfr_axes()
    every = every or ARROW_EVERY[lv]
    F = FineGrid(G, hlon, hlat, every, lv)
    print(f"{lv}: fine grid {F.shape[1]} x {F.shape[0]} at {F.dx_m:.0f} x {F.dy_m:.0f} m; "
          f"arrows every {every} points ({every * F.dx_m / 1e3:.1f} km)")
    lon_a, lat_a = np.meshgrid(F.lon, F.lat)
    jn, in_ = np.meshgrid(*F.node_ij, indexing="ij")
    in_hfr = (jn >= 0) & (jn < hlat.size) & (in_ >= 0) & (in_ < hlon.size)
    xlim = (G["lon_rho"].min(), G["lon_rho"].max())
    ylim = (G["lat_rho"].min(), G["lat_rho"].max())
    C = dict(lv=lv, G=G, F=F, every=every, xlim=xlim, ylim=ylim, HLON=np.meshgrid(hlon, hlat)[0],
             HLAT=np.meshgrid(hlon, hlat)[1], lon_a=lon_a[F.arrow], lat_a=lat_a[F.arrow],
             land=np.where(G["wet"], np.nan, 1.0), h_water=np.where(G["wet"], G["h"], np.nan),
             width_km=(xlim[1] - xlim[0]) * km_lon(np.mean(ylim)),
             ref=max(every, ARROW_EVERY[lv] / 2))   # dense test spacings: arrows at least half the default long

    MO = mooring_currents(lv, PAN)
    MM = model_at_moorings(lv, PAN)
    P = []
    for p in PAN:
        t, f = p["t"], p["f"]
        p.update(moor=mooring_at(MO, t), moor_model=MM.get((f, t), {}))
        if f is None:
            print(f"  {p['tl']:%Y-%m-%d %H:%M %Z}: no {lv} history file")
            P.append(p)
            continue
        if p["fc_day"] == 2:
            print(f"  {p['tl']:%Y-%m-%d %H:%M %Z}: day 2 of the {p['run']} run")
        mu, mv = model_panel(lv, G, F, f, p["rec"], t)
        h = hfr_uv(t, hlon, hlat) if t <= now_utc() else None
        p.update(mu=mu, mv=mv, h=h)
        if h is not None:   # stats at HFR cell centres inside the model domain
            hu, hv = np.full(in_hfr.shape, np.nan), np.full(in_hfr.shape, np.nan)
            hu[in_hfr], hv[in_hfr] = h[0][jn[in_hfr], in_[in_hfr]], h[1][jn[in_hfr], in_[in_hfr]]
            nu, nv = mu[F.node_in_arrow], mv[F.node_in_arrow]
            ok = np.isfinite(hu) & np.isfinite(nu)
            if ok.any():
                p["stats"] = (int(ok.sum()), np.hypot(nu, nv)[ok].mean(), np.hypot(hu, hv)[ok].mean(),
                              np.sqrt(np.mean((nu - hu)[ok] ** 2 + (nv - hv)[ok] ** 2)))
        P.append(p)
    return C, P


def qscale_for(C, s95):
    """Arrow scale (m/s across the axes width) that draws speed s95 ARROW_LEN arrow spacings long."""
    return s95 * C["width_km"] / (ARROW_LEN * C["ref"] * C["F"].dx_m / 1e3)


def draw_panel(ax, p, C, qscale, fontsize=11):
    """Bathymetry, land, model, HFR and mooring arrows of one panel, with its title."""
    lv, G, xlim, ylim = C["lv"], C["G"], C["xlim"], C["ylim"]
    ax.contour(G["lon_rho"], G["lat_rho"], C["h_water"], levels=BATHY[lv], colors="0.7", linewidths=0.4, zorder=1)
    ax.pcolormesh(G["lon_rho"], G["lat_rho"], C["land"], cmap=ListedColormap([LAND_COLOR]), shading="auto", zorder=2)
    ax.contour(G["lon_rho"], G["lat_rho"], G["wet"].astype(float), levels=[0.5], colors=COAST_COLOR,
               linewidths=0.6, zorder=3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1 / np.cos(np.radians(np.mean(ylim))))
    title = f"{p['tl']:%a %d %b %H:%M %Z}" + (f"  (day 2 of {p['run']} run)" if p["fc_day"] == 2 else "")
    draw_moorings(ax, p["moor"], p["moor_model"], qscale, C)
    if p["run"] is None:
        ax.text(0.5, 0.5, f"no {lv} run", transform=ax.transAxes, ha="center", color="0.4")
        ax.set_title(title, fontsize=fontsize)
        return
    g = np.isfinite(p["mu"])
    open_arrows(ax, C, C["lon_a"][g], C["lat_a"][g], p["mu"][g], p["mv"][g], qscale, colors=MODEL_COLOR[lv],
                linewidths=MODEL_LW * min(1, 1.25 * C["ref"] / ARROW_EVERY[lv]), zorder=5)   # thinner for denser test spacings
    if p.get("h") is not None:
        hu, hv = p["h"]
        HLON, HLAT = C["HLON"], C["HLAT"]
        gh = np.isfinite(hu) & (HLON >= xlim[0]) & (HLON <= xlim[1]) & (HLAT >= ylim[0]) & (HLAT <= ylim[1])
        open_arrows(ax, C, HLON[gh], HLAT[gh], hu[gh], hv[gh], qscale, colors=HFR_COLOR, linewidths=HFR_LW[lv], zorder=6)
    src = "no HFR yet" if p["t"] > now_utc() else "no HFR" if p.get("h") is None else ""
    if "stats" in p:
        n, sm, sh, rms = p["stats"]
        title += f"\nn={n}  speed {sm:.2f}/{sh:.2f}  rms diff {rms:.2f} m/s"
    else:
        title += f"\n{src or 'no HFR in domain'}"
    ax.set_title(title, fontsize=fontsize)


def speed_key(ax, qscale, C):
    """Reference arrow in the top-right corner, ending 5 % in from the right edge."""
    (x0, x1), (y0, y1) = C["xlim"], C["ylim"]
    start = 0.95 - QKEY / qscale                     # arrow length as a fraction of the axes width
    open_arrows(ax, C, [x0 + start * (x1 - x0)], [y0 + 0.95 * (y1 - y0)], [QKEY], [0], qscale, colors="k",
                linewidths=1.2, zorder=8)
    ax.text(0.95, 0.91, f"{QKEY:g} m/s", transform=ax.transAxes, ha="right", va="top", fontsize=10.5, zorder=8)


def legend_handles(C):
    lv, every, F = C["lv"], C["every"], C["F"]
    handles = [plt.Line2D([], [], color=MODEL_COLOR[lv], lw=MODEL_LW + 0.4,
                          label=f"PFM {lv} top {SURFACE_M:g} m mean, interpolated to ~100 m, every "
                                + (f"{every * F.dx_m / 1e3:.0f} km" if every * F.dx_m >= 1e3 else f"{every * F.dx_m:.0f} m")),
               plt.Line2D([], [], color=HFR_COLOR, lw=HFR_LW[lv] + 0.4, label="HF radar 6 km"),
               plt.Line2D([], [], color=MOOR_MODEL_COLOR, lw=MOOR_LW, label="PFM at each mooring, ADCP top layer")]
    handles += [plt.Line2D([], [], color=MOOR_COLOR[s], lw=MOOR_LW, alpha=MOOR_ALPHA, marker="o", mfc="none",
                           label=f"{MOOR_NAME[s]} ADCP top layer (observed)") for s in MOORINGS[lv]]
    return handles


def plot_level(lv, end_day, every=None, out_dir=OUT_DIR, name=None):
    """One figure, last panel end_day 00:00 local, into out_dir/<name>.png
    (default name Model_vs_HFR6km_<LV>_<end_day>, plus _arrows<N>m for a test spacing)."""
    C, P = prepare(lv, panels(lv, end_day), every)
    every, F, xlim, ylim = C["every"], C["F"], C["xlim"], C["ylim"]
    # one arrow scale per figure: the 95th percentile speed draws ARROW_LEN arrow spacings long
    sp = np.concatenate([np.hypot(p["mu"], p["mv"]).ravel() for p in P if p["run"]] +
                        [np.hypot(*p["h"]).ravel() for p in P if p.get("h") is not None] +
                        [np.array([np.hypot(u, v) for _, _, u, v in p["moor"].values()]) for p in P] +
                        [np.array([np.hypot(u, v) for u, v in p["moor_model"].values()] or [np.nan]) for p in P])
    qscale = qscale_for(C, np.nanpercentile(sp, 95))

    aspect = (ylim[1] - ylim[0]) * KM_LAT / C["width_km"]
    pw = 4.6
    fig, axes = plt.subplots(3, 4, figsize=(4 * pw, 3 * pw * aspect + 1.6), sharex=True, sharey=True)
    for ax, p in zip(axes.ravel(), P):
        draw_panel(ax, p, C, qscale)
    speed_key(axes[0, 0], qscale, C)
    for r in range(3):
        axes[r, 0].set_ylabel("latitude", fontsize=12)
    for c in range(4):
        axes[-1, c].set_xlabel("longitude", fontsize=12)
    for ax in axes.ravel():
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.tick_params(labelsize=11)
    fig.legend(handles=legend_handles(C), loc="lower center", ncol=3, frameon=False, fontsize=12,
               title="Circle with no arrow = no mooring data that hour", title_fontsize=11)
    fig.suptitle(f"Surface currents: PFM {lv} vs HF radar 6 km and moorings, every 6 h, local time, "
                 f"{P[0]['tl']:%d %b %H:%M} to {P[-1]['tl']:%d %b %H:%M %Z}\nModel: day 1 of the run for each "
                 f"panel's UTC date (day 2 of the previous run if marked).  Titles: HFR cells with both, "
                 f"mean speed model/HFR, RMS vector difference", fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95), h_pad=2.5)
    os.makedirs(out_dir, exist_ok=True)
    tag = "" if every == ARROW_EVERY[lv] else f"_arrows{every * F.dx_m:.0f}m"     # test spacings
    out = os.path.join(out_dir, (name or f"Model_vs_HFR6km_{lv}_{end_day:%Y%m%d}") + f"{tag}.png")
    fig.savefig(out, dpi=200 if tag else 130)
    plt.close(fig)
    print("wrote", out)


def update_moorings(levels):
    """Download new mooring obs from SIO for the moorings on these levels."""
    sys.path.insert(0, MOORINGS_COMMON)
    import download_obs
    for s in sorted({s for lv in levels for s in MOORINGS[lv]}):
        download_obs.update_site(s)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--date", help="local date of the last row, YYYY-MM-DD (default: today, local)")
    p.add_argument("--level", choices=LEVELS, nargs="+", default=list(LEVELS))
    p.add_argument("--arrow-km", type=float, help="model arrow spacing, km (test; default ~3 km LV3, ~2 km LV4); "
                   "rounded to a divisor of the 6 km HFR spacing")
    p.add_argument("--update-moorings", action="store_true", help="download new mooring obs from SIO first")
    a = p.parse_args()
    if a.update_moorings:
        update_moorings(a.level)
    day = (DT.datetime.strptime(a.date, "%Y-%m-%d").date() if a.date
           else DT.datetime.now(LOCAL_TZ).date())
    end_day = day + DT.timedelta(days=1)              # the last panel is the next day's 00:00
    every = None
    if a.arrow_km:   # nearest number of fine points that divides FINE, so HFR centres stay on arrow points
        every = min((n for n in range(1, FINE + 1) if FINE % n == 0), key=lambda n: abs(n * 6 / FINE - a.arrow_km))
    for lv in a.level:
        plot_level(lv, end_day, every, name=f"Model_vs_HFR6km_{lv}_{day:%Y%m%d}")   # named by the last row
        if not a.date and not a.arrow_km:   # daily run: keep only the newest figure for the website
            keep = f"Model_vs_HFR6km_{lv}_{day:%Y%m%d}.png"
            for old in glob.glob(os.path.join(OUT_DIR, f"Model_vs_HFR6km_{lv}_{'[0-9]' * 8}.png")):
                if os.path.basename(old) != keep:
                    os.remove(old)
                    print("removed", os.path.basename(old))


if __name__ == "__main__":
    main()
