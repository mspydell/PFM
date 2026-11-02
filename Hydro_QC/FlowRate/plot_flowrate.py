#!/usr/bin/env python3
"""QC-website plot: flow rates, observed (IBWC) vs the PFM model forcing: Punta Bandera + SAB, and the Tijuana River.

The first panel of PTB_TJRE_DyeFactor/figures/Dye_Factor_PTB.png, standalone:
  - observed Punta Bandera flow (IBWC AQWebportal, Discharge.Telemetry-ADS-mgd@11-PUNTA-BANDERA),
    hourly as in PTB_data.m (mean of the 5 samples centred on the first sample of each hour),
    converted from MGD to m^3/s;
  - observed SAB PreTreatment flow (Discharge.Telemetry-ADS-mgd@11-SAB-Pre, as IBWC_Flow.py and
    plot_observed_flows.py), hourly and daily mean the same way;
  - model Q_ptb: total discharge of the river sources carrying dye_01 (PTB) in the LV4 river
    forcing (river_LV4_<YYYYMMDDHH>.nc);
  - model Q_ptb,ww: raw-sewage part of it, sum of |river_transport| x river_dye_01 weighted over
    the layers by river_Vshape (as Dye_Loading.py).
A second panel (log axis) shows the Tijuana River, as Hourly_Flow_rate_only_SDTJRE.png of
Model_vs_Observation_river_flow.m (PFM_Master_Code_For_Plots.m, Obs_Lo 6):
  - observed IBWC flow (Discharge.Best Available@11013300, m^3/s), hourly the same way, and daily mean;
  - model Q_tj: total discharge of the TJR sources, rivers 1-5 as data_river.m (older forcing files tag
    only 4 of them with dye_02, so they are taken by index).
Each model hour comes from the latest forecast covering it. Days missing from the forcing archive up
to July 2025 (2024-12-05..2025-02-19 and 35 days in Mar-Jul 2025) take that day's forecast from Matt's
river files (riverQ_from_matt/LV4_riverQ_<YYYYMMDD>.nc): Q_tj = their TJRE discharge (the LV4 logs and
the neighbouring forcing files show each TJ source at exactly 0.2 x TJRE); Q_ptb and Q_ptb,ww, not in
them, are the forcing's hardcoded Punta Bandera values before 2025-04-01 (2.1906 m^3/s, dye 0.7).
Matt's files from August 2025 do not match the forcing and are not used; missing days after that are
filled from the earlier forecasts (up to 5 days ahead). Longer gaps keep the last value, and hours
before the first file take its first value (as Dye_Loading.py).
Nothing before 2024-12-05 (DATA_START, the first LV4 run with rivers) is plotted, read or cached.
The model lines run on past the window to the end of the latest forecast issued by then (its next
~5 days; files hold 111-123 hours), dashed after now (or after the window's end, for an older window),
with a dotted line at that split.

Cache (/dataSIO/PFM_Simulations/cache/FlowRate/, shared by swell and backwash): cache/obs/<data set>_<YYYYMM>.csv holds the raw IBWC samples per
UTC month (the current and previous month are downloaded again on every run, older months only once);
cache/model/river_LV4_<YYYYMMDDHH>.csv holds the hourly model flows of each forcing file (recomputed
if the forcing file changes); cache/model/river_LV4_<YYYYMMDD>.csv holds q_tj of each of Matt's files.
Delete a file to rebuild it.

Times on the plots are Pacific (PDT/PST, America/Los_Angeles), as on the dye vs ddPCR QC plots;
the data are read in UTC and converted. Days, the plot window and the daily means are Pacific days.

Daily run (default): past 3 weeks up to the end of today (Pacific), the same window as the SBOO / IB
QC plots, into plots_web/FlowRate_<YYYYMMDD>_3weeks.png; older *_3weeks.png are removed so
plots_web/ only holds the newest plot. For any start and end dates use plot_flowrate_range.py.

  python plot_flowrate.py                                  # past 3 weeks, for the website
  python plot_flowrate.py --date 2026-09-30                # 3 weeks ending on another day (old plots kept)
"""
import argparse
import glob
import io
import os
import re
import urllib.parse
import urllib.request
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "plots_web")
# shared cache on /dataSIO, seen from swell and backwash; the HYDRO_QC_DATA environment variable overrides the root
CACHE_DIR = os.path.join(os.environ.get("HYDRO_QC_DATA", "/dataSIO/PFM_Simulations/cache"), "FlowRate")
OBS_DIR = os.path.join(CACHE_DIR, "obs")      # raw IBWC samples, one CSV per data set and UTC month
MODEL_DIR = os.path.join(CACHE_DIR, "model")  # hourly flows of each river forcing file
RIVER_DIR = "/dataSIO/PFM_Simulations/Archive/Forcing"
# Matt's river discharge (Sweetwater, Otay, TJRE), LV4_riverQ_<YYYYMMDD>.nc: the forecast of the days
# missing from RIVER_DIR (2024-12-05..2025-02-19, and 35 days in Mar-Jul 2025). Q_tj = TJRE (5 sources x 0.2 Q in
# mk_lv4_river_nc): the LV4 logs and the neighbouring forcing files match it exactly up to July 2025;
# from August 2025 the forcing is 0.01-0.5 x TJRE (other TJ input), so later files are not used.
# Punta Bandera is not in them, so Q_ptb and Q_ptb,ww take get_pb_flow_and_dye's hardcoded values up
# to its switch (2.1906 m^3/s, dye 0.7) and the forcing values after it.
MATT_DIR = "/dataSIO/PFM_Simulations/Archive/riverQ_from_matt"
MATT_END = pd.Timestamp("2025-08-01")
PB_SWITCH, PB_Q1, PB_DYE1 = pd.Timestamp("2025-04-01"), 2.1906, 0.7
# first day (Pacific) of LV4 runs with rivers (LV4_His starts 2024-12-05); nothing earlier is read or cached
DATA_START = pd.Timestamp("2024-12-05")
TJ_SOURCES = slice(0, 5)   # rivers 1-5 (data_river.m); older files tag only 4 of them with dye_02
IBWC_URL = "https://waterdata.ibwc.gov/AQWebportal/Export/DataSet"
PB_DATASET = "Discharge.Telemetry-ADS-mgd@11-PUNTA-BANDERA"
SAB_DATASET = "Discharge.Telemetry-ADS-mgd@11-SAB-Pre"           # as IBWC_Flow.py
TJR_DATASET = "Discharge.Best Available@11013300"                 # Tijuana River, m^3/s (IBWC_data.m)
MGD_TO_M3S = 0.043812636
EPOCH = pd.Timestamp("1999-01-01")
DAYS, LABEL = 21, "3weeks"
TZ = "America/Los_Angeles"


def to_utc(t):
    """Naive Pacific time -> naive UTC."""
    return pd.Timestamp(t).tz_localize(TZ, ambiguous=False, nonexistent="shift_forward").tz_convert("UTC").tz_localize(None)


def to_pacific(s):
    """Series with a naive UTC index -> the same series with a naive Pacific index."""
    return s.set_axis(s.index.tz_localize("UTC").tz_convert(TZ).tz_localize(None))

# colours of PTB_TJRE_DyeFactor/plot_style.py
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_QPTB, C_QWW, C_SAB, C_QTJ = "#2a78d6", "#2a9d5c", "#7b4fb0", "#eb6834"   # C_SAB as plot_observed_flows.py


def download(dataset, start, end, mgd=True, timeout=900):
    """Raw IBWC samples (time UTC, flow in the data set's unit) between start and end (UTC)."""
    params = {"DataSet": dataset, "Calendar": "CALENDARYEAR",
              "StartTime": f"{start:%Y-%m-%d %H:%M:%S}", "EndTime": f"{end:%Y-%m-%d %H:%M:%S}",
              "DateRange": "Custom", "UnitID": 111 if mgd else 128, "Conversion": "Instantaneous",
              "IntervalPoints": "PointsAsRecorded", "ApprovalLevels": "False", "Qualifiers": "False", "Step": 1,
              "ExportFormat": "csv", "Compressed": "true", "RoundData": "True", "GradeCodes": "False",
              "InterpolationTypes": "False", "Timezone": 0}
    url = IBWC_URL + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("utf-8", "replace")
    # line 1 is a title, line 2 the header; the last line is a disclaimer
    df = pd.read_csv(io.StringIO(text), skiprows=1, usecols=[0, 1], names=["time", "flow"], header=0,
                     on_bad_lines="skip")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["flow"] = pd.to_numeric(df["flow"], errors="coerce")
    return df.dropna(subset=["time"])


def observed(dataset, start, end, mgd=True):
    """Hourly observed IBWC flow (m^3/s) between start and end (UTC); mgd: the data set is in MGD
    (UnitID 111, the telemetry data sets), else in m^3/s (UnitID 128, the gauges).
    The raw samples are cached per UTC month in cache/obs/<data set>_<YYYYMM>.csv; months missing
    from the cache and the current and previous month are downloaded, so revisions are picked up."""
    os.makedirs(OBS_DIR, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9]+", "_", dataset).strip("_")
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    frames = []
    for m in pd.period_range(start, min(end, now), freq="M"):
        path = os.path.join(OBS_DIR, f"{name}_{m.strftime('%Y%m')}.csv")
        if not os.path.exists(path) or m >= now.to_period("M") - 1:
            try:
                df = download(dataset, max(m.start_time, to_utc(DATA_START)), min(m.end_time.floor("s"), now), mgd)
                df.to_csv(path + ".part", index=False, date_format="%Y-%m-%d %H:%M:%S")
                os.replace(path + ".part", path)
            except Exception as e:                       # keep the cached month if IBWC is down
                if not os.path.exists(path):
                    raise
                print(f"{dataset} {m}: download failed ({e}), using the cached file")
        frames.append(pd.read_csv(path, parse_dates=["time"]))
    df = (pd.concat(frames) if frames else pd.DataFrame({"time": pd.to_datetime([]), "flow": []}))
    df = df.drop_duplicates("time", keep="last").sort_values("time")
    # hourly: mean of the samples 2 before to 2 after the first sample of each hour (PTB_data.m)
    hours = df["time"].dt.floor("h").to_numpy()
    first = np.flatnonzero(~pd.Index(hours).duplicated())
    v = df["flow"].to_numpy(float)
    out = [np.nanmean(w) if np.isfinite(w := v[max(i - 2, 0):i + 3]).any() else np.nan for i in first]
    return pd.Series(np.asarray(out) * (MGD_TO_M3S if mgd else 1), index=pd.DatetimeIndex(hours[first]))


def forcing_flows(t0, path):
    """Hourly q_ptb, q_ptb_ww and q_tj (m^3/s, index UTC) of one river forcing file, cached in
    cache/model/river_LV4_<YYYYMMDDHH>.csv; recomputed when the forcing file is newer than the cache."""
    cache = os.path.join(MODEL_DIR, os.path.basename(path).replace(".nc", ".csv"))
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        return pd.read_csv(cache, index_col="time", parse_dates=["time"])
    with Dataset(path) as nc:
        Q = np.abs(np.asarray(nc["river_transport"][:], float))          # time x river
        Vshape = np.asarray(nc["river_Vshape"][:], float)                # s_rho x river
        dye1 = np.asarray(nc["river_dye_01"][:], float)                  # time x s_rho x river
        tday = np.asarray(nc["river_time"][:], float)
    frac = np.sum(Vshape[None] * dye1, axis=1)                          # time x river
    src = np.any(frac > 0, axis=0)                                      # PTB sources
    tj = np.zeros(Q.shape[1], bool)
    tj[TJ_SOURCES] = True                                               # TJR sources
    hrs = t0 + pd.to_timedelta(np.arange(124), "h")
    td = ((hrs - EPOCH) / pd.Timedelta(days=1)).to_numpy()
    keep = td <= tday[-1] + 1e-6                          # only the hours the file covers (111-123)
    hrs, td = hrs[keep], td[keep]
    out = pd.DataFrame({"q_ptb": np.interp(td, tday, Q[:, src].sum(axis=1)),
                        "q_ptb_ww": np.interp(td, tday, (Q * frac).sum(axis=1)),
                        "q_tj": np.interp(td, tday, Q[:, tj].sum(axis=1))}, index=pd.DatetimeIndex(hrs, name="time"))
    os.makedirs(MODEL_DIR, exist_ok=True)
    out.to_csv(cache + ".part")
    os.replace(cache + ".part", cache)
    return out


def matt_flows(path):
    """Hourly q_tj (TJRE discharge, m^3/s, index UTC) of one of Matt's river files, cached in
    cache/model/river_LV4_<YYYYMMDD>.csv (8 digits, the forcing files' caches have 10); recomputed when
    the file is newer than the cache."""
    cache = os.path.join(MODEL_DIR, os.path.basename(path).replace("LV4_riverQ_", "river_LV4_").replace(".nc", ".csv"))
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        return pd.read_csv(cache, index_col="time", parse_dates=["time"])
    with Dataset(path) as nc:
        t = nc["time"]                                          # epoch differs between files
        hrs = pd.DatetimeIndex([str(x) for x in num2date(t[:], t.units, only_use_cftime_datetimes=False)],
                               name="time").round("h")
        ids = list(nc["station_id"][:])
        q = np.asarray(nc["discharge"][:, ids.index("TJRE")], float)
    out = pd.DataFrame({"q_tj": q}, index=hrs)
    os.makedirs(MODEL_DIR, exist_ok=True)
    out.to_csv(cache + ".part")
    os.replace(cache + ".part", cache)
    return out


def file_times(folder, pattern, regex, fmt):
    """{issue time: path} of the files in folder matching pattern."""
    files = {}
    for p in sorted(glob.glob(os.path.join(folder, pattern))):
        m = re.search(regex, os.path.basename(p))
        if m:
            files[pd.to_datetime(m.group(1), format=fmt)] = p
    return files


def model_flows(start, end):
    """Hourly model Q_ptb, Q_ptb,ww and Q_tj (m^3/s) from the river forcing, latest forecast per hour.
    Q_tj is the total discharge of the TJR sources, rivers 1-5 as in data_river.m.
    Matt's river files (MATT_DIR, up to MATT_END) count as the forecast of their day, so they fill the
    days missing from the forcing archive; a forcing file issued the same day wins (they agree)."""
    files = file_times(RIVER_DIR, "river_LV4_*.nc", r"river_LV4_(\d{10})\.nc$", "%Y%m%d%H")
    matt = file_times(MATT_DIR, "LV4_riverQ_*.nc", r"LV4_riverQ_(\d{8})\.nc$", "%Y%m%d")
    # forecasts that can cover the window (each runs 5 days ahead)
    use = [t for t in files if start - pd.Timedelta(days=6) <= t <= end]
    # plus the latest earlier forecast, to fill hours no forecast covers
    earlier = [t for t in files if t < start - pd.Timedelta(days=6)]
    use = [(t, 1) for t in ([max(earlier)] if earlier else []) + use]
    use += [(t, 0) for t in matt if max(start - pd.Timedelta(days=6), DATA_START) <= t <= end and t < MATT_END]
    q, qww, qtj = {}, {}, {}
    for t0, forcing in sorted(use):                           # later forecasts overwrite earlier ones
        if forcing:
            f = forcing_flows(t0, files[t0])
            for h, a, b, c in zip(f.index, f["q_ptb"], f["q_ptb_ww"], f["q_tj"]):
                q[h], qww[h], qtj[h] = a, b, c
            continue
        for h, c in matt_flows(matt[t0])["q_tj"].items():
            qtj[h] = c
            if h <= PB_SWITCH:
                q[h], qww[h] = PB_Q1, PB_Q1 * PB_DYE1
    if not qtj:                                               # nothing covers the window: first forecast
        f = forcing_flows(min(files), files[min(files)])
        for h, a, b, c in zip(f.index, f["q_ptb"], f["q_ptb_ww"], f["q_tj"]):
            q[h], qww[h], qtj[h] = a, b, c
    # hours nothing covers (no file for > 5 days, or before the first file) take the latest earlier
    # value, else the first one, as Dye_Loading.py fills from the nearest forcing file.
    # Past end the series runs on to the end of the latest forecast issued by end (its next ~5 days).
    idx = pd.date_range(start, max(end, max(q), max(qtj)), freq="h")
    out = []
    for s in (q, qww, qtj):
        s = pd.Series(s).sort_index()
        out.append(s.reindex(s.index.union(idx)).ffill().bfill().reindex(idx))
    return tuple(out)


FORECAST = Line2D([], [], color=INK2, lw=1.5, ls="--", label="Model forecast, next days (dashed)")


def model_line(ax, s, split, **kw):
    """Model series solid up to split, dashed after it (forecast)."""
    ax.plot(s[:split].index, s[:split], **kw)
    kw.pop("label", None)
    ax.plot(s[split:].index, s[split:], ls="--", **kw)


def plot(start, end, path):
    """Plot from start to end, naive Pacific times."""
    s_utc, e_utc = to_utc(max(start, DATA_START)), to_utc(end)
    obs, sab = (to_pacific(x[(x.index >= s_utc) & (x.index <= e_utc)])
                for x in (observed(d, s_utc, e_utc) for d in (PB_DATASET, SAB_DATASET)))
    tjr = observed(TJR_DATASET, s_utc, e_utc, mgd=False)
    tjr = to_pacific(tjr[(tjr.index >= s_utc) & (tjr.index <= e_utc)])
    q, qww, qtj = (to_pacific(x) for x in model_flows(s_utc, e_utc))
    # model hours after now (or after the window, for an older window) are forecast: dashed
    split = min(pd.Timestamp.now(tz=TZ).tz_localize(None).floor("h"), end)
    xend = max(end, q.index[-1])
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False})
    days = (end - start).days
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7.6), sharex=True, gridspec_kw={"hspace": 0.45})
    ax.plot(obs.index, obs, color=INK2, lw=0.9 if days <= 60 else 0.5, alpha=0.75,
            label="Observed Punta Bandera (IBWC), hourly")
    daily = obs.resample("D").mean()                     # Pacific days, plotted at midday
    ax.plot(daily.index + pd.Timedelta(hours=12), daily, color=INK, lw=1.5, label="Punta Bandera, daily mean")
    ax.plot(sab.index, sab, color=C_SAB, lw=0.9 if days <= 60 else 0.5, alpha=0.45,
            label="Observed SAB PreTreatment (IBWC), hourly")
    sab_daily = sab.resample("D").mean()
    ax.plot(sab_daily.index + pd.Timedelta(hours=12), sab_daily, color=C_SAB, lw=1.5, label="SAB PreTreatment, daily mean")
    model_line(ax, q, split, color=C_QPTB, lw=1.5, label="Model Q$_{ptb}$ (forcing)")
    model_line(ax, qww, split, color=C_QWW, lw=1.2, label="Model Q$_{ptb,ww}$ (forcing)")
    ax.set_ylabel("m$^3$/s")
    ax.set_ylim(bottom=0)
    # same axis in MGD on the right
    sec = ax.secondary_yaxis("right", functions=(lambda x: x / MGD_TO_M3S, lambda x: x * MGD_TO_M3S))
    sec.set_ylabel("MGD")
    ax.spines["right"].set_visible(True)
    ax.set_xlim(start, xend)
    ax.set_title(f"PTB discharge: observed vs model, {start:%Y-%m-%d} to {(end - pd.Timedelta(seconds=1)):%Y-%m-%d}"
                 f" + model forecast to {xend:%Y-%m-%d %H:%M}", pad=46)
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [FORECAST], loc="lower right", bbox_to_anchor=(1, 1),
              ncol=4, fontsize=9)

    # Tijuana River (Hourly_Flow_rate_only_SDTJRE of Model_vs_Observation_river_flow.m), log axis
    ax2.plot(tjr.index, tjr, color=INK2, lw=0.9 if days <= 60 else 0.5, alpha=0.75,
             label="Observed Tijuana River (IBWC 11013300), hourly")
    tjr_daily = tjr.resample("D").mean()
    ax2.plot(tjr_daily.index + pd.Timedelta(hours=12), tjr_daily, color=INK, lw=1.5, label="Observed, daily mean")
    model_line(ax2, qtj, split, color=C_QTJ, lw=1.5, label="Model Q$_{tj}$ (forcing)")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-2, max(3e2, 2 * np.nanmax([tjr.max(), qtj.max()])))   # as the MATLAB plot, storm peaks kept
    ax2.set_ylabel("m$^3$/s")
    sec2 = ax2.secondary_yaxis("right", functions=(lambda x: x / MGD_TO_M3S, lambda x: x * MGD_TO_M3S))
    sec2.set_ylabel("MGD")
    ax2.spines["right"].set_visible(True)
    ax2.set_title("Tijuana River discharge: observed vs model", pad=24)
    ax2.legend(handles=ax2.get_legend_handles_labels()[0] + [FORECAST], loc="lower right", bbox_to_anchor=(1, 1),
               ncol=4, fontsize=9)
    for a in (ax, ax2):
        a.axvline(split, color=INK2, lw=0.8, ls=":")                 # now: forecast to the right
    ax2.set_xlabel("Date (Pacific time, PDT/PST)")
    loc = mdates.DayLocator() if days <= 45 else mdates.AutoDateLocator()   # a labelled tick each day
    ax2.xaxis.set_major_locator(loc)
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax2.xaxis.set_minor_locator(mdates.DayLocator() if days <= 60 else mdates.MonthLocator())
    for a in (ax, ax2):
        a.tick_params(axis="x", which="minor", length=3, color=INK2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    if len(obs):
        print(f"observed to {obs.index.max()}, median {obs.median():.2f} m3/s; "
              f"model Q_ptb median {q[:split].median():.2f}, Q_ptb,ww median {qww[:split].median():.2f} m3/s")
    if len(sab):
        print(f"SAB to {sab.index.max()}, median {sab.median():.2f} m3/s")
    if len(tjr):
        print(f"TJR to {tjr.index.max()}, median {tjr.median():.3f} m3/s; model Q_tj median {qtj[:split].median():.3f} m3/s")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date", help="Last day of the 3-week plot, YYYY-MM-DD Pacific (default: today)")
    args = p.parse_args()
    today = pd.Timestamp.now(tz=TZ).tz_localize(None).normalize()
    last = pd.Timestamp(args.date) if args.date else today
    start, end = last - pd.Timedelta(days=DAYS - 1), last + pd.Timedelta(days=1)
    path = os.path.join(WEB_DIR, f"FlowRate_{last:%Y%m%d}_{LABEL}.png")
    plot(start, end, path)
    # daily run: keep only the newest plot for the website
    if not args.date:
        for old in glob.glob(os.path.join(WEB_DIR, f"FlowRate_{'[0-9]' * 8}_{LABEL}.png")):
            if old != path:
                os.remove(old)
                print("removed", os.path.basename(old))


if __name__ == "__main__":
    main()
