#!/usr/bin/env python3
"""QC-website plot: PTB discharge, observed Punta Bandera flow vs the PFM model forcing.

The first panel of PTB_TJRE_DyeFactor/figures/Dye_Factor_PTB.png, standalone:
  - observed Punta Bandera flow (IBWC AQWebportal, Discharge.Telemetry-ADS-mgd@11-PUNTA-BANDERA),
    hourly as in PTB_data.m (mean of the 5 samples centred on the first sample of each hour),
    converted from MGD to m^3/s;
  - model Q_ptb: total discharge of the river sources carrying dye_01 (PTB) in the LV4 river
    forcing (river_LV4_<YYYYMMDDHH>.nc);
  - model Q_ptb,ww: raw-sewage part of it, sum of |river_transport| x river_dye_01 weighted over
    the layers by river_Vshape (as Dye_Loading.py).
Each model hour comes from the latest forecast covering it, so days without a forcing file are
filled from the earlier forecasts (up to 5 days ahead); longer gaps keep the last forecast value,
and hours before the first forcing file take its first value (as Dye_Loading.py).

Daily run (default): past 3 weeks up to the end of today (UTC), the same window as the SBOO / IB
QC plots, into plots_web/PTB_discharge_<YYYYMMDD>_3weeks.png; older *_3weeks.png are removed so
plots_web/ only holds the newest plot. For any start and end dates use plot_ptb_q_range.py.

  python plot_ptb_q.py                                  # past 3 weeks, for the website
  python plot_ptb_q.py --date 2026-09-30                # 3 weeks ending on another day (old plots kept)
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
import numpy as np
import pandas as pd
from netCDF4 import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "plots_web")
RIVER_DIR = "/dataSIO/PFM_Simulations/Archive/Forcing"
IBWC_URL = "https://waterdata.ibwc.gov/AQWebportal/Export/DataSet"
PB_DATASET = "Discharge.Telemetry-ADS-mgd@11-PUNTA-BANDERA"
MGD_TO_M3S = 0.043812636
EPOCH = pd.Timestamp("1999-01-01")
DAYS, LABEL = 21, "3weeks"

# colours of PTB_TJRE_DyeFactor/plot_style.py
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_QPTB, C_QWW = "#2a78d6", "#2a9d5c"


def observed_pb(start, end, timeout=900):
    """Hourly observed Punta Bandera flow (m^3/s) between start and end (UTC)."""
    params = {"DataSet": PB_DATASET, "Calendar": "CALENDARYEAR",
              "StartTime": f"{start:%Y-%m-%d %H:%M:%S}", "EndTime": f"{end:%Y-%m-%d %H:%M:%S}",
              "DateRange": "Custom", "UnitID": 111, "Conversion": "Instantaneous",
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
    df = df.dropna(subset=["time"])
    # hourly: mean of the samples 2 before to 2 after the first sample of each hour (PTB_data.m)
    hours = df["time"].dt.floor("h").to_numpy()
    first = np.flatnonzero(~pd.Index(hours).duplicated())
    v = df["flow"].to_numpy(float)
    out = [np.nanmean(w) if np.isfinite(w := v[max(i - 2, 0):i + 3]).any() else np.nan for i in first]
    return pd.Series(np.asarray(out) * MGD_TO_M3S, index=pd.DatetimeIndex(hours[first]))


def model_ptb(start, end):
    """Hourly model Q_ptb and Q_ptb,ww (m^3/s) from the river forcing, latest forecast per hour."""
    files = {}
    for p in sorted(glob.glob(os.path.join(RIVER_DIR, "river_LV4_*.nc"))):
        m = re.search(r"river_LV4_(\d{10})\.nc$", p)
        if m:
            files[pd.Timestamp(m.group(1)[:8] + "T" + m.group(1)[8:])] = p
    # forecasts that can cover the window (each runs 5 days ahead)
    use = [t for t in files if start - pd.Timedelta(days=6) <= t <= end]
    # plus the latest earlier forecast (else the first one), to fill hours no forecast covers
    earlier = [t for t in files if t < start - pd.Timedelta(days=6)]
    use = ([max(earlier)] if earlier else []) + use or [min(files)]
    q, qww = {}, {}
    for t0 in sorted(use):                                    # later forecasts overwrite earlier ones
        with Dataset(files[t0]) as nc:
            Q = np.abs(np.asarray(nc["river_transport"][:], float))          # time x river
            Vshape = np.asarray(nc["river_Vshape"][:], float)                # s_rho x river
            dye1 = np.asarray(nc["river_dye_01"][:], float)                  # time x s_rho x river
            tday = np.asarray(nc["river_time"][:], float)
        frac = np.sum(Vshape[None] * dye1, axis=1)                          # time x river
        src = np.any(frac > 0, axis=0)                                      # PTB sources
        hrs = t0 + pd.to_timedelta(np.arange(121), "h")
        td = ((hrs - EPOCH) / pd.Timedelta(days=1)).to_numpy()
        qa = np.interp(td, tday, Q[:, src].sum(axis=1))
        qwa = np.interp(td, tday, (Q * frac).sum(axis=1))
        for h, a, b in zip(hrs, qa, qwa):
            q[h], qww[h] = a, b
    # hours no forecast covers (no forcing file for > 5 days, or before the first file) take the
    # latest earlier value, else the first one, as Dye_Loading.py fills from the nearest forcing file
    q, qww = pd.Series(q).sort_index(), pd.Series(qww).sort_index()
    idx = pd.date_range(start, end, freq="h")
    full = q.index.union(idx)
    q = q.reindex(full).ffill().bfill().reindex(idx)
    qww = qww.reindex(full).ffill().bfill().reindex(idx)
    return q, qww


def plot(start, end, path):
    obs = observed_pb(start, end)
    obs = obs[(obs.index >= start) & (obs.index <= end)]
    q, qww = model_ptb(start, end)
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False})
    days = (end - start).days
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(obs.index, obs, color=INK2, lw=0.9 if days <= 60 else 0.5, alpha=0.75,
            label="Observed Punta Bandera (IBWC), hourly")
    daily = obs.resample("D").mean()                     # plotted at midday
    ax.plot(daily.index + pd.Timedelta(hours=12), daily, color=INK, lw=1.5, label="Observed, daily mean")
    ax.plot(q.index, q, color=C_QPTB, lw=1.5, label="Model Q$_{ptb}$ (forcing)")
    ax.plot(qww.index, qww, color=C_QWW, lw=1.2, label="Model Q$_{ptb,ww}$ (forcing)")
    ax.set_ylabel("m$^3$/s")
    ax.set_ylim(bottom=0)
    # same axis in MGD on the right
    sec = ax.secondary_yaxis("right", functions=(lambda x: x / MGD_TO_M3S, lambda x: x * MGD_TO_M3S))
    sec.set_ylabel("MGD")
    ax.spines["right"].set_visible(True)
    ax.set_xlim(start, end)
    ax.set_title(f"PTB discharge: observed vs model, {start:%Y-%m-%d} to {(end - pd.Timedelta(seconds=1)):%Y-%m-%d} (UTC)",
                 pad=30)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=4, fontsize=9)
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.xaxis.set_minor_locator(mdates.DayLocator() if days <= 60 else mdates.MonthLocator())
    ax.tick_params(axis="x", which="minor", length=3, color=INK2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    if len(obs):
        print(f"observed to {obs.index.max()}, median {obs.median():.2f} m3/s; "
              f"model Q_ptb median {q.median():.2f}, Q_ptb,ww median {qww.median():.2f} m3/s")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date", help="Last day of the 3-week plot, YYYY-MM-DD UTC (default: today)")
    args = p.parse_args()
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    last = pd.Timestamp(args.date) if args.date else today
    start, end = last - pd.Timedelta(days=DAYS - 1), last + pd.Timedelta(days=1)
    path = os.path.join(WEB_DIR, f"PTB_discharge_{last:%Y%m%d}_{LABEL}.png")
    plot(start, end, path)
    # daily run: keep only the newest plot for the website
    if not args.date:
        for old in glob.glob(os.path.join(WEB_DIR, f"PTB_discharge_{'[0-9]' * 8}_{LABEL}.png")):
            if old != path:
                os.remove(old)
                print("removed", os.path.basename(old))


if __name__ == "__main__":
    main()
