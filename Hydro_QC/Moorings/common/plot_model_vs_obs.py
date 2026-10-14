#!/usr/bin/env python3
"""Shared plotting code: PFM day-1 forecast (LV4 at SBOO and the IB mini moorings, LV3 at PLOO and Del Mar)
vs mooring observations. Each mooring folder (../SBOO, ../IB, ../PLOO, ../DelMar) calls plot_window with its
group; the figures per group are listed in plot_window.

Used by plot_sboo_ib_3weeks.py (past 3 weeks, for the QC website) and plot_sboo_ib_range.py
(any start/end dates). plot_window() draws, for one time window:
  SBOO_currents, IB-C_currents  northward top 5 m (IB-C: 3 m), northward depth-averaged, eastward top 5 m (3 m),
                     eastward depth-averaged (only SBOO and IB-C have an ADCP; IB-N, IB-W, IB-S measure T/S only)
  SBOO_temperature   SBOO temperature at 1, 10, 18 and 26 m, all four depths in one panel
  SBOO_temperature_panels  the same, one panel per depth
  SBOO_salinity      SBOO salinity at 1, 10, 18 and 26 m
  SBOO_TS_contours   depth-time contours of observed and model temperature, observed and model salinity
                     (obs interpolated between the 4 sensors; model at all its levels; each pair on one scale)
  IB_temperature_salinity  temperature at 1 m and the deep sensor, salinity at 1 m and the deep sensor,
                     all four IB moorings in each panel
  In the IB figures each mooring has one colour; the model is solid and observations are dashed.

Model:  cache/model/<site>_<LV>_day1_hourly.csv from extract_model.py. Where there is an ADCP
  (SBOO, IB-C) the model is averaged over the same depths the ADCP measures (see extract_model.SITES).
Obs:    cache/obs via download_obs (suspect ADCP bins already removed there). As in the MATLAB
  *_Observed_DataExtraction code, each bin is averaged over [hour - 30 min, hour + 30 min) and
  the depth average is the mean of those hourly bins.
  "Top" obs = at SBOO (downward looking, blanked above ~3 m) the 3.3, 4.3 and 5.3 m bins (top 5 m),
  at IB-C the bins shallower than 3 m, 1.75 to 2.75 m (top 3 m). The model is averaged
  over the same depths.
  Deep T/S sensor: 8 m at IB-N/IB-C, 10 m at IB-W/IB-S (IB-S had 16 m before Dec 2025,
  not shown). T/S are calibrated but not yet QC'd for outliers.
"""
import os

import cmocean
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from download_obs import SBOO_SIDELOBE, load_adcp, load_ts
from extract_model import SITES as MODEL_SITES, level_of

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(OUT_DIR, "cache", "model")

NAME = {"SBOO": "SBOO", "IB-C": "IB-C", "PLOO": "PLOO (Point Loma)", "DelMar": "Del Mar"}
# "top N m" panels; must match the "top" ranges in extract_model.SITES. Del Mar's first ADCP bin is 5.42 m,
# so its top layer is the 5.42, 7.42 and 9.42 m bins ("top 10 m")
TOP_M = {"SBOO": 5.0, "IB-C": 3.0, "PLOO": 5.0, "DelMar": 10.0}
TOP_OBS_MAX = {"SBOO": 5.5, "IB-C": 3.0, "PLOO": 6.0, "DelMar": 10.0}  # obs bins shallower than this
# PLOO / Del Mar: obs depth average over the same range as the model (first bin to 92% of the model depth)
DAVG_OBS = {s: MODEL_SITES[s]["davg"] for s in ("PLOO", "DelMar")}
TS_DEFAULT = {"PLOO": [1, 10, 20, 30, 45, 60, 75, 87, 89], "DelMar": [1, 7, 15, 24, 35, 47, 60, 74, 90]}
# legend text: which depths each line uses
OBS_LABEL = {("SBOO", "top"): "Observed (ADCP bins at 3.3, 4.3 and 5.3 m)",
             ("SBOO", "davg"): f"Observed (mean of ADCP bins 3.3-{SBOO_SIDELOBE:.1f} m)",
             ("IB-C", "top"): "Observed (ADCP bins 1.75-2.75 m)",
             ("IB-C", "davg"): "Observed (mean of good ADCP bins, ~1.75-8.75 m)",
             ("PLOO", "top"): "Observed (ADCP bins at 3.43, 4.43 and 5.43 m)",
             ("PLOO", "davg"): f"Observed (mean of ADCP bins {DAVG_OBS['PLOO'][0]:g}-{DAVG_OBS['PLOO'][1]:.1f} m)",
             ("DelMar", "top"): "Observed (ADCP bins at 5.42, 7.42 and 9.42 m)",
             ("DelMar", "davg"): f"Observed (mean of ADCP bins {DAVG_OBS['DelMar'][0]:g}-{DAVG_OBS['DelMar'][1]:.1f} m)"}


def model_label(site, kind):
    """'PFM LVn day 1 (a-b m below ...)' for a current panel, from extract_model.SITES."""
    cfg = MODEL_SITES[site]
    top, bot = cfg[kind]
    ref = "mean sea level" if cfg["z_ref"] == "msl" else "surface"
    rng = f"{top:g}-{bot:.2f} m".replace(".00 m", " m") if bot is not None else f"{top:g} m to sea floor"
    return f"PFM {level_of(site)} day 1 (layer-weighted mean, {rng} below {ref})"
OBS_COLOR, MODEL_COLOR = "#2a78d6", "#eb6834"
IB_SITES = ["IB-N", "IB-C", "IB-W", "IB-S"]          # north to south
# one fixed colour per IB mooring (validated categorical slots 1-4); model solid, observed dashed
IB_COLOR = {"IB-N": "#2a78d6", "IB-C": "#eb6834", "IB-W": "#1baf7a", "IB-S": "#eda100"}
# SBOO temperature: one colour per sensor depth, same validated slots
SBOO_DEPTH_COLOR = {"1 m": "#2a78d6", "10 m": "#eb6834", "18 m": "#1baf7a", "26 m": "#eda100"}
DEEP_M = {"IB-N": 8, "IB-C": 8, "IB-W": 10, "IB-S": 10}
SBOO_TS_M = [1, 10, 18, 26]
_fc = None             # first model forecast hour (days 2-5 of the latest history file), set by plot_window
FC_SHADE = "0.94"      # background of the forecast period


def hour_of(t):
    """Whole UTC hour each sample belongs to: hour h gets samples in [h - 30 min, h + 30 min)."""
    return (t + pd.Timedelta(minutes=30)).dt.floor("h")


def hourly(df):
    df = df.copy()
    df["time"] = hour_of(df["time"])
    return df.groupby("time").mean()


def read_model_csv(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["time"])
    df["time"] = df["time"].dt.tz_localize("UTC")
    return df.set_index("time")


def model_csv(site, kind):
    """kind 1 (day 1) or 'forecast'."""
    return os.path.join(MODEL_DIR, f"{site}_{level_of(site)}_{'day1' if kind == 1 else kind}_hourly.csv")


def MODEL_COLUMNS(site):
    path = model_csv(site, 1)
    return set(pd.read_csv(path, nrows=0).columns) if os.path.exists(path) else set()


def load_model(site, forecast=True):
    """Day-1 series, followed (forecast=True) by days 2-5 of the latest history file after its last hour."""
    day1 = read_model_csv(model_csv(site, 1))
    fc = read_model_csv(model_csv(site, "forecast")) if forecast else None
    if day1 is None or fc is None:
        return day1
    return pd.concat([day1, fc[fc.index > day1.index.max()]])


def plot_model(ax, m, **kw):
    """Model line: solid, then fc_ls (dashed by default) over the forecast hours."""
    fc_ls = kw.pop("fc_ls", "--")
    if _fc is None:
        ax.plot(m, **kw)
        return
    ax.plot(m[m.index < _fc], **kw)
    kw.pop("label", None)
    ax.plot(m[m.index >= _fc - pd.Timedelta(hours=1)], ls=fc_ls, **kw)   # joined to the last day-1 hour


def shade_forecast(ax, end):
    if _fc is not None:
        ax.axvspan(_fc, end, color=FC_SHADE, zorder=0, lw=0)
        ax.text(_fc, 0.02, " model forecast", transform=ax.get_xaxis_transform(), ha="left", va="bottom",
                fontsize=8, color="0.35")


def obs_currents(site, start, end):
    """(depth-averaged, top) hourly east/north; None if the site has no ADCP."""
    uv = load_adcp(site, start, end)
    if uv.empty:
        return None, None
    cols = ["u_east", "v_north"]
    uv["time"] = hour_of(uv["time"])
    if site.startswith("IB"):  # IB bins follow the moving surface: group them to 0.5 m
        uv["z_m"] = (uv["z_m"] * 2).round() / 2
    prof = uv.groupby(["time", "z_m"])[cols].mean().reset_index()  # hourly profile
    top = prof[prof["z_m"] < TOP_OBS_MAX[site]]
    if site in DAVG_OBS:  # same depth range as the model average
        lo, hi = DAVG_OBS[site]
        prof = prof[(prof["z_m"] >= lo - 0.01) & (prof["z_m"] <= hi)]
    return prof.groupby("time")[cols].mean(), top.groupby("time")[cols].mean()


def obs_ts(ts, depth):
    ts = ts[ts["depth_m"] == depth]
    if ts.empty:
        return None
    return hourly(ts.pivot_table(index="time", columns="var", values="value").reset_index())


def panel(ax, idx, obs, model, title, ylabel, obs_label="Observed", robust=False, mod_label="PFM day 1"):
    o = obs.reindex(idx) if obs is not None else None
    m = model.reindex(idx) if model is not None else None
    if o is not None and o.notna().any():
        ax.plot(o, color=OBS_COLOR, lw=0.7, label=obs_label)
    else:
        o = None
    if m is not None and m.notna().any():
        plot_model(ax, m, color=MODEL_COLOR, lw=0.7, alpha=0.85, label=mod_label)
    else:
        m = None
    if not robust:
        ax.axhline(0, color="0.6", lw=0.8)
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="0.9")
    if robust:  # T/S are not outlier-QC'd: keep a few spikes from setting the axis
        vals = pd.concat([s for s in (o, m) if s is not None] or [pd.Series(dtype=float)]).dropna()
        if len(vals):
            lo, hi = vals.quantile([0.002, 0.998])
            pad = 0.05 * (hi - lo) if hi > lo else 0.1
            ax.set_ylim(lo - pad, hi + pad)
    if o is not None and m is not None:
        both = pd.concat([o, m], axis=1).dropna()
        if len(both) > 48:
            r = both.corr().iloc[0, 1]
            rmse = np.sqrt(((both.iloc[:, 1] - both.iloc[:, 0]) ** 2).mean())
            bias = (both.iloc[:, 1] - both.iloc[:, 0]).mean()
            ax.text(0.995, 0.95, f"r = {r:.2f}, RMSE = {rmse:.3g}, bias = {bias:+.3g}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="0.25",
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=2))
    missing = [w for w, s in (("observations", o), ("model", m)) if s is None]
    if missing:
        ax.text(0.5, 0.95, "no " + " or ".join(missing), transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, color="0.4")
    if ax.get_legend_handles_labels()[0]:
        # above the panel, right of the title, so it never covers the lines
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1), fontsize=8, frameon=False, ncol=2,
                  borderaxespad=0.2)


def stats(o, m):
    """r, RMSE, bias (model - obs) and hours compared, or None if fewer than 49 common hours."""
    both = pd.concat([o, m], axis=1).dropna()
    if len(both) <= 48:
        return None
    d = both.iloc[:, 1] - both.iloc[:, 0]
    return both.corr().iloc[0, 1], np.sqrt((d ** 2).mean()), d.mean(), len(both)


def overlay_panel(ax, idx, lines, title, ylabel, robust=False, colors=IB_COLOR):
    """Several moorings (or depths) in one panel. lines: [(key, obs series or None, model series or None)],
    key = the legend name and the colors key."""
    shown, notes = [], []
    for site, obs, model in lines:
        c = colors[site]
        o = obs.reindex(idx) if obs is not None else None
        m = model.reindex(idx) if model is not None else None
        if m is not None and m.notna().any():
            plot_model(ax, m, color=c, lw=0.9, label=f"{site} model", fc_ls=":")
            shown.append(m)
        else:
            m = None
        if o is not None and o.notna().any():
            ax.plot(o, color=c, lw=0.9, ls=(0, (3, 1.5)), label=f"{site} observed")
            shown.append(o)
            if m is not None and (st := stats(o, m)):
                notes.append(f"{site}: r = {st[0]:.2f}\n    RMSE = {st[1]:.3g}, bias = {st[2]:+.3g}")
    if not robust:
        ax.axhline(0, color="0.6", lw=0.8)
    elif shown:  # T/S are not outlier-QC'd: keep a few spikes from setting the axis
        vals = pd.concat(shown).dropna()
        if len(vals):
            lo, hi = vals.quantile([0.002, 0.998])
            pad = 0.05 * (hi - lo) if hi > lo else 0.1
            ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="0.9")
    if notes:  # right of the panel, outside the axes, so it never covers the lines
        ax.text(1.01, 1.0, "\n".join(notes), transform=ax.transAxes, ha="left", va="top", fontsize=8,
                color="0.25", linespacing=1.5)
    if not ax.get_legend_handles_labels()[0]:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color="0.4")


def ib_legend(fig, axes, keys=IB_SITES, y=0.972):
    """One legend above the panels: a column per mooring (or depth), model above observed."""
    seen = {l: h for ax in axes for h, l in zip(*ax.get_legend_handles_labels())}
    handles, labels = [], []
    for s in keys:
        for kind in ("model", "observed"):
            label = f"{s} {kind}"
            if label in seen:
                handles.append(seen[label])
            else:  # keep the grid aligned: invisible placeholder
                handles.append(plt.Line2D([], [], alpha=0))
                label = ""
            labels.append(label)
    fig.legend(handles, labels, loc="upper center", ncol=len(keys), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, y))


def finish(fig, axes, idx, title, out_path, top=1.0, ygroups=None):
    """ygroups: lists of panel indices that share one y range (default: all panels)."""
    for g in ygroups or [range(len(axes))]:
        lims = [axes[i].get_ylim() for i in g]
        for i in g:
            axes[i].set_ylim(min(l[0] for l in lims), max(l[1] for l in lims))
    span = (idx[-1] - idx[0]).days
    if span > 200:
        loc, fmt = mdates.MonthLocator(), "%b\n%Y"
    elif span > 60:
        loc, fmt = mdates.MonthLocator(), "%b %Y"
    elif span > 21:
        loc, fmt = mdates.WeekdayLocator(byweekday=mdates.MO), "%b %d"
    else:
        loc, fmt = mdates.DayLocator(interval=max(1, span // 10)), "%b %d"
    axes[-1].xaxis.set_major_locator(loc)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    for ax in axes:
        shade_forecast(ax, idx[-1])
    axes[-1].set_xlim(idx[0], idx[-1])
    axes[-1].set_xlabel("Date (UTC)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


GROUPS = {"SBOO": ["SBOO"], "IB": IB_SITES, "PLOO": ["PLOO"], "DelMar": ["DelMar"]}


def setup_window(start, end, sites):
    """Hourly index for UTC dates start..end, extended by the model forecast when the window reaches it;
    sets _fc. Returns (idx, model {site: DataFrame}, resolution text)."""
    global _fc
    last = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23)
    model = {s: load_model(s) for s in sites}
    # model forecast (days 2-5 of the latest history file): shown when it starts by the end of the window
    day1 = [read_model_csv(model_csv(s, 1)) for s in sites]
    d1_end = max((d.index.max() for d in day1 if d is not None), default=None)
    fc_end = max((m.index.max() for m in model.values() if m is not None), default=None)
    _fc = (d1_end + pd.Timedelta(hours=1) if d1_end is not None and fc_end > d1_end and d1_end < last else None)
    if _fc is not None:
        last = max(last, fc_end)
    else:
        model = {s: load_model(s, forecast=False) for s in sites}
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), last, freq="h")
    return idx, model, "hourly"


def col(df, c):
    return None if df is None or c not in df else df[c]


def fig_currents(s, idx, model, start, end, res, period, out):
    """North top layer, north depth-averaged, east top layer, east depth-averaged."""
    current_panels = [(f"top {TOP_M[s]:g} m", "North", "v_north", "v_top"),
                      ("depth-averaged", "North", "v_north", "v_davg"),
                      (f"top {TOP_M[s]:g} m", "East", "u_east", "u_top"),
                      ("depth-averaged", "East", "u_east", "u_davg")]
    davg, top = obs_currents(s, start, end)
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    for ax, (kind, comp, ocol, mcol) in zip(axes, current_panels):
        o = davg if kind == "depth-averaged" else top
        which = "top" if kind.startswith("top") else "davg"
        panel(ax, idx, col(o, ocol), col(model[s], mcol), f"{s} {kind}: {comp.lower()}ward",
              f"{comp} (m/s)", OBS_LABEL[(s, which)], mod_label=model_label(s, which))
    finish(fig, axes, idx, f"{NAME[s]} currents: PFM vs observations ({res}), {period}", out(f"{s}_currents"))


def fig_ts_panels(s, depths, idx, model, ts, res, period, out):
    """Temperature and salinity at each sensor depth, one panel per depth."""
    for var, mvar, unit, name in (("temp", "temp", "Temperature (°C)", f"{s}_temperature_panels"),
                                  ("salt", "sal", "Salinity", f"{s}_salinity")):
        fig, axes = plt.subplots(len(depths), 1, figsize=(13, max(11, 2.1 * len(depths))), sharex=True)
        for ax, d in zip(np.atleast_1d(axes), depths):
            o = obs_ts(ts, d)
            panel(ax, idx, col(o, var), col(model[s], f"{mvar}_{d:g}m"), f"{s}: {d:g} m", unit, robust=True,
                  obs_label=f"Observed (sensor at {d:g} m)",
                  mod_label=f"PFM {level_of(s)} day 1 (interpolated to {d:g} m below mean sea level)")
        finish(fig, list(np.atleast_1d(axes)), idx,
               f"{NAME[s]}, {unit.split(' (')[0].lower()}: PFM vs observations ({res}), {period}", out(name))


def sensor_depths(s, ts):
    """T/S sensor depths with data in the window (they vary a little between deployments), else the defaults."""
    have = sorted({float(d) for d in ts["depth_m"].unique()}) if len(ts) else []
    have = [d for d in have if f"temp_{d:g}m" in MODEL_COLUMNS(s)]
    return have or TS_DEFAULT[s]


def plot_window(start, end, out, period, group):
    """The figures of one group (SBOO, IB, PLOO or DelMar) for UTC dates start..end (both inclusive).

    out(name) gives the PNG path for figure `name`; period is the text used in the titles.
    """
    sites = GROUPS[group]
    idx, model, res = setup_window(start, end, sites)
    ts = {s: load_ts(s, start, end) for s in sites}

    if group == "SBOO":
        fig_currents("SBOO", idx, model, start, end, res, period, out)
        # all four sensor depths in one panel
        fig, ax = plt.subplots(figsize=(13, 6))
        lines = [(f"{d} m", col(obs_ts(ts["SBOO"], d), "temp"), col(model["SBOO"], f"temp_{d}m")) for d in SBOO_TS_M]
        overlay_panel(ax, idx, lines, "Temperature: observed at the sensor, model interpolated to the sensor depth "
                      "below mean sea level", "Temperature (°C)", robust=True, colors=SBOO_DEPTH_COLOR)
        ib_legend(fig, [ax], keys=list(SBOO_DEPTH_COLOR), y=0.93)
        finish(fig, [ax], idx, f"SBOO temperature: PFM (solid{', forecast dotted' if _fc else ''}) vs obs (dashed), "
               f"{res}, {period}", out("SBOO_temperature"), top=0.885)
        fig_ts_panels("SBOO", SBOO_TS_M, idx, model, ts["SBOO"], res, period, out)
        ts_contours("SBOO", idx, ts["SBOO"], SBOO_TS_M,
                    f"SBOO temperature and salinity contours: observed vs PFM ({res}), {period}", out("SBOO_TS_contours"))

    elif group == "IB":
        # currents only at IB-C (IB-N, IB-W, IB-S measure T/S only)
        fig_currents("IB-C", idx, model, start, end, res, period, out)
        # all four moorings in each panel, at 1 m and at the deep sensor
        deep = ", ".join(f"{s} {DEEP_M[s]} m" for s in IB_SITES)
        fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
        for ax, (var, mvar, unit, level) in zip(axes, [("temp", "temp", "Temperature (°C)", "1 m"),
                                                        ("temp", "temp", "Temperature (°C)", "deep"),
                                                        ("salt", "sal", "Salinity", "1 m"),
                                                        ("salt", "sal", "Salinity", "deep")]):
            lines = []
            for s in IB_SITES:
                d = 1 if level == "1 m" else DEEP_M[s]
                o = obs_ts(ts[s], d)
                lines.append((s, col(o, var), col(model[s], f"{mvar}_{d}m")))
            title = (unit.split(" (")[0] + (" at 1 m" if level == "1 m" else f" at the deep sensor ({deep})")
                     + ": observed at the sensor, model interpolated to the sensor depth below mean sea level")
            overlay_panel(ax, idx, lines, title, unit, robust=True)
        ib_legend(fig, axes)
        finish(fig, axes, idx, f"IB mini moorings temperature and salinity: PFM (solid{', forecast dotted' if _fc else ''}) "
               f"vs obs (dashed), {res}, {period}", out("IB_temperature_salinity"), top=0.955,
               ygroups=[[0, 1], [2, 3]])  # temperature panels, salinity panels

    else:  # PLOO, DelMar: deep moorings with ADCP and T/S at ~9 depths
        s = group
        depths = sensor_depths(s, ts[s])
        fig_currents(s, idx, model, start, end, res, period, out)
        fig_ts_panels(s, depths, idx, model, ts[s], res, period, out)
        ts_contours(s, idx, ts[s], depths,
                    f"{NAME[s]} temperature and salinity contours: observed vs PFM ({res}), {period}",
                    out(f"{s}_TS_contours"))


CONTOUR_CMAP = {"temp": cmocean.cm.thermal, "salt": cmocean.cm.haline}   # observed and model use the same one
CONTOUR_LEVELS = 14


def obs_ts_grid(ts, depths, grid, idx):
    """Observed T and S [hours x grid depths]: linear in depth between neighbouring sensors, only where
    both sensors of the pair have a value (no extrapolation above the top or below the bottom sensor)."""
    out = {}
    for var in ("temp", "salt"):
        cols = {}
        for d in depths:
            o = obs_ts(ts, d)
            cols[d] = o[var].reindex(idx) if o is not None and var in o else pd.Series(np.nan, idx)
        v = np.column_stack([cols[d].to_numpy(float) for d in depths])     # hours x sensors
        g = np.full((len(idx), len(grid)), np.nan)
        for i in range(len(depths) - 1):
            d0, d1 = depths[i], depths[i + 1]
            sel = (grid >= d0) & (grid <= d1)
            w = (grid[sel] - d0) / (d1 - d0)
            g[:, sel] = v[:, i:i + 1] * (1 - w) + v[:, i + 1:i + 2] * w     # NaN unless both sensors valid
        out[var] = g
    return out


def model_ts_profiles(site, idx):
    """Model T, S [hours x levels] and their depths below mean sea level (positive down), on idx. A first
    row at the free surface (depth -zeta) repeats the top level, so the contours reach the moving surface."""
    import extract_model
    p = extract_model.load_profiles(extract_model.profiles_path(site, level_of(site), 1))
    if p is None:
        return None
    t, d, a = p
    f = extract_model.load_profiles(extract_model.profiles_path(site, level_of(site), "forecast")) if _fc else None
    if f is not None:                                   # days 2-5 of the latest forecast, after day 1
        new = f[0] > t.max()
        t = t.append(f[0][new])
        d = {v: np.concatenate([d[v], f[1][v][new]]) for v in d}
    t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    zr, _ = extract_model.z_levels(a, d["zeta"])
    top = int(np.argmax(zr[0]))                         # uppermost rho level: first or last column
    at_end = top == zr.shape[1] - 1

    def surf(x, s):                                     # add the free-surface row on the surface side
        return np.column_stack([x, s] if at_end else [s, x])

    d = {**d, "temp": surf(d["temp"], d["temp"][:, top]), "salt": surf(d["salt"], d["salt"][:, top])}
    zr = surf(zr, np.asarray(d["zeta"], float).reshape(len(t), -1)[:, 0])
    pos = pd.Series(np.arange(len(t)), index=t)
    pos = pos[~pos.index.duplicated(keep="last")].reindex(idx)
    ok = pos.notna().to_numpy()
    k = pos.to_numpy()[ok].astype(int)
    out = {}
    for name, x in (("temp", d["temp"]), ("salt", d["salt"]), ("depth", -zr)):
        a_ = np.full((len(idx), x.shape[1]), np.nan)
        a_[ok] = x[k]
        out[name] = a_
    return out, float(a["h"])


def ts_contours(site, idx, ts, depths, title, out_path):
    """Depth-time contours: observed and model temperature, observed and model salinity, each pair on
    one colour scale. Obs are interpolated between the sensors (dotted lines); the model shows the whole
    water column at the depths of its levels below mean sea level, which move with the tide."""
    m = model_ts_profiles(site, idx)
    grid = np.arange(depths[0], depths[-1] + 0.01, 0.25)
    obs = obs_ts_grid(ts, depths, grid, idx)
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True, layout="constrained")
    h = m[1] if m else depths[-1] + 1
    # top of the axes: the highest model free surface in the window (above mean sea level, so negative)
    ytop = min(0.0, float(np.nanmin(m[0]["depth"]))) if m and np.isfinite(m[0]["depth"]).any() else 0.0
    idx = idx.tz_localize(None)                              # UTC, naive for matplotlib
    for row, (var, unit) in enumerate((("temp", "Temperature (°C)"), ("salt", "Salinity"))):
        ao, am = axes[2 * row], axes[2 * row + 1]
        vals = [obs[var][np.isfinite(obs[var])]]
        if m:
            vals.append(m[0][var][np.isfinite(m[0][var])])
        vals = np.concatenate(vals)
        if not len(vals):
            for ax in (ao, am):
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color="0.4")
            continue
        lo, hi = np.quantile(vals, [0.005, 0.995])           # T/S are not outlier-QC'd
        levels = np.linspace(lo, hi, CONTOUR_LEVELS + 1)
        kw = dict(levels=levels, cmap=CONTOUR_CMAP[var], extend="both")
        # observed
        if np.isfinite(obs[var]).any():
            cf = ao.contourf(idx, grid, obs[var].T, **kw)
        else:
            ao.text(0.5, 0.5, "no observations", transform=ao.transAxes, ha="center", color="0.4")
        ao.set_title(f"{site} {unit.split(' (')[0].lower()}: observed, interpolated between the sensors at "
                     + ", ".join(f"{d:g}" for d in depths) + " m", loc="left", fontsize=10.5)
        # model: each level at its own (tide-following) depth
        if m and np.isfinite(m[0][var]).any():
            x, z = m[0][var], m[0]["depth"]
            T = np.repeat(np.asarray(idx)[:, None], x.shape[1], axis=1)
            cf = am.contourf(T, z, x, **kw)
        else:
            am.text(0.5, 0.5, "no model", transform=am.transAxes, ha="center", color="0.4")
        am.set_title(f"{site} {unit.split(' (')[0].lower()}: PFM {level_of(site)} day 1, all model levels"
                     + (", then days 2-5 of the latest forecast (after the dashed line)" if _fc else ""),
                     loc="left", fontsize=10.5)
        if _fc is not None:
            am.axvline(_fc.tz_localize(None), color="white", lw=1.2, ls="--")
            am.text(_fc.tz_localize(None), 0.02, " model forecast", transform=am.get_xaxis_transform(),
                    ha="left", va="bottom", fontsize=8, color="white")
        for ax in (ao, am):
            for d in depths:
                ax.axhline(d, color="0.25", lw=0.6, ls=":")
            ax.set_ylim(h, ytop)
            ax.axhline(0, color="0.5", lw=0.6)                # mean sea level
            ax.set_ylabel("Depth below MSL (m)")
            ax.set_facecolor("0.93")                         # gaps (no data) show as light grey
        fig.colorbar(cf, ax=[ao, am], label=unit, pad=0.01, fraction=0.025,
                     format="%.1f" if var == "temp" else "%.2f")
    span = (idx[-1] - idx[0]).days
    loc = (mdates.MonthLocator() if span > 60 else mdates.WeekdayLocator(byweekday=mdates.MO) if span > 21
           else mdates.DayLocator(interval=max(1, span // 10)))
    axes[-1].xaxis.set_major_locator(loc)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y" if span > 200 else "%b %d" if span <= 60
                                                            else "%b %Y"))
    axes[-1].set_xlim(idx[0], idx[-1])
    axes[-1].set_xlabel("Date (UTC)")
    fig.suptitle(title + "\nGrey: no data. Dotted lines: sensor depths.", fontsize=13)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


def update_data(end, group):
    """Refresh the group's observations from SIO, and extract any new model days up to end (UTC date) for
    every site on the same model level (one pass over the history files serves them all)."""
    import download_obs
    import extract_model
    for s in GROUPS[group]:
        download_obs.update_site(s)
    os.makedirs(extract_model.MODEL_DIR, exist_ok=True)
    level = level_of(GROUPS[group][0])
    sites = [s for s in extract_model.SITES if level_of(s) == level]
    extract_model.extract(sites, level, 1, "2024-12-05", end, workers=6)
    extract_model.extract_forecast(sites, level, end)
    for s in sites:
        extract_model.export(s, level, 1)
        extract_model.export(s, level, "forecast")
