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
  <site>_density_panels, IB_density  the same for density sigma0 (sigma0(): TEOS-10 gsw, from the hourly
                     T and S; the same function for obs and model)
  <site>_N2_panels, IB_N2  observed N^2 between neighbouring sensors (gsw.Nsquared), no model
  <site>_density_N2_contours  like the TS contours: observed and model sigma0 (one scale), then observed N^2
                     between neighbouring sensors and model N^2 between neighbouring s-levels (own scales)
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
import gsw
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from download_obs import SBOO_SIDELOBE, load_adcp, load_ts
from extract_model import SITES as MODEL_SITES, level_of

from extract_model import MODEL_DIR   # shared cache on /dataSIO/PFM_Simulations/cache/Moorings/model

# plots and their date windows are in Pacific time (PDT/PST); data and caches stay UTC. Matplotlib takes
# tz-aware times (and naive ones as UTC) and draws the date axes in this zone.
TZ = "America/Los_Angeles"
matplotlib.rcParams["timezone"] = TZ

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
DENS_UNIT = "Density σ0 (kg/m³)"
N2_UNIT = "N² (10⁻⁴ s⁻²)"


def sa_ct_p(salt, temp, depth, site):
    """TEOS-10 (gsw) absolute salinity, conservative temperature and sea pressure from practical salinity,
    in-situ temperature and depth below mean sea level (m, positive down) at the site's position. Model
    levels above mean sea level at high tide get a slightly negative pressure; not clipped to 0, else two
    such levels would share p = 0 and gsw.Nsquared would divide by zero."""
    lon, lat = MODEL_SITES[site]["lon"], MODEL_SITES[site]["lat"]
    p = gsw.p_from_z(-np.asarray(depth, float), lat)
    sa = gsw.SA_from_SP(np.asarray(salt, float), p, lon, lat)
    return sa, gsw.CT_from_t(sa, np.asarray(temp, float), p), p


def sigma0(salt, temp, depth, site):
    """Potential density anomaly referenced to 0 dbar, sigma0 = rho(SA, CT, 0) - 1000 (kg/m^3), TEOS-10 (gsw).
    Same function for observations and model. Returns the type of salt (Series stays a Series)."""
    v = gsw.sigma0(*sa_ct_p(salt, temp, depth, site)[:2])
    return pd.Series(v, index=salt.index) if isinstance(salt, pd.Series) else v


def n2(up, dn, z_up, z_dn, site):
    """Buoyancy frequency squared (10^-4 s^-2) between two depths, gsw.Nsquared (TEOS-10, local gravity).
    up, dn: DataFrames with temp and salt on one time index; z_up, z_dn: their depths (m, positive down)."""
    sa, ct, p = sa_ct_p(np.vstack([up["salt"], dn["salt"]]), np.vstack([up["temp"], dn["temp"]]),
                        np.vstack([np.broadcast_to(z_up, len(up)), np.broadcast_to(z_dn, len(dn))]), site)
    return pd.Series(1e4 * gsw.Nsquared(sa, ct, p, MODEL_SITES[site]["lat"], axis=0)[0][0], index=up.index)
_fc = None             # first model forecast hour (days 2-5 of the latest history file), set by plot_window
FC_SHADE = "0.94"      # background of the forecast period


def hour_of(t):
    """Whole UTC hour each sample belongs to: hour h gets samples in [h - 30 min, h + 30 min)."""
    return (t + pd.Timedelta(minutes=30)).dt.floor("h")


def hourly(df):
    df = df.copy()
    df["time"] = hour_of(df["time"])
    return df.groupby("time").mean()


def model_ts_col(df, var, d, site=None):
    """Model temp / salt / dens (sigma0; needs site) at sensor depth d from the hourly CSV columns, or None."""
    if df is None:
        return None
    if var == "dens":
        t, s = df.get(f"temp_{d:g}m"), df.get(f"sal_{d:g}m")
        return None if t is None or s is None else sigma0(s, t, d, site)
    return df.get(f"{'sal' if var == 'salt' else var}_{d:g}m")


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


def obs_only(idx):
    """idx without the model-forecast hours (days 2-5), for observation-only figures."""
    return idx[idx < _fc] if _fc is not None else idx


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


def obs_ts(ts, depth, site=None):
    """Hourly observed temp, salt and (with site) dens = sigma0 of the sensor at depth, or None."""
    ts = ts[ts["depth_m"] == depth]
    if ts.empty:
        return None
    df = hourly(ts.pivot_table(index="time", columns="var", values="value").reset_index())
    if site and {"temp", "salt"} <= set(df.columns):
        df["dens"] = sigma0(df["salt"], df["temp"], depth, site)     # from the hourly T and S of this sensor
    return df


def panel(ax, idx, obs, model, title, ylabel, obs_label="Observed", robust=False, mod_label="PFM day 1",
          with_model=True):
    """with_model=False: an observations-only panel (no "no model" note)."""
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
    missing = [w for w, s in (("observations", o), ("model", m)) if s is None and (with_model or w != "model")]
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


def ib_legend(fig, axes, keys=IB_SITES, y=0.972, kinds=("model", "observed")):
    """One legend above the panels: a column per mooring (or depth), model above observed."""
    seen = {l: h for ax in axes for h, l in zip(*ax.get_legend_handles_labels())}
    handles, labels = [], []
    for s in keys:
        for kind in kinds:
            label = f"{s} {kind}"
            if label in seen:
                handles.append(seen[label])
            else:  # keep the grid aligned: invisible placeholder
                handles.append(plt.Line2D([], [], alpha=0))
                label = ""
            labels.append(label)
    fig.legend(handles, labels, loc="upper center", ncol=len(keys), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, y))


def date_ticks(ax, span):
    """A labelled tick each (Pacific) day up to 45 days: day numbers, the month name where it starts;
    longer windows weekly or monthly labels, with a small tick each day up to 60 days."""
    if span <= 45:
        loc = mdates.DayLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc, show_offset=False))
        return
    loc, fmt = ((mdates.MonthLocator(), "%b\n%Y") if span > 200 else (mdates.MonthLocator(), "%b %Y") if span > 60
                else (mdates.WeekdayLocator(byweekday=mdates.MO), "%b %d"))
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    if span <= 60:
        ax.xaxis.set_minor_locator(mdates.DayLocator())


def finish(fig, axes, idx, title, out_path, top=1.0, ygroups=None, forecast=True):
    """ygroups: lists of panel indices that share one y range (default: all panels). forecast=False: no
    model-forecast shading (observation-only figures)."""
    for g in ygroups or [range(len(axes))]:
        lims = [axes[i].get_ylim() for i in g]
        for i in g:
            axes[i].set_ylim(min(l[0] for l in lims), max(l[1] for l in lims))
    date_ticks(axes[-1], (idx[-1] - idx[0]).days)
    for ax in axes if forecast else []:
        shade_forecast(ax, idx[-1])
    axes[-1].set_xlim(idx[0], idx[-1])
    axes[-1].set_xlabel("Date (Pacific time, PDT/PST)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


GROUPS = {"SBOO": ["SBOO"], "IB": IB_SITES, "PLOO": ["PLOO"], "DelMar": ["DelMar"]}


def pacific_window(start, end):
    """(first, last) hour of the Pacific dates start..end (both inclusive), as tz-aware UTC."""
    first = pd.Timestamp(start).tz_localize(TZ)
    last = (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize(TZ) - pd.Timedelta(hours=1)
    return first.tz_convert("UTC"), last.tz_convert("UTC")


def setup_window(start, end, sites):
    """Hourly index for Pacific dates start..end, extended by the model forecast when the window reaches it;
    sets _fc. Returns (idx, model {site: DataFrame}, resolution text)."""
    global _fc
    first, last = pacific_window(start, end)
    model = {s: load_model(s) for s in sites}
    # model forecast (days 2-5 of the latest history file): shown when the window reaches the end of day 1
    # (also when day 1 ends exactly at the window's end), running on past the window to the forecast's end
    day1 = [read_model_csv(model_csv(s, 1)) for s in sites]
    d1_end = max((d.index.max() for d in day1 if d is not None), default=None)
    fc_end = max((m.index.max() for m in model.values() if m is not None), default=None)
    _fc = (d1_end + pd.Timedelta(hours=1) if d1_end is not None and fc_end > d1_end and d1_end <= last else None)
    if _fc is not None:
        last = max(last, fc_end)
    else:
        model = {s: load_model(s, forecast=False) for s in sites}
    idx = pd.date_range(first, last, freq="h")
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
    """Temperature, salinity and density (sigma0) at each sensor depth, one panel per depth."""
    for var, unit, name in (("temp", "Temperature (°C)", f"{s}_temperature_panels"),
                            ("salt", "Salinity", f"{s}_salinity"),
                            ("dens", DENS_UNIT, f"{s}_density_panels")):
        fig, axes = plt.subplots(len(depths), 1, figsize=(13, max(11, 2.1 * len(depths))), sharex=True)
        for ax, d in zip(np.atleast_1d(axes), depths):
            o = obs_ts(ts, d, s)
            panel(ax, idx, col(o, var), model_ts_col(model[s], var, d, s), f"{s}: {d:g} m", unit, robust=True,
                  obs_label=f"Observed (sensor at {d:g} m)",
                  mod_label=f"PFM {level_of(s)} day 1 (interpolated to {d:g} m below mean sea level)")
        finish(fig, list(np.atleast_1d(axes)), idx,
               f"{NAME[s]}, {unit.split(' (')[0].lower()}: PFM vs observations ({res}), {period}"
               + ("\nσ0 = ρ(SA, CT, 0) − 1000 from the hourly T and S (TEOS-10, gsw), the same for obs and model"
                  if var == "dens" else ""), out(name), top=0.97 if var == "dens" else 1.0)


def sensor_slots(ts, depths, idx):
    """One entry per T/S sensor on idx: (label, depth series, observed [temp, salt]), NaN where either is
    missing. A sensor whose depth changed a little between deployments (e.g. Del Mar 1 m / 4 m, 89 m / 90 m:
    within 3 m and never both with data) is one slot, its depth taken hour by hour from whichever has data."""
    def ts_frame(t, s):
        f = pd.DataFrame({"temp": t.reindex(idx) if t is not None else np.nan,
                          "salt": s.reindex(idx) if s is not None else np.nan}, index=idx)
        f[f.isna().any(axis=1)] = np.nan
        return f

    obs = {}
    for d in depths:
        o = obs_ts(ts, d)
        obs[d] = ts_frame(col(o, "temp"), col(o, "salt"))
    has = {d: obs[d]["temp"].notna() for d in depths}
    groups = []
    for d in depths:
        if groups and d - groups[-1][-1] <= 3 and not any((has[d] & has[g]).any() for g in groups[-1]):
            groups[-1].append(d)
        else:
            groups.append([d])
    out = []
    for g in groups:
        dep, o = pd.Series(float(g[0]), idx), obs[g[0]].copy()
        for d in g[1:]:
            dep[has[d]], o[has[d]] = float(d), obs[d][has[d]]
        out.append(("/".join(f"{d:g}" for d in g) + " m", dep, o))
    return out


def fig_n2_panels(s, depths, idx, ts, res, period, out):
    """Observed N^2 between each pair of neighbouring sensors, one panel per pair (no model)."""
    idx = obs_only(idx)
    slots = sensor_slots(ts, depths, idx)
    pairs = list(zip(slots, slots[1:]))
    fig, axes = plt.subplots(len(pairs), 1, figsize=(13, max(11, 2.1 * len(pairs))), sharex=True)
    for ax, ((l0, z0, o0), (l1, z1, o1)) in zip(np.atleast_1d(axes), pairs):
        a, b = l0.replace(" m", ""), l1
        panel(ax, idx, n2(o0, o1, z0, z1, s), None, f"{s}: N² between {a} and {b}", N2_UNIT, robust=True,
              obs_label=f"Observed (sensors at {a} and {b})", with_model=False)
        ax.axhline(0, color="0.6", lw=0.8)
    finish(fig, list(np.atleast_1d(axes)), idx,
           f"{NAME[s]}, observed buoyancy frequency N² ({res}), {period}\n"
           "N² between neighbouring sensors from their T and S, TEOS-10 gsw.Nsquared (local gravity)",
           out(f"{s}_N2_panels"), top=0.97, forecast=False)


def sensor_depths(s, ts):
    """T/S sensor depths with data in the window (they vary a little between deployments), else the defaults."""
    have = sorted({float(d) for d in ts["depth_m"].unique()}) if len(ts) else []
    have = [d for d in have if f"temp_{d:g}m" in MODEL_COLUMNS(s)]
    return have or TS_DEFAULT[s]


def plot_window(start, end, out, period, group):
    """The figures of one group (SBOO, IB, PLOO or DelMar) for Pacific dates start..end (both inclusive).

    out(name) gives the PNG path for figure `name`; period is the text used in the titles.
    """
    sites = GROUPS[group]
    idx, model, res = setup_window(start, end, sites)
    start, end = (f"{t:%Y-%m-%d}" for t in pacific_window(start, end))   # UTC dates, for loading the obs
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
        ts_contours("SBOO", idx, ts["SBOO"], SBOO_TS_M,
                    f"SBOO density (σ0) and N² contours: observed vs PFM ({res}), {period}",
                    out("SBOO_density_N2_contours"), variables=DENS_N2_CONTOUR)
        fig_n2_panels("SBOO", SBOO_TS_M, idx, ts["SBOO"], res, period, out)

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
        # density (sigma0) at 1 m and at the deep sensor, all four moorings in each panel
        fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
        for ax, level in zip(axes, ("1 m", "deep")):
            lines = []
            for s in IB_SITES:
                d = 1 if level == "1 m" else DEEP_M[s]
                lines.append((s, col(obs_ts(ts[s], d, s), "dens"), model_ts_col(model[s], "dens", d, s)))
            title = ("Density σ0" + (" at 1 m" if level == "1 m" else f" at the deep sensor ({deep})")
                     + ": from T and S at the sensor; model T and S interpolated to the sensor depth")
            overlay_panel(ax, idx, lines, title, DENS_UNIT, robust=True)
        ib_legend(fig, axes, y=0.935)
        finish(fig, axes, idx, f"IB mini moorings density σ0 = ρ(SA, CT, 0) − 1000 (TEOS-10): PFM (solid"
               f"{', forecast dotted' if _fc else ''}) vs obs (dashed), {res}, {period}", out("IB_density"), top=0.915)
        # observed N^2 between the 1 m and the deep sensor, all four moorings in one panel (no model)
        fig, ax = plt.subplots(figsize=(13, 6))
        lines, oidx = [], obs_only(idx)
        for s in IB_SITES:
            (_, z0, o0), (_, z1, o1) = sensor_slots(ts[s], [1, DEEP_M[s]], oidx)
            lines.append((s, n2(o0, o1, z0, z1, s), None))
        overlay_panel(ax, oidx, lines, f"N² between the 1 m and the deep sensor ({deep})", N2_UNIT, robust=True)
        ax.axhline(0, color="0.6", lw=0.8)
        ib_legend(fig, [ax], y=0.93, kinds=("observed",))
        finish(fig, [ax], oidx, f"IB mini moorings observed N² (TEOS-10 gsw.Nsquared), {res}, {period}",
               out("IB_N2"), top=0.9, forecast=False)

    else:  # PLOO, DelMar: deep moorings with ADCP and T/S at ~9 depths
        s = group
        depths = sensor_depths(s, ts[s])
        fig_currents(s, idx, model, start, end, res, period, out)
        fig_ts_panels(s, depths, idx, model, ts[s], res, period, out)
        ts_contours(s, idx, ts[s], depths,
                    f"{NAME[s]} temperature and salinity contours: observed vs PFM ({res}), {period}",
                    out(f"{s}_TS_contours"))
        ts_contours(s, idx, ts[s], depths,
                    f"{NAME[s]} density (σ0) and N² contours: observed vs PFM ({res}), {period}",
                    out(f"{s}_density_N2_contours"), variables=DENS_N2_CONTOUR)
        fig_n2_panels(s, depths, idx, ts[s], res, period, out)


CONTOUR_CMAP = {"temp": cmocean.cm.thermal, "salt": cmocean.cm.haline,   # observed and model use the same one
                "dens": cmocean.cm.dense, "n2": cmocean.cm.balance}
PANELS = {}       # contour panels per variable if not both, e.g. {"n2": ("obs",)} (default: obs and model)
MODEL_VAR = {}    # model_ts_profiles key, if not the variable name
CONTOUR_LEVELS = 14
TS_CONTOUR = (("temp", "Temperature (°C)"), ("salt", "Salinity"))
DENS_CONTOUR = (("dens", DENS_UNIT),)
N2_CONTOUR = (("n2", N2_UNIT),)
# one figure like the T/S one: observed and model density, then observed and model N^2 (each pair one colour scale)
DENS_N2_CONTOUR = DENS_CONTOUR + N2_CONTOUR


def obs_n2_grid(slots, grid, idx, site):
    """Observed N^2 [hours x grid depths]: between each pair of neighbouring sensors one value per hour, put
    on the grid depths between them (a layer, not interpolated)."""
    g = np.full((len(idx), len(grid)), np.nan)
    for (_, z0, o0), (_, z1, o1) in zip(slots, slots[1:]):
        v = n2(o0, o1, z0, z1, site).to_numpy(float)
        sel = (grid[None, :] >= z0.to_numpy()[:, None]) & (grid[None, :] <= z1.to_numpy()[:, None])
        g = np.where(sel & np.isnan(g), v[:, None], g)
    return g


def obs_ts_grid(ts, depths, grid, idx, site):
    """Observed T, S and sigma0 [hours x grid depths]: linear in depth between neighbouring sensors, only
    where both sensors of the pair have a value (no extrapolation above the top or below the bottom sensor).
    sigma0 is computed at each sensor from its hourly T and S, then interpolated the same way."""
    out = {}
    for var in ("temp", "salt", "dens"):
        cols = {}
        for d in depths:
            o = obs_ts(ts, d, site)
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
    # N^2 between neighbouring s-levels (gsw.Nsquared), halfway between them; before the surface row is added
    sa, ct, p = sa_ct_p(d["salt"], d["temp"], -zr, site)
    n2_lev = 1e4 * gsw.Nsquared(sa, ct, p, MODEL_SITES[site]["lat"], axis=1)[0]
    n2_depth = -(zr[:, 1:] + zr[:, :-1]) / 2
    top = int(np.argmax(zr[0]))                         # uppermost rho level: first or last column
    at_end = top == zr.shape[1] - 1

    def surf(x, s):                                     # add the free-surface row on the surface side
        return np.column_stack([x, s] if at_end else [s, x])

    d = {**d, "temp": surf(d["temp"], d["temp"][:, top]), "salt": surf(d["salt"], d["salt"][:, top])}
    zr = surf(zr, np.asarray(d["zeta"], float).reshape(len(t), -1)[:, 0])
    d["dens"] = sigma0(d["salt"], d["temp"], -zr, site)  # sigma0 at every level, same function as the obs
    pos = pd.Series(np.arange(len(t)), index=t)
    pos = pos[~pos.index.duplicated(keep="last")].reindex(idx)
    ok = pos.notna().to_numpy()
    k = pos.to_numpy()[ok].astype(int)
    out = {}
    for name, x in (("temp", d["temp"]), ("salt", d["salt"]), ("dens", d["dens"]), ("depth", -zr),
                    ("n2", n2_lev), ("n2_depth", n2_depth)):
        a_ = np.full((len(idx), x.shape[1]), np.nan)
        a_[ok] = x[k]
        out[name] = a_
    return out, float(a["h"])


def n2_var(var):
    return var.startswith("n2")


def weighted_quantile(v, w, q):
    """Quantiles q (0-1) of values v with weights w (midpoint rule; equal weights give np.quantile's 'hazen')."""
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return np.interp(q, (cw - 0.5 * w[o]) / cw[-1], v[o])


def obs_sensor_values(ts, depths, idx, site):
    """Hourly observed temp, salt and dens (sigma0) of every sensor on idx, all sensors in one array per variable."""
    out = {}
    for var in ("temp", "salt", "dens"):
        cols = [col(obs_ts(ts, d, site), var) for d in depths]
        out[var] = np.concatenate([c.reindex(idx).to_numpy(float) for c in cols if c is not None] or [[]])
    return out


def ts_contours(site, idx, ts, depths, title, out_path, variables=TS_CONTOUR):
    """Depth-time contours, an observed and a model panel per variable ((var, label) in variables: temp,
    salt, dens, n2), each pair on one colour scale (PANELS can drop one of the two). Obs are interpolated between
    the sensors (dotted lines; N^2 one value per layer); the model shows the whole water column at the depths of its levels below mean
    sea level, which move with the tide."""
    m = model_ts_profiles(site, idx)
    grid = np.arange(depths[0], depths[-1] + 0.01, 0.25)
    obs = obs_ts_grid(ts, depths, grid, idx, site)
    real = obs_sensor_values(ts, depths, idx, site)        # for the colour limits (the grid is only for drawing)
    if any(v == "n2" for v, _ in variables):
        slots = sensor_slots(ts, depths, idx)
        obs["n2"] = obs_n2_grid(slots, grid, idx, site)
        real["n2"] = np.concatenate([n2(o0, o1, z0, z1, site).to_numpy(float)
                                     for (_, z0, o0), (_, z1, o1) in zip(slots, slots[1:])] or [[]])
    kinds = [PANELS.get(v, ("obs", "model")) for v, _ in variables]     # panels per variable, top to bottom
    npan = sum(len(k) for k in kinds)
    fig, axes = plt.subplots(npan, 1, figsize=(13, max(4.5, 3 * npan + 0.5)), sharex=True, layout="constrained")
    axes = list(np.atleast_1d(axes))
    free = iter(axes)
    h = m[1] if m else depths[-1] + 1
    # top of the axes: the highest model free surface in the window (above mean sea level, so negative)
    ytop = min(0.0, float(np.nanmin(m[0]["depth"]))) if m and np.isfinite(m[0]["depth"]).any() else 0.0
    idx = idx.tz_localize(None)                              # UTC, naive for matplotlib
    for (var, unit), kind in zip(variables, kinds):
        pans = {k: next(free) for k in kind}
        mvar = MODEL_VAR.get(var, var)
        mx = m[0][mvar] if m else None
        # colour limits from the measured values, not the drawing grid (which would weight each sensor layer by its
        # thickness and count it once per grid row): T, S, sigma0 from the hourly sensor values and the model
        # values pooled; N^2 from the observed sensor-pair values and the model values with equal total weight
        # (the model has ~40 level pairs per hour against ~8 sensor pairs, and much higher values in thin layers)
        ov = real[var][np.isfinite(real[var])] if "obs" in pans else np.array([])
        mv = mx[np.isfinite(mx)] if "model" in pans and m else np.array([])
        if not len(ov) + len(mv):
            for ax in pans.values():
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color="0.4")
            continue
        if n2_var(var) and len(ov) and len(mv):
            lo, hi = weighted_quantile(np.r_[ov, mv], np.r_[np.full(len(ov), 0.5 / len(ov)),
                                                            np.full(len(mv), 0.5 / len(mv))], [0.005, 0.995])
        else:
            lo, hi = np.quantile(np.r_[ov, mv], [0.005, 0.995])     # T/S are not outlier-QC'd
        levels = np.linspace(lo, hi, CONTOUR_LEVELS + 1)
        kw = dict(levels=levels, cmap=CONTOUR_CMAP[var], extend="both")
        n2var = n2_var(var)
        if n2var:   # diverging, white at 0: blue negative (unstable), red positive (stable)
            lo, hi = min(lo, -0.05 * hi), max(hi, 1e-6)
            nneg = max(2, round(CONTOUR_LEVELS * -lo / (hi - lo)))           # levels below 0
            levels = np.r_[np.linspace(lo, 0, nneg + 1)[:-1], np.linspace(0, hi, CONTOUR_LEVELS - nneg + 1)]
            kw = dict(levels=levels, cmap=CONTOUR_CMAP[var], extend="both",
                      norm=matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=lo, vmax=hi))
        name = unit.split(" (")[0] if n2var else unit.split(" (")[0].lower()     # keep "N²" upper case
        if "obs" in pans:
            ao = pans["obs"]
            if np.isfinite(obs[var]).any():
                cf = ao.contourf(idx, grid, obs[var].T, **kw)
            else:
                ao.text(0.5, 0.5, "no observations", transform=ao.transAxes, ha="center", color="0.4")
            how = ("from neighbouring sensors, one value per layer between the sensors at " if n2var
                   else "interpolated between the sensors at ")
            ao.set_title(f"{site} {name}: observed, {how}"
                         + ", ".join(f"{d:g}" for d in depths) + " m", loc="left", fontsize=10.5)
        if "model" in pans:   # each level at its own (tide-following) depth; N^2 halfway between levels
            am = pans["model"]
            if m and np.isfinite(mx).any():
                z = m[0][f"{mvar}_depth"] if f"{mvar}_depth" in m[0] else m[0]["depth"]
                T = np.repeat(np.asarray(idx)[:, None], mx.shape[1], axis=1)
                cf = am.contourf(T, z, mx, **kw)
            else:
                am.text(0.5, 0.5, "no model", transform=am.transAxes, ha="center", color="0.4")
            am.set_title(f"{site} {name}: PFM {level_of(site)} day 1, "
                         + ("between neighbouring s-levels" if n2var else "all model levels")
                         + (", then days 2-5 of the latest forecast (after the dashed line)" if _fc else ""),
                         loc="left", fontsize=10.5)
            if _fc is not None:
                am.axvline(_fc.tz_localize(None), color="white", lw=1.2, ls="--")
                am.text(_fc.tz_localize(None), 0.02, " model forecast", transform=am.get_xaxis_transform(),
                        ha="left", va="bottom", fontsize=8, color="white")
        for ax in pans.values():
            for d in depths:
                ax.axhline(d, color="0.25", lw=0.6, ls=":")
            ax.set_ylim(h, ytop)
            ax.axhline(0, color="0.5", lw=0.6)                # mean sea level
            ax.set_ylabel("Depth below MSL (m)")
            ax.set_facecolor("0.93")                         # gaps (no data) show as light grey
        fig.colorbar(cf, ax=list(pans.values()), label=unit, pad=0.01, fraction=0.025,
                     format="%.1f" if var == "temp" else "%.2f")
    date_ticks(axes[-1], (idx[-1] - idx[0]).days)
    axes[-1].set_xlim(idx[0], idx[-1])
    axes[-1].set_xlabel("Date (Pacific time, PDT/PST)")
    fig.suptitle(title + "\nGrey: no data. Dotted lines: sensor depths.", fontsize=13)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path)


def update_data(end, group):
    """Refresh the group's observations from SIO, and extract any new model days up to end for
    every site on the same model level (one pass over the history files serves them all). end is a Pacific
    date; its evening is the next UTC day, so the model is extracted one UTC day further."""
    end = f"{pd.Timestamp(end) + pd.Timedelta(days=1):%Y-%m-%d}"
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
