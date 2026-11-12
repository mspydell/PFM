#!/usr/bin/env python3
"""Daily 3-panel plot of PFM dye forecasts (day1/day5) vs ddPCR Enterococcus.

Window: past 2 weeks (default), month, or year, through the next 5 days (forecast).
Left y-axis: total dye (log) with green/yellow/red risk bands.
Right y-axis: ddPCR Enterococcus (copies/100 ml); red markers > 1413, else small black.

Data come from the shared cache $HYDRO_QC_DATA/Dye_ddPCRENT (default
/dataSIO/PFM_Simulations/cache/Dye_ddPCRENT):
  dye:   the newest sites_dye_tot_day1_day5_<YYYYMMDDHH>.csv (make_forecast_csv.py)
  ddPCR: ddpcr_ent_recent.csv (update_cache.py); this script refreshes it first unless --no-update.
Times are kept timezone-aware and the x-axis is drawn in Pacific time (PDT/PST).
"""
import argparse
import glob
import os
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from update_cache import update_ddpcr

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.environ.get("HYDRO_QC_DATA", "/dataSIO/PFM_Simulations/cache"), "Dye_ddPCRENT")
# make_forecast_csv.py names its output after the newest run; use the latest one
DYE_CSV_GLOB = os.path.join(CACHE, "sites_dye_tot_day1_day5_??????????.csv")
DDPCR_CSV = os.path.join(CACHE, "ddpcr_ent_recent.csv")
PLOT_DIR = os.path.join(HERE, "plots")

TZ = ZoneInfo("America/Los_Angeles")  # plot time zone

# --period choice -> (days back, x tick locator, tick format)
PERIODS = {
    "2weeks": (14,  mdates.DayLocator(interval=2, tz=TZ), "%b %d"),
    "month":  (30,  mdates.DayLocator(interval=4, tz=TZ), "%b %d"),
    "year":   (365, mdates.MonthLocator(tz=TZ),           "%b %Y"),
}
FUTURE_DAYS = 5
ENT_THRESHOLD = 1413  # copies/100 ml

# dye risk bands
Y0, Y1, Y2, Y3 = 10**-7, 1e-5, 1e-3, 10**-1.5

DYE_LO, DYE_HI = 1e-6, 1e-2  # left-axis limits
PAIR_HOURS = 3  # each ddPCR sample is paired with the mean dye within +-PAIR_HOURS h (as dt in the Matlab scripts)

# right-axis limits: left axis scaled by ENT_THRESHOLD / Y1, so both span the same
# decades and ENT_THRESHOLD lines up with Y1 (top of the green band)
_scale = ENT_THRESHOLD / Y1
ENT_LO, ENT_HI = DYE_LO * _scale, DYE_HI * _scale

# dye site prefix -> (panel title, ddPCR station ID)
SITES = {
    "IB_pier":          ("Imperial Beach Pier", "EH-030"),
    "SilverStrand":     ("Silver Strand",       "IB-068"),
    "Coronado_AvLunar": ("Coronado (Avenida Lunar)", "IB-079"),
}


def load_ddpcr(start, end):
    """ddPCR Enterococcus at the three stations from the cache, sample times in Pacific time."""
    df = pd.read_csv(DDPCR_CSV, parse_dates=["time_utc"])
    df = df[df["station"].isin([s for _, s in SITES.values()])].dropna(subset=["time_utc", "ent"])
    df["time"] = df["time_utc"].dt.tz_localize("UTC").dt.tz_convert(TZ)
    df["value"] = df["ent"]
    return df[(df["time"] >= start) & (df["time"] <= end)]


def dye_ent_pairs(dye, e, col):
    """(ddPCR ENT, dye) of each sample in e, dye the mean of column col within +-PAIR_HOURS h of it; both > 0."""
    win = pd.Timedelta(hours=PAIR_HOURS)
    pairs = [(v, dye[col].loc[t - win:t + win].mean()) for t, v in zip(e["time"], e["value"])]
    x, y = np.array(pairs, float).reshape(-1, 2).T
    ok = (x > 0) & (y > 0)
    return x[ok], y[ok]


def dye_ent_corr(dye, e, col):
    """Pearson r of log10(dye) and log10(ddPCR ENT) over dye_ent_pairs. (r, n), r nan when fewer than 3 pairs."""
    x, y = dye_ent_pairs(dye, e, col)
    if x.size < 3:
        return np.nan, int(x.size)
    return np.corrcoef(np.log10(x), np.log10(y))[0, 1], int(x.size)


def r_label(dye, e, prefix):
    """', r = <day-1 r>' for a panel title ('' with fewer than 3 pairs)."""
    r, _ = dye_ent_corr(dye, e, f"{prefix}_day1")
    return f", r = {r:.2f}" if np.isfinite(r) else ""


def load_window(t0, t1):
    """(dye, ddPCR) of the window t0..t1 (tz-aware); ddPCR up to min(t1, now)."""
    dye_csv = max(glob.glob(DYE_CSV_GLOB))
    print("reading", os.path.basename(dye_csv))
    dye = pd.read_csv(dye_csv, parse_dates=["datetime_utc"], index_col="datetime_utc")
    dye.index = dye.index.tz_localize("UTC")
    return dye.loc[t0:t1], load_ddpcr(t0, min(t1, pd.Timestamp.now(TZ)))


def loglog_fit(x, y):
    """Least-squares line log10(y) = m log10(x) + c: (m, c, its equation 'dye = 10^c ENT^m' with the standard
    errors of c and m in brackets, as on the ddPCR dashboard), None with fewer than 3 points."""
    lx, ly = np.log10(x), np.log10(y)
    n = lx.size
    if n < 3:
        return None
    mx, my = lx.mean(), ly.mean()
    sxx, sxy, syy = ((lx - mx) ** 2).sum(), ((lx - mx) * (ly - my)).sum(), ((ly - my) ** 2).sum()
    m = sxy / sxx
    c = my - m * mx
    s2 = (syy - m * sxy) / (n - 2)                     # residual variance
    sm, sc = np.sqrt(s2 / sxx), np.sqrt(s2 * (1 / n + mx * mx / sxx))
    pm = lambda v: f"(\\pm{v:.3f})" if v < 0.1 else f"(\\pm{v:.2f})"
    return m, c, f"dye = 10$^{{{c:.2f}{pm(sc)}}}$ ENT$^{{{m:.2f}{pm(sm)}}}$"


def plot_scatter(t0, t1, out_path, title):
    """Dye vs ddPCR ENT scatter: a panel for day 1 and one for day 5, the three beaches in each with their own
    marker, in the time-series colours (day 1 blue, day 5 purple) over the dye risk bands. Each past sample
    with the mean dye within +-PAIR_HOURS h; Pearson r of the logs per beach (legend) and for all three
    (title), and the least-squares line of all three in log-log space (as the Matlab scripts)."""
    dye, ent = load_window(t0, t1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.4), sharex=True, sharey=True)
    for ax, (day, color) in zip(axes, ((1, "tab:blue"), (5, "tab:purple"))):
        ax.axhspan(Y0, Y1, color="green", alpha=0.2, lw=0)
        ax.axhspan(Y1, Y2, color="yellow", alpha=0.3, lw=0)
        ax.axhspan(Y2, Y3, color="red", alpha=0.2, lw=0)
        ax.axvline(ENT_THRESHOLD, color="red", lw=0.8, ls=":")
        xs, ys = [], []
        for (prefix, (site_name, station)), marker in zip(SITES.items(), ("o", "s", "^")):
            x, y = dye_ent_pairs(dye, ent[ent["station"] == station], f"{prefix}_day{day}")
            xs.append(x)
            ys.append(y)
            label = f"{site_name} ({station})"
            fit = loglog_fit(x, y)                     # this beach: r and least-squares line in log-log space
            if fit:
                label += f", r = {np.corrcoef(np.log10(x), np.log10(y))[0, 1]:.2f}, {fit[2]}"
            ax.scatter(x, np.maximum(y, DYE_LO), s=55, c=color, marker=marker, edgecolors="black", linewidths=0.6,
                       zorder=3, label=label)
        x, y = np.concatenate(xs), np.concatenate(ys)
        r_all = ""
        if x.size >= 3:
            lx, ly = np.log10(x), np.log10(y)
            b, a, eq = loglog_fit(x, y)
            xf = np.array([x.min(), x.max()])
            ax.plot(xf, 10 ** (a + b * np.log10(xf)), color=color, lw=1.8, ls="--", zorder=2)
            r_all = f", r = {np.corrcoef(lx, ly)[0, 1]:.2f} (all {x.size})"
            ax.text(0.02, 0.03, f"Fit, all three: {eq}", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=9, color="black", zorder=5,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(ENT_LO, ENT_HI)                      # the time series' ddPCR axis
        ax.set_ylim(DYE_LO, DYE_HI)
        ax.set_xlabel("ddPCR ENT (copies/100 ml)")
        ax.set_title(f"Dye day {day}{r_all}", loc="left", fontsize=11)
        ax.grid(True, which="major", color="gray", lw=0.6, alpha=0.5)   # one line per decade
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8.5, frameon=False)
    axes[0].set_ylabel("Total dye")
    fig.suptitle(title)
    fig.text(0.01, 0.01, f"Each past ddPCR sample paired with the mean dye within ±{PAIR_HOURS} h; r: Pearson "
                         f"correlation of log10(dye) and log10(ddPCR ENT);\ndashed: least-squares line of the three "
                         f"beaches in log-log space; (±): standard errors of the fit. Dotted: ENT = {ENT_THRESHOLD}. Dye below 10$^{{-6}}$ drawn at "
                         f"10$^{{-6}}$.", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("wrote", out_path)


def auto_ticks(span_days):
    """(locator, tick format) suited to a window of span_days."""
    if span_days <= 20:
        return mdates.DayLocator(interval=2, tz=TZ), "%b %d"
    if span_days <= 45:
        return mdates.DayLocator(interval=4, tz=TZ), "%b %d"
    if span_days <= 120:
        return mdates.WeekdayLocator(byweekday=mdates.MO, tz=TZ), "%b %d"
    if span_days <= 550:
        return mdates.MonthLocator(tz=TZ), "%b %Y"
    return mdates.MonthLocator(interval=3, tz=TZ), "%b %Y"


def plot(today, period, out_path):
    past_days, locator, tick_fmt = PERIODS[period]
    t0 = today - pd.Timedelta(days=past_days)
    t1 = today + pd.Timedelta(days=FUTURE_DAYS + 1)
    label = "2 weeks" if period == "2weeks" else period
    title = f"PFM dye forecast vs ddPCR Enterococcus — {today:%Y-%m-%d} (past {label})"
    plot_window(t0, t1, out_path, title, locator, tick_fmt)


def plot_window(t0, t1, out_path, title, locator=None, tick_fmt=None):
    """3-panel plot for any window t0..t1 (tz-aware); ddPCR is fetched up to min(t1, now)."""
    span_days = (t1 - t0).days
    if locator is None:
        locator, tick_fmt = auto_ticks(span_days)
    now = pd.Timestamp.now(TZ)
    dye, ent = load_window(t0, t1)
    ms = 1.0 if span_days <= 31 else 0.35  # shrink markers on long windows

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    for ax, (prefix, (site_name, station)) in zip(axes, SITES.items()):
        ax.axhspan(Y0, Y1, color="green", alpha=0.2, lw=0)
        ax.axhspan(Y1, Y2, color="yellow", alpha=0.3, lw=0)
        ax.axhspan(Y2, Y3, color="red", alpha=0.2, lw=0)
        ax.plot(dye.index, dye[f"{prefix}_day1"], color="tab:blue", lw=1.5, label="Dye day 1")
        ax.plot(dye.index, dye[f"{prefix}_day5"], color="tab:purple", lw=1.5, ls="--", label="Dye day 5")
        ax.axvline(now, color="gray", lw=1, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(DYE_LO, DYE_HI)
        ax.set_ylabel("Total dye")

        axr = ax.twinx()
        e = ent[ent["station"] == station]
        ax.set_title(f"{site_name} ({station}){r_label(dye, e, prefix)}", loc="left", fontsize=11)
        hi = e[e["value"] > ENT_THRESHOLD]
        lo = e[e["value"] <= ENT_THRESHOLD]
        axr.scatter(lo["time"], lo["value"], s=25 * ms, c="black", marker="o", zorder=3,
                    label=f"ddPCR ENT ≤ {ENT_THRESHOLD}")
        axr.scatter(hi["time"], hi["value"], s=70 * ms, c="red", edgecolors="black", linewidths=0.8 * ms,
                    marker="o", zorder=4, label=f"ddPCR ENT > {ENT_THRESHOLD}")
        axr.axhline(ENT_THRESHOLD, color="red", lw=0.8, ls=":")
        axr.text(0.995, ENT_THRESHOLD, f"ddPCR ENT = {ENT_THRESHOLD}", color="red", fontsize=8,
                 va="bottom", ha="right", zorder=5, transform=axr.get_yaxis_transform())
        axr.set_yscale("log")
        axr.set_ylim(ENT_LO, ENT_HI)
        axr.set_ylabel("ddPCR ENT (copies/100 ml)")

    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, 0.965), fontsize=8, ncol=4, frameon=False)

    # end the axis at the midnight after the last dye value (the forecast can stop before t1)
    has = dye.dropna(how="all").index
    xmax = min(t1, has.max().tz_convert(TZ).ceil("D")) if len(has) else t1
    axes[-1].set_xlim(t0, max(xmax, min(t1, now)))
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter(tick_fmt, tz=TZ))
    axes[-1].set_xlabel("Date (Pacific time, PDT/PST)")
    fig.suptitle(title)
    fig.text(0.01, 0.005, f"r: Pearson correlation of log10(day-1 dye) and log10(ddPCR ENT); each sample paired with "
             f"the mean dye within ±{PAIR_HOURS} h, past samples in the window", fontsize=8)
    fig.tight_layout(rect=(0, 0.015, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("wrote", out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date", help="Plot date YYYY-MM-DD (default: today)")
    p.add_argument("--period", choices=PERIODS, default="2weeks",
                   help="How far back to plot (default: 2weeks)")
    p.add_argument("--no-update", action="store_true", help="Plot the cached ddPCR without downloading new samples")
    args = p.parse_args()
    if not args.no_update:
        os.umask(0o002)  # the cache is shared with the group
        update_ddpcr()
    # local midnight of the plot day
    today = pd.Timestamp(args.date).tz_localize(TZ) if args.date else pd.Timestamp.now(TZ).normalize()
    os.makedirs(PLOT_DIR, exist_ok=True)
    suffix = f"_{args.period}"
    out_path = os.path.join(PLOT_DIR, f"dye_ddpcr_{today:%Y%m%d}{suffix}.png")
    plot(today, args.period, out_path)
    # daily run: keep only the newest plot for this period
    if not args.date:
        for old in glob.glob(os.path.join(PLOT_DIR, f"dye_ddpcr_{'[0-9]' * 8}{suffix}.png")):
            if old != out_path:
                os.remove(old)
                print("removed", os.path.basename(old))


if __name__ == "__main__":
    main()
