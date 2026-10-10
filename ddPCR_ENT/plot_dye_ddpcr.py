#!/usr/bin/env python3
"""Daily 3-panel plot of PFM dye forecasts (day1/day5) vs ddPCR Enterococcus.

Window: past 2 weeks (default), month, or year, through the next 5 days (forecast).
Left y-axis: total dye (log) with green/yellow/red risk bands.
Right y-axis: ddPCR Enterococcus (copies/100 ml); red markers > 1413, else small black.

Dye times are UTC and ddPCR sample times are Pacific local; both are kept timezone-aware
and the x-axis is drawn in Pacific time (PDT/PST switch handled automatically).
"""
import argparse
import glob
import os
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from download_water_quality import download_sd_water_quality

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
# make_forecast_csv.py names its output after the newest run; use the latest one
DYE_CSV_GLOB = os.path.join(OUT_DIR, "sites_dye_tot_day1_day5_??????????.csv")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

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


def load_ddpcr(start, end, force_beachwatch=False):
    """ddPCR Enterococcus for the three stations, with Pacific-time sample times.

    The County portal only reaches back a few months, so long windows use State BeachWatch.
    """
    csv_path = os.path.join(OUT_DIR, "ddpcr_ent_recent.csv")
    df = download_sd_water_quality(
        output_csv=csv_path,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        parameter="Enterococcus",
        ddpcr_only=True,
        force_beachwatch=force_beachwatch,
    )
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["station", "time", "value"])
    # BeachWatch puts a numeric id in Station_ID; Station Name holds the code in both sources
    df["station"] = df["Station Name"].str.strip().str.upper()
    df = df[df["station"].isin([s for _, s in SITES.values()])].copy()
    local = pd.to_datetime(df["SampleDate"].dt.strftime("%Y-%m-%d") + " "
                           + df["SampleTime"].fillna("12:00:00"))
    df["time"] = local.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
    df["value"] = df["Result_Numeric"]
    return df.dropna(subset=["time", "value"])


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

    dye_csv = max(glob.glob(DYE_CSV_GLOB))
    print("reading", os.path.basename(dye_csv))
    dye = pd.read_csv(dye_csv, parse_dates=["datetime_utc"], index_col="datetime_utc")
    dye.index = dye.index.tz_localize("UTC")
    dye = dye.loc[t0:t1]
    # the County portal only reaches back a few months
    ent = load_ddpcr(t0, min(t1, now), force_beachwatch=(now - t0).days > 60)
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
        ax.set_title(f"{site_name} ({station})", loc="left", fontsize=11)

        axr = ax.twinx()
        e = ent[ent["station"] == station]
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

    axes[-1].set_xlim(t0, t1)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter(tick_fmt, tz=TZ))
    axes[-1].set_xlabel("Date (Pacific time, PDT/PST)")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("wrote", out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date", help="Plot date YYYY-MM-DD (default: today)")
    p.add_argument("--period", choices=PERIODS, default="2weeks",
                   help="How far back to plot (default: 2weeks)")
    args = p.parse_args()
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
