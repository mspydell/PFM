"""Operational vs PFM DA shoreline total dye (dye_01 + dye_02) vs ddPCR ENT at IB-079 (Avenida Lunar),
IB-068 (Silver Strand) and EH-030 (Imperial Beach Pier, the QC-website IB site).

Inputs (all caches, nothing re-read from the his files here):
  - operational Day 1 dye: station_dye01_dye02_day1.csv and ddPCR: ddpcr_ent_recent.csv, both in the shared
    cache /dataSIO/PFM_Simulations/cache/Dye_ddPCRENT (~/modelQC/Dye_vs_ddPCRENT/update_cache.py, refreshed
    daily by the dashboard cron)
  - DA dye: da_station_dye.csv (extract_da_dye.py)
Outputs (plots_web/):
  - da_dye_ddpcr_timeseries.png: 3 panels, dye left axis / ENT right axis tied so ENT 1413 <-> dye 1e-5,
    QC-website risk bands, x in Pacific time, r vs ddPCR (log10, same samples as the scatter) per source
  - da_dye_ddpcr_scatter.png: 2x2, ddPCR ENT (x) vs dye (mean over sample hour +/- 3 h, y) as on the
    dashboard, per station + all, r of log10 and log-log least-squares best-fit lines
  - da_dye_ddpcr_samples.csv (in the DA cache, not figures): the matched samples
Usage: python plot_da_dye_ddpcr.py [start, default 2026-10-19]
"""

import os
import sys
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# fit equation written as on the ddPCR dashboard / QC dye vs ddPCR plot: dye = 10^c(±se) ENT^m(±se)
from qc_paths import DYE_DDPCR  # noqa: E402
sys.path.insert(0, DYE_DDPCR)
from plot_dye_ddpcr import loglog_fit  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DA_CACHE = os.path.join(os.environ.get('HYDRO_QC_DATA', '/dataSIO/PFM_Simulations/cache'), 'DA')  # CSV caches
CACHE = os.path.join(os.environ.get('HYDRO_QC_DATA', '/dataSIO/PFM_Simulations/cache'), 'Dye_ddPCRENT')
OPS_CSV = os.path.join(CACHE, 'station_dye01_dye02_day1.csv')
DA_CSV = os.path.join(DA_CACHE, 'da_station_dye.csv')
DDPCR_CSV = os.path.join(CACHE, 'ddpcr_ent_recent.csv')
# dye column suffix of each station: the cache uses the website site names, da_station_dye.csv the codes
OPS_SITE = {'IB-079': 'Coronado_AvLunar', 'IB-068': 'SilverStrand', 'EH-030': 'IB_pier'}
DA_SITE = {st: st.replace('-', '') for st in OPS_SITE}
FIG = os.path.join(HERE, 'plots_web')   # website figures, as in the other QC folders
TZ = ZoneInfo('America/Los_Angeles')

STATIONS = {'EH-030': 'EH-030 Imperial Beach Pier', 'IB-068': 'IB-068 Silver Strand', 'IB-079': 'IB-079 Avenida Lunar'}  # south to north (panel order)
SOURCES = {'ops': ('Operational Day 1', '#eb6834', 'o'), 'da': ('PFM DA', '#2a78d6', '^')}   # same as the SBOO currents plot
DT = 3                                   # +/- hours around the sample hour
DYE_THR, ENT_THR = 1e-5, 1413
Y0, Y1, Y2, Y3 = 1e-7, 1e-5, 1e-3, 10**-1.5   # QC-website dye risk bands
DYE_LO, DYE_HI = 1e-6, 1e-2
ENT_LO, ENT_HI = DYE_LO * ENT_THR / DYE_THR, DYE_HI * ENT_THR / DYE_THR


def total_dye(path, start, site):
    d = pd.read_csv(path, parse_dates=['time'], index_col='time')
    d = d[d.index >= start.tz_localize(None)].asfreq('h')
    out = pd.DataFrame({st: d[f'ptb_{site[st]}'] + d[f'tjre_{site[st]}'] for st in STATIONS})
    out.index = out.index.tz_localize('UTC')
    return out


def load_ddpcr(start):
    r = pd.read_csv(DDPCR_CSV, parse_dates=['time_utc'])
    r = r[r['station'].isin(STATIONS)]
    s = pd.DataFrame({'station': r['station'], 'time': r['time_utc'].dt.tz_localize('UTC'),
                      'ent': r['ent']}).dropna()
    s = s[s['time'] >= start].drop_duplicates().sort_values('time').reset_index(drop=True)
    return s


def window_mean(series, t):
    h = t.floor('h')
    w = series.loc[h - pd.Timedelta(hours=DT): h + pd.Timedelta(hours=DT)]
    return w.mean() if w.notna().sum() else np.nan


def log_fit(o, src):
    """Matched (ENT, dye) for one source, r of log10 values and the log-log least-squares fit
    (slope m, intercept c, equation 'dye = 10^c ENT^m' with standard errors, as on the dashboard)."""
    m = o[src].notna() & (o[src] > 0) & (o['ent'] > 0)
    x, y = o.loc[m, 'ent'], o.loc[m, src]
    if len(x) <= 2:
        return x, y, np.nan, None
    return x, y, np.corrcoef(np.log10(x), np.log10(y))[0, 1], loglog_fit(x.to_numpy(), y.to_numpy())


def plot_timeseries(dye, obs, t0, t1):
    fig, axes = plt.subplots(len(STATIONS), 1, figsize=(12, 10), sharex=True)
    for ax, (st, title) in zip(axes, STATIONS.items()):
        ax.axhspan(Y0, Y1, color='green', alpha=0.2, lw=0)
        ax.axhspan(Y1, Y2, color='yellow', alpha=0.3, lw=0)
        ax.axhspan(Y2, Y3, color='red', alpha=0.2, lw=0)
        for src, (label, color, _) in SOURCES.items():
            ax.plot(dye[src].index, dye[src][st], color=color, lw=1.5, label=label)
        ax.set_yscale('log')
        ax.set_ylim(DYE_LO, DYE_HI)
        ax.set_ylabel('Total dye')
        ax.set_title(title, loc='left', fontsize=11)
        ax.grid(alpha=0.3, which='major')
        axr = ax.twinx()
        o = obs[obs['station'] == st]
        # r vs ddPCR (log10, dye = mean of sample hour +/- DT h, same samples as the scatter plot)
        for i, (src, (label, color, _)) in enumerate(SOURCES.items()):
            x, _, r, _ = log_fit(o, src)
            ax.text(1 - 0.27 * (len(SOURCES) - 1 - i), 1.02, f'{label}: r = {r:.2f} (n={len(x)})',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=10, color=color, fontweight='bold')
        hi = o[o['ent'] > ENT_THR]
        lo = o[o['ent'] <= ENT_THR]
        axr.scatter(hi['time'], hi['ent'], s=60, c='red', edgecolors='black', lw=0.8, zorder=5,
                    label=f'ddPCR ENT > {ENT_THR}')
        axr.scatter(lo['time'], lo['ent'], s=25, c='black', zorder=5, label=f'ddPCR ENT <= {ENT_THR}')
        axr.axhline(ENT_THR, color='red', lw=0.8, ls=':')
        axr.set_yscale('log')
        axr.set_ylim(ENT_LO, ENT_HI)
        axr.set_ylabel('ddPCR ENT (copies/100 ml)')
        if ax is axes[0]:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = axr.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8, ncol=4, framealpha=0.9)
    axes[-1].set_xlim(t0, t1)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2, tz=TZ))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d', tz=TZ))
    axes[-1].set_xlabel('Date (Pacific time, PDT/PST)')
    fig.suptitle('Shoreline total dye: operational Day 1 vs PFM DA, with ddPCR ENT', fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIG, 'da_dye_ddpcr_timeseries.png')
    fig.savefig(path, dpi=150)
    print(path)


def plot_scatter(obs):
    panels = list(STATIONS) + ['all']
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
    for ax, st in zip(axes.flat, panels):
        o = obs if st == 'all' else obs[obs['station'] == st]
        for src, (label, color, marker) in SOURCES.items():
            x, y, r, fit = log_fit(o, src)
            ax.scatter(x, y, s=45, c=color, marker=marker, edgecolors='white', lw=0.8,
                       label=f'{label} (n={len(x)}, r={r:.2f})')
            if fit is not None:   # best-fit line in log-log space, drawn over the sampled ENT range
                slope, icpt, eq = fit
                xf = np.logspace(np.log10(x.min()), np.log10(x.max()), 50)
                ax.plot(xf, 10 ** (icpt + slope * np.log10(xf)), color=color, lw=1.8, label=f'  {eq}')
        ax.axhline(DYE_THR, color='gray', lw=0.8, ls='--')
        ax.axvline(ENT_THR, color='red', lw=0.8, ls=':')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(ENT_LO, ENT_HI)   # same ddPCR range as the time-series right axis (tied to the dye axis)
        ax.set_ylim(DYE_LO, DYE_HI)
        ax.grid(alpha=0.3)
        ax.set_title('All three stations' if st == 'all' else STATIONS[st], fontsize=11)
        ax.legend(loc='lower right', fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel('ddPCR ENT (copies/100 ml)')
    for ax in axes[:, 0]:
        ax.set_ylabel(f'Total dye (mean of sample hour +/- {DT} h)')
    fig.suptitle(f'ddPCR ENT vs dye, {obs["time"].min():%Y-%m-%d} to {obs["time"].max():%Y-%m-%d} '
                 '(r and best-fit lines of log10 values)', fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIG, 'da_dye_ddpcr_scatter.png')
    fig.savefig(path, dpi=150)
    print(path)


def main():
    start = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else '2026-10-19', tz='UTC')
    os.makedirs(FIG, exist_ok=True)
    dye = {'ops': total_dye(OPS_CSV, start, OPS_SITE), 'da': total_dye(DA_CSV, start, DA_SITE)}
    obs = load_ddpcr(start)
    for src in SOURCES:
        obs[src] = [window_mean(dye[src][st], t) for st, t in zip(obs['station'], obs['time'])]
    obs.to_csv(os.path.join(DA_CACHE, 'da_dye_ddpcr_samples.csv'), index=False)
    t1 = max(dye['ops'].index.max(), dye['da'].index.max())
    plot_timeseries(dye, obs, start, t1)
    plot_scatter(obs)
    print(obs.to_string())


if __name__ == '__main__':
    main()
