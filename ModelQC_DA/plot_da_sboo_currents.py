"""SBOO currents: observed ADCP vs operational PFM day 1 (QC-website figure, no forecast) with PFM DA on top.

Same 4 panels, obs, depth ranges and styling as ~/modelQC/Moorings SBOO_currents (plot_model_vs_obs.py),
plus the DA series (da_sboo_currents.csv from extract_da_sboo.py) and its stats vs obs.
Operational model and obs come from the shared Moorings caches; --update refreshes the SBOO obs first.

Usage: ~/anaconda3/bin/python plot_da_sboo_currents.py [--start 2026-10-19] [--end today] [--update]
"""

import glob
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DA_CACHE = os.path.join(os.environ.get('HYDRO_QC_DATA', '/dataSIO/PFM_Simulations/cache'), 'DA')  # CSV caches
from qc_paths import MOORINGS_COMMON  # noqa: E402
sys.path.insert(0, MOORINGS_COMMON)
import plot_model_vs_obs as pm  # noqa: E402
from plot_model_vs_obs import OBS_LABEL, TOP_M, col, finish, model_label, obs_currents, panel, pacific_window  # noqa: E402

DA_CSV = os.path.join(DA_CACHE, 'da_sboo_currents.csv')
DA_COLOR = '#2a78d6'          # blue (obs black, operational orange)
S = 'SBOO'
pm.OBS_COLOR = 'black'        # observed in black (this figure only; QC website keeps blue)


def stats_text(o, m):
    both = pd.concat([o, m], axis=1).dropna()
    if len(both) <= 48:
        return None
    d = both.iloc[:, 1] - both.iloc[:, 0]
    return f'r = {both.corr().iloc[0, 1]:.2f}, RMSE = {np.sqrt((d ** 2).mean()):.3g}, bias = {d.mean():+.3g}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', default='2026-10-19', help='Pacific start date')
    p.add_argument('--end', default='today', help='Pacific end date (inclusive)')
    p.add_argument('--update', action='store_true', help='download new SBOO obs first')
    a = p.parse_args()
    end = pd.Timestamp.now(tz=pm.TZ).tz_localize(None).normalize() if a.end == 'today' else pd.Timestamp(a.end)
    end = f'{end:%Y-%m-%d}'
    if a.update:
        import download_obs
        download_obs.update_site(S)

    # day 1 only, no model forecast: hourly index over the Pacific dates start..end
    first, last = pacific_window(a.start, end)
    idx, res = pd.date_range(first, last, freq='h'), 'hourly'
    model = {S: pm.load_model(S, forecast=False)}
    pm._fc = None
    ustart, uend = (f'{t:%Y-%m-%d}' for t in pacific_window(a.start, end))
    davg, top = obs_currents(S, ustart, uend)
    da = pd.read_csv(DA_CSV, parse_dates=['time'], index_col='time')
    da.index = da.index.tz_localize('UTC') if da.index.tz is None else da.index.tz_convert('UTC')

    import matplotlib.pyplot as plt
    rows = [(f'top {TOP_M[S]:g} m', 'North', 'v_north', 'v_top'), ('depth-averaged', 'North', 'v_north', 'v_davg'),
            (f'top {TOP_M[S]:g} m', 'East', 'u_east', 'u_top'), ('depth-averaged', 'East', 'u_east', 'u_davg')]
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    for ax, (kind, comp, ocol, mcol) in zip(axes, rows):
        o = davg if kind == 'depth-averaged' else top
        which = 'top' if kind.startswith('top') else 'davg'
        obs = col(o, ocol)
        panel(ax, idx, obs, col(model[S], mcol), f'{S} {kind}: {comp.lower()}ward', f'{comp} (m/s)',
              OBS_LABEL[(S, which)], mod_label=model_label(S, which))
        # stats (vs obs) right of the panel, outside the axes, so they never cover the lines
        for t in [t for t in ax.texts if t.get_text().startswith('r = ')]:
            t.remove()
        dser = da[mcol].reindex(idx)
        ax.plot(dser, color=DA_COLOR, lw=0.9, label='PFM DA (same depths)')
        notes = []
        for name, m in (('PFM day 1', col(model[S], mcol)), ('PFM DA', dser)):
            if obs is not None and m is not None and (txt := stats_text(obs.reindex(idx), m.reindex(idx))):
                notes.append(name + ':\n  ' + txt.replace(', RMSE', '\n  RMSE'))
        ax.text(1.01, 1.0, '\n'.join(notes), transform=ax.transAxes, ha='left', va='top', fontsize=8,
                color='0.25', linespacing=1.5)
        ax.legend(loc='lower right', bbox_to_anchor=(1, 1), fontsize=7.5, frameon=False, ncol=3, borderaxespad=0.2)
    period = f'{a.start} to {end}'
    out = os.path.join(HERE, 'plots_web', f'SBOO_currents_DA_{a.start.replace("-", "")}_{end.replace("-", "")}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # plots_web keeps only the latest figure, as in the other QC folders
    for old in glob.glob(os.path.join(os.path.dirname(out), 'SBOO_currents_DA_*.png')):
        if old != out:
            os.remove(old)
    finish(fig, list(axes), idx, f'SBOO currents: observed vs PFM day 1 vs PFM DA ({res}), {period}', out,
           forecast=False)


if __name__ == '__main__':
    main()
