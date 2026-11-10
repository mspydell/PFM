"""Hourly shoreline dye_01 / dye_02 at a station from an LV4 his file.

Same method as ~/Python_files/PTB_TJRE_DyeFactor/update_station_dye.py, a Python port of the MATLAB
Shoreline_Dye_5days.m + calculate_shorelineS_nadb.m (matches its station_dye_day<N>.csv to float precision):
  - shoreline index SHOREI(j): first land cell from offshore in each eta row, with a fake land line
    across the bay mouth (pxy1 -> pxy2);
  - each hour, the shoreline cell is the most onshore water cell with depth (zeta + h) <= 1.2 m
    (else < 1.5 m, else < 3.5 m), ignoring xi < 90;
  - dye is the surface layer, masked by wetdry_mask_rho, plus 1e-8.
"""
import netCDF4 as nc
import numpy as np
import pandas as pd

HIS = '/dataSIO/PFM_Simulations/Archive/LV4_His/LV4_ocean_his_*.nc'
EPOCH = pd.Timestamp('1999-01-01')


def shore_index(mask):
    """MATLAB SHOREI (1-based xi of the first land cell) for every eta row."""
    mf = mask.copy()
    (x1, y1), (x2, y2) = (319, 629), (343, 559)
    for iy in range(y1, y2 - 1, -1):
        ix = int(np.ceil(x1 + (iy - y1) * (x1 - x2) / (y1 - y2)))
        mf[iy - 1, ix - 1] = 0
    out = np.zeros(mask.shape[0], int)
    for j in range(1, mask.shape[0] + 1):
        start = 0 if j <= 1035 else 119
        out[j - 1] = np.flatnonzero(mf[j - 1, start:] == 0).min() + start + 1
    return out


def shore_cell(depth_row, ishore):
    """1-based xi of the shoreline cell in one eta row (calculate_shorelineS_nadb), or None."""
    d0 = depth_row[:ishore - 1]
    with np.errstate(invalid='ignore'):
        for cand in (np.flatnonzero(d0 <= 1.2), np.flatnonzero(d0 < 1.5), np.flatnonzero(d0 < 3.5)):
            if len(cand):
                break
    cand = cand + 1
    cand = cand[cand >= 90]
    return int(cand.max()) if len(cand) else None


def read_file(path, records, shorei, stations):
    """{time: {'ptb_<station>': dye_01, 'tjre_<station>': dye_02}} for the given records of one his file.

    stations: {station: MATLAB 1-based eta row of its shoreline cell}.
    """
    out = {}
    with nc.Dataset(path) as d:
        nrec = len(d.dimensions['ocean_time'])
        t = d['ocean_time'][:]
        h = d['h'][:]
        for k in records:
            if k >= nrec:
                break
            row = {}
            for st, j in stations.items():
                wet = d['wetdry_mask_rho'][k, j - 1, :].filled(0).astype(float)
                wet[wet == 0] = np.nan
                depth = d['zeta'][k, j - 1, :].filled(np.nan) * wet + 1e-8 + h[j - 1, :]
                i = shore_cell(depth, shorei[j - 1])
                sid = st.replace('-', '')
                if i is None:
                    row[f'ptb_{sid}'] = row[f'tjre_{sid}'] = np.nan
                    continue
                for name, col in (('dye_01', f'ptb_{sid}'), ('dye_02', f'tjre_{sid}')):
                    v = float(d[name][k, -1, j - 1, i - 1]) * wet[i - 1] + 1e-8
                    row[col] = v if v >= 0 else np.nan
            out[(EPOCH + pd.Timedelta(seconds=float(t[k]))).round('h')] = row
    return out
