"""Python port of Dye_Volume.m.

Hourly PTB (dye_01) and TJRE (dye_02) dye volumes (m^3) integrated over the LV4 domain.
Vol1_*: from day-1 of each forecast (hours 1-24), Vol5_*: from day-5 (hours 97-120).
Volume = sum_xy( sum_z( dye * Hz ) * dx * dy ), dx = 1/pm, dy = 1/pn.
time1, time5 are hourly times (UTC); duplicate hours keep the latest forecast.

Note on array order: MATLAB's ncread returns dimensions reversed relative to
netCDF4-python. A MATLAB array of size [nx ny nz nt] is read here as
(nt, nz, ny, nx), i.e. (ocean_time, s_rho, eta_rho, xi_rho).
"""

import glob
import os
import re
from datetime import date, datetime

import numpy as np
from netCDF4 import Dataset


def _read(nc, name, index=slice(None)):
    """Read a variable like MATLAB's ncread: fill values become NaN."""
    return np.ma.filled(np.ma.asarray(nc.variables[name][index]).astype(float), np.nan)


def hourly_series(V, T):
    """Flatten (hour, file) arrays to a continuous hourly series; repeated hours keep the latest forecast."""
    V = V.ravel(order='F')
    T = T.ravel(order='F')
    ok = ~np.isnat(T)
    V, T = V[ok], T[ok]
    t_full = np.arange(T.min(), T.max() + np.timedelta64(1, 'h'), np.timedelta64(1, 'h'))
    vol = np.full(t_full.shape, np.nan)
    idx = ((T - t_full[0]) // np.timedelta64(1, 'h')).astype(int)
    vol[idx] = V  # later files come later in V, so they overwrite earlier ones
    return vol, t_full


HIS_FILES = '/dataSIO/PFM_Simulations/Archive/LV4_His/LV4_ocean_his_*.nc'
TIMEREF0 = np.datetime64('1999-01-01T00:00:00', 's')


RECORDS_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'his_records_cache.json')


def forecast_files():
    """All LV4 forecasts: list of (path, start time, number of hourly records).
    Record counts are cached in his_records_cache.json (by file size and modification time),
    so only new or changed his files are opened."""
    import json
    try:
        with open(RECORDS_CACHE) as fh:
            cache = json.load(fh)
    except (OSError, ValueError):
        cache = {}
    out, changed = [], False
    for path in sorted(glob.glob(HIS_FILES)):
        m = re.fullmatch(r'LV4_ocean_his_(\d{12})\.nc', os.path.basename(path))
        if not m:
            continue
        st = os.stat(path)
        key = f'{st.st_size}:{int(st.st_mtime)}'
        if cache.get(path, [None])[0] != key:
            with Dataset(path) as nc:
                cache[path] = [key, len(nc.dimensions['ocean_time'])]
            changed = True
        out.append((path, np.datetime64(datetime.strptime(m.group(1), '%Y%m%d%H%M'), 's'), cache[path][1]))
    if changed:
        with open(RECORDS_CACHE, 'w') as fh:
            json.dump(cache, fh)
    return out


def gap_plan(times, missing, files):
    """For each missing hour, the latest forecast that covers it (any lead up to its last record).
    times: hourly datetime64; missing: bool array; files: from forecast_files().
    Returns {path: [(record index, time), ...]} and the hours no forecast covers."""
    plan, uncovered = {}, []
    starts = np.array([f[1] for f in files])
    for t in np.asarray(times, dtype='datetime64[s]')[np.asarray(missing)]:
        for i in np.flatnonzero(starts <= t)[::-1]:
            path, t0, n = files[i]
            k = int((t - t0) // np.timedelta64(1, 'h'))
            if k < n:
                plan.setdefault(path, []).append((k, t))
                break
        else:
            uncovered.append(t)
    return plan, uncovered


def file_volume(hisfile, records, wet_only=True):
    """PTB and TJRE sewage volume (m^3) at the given record indices of one his file: array (n, 2).
    Same integral as Dye_Volume: sum over wet cells of dye * Hz * dx * dy
    (wet_only=False: over all water cells, including cells flagged dry)."""
    V = np.full((len(records), 2), np.nan)
    with Dataset(hisfile) as nc:
        mask_rho = _read(nc, 'mask_rho')
        mask_rho[mask_rho == 0] = np.nan
        h = _read(nc, 'h')
        dA = 1.0 / (_read(nc, 'pm') * _read(nc, 'pn'))
        hc = float(nc.variables['hc'][:])
        S_w = (hc * _read(nc, 's_w')[:, None, None] + h * _read(nc, 'Cs_w')[:, None, None]) / (hc + h)
        nz = S_w.shape[0] - 1
        for n, k in enumerate(records):
            wet = _read(nc, 'wetdry_mask_rho', (k, slice(None), slice(None))) if wet_only else np.ones_like(h)
            wet[wet == 0] = np.nan
            zeta = _read(nc, 'zeta', (k, slice(None), slice(None)))
            Hz = np.diff(zeta + (zeta + h) * S_w, axis=0)
            for idye, dye_name in enumerate(('dye_01', 'dye_02')):
                dye = _read(nc, dye_name, (k, slice(0, nz), slice(None), slice(None)))
                V[n, idye] = np.nansum(np.sum(dye * Hz, axis=0) * wet * mask_rho * dA)
    return V


def Dye_Volume(start_date, end_date):
    if isinstance(start_date, date) and not isinstance(start_date, datetime):
        start_date = datetime.combine(start_date, datetime.min.time())
    if isinstance(end_date, date) and not isinstance(end_date, datetime):
        end_date = datetime.combine(end_date, datetime.min.time())

    # List and filter .nc files
    base_hisfile = '/dataSIO/PFM_Simulations/Archive/LV4_His/LV4_ocean_his_*.nc'
    D_all = sorted(glob.glob(base_hisfile))

    valid_files = []
    for path in D_all:
        m = re.fullmatch(r'LV4_ocean_his_(\d{8})\d{4}\.nc', os.path.basename(path))
        if m:
            file_date = datetime.strptime(m.group(1), '%Y%m%d')
            if start_date <= file_date <= end_date:
                valid_files.append(path)

    if not valid_files:
        raise FileNotFoundError(
            f'No LV4 his files between {start_date:%Y-%m-%d} and {end_date:%Y-%m-%d}')

    # grid and vertical coordinate (same for all files)
    with Dataset(valid_files[0]) as nc:
        mask_rho = _read(nc, 'mask_rho')
        h = _read(nc, 'h')
        dA = 1.0 / (_read(nc, 'pm') * _read(nc, 'pn'))  # dx*dy (m^2)
        hc = float(nc.variables['hc'][:])
        s_w = _read(nc, 's_w')[:, None, None]
        Cs_w = _read(nc, 'Cs_w')[:, None, None]
        if int(nc.variables['Vtransform'][:]) != 2:
            raise NotImplementedError('Only Vtransform = 2 is implemented')
    S_w = (hc * s_w + h * Cs_w) / (hc + h)  # (nz+1, ny, nx)
    nz = S_w.shape[0] - 1
    mask_rho[mask_rho == 0] = np.nan

    num_hisfile = len(valid_files)
    timeref0 = np.datetime64('1999-01-01T00:00:00', 's')
    day_hours = [range(0, 24), range(96, 120)]  # day 1 and day 5 (0-based)
    nh = 24
    dye_names = ['dye_01', 'dye_02']  # PTB, TJRE
    VOL = np.full((nh, num_hisfile, 2, 2), np.nan)  # hour * file * day(1,5) * dye(PTB,TJRE)
    TIME = np.full((nh, num_hisfile, 2), np.datetime64('NaT'), dtype='datetime64[s]')

    for i, hisfile in enumerate(valid_files):
        print(f'Processing {hisfile}')
        with Dataset(hisfile) as nc:
            ocean_time = np.asarray(nc.variables['ocean_time'][:], dtype=float)

            # round to nearest full hour
            hours = np.floor(ocean_time / 3600 + 0.5).astype('int64')
            time_dnum = timeref0 + hours * np.timedelta64(3600, 's')

            for iday in range(2):
                for ih, itime in enumerate(day_hours[iday]):
                    if itime >= len(ocean_time):
                        continue
                    print(f'Doing i={i + 1},  itime = {itime + 1},  date = {time_dnum[itime]}')

                    # load wetdry mask
                    masknan_rho = _read(nc, 'wetdry_mask_rho', (itime, slice(None), slice(None)))
                    masknan_rho[masknan_rho == 0] = np.nan

                    # layer thickness Hz from z_w (Vtransform = 2)
                    zeta = _read(nc, 'zeta', (itime, slice(None), slice(None)))
                    z_w = zeta + (zeta + h) * S_w
                    Hz = np.diff(z_w, axis=0)

                    for idye, dye_name in enumerate(dye_names):
                        dye = _read(nc, dye_name, (itime, slice(0, nz), slice(None), slice(None)))

                        # weighted integration in z, then in x and y
                        dye_col = np.sum(dye * Hz, axis=0)  # m
                        dye_col = dye_col * masknan_rho * mask_rho
                        VOL[ih, i, iday, idye] = np.nansum(dye_col * dA)
                    TIME[ih, i, iday] = time_dnum[itime]

    # hourly series; for repeated hours keep the latest forecast
    Vol1_ptb, time1 = hourly_series(VOL[:, :, 0, 0], TIME[:, :, 0])
    Vol5_ptb, time5 = hourly_series(VOL[:, :, 1, 0], TIME[:, :, 1])
    Vol1_tjre, _ = hourly_series(VOL[:, :, 0, 1], TIME[:, :, 0])
    Vol5_tjre, _ = hourly_series(VOL[:, :, 1, 1], TIME[:, :, 1])
    return Vol1_ptb, Vol5_ptb, Vol1_tjre, Vol5_tjre, time1, time5


if __name__ == '__main__':
    import sys

    start = datetime.strptime(sys.argv[1], '%Y-%m-%d')
    end = datetime.strptime(sys.argv[2], '%Y-%m-%d')
    Vol1_ptb, Vol5_ptb, Vol1_tjre, Vol5_tjre, time1, time5 = Dye_Volume(start, end)
    print('             time      PTB (m^3)   TJRE (m^3)')
    for t, vp, vt in zip(time1, Vol1_ptb, Vol1_tjre):
        print(f'Day1 {t}  {vp:10.6g}  {vt:10.6g}')
    for t, vp, vt in zip(time5, Vol5_ptb, Vol5_tjre):
        print(f'Day5 {t}  {vp:10.6g}  {vt:10.6g}')
