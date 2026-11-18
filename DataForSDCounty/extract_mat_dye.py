"""One-time: hourly total shoreline dye at the 3 beaches from the MATLAB Shoreline_Dye_5days output, for the
forecasts before the PFM website files have the site (2025-04-12; Avenida Lunar only from 2025-06-23, the
website's Coronado site was 32.6778N before), so make_county_csv.py can cover Jan 2025 on.

Total dye = DSHORE_PTB + DSHORE_TJRE at the shoreline points (index into V.LAT) nearest the website sites;
against the website's sites_dye_tot (Jul-Dec 2025) log r = 0.999, median ratio 1.00.

Output: mat_dye_hourly.csv (run, valid_utc, <beach>), runs from FIRST to LAST.
Needs scipy: run with /home/akg004/anaconda3/bin/python3 extract_mat_dye.py
"""

import os

import numpy as np
import pandas as pd
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
MAT = '/home/akg004/Matlab_files/Dye_LV4/DSHORE_PTB_TJRE_05-Dec-2024_31-Mar-2026_model.mat'
FIRST, LAST = pd.Timestamp('2024-12-31'), pd.Timestamp('2025-06-23')
POINTS = {'Avenida_Lunar': 208, 'Silver_Strand': 177, 'Imperial_Beach_Pier': 146}


def main():
    V = sio.loadmat(MAT, squeeze_me=True, struct_as_record=False, variable_names=['V'])['V']
    tot = V.DSHORE_PTB + V.DSHORE_TJRE
    rows = []
    for f in range(V.time_dnum.shape[1]):
        t = pd.to_datetime(V.time_dnum[:, f] - 719529, unit='D').round('h')   # MATLAB datenum -> UTC
        if pd.isna(t[0]) or not FIRST <= t[0] < LAST:
            continue
        for k in np.flatnonzero(~pd.isna(t)):
            rows.append({'run': t[0], 'valid_utc': t[k], **{b: tot[i, k, f] for b, i in POINTS.items()}})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, 'mat_dye_hourly.csv'), index=False, date_format='%Y-%m-%d %H:%M',
               float_format='%.6e')
    print(f'mat_dye_hourly.csv: {out.run.nunique()} runs, {out.run.min()} to {out.run.max()}')


if __name__ == '__main__':
    main()
