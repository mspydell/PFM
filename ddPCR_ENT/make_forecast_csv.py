#!/usr/bin/env python3
"""Build/update an hourly CSV of day-1 and day-5 PFM forecasts of sites_dye_tot.

For each valid datetime (UTC):
  day1 = value from the run whose forecast hours 1-24   cover that time
  day5 = value from the run whose forecast hours 97-120 cover that time
If two runs cover the same time (e.g. an 06Z run followed by a 00Z run), the newer run wins.

Every web file is read each time (about 10 s for all of them). The output goes to the shared cache folder
$HYDRO_QC_DATA/Dye_ddPCRENT (default /dataSIO/PFM_Simulations/cache/Dye_ddPCRENT); only the newest
output file is kept there.
"""
import glob
import os
import sys

import netCDF4 as nc
import numpy as np
import pandas as pd

NC_DIR = "/dataSIO/PFM_Simulations/Archive/web"
OUT_DIR = os.path.join(os.environ.get("HYDRO_QC_DATA", "/dataSIO/PFM_Simulations/cache"), "Dye_ddPCRENT")
# output is named after the newest run it includes, e.g. sites_dye_tot_day1_day5_2026092900.csv
OUT_PREFIX = os.path.join(OUT_DIR, "sites_dye_tot_day1_day5")

VAR = "sites_dye_tot"
EPOCH = pd.Timestamp("1999-01-01")
DAY1_HOURS = range(1, 25)     # forecast hours 1-24
DAY5_HOURS = range(97, 121)   # forecast hours 97-120
# Sites matched by coordinates so older files (6 sites, different order) line up
SITES = {
    "PlayasTijuana":    (32.51998, -117.12480),
    "IB_pier":          (32.58008, -117.13338),
    "SilverStrand":     (32.62492, -117.13978),
    "Coronado_AvLunar": (32.67415, -117.17386),
}
TOL = 1e-3


def extract(path):
    """Return long DataFrame: run, lead_hr, valid, <site columns> for one file."""
    with nc.Dataset(path) as d:
        t = np.ma.filled(d["time"][:].astype(float), np.nan)
        data = np.ma.filled(d[VAR][:].astype(float), np.nan)
        lat = np.asarray(d["sites_lat"][:])
        lon = np.asarray(d["sites_lon"][:])
    valid = (EPOCH + pd.to_timedelta(t, "D")).round("h")
    df = pd.DataFrame({
        "run": valid[0],
        "lead_hr": np.arange(len(t)),
        "valid": valid,
    })
    for name, (la, lo) in SITES.items():
        idx = np.where((np.abs(lat - la) < TOL) & (np.abs(lon - lo) < TOL))[0]
        df[name] = data[:, idx[0]] if len(idx) else np.nan
    keep = set(DAY1_HOURS) | set(DAY5_HOURS)
    df = df[df["lead_hr"].isin(keep)]
    df.insert(0, "file", os.path.basename(path))
    return df


def main():
    os.umask(0o002)  # the cache is shared with the group
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = []
    for f in sorted(glob.glob(os.path.join(NC_DIR, "web_data_*.nc"))):
        try:
            runs.append(extract(f))
        except Exception as e:  # e.g. file still being written
            print("skip", os.path.basename(f), e, file=sys.stderr)
    cache = pd.concat(runs, ignore_index=True)
    print(f"read {len(runs)} web files")

    sites = list(SITES)
    cache = cache.sort_values("run")
    out = []
    for label, hours in (("day1", DAY1_HOURS), ("day5", DAY5_HOURS)):
        sub = cache[cache["lead_hr"].isin(hours)]
        sub = sub.drop_duplicates("valid", keep="last").set_index("valid")[sites]
        out.append(sub.add_suffix(f"_{label}"))
    res = pd.concat(out, axis=1, sort=True)
    res = res.reindex(pd.date_range(res.index.min(), res.index.max(), freq="h"))
    res.index.name = "datetime_utc"
    # order columns site by site: day1 then day5
    res = res[[f"{s}_{l}" for s in sites for l in ("day1", "day5")]]
    last_run = cache["run"].max()
    out_csv = f"{OUT_PREFIX}_{last_run:%Y%m%d%H}.csv"
    tmp = out_csv + ".tmp"
    res.to_csv(tmp, date_format="%Y-%m-%d %H:%M", float_format="%.6e")
    os.replace(tmp, out_csv)
    print("wrote", out_csv, res.shape)
    for old in glob.glob(f"{OUT_PREFIX}_??????????.csv"):  # keep only the newest
        if old != out_csv:
            os.remove(old)


if __name__ == "__main__":
    main()
