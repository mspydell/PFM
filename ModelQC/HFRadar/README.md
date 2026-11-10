# PFM surface currents vs HF radar

Compares PFM surface currents (LV3 and LV4) with the HF radar near-real-time products (6, 2 and 1 km) and the ADCP top layer at the moorings, every 6 hours. Times on the plots are Pacific (PDT/PST, America/Los_Angeles); the data and caches stay UTC.

| File | What it does |
|---|---|
| `plot_model_vs_hfr.py` | QC-website figures, into `plots_web/`: HF radar 6, 2 and 1 km on LV3 and LV4. Also holds the shared plotting code. |
| `plot_model_vs_hfr_range.py` | The same figures for any start and end dates, one page per 3 days, into `plots/` (not for the website). |

The HF radar-only QC figures (radar maps and coverage) are made by `qc_obs_py_files/make_obs_qc_figures.py`, not here.

## Running

```bash
python plot_model_vs_hfr.py                     # all 6 figures, rows: day before yesterday, yesterday, today
python plot_model_vs_hfr.py --res 6             # only the 6 km figures (--res 6 2 1)
python plot_model_vs_hfr.py --date 2026-10-18   # rows 16, 17 and 18 Oct (last panel 19 Oct 00:00)
python plot_model_vs_hfr.py --level LV4
python plot_model_vs_hfr.py --update-moorings   # download new mooring obs first
python plot_model_vs_hfr.py --level LV4 --arrow-km 0.5   # test arrow spacing, saved as *_arrows<N>m.png
```

Writes `plots_web/Model_vs_HFR<res>km_<LV>_<YYYYMMDD>.png` (6, 2 and 1 km, each for LV3 and LV4), named by the last row's date. A normal run (without `--date`) deletes the older copies of each, so `plots_web/` only holds the newest; with `--date` older ones are kept. It must run on a machine that can read `/dataSIO`, and needs `../Moorings/common` (the mooring obs and model code).

Made for the QC website run at about 06:00 Pacific. By then the HF radar reaches about 03:00, so today's row is mostly the model forecast, marked "no HFR yet" in the panel titles.

## Date-range plots: plot_model_vs_hfr_range.py

```bash
python plot_model_vs_hfr_range.py --start 2026-10-01 --end 2026-10-20
python plot_model_vs_hfr_range.py --end today --duration 9          # the last 9 days, today included
python plot_model_vs_hfr_range.py --start 2026-10-10 --level LV3 --update-moorings
python plot_model_vs_hfr_range.py --start 2026-09-01 --res 2 1   # default --res 6
```

- Dates are Pacific days, both included. `--end` takes a date or `today` (the default); `--duration N` can't be combined with `--start`.
- One page per 3 days from `--start` on: `plots/Model_vs_HFR<res>km_<LV>_<first day>_<last day>.png`. The last page always has 3 rows, so it can run up to 2 days past `--end`.
- A page is the same figure as the website one for its last day (for example, the 18-20 Oct page is the 20 Oct website figure).

## Figure

3 rows x 4 columns, every 6 h in Pacific time: each row is one day at 06:00, 12:00, 18:00 and the next day's 00:00. The 21 Oct figure covers 19 Oct 06:00 to 22 Oct 00:00: the day before yesterday, yesterday and today.

| Item | What is plotted |
|---|---|
| Model (grey) | The newest PFM run started at or before the panel's time that reaches it within 48 h: normally day 1 of that UTC date's 00 UTC run; some days only have a 06 or 12 UTC run; if the day's run isn't written yet (or was cut short), day 2 of an earlier run, marked in the panel title. Instantaneous at the hour. |
| Model depth | u, v averaged over the top 2 m below the moving free surface (each s-layer weighted by how much of it lies in 0-2 m), rotated to east/north. |
| Model grid | Linearly interpolated onto a regular ~100 m lon/lat grid lined up with the HF radar grids (1/60 of the 6 km spacing; the 1, 2 and 6 km grids share nodes). Arrows: 6 km figures every 3 km on LV3 and every 2 km on LV4; 2 and 1 km figures at every radar cell. So each radar cell centre has a model arrow. Cells touching land are left blank. |
| HF radar (red) | 6, 2 or 1 km, hourly, as published. From the shared near-real-time cache that `make_obs_qc_figures.py` fills (`/dataSIO/DA_Simulations/input_files/hfr_nrt`, from Aug 2026); else, 6 km only, the monthly archive in `/dataSIO/HFRadar_files` (Nov 2024 - 18 Mar 2026; the 1 and 2 km archives there are daily means, not used); else live from NDBC (it keeps about the last 3 months). For 6 km nothing covers 18 Mar - 20 Jul 2026; 1 and 2 km start about 20 Jul 2026. Those panels show the model only. The LV3 1 km figure is twice the size of the others, so its 1 km arrows over the whole domain stay readable. |
| Moorings (colours) | Observed ADCP top layer, hourly: SBOO on LV4; SBOO, PLOO and Del Mar on LV3. From the `../Moorings` cache. |
| Panel title | At HF radar cells where both exist: number of cells, mean speed model / HF radar, RMS vector difference. |

All arrows share one scale per figure (the 95th percentile speed is drawn 1.35 model arrow spacings long) and have a small open head; the reference arrow is 0.25 m/s. Model arrows are thin and grey (darker on LV4), HF radar arrows bright red and thicker (both thinner on the denser 1 and 2 km figures), observed mooring arrows coloured by mooring and partly transparent. Isobaths: 10, 20 and 50 m. Colours and line widths are set at the top of `plot_model_vs_hfr.py`.

## Caches

In `/dataSIO/PFM_Simulations/cache/HFRadar/`, shared by swell and backwash (group-writable; the environment variable `HYDRO_QC_DATA` overrides the `/dataSIO/PFM_Simulations/cache` root). A run never changes once written, so nothing in the cache goes out of date.

| File | Contents |
|---|---|
| `weights_<LV>_<key>.npz` | Interpolation weights, model grid to the ~100 m grid; rebuilt by itself if the grid file, radar grid or spacing changes |
| `model_<LV>_<run>_rec<NNN>_top2m_every<N>_<key>.npz` | Model arrows of one panel |
| `hfr6km_<archive file>_<key>.npz` | An HF radar archive `.mat` file cut to the LV3 box, made once |

`<run>` is YYYYMMDD for a 00 UTC run, YYYYMMDDHH for one started later.

A first run takes a few minutes; a run with everything cached takes about 15 s.
