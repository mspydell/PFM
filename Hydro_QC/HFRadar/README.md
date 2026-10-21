# PFM surface currents vs HF radar

Compares PFM surface currents (LV3 and LV4) with the HF radar 6 km near-real-time product and the ADCP top layer at the moorings, every 6 hours. Times on the plots are Pacific (PDT/PST, America/Los_Angeles); the data and caches stay UTC.

| File | What it does |
|---|---|
| `plot_model_vs_hfr.py` | QC-website figures: one for LV3, one for LV4, into `figures/`. Also holds the shared plotting code. |
| `plot_model_vs_hfr_range.py` | The same figures for any start and end dates, one page per 3 days, into `plots/` (not for the website). |

The HF radar-only QC figures (radar maps and coverage) are made by `qc_obs_py_files/make_obs_qc_figures.py`, not here.

## Running

```bash
python plot_model_vs_hfr.py                     # LV3 and LV4, last panel = today 00:00 Pacific
python plot_model_vs_hfr.py --date 2026-10-18   # last panel = 18 Oct 00:00 Pacific
python plot_model_vs_hfr.py --level LV4
python plot_model_vs_hfr.py --update-moorings   # download new mooring obs first
python plot_model_vs_hfr.py --level LV4 --arrow-km 0.5   # test arrow spacing, saved as *_arrows<N>m.png
```

Writes `figures/Model_vs_HFR6km_<LV>_<YYYYMMDD>.png`. Older dates are kept. It must run on a machine that can read `/dataSIO`, and needs `../Moorings/common` (the mooring obs and model code).

Made for the QC website run at about 06:00 Pacific. By then the HF radar reaches about 03:00, so the last panel is 00:00 today.

## Date-range plots: plot_model_vs_hfr_range.py

```bash
python plot_model_vs_hfr_range.py --start 2026-10-01 --end 2026-10-20
python plot_model_vs_hfr_range.py --end today --duration 9          # the last 9 days, today included
python plot_model_vs_hfr_range.py --start 2026-10-10 --level LV3 --update-moorings
```

- Dates are Pacific days, both included. `--end` takes a date or `today` (the default); `--duration N` can't be combined with `--start`.
- One page per 3 days from `--start` on: `plots/Model_vs_HFR6km_<LV>_<first day>_<last day>.png`. The last page always has 3 rows, so it can run up to 2 days past `--end`.
- A page is the same figure as the website one ending at the next day's 00:00 (for example, the 18-20 Oct page is the 21 Oct website figure).

## Figure

3 rows x 4 columns, every 6 h in Pacific time: each row is one day at 06:00, 12:00, 18:00 and the next day's 00:00. The 21 Oct figure covers 18 Oct 06:00 to 21 Oct 00:00.

| Item | What is plotted |
|---|---|
| Model (black) | Day 1 of the PFM run for the panel's UTC date. If that run isn't written yet, day 2 of the previous run, marked in the panel title. Instantaneous at the hour. |
| Model depth | u, v averaged over the top 2 m below the moving free surface (each s-layer weighted by how much of it lies in 0-2 m), rotated to east/north. |
| Model grid | Linearly interpolated onto a regular ~100 m lon/lat grid lined up with the HF radar grid (1/60 of the 6 km spacing). Arrows every 3 km on LV3 and every 2 km on LV4, so every 6 km node is an HF radar cell centre. Cells touching land are left blank. |
| HF radar (red) | NDBC 6 km near-real-time product, hourly, as published. Read from the shared cache that `make_obs_qc_figures.py` fills (`/dataSIO/DA_Simulations/input_files/hfr_nrt`); hours missing there are read from NDBC. |
| Moorings (colours) | Observed ADCP top layer, hourly: SBOO on LV4; SBOO, PLOO and Del Mar on LV3. From the `../Moorings` cache. |
| Model at the moorings (thick black) | Averaged over the same depths as the ADCP top layer (SBOO 2.8-5.8 m, PLOO 2.9-5.9 m, Del Mar 4.4-10.4 m), from the same run and hour as the panel. |
| Panel title | At HF radar cells where both exist: number of cells, mean speed model / HF radar, RMS vector difference. |

All arrows share one scale per figure; the reference arrow is 0.25 m/s. Observed mooring arrows are drawn partly transparent so the model arrow beneath stays visible.

## Caches

In `/dataSIO/Hydro_QC/HFRadar/cache/`, shared by swell and backwash (group-writable; the environment variable `HYDRO_QC_DATA` overrides the `/dataSIO/Hydro_QC` root). A run never changes once written, so nothing in the cache goes out of date.

| File | Contents |
|---|---|
| `weights_<LV>_<key>.npz` | Interpolation weights, model grid to the ~100 m grid; rebuilt by itself if the grid file, radar grid or spacing changes |
| `model_<LV>_<run>_rec<NNN>_top2m_every<N>_<key>.npz` | Model arrows of one panel |
| `mooring_<LV>_<run>_<site>.npz` | Model at a mooring, days 1-2, where the Moorings cache doesn't have it |

A first run takes a few minutes; a run with everything cached takes about 15 s.
