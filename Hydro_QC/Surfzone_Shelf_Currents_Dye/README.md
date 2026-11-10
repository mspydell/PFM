# Surfzone and shelf currents: PFM LV4 model

Hourly depth-averaged surfzone and shelf (10, 12 and 15 m isobaths) currents, and the near-surface (top 3 m) current at the 12 m point, of the PFM LV4 model at three stations: Coronado Avenida Lunar, the Silver Strand guard shack and the Imperial Beach Pier (south). Model only, no observations yet.

Python port of `~/Matlab_files/SurfZone_transect_forOneLocation.m` (with `ShorelineOrientation.m`), PFM runs only.

| File | What it does |
|---|---|
| `plot_surfzone_shelf.py` | QC-website plots: the past 3 weeks, into `plots_web/` (with the alongshore dye figure and the two maps). Also holds the shared code (transects, model, cache). `--map` draws the transects map. |
| `plot_dye_alongshore.py` | Alongshore dye Hovmoller (shoreline, 10 and 15 m isobaths) for any dates into `plots/`; `--map` for its map. |
| `plot_surfzone_shelf_range.py` | The same plot for any start and end dates, into `plots/` (not for the website). |
| `plots/transects_map.png` | Map of the three transects and the 10/12/15 m shelf points, with their directions from east and distances offshore. |

## Method

| Step | How |
|---|---|
| Stations | Positions of the MATLAB code: `COAvenidaLunar` (32.674066, −117.1723), `SSGuardShack` (32.6249, −117.1398), `IBPierSouth` (32.5787, −117.133). |
| Coastline orientation | Straight line fitted to the coastline (wet cells next to land, the mask_rho = 0.5 contour) within 220 m of the station, as `ShorelineOrientation.m` (0.002°). Fitted by total least squares, since the coast runs nearly north–south and a fit of latitude on longitude is ill-conditioned there. |
| Transect | Normal to that line, from 100 m onshore of it to past the deepest shelf isobath (15 m), a point every 7 m. Offshore is the side with deeper water, so the direction is the same kind at every station. |
| Currents | ROMS `ubar`, `vbar` (the depth average; identical to the dz-weighted mean of `u`, `v`), dry faces removed with the wet/dry masks, averaged to rho points, rotated to east/north with the grid `angle`, then to **alongshore (+ up-coast, northward)** and **cross-shore (+ onshore)**. |
| Values on the transect | Inverse-distance-squared weighted mean of the 4 nearest wet rho cells (as the MATLAB code). |
| Shoreline | Each hour, the last wet transect point going shoreward from offshore, so it moves with the tide. |
| Surfzone | From the shoreline to the H<sub>s</sub> maximum (the breaking point), found as described in "Finding the H<sub>s</sub> maximum" below. The MATLAB code went to 1.25 × that distance. Surfzone currents are the mean over those points. |
| Shelf | The points of the same transect where the grid depth h reaches 10, 12 and 15 m (`SHELF_DEPTHS`; static h, not h + ζ). |
| Waves | At each shelf point: H<sub>s</sub> (`Hwave`, inverse-distance weighted as the currents), and the peak period T<sub>p</sub> (`Pwave_top`) and mean wave direction D<sub>m</sub> (`Dwave`, degrees from N, the direction the waves come from; not the peak direction) at the nearest grid cell, as `WaveData_timeseries.m` (periods come in discrete bins, and directions do not average linearly). |
| Near-surface current | At the 12 m point only (`TOP_DEPTH`): the 3-D `u`, `v` averaged over the top 3 m of the water column (`TOP`), each layer weighted by its thickness inside the top 3 m (ROMS Vtransform 2 levels, with ζ). LV4 has 10 layers, so that is about 2–3 layers. Only the upper 5 layers are read: the top 3 m is always inside them, and they are one stored chunk of the files. Then rotated and put on the point as the depth-averaged currents. |

### Finding the H<sub>s</sub> maximum (surfzone edge)

Waves shoal (H<sub>s</sub> grows) as they come into shallower water, and then break and lose height towards the shore. So along the transect, H<sub>s</sub> rises going shoreward to a peak at the breaking point and drops inside the surfzone. The peak marks the outer edge of the surfzone. It is found every hour, in `compute()` of `plot_surfzone_shelf.py`, as in `SurfZone_transect_forOneLocation.m`:

1. **H<sub>s</sub> on the transect.** Model `Hwave` on the wet cells (`wetdry_mask_rho`), put on the transect points (every 7 m) as the inverse-distance-squared weighted mean of the 4 nearest wet rho cells.
2. **Start at the shoreline.** The shoreline is the last wet point going shoreward from offshore (it moves with the tide). Only the points from the shoreline outward are searched.
3. **First local maximum going offshore.** Walking offshore from the shoreline, the edge is the first point k whose H<sub>s</sub> is higher than both the point two before it and the point two after it:

   H<sub>s</sub>(k) > H<sub>s</sub>(k − 2) and H<sub>s</sub>(k) > H<sub>s</sub>(k + 2), points 7 m apart (so ±14 m).

   Comparing with points 2 away rather than the next point keeps tiny wiggles of the interpolated profile from counting as a peak. The search starts at the third point from the shoreline, so that there are two points on each side.
4. **Fallback.** If there is no such point (H<sub>s</sub> only falls or only rises along the transect), the edge is the point with the largest H<sub>s</sub>.
5. **Results.** `sz_hs` = H<sub>s</sub> at the edge, `sz_width` = distance from the shoreline to the edge, `sz_depth` = water depth h + ζ there, and `s_edge` (in the `_hs.npz` profiles) = its distance offshore of the fitted coastline. The surfzone currents `sz_along` and `sz_cross` are the mean of the depth-averaged currents over the points from the shoreline to the edge, included.

Things to know when reading the results:

- It is the **first** maximum from shore, not the largest on the transect. With a bar or a terrace there can be two peaks; the inner one is used, and when the two are about the same height the edge can jump between them from hour to hour (Avenida Lunar: the grid has a flat terrace, h ≈ 2.8–3.1 m, from ~105 m to ~175 m offshore; H<sub>s</sub> is almost constant across it, within a few mm, so the first maximum is at its inner end (~105 m) or its outer end (~175 m) depending on mm differences, and the edge flips between the two). The surfzone currents then average over different widths.
- The edge moves in steps of the 7 m transect spacing, but the model grid is ~28 m, so the profile between grid cells is interpolated; widths are good to about a grid cell.
- With large waves, breaking starts farther out and the edge moves offshore (IB Pier went to ~500 m in the 11 Oct 2026 swell). The `_hs_0-700m` and `_hs_to15m` figures show the H<sub>s</sub> profile with the chosen edge (orange line), to check it.

Transect directions, counter-clockwise from east (0° = E, 90° = N):

| Station | Offshore | Onshore (+ cross-shore) | + Alongshore | 10 m point | 12 m point | 15 m point |
|---|---|---|---|---|---|---|
| IBPierSouth | −177.0° | 3.0° | 93.0° | 694 m | 1342 m | 2290 m |
| SSGuardShack | −163.0° | 17.0° | 107.0° | 366 m | 608 m | 1377 m |
| COAvenidaLunar | −134.6° | 45.4° | 135.4° | 602 m | 1143 m | 2919 m |

(Shelf points: distance offshore of the fitted coastline.)

Stations are plotted south to north: Imperial Beach Pier, the guard shack, Avenida Lunar. Each model hour comes from the latest history file that covers it (files run 2.5–5 days ahead); hours no file covers stay empty. The lines continue past the plot window to the end of the latest file, **dashed** after now ("Future dates" in the legend; every value is model output, the dashed part is just still ahead), with a dotted vertical line at now. Titles give the plot window and how far the latest run reaches.

Times on the plots are Pacific (PDT/PST); the data are read and cached in UTC.

## What is plotted

| Panel | Lines |
|---|---|
| Top | H<sub>s</sub> at the surfzone edge, one line per station |
| Two per station | Alongshore, then cross-shore current: depth-averaged surfzone mean (orange) and shelf at 10, 12 and 15 m (light to dark blue), and the top 3 m at the 12 m point (magenta) |

Each run also saves 16 more figures next to it:

| Figure | What it shows |
|---|---|
| `<name>_alongshore.png` | Alongshore currents only: one panel per station (same y axis), depth-averaged surfzone mean (orange) and 12 m shelf point (blue; `ALONG_DEPTH` in the script), and the top 3 m at the 12 m point (magenta). |
| `<name>_hs_0-700m.png` | H<sub>s</sub> along the transect, time × offshore distance (0–700 m, `HS_NEAR`), one panel per station, one colour scale. Lines: shoreline (white), surfzone edge (orange), shelf points inside the range (dashed). |
| `<name>_hs_to15m.png` | The same out to just past the 15 m point (1.4–2.9 km). |
| `<name>_dye_0-700m.png`, `<name>_dye_to15m.png` | The same two ranges for the total dye, dye_01 (PTB) + dye_02 (TJRE), in the **surface layer** (top σ-level; the surfzone is well mixed, so nearly the depth mean there), inverse-distance weighted as H<sub>s</sub>. Same colours as `Matlab_files/Hovmoller_plot_dye.m`: cmocean `matter`, log scale fixed at 10<sup>−5.5</sup>–10<sup>−1.5</sup> (unitless: dye is the fraction of PTB or TJRE water; the "kilogram meter-3" units in the history files do not apply) (`DYE_RANGE`), in 4 discrete colour blocks per power of ten (`DYE_STEPS`, steps of 10<sup>0.25</sup>); lower values take the lightest colour. The shoreline and shelf lines are black and the surfzone edge blue on these. |
| `<name>_tempcolumn_<station>.png`, `<name>_saltcolumn_<station>.png` | The same 7 panels for temperature (cmocean `thermal`) and salinity (cmocean `haline`): one colour scale shared by the 3 stations (`COLUMN_SCALE`): temperature 5 °C wide in 0.25 °C steps, centred on the median of all stations (rounded), widened to the 1st–99th percentile when the data spans more (e.g. the long-range figures); salinity 1st–99th percentile in about 20 round steps. |
| `<name>_dyerho_10-15m.png` | **QC-website figure**, 3 × 4 panels: one row per station (IB Pier, guard shack, Avenida Lunar), columns dye and density σ<sub>0</sub> over the water column at the 10 m isobath, then dye and σ<sub>0</sub> at the 15 m isobath (`GRID_FIGS`, `GRID_DEPTHS`). Same panels as the column figures; one dye scale (stepped `matter`) and one σ<sub>0</sub> scale (1st–99th percentile of the 10 and 15 m points of all stations), horizontal colour bars at the bottom. |
| `<name>_salttemp_10-15m.png` | **QC-website figure**, the same 3 × 4 layout with salinity in place of dye and temperature in place of density (`GRID_FIGS`); salinity and temperature scales both 1st–99th percentile of the 10 and 15 m points of all stations. |
| `<name>_rhocolumn_<station>.png` | The same 7 panels for potential density anomaly σ<sub>0</sub> = ρ(S<sub>A</sub>, Θ, p = 0) − 1000 kg m<sup>−3</sup> (cmocean `dense`), from the model's temperature (potential), salinity (practical) and layer depth with the Gibbs SeaWater toolbox (`gsw`, TEOS-10: `p_from_z`, `SA_from_SP`, `CT_from_pt`, `sigma0`); computed at plot time from the cached temp/salt/z, nothing extra cached. Colour scale 1st–99th percentile, shared by the 3 stations. |
| `<name>_dyecolumn_<station>.png` (one per station) | Dye, dye_01 (PTB) + dye_02 (TJRE), over the water column (time × depth z, all 10 layers, moving with ζ), 7 panels: at the surfzone edge (first H<sub>s</sub> maximum; each hour at the grid cell nearest that hour's edge, so the bottom moves with the edge depth), and at the 2, 5 and 8 m (`COLUMN_DEPTHS`, used only here; 1 m often dries), 10, 12 and 15 m points, at the nearest grid cell. Offshore distances of the 2/5/8 m points: IB 111/243/423 m, guard shack 119/205/272 m, Avenida 82/254/383 m. Same stepped `matter` colour scale as the other dye figures. |
| `<name>_waves.png` | Waves at the 15 m point (`WAVE_DEPTH`), one line per station, 3 panels: H<sub>s</sub>, T<sub>p</sub>, mean direction D<sub>m</sub>. T<sub>p</sub> moves in steps (frequency bins). D<sub>m</sub> differs between the stations because the waves refract towards each beach's shore normal (onshore-normal from-directions: IB 267°, guard shack 253°, Avenida 225°). |
| `<name>_surfzone.png` | The surfzone edge, one line per station, 4 panels: H<sub>s</sub> at the edge (`sz_hs`), surfzone width from the shoreline (`sz_width`), water depth h + ζ at the edge (`sz_depth`) and the moving shoreline (last wet point) relative to the fixed, fitted coastline, + offshore (`s_sl`). The shoreline comes from the wet/dry mask of the nearest ~28 m grid cell, so it moves in steps. |

## plot_surfzone_shelf.py: plot for the QC website

```bash
python plot_surfzone_shelf.py                      # past 3 weeks to the end of today (Pacific) -> plots_web/
python plot_surfzone_shelf.py --date 2026-09-30    # 3 weeks ending on another day (older plots are kept)
python plot_surfzone_shelf.py --map                # transects map -> plots/transects_map.png
```

A normal run keeps only the website figures in `plots_web/`: 5 from `WEB_VARIANTS`, `SurfzoneShelf_<YYYYMMDD>_3weeks_alongshore`, `_dye_to15m`, `_dyerho_10-15m`, `_waves` and `_surfzone`; plus the alongshore dye Hovmoller of the same 3 weeks, `DyeAlongshore_surf_<YYYYMMDD>_3weeks.png` (`plot_alongshore()` of `plot_dye_alongshore.py`), and two static maps, `DyeAlongshore_map.png` and `transects_map.png` (drawn only when missing; delete them to redraw). The other figures it draws are deleted, and so are older dates (with `--date`, only that date's non-website figures are deleted, and older DyeAlongshore figures are kept). The range script draws the same 5 `WEB_VARIANTS` figures into `plots/`. The other figures (the full figure, `_dye_0-700m`, `_hs_0-700m`, `_hs_to15m`, `_salttemp_10-15m`, the 7-panel `_<field>column_<station>` figures) are still in the code: `plot(..., only=None)` draws them all.

## plot_surfzone_shelf_range.py: any start and end dates

```bash
python plot_surfzone_shelf_range.py --start YYYY-MM-DD [--end YYYY-MM-DD|today] [--workers N] [--out PATH]
python plot_surfzone_shelf_range.py --duration N [--end YYYY-MM-DD|today] [--workers N] [--out PATH]
```

Both dates are Pacific days and included. `--workers` is the number of history files read in parallel (default 4).

## Cache

The cache is on the shared disk, `/dataSIO/PFM_Simulations/cache/Surfzone_ShelfCurrents/`, so runs on swell and backwash use and update the same files (group `feddersen-lab`, group-writable). Set the environment variable `HYDRO_QC_DATA` to use another root.

| Folder | Contents | Refreshed |
|---|---|---|
| `model/` | Hourly results of each history file and station, `<station>_LV4_<YYYYMMDDHHMM>_<hash>.csv`: time (UTC), `sz_along`, `sz_cross` (m/s), `sz_hs` (H<sub>s</sub> at the surfzone edge, m), `sz_width` (shoreline to the edge, m), `sz_depth` (h + ζ at the edge, m), `shelf<d>_along`, `shelf<d>_cross` (m/s), `shelf<d>_hs` (m), `shelf<d>_tp` (s), `shelf<d>_dp` (mean wave direction `Dwave`, ° from N, coming from; named dp for history) for d = 10, 12, 15, and `top12_along`, `top12_cross` (m/s, top 3 m at the 12 m point). Next to it, `..._hs.npz`: the hourly H<sub>s</sub> profile along the transect (every 7 m, to 100 m past the 15 m point, nan onshore of the shoreline; float16, ~40 kB) and the same for the surface dye (`dye`, float32), and the offshore distances of the shoreline (`s_sl`) and surfzone edge (`s_edge`). And `..._column4.npz`: dye_01 + dye_02, temperature and salinity in every layer (`dye2`, `temp2`, `salt2`, ... `dye15`, `temp15`, `salt15`, and `dyesz`, `tempsz`, `saltsz` at the surfzone edge; time × 10) and the layer depths (`z2`, ..., `zsz`, m, + up). It is checked and computed on its own, so it was added without rebuilding the rest; it reads the full 3-D dye, so it is the slowest part (~10 min per history file and station). | Computed once per history file; recomputed if the history file is newer. `<hash>` changes with the station position or the transect settings, so changing those rebuilds the cache. |

Reading one history file takes about 1 min the first time (the files are ~30 GB; only a small box around each transect is read). Delete a file to rebuild it.

## Requirements

Python 3 (base anaconda on backwash) with `pandas`, `matplotlib`, `numpy`, `scipy`, `netCDF4`, `cmocean` (colour maps) and `gsw` (Gibbs SeaWater, TEOS-10: the density figures).
It needs read access to `/dataSIO/PFM_Simulations/Archive/LV4_His` and `/dataSIO/PFM_Simulations/Grid`.
