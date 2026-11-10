# modelQC_DA

QC of the PFM data-assimilation (DA) test runs against the operational Day 1 forecast and observations. Moved here from `~/Python_files/DA_testing` on 2026-11-09.

Website figures go to `plots_web/` (only the latest SBOO figure is kept; older ones are in `plots/`). The CSV caches live in `/dataSIO/PFM_Simulations/cache/DA` (see its README.md; `HYDRO_QC_DATA` overrides the parent folder).

Update by hand when needed, running these in order with `~/anaconda3/bin/python`:

1. `extract_da_dye.py` — DA shoreline dye at IB-079, IB-068, IB Pier (EH-030) -> `da_station_dye.csv`
2. `extract_da_sboo.py` — DA SBOO currents -> `da_sboo_currents.csv`
3. `plot_da_dye_ddpcr.py` — dye vs ddPCR time series and scatter
4. `plot_da_sboo_currents.py --update` — SBOO currents figure (refreshes SBOO obs first)

Dependencies: the mooring code `Moorings/common` and the dye vs ddPCR code (`plot_dye_ddpcr.py`, `station_dye.py`, the shoreline-cell method of `update_station_dye.py`), found by `qc_paths.py` next to this folder: `../ModelQC/Moorings/common` (or `../Hydro_QC/...`) and `../ddPCR_ENT` in the PFM repository, `../modelQC/Moorings/common` and `../modelQC/Dye_vs_ddPCRENT` in the home directory; and the `Dye_ddPCRENT` cache (ops dye + ddPCR, refreshed 07:01).

The DA his files land around 07:46 and the ops cache at 07:01, so run it after ~08:00 to include the current day's runs.
