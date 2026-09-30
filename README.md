# README for PFM

PFM is the code base for running the Pathogen Forecast Model (San Diego), a
collection of nested regional ocean simulations.

The code here handles all of the automatic running of the PFM. **This does not
make ROMS executables.**

---

## What PFM runs

Four nested grids, run in order, each taking boundary conditions from the one
above it:

| level | grid (eta × xi) | dt   | executable          | notes                        |
|-------|-----------------|------|---------------------|------------------------------|
| LV1   | 390 × 253       | 60 s | `LV3_romsM_INTEL`   | ROMS, 40 s-levels            |
| LV2   | 398 × 266       | 30 s | `LV3_romsM_INTEL`   | ROMS                         |
| LV3   | 413 × 251       | 15 s | `LV3_romsM_INTEL`   | ROMS                         |
| LV4   | 1141 × 486      | 2 s  | `coawstM_INTEL`     | COAWST: ROMS + SWAN via MCT, `WET_DRY`, `CURVGRID` |

Executables live in `/scratch/PFM_Simulations/executables/` and are built
elsewhere — `coawstM_INTEL` from `/home/mspydell/models/COAWST`.

The COAWST source in use from 2026-05-08 is tracked in a **private** repository,
<https://github.com/mspydell/coawst_new>, which has the build details and the
model-side history. Anything about how the executables themselves were
configured or changed lives there, not here; this repo only drives them. Access
is by request to MSS.

Default forecast length is 5 days, shortened automatically when the ocean
boundary data cannot cover it (see *Operational changes*, 2026-09-12).

## Why would you clone this repo?

To port this code to other computers and make developing easier. And to use the
python functions for creating the files ROMS needs — `atm_forcing.nc`,
`boundary_condition.nc`, etc.

## How a forecast runs

```
cron
 └─ run_forecast_LVs_v2.sh          (operational runner, F. Feddersen)
     └─ driver_run_forecast_LV1234.py
         └─ driver_functions.py  ->  per-level forcing, then ROMS under SLURM
```

Other entry points in `driver/` (`run_PFMv2.sh`, `run_PFM_LV1234.sh`,
`run_PFM_LV4_mv_webnc.sh`, …) run the same machinery for manual and partial
reruns. Each checks `EXPECTED_BRANCH` and exits if it cannot get onto it.

Working output goes to `/scratch/PFM_Simulations/LV?_Forecast/{Run,Forc,His}`;
archives to `/dataSIO/PFM_Simulations/Archive/`.

## Data sources

| input | source | notes |
|-------|--------|-------|
| atmosphere | `pipeline.ucsd.edu:/transfer` over SFTP (`A3D` files) | falls back to `/transfer/FALK`, then to ECMWF |
| atmosphere (fallback) | `ftp://diss.ecmwf.int/` (`T1D` files) | credentials in `PFM/.env`, not in git |
| ocean | HYCOM ESPC-D-V02, `tds.hycom.org` | cached in `/scratch/PFM_Simulations/hycom_data/` |
| rivers | National Water Model v3.1 (nomads, S3 fallback) | TJ uses persistence; see below |
| waves | CDIP | LV4 / SWAN boundary |

Credentials for all of these live in `PFM/.env`, which is git-ignored. Load them
with `set -a; source <PFM>/.env; set +a` before running anything by hand.

---

# Significant changes

**Dates below are deployment dates** — when a change reached the operational
clone `/home/ffeddersen/PFM_NEW` and could affect forecast output — **not commit
dates.** The two differ by days to weeks. A change affects the first nightly run
*after* the listed time.

Deployment dates come from that clone's reflog, which begins 2026-06-25. Earlier
changes can sometimes be dated from the archived output itself, which is better
evidence than either a commit or a reflog — `/dataSIO/PFM_Simulations/Archive/Forcing`
records the forcing source in the filename, so the GFS → ECMWF switch below is
dated from the files that were actually produced. Entries with neither are dated
as originally recorded and have not been re-verified.

## Model and forcing changes

These change what the model produces.

### Punta Bandera outfall — Q_PB & C_PB

| discharge | fraction raw WW | period |
|-----------|-----------------|--------|
| 2.2 m³/s  | 0.7             | t < 2025-04-04 |
| 2.5 m³/s  | 0.5088          | 2025-04-04 < t < 2025-05-02 |
| 2 m³/s    | 0.5             | 2025-05-02 < t < 2026-08-04 |
| 2 m³/s    | **0.39**        | t > 2026-08-04 |

`dye_PB` 0.5 → 0.39 in `pfm_operational_input_new.py`, committed directly in the
operational clone, so effective the same day.

### Tijuana River — Q_TJ

- t ≤ 2025-09-02 — NWM forecast.
- t ≥ 2025-09-02 — persistence method, and NWM if rain is detected in NWM Q_TJ.

### Tijuana River — C_TJ

- t < 2025-04-04 — 0.3 for Q_TJ < 1.83; 0.3·1.83/Q for Q_TJ > 1.83. Wastewater
  capped at 0.3·1.83 = 0.549.
- 2025-04-04 < t < 2025-12-16 — C0 = 0.65, Cf = 0.045, Qmx = 2.25.
- 2025-12-16 < t < 2026-01-14 — as above, Qww capped at 5 m³/s.
- t > 2026-01-14 — C0 = 0.3, Cf = 0.04, Qmx = 2.25, Qww capped at 5 m³/s.

### Other model changes

**2025-03-11 — atmospheric forcing switched from GFS to ECMWF.** The last
GFS-forced run is `2025-03-11 00Z` and ECMWF runs continuously from
`2025-03-11 06Z`, matching `5e353c9` ("FF got working full ecmwf stuff") the
same day. The two overlap through February and early March 2025 while ECMWF was
being brought up (`38002b5`, 2025-02-12), so archived forcing in that window is
a mix — check the filename prefix (`atm_gfs_*` vs `atm_ecmwf_*`) rather than
assuming. This is the single largest change to the surface forcing in the
record.

**2026-05-08 — tracer advection switched from MPDATA to HSIMT, on all four
levels at once.** New ROMS executables (`7184c4b`); applies to both temp and
salt, horizontal and vertical. LV1–LV3 and LV4 run different executables, so
both were replaced together. The COAWST side of that rebuild is documented in
the private <https://github.com/mspydell/coawst_new>.

The switch is sharp and dateable from the archived history files, which carry
the scheme in the `NLM_TADV` global attribute. Every level changes on the same
run:

| level | last MPDATA | first HSIMT |
|-------|-------------|-------------|
| LV1 | `LV1_ocean_his_202605080000.nc` | `LV1_ocean_his_202605080600.nc` |
| LV2 | `LV2_ocean_his_202605080000.nc` | `LV2_ocean_his_202605080600.nc` |
| LV3 | `LV3_ocean_his_202605080000.nc` | `LV3_ocean_his_202605080600.nc` |
| LV4 | `LV4_ocean_his_202605080000.nc` | `LV4_ocean_his_202605080600.nc` |

Everything before that, back through 2025 and earlier, is MPDATA. The scheme is
also printed in each run's `LV?_forecast.log` ("Tracer Advection Scheme") and set
by `Hadvection`/`Vadvection` in the `.in` files, so any run can be checked
directly rather than inferred from its date.

**t < 2025-09-09 — Q_TJ split across 4 of 5 cells.** Only 80% of the intended
Q_TJ reached the model. Because C_TJ was fixed, Q_WW_TJ was also 80% of intent.
Fixed after 2025-09-09.

**2026-06-09 to 2026-06-11 — `river_Vshape` flipped, then reverted.** The array
was flipped on the belief that flow was being put at the bottom rather than the
top. This was wrong and was reverted two days later (`ac746d8`): the original
`vshape_raw` is correct, giving u(z) = constant = Q/(h·dx) in each cell. **The
current code does not flip.** An earlier version of this README recorded only the
flip and was misleading for three months.

**2026-09-04 — `uvb` carried into the LV atm forcing files.** Surface downward UV
radiation, de-accumulated onto `srf_time` like the other fluxes. ROMS does not
read it; it is carried so the archived forcing has it. Only present when the
source grib has it.

**2026-09-23 — `ubar_north` boundary fix.** In
`ocnr_2_BCdict_1hrzeta_from_tmppkls`, the northern depth-averaged velocity was
computed from the *southern* boundary's velocity profile weighted by the northern
boundary's geometry. LV1's northern barotropic forcing changes from this date,
and every nested level inherits it. Magnitude is small (mean |ubar_north| ≈
0.0085 m/s) but it is a real change to the boundary condition.

### Winds — coordinate convention

Worth stating because the file metadata was wrong until 2026-09-23. ECMWF winds
arrive in earth (east/north) coordinates. `get_atm_data_on_roms_grid` **rotates
them into ROMS ξ/η** before writing, so `LV?_ATM_FORCING.nc` holds grid-direction
winds: `Uwind` is ξ, `Vwind` is η. ROMS does **not** rotate them again — LV1–3
are built without `CURVGRID`, and for LV4 the fields match the model grid
dimensions, so `set_data.F` treats them as already gridded and skips its
rotation. One rotation total.

## Operational changes

These affect whether and how the model runs, not what it produces.

- **2026-06-14** (in the operational clone by 2026-06-25) — ECMWF gribs fetched
  directly from ECMWF rather than via CDIP (`70f3c10`). Same product, different
  delivery path. Earlier, if files were missing at CDIP, PFM fell back to an
  older ECMWF forecast that might still be there (`b7401eb`, 2025-05-09).
- **2026-07-22** — atm interpolator rebuilt per time slice instead of reusing one
  `RegularGridInterpolator` (`aaf537a`). See *Near misses*.
- **2026-07-22 / 2026-07-29** — restart `ocean_time` snapped to the nearest 6
  hours; `sanitize_all_restart_files` added as a pre-flight to the forecast and
  hindcast drivers. Float32 drift in `ocean_time` was causing ROMS to reject
  otherwise-valid restarts.
- **2026-08-23** — NWM v3.0 deprecated; now v3.1 with version probing and an
  Amazon S3 fallback. Downloads parallelised (~333 s → ~30 s).
- **2026-08-31** — ECMWF flattened their dissemination layout from `/YYYYMMDD/`
  to the server root. PFM now probes both, validates GRIB magic bytes rather than
  just file size, and fails loudly instead of silently reusing a stale pickle.
- **2026-08-31** — LV4 atm steps run under `srun`. The login node's 8 GiB per-user
  cgroup cap was SIGKILLing them.
- **2026-09-01 → 2026-09-03** — atm data now comes from `pipeline.ucsd.edu` over
  SFTP, ECMWF as fallback. Searches `/transfer` then `/transfer/FALK`, at the
  requested cycle then 6 h earlier, and requires the whole cycle to be present
  before committing to it (a cycle takes ~2 h to land and `/transfer` is the
  active push target).
- **2026-09-02 / 2026-09-06** — daily observation QC figures (HF radar,
  SBOO/PLOO currents, CDIP waves over LV3) and the ddPCR wrappers.
- **2026-09-12** — `get_longest_forecast` fixed. A partial HYCOM forecast now
  correctly shortens the run instead of crashing, and a gap in the data is no
  longer silently reported as continuous coverage. Runs shorter than 3 days still
  abort.
- **2026-09-29** — default branch renamed `PHM_development` → `main`; `master`
  deleted. Runner scripts now verify the branch switch succeeded and exit if it
  did not, instead of continuing on whatever branch was checked out.

## Near misses

Changes that could have corrupted output but did not reach operations. Recorded
so that re-analysis of these periods is not thrown by the commit history.

**2026-06-25 → 2026-07-22 — frozen atm fields (development only).** The atm
interpolation loop reused a single `RegularGridInterpolator` and poked new data
into it via `F._values`. Silently ignored by scipy ≥ 1.14 — the values cached at
`__init__` are what `__call__` reads — which would have made every atm field a
copy of the first time slice. It was never deployed: the operational clone sat at
`b276ca9` for this entire window and its next pull was the fix itself. Archived
forcing from this period was checked and is correctly time-varying.

---

## Notes for anyone editing this file

- Date changes by when they reached `/home/ffeddersen/PFM_NEW`, not by commit
  date. A commit is not a deployment.
- Use ISO dates (`YYYY-MM-DD`). Earlier revisions mixed `2025-4-4`, `16 Dec` and
  `6/9/2026`, the last of which was only resolvable by digging through git.
- If a change is later reverted, edit the original entry. A timeline that records
  only the change and not the revert is worse than no timeline.
