# DA_files provenance

The DA science is developed in **`roms_fourdvar`**; this directory is a *copy*,
so PFM is self-contained and the nightly job has as few moving parts as possible.
When you change the science, change it there and re-copy, updating the commit
below.

Source repo: `/home/mspydell/research/sdtjreL3/v_intel/roms_fourdvar`
Copied at commit: **`11efb73`** (2026-10-06)

## Copied verbatim

| file | from (relative to roms_fourdvar) |
|---|---|
| `python_files/make_child_increment.py` | `DA_experiment/LV2/python_files/` |
| `python_files/oi_interp_increment.py` | `DA_experiment/LV2/python_files/` |
| `python_files/apply_child_increment.py` | `DA_experiment/LV2/python_files/` |
| `python_files/parent_history_to_BC.py` | `DA_experiment/LV2/python_files/` |
| `python_files/his_to_ic.py` | `DA_experiment/python_files/` |
| `python_files/download_hfr_range.py` | `DA_experiment/python_files/` |
| `python_files/DA_setup_python_funs.py` | `testing/` |
| `masks/*.nc` (3 files) | `DA_experiment/masks/` — see *Rebuilding the masks* |
| `run_in/LV1_run_4dvar.in`, `run_in/s4dvar.in` | `DA_experiment/run_in/` |
| `run_in/LV{2,3}_oi_seed.in` | `/scratch/matt/DA_Simulations/LV{2,3}/Run/` |
| `slurm/slurm_4dvar.sb`, `slurm/slurm_make_increment.sb` | `DA_experiment/scripts/` |
| `slurm/slurm_LV{2,3,4}_fwd.sb` | `/scratch/matt/DA_Simulations/LV{2,3,4}/Run/` |

## Copied WITH changes

All three were modified only to stop PFM reaching back into the `roms_fourdvar`
tree at run time. Re-apply if you re-copy.

**`python_files/make_obs_from_hfr.py`** — the absolute
`sys.path.insert(0, ".../roms_fourdvar/testing")` before
`from DA_setup_python_funs import _helmholtz_div` is replaced by a
`Path(__file__).parent` insert, and `DA_setup_python_funs.py` is vendored
alongside.

**`python_files/plot_increment_smoothing.py`** — `OUTDIR` pointed at
`roms_fourdvar/DA_experiment/figures/`; now
`/scratch/PFM_Simulations/DA_Sim/Log/figures`. Only used for figure output;
`make_child_increment` imports this module for `load_grid`, `geom` and
`grad_mag`, not for plotting.

**`python_files/make_bay_mask.py`** — the same `OUTDIR` change, plus `MASKDIR`
(where the built mask is written) changed from `OUTDIR.parent.parent / "masks"`
to `Path(__file__).parent.parent / "masks"`, so it lands in the `masks/` that
`da_config.MASKS` reads instead of following the relocated figure dir. Only
needed to rebuild a mask — the pipeline never imports it. See *Rebuilding the
masks*.

## Rebuilding the masks

The three `masks/*.nc` are committed, binary, ~20 MB total, because the nightly
job must not have a build step. They are static: a different σ or a different
mouth line would get a different filename, so these three will not be
regenerated. `python_files/make_bay_mask.py` is vendored anyway, patched the same
way as `plot_increment_smoothing.py` (`OUTDIR` → the DA scratch log dir) plus
`MASKDIR` → `DA_files/masks` so a rebuild lands where `da_config.MASKS` looks.

Each mask records its own arguments as global netCDF attributes, which is where
these three commands come from — read them back with `ncdump -h` if you ever
doubt the file on disk:

```bash
cd DA_files/python_files
# bay mouth, LV3 and LV4: a line across the Zuniga jetty gap, seed inside the bay
python3 make_bay_mask.py --level LV3 \
    --p1 -117.2389,32.6677 --p2 -117.22,32.6865 --seed -117.17,32.7 --sigma-km 1.0
python3 make_bay_mask.py --level LV4 \
    --p1 -117.2389,32.6677 --p2 -117.22,32.6865 --seed -117.17,32.7 --sigma-km 1.0
# TJRE, LV4 only: a line across the river mouth, seed inside the estuary
python3 make_bay_mask.py --level LV4 --tag tjre \
    --p1 -117.1304,32.558 --p2 -117.1273,32.5532 --seed -117.1144,32.5577 --sigma-km 1.0
```

The 1 km taper is diffused **through water** with no-flux at land, not with a
Euclidean blur: a bay cell is a few hundred metres of *land* from the open ocean
across the Silver Strand, so a Euclidean taper would lift the mask in the middle
of the bay while dropping it on the ocean side of the spit. The taper has to
travel the ~20 km around through the mouth.

`zeta` is deliberately **not** masked. The bay fills and drains in phase with the
coastal ocean, so zeroing its `zeta` increment would manufacture a head
difference across the mouth. The mask applies to `u`, `v`, `temp`, `salt`, whose
bay values are tidal and topographically steered, carry no HFR information, and
are not covered by the balance operator. LV2 has no mask: at 600 m the bay is
too coarsely resolved for the distinction to mean anything.

## Deliberately NOT copied

**`make_atm_from_nam.py`** and **`make_BCnc_from_hycom.py`**. PFM builds its own
forcing from ECMWF and HYCOM, and Phase 0 (2026-10-06) established that both of
PFM's builders are already correct:

* `sdpm_py_util/atm_functions.py:610-611` rotates winds earth → ROMS XI/ETA,
  matching `set_data.F`'s `cff1`/`cff2`, and has since 2025-07-02.
* `sdpm_py_util/ocn_funs_forecast.py` converts in-situ → potential temperature
  with `gsw.pt_from_t`. The adjacent commented `seawater.ptmp` lines are the
  superseded EOS-80 implementation, 7 commented against 7 active — **not** a
  missing conversion, which is how it first looked on a grep.

Porting our NAM/HYCOM builders would reintroduce the bugs September fixed and
would not reproduce PFM's own forcing. Stages `lv1_atm` and `lv1_bc` therefore
call PFM's functions, and the acceptance test is that day 1 of the result
matches the operational `LV1_ATM_FORCING.nc` / `LV1_OCEAN_BC.nc` to round-off.

## Settings carried over, with their reasons

| setting | value | why |
|---|---|---|
| `BLEND_SPACE` | `D` | blending `h` needs a `Dcrit − ζ` clip that inverts the parent column at a deep low tide (50 → 0.145 °C/km on LV4, 2026-02-03) |
| `ROMS_EXE_DA` | `romsM_DA_IC` | IC-only; boundary adjustment blew up at the NW corner |
| `ROMS_EXE_LV4` | `coawstM_noanapassive` | `ANA_PASSIVE` undef so IC-file dye is honoured |
| `HFR_RES` | `2km` | measured 2026-10-06: 6 km lags ~4.8 h, 1 and 2 km ~1.8 h. Latency, not resolution |
| `OBS_ERROR_VAR` | 0.015 | smallest that stayed stable from a poor IC; Desroziers suggests ~2× too large |
| `NRREC` | 0 everywhere | `ini_fields` then recomputes `ubar` from `u3d`, so the IC's `ubar` is discarded — intended |
