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

**`make_atm_from_nam.py`**, **`download_nam_range.py`** and
**`make_BCnc_from_hycom.py`**. PFM builds its own forcing and stages 2–3 call
PFM's builders through `da_pfm_bridge`. The atm result is **bit-identical** to
the operational file — `max|diff| = 0.000e+00` across all nine fields over 25
shared records, 2026-10-06..10-07 — so there is nothing for a ported builder to
improve on.

I briefly vendored the two NAM scripts and then removed them. The reasoning is
worth recording because the first reading was wrong in both directions:

* `driver_functions.run_hind_LV1` is the only 1-day-at-a-time driver, and it
  builds atm from `hind_functions.py` (NAM), so that looked like the path a
  closed window must take.
* `ecmwf_data/` looked like a per-cycle working directory. It is not: it keeps
  one init group per day, 99 groups back to 2026-07-07, 101 hourly steps each,
  never pruned (the delete at `atm_functions.py:111-113` is commented out, though
  it still *prints* that it deleted — misleading log line, no effect).

So a closed window **is** already on disk, ECMWF is the operational product, and
the ECMWF path rotates correctly. Stage 2 uses it.

### A real bug this turned up in PFM's NAM path

Not ours to fix here, but it should be written down: **PFM's NAM path applies no
wind rotation at all.** `hind_functions.grb2_to_pickle` reads `UGRD`/`VGRD`
straight out of the grib2, and `nam_pkls_2_romsatm_pkl` interpolates them onto
the ROMS grid as if they were already ROMS components. `hind_functions` does not
import `atm_functions`, so the `cosang`/`sinang` rotation at
`atm_functions.py:610` is not on that path at all. NAM grid 218 is Lambert
Conformal and publishes u/v along grid axes, so **both** rotation stages are
missing and wind stress is mis-aimed by ~14°.

This does not affect the operational forecast (ECMWF, correctly rotated) or the
DA pre-step (ditto). It does affect anything run through `run_hind_LV*` — the PHM
hindcasts.

Phase 0's conclusion, "reuse PFM's builder, do not port ours," therefore holds
for the ECMWF forecast path and **not** for NAM. The distinction is the product,
not the direction of time: ECMWF is lat/lon so its u/v are earth-relative and the
single `RMG['angle']` stage is the whole rotation; NAM needs two stages and gets
none.

### What was confirmed correct, and how

* `atm_functions.py:596-598,610` rotates earth → ROMS XI/ETA with
  `angr = RMG['angle']`, matching `set_data.F`'s `cff1`/`cff2`.
* `ocn_funs_forecast.ocnr_2_BCdict_1hrzeta_from_tmppkls` converts in-situ →
  potential temperature with `gsw.pt_from_t` (lines 3551/3554/3557). The adjacent
  commented `seawater.ptmp` lines are the superseded EOS-80 version, 7 commented
  against 7 active — **not** a missing conversion, which is how it first looks on
  a grep.

## Isolation from the operational forecast

Stages 2–3 run PFM's code, so this is the part that keeps Phase 1 bit-safe for
the forecast. `da_pfm_bridge.build_minfo` writes a DA-private model-info pickle;
every PFM builder reads its config from the `pkl_fnm` it is handed
(`get_model_info` is a plain pickle load, no globals, no fallback), so that one
redirect moves the entire computation.

Three things are **not** reached by redirecting directories, and each is handled
explicitly:

| leak | why it escapes | fix |
|---|---|---|
| derived ECMWF pickles | written to `ecmwf_dir + ecmwf_all_pkl_name`, and `ecmwf_dir` is the shared grib cache | gribs stay shared (downloads are additive, we want the cache); every derived pickle gets a `_DA_<t_fore>` tag |
| `lv{N}_his_name_full`, `rst_name_full` | `create_model_info_dict` derives them from the operational dirs *and* the operational timestamp, before our patch | re-run PFM's own `add_file_names_2MI` after writing the pickle |
| `lv4_coawst_varinfo_full`, `qtj_obs_fname_full`, `lv4_web_name_full`, `*_plot_dir` | built as absolute paths, not from the `*_dir` keys | redirected by hand |

`_operational_leaks()` then walks the finished pickle for any value still naming
the operational tree and **refuses to run** if one remains. That guard is not a
formality: the first build left 11 such keys.

Why the DA-tagged ECMWF pickle name is not cosmetic — PFM's own code raises on a
failed rebuild with the comment that continuing "would leave the previous cycle's
ecmwf_all.pkl in place and silently build atm forcing for the wrong day (this is
exactly what happened 2026-08-31)". A DA run writing the operational name would
do that to the forecast deliberately. Stage 3 also checks its output's time axis
covers the window, which is the only check that notices that class of failure:
the file is perfectly valid, just for the wrong day.

Verified after the first real run: the shared cache went 19927 → 19928 files
(our one DA-tagged pickle), init groups 99 → 99, and the operational
`ecmwf_all.pkl` was untouched.

## Credentials

`run_pfm` loads `PFM/.env` into the subprocess environment. Without it ECMWF
fetches raise "credentials missing: WGET_ECMWF_USER / WGET_ECMWF_PASS" and return
`rc=1`, and PFM's fetch loop then continues on whatever is cached. The first run
passed only because every grib it needed was already present; a nightly run
reaching for an uncached init would have failed in the confusing way, so a
missing credential is now an error rather than a silent dependency on the cache.

## Settings carried over, with their reasons

| setting | value | why |
|---|---|---|
| `BLEND_SPACE` | `D` | blending `h` needs a `Dcrit − ζ` clip that inverts the parent column at a deep low tide (50 → 0.145 °C/km on LV4, 2026-02-03) |
| `ROMS_EXE_DA` | `romsM_DA_IC` | IC-only; boundary adjustment blew up at the NW corner |
| `ROMS_EXE_LV4` | `coawstM_noanapassive` | `ANA_PASSIVE` undef so IC-file dye is honoured |
| `HFR_RES` | `2km` | measured 2026-10-06: 6 km lags ~4.8 h, 1 and 2 km ~1.8 h. Latency, not resolution |
| `OBS_ERROR_VAR` | 0.015 | smallest that stayed stable from a poor IC; Desroziers suggests ~2× too large |
| `NRREC` | 0 everywhere | `ini_fields` then recomputes `ubar` from `u3d`, so the IC's `ubar` is discarded — intended |
