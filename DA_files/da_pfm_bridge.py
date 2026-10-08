"""
da_pfm_bridge.py — run PFM's own forcing builders over the DA window.

Stages 2 and 3 do not reimplement PFM's atm and ocean-BC builders; they call
them.  This module is the whole of that reuse: it builds a DA-private copy of
PFM's model-info dict and runs PFM's functions against it.

WHY A PRIVATE DICT, AND WHY THAT IS ENOUGH
Every PFM builder starts with `PFM = initfuns.get_model_info(pkl_fnm)`, and
get_model_info is a plain pickle load -- no globals, no defaults, no fallback
path.  Every driver threads pkl_fnm down into each subprocess.  So handing them
our pickle redirects the entire computation: the operational
/scratch/PFM_Simulations/PFM_run_info.pkl is neither read nor written, and the
forecast cannot see that we ran.  That is what makes Phase 1 bit-safe.

The dict starts as PFM's REAL operational config (pfm_operational_input.py,
through PFM's own evaluate_function_from_file) and is then patched.  Starting
from the real thing means the science settings -- the vertical stretching, the
grids, the bottom roughness, atm_model, forecast_days -- are whatever PFM
currently says they are, not a copy of them that can drift.  We override only
times, directories and output names.

WHAT STILL HAS TO BE ISOLATED BY HAND
Two things are not reached by redirecting the dict's directories:

  * the ECMWF derived pickles.  atm_functions writes them to
    ecmwf_dir + ecmwf_all_pkl_name, and ecmwf_dir is the shared grib cache.
    Writing the operational NAME there would hand the forecast atm forcing for
    the wrong day -- PFM's own code calls that out as the 2026-08-31 failure.
    So the grib cache stays shared (downloads are additive, and we want the
    cache) while every derived pickle gets a DA tag.

  * the river pickle, PFM['river_pckl_file_full'], which is built from an
    absolute path rather than from lv4_forc_dir.  We do not run the river
    builder, but it is re-pointed anyway so that a later stage cannot write it.
"""
from __future__ import annotations

import datetime as DT
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import da_config as C

# Bump whenever build_minfo changes what it writes.  A cached pickle is reused on
# the strength of _minfo_is_sane, which checks the window -- so without a version
# marker an edit to build_minfo would silently not take on any t_fore whose
# pickle already existed, and the bug it fixed would look unfixed.
MINFO_VERSION = 7


# ---------------------------------------------------------------------------
# building the DA-private model-info pickle
# ---------------------------------------------------------------------------
def _load_pfm_operational_dict() -> dict:
    """PFM's real operational config, loaded the way PFM loads it.

    pfm_operational_input.py is not importable as a module -- it is read and
    exec'd by init_funs*.evaluate_function_from_file, which is also why it has no
    guard against running at import time.  We use PFM's own loader rather than a
    second implementation of it, so a change to how PFM evaluates its input file
    cannot leave us reading it a different way.  The loader needs sdpm_py_util on
    sys.path because the input file imports grid_functions.
    """
    pfm_py = str(C.PFM_PYDIR)
    if pfm_py not in sys.path:
        sys.path.insert(0, pfm_py)
    cwd = os.getcwd()
    try:
        os.chdir(pfm_py)                 # the input file reads grids by relative name
        from init_funs_forecast import evaluate_function_from_file
        MI = evaluate_function_from_file(str(C.PFM_INPUT_PY),
                                        "create_model_info_dict")
    finally:
        os.chdir(cwd)
    if not isinstance(MI, dict):
        # evaluate_function_from_file swallows exceptions and returns None, so a
        # broken input file would otherwise surface much later as a KeyError.
        raise RuntimeError(
            f"could not load create_model_info_dict() from {C.PFM_INPUT_PY}; "
            f"got {type(MI).__name__}.  PFM's loader prints the reason above "
            f"and returns None rather than raising.")
    return MI


def build_minfo(t_fore: DT.datetime, t_da: DT.datetime, *,
                force: bool = False) -> Path:
    """Write the DA run's model-info pickle and return its path.

    fetch_time is t_da, not t_fore: PFM's builders describe a window that STARTS
    at fetch_time and runs forecast_days forward, and our window is
    [t_fore - 24 h, t_fore].  With forecast_days = 1.0 that window ends exactly
    at t_fore, which is the identity the whole pre-step rests on.
    """
    out = C.minfo_pkl(t_fore.strftime("%Y%m%d%H"))
    if out.exists() and not force and _minfo_is_sane(out, t_fore, t_da):
        return out

    MI = _load_pfm_operational_dict()

    # -- times.  One day, ending at t_fore. --------------------------------
    MI["run_type"] = "forecast"      # not 'hindcast': the hindcast branch is the
                                     # NAM path, and it applies no wind rotation
                                     # at all (see da_config.ATM_SOURCE)
    MI["forecast_days"] = 1.0
    MI["fetch_time"] = t_da
    MI["fore_end_time"] = t_fore
    MI["yyyymmdd"] = t_da.strftime("%Y%m%d")
    MI["hhmm"] = t_da.strftime("%H%M")
    MI["sim_start_time"] = t_da
    MI["sim_end_time"] = t_fore
    MI["start_time"] = DT.datetime.now()
    MI["utc_time"] = DT.datetime.now(DT.timezone.utc)

    # forecast_days is copied into the per-level timing dict, and the .in writers
    # read it from there rather than from the top level.
    for lv in ("L1", "L2", "L3", "L4"):
        if ("tinfo" in MI) and ((lv, "forecast_days") in MI["tinfo"]):
            MI["tinfo"][lv, "forecast_days"] = 1.0

    # -- directories.  Everything we write goes under the DA scratch tree. --
    stamp_da = t_da.strftime("%Y%m%d%H")
    for lv, n in (("lv1", "LV1"), ("lv2", "LV2"), ("lv3", "LV3"), ("lv4", "LV4")):
        MI[f"{lv}_run_dir"] = str(C.scratch_dir(n, "Run"))
        MI[f"{lv}_forc_dir"] = str(C.scratch_dir(n, "Forc"))
        MI[f"{lv}_his_dir"] = str(C.scratch_dir(n, "His"))
    MI["restart_files_dir"] = str(C.SCRATCH / "Rst")

    # -- restarts: READ the operational SWAN hotstarts, WRITE our own ---------
    # The SWAN initial condition is the hotstart the operational forecast already
    # deposited.  _in points at a DA-PRIVATE directory of symlinks, NOT at the
    # shared /scratch/PFM_Simulations/restart_swan_data, because PFM's
    # restart_setup calls three functions -- remove_swan_rst_nohour,
    # remove_old_restart_files and remove_swan_restarts_eq_foretime -- that each
    # os.remove() files globbed straight out of restart_swanfiles_dir_in.  Aimed
    # at the shared directory that would delete operational restarts out of a
    # 357 GB store.  stage_swan_hotstart() populates the private directory and
    # pick_swan_hotstart() selects from it read-only.  _out is in the DA tree so
    # this run's own restarts never land among the forecast's.
    MI["restart_swanfiles_dir_in"] = str(C.SWAN_RST_DA_IN)
    MI["restart_swanfiles_dir_out"] = str(C.SCRATCH / "Rst")
    MI["restart_ncfiles_dir_in"] = str(C.SCRATCH / "Rst")
    MI["restart_ncfiles_dir_out"] = str(C.SCRATCH / "Rst")

    # cdip_data_dir MUST be redirected.  swan_functions.cdip_ncs_to_dict sets
    # refresh='refresh' unconditionally on its second line, overriding the
    # argument, and then os.remove()s every *.nc in cdip_data_dir before
    # downloading replacements.  Left at the inherited
    # /scratch/PFM_Simulations/cdip_data that would delete the operational
    # forecast's 81 live CDIP files and hit the network -- both of which PFM_DA
    # must never do.
    MI["cdip_data_dir"] = str(C.CDIP_DA)
    MI["cdip_archive_dir"] = str(C.CDIP_ARCHIVE)

    # LV4 runs coawstM_noanapassive: ANA_PASSIVE is undef so dye_01/dye_02 from
    # the IC file are honoured rather than zeroed at t=0.
    MI["lv4_executable"] = C.ROMS_EXE_LV4
    MI["river_pckl_file_full"] = str(C.scratch_dir("LV4", "Forc") / "river_Q.pkl")
    for lv, n in (("lv1", "LV1"), ("lv2", "LV2"), ("lv3", "LV3"), ("lv4", "LV4")):
        MI[f"{lv}_plot_dir"] = str(C.SCRATCH / "Log" / "figures" / n)
    # Three more absolute names that create_model_info_dict builds itself rather
    # than from the *_dir keys, so patching the dirs does not move them.
    MI["lv4_coawst_varinfo_full"] = str(C.scratch_dir("LV4", "Run")
                                        / "LV4_coawst_varinfo.dat")
    MI["qtj_obs_fname_full"] = str(C.scratch_dir("LV4", "Forc")
                                   / "qtjr_obs_tmp.csv")
    MI["lv4_web_name_full"] = str(C.scratch_dir("LV4", "His")
                                  / f"web_data_{stamp_da}.nc")

    # -- ECMWF: share the grib cache, never the derived pickles. -----------
    MI["ecmwf_dir"] = str(C.ECMWF_CACHE) + "/"      # trailing slash: PFM
                                                    # concatenates, never joins
    tag = C.ECMWF_PKL_TAG
    stamp = t_fore.strftime("%Y%m%d%H")
    MI["ecmwf_all_pkl_name"] = f"ecmwf_all_{tag}_{stamp}.pkl"
    MI["ecmwf_pkl_roms_vars"] = f"ecmwf_roms_vars_{tag}_{stamp}.pkl"
    MI["ecmwf_pkl_on_roms_grid"] = f"ecmwf_on_romsgrid_{tag}_{stamp}.pkl"

    # -- output names.  Date-stamped, so a rerun cannot read a stale file. -
    # ALL FOUR levels, not just LV1.  PFM names its output from these keys, so a
    # level left at its operational default writes LV2_ATM_FORCING.nc into the DA
    # tree and the stage then cannot find the file it asked for -- which is
    # exactly how the first LV2 forward failed.  The date tag also keeps one
    # cycle's forcing from being mistaken for another's.
    ymd = C.ymd(t_da)
    for lv, n in (("lv1", "LV1"), ("lv2", "LV2"), ("lv3", "LV3"), ("lv4", "LV4")):
        MI[f"{lv}_atm_file"] = f"{n}_DA_ATM_{ymd}.nc"
        MI[f"{lv}_bc_file"] = f"{n}_DA_BC_{ymd}.nc"
        MI[f"{lv}_ini_file"] = f"{n}_DA_IC_{ymd}.nc"
    # NRREC = 0 at EVERY level: each one starts from a single-record IC this
    # pipeline built, not from a ROMS restart.  The distinction is not cosmetic --
    # with lv4_use_restart = 1 the generator writes NRREC == 1 and ROMS then
    # demands the restart-only variables rzeta, rubar, rvbar, ru, rv, which a
    # single-record IC does not carry.  That is what the second LV4 attempt died
    # on: "CHECKVARS_NF90 - unable to find model variable: rzeta".
    #
    # LV2 and LV3 were insulated by their seed .in hardcoding NRREC == 0; LV4
    # takes it from here, which is why only LV4 showed the bug.  NRREC = 0 also
    # means ini_fields recomputes ubar from u3d, which is what we want and is
    # consistent with how the increment was built.
    for lv in ("lv1", "lv2", "lv3", "lv4"):
        MI[f"{lv}_use_restart"] = 0

    MI["da_t_fore"] = t_fore         # breadcrumbs, so a stray pickle is
    MI["da_t_da"] = t_da             # identifiable months later
    MI["da_written"] = DT.datetime.now()
    MI["da_minfo_version"] = MINFO_VERSION

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as fp:
        pickle.dump(MI, fp, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(out)                 # atomic: a half-written config read by a
                                     # subprocess would be hard to diagnose

    # create_model_info_dict derives lv{N}_his_name_full / rst_name_full from the
    # OPERATIONAL dirs and the OPERATIONAL fetch_time, both of which we just
    # changed, so those names are stale on two counts.  PFM has the function that
    # rebuilds them -- use it rather than a second copy of the naming rules, so a
    # change to PFM's naming follows through here automatically.  It rewrites the
    # pickle in place, which is why it runs after the write.
    run_pfm("init_funs_forecast.py", "add_file_names_2MI", stamp_da, str(out))

    # add_file_names_2MI (just run) names LV4's history LV4_ocean_his_<stamp>.nc
    # from PFM's convention.  LV2 and LV3 get their HISNAME substituted into the
    # seed .in directly, but LV4's comes from the model-info, so override it here
    # or the run writes a file the stage will not recognise as its keeper.
    with open(out, "rb") as fp:
        MI2 = pickle.load(fp)
    ymd2 = C.ymd(t_da)
    MI2["lv4_his_name"] = f"LV4_DA_his_{ymd2}.nc"
    MI2["lv4_his_name_full"] = str(C.scratch_dir("LV4", "His")
                                   / MI2["lv4_his_name"])
    with open(out, "wb") as fp:
        pickle.dump(MI2, fp, protocol=pickle.HIGHEST_PROTOCOL)

    leaks = _operational_leaks(out)
    if leaks:
        raise RuntimeError(
            f"the DA model-info pickle still points at the operational tree for "
            f"{sorted(leaks)}.  Refusing to continue: a builder using one of "
            f"these would write into the forecast's own directories.")
    return out


_OPERATIONAL_MARKERS = ("LV1_Forecast", "LV2_Forecast", "LV3_Forecast",
                        "LV4_Forecast", "PFM_run_info.pkl")


def _operational_leaks(pkl: Path) -> dict:
    """Every dict entry that still names an operational path.

    This is a real guard, not a formality: the first build of this pickle left 11
    such keys, including lv{N}_his_name_full carrying the operational His dir and
    the CURRENT operational timestamp.  Whatever we have not thought to redirect,
    this catches before a builder writes through it.
    """
    with open(pkl, "rb") as fp:
        MI = pickle.load(fp)
    return {k: v for k, v in MI.items()
            if isinstance(v, (str, Path))
            and any(m in str(v) for m in _OPERATIONAL_MARKERS)}


def _minfo_is_sane(pkl: Path, t_fore: DT.datetime, t_da: DT.datetime) -> bool:
    """Is a cached pickle still the one this window needs?

    Reuse used to key on existence alone, which is not enough: anything that
    edits a cached pickle in place leaves a file that describes a DIFFERENT
    window and is silently reused.  That happened here -- a 5-day experiment
    rewrote forecast_days to 5.0 on the cached 1-day config, and the next run
    would have built forcing for a 5-day window while reporting a 1-day one.
    So the window is checked, not assumed.
    """
    try:
        with open(pkl, "rb") as fp:
            MI = pickle.load(fp)
    except Exception:  # noqa: BLE001
        return False
    return (MI.get("da_minfo_version") == MINFO_VERSION
            and MI.get("da_t_fore") == t_fore
            and MI.get("da_t_da") == t_da
            and MI.get("fetch_time") == t_da
            and float(MI.get("forecast_days", -1)) == 1.0)


def stage_cdip_from_archive(t_da: DT.datetime, dest: Path | None = None) -> int:
    """Put the window's CDIP spectra where the SWAN boundary builder expects them.

    Returns the number of station files staged.

    The archive is date-stamped and complete -- 81 stations per cycle, named
    cdip_<station>_<YYYYMMDDHH>.nc, exactly as get_cdip_data would name a fresh
    download -- so the boundary file can be built for a past window with no
    network access at all.
    """
    dest = Path(dest or C.CDIP_DA)
    dest.mkdir(parents=True, exist_ok=True)
    stamp = t_da.strftime("%Y%m%d%H")
    src = sorted(C.CDIP_ARCHIVE.glob(f"cdip_*_{stamp}.nc"))
    if not src:
        raise FileNotFoundError(
            f"no archived CDIP spectra for {stamp} in {C.CDIP_ARCHIVE}.  Without "
            f"them the SWAN boundary file cannot be built for this window, and "
            f"downloading is not an option for a window this far past.")
    n = 0
    for f in src:
        d = dest / f.name
        if not (d.is_file() and d.stat().st_size == f.stat().st_size):
            shutil.copy2(f, d)
        n += 1
    return n


def cdip_dict_from_archive(pkl: Path, t_da: DT.datetime) -> None:
    """Build the CDIP pickle the SWAN boundary builder reads, from the archive.

    cdip_ncs_to_dict does two things: refresh the .nc files, then digest them into
    a pickle.  Only the second is wanted here -- but the refresh is not optional
    in that function (refresh='refresh' is assigned over the argument), and it
    deletes before it fetches.  So the fetch is replaced in-process with a copy
    from the archive: the delete then removes our own staged copies and the
    "download" puts them straight back, leaving the digest step to run unchanged.

    Called in-process rather than as a subprocess precisely so that substitution
    is possible.
    """
    pfm_py = str(C.PFM_PYDIR)
    if pfm_py not in sys.path:
        sys.path.insert(0, pfm_py)
    cwd = os.getcwd()
    try:
        os.chdir(pfm_py)
        import swan_functions as SW
        real = SW.get_cdip_data
        SW.get_cdip_data = lambda _pkl: stage_cdip_from_archive(t_da)
        try:
            SW.cdip_ncs_to_dict("from-archive", str(pkl))
        finally:
            SW.get_cdip_data = real
    finally:
        os.chdir(cwd)


def build_swan_wnd(pkl: Path, fout: Path) -> bool:
    """Make the SWAN wind file if this configuration wants one.  True if written.

    Mirrors what PFM's own driver does: rc != 0 is only fatal when
    swan_wind_wave_generation is on.  With it off, mk_swan_wnd_file exits 1 on
    purpose and there is nothing to write -- SWAN is then driven by the boundary
    spectra alone.
    """
    import pickle as _p
    with open(pkl, "rb") as fp:
        want = bool(_p.load(fp).get("swan_wind_wave_generation", False))
    rc = run_pfm("swan_functions.py", "mk_swan_wnd_file", str(fout), str(pkl),
                 timeout=3000, ok_rc=(0, 1))
    if want and rc != 0:
        raise RuntimeError(
            f"swan_wind_wave_generation is on but mk_swan_wnd_file returned "
            f"{rc}; SWAN would run without the winds it is configured to use")
    if not want:
        print("    SWAN wind-wave generation is OFF -- no .wnd file needed "
              "(mk_swan_wnd_file exits 1 by design in that case)")
    return rc == 0


def stage_lv4_clm(t_da: DT.datetime, t_fore: DT.datetime,
                  out: Path, template: Path | None = None) -> Path:
    """LV4 interior-nudging climatology for this window.

    NOT built with PFM's mk_lv4_clm_nc: that reads BC_LV4_tmp_file.pkl, which
    PFM's own LV4 flow produces from HYCOM.  PFM_DA takes LV4's boundaries from
    the LV3 DA history instead, so the pickle never exists here.

    Copy the operational file and rewrite only its time variables, which is what
    the research chain settled on.  ROMS refuses to start when the initialisation
    time lies outside the climatology's span, so the span is the only thing that
    has to be right; the VALUES are left alone because interior nudging is weak
    and a 24 h window cannot respond much to slightly stale climatology.

    The span is padded a day either side so a run that drifts past t_fore -- or
    starts a moment early -- is still inside it.
    """
    template = Path(template or
                    "/scratch/PFM_Simulations/LV4_Forecast/Forc/LV4_clm.nc")
    if not template.is_file():
        raise FileNotFoundError(f"no LV4 climatology template at {template}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    shutil.copyfile(template, out)

    import netCDF4 as nc_
    import numpy as np_
    epoch = DT.datetime(1999, 1, 1)
    lo = (t_da - DT.timedelta(days=1) - epoch).total_seconds() / 86400.0
    hi = (t_fore + DT.timedelta(days=1) - epoch).total_seconds() / 86400.0
    touched = []
    with nc_.Dataset(out, "a") as d:
        for name in [v for v in d.variables if v.endswith("_time")]:
            v = d.variables[name]
            n = v.shape[0] if v.shape else 1
            v[:] = (np_.array([lo, hi]) if n == 2
                    else np_.linspace(lo, hi, n))
            touched.append(f"{name}[{n}]")
    print(f"    LV4 clm retimed to {t_da - DT.timedelta(days=1):%Y-%m-%d} .. "
          f"{t_fore + DT.timedelta(days=1):%Y-%m-%d}  ({', '.join(touched)})")
    return out


def stage_swan_hotstart(t_da: DT.datetime, dest: Path | None = None) -> int:
    """Symlink the SWAN restarts valid at t_da into a DA-private directory.

    Returns the number of parts linked (72 when complete, one per SWAN rank).

    Symlinks rather than copies: each candidate set is ~2.5 GB and SWAN only
    reads them.  Private rather than shared because PFM's restart bookkeeping
    deletes from this directory -- see build_minfo.
    """
    dest = Path(dest or C.SWAN_RST_DA_IN)
    dest.mkdir(parents=True, exist_ok=True)
    stamp = t_da.strftime("%Y%m%d%H%M")
    src = sorted(C.SWAN_RST_SHARED.glob(f"LV4_swan_rst_{stamp}_000.dat-*"))
    if not src:
        raise FileNotFoundError(
            f"no SWAN restart valid at {t_da:%Y-%m-%d %H}Z in "
            f"{C.SWAN_RST_SHARED} (looked for LV4_swan_rst_{stamp}_000.dat-*).  "
            f"Without one SWAN would have to cold-start from a flat sea, which "
            f"takes longer than the 24 h window to spin up.")
    for f in src:
        link = dest / f.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(f)
    return len(src)


def pick_swan_hotstart(pkl: Path) -> str | None:
    """PFM's own choice of hotstart file, read-only.

    get_swan_restart_file_name parses init and lead from each name and takes the
    one where init + lead == fetch_time, returning the base name with the rank
    suffix stripped -- SWAN appends -001..-072 itself.  Called instead of
    restart_setup, which would make the same choice but delete as it went.
    """
    pfm_py = str(C.PFM_PYDIR)
    if pfm_py not in sys.path:
        sys.path.insert(0, pfm_py)
    cwd = os.getcwd()
    try:
        os.chdir(pfm_py)
        import init_funs_forecast as IF
        return IF.get_swan_restart_file_name(str(pkl))
    finally:
        os.chdir(cwd)


def set_swan_init(pkl: Path, hotstart: str | None) -> str:
    """Write swan_init_txt_full into the model-info, as restart_setup would."""
    with open(pkl, "rb") as fp:
        MI = pickle.load(fp)
    MI["swan_init_txt_full"] = f"HOTSTART '{hotstart}'" if hotstart else "ZERO"
    tmp = Path(pkl).with_suffix(".tmp")
    with open(tmp, "wb") as fp:
        pickle.dump(MI, fp, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(pkl)
    return MI["swan_init_txt_full"]


def make_lv4_dotins(pkl: Path) -> None:
    """PFM's generator for LV4's three .in files (coupled / ROMS / SWAN).

    In-process: run_funs reads the model-info by path, so this needs no argv
    plumbing, and PFM's own driver calls it the same way.
    """
    pfm_py = str(C.PFM_PYDIR)
    if pfm_py not in sys.path:
        sys.path.insert(0, pfm_py)
    cwd = os.getcwd()
    try:
        os.chdir(pfm_py)
        import run_funs as R
        R.make_LV4_coawst_dotins_dotsb(str(pkl), "fore")
    finally:
        os.chdir(cwd)


def assert_not_operational(pkl: Path) -> None:
    """Guard: refuse to proceed if the DA pickle is the operational one.

    Cheap, and the failure it prevents is the expensive kind -- a DA run that
    silently rewrites the forecast's config.
    """
    pkl = Path(pkl).resolve()
    for bad in (Path("/scratch/PFM_Simulations/PFM_run_info.pkl"),
                Path("/scratch/PHM_Simulations/PFM_run_info.pkl")):
        if pkl == bad:
            raise RuntimeError(
                f"refusing to run: the DA model-info pickle resolves to the "
                f"OPERATIONAL config {bad}.  Writing it would change the "
                f"forecast's own settings.")
    if C.SCRATCH not in pkl.parents:
        raise RuntimeError(
            f"refusing to run: the DA model-info pickle {pkl} is outside the DA "
            f"scratch tree {C.SCRATCH}")


# ---------------------------------------------------------------------------
# running PFM's functions
# ---------------------------------------------------------------------------
def _import_ocn_funs():
    """Import PFM's ocean module with sdpm_py_util importable and cwd correct."""
    pfm_py = str(C.PFM_PYDIR)
    if pfm_py not in sys.path:
        sys.path.insert(0, pfm_py)
    cwd = os.getcwd()
    try:
        os.chdir(pfm_py)
        import ocn_funs_forecast as O
    finally:
        os.chdir(cwd)
    return O


def hycom_file_list(init_day: str, t_da: DT.datetime, t_fore: DT.datetime,
                    pkl: Path) -> list[Path]:
    """Every HYCOM file PFM will ask for, for this init and window.

    Uses PFM's OWN get_hycom_nc_file_names rather than a second copy of the naming
    rules: the names encode the init time (hardcoded T12:00), the 3-hourly cadence
    for s3z/t3z/u3z/v3z and the 1-hourly cadence for ssh, and all of that has to
    stay in step with PFM.
    """
    O = _import_ocn_funs()
    return [Path(f) for f in O.get_hycom_nc_file_names(
        init_day, t_da.strftime("%Y%m%d%H%M"), t_fore.strftime("%Y%m%d%H%M"),
        str(pkl))]


def hycom_init_for_window(t_da: DT.datetime, t_fore: DT.datetime,
                         pkl: Path, allow_download: bool = True) -> str:
    """The HYCOM init day ('YYYYMMDD') to read, filling gaps only if we must.

    PFM_DA READS WHAT IS ON DISK.  The forecast has normally fetched this window
    already, so re-fetching would re-pull data we have, hit an external service
    for nothing, and make the pre-step depend on tds.hycom.org being up.  The
    first version of this function enforced that absolutely and raised when
    nothing covered the window.

    That is too strict for the case that will actually occur: if a PFM cycle fails
    outright and is rerun off-schedule, the next PFM_DA window was never fetched
    by anybody.  hycom_data/ is also cleaned each cycle and holds only ~2 inits,
    so the window can age out.  So: prefer disk, verify, and download only the
    gap -- never the whole window when most of it is present.

    Picks the latest init that is complete.  If none is, takes the init needing
    the fewest files and fetches just those.
    """
    from da_preflight import check_hycom

    item = check_hycom(t_da, t_fore, pkl)
    if item.ok:
        return item.resolved["init"]

    init = item.resolved["init"]
    missing = item.resolved["missing"]
    if not allow_download:
        raise FileNotFoundError(
            f"HYCOM is short for {t_da:%Y-%m-%d %H}Z..{t_fore:%Y-%m-%d %H}Z and "
            f"downloading is disabled (--no-download): best init {init}T12 is "
            f"missing {len(missing)} file(s) in {C.HYCOM_CACHE}.  "
            f"First missing: {missing[0].name}")

    print(f"    HYCOM gap: init {init}T12 is missing {len(missing)} file(s); "
          f"fetching only those (PFM_DA normally reads from disk -- this means "
          f"the forecast for this window did not run, or hycom_data/ has been "
          f"cleaned since)", flush=True)

    # PFM's check_and_redownload_ncfiles is the right downloader -- it chunks to
    # <=10 files to stay polite to HYCOM and retries 3 times -- but it decides
    # what to fetch with os.path.getsize(), so it handles a TRUNCATED file and
    # raises on an ABSENT one.  Creating each absent file as zero bytes turns the
    # case it cannot see into the case it handles, and its own delete_files()
    # removes the placeholder before fetching.  If a fetch still fails the file
    # ends up absent again and the verification below catches it.
    for f in missing:
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            f.touch()
    run_pfm("ocn_funs_forecast.py", "check_and_redownload_ncfiles", init, pkl,
            timeout=5400)

    recheck = check_hycom(t_da, t_fore, pkl)
    if not recheck.ok:
        still = recheck.resolved.get("missing", [])
        # do not leave zero-byte placeholders behind to be mistaken for data
        for f in still:
            try:
                if f.is_file() and f.stat().st_size == 0:
                    f.unlink()
            except OSError:
                pass
        raise FileNotFoundError(
            f"HYCOM still incomplete after a download attempt: {recheck.detail}. "
            f"Check that tds.hycom.org is reachable and that the window is within "
            f"HYCOM's retention.")
    print(f"    HYCOM gap filled; init {init}T12 now complete", flush=True)
    return recheck.resolved["init"]


def pfm_env() -> dict:
    """os.environ plus PFM/.env, which holds the download credentials.

    Measured 2026-10-06: without this, get_ecmwf_forecast_grbs_v2 raises
    "ECMWF credentials missing: WGET_ECMWF_USER / WGET_ECMWF_PASS" and returns
    rc=1.  That run still succeeded -- but only because every grib it needed was
    already cached, and got_ecmwf_files() then reported the window covered.  A
    nightly run reaching for an init that has not been cached yet would fail, and
    would fail in the confusing way: a download error swallowed by a success
    path.  So the credentials are loaded explicitly, and their absence is an
    error rather than a silent dependency on the cache.
    """
    env = dict(os.environ)
    dotenv = C.PFM_ROOT / ".env"
    if dotenv.is_file():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip().strip("'\""))
    missing = [k for k in ("WGET_ECMWF_USER", "WGET_ECMWF_PASS")
               if not env.get(k)]
    if missing:
        raise RuntimeError(
            f"ECMWF credentials missing: {', '.join(missing)}.  They live in "
            f"{dotenv}.  Without them a grib that is not already cached cannot "
            f"be downloaded, and the failure is easy to miss because PFM's "
            f"fetch returns rc=1 and the run continues on whatever is cached.")
    return env


def run_pfm(script: str, func: str, *args: str, timeout: int = 3600,
            ok_rc: tuple[int, ...] = (0,)) -> int:
    """Run one of PFM's module-level entry points, as its drivers do.

    PFM's modules dispatch on argv (`globals()[args[1]](*args[2:])`) and are run
    as subprocesses from sdpm_py_util, which is also where their relative imports
    and `sys.path.append('../sdpm_py_util')` expect to be.  Calling them in-process
    instead would work until one of them chdir'd or mutated a module global, so we
    keep PFM's own invocation style rather than inventing a cleaner one.

    Raises unless the return code is in ok_rc.  Unlike ROMS these are plain
    python, so a non-zero rc usually does mean failure -- but not always, and the
    exception is deliberate on PFM's side: mk_swan_wnd_file calls sys.exit(1) when
    swan_wind_wave_generation is False, because then no .wnd file is wanted at
    all.  PFM's own driver prints that return code as "(0 or 1 = good)".  Pass
    ok_rc=(0, 1) for that call rather than relaxing the default.
    """
    cmd = [C.PY, "-W", "ignore", script, func, *[str(a) for a in args]]
    r = subprocess.run(cmd, cwd=str(C.PFM_PYDIR), text=True, timeout=timeout,
                       env=pfm_env())
    if r.returncode not in ok_rc:
        raise RuntimeError(
            f"PFM call failed (rc={r.returncode}, accepted {ok_rc}): {script} "
            f"{func} {' '.join(str(a) for a in args)}")
    return r.returncode
