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
import subprocess
import sys
from pathlib import Path

import da_config as C


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
    if out.exists() and not force:
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
    MI["lv1_atm_file"] = f"LV1_DA_ATM_{C.ymd(t_da)}.nc"
    MI["lv1_bc_file"] = f"LV1_DA_BC_{C.ymd(t_da)}.nc"
    MI["lv1_ini_file"] = f"LV1_DA_IC_{C.ymd(t_da)}.nc"
    MI["lv1_use_restart"] = 0        # we supply the IC ourselves, in stage 1

    MI["da_t_fore"] = t_fore         # breadcrumbs, so a stray pickle is
    MI["da_t_da"] = t_da             # identifiable months later
    MI["da_written"] = DT.datetime.now()

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


def run_pfm(script: str, func: str, *args: str, timeout: int = 3600) -> None:
    """Run one of PFM's module-level entry points, as its drivers do.

    PFM's modules dispatch on argv (`globals()[args[1]](*args[2:])`) and are run
    as subprocesses from sdpm_py_util, which is also where their relative imports
    and `sys.path.append('../sdpm_py_util')` expect to be.  Calling them in-process
    instead would work until one of them chdir'd or mutated a module global, so we
    keep PFM's own invocation style rather than inventing a cleaner one.

    Raises on a non-zero return code.  Unlike ROMS, these are plain python and a
    non-zero rc means failure -- there is no MPI_Finalize noise to forgive.
    """
    cmd = [C.PY, "-W", "ignore", script, func, *[str(a) for a in args]]
    r = subprocess.run(cmd, cwd=str(C.PFM_PYDIR), text=True, timeout=timeout,
                       env=pfm_env())
    if r.returncode != 0:
        raise RuntimeError(
            f"PFM call failed (rc={r.returncode}): {script} {func} "
            f"{' '.join(str(a) for a in args)}")
