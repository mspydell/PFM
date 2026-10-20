#!/usr/bin/env python3
"""
start_PFM_plus_DA -- one entry point for the DA pre-step and the forecast.

    start_PFM_plus_DA                      DA, then PFM started from its analysis
    start_PFM_plus_DA DA_only              the pre-step alone
    start_PFM_plus_DA PFM_only             forecast only; use DA ICs IF present
    start_PFM_plus_DA PFM_only no_DA       forecast only, restart files (today's PFM)
    start_PFM_plus_DA 2026101100           pin the forecast time
    start_PFM_plus_DA 2026101100 testing --length 1
    start_PFM_plus_DA PFM_only --level LV4 --fore_time 2026101100

HOW IT AVOIDS TOUCHING OPERATIONAL CODE.  driver_run_pfm_phm.py does
initialize_model(), then hands a pkl to driver_run_forecast_LV1234.py.  This
script does the same two things with the DA work in between: build the pkl,
redirect it into the sandbox if testing, point it at the DA ICs, prove it is not
operational, save it, and hand it to the same forecast driver.  No PFM driver is
modified.  The one change in PFM itself is PFM_FORCE_FETCH_TIME in
pfm_operational_input_new.py, three env-gated lines that are inert unless set.

FALLING THROUGH IS THE POLICY.  If the DA has no usable IC for a level, that
level keeps its restart settings and starts the way it always has.  A forecast on
restarts beats no forecast, so a DA failure must never stop the forecast.  The
one thing that WILL stop it is a testing run whose paths are not fully
sandboxed -- see da_sandbox.
"""
from __future__ import annotations

import argparse
import datetime as DT
import os
import pickle
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sdpm_py_util"))

import da_config as C            # noqa: E402
import da_ic_handoff as ICH      # noqa: E402
import da_pfm_bridge as B        # noqa: E402
import da_sandbox as SB          # noqa: E402

DRIVER_DIR = HERE.parent / "driver"
INPUT_PY = C.PFM_INPUT_PY
OPER_PKL = Path("/scratch/PFM_Simulations/forecast_info.pkl")
TEST_PKL = Path(SB.SANDBOX_SCRATCH) / "forecast_info_testing.pkl"

KEYWORDS = {"PFM_only", "DA_only", "no_DA", "testing"}


def load_pfm_env() -> None:
    """Merge PFM/.env into os.environ so the forecast can download.

    PFM_DA never needs this -- it reads ECMWF and HYCOM off disk by design and
    does not fetch.  The FORECAST does: LV1's atm step calls
    get_ecmwf_forecast_grbs_v2, which raises

        ECMWF credentials missing: WGET_ECMWF_USER / WGET_ECMWF_PASS

    when they are unset.  On 2026-10-12 that cost 71 minutes before LV1 gave up,
    so the check happens here, at startup, before anything expensive.

    Merged into os.environ rather than passed per-subprocess because
    initialize_model runs IN THIS PROCESS and the level drivers are started
    several subprocess layers down; setting it once covers both.
    """
    env = B.pfm_env()                       # raises with a useful message
    for k in ("WGET_ECMWF_USER", "WGET_ECMWF_PASS"):
        os.environ[k] = env[k]
    # carry anything else .env defines that we do not already have
    for k, v in env.items():
        os.environ.setdefault(k, v)


def _parse(argv):
    # Split the bare keywords and a bare YYYYMMDDHH out of argv, leaving the
    # flags for argparse.  The skip matters: --fore_time's VALUE is also a
    # 10-digit number, and without it the stamp was stolen from the flag and
    # argparse died with "expected one argument".
    _takes_value = {"--fore_time", "--length", "--level"}
    words, stamps, rest, skip = [], [], [], False
    for tok in argv:
        if skip:
            rest.append(tok); skip = False; continue
        if tok in _takes_value:
            rest.append(tok); skip = True; continue
        if tok in KEYWORDS:
            words.append(tok)
        elif tok.isdigit() and len(tok) == 10:
            stamps.append(tok)
        else:
            rest.append(tok)

    ap = argparse.ArgumentParser(prog="start_PFM_plus_DA", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fore_time", default=None, help="YYYYMMDDHH, same as the bare stamp")
    ap.add_argument("--length", type=float, default=None,
                    help="forecast days; short runs are for testing the plumbing")
    ap.add_argument("--level", action="append", default=None,
                    help="LV1..LV4; repeatable.  Only LV4 can run alone today.")
    ap.add_argument("--dry-run", action="store_true",
                    help="do everything except launch the forecast")
    a = ap.parse_args(rest)

    a.t_fore = a.fore_time or (stamps[0] if stamps else None)
    a.testing = "testing" in words
    a.run_da = "PFM_only" not in words
    a.run_pfm = "DA_only" not in words
    a.use_da_ics = "no_DA" not in words
    if "no_DA" in words and "PFM_only" not in words:
        ap.error("no_DA only makes sense with PFM_only -- "
                 "running the DA and then ignoring it is never what you want")
    return a


def _run(cmd, **kw) -> int:
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], **kw).returncode


def _build_pkl(pkl: Path, t_fore: str | None) -> dict:
    """initialize_model into `pkl`, with the forecast time pinned if asked."""
    import init_funs_forecast as initfuns
    env_before = os.environ.get("PFM_FORCE_FETCH_TIME")
    if t_fore:
        os.environ["PFM_FORCE_FETCH_TIME"] = t_fore
    try:
        pkl.parent.mkdir(parents=True, exist_ok=True)
        initfuns.initialize_model(str(INPUT_PY), str(pkl))
    finally:
        if env_before is None:
            os.environ.pop("PFM_FORCE_FETCH_TIME", None)
        else:
            os.environ["PFM_FORCE_FETCH_TIME"] = env_before
    with open(pkl, "rb") as fh:
        return pickle.load(fh)


def main(argv=None) -> int:
    a = _parse(argv if argv is not None else sys.argv[1:])
    print("  start_PFM_plus_DA")
    # Fail in seconds, not 71 minutes into LV1's atm step.
    try:
        load_pfm_env()
        print("    credentials: PFM/.env loaded")
    except RuntimeError as e:
        print("    " + str(e))
        return 2
    print(f"    DA pre-step : {'yes' if a.run_da else 'no'}")
    print(f"    forecast    : {'yes' if a.run_pfm else 'no'}"
          + ("" if a.use_da_ics else "   (no_DA: restart files, as PFM always has)"))
    print(f"    testing     : {a.testing}"
          + (f"   sandbox {SB.SANDBOX_SCRATCH}" if a.testing else ""))
    print(f"    forecast time: {a.t_fore or 'derived from now'}")

    # ---- 1. the DA pre-step ------------------------------------------------
    if a.run_da:
        if not a.t_fore:
            # the DA needs a concrete t_fore, and PFM's own derivation is the
            # only thing allowed to decide it -- so build the pkl first and read
            # it back rather than re-implementing the 6-hourly rounding here.
            print("  deriving the forecast time from PFM itself...")
            MI = _build_pkl(TEST_PKL if a.testing else OPER_PKL, None)
            a.t_fore = MI["sim_time_1"].strftime("%Y%m%d%H")
            print(f"    -> {a.t_fore}")
        # -u, or the pre-step's own prints are block-buffered when this
        # script's stdout is a file: on 2026-10-19 the log sat at the CDIP
        # step while lv4_fwd's slurm job had already been running for five
        # minutes, which reads as a hang.  The forecast driver below has
        # always had -u; this call was the one that did not.
        rc = _run([C.PY, "-u", HERE / "run_PFM_DA.py", a.t_fore], cwd=HERE)
        if rc != 0:
            print(f"  DA pre-step returned {rc}.  Continuing: the forecast falls "
                  f"through to whatever ICs it can find.")

    if not a.run_pfm:
        return 0

    # ---- 2. build the model-info pkl ---------------------------------------
    pkl = TEST_PKL if a.testing else OPER_PKL
    print(f"  building {pkl}")
    MI = _build_pkl(pkl, a.t_fore)
    t_fore = MI["sim_time_1"]
    print(f"    forecast {t_fore:%Y-%m-%d %H}Z .. {MI['sim_time_2']:%Y-%m-%d %H}Z")

    # ---- 3. sandbox --------------------------------------------------------
    if a.testing:
        print("  redirecting into the testing sandbox")
        n = SB.apply_testing_paths(MI, verbose=False)
        # Tag the OUTPUT names, not just the paths.  A sandboxed path is not
        # isolation on its own: on 2026-10-17 RSTNAME pointed at a sandbox path
        # that was a SYMLINK to the operational restart of the same name, and
        # ROMS wrote through it into the operational file.  A tagged name cannot
        # collide with an operational one, so no symlink can match it.
        SB.tag_testing_output_names(MI)
        made = SB.ensure_sandbox_dirs(MI)
        print(f"    {n} path(s) redirected, {made} directory/ies created")
        # The ocean starts from the DA ICs, but SWAN is initialised separately
        # and still wants its hotstart -- an empty sandbox restart dir makes the
        # generator fall back to ZERO and cold-start from a flat sea.
        SB.link_restarts_into_sandbox(MI, t_fore)
        # ROMS reads VARNAME from each level's Run dir; an empty sandbox Run dir
        # makes it abort in initialisation and still exit COMPLETED.
        SB.stage_static_run_inputs(MI)
        SB.assert_not_operational(MI)        # raises, deliberately
        print("    assert_not_operational: clean")

    # ---- 4. shorten, for plumbing tests ------------------------------------
    if a.length:
        MI["forecast_days"] = a.length
        MI["sim_time_2"] = t_fore + DT.timedelta(days=a.length)
        for lv in ("L1", "L2", "L3", "L4"):
            if ("tinfo" in MI) and ((lv, "forecast_days") in MI["tinfo"]):
                MI["tinfo"][lv, "forecast_days"] = a.length
        print(f"  forecast shortened to {a.length} day(s) -> "
              f"{MI['sim_time_2']:%Y-%m-%d %H}Z")

    # ---- 5. the DA initial conditions --------------------------------------
    if a.use_da_ics:
        levels = tuple(a.level) if a.level else ICH.LEVELS
        found = ICH.detect_da_ics(t_fore, levels=levels)
        print(found.text())
        ICH.apply_da_ics(MI, t_fore, found, levels=levels)
    else:
        print("  no_DA: every level keeps its restart settings")

    # ---- 6. save, re-check, launch -----------------------------------------
    if a.testing:
        SB.assert_not_operational(MI)        # again: step 5 copied files around
    with open(pkl, "wb") as fh:
        pickle.dump(MI, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  wrote {pkl}")

    only4 = a.level == ["LV4"]
    drv = "driver_run_forecast_LV4_only.py" if only4 else "driver_run_forecast_LV1234.py"
    cmd = [C.PY, "-u", "-W", "ignore", drv]
    if not only4:
        cmd.append("driver_run_forecast_LV1234")
    cmd.append(str(pkl))
    if a.dry_run:
        print("  DRY RUN -- would now launch:")
        print("    $ " + " ".join(str(c) for c in cmd) + f"   (cwd {DRIVER_DIR})")
        return 0
    rc = _run(cmd, cwd=DRIVER_DIR)

    # Archive the histories.  PFM's own copier (end_of_sim_housekeeping) never
    # runs -- driver_run_forecast_LV1234 hardcodes use_FF = 1 and uses the
    # shell script instead, which covers operational but has no testing
    # equivalent -- so without this the testing histories only ever existed on
    # /scratch until the next purge.  After the forecast, so a level that
    # failed simply has nothing to copy.
    if a.testing:
        n = SB.archive_testing_histories(MI)
        print(f"  archived {n}/4 testing history file(s) to "
              f"{SB.SANDBOX_ARCHIVE}/LV*_His/")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
