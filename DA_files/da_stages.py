"""
da_stages.py — one function per stage of the PFM DA pre-step.

Every stage has the same shape, which is what makes a ~4.9 h pipeline
survivable overnight:

    1. if the output already exists AND validates -> SKIPPED
    2. produce it
    3. validate the CONTENT, not just the existence
    4. copy the keeper to the archive and verify that copy by size -- on the
       SKIP path as well, via ensure_archived(), or an interrupted run can leave
       a valid scratch file that is never archived
    5. record status, wall time, both paths and the evidence in the manifest

Re-running after a failure therefore costs only the stages that did not finish,
and a keeper is safe in the archive the moment its stage ends -- not at the end
of a pipeline that may never reach its last stage.

Stage order and dependencies:

    lv1_ic ─┐
    lv1_atm ├─> lv1_da ─> increments ─> lv2_fwd ─> lv3_fwd ─> lv4_fwd ─> handoff
    lv1_bc  │                             (LV3 takes boundaries from LV2,
    lv1_obs ┘                              LV4 from LV3 -- strictly sequential)
"""
from __future__ import annotations

import datetime as DT
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import da_config as C
import da_pfm_bridge as B
import da_preflight as PF
import da_validate as V
from da_manifest import Manifest

sys.path.insert(0, str(C.PYDIR))

STAGES = ("preflight", "lv1_ic", "lv1_atm", "lv1_bc", "lv1_obs", "lv1_da",
          "increments", "lv2_fwd", "lv3_fwd", "lv4_fwd", "handoff")


@dataclass
class Ctx:
    t_fore: DT.datetime
    t_da: DT.datetime
    man: Manifest
    deadline: DT.datetime
    dry_run: bool = False
    allow_download: bool = True

    @property
    def yda(self) -> str:
        return C.ymd(self.t_da)

    @property
    def yfore(self) -> str:
        return C.ymd(self.t_fore)

    def over_deadline(self) -> bool:
        return DT.datetime.now() > self.deadline

    def left_h(self) -> float:
        return (self.deadline - DT.datetime.now()).total_seconds() / 3600.0


class StageSkip(Exception):
    """Raised by a stage that finds its output already valid."""


def _run(cmd: list[str], log: Path | None = None, cwd: Path | None = None,
         timeout: float | None = None) -> int:
    """Run a command, tee-ing to a log.  Returns the exit code (never raises on
    a non-zero rc -- ROMS success is judged from outputs, not rc)."""
    log = Path(log) if log else None
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
    with (open(log, "a") if log else subprocess.DEVNULL) as fh:
        p = subprocess.run([str(c) for c in cmd],
                           stdout=fh if log else subprocess.DEVNULL,
                           stderr=subprocess.STDOUT,
                           cwd=str(cwd) if cwd else None, timeout=timeout)
    return p.returncode


def _sbatch_wait(sb: Path, exports: dict, cwd: Path,
                 poll_s: int = 30, timeout_s: float | None = None) -> tuple[int, str]:
    """Submit with --wait and return (rc, jobid).

    --wait blocks, so the rc is the job's.  It is reported but NOT used as the
    success criterion -- see da_validate.roms_job_ok for why.
    """
    exp = "ALL," + ",".join(f"{k}={v}" for k, v in exports.items())
    out = subprocess.run(["sbatch", "--parsable", "--export=" + exp, str(sb)],
                         capture_output=True, text=True, cwd=str(cwd))
    if out.returncode != 0:
        raise RuntimeError(f"sbatch failed: {out.stderr.strip()}")
    job = out.stdout.strip().split(";")[0]
    t0 = time.time()
    while True:
        q = subprocess.run(["squeue", "-j", job, "-h", "-o", "%T"],
                           capture_output=True, text=True)
        if not q.stdout.strip():
            break
        if timeout_s and time.time() - t0 > timeout_s:
            subprocess.run(["scancel", job])
            raise TimeoutError(f"job {job} exceeded {timeout_s/3600:.1f} h; cancelled")
        time.sleep(poll_s)
    rc = 0
    acct = subprocess.run(["sacct", "-j", job, "-n", "-X", "-o", "ExitCode"],
                          capture_output=True, text=True).stdout.strip()
    if acct:
        try:
            rc = int(acct.split(":")[0])
        except ValueError:
            rc = 0
    return rc, job


# ---------------------------------------------------------------------------
# stage 0: preflight -- do we have every input, before spending 3 h of 4D-Var?
# ---------------------------------------------------------------------------
def stage_preflight(ctx: Ctx) -> dict:
    """Check every input up front and fail in seconds rather than hours.

    Two outcomes are distinguished, because they need different responses:
      * MISSING AND FIXABLE (ECMWF / HYCOM gaps) -- the owning stage downloads
        just the gap.  Reported here so it is not a surprise later.
      * MISSING AND NOT FIXABLE (no LV1 history covering t_da) -- nothing can be
        downloaded, so stop now.
    """
    pkl = _minfo(ctx)
    rep = PF.check_inputs(ctx.t_fore, ctx.t_da, pkl)
    print(rep.text())
    if rep.blocking:
        raise FileNotFoundError(
            "preflight: input(s) missing that NO download can fix: "
            + "; ".join(f"{i.name} -- {i.detail}" for i in rep.blocking))
    if rep.downloadable:
        if not ctx.allow_download:
            raise FileNotFoundError(
                "preflight: input(s) missing and downloading is disabled "
                "(--no-download): "
                + "; ".join(f"{i.name} -- {i.detail}" for i in rep.downloadable))
        print(f"    {len(rep.downloadable)} input(s) incomplete on disk; the "
              f"owning stage will fetch only the gap")
    return {}


# ---------------------------------------------------------------------------
# stage 1: LV1 initial condition, from a PFM history covering t_da
# ---------------------------------------------------------------------------
def stage_lv1_ic(ctx: Ctx) -> dict:
    out = C.scratch_dir("LV1", "Forc") / f"LV1_DA_IC_{ctx.yda}.nc"
    chk = V.nc_ok(out, need_vars=("zeta", "u", "v", "temp", "salt"),
                  expect_time=ctx.t_da, min_bytes=50 << 20)
    if chk:
        # archive on the skip path too -- see da_validate.ensure_archived
        V.ensure_archived(out, C.ARCHIVE / "IC_out")
        raise StageSkip(str(out))
    # The IC does NOT have to come from the forecast initialised at t_da.  A PFM
    # LV1 history holds 121 hourly records spanning 5 days, so the state at t_da
    # appears in up to five different runs; find_lv1_history searches them
    # freshest-first (least forecast drift) and verifies the record is really
    # there rather than trusting the filename.  This matters exactly when a PFM
    # cycle has failed and been rerun off-schedule -- which is when PFM_DA is
    # being asked to cover the hole, and when an exact-filename lookup would
    # fail.
    src, lead_h, note = PF.find_lv1_history(ctx.t_da)
    if src is None:
        raise FileNotFoundError(
            f"no PFM LV1 history within {PF.LV1_HIS_SPAN_DAYS} days holds a "
            f"record at {ctx.t_da:%Y-%m-%d %H}Z.  Nothing can be downloaded to "
            f"fix this -- a PFM forecast whose window contains t_da has to "
            f"exist.  Searched {[str(d) for d in C.LV1_HIS_DIRS]}")
    print(f"    IC from {note}")
    if lead_h > 0:
        print(f"    NOTE: this is a {lead_h} h forecast of the t_da state, not an "
              f"analysis at t_da -- the run initialised at t_da is missing.")
    if ctx.dry_run:
        return {"would_run": f"his_to_ic.py --his {src.name} --time {ctx.t_da:%Y-%m-%dT%H:%M:%S}"}
    rc = _run([C.PY, C.PYDIR / "his_to_ic.py",
               "--his", src, "--time", f"{ctx.t_da:%Y-%m-%dT%H:%M:%S}",
               "--out", out, "--sanitize"],
              log=C.SCRATCH / "Log" / f"{ctx.yfore}_lv1_ic.log")
    chk = V.nc_ok(out, need_vars=("zeta", "u", "v", "temp", "salt"),
                  expect_time=ctx.t_da, min_bytes=50 << 20,
                  finite_vars=("zeta", "temp", "salt"),
                  water_range={"temp": (-3.0, 40.0), "salt": (0.0, 45.0),
                               "zeta": (-5.0, 5.0)})
    if not chk:
        raise RuntimeError(f"LV1 IC invalid: {chk.why} (rc={rc})")
    arch = V.ensure_archived(out, C.ARCHIVE / "IC_out")
    return {"lv1_ic": {"scratch": out, "archive": arch}}


# ---------------------------------------------------------------------------
# stages 2-3: LV1 atm and ocean BC, over the DA window.
#
# Both reuse PFM's OWN builders rather than reimplementing or porting ours, via
# a DA-private model-info pickle -- see da_pfm_bridge for why that one change is
# enough to isolate us from the operational forecast.
#
# Why PFM's and not ours, concretely:
#   * atm: get_atm_data_on_roms_grid rotates earth -> ROMS XI/ETA with
#     angr = RMG['angle'], and ECMWF is lat/lon so its u/v are earth-relative --
#     that single stage is the whole rotation, and it is correct.
#   * BC: ocnr_2_BCdict_1hrzeta_from_tmppkls converts in-situ -> potential
#     temperature with gsw.pt_from_t (ocn_funs_forecast.py:3551-3557).  The
#     adjacent commented seawater.ptmp lines are the superseded EOS-80 version,
#     7 commented against 7 active -- not a missing conversion, which is how it
#     first looks on a grep.
#
# The acceptance test for both: day 1 of the result matches the operational
# LV1_ATM_FORCING.nc / LV1_OCEAN_BC.nc to round-off.  Matching proves the adapter
# changed the window and the paths and nothing else.  `--check-vs-operational`
# runs that comparison.
# ---------------------------------------------------------------------------
def _minfo(ctx: Ctx) -> Path:
    """The DA model-info pickle, built once per run and reused by both stages."""
    pkl = B.build_minfo(ctx.t_fore, ctx.t_da)
    B.assert_not_operational(pkl)
    return pkl


def stage_lv1_atm(ctx: Ctx) -> dict:
    out = C.scratch_dir("LV1", "Forc") / f"LV1_DA_ATM_{ctx.yda}.nc"
    need = ("Uwind", "Vwind", "Tair", "rain")
    chk = V.nc_ok(out, need_vars=need, min_bytes=10 << 20)
    if chk:
        V.ensure_archived(out, C.ARCHIVE / "Log" / "forcing")
        raise StageSkip(str(out))

    pkl = _minfo(ctx)
    if ctx.dry_run:
        return {"would_run": f"atm_functions.py x3 with {pkl.name}"}

    # PFM's own three steps, in the driver's order.  get_atm_data_as_dict walks
    # back through earlier ECMWF inits until got_ecmwf_files() says the window is
    # covered, so the closed DA window resolves to the init at t_da -- i.e. the
    # PREVIOUS day's 00Z run -- at leads 0..24.
    #
    # Do NOT "improve" this to the freshest init.  The forcing the ocean actually
    # experienced over [t_da, t_fore] is what yesterday's forecast ran with,
    # which is exactly that init at leads 0..24.  The fresher init would be a
    # better estimate of the weather and the wrong forcing for this analysis: the
    # increment would then correct for a wind the model never felt.  (The two
    # differ by ECMWF's 24 h error -- measured 2026-10-06 on the one shared
    # record: up to 5.9 m/s in Uwind, 3.3 C in Tair, 212 W/m2 in swrad.  Large
    # enough that confusing them would matter, and large enough that it is NOT
    # the test of whether this adapter works.)
    B.run_pfm("atm_functions.py", "get_atm_data_as_dict", pkl, timeout=3600)
    B.run_pfm("atm_functions.py", "get_atm_data_on_roms_grid", "1", pkl)
    B.run_pfm("atm_functions.py", "atm_roms_dict_to_netcdf", "1", pkl)

    # PFM names the file from the dict; we asked for the DA name, so this is
    # where that is confirmed rather than assumed.
    if not out.exists():
        raise RuntimeError(
            f"PFM's atm builder did not produce {out.name}.  It writes "
            f"lv1_forc_dir + lv1_atm_file from the model-info dict, so a "
            f"different name here means build_minfo and this stage disagree.")

    # NOTE: no wind_coordinate_frame attribute check.  That attribute is written
    # by OUR make_atm_from_nam.py, not by PFM -- the operational
    # LV1_ATM_FORCING.nc carries only type/time info/coordinates.  Requiring it
    # here rejected a perfectly good file.  Rotation is instead established by
    # the same-init comparison against the operational builder; see
    # check_atm_vs_operational().
    chk = V.nc_ok(out, need_vars=need, min_bytes=10 << 20,
                  finite_vars=("Uwind", "Vwind", "Tair", "Pair"),
                  water_range={"Tair": (-20.0, 55.0), "Uwind": (-60.0, 60.0),
                               "Vwind": (-60.0, 60.0)})
    if not chk:
        raise RuntimeError(f"LV1 DA atm invalid: {chk.why}")
    _assert_covers_window(out, ctx, "wind_time")
    arch = V.ensure_archived(out, C.ARCHIVE / "Log" / "forcing")
    return {"lv1_atm": {"scratch": out, "archive": arch}}


def stage_lv1_bc(ctx: Ctx) -> dict:
    out = C.scratch_dir("LV1", "Forc") / f"LV1_DA_BC_{ctx.yda}.nc"
    need = ("temp_south", "temp_north", "temp_west",
            "zeta_south", "zeta_north", "zeta_west")
    chk = V.nc_ok(out, need_vars=need, min_bytes=5 << 20)
    if chk:
        V.ensure_archived(out, C.ARCHIVE / "Log" / "forcing")
        raise StageSkip(str(out))

    pkl = _minfo(ctx)
    forc = C.scratch_dir("LV1", "Forc")
    if ctx.dry_run:
        return {"would_run": f"ocn_funs_forecast.py x5 with {pkl.name}"}

    # HYCOM comes off disk, never from the network.  The forecast has already
    # fetched this window, so hycom_init_for_window proves every file PFM will
    # ask for is present and picks the init -- rather than calling a fetch.
    #
    # The earlier version of this stage called get_hycom_hind_data, which
    # re-downloaded ~61 files we already had via ncks against tds.hycom.org.  That
    # was wrong twice over: it duplicated data on disk, and it made the pre-step
    # depend on an external service inside a 5 h budget.  The on-disk files use
    # the FORECAST naming (hy_<var>_<init>_<valid>.nc), which is why the hindcast
    # reader could not see them.
    init_day = B.hycom_init_for_window(ctx.t_da, ctx.t_fore, pkl,
                                       allow_download=ctx.allow_download)
    B.run_pfm("ocn_funs_forecast.py", "hycom_ncfiles_to_pickle", init_day, pkl,
              timeout=3600)

    # onto the ROMS grid, then the depth pickle the BC builder needs
    B.run_pfm("ocn_funs_forecast.py", "make_all_tmp_pckl_ocnR_files_1hrzeta", pkl,
              timeout=3600)
    B.run_pfm("ocn_funs_forecast.py", "make_rom_depths_1hrzeta",
              forc / "roms_tmp_depth_file.pkl", pkl)

    # the BC dict, then the NetCDF.  Note the second call takes the pickle and
    # the OUTPUT path but no pkl_fnm -- PFM's signature, not an omission.
    bc_pkl = forc / "ocnBC_tmp_pckl_file.pkl"
    B.run_pfm("ocn_funs_forecast.py", "ocnr_2_BCdict_1hrzeta_from_tmppkls",
              bc_pkl, pkl, timeout=3600)
    B.run_pfm("ocn_funs_forecast.py", "ocn_roms_BC_dict_to_netcdf_pckl_1hrzeta",
              bc_pkl, out)

    chk = V.nc_ok(out, need_vars=need, min_bytes=5 << 20,
                  finite_vars=("temp_south", "temp_west", "zeta_south"),
                  water_range={"temp_south": (-3.0, 40.0),
                               "temp_north": (-3.0, 40.0),
                               "temp_west": (-3.0, 40.0),
                               "zeta_south": (-5.0, 5.0)})
    if not chk:
        raise RuntimeError(f"LV1 DA BC invalid: {chk.why}")
    _assert_covers_window(out, ctx, "zeta_time")
    _assert_potential_temperature(out)
    arch = V.ensure_archived(out, C.ARCHIVE / "Log" / "forcing")
    return {"lv1_bc": {"scratch": out, "archive": arch}}


def _assert_covers_window(path: Path, ctx: Ctx, tvar: str) -> None:
    """The forcing must span [t_da, t_fore], or ROMS extrapolates silently.

    This is the check that catches the failure mode PFM's own code warns about:
    a stale derived pickle left in place builds forcing for the WRONG DAY, and
    the file is perfectly valid -- just not for this window.  Comparing the time
    axis against the window is the only thing that notices.
    """
    import netCDF4 as nc
    import numpy as np
    with nc.Dataset(path) as d:
        if tvar not in d.variables:
            cands = [v for v in d.variables if v.endswith("_time")]
            if not cands:
                return
            tvar = cands[0]
        tv = np.asarray(d[tvar][:]).ravel().astype(float)
        units = getattr(d[tvar], "units", "")
    if tv.size == 0:
        raise RuntimeError(f"{path.name}: {tvar} is empty")
    scale = 86400.0 if units.strip().lower().startswith("day") else 1.0
    t0 = V.ROMS_EPOCH + DT.timedelta(seconds=float(tv.min()) * scale)
    t1 = V.ROMS_EPOCH + DT.timedelta(seconds=float(tv.max()) * scale)
    # 1 h of slack: the atm axis is hourly and may start on the hour boundary
    slack = DT.timedelta(hours=1)
    if t0 > ctx.t_da + slack or t1 < ctx.t_fore - slack:
        raise RuntimeError(
            f"{path.name}: {tvar} spans {t0:%Y-%m-%d %H:%M}..{t1:%Y-%m-%d %H:%M}, "
            f"which does not cover the DA window "
            f"{ctx.t_da:%Y-%m-%d %H:%M}..{ctx.t_fore:%Y-%m-%d %H:%M}.  A stale "
            f"derived pickle builds valid forcing for the wrong day -- that is "
            f"PFM's 2026-08-31 failure, and only this check sees it.")


def _assert_potential_temperature(path: Path) -> None:
    """Sanity-check that the BC temperature was converted, not passed through.

    pt < t in situ at depth, so the deepest level of the western boundary should
    sit below the surface value by a small but non-zero amount.  This cannot
    prove gsw.pt_from_t ran, but it does catch the conversion being dropped
    wholesale -- the exact regression that reading a grep of commented
    seawater.ptmp lines would have talked us into.
    """
    import netCDF4 as nc
    import numpy as np
    with nc.Dataset(path) as d:
        if "temp_west" not in d.variables:
            return
        a = np.asarray(d["temp_west"][:], dtype=float)
    if a.ndim < 3:
        return
    a = np.where(np.abs(a) > V.FILL, np.nan, a)
    bot = np.nanmean(a[..., 0, :])      # s_rho index 0 = deepest
    top = np.nanmean(a[..., -1, :])
    if not np.isfinite(bot) or not np.isfinite(top):
        return
    if bot >= top:
        raise RuntimeError(
            f"{path.name}: temp_west is not colder at depth "
            f"(bottom {bot:.3f} C >= surface {top:.3f} C).  Expected stable "
            f"stratification on the open western boundary; check that the "
            f"in-situ -> potential temperature conversion ran.")


# ---------------------------------------------------------------------------
# stage 4: HF-radar observations
#
# The one input PFM_DA fetches itself.  PFM does not use HF radar, so nothing
# else brings it to disk -- downloading here is the normal path, unlike ECMWF and
# HYCOM where a download means the forecast cycle did not run.  It still fetches
# only the gap: cached hours are skipped.
#
# Two steps, deliberately separate:
#   1. fetch_hfr_nrt.py caches the window's 25 hourly surveys, writing them in
#      NCEI's retrospective LAYOUT so step 2 needs no changes at all.
#   2. make_obs_from_hfr.py does the science -- 3x3 stencil, Helmholtz
#      decomposition keeping the rotational part, coast and boundary culls,
#      1-in-2 thinning outside LV3, and the error ellipse.  That is the tuned
#      part and it is reused verbatim.
# ---------------------------------------------------------------------------
def stage_lv1_obs(ctx: Ctx) -> dict:
    out = C.scratch_dir("LV1", "Forc") / f"obs_HF_LV1_{ctx.yda}.nc"
    need = ("obs_type", "obs_value", "obs_error", "obs_Xgrid", "obs_Ygrid")
    chk = V.nc_ok(out, need_vars=need, min_bytes=1 << 20)
    if chk:
        V.ensure_archived(out, C.ARCHIVE / "Obs")
        raise StageSkip(str(out))

    if ctx.dry_run:
        return {"would_run": f"fetch_hfr_nrt.py + make_obs_from_hfr.py {ctx.yda}"}

    # -- 1. cache the surveys (gap only) --------------------------------------
    rep = PF.check_hfr(ctx.t_da, ctx.t_fore)
    if not rep.ok:
        if not ctx.allow_download:
            raise FileNotFoundError(
                f"HFR is short ({rep.detail}) and downloading is disabled "
                f"(--no-download).  Note PFM never fetches HFR, so with "
                f"--no-download a cycle can only use hours already cached.")
        rc = _run([C.PY, C.PYDIR / "fetch_hfr_nrt.py",
                   f"{ctx.t_da:%Y-%m-%d}", "--res", C.HFR_RES,
                   "--outdir", C.HFR_CACHE_NRT,
                   "--also-cache", C.HFR_CACHE_RETRO],
                  log=C.SCRATCH / "Log" / f"{ctx.yfore}_lv1_obs.log",
                  timeout=3600)
        # rc != 0 only means some hour is missing, which is survivable -- the
        # coverage test below is the real gate.
        rep = PF.check_hfr(ctx.t_da, ctx.t_fore)
        print(f"    HFR after fetch: {rep.detail} (fetch rc={rc})")

    have, need_n = rep.resolved["have"], rep.resolved["need"]
    if have < C.HFR_MIN_HOURS:
        raise RuntimeError(
            f"only {have}/{need_n} HFR surveys available for "
            f"{ctx.t_da:%Y-%m-%d}, below the {C.HFR_MIN_HOURS} needed.  A cycle "
            f"with too few observations produces an increment that is mostly "
            f"prior, which is worse than not cycling: check whether the HFR feed "
            f"is down (NDBC and CoastWatch both) before forcing this through.")

    # -- 2. build the ROMS obs file ------------------------------------------
    # Caches passed in precedence order: retrospective first, so reprocessed data
    # beats the near-real-time feed for any hour both hold.
    cache_args = []
    for c in C.HFR_CACHES:
        cache_args += ["--hfr-cache", str(c)]
    rc = _run([C.PY, C.PYDIR / "make_obs_from_hfr.py", f"{ctx.t_da:%Y-%m-%d}",
               "--outdir", C.scratch_dir("LV1", "Forc")] + cache_args,
              log=C.SCRATCH / "Log" / f"{ctx.yfore}_lv1_obs.log", timeout=3600)

    chk = V.nc_ok(out, need_vars=need, min_bytes=1 << 20,
                  finite_vars=("obs_value", "obs_error"))
    if not chk:
        raise RuntimeError(f"LV1 obs file invalid: {chk.why} (rc={rc})")
    _report_obs(out, have, need_n)
    arch = V.ensure_archived(out, C.ARCHIVE / "Obs")
    return {"lv1_obs": {"scratch": out, "archive": arch}}


def _report_obs(path: Path, have: int, need_n: int) -> None:
    """Print the obs count and error range -- the numbers that explain a cycle.

    Worth having in the log because when an analysis looks weak the first question
    is always whether it had observations, and the second is whether their errors
    were sane.  OBS_ERROR_VAR is the value at the ellipse centre and grows
    outward, so the minimum should sit at about that value.
    """
    try:
        import netCDF4 as nc_
        import numpy as np_
        with nc_.Dataset(path) as d:
            n = int(d["obs_value"].shape[0])
            err = np_.asarray(d["obs_error"][:], dtype=float)
            val = np_.asarray(d["obs_value"][:], dtype=float)
        print(f"    {n:,} obs from {have}/{need_n} surveys; "
              f"|value| max {np_.nanmax(np_.abs(val)):.3f} m/s; "
              f"obs_error {np_.nanmin(err):.4f}..{np_.nanmax(err):.4f} "
              f"(centre value is {C.OBS_ERROR_VAR})")
    except Exception as e:  # noqa: BLE001
        print(f"    (could not summarise obs: {type(e).__name__}: {e})")


# ---------------------------------------------------------------------------
# stage 5: the LV1 4D-Var cycle itself
# ---------------------------------------------------------------------------
def stage_lv1_da(ctx: Ctx) -> dict:
    his = C.scratch_dir("LV1", "His")
    outer0 = his / f"LV1_DA_outer0_{ctx.yda}.nc"
    outer1 = his / f"LV1_DA_outer1_{ctx.yda}.nc"
    dai = his / f"LV1_DA_dai_{ctx.yda}.nc"
    if all(V.nc_ok(p, min_bytes=100 << 20) for p in (outer0, outer1, dai)):
        raise StageSkip(f"{outer1.name} + prior + dai")
    raise NotImplementedError(
        "LV1 4D-Var submission not wired yet.  Submit slurm/slurm_4dvar.sb with "
        "ROMS_EXE=romsM_DA_IC (IC-only: BC adjustment blew up at the NW corner) "
        "and judge success with da_validate.roms_job_ok, NOT the slurm exit "
        "state -- these jobs routinely land in state FAILED at ~3h13 because "
        "MVAPICH2 raises SIGTERM in MPI_Finalize after the model is done.")


# ---------------------------------------------------------------------------
# stage 6: increments LV1 -> LV2/LV3/LV4, one date
# ---------------------------------------------------------------------------
def stage_increments(ctx: Ctx) -> dict:
    outs, need = {}, []
    for lv in ("LV2", "LV3", "LV4"):
        p = C.SCRATCH / "Inc" / f"INC_{lv}_{ctx.yda}.nc"
        if V.nc_ok(p, need_vars=("zeta", "u", "v", "temp", "salt"),
                   attrs={"blend_space": C.BLEND_SPACE}, min_bytes=10 << 20):
            outs[lv] = p
        else:
            need.append(lv)
    if not need:
        raise StageSkip(", ".join(p.name for p in outs.values()))
    raise NotImplementedError(
        f"increment build not wired yet (need {need}).  Submit "
        f"slurm/slurm_make_increment.sb per level with DATES={ctx.yda}..{ctx.yda}; "
        f"the increment MUST carry blend_space='{C.BLEND_SPACE}'.")


# ---------------------------------------------------------------------------
# stage 7: the child forward runs, strictly sequential down the nest
# ---------------------------------------------------------------------------
def _child_fwd(ctx: Ctx, lv: str, parent_his: Path | None) -> dict:
    out = C.scratch_dir(lv, "His") / f"{lv}_DA_his_{ctx.yda}.nc"
    if V.nc_ok(out, need_vars=("zeta", "temp"), min_bytes=100 << 20):
        raise StageSkip(str(out))
    if parent_his is not None and not parent_his.exists():
        raise FileNotFoundError(
            f"{lv} needs its parent's history first: {parent_his}")
    raise NotImplementedError(
        f"{lv} forward not wired yet.  Base IC = previous day's {lv} analysis at "
        f"t=24 h, else the PFM {lv} history at 00Z; add INC_{lv}_{ctx.yda}.nc "
        f"with apply_child_increment.py --grid {C.GRID[lv].name}; boundaries "
        f"from {'the ' + parent_his.name if parent_his else 'the LV1 analysis'}.")


# LV4 is not just "LV2 with a finer grid", and whoever writes stage 7 should know
# the two things that cost days the first time round:
#
#   1. Its per-day .in is GENERATED, not templated -- by stage_LV4_ins.py, which
#      is NOT vendored here (it still reads the roms_fourdvar LV4 tree).  Either
#      vendor it with its templates or build the .in inline.  LV2/LV3 get away
#      with a seed .in plus a DSTART sed; LV4 does not, because it is coupled.
#
#   2. SWAN's clock drifts ~1280 s per RESTART FREE hop when no interval is
#      given, so a chained day silently starts the wave model at the wrong time.
#      The fix is RESTART FREE 24 HR with HOTSTART pointed at the _024-suffixed
#      file.  Do not drop this: it is what blocked 2026-03-09 for weeks, and it
#      presents as a crash with no obvious wave cause.


def stage_lv2_fwd(ctx: Ctx) -> dict:
    return _child_fwd(ctx, "LV2", None)


def stage_lv3_fwd(ctx: Ctx) -> dict:
    return _child_fwd(ctx, "LV3",
                      C.scratch_dir("LV2", "His") / f"LV2_DA_his_{ctx.yda}.nc")


def stage_lv4_fwd(ctx: Ctx) -> dict:
    return _child_fwd(ctx, "LV4",
                      C.scratch_dir("LV3", "His") / f"LV3_DA_his_{ctx.yda}.nc")


# ---------------------------------------------------------------------------
# stage 8: the handoff -- extract t=24 h from each level
# ---------------------------------------------------------------------------
def stage_handoff(ctx: Ctx) -> dict:
    """The t=24 h state of each level IS the state at t_fore, which is exactly
    the IC the forecast needs.  Written to IC_out/ on scratch for the forecast
    to read, and to the archive for the record.

    Note for Phase 2: these are SINGLE-RECORD files, so each level needs
    NRREC=0 rather than the restart path's NRREC!=0.  That also means
    ini_fields recomputes ubar from u3d -- which is what we want, but it is a
    behaviour change from restarting.
    """
    raise NotImplementedError(
        "handoff not wired yet.  For each level, his_to_ic.py --time "
        f"{ctx.t_fore:%Y-%m-%dT%H:%M:%S} on that level's DA history -> "
        "IC_out/LV{n}_DA_IC_for_<t_fore>.nc, validate, archive, and record with "
        "man.set_ic(). LV4 must carry dye_01/dye_02.")


FUNCS = {
    "preflight": stage_preflight,
    "lv1_ic": stage_lv1_ic, "lv1_atm": stage_lv1_atm, "lv1_bc": stage_lv1_bc,
    "lv1_obs": stage_lv1_obs, "lv1_da": stage_lv1_da,
    "increments": stage_increments, "lv2_fwd": stage_lv2_fwd,
    "lv3_fwd": stage_lv3_fwd, "lv4_fwd": stage_lv4_fwd, "handoff": stage_handoff,
}


def purge_scratch(retain_days: int = C.SCRATCH_RETAIN_DAYS) -> int:
    """Age out Forc/ and His/ -- never Run/, which holds the executable, the .in
    templates and the NRM cache.  Runs FIRST, not last, so a crashed cycle is
    cleaned up on the next night instead of accumulating unnoticed.
    """
    n = 0
    cutoff = time.time() - retain_days * 86400
    for lv in C.LEVELS:
        for sub in C.SCRATCH_PURGE_SUBDIRS:
            d = C.SCRATCH / lv / sub
            if not d.is_dir():
                continue
            for f in d.iterdir():
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
                    n += 1
    return n
