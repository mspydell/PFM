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
import re
import shutil
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
    cold_lv1: bool = False

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
    # a node count in da_config that disagrees with the slurm script that actually
    # allocates is not fatal, but it makes every scheduling estimate wrong
    for msg in C.check_nodes_match_slurm():
        print(f"    WARNING: {msg}")
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
    # WARM FIRST, exactly as the children do in _child_base_ic.  The previous
    # cycle's handoff IC IS its analysis at t = 24 h, and that cycle's t = 24 h
    # is this cycle's t_da -- so LV1 continues from the DA system's own state
    # instead of being yanked back onto the operational forecast every night.
    # Without this, LV1 cold-started from PFM every cycle while LV2/LV3/LV4
    # warm-started from the DA, which is an inconsistency, not a choice: the
    # children would be descended from an LV1 analysis their own parent had
    # already discarded.
    #
    # The file is already a sanitised single record at t_da, written by
    # stage_handoff through the same his_to_ic.py, so it is copied rather than
    # rebuilt.  Verified 2026-10-08 on the 10/06 cycle: bit-identical to
    # ocean_fwd_outer1.nc's t = 24 h record, max|dzeta| = 0 over 81,217 cells.
    warm = C.SCRATCH / "IC_out" / f"LV1_DA_IC_for_{ctx.t_da:%Y%m%d%H}.nc"
    if not warm.is_file():
        warm = C.ARCHIVE / "IC_out" / warm.name
    warm_ok = V.nc_ok(warm, need_vars=("zeta", "u", "v", "temp", "salt"),
                      expect_time=ctx.t_da, min_bytes=50 << 20)
    if warm_ok and not ctx.cold_lv1:
        print(f"    IC WARM from the previous DA cycle's analysis at t=24h "
              f"({warm.name})")
        if ctx.dry_run:
            return {"would_run": f"copy {warm.name} -> {out.name}"}
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(warm, out)
        chk = V.nc_ok(out, need_vars=("zeta", "u", "v", "temp", "salt"),
                      expect_time=ctx.t_da, min_bytes=50 << 20,
                      finite_vars=("zeta", "temp", "salt"),
                      water_range={"temp": (-3.0, 40.0), "salt": (0.0, 45.0),
                                   "zeta": (-5.0, 5.0)})
        if not chk:
            raise RuntimeError(f"LV1 warm IC invalid: {chk.why}")
        arch = V.ensure_archived(out, C.ARCHIVE / "IC_out")
        return {"lv1_ic": {"scratch": out, "archive": arch, "warm": True}}
    if ctx.cold_lv1 and warm_ok:
        print(f"    IC: --cold-lv1 given, ignoring the available warm start "
              f"({warm.name})")

    # COLD.  The IC does NOT have to come from the forecast initialised at t_da.
    # A PFM LV1 history holds 121 hourly records spanning 5 days, so the state at
    # t_da appears in up to five different runs; find_lv1_history searches them
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
    print(f"    IC COLD from PFM's operational history: {note}")
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
    # ARCHIVE FIRST.  An LV1 atm file archived by an earlier run is already the
    # forcing for this window, on the LV1 grid with the winds rotated, so the
    # whole chain below can be skipped.  This is the LAST-RESORT recovery path:
    # the raw grib cache sits one level deeper inside get_atm_data_as_dict and is
    # preferred, because from gribs this stage rebuilds exactly as it always has.
    #
    # It matters here more than in PFM.  The DA only ever runs on a PAST window,
    # so it is precisely the caller for which upstream has already dropped the
    # data -- pipeline keeps the current cycle, ecmwf ftp ~2-3 days, CDIP nothing
    # since July 2026.  On 2026-10-16 this stage failed for exactly that reason
    # and took the 10/14 cycle down with it.
    # ALWAYS FROM GRIBS.  get_atm_data_as_dict resolves them itself: local
    # scratch, then the raw grib archive, then the server.  That archive is what
    # makes a past window rerunnable at all -- upstream keeps nothing (pipeline
    # the current cycle, ecmwf ftp ~2-3 days, CDIP nothing since 2026/07), which
    # is why this stage failed on 2026-10-16 and took the 10/14 cycle with it.
    #
    # Do NOT short-circuit with an archived LV1_atm.  _child_atm relies on this
    # step having written lv1_forc_dir/atm_tmp_pckl_file.pkl, so skipping it
    # leaves LV2/3/4 on the previous cycle's pickle.  The DA would catch that in
    # _assert_covers_window, but PFM would not, and the fix belongs in one place.
    B.run_pfm("atm_functions.py", "get_atm_data_as_dict", pkl, timeout=3600)
    B.run_pfm("atm_functions.py", "get_atm_data_on_roms_grid", "1", pkl)
    B.run_pfm("atm_functions.py", "atm_roms_dict_to_netcdf", "1", pkl)

    if str(C.PFM_PYDIR) not in sys.path:
        sys.path.insert(0, str(C.PFM_PYDIR))
    import util_functions as UF
    UF.archive_lv1_atm(str(pkl))

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
# stage 5: the LV1 4D-Var cycle -- the expensive one, ~3.25 h on 9 nodes
# ---------------------------------------------------------------------------
# Staging and .in generation reuse the research chain's own shell scripts rather
# than a Python reimplementation.  set_up_DA_simulation_v2.sh is 205 lines of sed
# that encodes the entire DA configuration -- DSTART, the NRM and STD file names
# with their decorrelation scales, the obs-error ellipse, every path ROMS reads.
# Re-expressing that would be a silent-divergence machine; it is also already
# parameterised by environment variables, so PFM can drive it unchanged.
#
# PFM_DA is ONE cycle, not a chain, which removes the subtlest trap in the
# research driver: there, the IC had to come from the archived date-tagged
# ocean_dai because DAINAME is IO_DIR/ocean_dai.nc, and using that path for both
# would have ROMS open one file for read and write (the symptom was
# DEF_DAI_NF90 failing to create the DA restart file).  Here the IC always comes
# from stage 1, in LV1/Forc, so the collision cannot arise.
# ---------------------------------------------------------------------------
# Per-keeper minimum sizes.  These are NOT cosmetic: a single blanket floor
# rejected a perfectly good cycle.  roms_mod.nc holds one value per OBSERVATION,
# not a 3D state, so it is tens of MB where outer0/outer1 are ~6.8 GB.  A 100 MB
# floor applied to all three failed the 2026-10-06 cycle on roms_mod.nc at
# 29.0 MB -- against the research chain's 86 files spanning 18.7-30.3 MB, mean
# 26.3.  Floors are set to roughly a third of the observed size, low enough to
# pass a thin cycle and high enough to catch a truncated write.
_DA_KEEPERS = {
    # IO_DIR name          -> (archive subdir,  archived basename, min bytes)
    "ocean_fwd_outer0.nc": ("LV1_Prior",    "LV1_fwd_outer0_{ymd}.nc", 1 << 30),
    "ocean_fwd_outer1.nc": ("LV1_Analysis", "LV1_fwd_outer1_{ymd}.nc", 1 << 30),
    "roms_mod.nc":         ("LV1_Analysis", "roms_mod_{ymd}.nc",       6 << 20),
    "ocean_dai.nc":        ("LV1_Dai",      "ocean_dai_{ymd}.nc",     50 << 20),
}


def stage_lv1_da(ctx: Ctx) -> dict:
    """Run the 24 h RBL4DVAR cycle over [t_da, t_fore].

    outer0 is the prior, outer1 the analysis.  The increment the children need is
    outer1(t=0) - outer0(t=0), which is why BOTH are keepers.
    """
    io = C.io_dir()
    ymd = ctx.yda
    analysis = io / "ocean_fwd_outer1.nc"
    prior = io / "ocean_fwd_outer0.nc"

    # Idempotent skip: the prior and the analysis must be present, sane, AND for
    # THIS window.
    #
    # expect_time is the whole point of this check and was missing.  These two
    # files live in tmp_input_output under fixed, UN-DATED names, and that
    # directory is not in SCRATCH_PURGE_SUBDIRS -- so the previous cycle's
    # outer0/outer1 are still sitting there when the next cycle starts.  Without
    # a date test the skip fired on yesterday's files and the whole chain
    # silently downscaled YESTERDAY's analysis onto today's children, producing a
    # complete, plausible, wrong result.  Caught on the 2026-10-07 dry run, which
    # reported "[lv1_da] skipped -- output already valid" before the 10/07 cycle
    # had been run at all.
    #
    # record 0 of outer0/outer1 is t_da, so this is the correct record to test.
    if (V.nc_ok(analysis, need_vars=("zeta", "u", "v", "temp", "salt"),
                min_bytes=100 << 20, expect_time=ctx.t_da)
            and V.nc_ok(prior, need_vars=("zeta", "temp"), min_bytes=100 << 20,
                        expect_time=ctx.t_da)):
        for name, (sub, tmpl, _mb) in _DA_KEEPERS.items():
            f = io / name
            if f.is_file():
                V.ensure_archived(f, C.ARCHIVE / sub, tmpl.format(ymd=ymd),
                                  expect_time=ctx.t_da)
        raise StageSkip(str(analysis))

    rin = C.scratch_dir("LV1", "Run") / f"LV1_run_4dvar_{ymd}.in"
    log = C.SCRATCH / "Log" / f"LV1_4dvar_{ymd}.log"
    if ctx.dry_run:
        # before the input checks, not after: in a dry run the earlier stages
        # produced nothing, so demanding their outputs would stop --dry-run at the
        # first stage that has a precondition and defeat the point of it.
        return {"would_run": f"sbatch slurm_4dvar.sb for {ymd} "
                             f"({C.NODES['lv1_da']} nodes, ~"
                             f"{C.EXPECTED_WALL_S['lv1_da']/3600:.2f} h)"}

    ic = C.scratch_dir("LV1", "Forc") / f"LV1_DA_IC_{ymd}.nc"
    bry = C.scratch_dir("LV1", "Forc") / f"LV1_DA_BC_{ymd}.nc"
    frc = C.scratch_dir("LV1", "Forc") / f"LV1_DA_ATM_{ymd}.nc"
    obs = C.scratch_dir("LV1", "Forc") / f"obs_HF_LV1_{ymd}.nc"
    for f in (ic, bry, frc, obs):
        if not f.is_file():
            raise FileNotFoundError(f"stage 5 needs {f.name} from an earlier "
                                    f"stage: {f}")
    exe = Path(C.ROMS_EXE_DA)
    if not (exe.is_file() and os.access(exe, os.X_OK)):
        raise FileNotFoundError(
            f"DA executable missing or not executable: {exe}.  It is built "
            f"outside this repo; copy it into {exe.parent}.")
    for d, what in ((C.NRM_DIR, "normalisation"), (C.STD_DIR, "background-error")):
        if not any(d.glob("*.nc")):
            raise FileNotFoundError(
                f"no {what} files in {d} -- the 4D-Var cannot run without them, "
                f"and they are hours of compute to rebuild.")

    # -- stage the inputs into IO_DIR ---------------------------------------
    env = dict(os.environ, IC_SRC=str(ic), BRY_SRC=str(bry), FRC_SRC=str(frc),
               OBS_SRC=str(obs), IO_DIR=str(io))
    r = subprocess.run([str(C.SCRIPTSDIR / "stage_cycle_inputs.sh")],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"stage_cycle_inputs.sh failed: {r.stderr.strip()}")
    staged = [s for s in r.stdout.strip().splitlines() if s.strip()]
    if len(staged) != 4:
        raise RuntimeError(
            f"stage_cycle_inputs.sh printed {len(staged)} destination(s), "
            f"expected 4 (IC, BRY, FRC, OBS): {staged}")
    ic_s, bry_s, frc_s, obs_s = staged

    # -- generate the .in ---------------------------------------------------
    env = dict(os.environ,
               START_DATE=f"{ctx.t_da:%Y-%m-%d}",
               IC_FILE=ic_s, BRY_FILE=bry_s, FRC_FILE=frc_s, OBS_FILE=obs_s,
               RUN_DIR=str(C.scratch_dir("LV1", "Run")), IO_DIR=str(io),
               RIN_TEMPLATE=str(C.RUN_IN / "LV1_run_4dvar.in"),
               RIN_OUT=str(rin),
               NRM_DIR=str(C.NRM_DIR), STD_DIR=str(C.STD_DIR),
               GRID_TAG="LV1",
               HDECAY_IC_KM=str(C.HDECAY_IC_KM),
               HDECAY_BC_KM=str(C.HDECAY_BC_KM),
               HDECAY_F_KM=str(C.HDECAY_F_KM),
               URMS_TAG=C.URMS_TAG, WSRMS_TAG=C.WSRMS_TAG)
    r = subprocess.run([str(C.SCRIPTSDIR / "set_up_DA_simulation_v2.sh")],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"set_up_DA_simulation_v2.sh failed: "
                           f"{(r.stderr or r.stdout).strip()[-800:]}")
    if not rin.is_file():
        raise RuntimeError(f"the .in was not produced: {rin}")
    _assert_in_consistent(rin, ctx, (ic_s, bry_s, frc_s, obs_s))

    # Defeat a stale Lustre attribute cache: ROMS has opened an input whose size
    # the client still believed was the previous cycle's.
    for f in (ic_s, bry_s, frc_s, obs_s, str(rin)):
        try:
            os.utime(f, None)
        except OSError:
            pass

    # -- submit -------------------------------------------------------------
    t0 = time.time()
    rc, job = _sbatch_wait(C.SLURMDIR / "slurm_4dvar.sb",
                           {"ROMS_EXE": str(exe), "RIN": str(rin),
                            "LOG": str(log)},
                           cwd=C.scratch_dir("LV1", "Run"),
                           timeout_s=ctx.left_h() * 3600)
    # each keeper against its own floor -- see _DA_KEEPERS
    for name in ("ocean_fwd_outer0.nc", "ocean_fwd_outer1.nc", "roms_mod.nc"):
        chk = V.roms_job_ok(log, (io / name,), t0, rc,
                            min_bytes=_DA_KEEPERS[name][2])
        if not chk:
            raise RuntimeError(f"LV1 4D-Var failed (job {job}): {chk.why}.  "
                               f"Log: {log}")
    print(f"    job {job} accepted: {chk.why}")

    # RBL4DVAR needs NQCK>0; without it the failure presents as a SIGSEGV rather
    # than a clear message, so grep for the real marker.
    _warn_on_roms_errors(log)

    out = {}
    for name, (sub, tmpl, _mb) in _DA_KEEPERS.items():
        f = io / name
        if not f.is_file():
            if name == "ocean_dai.nc":
                print(f"    note: {name} absent -- expected only if this build "
                      f"does not write a DA restart")
                continue
            raise RuntimeError(f"keeper missing after a successful job: {f}")
        arch = V.ensure_archived(f, C.ARCHIVE / sub, tmpl.format(ymd=ymd),
                                 expect_time=ctx.t_da)
        out[name] = {"scratch": f, "archive": arch}
    V.ensure_archived(log, C.ARCHIVE / "Log")
    return out


def _assert_in_consistent(rin: Path, ctx: Ctx, staged: tuple) -> None:
    """Check the generated .in really describes THIS cycle, before 3 h of compute.

    The .in is produced by sed substitution, so a template change or an unmatched
    pattern yields a file that is perfectly valid ROMS input for the WRONG cycle,
    and the only symptom is an analysis that quietly disagrees with its
    observations.

    The inputs are split across TWO files and this has to follow that: the main
    .in carries the IC, boundary and forcing paths, while the OBSERVATION path
    lives in the assimilation-parameter file that the main .in names in APARNAM.
    An earlier version of this check looked for the obs file in the main .in and
    reported a false failure.  Following the link also tests the link, which is
    the worse failure of the two -- a stale APARNAM pointing at another day's
    s4dvar file would assimilate the wrong observations with nothing to show it.
    """
    txt = rin.read_text()
    ic_s, bry_s, frc_s, obs_s = staged

    for f in (ic_s, bry_s, frc_s):
        if Path(f).name not in txt:
            raise RuntimeError(
                f"{rin.name} does not reference {Path(f).name}: the template "
                f"substitution did not take.  Running this would do a valid "
                f"4D-Var with the wrong inputs.")

    m = re.search(r"^\s*APARNAM\s*=\s*(\S+)", txt, re.M)
    if not m:
        raise RuntimeError(f"{rin.name} has no APARNAM line, so the "
                           f"assimilation parameters (and the observation file) "
                           f"are unset.")
    apar = Path(m.group(1))
    if not apar.is_file():
        raise RuntimeError(f"{rin.name} points APARNAM at {apar}, which does "
                           f"not exist.")
    if Path(obs_s).name not in apar.read_text():
        raise RuntimeError(
            f"{apar.name} (APARNAM of {rin.name}) does not reference "
            f"{Path(obs_s).name} -- this cycle would assimilate a different "
            f"day's observations.")

    # DSTART is days since the ROMS epoch; confirm it matches t_da
    m = re.search(r"^\s*DSTART\s*=\s*([0-9.eEdD+-]+)", txt, re.M)
    if m:
        want = (ctx.t_da - V.ROMS_EPOCH).total_seconds() / 86400.0
        got = float(m.group(1).lower().replace("d", "e"))
        if abs(got - want) > 1e-6:
            raise RuntimeError(
                f"{rin.name}: DSTART={got} but t_da={ctx.t_da:%Y-%m-%d} is "
                f"{want} days after the ROMS epoch -- the .in is for a "
                f"different day.")


def _warn_on_roms_errors(log: Path) -> None:
    """Surface ROMS's own error lines even when the outputs look fine."""
    try:
        txt = log.read_text(errors="ignore")
    except OSError:
        return
    hits = [l.strip() for l in txt.splitlines() if "Found Error:" in l]
    if hits:
        print(f"    WARNING: {len(hits)} 'Found Error:' line(s) in the ROMS log, "
              f"first: {hits[0][:110]}")


# ---------------------------------------------------------------------------
# stage 6: downscale the LV1 increment onto LV2 / LV3 / LV4
# ---------------------------------------------------------------------------
def _lv1_increment_root(ctx: Ctx) -> Path:
    """A directory shaped the way make_child_increment expects, pointing at ours.

    It looks for the increment source as
        <root>/analysis_files/LV1_fwd_outer1_<date>.nc
      - <root>/prior_files/LV1_fwd_outer0_<date>.nc
    whereas our archive uses LV1_Analysis/ and LV1_Prior/.  Rather than rename
    the archive or teach the downscaler a second layout, this builds a directory
    of SYMLINKS with the names it wants.

    The links point at IO_DIR, not at the archive: those are the same files, on
    BeeGFS rather than /dataSIO, and the downscaler reads both 6.8 GB files in
    full -- so this avoids pulling 13.6 GB back off /dataSIO whose throughput is
    erratic.
    """
    io = C.io_dir()
    root = C.SCRATCH / "Inc" / "lv1_src"
    pairs = (("analysis_files", f"LV1_fwd_outer1_{ctx.yda}.nc",
              io / "ocean_fwd_outer1.nc"),
             ("prior_files", f"LV1_fwd_outer0_{ctx.yda}.nc",
              io / "ocean_fwd_outer0.nc"))
    for sub, name, src in pairs:
        if not src.is_file():
            raise FileNotFoundError(
                f"stage 6 needs the LV1 {sub.split('_')[0]} from stage 5: {src}")
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        link = d / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src)
    return root


def _child_vparams_source(lv: str) -> Path:
    """A file carrying this level's vertical coordinate.

    Only Vtransform/hc/Cs_r/s_rho are read, and those are static per level.
    Verified 2026-10-07 that PFM's operational child histories match the research
    chain's fix_v* files exactly for LV2, LV3 and LV4, so PFM's own are used --
    which keeps stage 6 off the research tree.
    """
    hits = sorted(C.PFM_HIS[lv].glob(f"{lv}_ocean_his_*.nc"))
    if not hits:
        raise FileNotFoundError(
            f"no PFM {lv} history in {C.PFM_HIS[lv]} to read vertical "
            f"parameters from")
    return hits[-1]


def stage_increments(ctx: Ctx) -> dict:
    """One objective-mapping run per child level, for this one date.

    blend_space='D' is asserted on the OUTPUT, not just passed in: blending the
    bathymetry h instead of the water column D needs a Dcrit clip that inverts the
    parent column at a deep low tide, and the symptom is a temperature-gradient
    blow-up two levels down rather than anything visible here.
    """
    outs, need = {}, []
    for lv in ("LV2", "LV3", "LV4"):
        p = C.SCRATCH / "Inc" / f"INC_{lv}_{ctx.yda}.nc"
        if V.nc_ok(p, need_vars=("zeta", "u", "v", "temp", "salt"),
                   attrs={"blend_space": C.BLEND_SPACE}, min_bytes=10 << 20):
            outs[lv] = p
            V.ensure_archived(p, C.ARCHIVE / "Inc")
        else:
            need.append(lv)
    if not need:
        raise StageSkip(", ".join(p.name for p in outs.values()))
    if ctx.dry_run:
        return {"would_run": f"slurm_make_increment.sb for {need} on {ctx.yda}"}

    root = _lv1_increment_root(ctx)
    for lv in need:
        log = C.SCRATCH / "Log" / f"{ctx.yfore}_inc_{lv}.log"
        t0 = time.time()
        rc, job = _sbatch_wait(
            C.SLURMDIR / "slurm_make_increment.sb",
            {"LEVEL": lv, "DATES": f"{ctx.yda}..{ctx.yda}",
             "OUTDIR": str(C.SCRATCH / "Inc"),
             # ROOT and VPARAMS MUST be passed.  Their defaults in the slurm
             # script are the RESEARCH chain's paths, so omitting them would
             # build the increment from a different LV1 run entirely.
             "ROOT": str(root),
             "VPARAMS": str(_child_vparams_source(lv)),
             # slurm runs a COPY of the batch script from its spool directory, so
             # the script cannot find DA_files from its own path -- it must be
             # told.
             "DA_FILES": str(C.DA_FILES),
             "SKIP": "1"},
            cwd=C.SCRATCH / "Inc", timeout_s=ctx.left_h() * 3600)
        p = C.SCRATCH / "Inc" / f"INC_{lv}_{ctx.yda}.nc"
        chk = V.nc_ok(p, need_vars=("zeta", "u", "v", "temp", "salt"),
                      attrs={"blend_space": C.BLEND_SPACE}, min_bytes=10 << 20,
                      finite_vars=("zeta", "temp", "salt"))
        if not chk:
            raise RuntimeError(f"{lv} increment invalid (job {job}, rc={rc}): "
                               f"{chk.why}")
        print(f"    {lv} increment ok ({(time.time()-t0)/60:.1f} min, job {job})")
        outs[lv] = p
        V.ensure_archived(p, C.ARCHIVE / "Inc")
    return {lv: {"scratch": p, "archive": C.ARCHIVE / "Inc" / p.name}
            for lv, p in outs.items()}


# ---------------------------------------------------------------------------
# stage 7: the child forward runs, strictly sequential down the nest
#
# Each level, for the one DA day:
#     base IC  <- the previous night's DA history at t_da (warm) or PFM's
#                 operational history at t_da (cold)
#     IC       =  base + the pre-built increment
#     atm      <- PFM's ECMWF builder for THIS level
#     BC       <- the parent's history for this day
#     .in      <- the seed, with every path and the date substituted
#     run      <- sbatch, judged by outputs rather than exit code
#
# Warm-start is preferred so the DA system carries its own state forward, which is
# the point of cycling; cold-start from PFM's own forecast is the fallback when a
# night was missed.  Same preference as the research chain.
#
# ATM COMES FROM PFM'S ECMWF PATH, not from the research chain's
# make_atm_LV{2,3,4}.py.  Those are NAM-based, and PFM's NAM path applies no wind
# rotation at all (see PROVENANCE).  PFM's ECMWF builder rotates per level with
# that level's own grid angle, which is what these runs need, and it is the same
# forcing the operational child runs use.
# ---------------------------------------------------------------------------
_CHILD_IN_SUBS = ("VARNAME", "ININAME", "BRYNAME", "FRCNAME",
                  "HISNAME", "RSTNAME")


def _stage_child_in(lv: str, ctx: Ctx, paths: dict) -> Path:
    """Write the staged .in for one child level and one day.

    The research chain retags its seed with `sed s|20260307|<ymd>|g`, which works
    there because only the DATE differs between cycles.  For PFM every DIRECTORY
    differs too -- the seed names /dataSIO/DA_Simulations/... and
    /scratch/matt/... -- so each path is substituted explicitly by keyword.  A
    date-only retag would have produced a valid .in pointing at the research tree.

    Every substitution is verified to have taken, and every input the .in names is
    verified to exist, because the alternative is discovering it after a 216-rank
    allocation has started.
    """
    seed = C.RUN_IN / f"{lv}_oi_seed.in"
    if not seed.is_file():
        raise FileNotFoundError(f"no seed .in for {lv}: {seed}")
    out = C.scratch_dir(lv, "Run") / f"{lv}_DA_{ctx.yda}.in"
    if out.resolve() == seed.resolve():
        raise RuntimeError(f"refusing to overwrite the seed with the staged .in: "
                           f"{out}")
    txt = seed.read_text()

    dstart = (ctx.t_da - V.ROMS_EPOCH).total_seconds() / 86400.0
    subs = {
        "VARNAME": C.VARINFO[lv],
        "ININAME": paths["ic"],
        "BRYNAME": paths["bc"],
        "FRCNAME": paths["atm"],
        "HISNAME": paths["his"],
        "RSTNAME": C.scratch_dir(lv, "Run") / f"{lv}_DA_rst_{ctx.yda}.nc",
    }
    for key, val in subs.items():
        # '=' for VARNAME, '==' for the rest -- ROMS's own convention, and the
        # seed uses both.
        pat = rf"^(\s*{key}\s*==?\s*).*$"
        new, n = re.subn(pat, lambda m: m.group(1) + str(val), txt,
                         count=1, flags=re.M)
        if n != 1:
            raise RuntimeError(
                f"{seed.name}: could not substitute {key} ({n} matches).  The "
                f"seed template has changed shape; a silent miss here leaves the "
                f"run pointing at the research tree.")
        txt = new
    txt, n = re.subn(r"^(\s*DSTART\s*=).*$",
                     lambda m: f"{m.group(1)}  {dstart}d0"
                               f"                 ! {ctx.t_da:%Y-%m-%d} 00 UTC",
                     txt, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError(f"{seed.name}: could not set DSTART")

    # The seed also names a dozen files this configuration does not use --
    # CLMNAME, NUDNAME, SSFNAME, TIDENAME, DAINAME, GSTNAME, QCKNAME, AVGNAME,
    # DIANAME -- all pointed at a `Run/junk/` directory.  They are unused but ROMS
    # may still create some of them, so they are redirected into the DA tree and
    # the directory is created, rather than left pointing at the research tree.
    junk = C.scratch_dir(lv, "Run") / "junk"
    junk.mkdir(parents=True, exist_ok=True)
    txt = re.sub(r"(/[^\s]*?/)junk/([A-Za-z0-9_.]+\.nc)",
                 lambda m: f"{junk}/{m.group(2)}", txt)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt)

    # nothing from the research tree may survive in an active line
    bad = [l.strip() for l in txt.splitlines()
           if not l.lstrip().startswith("!")
           and ("/scratch/matt" in l or "DA_Simulations" in l)]
    if bad:
        raise RuntimeError(
            f"{out.name} still references the research tree on "
            f"{len(bad)} active line(s), first: {bad[0][:110]}")
    # and every input it names must exist
    for key in ("VARNAME", "ININAME", "BRYNAME", "FRCNAME"):
        f = Path(str(subs[key]))
        if not f.is_file():
            raise FileNotFoundError(
                f"{out.name}: {key} points at a missing file: {f}")
    return out


def _child_base_ic(ctx: Ctx, lv: str) -> tuple[Path, str]:
    """(source history, note) for the base state at t_da.

    Warm first: the previous night's DA history holds t_da as its t = 24 h record,
    so cycling continues from the DA system's own state.  Cold otherwise, from
    PFM's operational history for this day -- which is what the forecast itself
    started from, so a missed night costs continuity but not correctness.
    """
    prev = C.scratch_dir(lv, "His") / f"{lv}_DA_his_{C.ymd(ctx.t_da - DT.timedelta(days=1))}.nc"
    if prev.is_file():
        return prev, f"WARM from the previous DA day at t=24h ({prev.name})"
    cold = C.PFM_HIS[lv] / f"{lv}_ocean_his_{ctx.yda}0000.nc"
    if cold.is_file():
        return cold, f"COLD from PFM's operational history ({cold.name})"
    raise FileNotFoundError(
        f"no base state for {lv} on {ctx.yda}: neither {prev} nor {cold}")


def _child_atm(ctx: Ctx, lv: str, pkl: Path) -> Path:
    """Build this level's atm from PFM's ECMWF path, or reuse it."""
    out = C.scratch_dir(lv, "Forc") / f"{lv}_DA_ATM_{ctx.yda}.nc"
    if V.nc_ok(out, need_vars=("Uwind", "Vwind", "Tair"), min_bytes=1 << 20):
        _assert_covers_window(out, ctx, "wind_time")
        return out
    n = lv[-1]
    # the ECMWF data is already read and on the roms LV1 grid from stage 2; this
    # re-interpolates it onto THIS level's grid and rotates with THIS level's angle
    B.run_pfm("atm_functions.py", "get_atm_data_on_roms_grid", n, pkl)
    B.run_pfm("atm_functions.py", "atm_roms_dict_to_netcdf", n, pkl)
    chk = V.nc_ok(out, need_vars=("Uwind", "Vwind", "Tair"), min_bytes=1 << 20,
                  finite_vars=("Uwind", "Vwind", "Tair"))
    if not chk:
        raise RuntimeError(f"{lv} atm invalid: {chk.why}")
    # PFM's builder goes through a per-level tmp pickle under a FIXED name
    # (atm_tmp_LV<n>_pckl_file.pkl).  A stale one left by the previous cycle
    # builds a perfectly valid file for the WRONG DAY under today's name, and
    # nothing but the time axis notices.  Same check, same reason, as the LV1
    # forcing stages.
    _assert_covers_window(out, ctx, "wind_time")
    return out


def _child_fwd(ctx: Ctx, lv: str) -> dict:
    out = C.scratch_dir(lv, "His") / f"{lv}_DA_his_{ctx.yda}.nc"
    need = ("zeta", "u", "v", "temp", "salt")
    if V.nc_ok(out, need_vars=need, min_bytes=100 << 20):
        V.ensure_archived(out, C.ARCHIVE / f"{lv}_His")
        raise StageSkip(str(out))
    if ctx.dry_run:
        return {"would_run": f"{lv} forward for {ctx.yda} "
                             f"({C.NODES[lv]} nodes)"}

    # the increment, and it must be the D-space one
    inc = C.SCRATCH / "Inc" / f"INC_{lv}_{ctx.yda}.nc"
    chk = V.nc_ok(inc, need_vars=("zeta", "u", "v", "temp", "salt"),
                  attrs={"blend_space": C.BLEND_SPACE})
    if not chk:
        raise FileNotFoundError(f"{lv} needs its increment from stage 6: "
                                f"{chk.why}")

    # the parent's history, for the boundaries
    parent = C.BC_PARENT[lv]
    if parent == "LV1":
        parent_his = C.io_dir() / "ocean_fwd_outer1.nc"
    else:
        parent_his = C.scratch_dir(parent, "His") / f"{parent}_DA_his_{ctx.yda}.nc"
    if not parent_his.is_file():
        raise FileNotFoundError(
            f"{lv} takes boundaries from {parent}, whose history is not there "
            f"yet: {parent_his}.  The child forwards are strictly sequential.")

    # Check the executable BEFORE doing 5 minutes of input preparation and
    # claiming a 6-node allocation.  LV3's first attempt built its atm, staged
    # everything, submitted, and died in 3 seconds with
    #   /usr/bin/env: 'romsM_forward': No such file or directory
    # repeated 216 times -- because the binary had been copied into LV2/Run but
    # not LV3/Run.  Stage 5 already checks its own executable; this did not.
    exe = C.scratch_dir(lv, "Run") / C.ROMS_EXE_FWD
    if not (exe.is_file() and os.access(exe, os.X_OK)):
        raise FileNotFoundError(
            f"{lv} forward executable missing or not executable: {exe}\n"
            f"    It is built outside this repo.  Copy it in, e.g.\n"
            f"      cp -p {C.scratch_dir('LV2', 'Run') / C.ROMS_EXE_FWD} {exe}")

    forc = C.scratch_dir(lv, "Forc")
    forc.mkdir(parents=True, exist_ok=True)

    # 1. base state
    src, note = _child_base_ic(ctx, lv)
    print(f"    base: {note}")
    base = forc / f"{lv}_DA_base_{ctx.yda}.nc"
    rc = _run([C.PY, C.PYDIR / "his_to_ic.py", "--his", src,
               "--time", f"{ctx.t_da:%Y-%m-%dT%H:%M:%S}",
               "--out", base, "--sanitize"],
              log=C.SCRATCH / "Log" / f"{ctx.yfore}_{lv}_fwd.log")
    if not V.nc_ok(base, need_vars=need, expect_time=ctx.t_da,
                   min_bytes=20 << 20):
        raise RuntimeError(f"{lv} base IC invalid (rc={rc}): {base}")

    # 2. IC = base + increment, masked to the RUN grid
    ic = forc / f"{lv}_DA_IC_{ctx.yda}.nc"
    rc = _run([C.PY, C.PYDIR / "apply_child_increment.py",
               "--base", base, "--incr", inc, "--out", ic,
               "--grid", C.GRID[lv]],
              log=C.SCRATCH / "Log" / f"{ctx.yfore}_{lv}_fwd.log")
    if not V.nc_ok(ic, need_vars=need, expect_time=ctx.t_da, min_bytes=20 << 20,
                   finite_vars=("zeta", "temp", "salt"),
                   water_range={"temp": (-3.0, 40.0), "salt": (0.0, 45.0),
                                "zeta": (-5.0, 5.0)}):
        raise RuntimeError(f"{lv} IC invalid after adding the increment (rc={rc})")

    # 3. atm and 4. BC
    atm = _child_atm(ctx, lv, _minfo(ctx))
    bc = forc / f"{lv}_DA_BC_{ctx.yda}.nc"
    if not V.nc_ok(bc, need_vars=("temp_south",), min_bytes=1 << 20):
        cmd = [C.PY, C.PYDIR / "parent_history_to_BC.py",
               "--parent-grid", C.GRID[parent], "--child-grid", C.GRID[lv],
               "--history", parent_his, "--out", bc]
        if lv == "LV4":
            # LV4 has 10 s-levels against LV3's 40, so its vertical parameters
            # cannot be inferred from the parent -- they must be read from the
            # child grid's own run configuration.
            cmd += ["--child-vparams-from", str(ic)]
        rc = _run(cmd, log=C.SCRATCH / "Log" / f"{ctx.yfore}_{lv}_fwd.log")
        if not V.nc_ok(bc, need_vars=("temp_south",), min_bytes=1 << 20):
            raise RuntimeError(f"{lv} BC build failed (rc={rc}): {bc}")

    # 5. the .in, 6. submit
    rin = _stage_child_in(lv, ctx, {"ic": ic, "bc": bc, "atm": atm, "his": out})
    log = C.SCRATCH / "Log" / f"{lv}_DA_fwd_{ctx.yda}.log"
    t0 = time.time()
    rc, job = _sbatch_wait(C.SLURMDIR / f"slurm_{lv}_fwd.sb",
                           {"ROMS_EXE": str(C.scratch_dir(lv, "Run")
                                            / C.ROMS_EXE_FWD),
                            "RIN": str(rin), "LOG": str(log)},
                           cwd=C.scratch_dir(lv, "Run"),
                           timeout_s=ctx.left_h() * 3600)
    chk = V.roms_job_ok(log, (out,), t0, rc, min_bytes=100 << 20)
    if not chk:
        raise RuntimeError(f"{lv} forward failed (job {job}): {chk.why}. "
                           f"Log: {log}")
    print(f"    job {job} accepted: {chk.why} "
          f"({(time.time()-t0)/60:.1f} min)")
    _warn_on_roms_errors(log)
    arch = V.ensure_archived(out, C.ARCHIVE / f"{lv}_His")
    V.ensure_archived(log, C.ARCHIVE / "Log")
    return {lv: {"scratch": out, "archive": arch}}


def stage_lv2_fwd(ctx: Ctx) -> dict:
    return _child_fwd(ctx, "LV2")


def stage_lv3_fwd(ctx: Ctx) -> dict:
    return _child_fwd(ctx, "LV3")


def stage_lv4_fwd(ctx: Ctx) -> dict:
    """LV4 is COAWST (ROMS + SWAN via MCT), so it needs five things the others do
    not, and all five come from PFM's own builders rather than the research
    chain's LV4 scripts:

        rivers / nudging / clm   archived + retimed (see below)
        SWAN bnd                 swan_functions.mk_swan_bnd_file, from CDIP
        SWAN wnd                 only when swan_wind_wave_generation is on
        SWAN hotstart            the operational restart valid at t_da
        three .in files          run_funs.make_LV4_coawst_dotins_dotsb

    THE SWAN CLOCK HAZARD IS ALREADY HANDLED BY PFM.  SWAN's clock drifts
    ~1280 s per hop when RESTART is given FREE with no interval -- what blocked
    2026-03-09 on the research side.  PFM's LV4_SWAN_BLANK.in writes
    RESTART '...' FREE $swan_rst_int_hr$ HR, with the interval present.  At the
    default rst_interval of 0.25 d that is 6 HR, so a 1-day run writes
    _006/_012/_018/_024 and the next night hot-starts from _024.  No override
    needed; it just writes three files we do not use.

    Dye: coawstM_noanapassive has ANA_PASSIVE undef so IC-file dye is honoured
    rather than zeroed at t=0, and his_to_ic carries dye_* through.  Asserted
    below, because an LV4 IC without dye runs and silently produces nothing.
    """
    out = C.scratch_dir("LV4", "His") / f"LV4_DA_his_{ctx.yda}.nc"
    need = ("zeta", "u", "v", "temp", "salt")
    if V.nc_ok(out, need_vars=need, min_bytes=500 << 20):
        V.ensure_archived(out, C.ARCHIVE / "LV4_His")
        raise StageSkip(str(out))
    if ctx.dry_run:
        return {"would_run": f"LV4 coupled forward for {ctx.yda} "
                             f"({C.NODES['LV4']} nodes)"}

    exe = C.scratch_dir("LV4", "Run") / C.ROMS_EXE_LV4
    if not (exe.is_file() and os.access(exe, os.X_OK)):
        raise FileNotFoundError(
            f"LV4 executable missing or not executable: {exe}.  LV4 needs "
            f"{C.ROMS_EXE_LV4} (ANA_PASSIVE undef), not {C.ROMS_EXE_FWD}.")

    inc = C.SCRATCH / "Inc" / f"INC_LV4_{ctx.yda}.nc"
    chk = V.nc_ok(inc, need_vars=need, attrs={"blend_space": C.BLEND_SPACE})
    if not chk:
        raise FileNotFoundError(f"LV4 needs its increment from stage 6: {chk.why}")

    parent_his = C.scratch_dir("LV3", "His") / f"LV3_DA_his_{ctx.yda}.nc"
    if not parent_his.is_file():
        raise FileNotFoundError(
            f"LV4 takes boundaries from LV3, whose history is not there yet: "
            f"{parent_his}")

    pkl = _minfo(ctx)
    import pickle as _pk
    with open(pkl, "rb") as fp:
        MI = _pk.load(fp)
    forc = C.scratch_dir("LV4", "Forc")
    forc.mkdir(parents=True, exist_ok=True)

    # -- 1. base state, carrying dye ----------------------------------------
    src, note = _child_base_ic(ctx, "LV4")
    print(f"    base: {note}")
    base = forc / f"LV4_DA_base_{ctx.yda}.nc"
    if not V.nc_ok(base, need_vars=need, expect_time=ctx.t_da, min_bytes=200 << 20):
        rc = _run([C.PY, C.PYDIR / "his_to_ic.py", "--his", src,
                   "--time", f"{ctx.t_da:%Y-%m-%dT%H:%M:%S}",
                   "--out", base, "--sanitize"],
                  log=C.SCRATCH / "Log" / f"{ctx.yfore}_LV4_fwd.log",
                  timeout=5400)
        if not V.nc_ok(base, need_vars=need, expect_time=ctx.t_da,
                       min_bytes=200 << 20):
            raise RuntimeError(f"LV4 base IC invalid (rc={rc}): {base}")

    # -- 2. IC = base + increment -------------------------------------------
    ic = forc / f"LV4_DA_IC_{ctx.yda}.nc"
    if not V.nc_ok(ic, need_vars=need, expect_time=ctx.t_da, min_bytes=200 << 20):
        rc = _run([C.PY, C.PYDIR / "apply_child_increment.py",
                   "--base", base, "--incr", inc, "--out", ic,
                   "--grid", C.GRID["LV4"]],
                  log=C.SCRATCH / "Log" / f"{ctx.yfore}_LV4_fwd.log",
                  timeout=5400)
        if not V.nc_ok(ic, need_vars=need, expect_time=ctx.t_da,
                       min_bytes=200 << 20,
                       water_range={"temp": (-3.0, 40.0), "salt": (0.0, 45.0),
                                    "zeta": (-5.0, 5.0)}):
            raise RuntimeError(f"LV4 IC invalid after the increment (rc={rc})")
    _assert_lv4_dye(ic)

    # make_LV4_coawst_dotins_dotsb composes the IC path as
    #     PFM['restart_ncfiles_dir_in'] + '/' + PFM['lv4_ini_file']
    # unconditionally -- it treats the initial state as a restart, which for a
    # single-record IC it effectively is.  Our IC is built in LV4/Forc, so link
    # it where the generator expects, rather than repurposing a run-wide key for
    # one level's benefit.  A symlink, because the file is ~2 GB and ROMS only
    # reads it.  (This is what the first LV4 attempt died on: FIND_FILE could not
    # find .../Rst/LV4_DA_IC_20261006.nc, 25 s into a 14-node allocation.)
    # LV4_BLANK_new.in hardcodes  FRCNAME == $lv4_forc_dir$/LV4_ATM_FORCING.nc
    # -- only the directory is substituted, the basename is literal, and the
    # generator never reads lv4_atm_file.  Link our date-tagged atm to the name
    # the template insists on, keeping the tagged file as the real artefact.
    # Build the atm and stage the river FIRST.  Neither existed for 2026-10-07:
    # stage_lv4_fwd only ever linked and then checked them, so the 10/06 cycle
    # worked solely because both had been made by hand that session.  The
    # symptom was "LV4 atm missing" after the 4D-Var and all three children had
    # already run -- 3.5 h in.  Note the link is created BELOW this, so an
    # earlier version left LV4_ATM_FORCING.nc dangling at a file that was never
    # built.
    _child_atm(ctx, "LV4", pkl)
    _stage_lv4_river(ctx, MI)
    _stage_lv4_window_inputs(ctx, MI, pkl)

    atm_link = Path(MI["lv4_forc_dir"]) / "LV4_ATM_FORCING.nc"
    atm_real = Path(MI["lv4_forc_dir"]) / MI["lv4_atm_file"]
    if atm_link.resolve() != atm_real.resolve():
        if atm_link.is_symlink() or atm_link.exists():
            atm_link.unlink()
        atm_link.symlink_to(atm_real)
        print(f"    atm linked for the template: {atm_link.name} -> {atm_real.name}")

    rst_in = Path(MI_rst_dir(pkl))
    rst_in.mkdir(parents=True, exist_ok=True)
    link = rst_in / ic.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(ic)
    print(f"    IC linked for the .in generator: {link}")

    # -- 3. BC from the LV3 DA history --------------------------------------
    bc = forc / f"LV4_DA_BC_{ctx.yda}.nc"
    if not V.nc_ok(bc, need_vars=("temp_south",), min_bytes=1 << 20):
        rc = _run([C.PY, C.PYDIR / "parent_history_to_BC.py",
                   "--parent-grid", C.GRID["LV3"], "--child-grid", C.GRID["LV4"],
                   "--history", parent_his, "--out", bc,
                   # LV4 has 10 s-levels against LV3's 40, so the vertical
                   # parameters cannot be inferred from the parent
                   "--child-vparams-from", str(ic)],
                  log=C.SCRATCH / "Log" / f"{ctx.yfore}_LV4_fwd.log",
                  timeout=5400)
        if not V.nc_ok(bc, need_vars=("temp_south",), min_bytes=1 << 20):
            raise RuntimeError(f"LV4 BC build failed (rc={rc}): {bc}")

    # LV4's ROMS is built with DYE_01 and DYE_02, so it requires dye boundary
    # variables -- even under RadNud, where the boundary value is what the
    # interior is nudged toward.  parent_history_to_BC builds from LV3, which
    # carries no dye, so the slots have to be added afterwards, zero-filled
    # (no dye flux through the open boundaries; all LV4 dye comes from the
    # rivers and the IC).  Without them ROMS stops at
    #   INQUIRY_NF90 - unable to find requested variable: dye_west_01
    if not V.nc_ok(bc, need_vars=("dye_west_01", "dye_west_02")):
        rc = _run([C.PY, C.PYDIR / "add_dye_bc_LV4.py", bc],
                  log=C.SCRATCH / "Log" / f"{ctx.yfore}_LV4_fwd.log")
        chk = V.nc_ok(bc, need_vars=("dye_south_01", "dye_north_01",
                                     "dye_west_01", "dye_south_02",
                                     "dye_north_02", "dye_west_02"))
        if not chk:
            raise RuntimeError(f"LV4 dye BC variables missing after "
                               f"add_dye_bc_LV4 (rc={rc}): {chk.why}")
        print("    dye BC variables added (zero-filled, 3 open boundaries x 2 "
              "tracers)")

    # -- 4. the forcing staged earlier must all be present -------------------
    for key, label in (("lv4_atm_file", "atm"), ("lv4_clm_file", "climatology"),
                       ("lv4_nud_file", "nudging"), ("lv4_river_file", "river"),
                       ("lv4_swan_bnd_file", "SWAN boundary")):
        f = Path(MI["lv4_forc_dir"]) / MI[key]
        if not f.is_file():
            raise FileNotFoundError(f"LV4 {label} missing: {f}")

    # -- 5. SWAN hotstart, selected read-only --------------------------------
    n = B.stage_swan_hotstart(ctx.t_da)
    hot = B.pick_swan_hotstart(pkl)
    txt = B.set_swan_init(pkl, hot)
    print(f"    SWAN: {n} restart parts linked; {txt[:88]}")
    if hot is None:
        raise RuntimeError(
            "no SWAN hotstart found for this window, so SWAN would cold-start "
            "from a flat sea -- which cannot spin up inside 24 h.  Check "
            f"{C.SWAN_RST_SHARED} for a set valid at {ctx.t_da:%Y-%m-%d %H}Z.")

    # -- 6. the three .in files, from PFM's own generator --------------------
    B.make_lv4_dotins(pkl)
    run_dir = C.scratch_dir("LV4", "Run")
    coupled = run_dir / "LV4_forecast_coupled.in"
    for f in (coupled, run_dir / "LV4_forecast_run.in",
              run_dir / "LV4_forecast_swan.in"):
        if not f.is_file():
            raise RuntimeError(f"make_LV4_coawst_dotins_dotsb did not write {f}")
    _assert_lv4_in_consistent(run_dir, ctx, ic, bc)

    # -- 7. submit -----------------------------------------------------------
    log = C.SCRATCH / "Log" / f"LV4_DA_fwd_{ctx.yda}.log"
    t0 = time.time()
    rc, job = _sbatch_wait(C.SLURMDIR / "slurm_LV4_fwd.sb",
                           {"ROMS_EXE": str(exe), "RIN": str(coupled),
                            "LOG": str(log)},
                           cwd=run_dir, timeout_s=ctx.left_h() * 3600)
    chk = V.roms_job_ok(log, (out,), t0, rc, min_bytes=500 << 20)
    if not chk:
        raise RuntimeError(f"LV4 forward failed (job {job}): {chk.why}. "
                           f"Log: {log}")
    print(f"    job {job} accepted: {chk.why} ({(time.time()-t0)/60:.1f} min)")
    _warn_on_roms_errors(log)
    arch = V.ensure_archived(out, C.ARCHIVE / "LV4_His")
    V.ensure_archived(log, C.ARCHIVE / "Log")
    return {"LV4": {"scratch": out, "archive": arch}}


def MI_rst_dir(pkl: Path) -> str:
    """The directory make_LV4_coawst_dotins_dotsb composes the IC path from."""
    import pickle as _pk
    with open(pkl, "rb") as fp:
        return _pk.load(fp)["restart_ncfiles_dir_in"]


def _assert_swan_bnd_covers(bnd: Path, ctx: Ctx) -> None:
    """The SWAN boundary spectra must carry THIS window's timestamps.

    SWAN does not fail on a boundary file for the wrong day -- it holds the last
    available spectrum -- so a stale .bnd runs to completion and produces a
    plausible wave field driven by yesterday's swell.  That is exactly what
    happened on 2026-10-07: LV4 ran green on a .bnd carrying 20261006.0000 ..
    20261007.0000.  Only reading the timestamps notices.
    """
    # SWAN writes "YYYYMMDD.HHMMSS" followed by a "date and time" comment on the
    # same line, so take the first token rather than the whole line.
    stamps = []
    with open(bnd, "r", errors="ignore") as fh:
        for line in fh:
            tok = line.split()[0] if line.split() else ""
            if re.fullmatch(r"\d{8}\.\d{6}", tok):
                stamps.append(tok)
    if not stamps:
        raise RuntimeError(f"{bnd.name}: no SWAN timestamps found at all")
    fmt = "%Y%m%d.%H%M%S"
    lo = DT.datetime.strptime(stamps[0], fmt)
    hi = DT.datetime.strptime(stamps[-1], fmt)
    # COVERS the window, not "starts exactly on it".  The earlier version
    # demanded equality and broke on 2026-10-10, when CDIP changed its lead
    # convention: the init stamped <date>00 went from beginning 3-9 h BEFORE
    # that date to beginning 3 h AFTER it.  cdip_init_covering now picks an
    # init that spans the window, so the bnd may legitimately start early --
    # what must never happen is a bnd that does not reach the window's ends,
    # because SWAN holds the last spectrum instead of complaining.
    if lo > ctx.t_da or hi < ctx.t_fore:
        raise RuntimeError(
            f"{bnd.name} spans {stamps[0]}..{stamps[-1]}, which does not cover "
            f"the DA window {ctx.t_da:%Y%m%d.%H%M%S}..{ctx.t_fore:%Y%m%d.%H%M%S}."
            f"  SWAN would run on held spectra and NOT complain.")
    print(f"    SWAN bnd covers {stamps[0]} .. {stamps[-1]} ({len(stamps)} times)")


def _stage_lv4_window_inputs(ctx: Ctx, MI: dict, pkl: Path) -> None:
    """Rebuild the LV4 inputs that carry a window but NOT a date in their name.

    LV4_clm.nc and LV4_swan.bnd are written under FIXED names, so a file left by
    the previous cycle looks present and valid to every existence check.  Both
    were built by hand during the 2026-10-06 session and then silently reused:

      * LV4_clm.nc spanned 10/05..10/08, which happens to contain the 10/06 and
        10/07 windows.  The 10/08 cycle fell outside it and ROMS stopped at
        MULTIFILE_INFO_S2D, "Error while checking input Climatology file".
      * LV4_swan.bnd carried 20261006.0000..20261007.0000 for all three cycles.
        SWAN does not reject that, so 10/07 ran green on 10/06's spectra.

    The bridge had stage_lv4_clm, stage_cdip_from_archive and
    cdip_dict_from_archive ready for exactly this; nothing ever called them.
    stage_lv4_fwd's own docstring already claimed "SWAN bnd --
    swan_functions.mk_swan_bnd_file, from CDIP", which was intent, not code.
    """
    fd = Path(MI["lv4_forc_dir"])

    # climatology: a retimed copy of the operational file.  Rebuilt every cycle,
    # unconditionally -- the span is the only thing that has to be right and it
    # is different every day.
    B.stage_lv4_clm(ctx.t_da, ctx.t_fore, fd / MI["lv4_clm_file"])

    # interior nudging coefficients.  BUILT, with PFM's own mk_lv4_nud_nc, which
    # needs only lv4_grid_file and stretching['L4','Nz'] -- no BC pickle, no
    # network -- and writes to lv4_forc_dir/lv4_nud_file, exactly where the
    # check below looks.
    #
    # Nothing used to produce this.  It was placed by hand when the chain was
    # first set up and simply sat in LV4/Forc, until SCRATCH_PURGE_SUBDIRS
    # ("Forc", "His") with SCRATCH_RETAIN_DAYS = 7 finally deleted it and the
    # 2026-10-13 cycle died here, after LV1's 4D-Var and both LV2/LV3 forwards
    # had already succeeded:
    #     [lv4_fwd] FAILED: LV4 nudging missing: .../LV4/Forc/LV4_nud.nc
    # A static input living in a purged directory is a time bomb; building it is
    # what makes the cycle self-sufficient.
    #
    # Only when missing: the field is purely static (temp/salt/tracer_NudgeCoef
    # on (s_rho, er, xr), no time axis at all), so unlike the climatology there
    # is nothing per-cycle to get right, and it is 142 MB to rewrite.
    nud = fd / MI["lv4_nud_file"]
    if not nud.is_file():
        B.run_pfm("ocn_funs_forecast.py", "mk_lv4_nud_nc", pkl, timeout=1800)
        if not nud.is_file():
            raise FileNotFoundError(
                f"mk_lv4_nud_nc did not write {nud}.  It is static, so a copy "
                f"can be staged by hand from "
                f"/dataSIO/DA_Simulations/LV4/input_files/LV4_nud.nc")
        print(f"    LV4 nudging coefficients built -> {nud.name}")

    # CDIP spectra -> the pickle -> the SWAN boundary file.  All from the
    # date-stamped archive, so no network call.
    n = B.stage_cdip_from_archive(ctx.t_da, ctx.t_fore)
    B.cdip_dict_from_archive(pkl, ctx.t_da, ctx.t_fore)
    bnd = fd / MI["lv4_swan_bnd_file"]
    B.run_pfm("swan_functions.py", "mk_swan_bnd_file", str(bnd), str(pkl),
              timeout=3000)
    print(f"    SWAN bnd rebuilt from {n} archived CDIP stations -> {bnd.name}")
    _assert_swan_bnd_covers(bnd, ctx)

    # SWAN winds: a no-op unless swan_wind_wave_generation is on (it is off in
    # this configuration, and mk_swan_wnd_file exits 1 on purpose).
    B.build_swan_wnd(pkl, fd / MI["lv4_swan_wnd_file"])


def _stage_lv4_river(ctx: Ctx, MI: dict) -> Path:
    """Put this window's river file where the LV4 .in generator expects it.

    COPIED from PFM's operational Forcing archive, not rebuilt.  PFM's own
    mk_lv4_river_nc calls rivfuns.get_river_flow_nwm, which goes to the network
    -- and PFM_DA sources only what is already on disk.  The operational cycle
    for this day has already produced the file, and it covers t_da .. t_da + 5 d,
    which contains our 24 h window.

    Verified 2026-10-08: the 10/06 cycle's river_20261006_20261007.nc is
    BYTE-IDENTICAL to river_LV4_2026100600.nc (md5 dcfb7370...), i.e. the file
    that was staged by hand last session is exactly this copy.  This only makes
    that step reproducible.
    """
    dst = Path(MI["lv4_forc_dir"]) / MI["lv4_river_file"]
    if dst.is_file():
        return dst
    src = C.PFM_FORCING_ARCHIVE / f"river_LV4_{ctx.t_da:%Y%m%d}00.nc"
    if not src.is_file():
        raise FileNotFoundError(
            f"LV4 river: {src} is not on disk.  It is written by PFM's own "
            f"cycle for {ctx.t_da:%Y-%m-%d}; nothing here rebuilds it, because "
            f"PFM's mk_lv4_river_nc fetches from the network.  If that cycle "
            f"did not run, stage a neighbouring day by hand and say so.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"    river staged from the operational archive: "
          f"{src.name} -> {dst.name}")
    return dst


def _assert_lv4_dye(ic: Path) -> None:
    """LV4's IC must carry dye_01 and dye_02, and they must not be all zero.

    coawstM_noanapassive has ANA_PASSIVE undef, so dye comes from the IC file
    rather than an analytical profile.  Without it the run completes normally and
    produces identically zero dye -- the whole point of LV4 lost, with nothing in
    the log to say so.  A variable that is PRESENT but zero fails the same way,
    which is why the magnitude is checked too.

    ROMS writes ~1e37 on land, so the fill must be masked before summing.  The
    first version did not, and reported sum|dye_01| = 1.6e43 -- that is 1e37
    times the land-cell count, not dye.  Masked, the same field is 7.1e2 over
    3.8M wet cells.  (Third time this bug has appeared in this project: see
    da_validate.nc_ok and the sampler in compare_PFM_PFMDA.ipynb.  ROMS fill is
    FINITE and large, so it survives every isfinite() check.)
    """
    import netCDF4 as nc_
    import numpy as np_
    with nc_.Dataset(ic) as d:
        missing = [v for v in ("dye_01", "dye_02") if v not in d.variables]
        if missing:
            raise RuntimeError(
                f"{ic.name} has no {', '.join(missing)}.  LV4 runs "
                f"coawstM_noanapassive (ANA_PASSIVE undef), so it would run to "
                f"completion with zero dye everywhere.  his_to_ic carries dye "
                f"through only when the SOURCE history has it -- check the base.")
        rep = {}
        for v in ("dye_01", "dye_02"):
            a = np_.ma.filled(d[v][:], np_.nan).astype(float)
            a[np_.abs(a) > 1e30] = np_.nan          # ROMS land fill
            fin = a[np_.isfinite(a)]
            rep[v] = (float(np_.abs(fin).sum()), int((np_.abs(fin) > 0).sum()),
                      float(np_.nanmax(fin)) if fin.size else 0.0)
    for v, (tot, nz, mx) in rep.items():
        if nz == 0 or tot == 0.0:
            raise RuntimeError(
                f"{ic.name}: {v} is present but identically zero over every wet "
                f"cell.  LV4 would run and produce no dye at all.")
    print("    dye ok: " + ", ".join(
        f"{v} sum={tot:.3e} over {nz:,} wet cells (max {mx:.3e})"
        for v, (tot, nz, mx) in rep.items()))


def _assert_lv4_in_consistent(run_dir: Path, ctx: Ctx, ic: Path,
                              bc: Path) -> None:
    """The generated .in files must name THIS cycle's inputs and date."""
    roms = (run_dir / "LV4_forecast_run.in").read_text()
    # Check the FULL path each key resolves to, not just the basename.  The
    # basename-only version passed an .in whose ININAME pointed at the restart
    # directory instead of LV4/Forc -- same filename, wrong place, and ROMS died
    # on FIND_FILE after the allocation had started.
    for key, want in (("ININAME", ic), ("BRYNAME", bc)):
        m = re.search(rf"^\s*{key}\s*==?\s*(\S+)", roms, re.M)
        if not m:
            raise RuntimeError(f"LV4_forecast_run.in has no {key} line")
        got = Path(m.group(1))
        if not got.is_file():
            raise FileNotFoundError(
                f"LV4_forecast_run.in {key} points at a file that does not "
                f"exist: {got}")
        if got.resolve() != want.resolve():
            raise RuntimeError(
                f"LV4_forecast_run.in {key} resolves to {got.resolve()}, but "
                f"this stage built {want.resolve()}")
    # EVERY input the .in names must exist, not just the two this stage happens
    # to build.  Listing them by hand missed FRCNAME, which the template
    # hardcodes -- it would have been the next 14-node crash.
    # An ALLOWLIST of the inputs a COAWST forward run actually reads, not a
    # denylist of outputs: the template also carries 4D-Var placeholders such as
    # ITLNAME == ocean_itl.nc that are never opened here, and a denylist would
    # have to grow every time one appeared.
    for key in ("GRDNAME", "ININAME", "BRYNAME", "FRCNAME", "CLMNAME",
                "NUDNAME", "VARNAME"):
        m = re.search(rf"^\s*{key}\s*==?\s*(\S+)", roms, re.M)
        if not m:
            continue                      # not every key is used at every level
        pth = Path(m.group(1))
        if not pth.is_file():
            raise FileNotFoundError(
                f"LV4_forecast_run.in {key} points at a missing file: {pth}")

    swan = (run_dir / "LV4_forecast_swan.in").read_text()
    if "HOTSTART" not in swan and "INITIAL" not in swan:
        raise RuntimeError("LV4_forecast_swan.in has no INITIAL/HOTSTART line")
    if "FREE" in swan and "HR" not in swan:
        raise RuntimeError(
            "LV4_forecast_swan.in has RESTART ... FREE with no interval.  SWAN's "
            "clock drifts ~1280 s per hop in that form -- the 2026-03-09 bug.")


# ---------------------------------------------------------------------------
# stage 10: the handoff -- extract t = 24 h from each level
# ---------------------------------------------------------------------------
def _handoff_range(lv: str) -> dict:
    """Acceptable water-cell ranges, per level.

    LV4 gets a wider salinity bound because it has WET_DRY and intertidal cells
    in the bay and estuary that carry genuinely hypersaline water.  Measured
    2026-10-08: 230 wet cells above 45 PSU, 0.0058% of the grid, topping out at
    54.1 -- and they are INHERITED, not produced here.  PFM's own operational LV4
    carries exactly the same 230 cells at the same values, the increment does not
    touch them, and after 24 h our 54.37 tracks PFM's own 54.36.

    The bound is 60 rather than something larger so a real salinity blow-up is
    still caught; the research notes record LV4 histories reaching 194 PSU in
    dried cells, which this would still reject.
    """
    if lv == "LV4":
        return {"temp": (-3.0, 40.0), "salt": (0.0, 60.0), "zeta": (-5.0, 5.0)}
    return {"temp": (-3.0, 40.0), "salt": (0.0, 45.0), "zeta": (-5.0, 5.0)}


def stage_handoff(ctx: Ctx) -> dict:
    """The t = 24 h state of each level IS the state at t_fore, so it is exactly
    the IC the forecast needs.  Written to IC_out/ and archived.

    NRREC = 0 for these, because they are SINGLE-RECORD files rather than
    restarts.  That also means ini_fields recomputes ubar from u3d -- which is
    what we want and is consistent with how the increment was built (Vtransform=2
    makes the depth-average weights zeta-independent, so depth_avg(u) is exactly
    the ubar ROMS will use), but it IS a behaviour change from restarting, and
    Phase 2 has to set NRREC accordingly.
    """
    outs = {}
    srcs = {
        "LV1": C.io_dir() / "ocean_fwd_outer1.nc",
        "LV2": C.scratch_dir("LV2", "His") / f"LV2_DA_his_{ctx.yda}.nc",
        "LV3": C.scratch_dir("LV3", "His") / f"LV3_DA_his_{ctx.yda}.nc",
        "LV4": C.scratch_dir("LV4", "His") / f"LV4_DA_his_{ctx.yda}.nc",
    }
    for lv, src in srcs.items():
        out = C.SCRATCH / "IC_out" / f"{lv}_DA_IC_for_{ctx.t_fore:%Y%m%d%H}.nc"
        need = ("zeta", "u", "v", "temp", "salt")
        if lv == "LV4":
            need += ("dye_01", "dye_02")     # coawstM_noanapassive needs IC dye
        chk = V.nc_ok(out, need_vars=need, expect_time=ctx.t_fore,
                      min_bytes=20 << 20)
        if chk:
            arch = V.ensure_archived(out, C.ARCHIVE / "IC_out")
            outs[lv] = {"scratch": out, "archive": arch}
            ctx.man.set_ic(lv, out, arch)
            continue
        if ctx.dry_run:
            outs[lv] = {"would_run": f"his_to_ic.py {src.name} --time "
                                     f"{ctx.t_fore:%Y-%m-%dT%H:%M:%S}"}
            continue
        if not src.is_file():
            raise FileNotFoundError(
                f"{lv} handoff needs {src}, which stage "
                f"{'5' if lv == 'LV1' else '7'} should have produced")
        rc = _run([C.PY, C.PYDIR / "his_to_ic.py",
                   "--his", src,
                   "--time", f"{ctx.t_fore:%Y-%m-%dT%H:%M:%S}",
                   "--out", out, "--sanitize"],
                  log=C.SCRATCH / "Log" / f"{ctx.yfore}_handoff.log")
        chk = V.nc_ok(out, need_vars=need, expect_time=ctx.t_fore,
                      min_bytes=20 << 20, finite_vars=("zeta", "temp", "salt"),
                      water_range=_handoff_range(lv), grid=C.GRID[lv])
        if not chk:
            raise RuntimeError(f"{lv} handoff IC invalid: {chk.why} (rc={rc})")
        arch = V.ensure_archived(out, C.ARCHIVE / "IC_out")
        outs[lv] = {"scratch": out, "archive": arch}
        ctx.man.set_ic(lv, out, arch)
        print(f"    {lv} IC at {ctx.t_fore:%Y-%m-%d %H}Z -> {out.name}")
    return outs


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
