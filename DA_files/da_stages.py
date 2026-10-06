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
import da_validate as V
from da_manifest import Manifest

sys.path.insert(0, str(C.PYDIR))

STAGES = ("lv1_ic", "lv1_atm", "lv1_bc", "lv1_obs", "lv1_da",
          "increments", "lv2_fwd", "lv3_fwd", "lv4_fwd", "handoff")


@dataclass
class Ctx:
    t_fore: DT.datetime
    t_da: DT.datetime
    man: Manifest
    deadline: DT.datetime
    dry_run: bool = False

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
# stage 1: LV1 initial condition, from the PFM operational history at t_da
# ---------------------------------------------------------------------------
def stage_lv1_ic(ctx: Ctx) -> dict:
    out = C.scratch_dir("LV1", "Forc") / f"LV1_DA_IC_{ctx.yda}.nc"
    chk = V.nc_ok(out, need_vars=("zeta", "u", "v", "temp", "salt"),
                  expect_time=ctx.t_da, min_bytes=50 << 20)
    if chk:
        # archive on the skip path too -- see da_validate.ensure_archived
        V.ensure_archived(out, C.ARCHIVE / "IC_out")
        raise StageSkip(str(out))
    src = C.PFM_ARCHIVE / "LV1_His" / f"LV1_ocean_his_{ctx.yda}0000.nc"
    if not src.exists():
        raise FileNotFoundError(f"no PFM LV1 history for {ctx.yda}: {src}")
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
# stages 2-3: LV1 atm and BC.
#
# These are the only real integration work in Phase 1.  Phase 0 established
# that PFM's own builders are correct and should be REUSED rather than ported:
#   * atm_functions.py rotates winds earth -> ROMS XI/ETA (lines 610-611),
#     matching set_data.F, and has done since 2025-07-02.
#   * ocn_funs_forecast.py converts in-situ -> potential temperature with
#     gsw.pt_from_t (the commented seawater.ptmp lines are superseded, 7 for 7).
# Porting our NAM/HYCOM builders instead would reintroduce exactly the bugs
# September fixed, and would not reproduce PFM's own forcing.
#
# Both PFM builders are driven by the PFM dict and a pickle, so the work is an
# adapter that constructs a 1-day PFM dict pointing at DA paths without
# disturbing the operational one.  Left unimplemented deliberately: it is the
# first thing to write, and the acceptance test is that day 1 of the result
# matches the operational LV1_ATM_FORCING.nc / LV1_OCEAN_BC.nc to round-off.
# ---------------------------------------------------------------------------
def stage_lv1_atm(ctx: Ctx) -> dict:
    out = C.scratch_dir("LV1", "Forc") / f"LV1_DA_ATM_{ctx.yda}.nc"
    chk = V.nc_ok(out, need_vars=("Uwind", "Vwind", "Tair", "rain"),
                  attrs={"wind_coordinate_frame": "ROMS grid (XI/ETA)"},
                  min_bytes=10 << 20)
    if chk:
        raise StageSkip(str(out))
    raise NotImplementedError(
        "PFM atm adapter not written yet.  Reuse sdpm_py_util/atm_functions.py "
        "for a 1-day window ending at t_fore; do NOT port make_atm_from_nam.py. "
        "Validate that the result carries wind_coordinate_frame='ROMS grid "
        "(XI/ETA)' and reproduces day 1 of LV1_ATM_FORCING.nc.")


def stage_lv1_bc(ctx: Ctx) -> dict:
    out = C.scratch_dir("LV1", "Forc") / f"LV1_DA_BC_{ctx.yda}.nc"
    chk = V.nc_ok(out, need_vars=("temp_south", "temp_north", "temp_west"),
                  min_bytes=5 << 20)
    if chk:
        raise StageSkip(str(out))
    raise NotImplementedError(
        "PFM ocean-BC adapter not written yet.  Reuse ocn_funs_forecast."
        "ocnr_2_BCdict_1hrzeta_from_tmppkls + ocn_roms_BC_dict_to_netcdf_pckl_"
        "1hrzeta for a 1-day window; do NOT port make_BCnc_from_hycom.py. "
        "Validate against day 1 of LV1_OCEAN_BC.nc.")


# ---------------------------------------------------------------------------
# stage 4: HF-radar observations
# ---------------------------------------------------------------------------
def stage_lv1_obs(ctx: Ctx) -> dict:
    out = C.scratch_dir("LV1", "Forc") / f"obs_HF_LV1_{ctx.yda}.nc"
    chk = V.nc_ok(out, need_vars=("obs_type", "obs_value", "obs_error",
                                  "obs_Xgrid", "obs_Ygrid"), min_bytes=1 << 20)
    if chk:
        raise StageSkip(str(out))
    raise NotImplementedError(
        f"NRT HFR fetch not written yet.  make_obs_from_hfr.py reads the NCEI "
        f"retrospective archive; operationally we need NDBC THREDDS "
        f"(dods.ndbc.noaa.gov/thredds/dodsC/hfradar_uswc_{C.HFR_RES}). "
        f"Measured 2026-10-06: 6 km lags ~4.8 h, 1 and 2 km ~1.8 h -- which is "
        f"why HFR_RES defaults to {C.HFR_RES}.")


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
