#!/usr/bin/env python3
"""
da_ic_handoff.py -- the PFM side of the DA handoff.

Two functions, and nothing else:

    detect_da_ics(t_fore)        can PFM start this forecast from the DA?
    apply_da_ics(PFM, t_fore)    point the model-info dict at those ICs

WHY THERE IS NO NEW IC MACHINERY HERE.  run_funs.py already branches on
PFM['lv{N}_use_restart'] at every level:

    if start_type == 1:  nrrec = PFM['lv{N}_nrrec']; dir = restart_ncfiles_dir_in
    else:                nrrec = '0';                dir = PFM['lv{N}_forc_dir']
    ininame = PFM['lv{N}_ini_file']

So "use the DA IC" is exactly: use_restart = 0, the file in lv{N}_forc_dir,
ini_file naming it.  That path is not speculative -- every PFM_DA cycle runs all
four of its own levels through it.  apply_da_ics sets those three things and
nothing more.

FILES ARE THE TRUTH, THE MANIFEST IS CORROBORATION.  The manifest verdict is
read and reported, but a cycle finished by hand in stages can carry a verdict
that understates it, and a verdict cannot tell you a file was later corrupted or
overwritten.  Every IC is therefore opened and checked against t_fore.  Checking
the NAME is not enough and that is not hypothetical: LV1_fwd_outer0_20261007.nc
in the archive held the 10/06 window for three cycles because ensure_archived
skipped the copy on a size match.

FALLING THROUGH IS NORMAL.  A forecast on restart files beats no forecast, so
every failure here is a reported reason, never an exception.
"""
from __future__ import annotations

import datetime as DT
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import da_config as C          # noqa: E402
import da_validate as V        # noqa: E402
from da_manifest import Manifest   # noqa: E402

LEVELS = ("LV1", "LV2", "LV3", "LV4")

# What each level's IC must contain.  LV4 carries dye because coawstM needs it
# in the IC, and its salt bound is wider because the bay and estuary hold
# genuinely hypersaline water in intertidal cells -- 230 of them above 45 PSU,
# inherited from PFM's own LV4, not produced by the DA.
_NEED = {lv: ("zeta", "u", "v", "temp", "salt") for lv in LEVELS}
_NEED["LV4"] = _NEED["LV4"] + ("dye_01", "dye_02")
_RANGE = {lv: {"temp": (-3.0, 40.0), "salt": (0.0, 45.0), "zeta": (-5.0, 5.0)}
          for lv in LEVELS}
_RANGE["LV4"] = {"temp": (-3.0, 40.0), "salt": (0.0, 60.0), "zeta": (-5.0, 5.0)}

# the DA history that holds t_fore as its LAST record, used only as a fallback
_HIS = {"LV1": lambda y: C.io_dir() / "ocean_fwd_outer1.nc",
        "LV2": lambda y: C.scratch_dir("LV2", "His") / f"LV2_DA_his_{y}.nc",
        "LV3": lambda y: C.scratch_dir("LV3", "His") / f"LV3_DA_his_{y}.nc",
        "LV4": lambda y: C.scratch_dir("LV4", "His") / f"LV4_DA_his_{y}.nc"}


@dataclass
class ICStatus:
    level: str
    ok: bool = False
    path: Path | None = None
    source: str = "none"          # scratch | archive | built | none
    why: str = "not looked for"


@dataclass
class DetectResult:
    t_fore: DT.datetime
    status: str = "none"          # ok | partial | none
    verdict: str = "absent"       # the manifest's own view
    levels: dict = field(default_factory=dict)

    def usable(self) -> list[str]:
        return [lv for lv in LEVELS if self.levels.get(lv) and self.levels[lv].ok]

    def text(self) -> str:
        out = [f"  DA ICs for {self.t_fore:%Y-%m-%d %H}Z: {self.status.upper()} "
               f"(manifest verdict: {self.verdict})"]
        for lv in LEVELS:
            st = self.levels.get(lv)
            if st is None:
                out.append(f"    {lv}  -- not checked")
            elif st.ok:
                out.append(f"    {lv}  OK   [{st.source}] {st.path.name}")
            else:
                out.append(f"    {lv}  no   {st.why}")
        return "\n".join(out)


def _ic_name(level: str, t_fore: DT.datetime) -> str:
    return f"{level}_DA_IC_for_{t_fore:%Y%m%d%H}.nc"


def _check(path: Path, level: str, t_fore: DT.datetime):
    """nc_ok with this level's requirements, against t_fore."""
    return V.nc_ok(path, need_vars=_NEED[level], expect_time=t_fore,
                   min_bytes=20 << 20, finite_vars=("zeta", "temp", "salt"),
                   water_range=_RANGE[level], grid=C.GRID[level])


def _build_from_history(level: str, t_fore: DT.datetime, out: Path) -> ICStatus:
    """Last resort: cut the IC out of the DA history that still holds t_fore.

    stage_handoff normally does this at the end of a cycle.  It is repeated here
    for the case the user described: the cycle ran, its histories are on disk,
    but IC_out was never written (handoff died, or the files were cleaned).
    Uses the same his_to_ic.py, so the result is the same artefact.
    """
    src = _HIS[level](C.ymd(t_fore - DT.timedelta(days=1)))
    if not src.is_file():
        return ICStatus(level, False, why=f"no handoff IC and no history at {src}")
    if not V.nc_ok(src, need_vars=("zeta", "temp"), min_bytes=100 << 20):
        return ICStatus(level, False, why=f"history {src.name} is not readable")
    out.parent.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run([C.PY, str(C.PYDIR / "his_to_ic.py"), "--his", str(src),
                         "--time", f"{t_fore:%Y-%m-%dT%H:%M:%S}",
                         "--out", str(out), "--sanitize"],
                        capture_output=True, text=True).returncode
    chk = _check(out, level, t_fore)
    if not chk:
        return ICStatus(level, False, why=f"built from {src.name} but invalid: "
                                          f"{chk.why} (rc={rc})")
    return ICStatus(level, True, out, "built", f"cut from {src.name}")


def detect_da_ics(t_fore, *, allow_build: bool = True,
                  levels: tuple[str, ...] = LEVELS) -> DetectResult:
    """Can PFM start the forecast at t_fore from PFM_DA output?

    t_fore is a datetime or a 'YYYYMMDDHH' string -- the FORECAST init time,
    which is also the DA window's end, so the handoff IC is named for it exactly.

    status is 'ok' when every requested level has a valid IC, 'partial' when some
    do, 'none' when none do.  'partial' is a real and useful outcome: LV1-LV3
    good with LV4 missing should start three levels from the analysis rather than
    discard all of it.  The caller decides.
    """
    if isinstance(t_fore, str):
        t_fore = DT.datetime.strptime(t_fore, "%Y%m%d%H")
    stamp = f"{t_fore:%Y%m%d%H}"
    res = DetectResult(t_fore=t_fore, verdict=Manifest.verdict_for(stamp))

    for lv in levels:
        name = _ic_name(lv, t_fore)
        found = None
        for where, d in (("scratch", C.SCRATCH / "IC_out"),
                         ("archive", C.ARCHIVE / "IC_out")):
            p = d / name
            if not p.is_file():
                continue
            chk = _check(p, lv, t_fore)
            if chk:
                found = ICStatus(lv, True, p, where, "valid")
                break
            found = ICStatus(lv, False, p, where, f"{where} copy invalid: {chk.why}")
        if found is None:
            found = ICStatus(lv, False, why=f"{name} not in IC_out (scratch or archive)")
        if not found.ok and allow_build:
            built = _build_from_history(lv, t_fore, C.SCRATCH / "IC_out" / name)
            if built.ok:
                found = built
            else:
                # keep BOTH reasons: "no IC_out file" and why cutting one from
                # the history did not work either.  Debugging a cycle that
                # silently fell through to restarts needs the second one.
                found.why = f"{found.why}; build: {built.why}"
        res.levels[lv] = found

    n = len(res.usable())
    res.status = "ok" if n == len(levels) else ("partial" if n else "none")
    return res


def apply_da_ics(PFM: dict, t_fore, found: DetectResult | None = None,
                 *, levels: tuple[str, ...] = LEVELS) -> list[str]:
    """Point the model-info dict at the DA ICs.  Returns the levels applied.

    Copies each IC into that level's forc dir and sets the three keys run_funs
    already reads.  Levels without a usable IC are left exactly as they were, so
    they fall through to the restart path -- which is the whole point: a forecast
    on restarts beats no forecast.

    When LV4's ocean IC comes from the DA, its WAVE state is taken from the DA
    too -- see apply_da_swan_hotstart, called at the end.
    """
    if found is None:
        found = detect_da_ics(t_fore, levels=levels)
    applied = []
    for lv in levels:
        st = found.levels.get(lv)
        if st is None or not st.ok:
            continue
        n = lv[-1]
        forc = Path(PFM[f"lv{n}_forc_dir"])
        forc.mkdir(parents=True, exist_ok=True)
        dst = forc / st.path.name
        if not (dst.is_file() and dst.stat().st_size == st.path.stat().st_size):
            shutil.copy2(st.path, dst)
        PFM[f"lv{n}_use_restart"] = 0      # -> NRREC 0, ini from lv{n}_forc_dir
        PFM[f"lv{n}_ini_file"] = dst.name
        # Tell the driver the IC already exists.  use_restart == 0 alone means
        # "build an IC from HYCOM and write it to lv{n}_ini_file" -- which is
        # exactly this file.  Without this flag PFM overwrites the DA analysis
        # with a HYCOM cold start before ROMS reads it, and the run still looks
        # like a success.  Found 2026-10-14, after the varinfo failure had been
        # masking it.
        PFM[f"lv{n}_ic_prebuilt"] = True
        # NRREC 0: the DA IC is a SINGLE-RECORD file, so record 0 is the only
        # one there.  run_funs does not read this key when use_restart == 0 (it
        # hardcodes '0'), but the DRIVER prints it unconditionally on the branch
        # we now take -- "Going to use the file ... to restart the simulation /
        # with time index <lv{n}_nrrec>" -- and raised KeyError: 'lv1_nrrec'
        # 1.5 s into LV1.  The same print exists at all four levels and in the
        # hindcast and fillin paths, so setting the key here fixes every one of
        # them without touching PFM.  It also keeps the driver's
        # "if nrrec < 0: RESTARTING WILL NOT WORK" check honest.
        PFM[f"lv{n}_nrrec"] = 0
        applied.append(lv)
        print(f"    {lv}: DA IC {dst.name} (use_restart=0, NRREC=0)")
    for lv in levels:
        if lv not in applied:
            print(f"    {lv}: no DA IC -- falling through to the restart file")
    if "LV4" in applied:
        apply_da_swan_hotstart(PFM, t_fore)
    return applied


def apply_da_swan_hotstart(PFM: dict, t_fore, lookback_days: int = 3) -> str | None:
    """Point SWAN at the DA chain's own LV4 wave field at t_fore.

    Only called when LV4's ocean IC came from the DA, because the two have to
    agree: starting LV4's ocean from the analysis and its waves from the
    operational forecast is the inconsistency this exists to remove.

    The DA cycle that produced the ICs for t_fore ran LV4 over
    [t_fore - 1 day, t_fore] with

        RESTART '<SWAN_RST_DA_OUT>/LV4_swan_rst_<t_fore - 1 day>.dat' FREE <n> HR

    and, with no _HHH suffix and no check_and_move watcher, overwrote that one
    name every interval -- so what survives is the LAST write, the wave field at
    t_fore exactly.  SWAN appends the rank suffix to a HOTSTART name itself, so
    the unsuffixed stem is handed over as-is; FILENM is CHARACTER*140
    (SWAN/src_coawst/swmod1.ftn), so the ~65-character path is nowhere near the
    80-character limit that bites INPFIL.

    Set directly rather than symlinked into restart_swanfiles_dir_in, because
    the name PFM's lookup would want -- LV4_swan_rst_<t_fore - 1 day>_024 for a
    daily cycle -- is ALREADY OWNED by the operational day N-1 forecast's own
    t = 24 h restart.  Linking there would shadow real operational data.
    lv4_swan_ic_prebuilt then stops restart_setup running the lookup and
    replacing this with PFM's own choice.

    Returns the HOTSTART line, or None if no DA restart was found (SWAN then
    falls through to PFM's usual lookup, which is the right fallback).
    """
    import datetime as _dt

    if isinstance(t_fore, str):
        t_fore = _dt.datetime.strptime(t_fore, "%Y%m%d%H")

    for k in range(1, lookback_days + 1):
        t_prev = t_fore - _dt.timedelta(days=k)
        # three name forms, newest first -- see stage_swan_hotstart.  SWAN takes
        # the stem and appends the rank itself, so the _024 in the name is
        # inert here; it is carried only so the file says what it holds.
        stamp = t_prev.strftime("%Y%m%d%H%M")
        hrs = int(round((t_fore - t_prev).total_seconds() / 3600.0))
        stem = parts = None
        for nm in (f"LV4_DA_swan_rst_{stamp}_{hrs:03d}.dat",
                   f"LV4_DA_swan_rst_{stamp}.dat",
                   f"LV4_swan_rst_{stamp}.dat"):
            cand = sorted(C.SWAN_RST_DA_OUT.glob(nm + "-*"))
            if cand:
                stem, parts = C.SWAN_RST_DA_OUT / nm, cand
                break
        if not parts:
            continue
        txt = f"HOTSTART '{stem}'"
        PFM["swan_init_txt_full"] = txt
        PFM["lv4_swan_ic_prebuilt"] = True
        print(f"    LV4 swan: DA hotstart {stem.name} "
              f"({len(parts)} ranks, the DA's own wave field at "
              f"{t_fore:%Y-%m-%d %H}Z)")
        return txt

    print(f"    LV4 swan: no DA wave restart within {lookback_days} day(s) of "
          f"{t_fore:%Y-%m-%d %H}Z in {C.SWAN_RST_DA_OUT} -- leaving SWAN to "
          f"PFM's own restart lookup")
    return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="report whether PFM can start from the DA")
    ap.add_argument("t_fore", help="YYYYMMDDHH, the forecast init time")
    ap.add_argument("--no-build", action="store_true",
                    help="do not cut an IC from a DA history when IC_out lacks one")
    a = ap.parse_args()
    r = detect_da_ics(a.t_fore, allow_build=not a.no_build)
    print(r.text())
    raise SystemExit(0 if r.status == "ok" else (1 if r.status == "partial" else 2))
