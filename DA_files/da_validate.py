"""
da_validate.py — "did this actually work?" checks.

Existence is not evidence.  Two failures this project has already had turn on
that: a staged BC file was silently reused because it was the same SIZE as the
regenerated one, and a chain's idempotent skip would have treated a truncated
history as a finished day.  So every check here looks at content, and the
success of a ROMS job is judged from its outputs rather than its exit code.
"""
from __future__ import annotations

import datetime as DT
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import netCDF4 as nc
except ImportError:  # pragma: no cover
    nc = None

ROMS_EPOCH = DT.datetime(1999, 1, 1)
FILL = 1e30


@dataclass
class Check:
    ok: bool
    why: str = ""
    detail: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok


def _san(a):
    return np.where(np.abs(a) > FILL, np.nan, a)


def _wet_mask(grid: Path):
    """mask_rho from a ROMS grid, or None."""
    if nc is None or grid is None:
        return None
    try:
        with nc.Dataset(grid) as g:
            if "mask_rho" not in g.variables:
                return None
            return np.asarray(g["mask_rho"][:]) > 0
    except Exception:  # noqa: BLE001
        return None


def nc_ok(path: Path, *, min_bytes: int = 1 << 20,
          need_vars: tuple[str, ...] = (),
          expect_time: DT.datetime | None = None,
          time_var: str = "ocean_time",
          finite_vars: tuple[str, ...] = (),
          water_range: dict | None = None,
          grid: Path | None = None,
          attrs: dict | None = None) -> Check:
    """Open a NetCDF file and prove it is what we think it is.

    expect_time is checked against ROMS's days-or-seconds-since-1999 convention
    (the units string decides which), to a 60 s tolerance -- enough to catch a
    wrong day, loose enough to survive float32 round-off in a restart.
    """
    path = Path(path)
    if not path.exists():
        return Check(False, f"missing: {path}")
    sz = path.stat().st_size
    if sz < min_bytes:
        return Check(False, f"too small ({sz} B < {min_bytes} B): {path.name}")
    if nc is None:
        return Check(True, "netCDF4 unavailable; size-only check")
    wet = _wet_mask(grid) if (water_range and grid) else None
    try:
        with nc.Dataset(path) as d:
            missing = [v for v in need_vars if v not in d.variables]
            if missing:
                return Check(False, f"{path.name}: missing {missing}")
            if attrs:
                for k, want in attrs.items():
                    got = getattr(d, k, None)
                    if got != want:
                        return Check(False,
                                     f"{path.name}: attr {k}={got!r}, want {want!r}")
            if expect_time is not None and time_var in d.variables:
                tv = np.asarray(d[time_var][:]).ravel()
                if tv.size == 0:
                    return Check(False, f"{path.name}: empty {time_var}")
                units = getattr(d[time_var], "units", "")
                scale = 86400.0 if units.strip().lower().startswith("day") else 1.0
                got = ROMS_EPOCH + DT.timedelta(seconds=float(tv[0]) * scale)
                if abs((got - expect_time).total_seconds()) > 60:
                    return Check(False,
                                 f"{path.name}: {time_var}[0]={got:%Y-%m-%d %H:%M}, "
                                 f"want {expect_time:%Y-%m-%d %H:%M}")
            for v in finite_vars:
                if v not in d.variables:
                    continue
                a = np.asarray(d[v][:])
                # Check the RAW array.  ROMS writes ~1e37 on land, which IS
                # finite -- sanitising it to NaN first and then testing
                # finiteness just rediscovers the land mask.  (It did: an LV1 IC
                # was rejected for "698,120 non-finite", which was exactly
                # 17,453 land cells x 40 levels.)  Genuine corruption is NaN or
                # Inf, and that is what this catches.
                if not np.isfinite(a).all():
                    bad = int((~np.isfinite(a)).sum())
                    return Check(False, f"{path.name}: {v} has {bad:,} NaN/Inf")
                fin = a[np.abs(a) <= FILL]
                if fin.size == 0:
                    return Check(False, f"{path.name}: {v} is entirely fill")
                lo, hi = water_range.get(v, (None, None)) if water_range else (None, None)
                if lo is not None:
                    # water_range means WATER.  Land and dried cells carry values
                    # that are meaningless to range-check: on LV4, which has
                    # WET_DRY, zeta on land sits near -h, and a handoff IC was
                    # rejected for "zeta out of range [-16.2, 1.9]" when all 405
                    # offending cells were land and every wet cell was within
                    # -0.50..1.32.  With no grid supplied the check falls back to
                    # all finite values, which is the old behaviour.
                    vals = fin
                    if wet is not None and a.ndim >= 2 and a.shape[-2:] == wet.shape:
                        m = np.broadcast_to(wet, a.shape)
                        sel = a[m]
                        vals = sel[np.abs(sel) <= FILL]
                    if vals.size and (vals.min() < lo or vals.max() > hi):
                        return Check(False, f"{path.name}: {v} out of range "
                                            f"[{vals.min():.3g}, {vals.max():.3g}] "
                                            f"not within [{lo}, {hi}]"
                                            + ("" if wet is None
                                               else " (water cells only)"))
    except Exception as e:  # noqa: BLE001
        return Check(False, f"{path.name}: unreadable ({type(e).__name__}: {e})")
    return Check(True, "", {"bytes": sz})


def roms_job_ok(log: Path, keepers: tuple[Path, ...], started: float,
                rc: int, min_bytes: int = 10 << 20) -> Check:
    """Did a ROMS job succeed?  Do NOT trust the slurm exit state.

    LV1 4D-Var jobs routinely land in slurm state FAILED at ~3h13 because
    MVAPICH2 raises SIGTERM in MPI_Finalize after the model is done, and an HDF5
    shutdown SEGV can eat the DONE line as well.  Three cases, matching
    run_da_cycle_v2.sh:

      a. rc == 0
      b. 'ROMS/TOMS: DONE' in the log AND the keepers are fresh
      c. no DONE, but every keeper is present, newer than the job start, and
         over a sane size floor -- the outputs are on disk and valid
    """
    log = Path(log)
    fresh = []
    for k in keepers:
        k = Path(k)
        if not k.exists():
            return Check(False, f"keeper missing: {k.name}", {"rc": rc})
        if k.stat().st_mtime <= started:
            return Check(False, f"keeper stale (predates job start): {k.name}",
                         {"rc": rc})
        if k.stat().st_size < min_bytes:
            return Check(False,
                         f"keeper too small: {k.name} "
                         f"({k.stat().st_size} B < {min_bytes} B)", {"rc": rc})
        fresh.append(k.name)

    if rc == 0:
        return Check(True, "rc=0", {"keepers": fresh})
    done = log.exists() and "ROMS/TOMS: DONE" in log.read_text(errors="ignore")
    if done:
        return Check(True, "rc!=0 but ROMS/TOMS: DONE + fresh keepers "
                           "(MVAPICH2 MPI_Finalize SIGTERM)",
                     {"rc": rc, "keepers": fresh})
    return Check(True, "rc!=0, no DONE, but all keepers fresh and sane "
                       "(probably HDF5 shutdown SEGV)",
                 {"rc": rc, "keepers": fresh})


def copied_ok(src: Path, dst: Path) -> Check:
    """Verify an archive copy landed, by size -- not by cp's exit code.

    /dataSIO throughput has been erratic, and a stage must not report success
    with nothing archived.
    """
    src, dst = Path(src), Path(dst)
    if not dst.exists():
        return Check(False, f"archive copy absent: {dst}")
    a, b = src.stat().st_size, dst.stat().st_size
    if a != b:
        return Check(False, f"archive copy size mismatch: {a} != {b} ({dst.name})")
    return Check(True, "", {"bytes": b})


def archive_copy(src: Path, dst_dir: Path, name: str | None = None) -> Path:
    """Copy a keeper to the archive immediately and verify it.  Raises on failure.

    Copy, not move: the scratch copy stays for ~a week so a bad night can be
    debugged, and the archive is safe the moment the stage finishes rather than
    at the end of a 4.5 h pipeline that may never reach its last stage.
    """
    src = Path(src)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (name or src.name)
    subprocess.run(["cp", "-f", str(src), str(dst)], check=True)
    chk = copied_ok(src, dst)
    if not chk:
        raise RuntimeError(f"archive copy failed: {chk.why}")
    return dst


def ensure_archived(src: Path, dst_dir: Path, name: str | None = None,
                    expect_time: DT.datetime | None = None) -> Path:
    """Idempotent archive_copy: copy only if the archive copy is absent or the
    wrong size.

    Must be called on the SKIP path as well as the produce path.  Otherwise a
    run that produces an output and then dies before archiving leaves a valid
    scratch file that every later run skips -- and never archives.  That is not
    hypothetical: it happened on the first LV1 IC, which sat valid on scratch
    with an empty archive because the run failed in validation after writing it.
    """
    src, dst_dir = Path(src), Path(dst_dir)
    dst = dst_dir / (name or src.name)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        # Size alone is not enough when the file is a date-tagged copy of a
        # FIXED-name output.  Every LV1 ocean_fwd_outer0/outer1 is exactly the
        # same size every cycle (same grid, same 25 records), so once
        # LV1_fwd_outer0_<ymd>.nc exists the size test matches forever and a
        # later cycle's data is never written -- the archive silently keeps the
        # FIRST cycle that ever landed on that name.
        #
        # That happened: LV1_fwd_outer0_20261007.nc and _outer1_20261007.nc both
        # held the 10/06 window, and the real 10/07 fields were overwritten in
        # tmp_input_output by the next cycle before anyone noticed.  roms_mod.nc
        # escaped only because its size varies with the observation count.
        #
        # So when the caller says which window this copy is for, check it.
        if expect_time is None or _first_time_is(dst, expect_time):
            return dst
        print(f"    re-archiving {dst.name}: it holds a different window than "
              f"{expect_time:%Y-%m-%d %H}Z (same size, so the size test passed)")
    return archive_copy(src, dst_dir, name)


def _first_time_is(path: Path, want: DT.datetime, tol_s: float = 60.0) -> bool:
    """True when the file's first ocean_time record is `want`."""
    try:
        import netCDF4 as nc
        import numpy as np
        with nc.Dataset(path) as d:
            if "ocean_time" not in d.variables:
                return True            # nothing to check against
            tv = np.asarray(d["ocean_time"][:]).ravel().astype(float)
            units = getattr(d["ocean_time"], "units", "")
        if tv.size == 0:
            return False
        scale = 86400.0 if units.strip().lower().startswith("day") else 1.0
        got = ROMS_EPOCH + DT.timedelta(seconds=float(tv[0]) * scale)
        return abs((got - want).total_seconds()) <= tol_s
    except Exception:
        return True                    # unreadable: let archive_copy decide
