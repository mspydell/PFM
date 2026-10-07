#!/usr/bin/env python3
"""
run_PFM_DA.py — the DA pre-step for a PFM forecast.

    run_PFM_DA(t_fore)        t_fore = 'YYYYMMDDHH', the FORECAST init time

Runs a 24 h LV1 4D-Var cycle over [t_fore - 24 h, t_fore], downscales the
analysis increment onto LV2/LV3/LV4, runs each child forward for that day, and
leaves four t = 24 h initial conditions in IC_out/.  Because the window ends at
t_fore, the t = 24 h state IS the state at t_fore -- exactly the IC the forecast
needs.  That identity is why the entry point takes the forecast time and derives
the DA time rather than the other way round.

PHASE 1 SCOPE.  This runs by hand on a past date and changes NOTHING in PFM.
There is no IC override, no flag in pfm_operational_input.py, no cron change.
The forecast is bit-unaffected: it reads none of what this writes.  Phase 2 adds
a single consumer of one value -- Manifest.verdict_for(t_fore) == 'ok'.

    python run_PFM_DA.py 2026030800
    python run_PFM_DA.py 2026030800 --stages lv1_ic,lv1_atm --dry-run
    python run_PFM_DA.py 2026030800 --status
"""
from __future__ import annotations

import argparse
import datetime as DT
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import da_config as C          # noqa: E402
import da_stages as S          # noqa: E402
from da_manifest import Manifest   # noqa: E402


def run_PFM_DA(t_fore: str, *, stages: list[str] | None = None,
               dry_run: bool = False, deadline_h: float | None = None,
               purge: bool = True, allow_download: bool = True) -> dict:
    tf, td = C.parse_t_fore(t_fore)
    C.ensure_tree()

    exp, dl, budget_msg = C.budget_check()
    deadline = DT.datetime.now() + DT.timedelta(
        hours=deadline_h if deadline_h is not None else dl)

    man = Manifest(t_fore)
    ctx = S.Ctx(t_fore=tf, t_da=td, man=man, deadline=deadline, dry_run=dry_run,
                allow_download=allow_download)
    todo = stages or list(S.STAGES)

    print(f"  PFM DA pre-step")
    print(f"    t_fore   {tf:%Y-%m-%d %H}Z   (forecast init)")
    print(f"    t_da     {td:%Y-%m-%d %H}Z   (DA window start; ends at t_fore)")
    print(f"    scratch  {C.SCRATCH}")
    print(f"    archive  {C.ARCHIVE}")
    print(f"    budget   {budget_msg}")
    print(f"    deadline {deadline:%H:%M:%S}")
    print(f"    forcing  read from disk; "
          + ("gaps downloaded if the forecast did not fetch them"
             if allow_download else "DOWNLOADS DISABLED (--no-download)"))
    if dry_run:
        print(f"    DRY RUN -- nothing will be produced")

    if purge and not dry_run:
        n = S.purge_scratch()
        if n:
            print(f"    purged {n} file(s) older than {C.SCRATCH_RETAIN_DAYS} d "
                  f"from Forc/ and His/")

    t_all = time.time()
    for name in todo:
        # NOTE: deliberately no manifest-level short-circuit here.  Each stage
        # decides for itself whether its output already exists and validates,
        # and that check also REPAIRS -- it re-archives a keeper whose archive
        # copy is missing.  Trusting the manifest instead meant a stage recorded
        # as "skipped" never ran again, so an output that had been produced but
        # not archived stayed unarchived forever.  The manifest is a record, not
        # control flow; the files on disk are the truth.
        if ctx.over_deadline():
            man.record(name, status="timeout",
                       error="deadline passed before this stage started")
            print(f"  [{name}] DEADLINE PASSED -- abandoning the pre-step")
            return {"verdict": man.finalize("timeout",
                                            f"deadline at {name}"),
                    "manifest": str(man.path)}
        t0 = time.time()
        print(f"  [{name}] starting ({ctx.left_h():.2f} h left)", flush=True)
        try:
            outputs = S.FUNCS[name](ctx) or {}
            man.record(name, status="ok", wall_s=time.time() - t0,
                       outputs={k: {kk: str(vv) for kk, vv in v.items()}
                                for k, v in outputs.items()
                                if isinstance(v, dict)},
                       evidence="produced and validated")
            print(f"  [{name}] ok ({(time.time()-t0)/60:.1f} min)")
        except S.StageSkip as e:
            man.record(name, status="skipped", evidence=f"already valid: {e}")
            print(f"  [{name}] skipped -- output already valid")
        except NotImplementedError as e:
            man.record(name, status="not_implemented", error=str(e))
            print(f"  [{name}] NOT IMPLEMENTED")
            print(f"           {e}")
            return {"verdict": man.finalize("failed", f"{name} not implemented"),
                    "manifest": str(man.path)}
        except Exception as e:  # noqa: BLE001
            man.record(name, status="failed", wall_s=time.time() - t0,
                       error=f"{type(e).__name__}: {e}")
            print(f"  [{name}] FAILED: {type(e).__name__}: {e}")
            return {"verdict": man.finalize("failed", f"{name}: {e}"),
                    "manifest": str(man.path)}

    verdict = man.finalize("ok" if all(man.done(s) for s in S.STAGES) else "failed")
    print(f"\n  verdict: {verdict}   ({(time.time()-t_all)/3600:.2f} h)")
    print(f"  manifest: {man.path}")
    return {"verdict": verdict, "manifest": str(man.path),
            "ic_out": man.data.get("ic_out", {})}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("t_fore", help="YYYYMMDDHH, the FORECAST init time")
    ap.add_argument("--stages", default=None,
                    help="comma-separated subset, e.g. lv1_ic,lv1_atm")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline-h", type=float, default=None)
    ap.add_argument("--no-purge", action="store_true")
    ap.add_argument("--status", action="store_true",
                    help="print the manifest for this t_fore and exit")
    ap.add_argument("--check", action="store_true",
                    help="run the input preflight and exit (reads nothing else, "
                         "downloads nothing; exit 0 = every input is on disk)")
    ap.add_argument("--no-download", action="store_true",
                    help="never download: fail if any forcing is missing from "
                         "disk.  The default is to read from disk and fetch only "
                         "a genuine gap -- which happens when the PFM cycle for "
                         "this window did not run.")
    a = ap.parse_args()

    if a.status:
        m = Manifest(a.t_fore)
        print(m.summary())
        return 0

    if a.check:
        import da_pfm_bridge as B          # noqa: PLC0415
        import da_preflight as PF          # noqa: PLC0415
        tf, td = C.parse_t_fore(a.t_fore)
        C.ensure_tree()
        pkl = B.build_minfo(tf, td)
        rep = PF.check_inputs(tf, td, pkl)
        print(f"  input preflight for t_fore {tf:%Y-%m-%d %H}Z "
              f"(window {td:%Y-%m-%d %H}Z .. {tf:%Y-%m-%d %H}Z)")
        print(rep.text())
        if rep.blocking:
            print(f"\n  BLOCKED: {len(rep.blocking)} input(s) that no download "
                  f"can fix.")
            return 2
        if rep.downloadable:
            print(f"\n  {len(rep.downloadable)} input(s) not fully on disk; a run "
                  f"would fetch just the gap (use --no-download to forbid that).")
            return 1
        print(f"\n  all inputs present on disk; a run would download nothing.")
        return 0

    st = a.stages.split(",") if a.stages else None
    if st:
        bad = [s for s in st if s not in S.STAGES]
        if bad:
            ap.error(f"unknown stage(s): {bad}.  Valid: {list(S.STAGES)}")
    r = run_PFM_DA(a.t_fore, stages=st, dry_run=a.dry_run,
                   deadline_h=a.deadline_h, purge=not a.no_purge,
                   allow_download=not a.no_download)
    return 0 if r["verdict"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
