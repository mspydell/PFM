"""
da_manifest.py — the record of what a DA pre-step did, and whether it worked.

One JSON per t_fore.  Two jobs:

  1. Resume.  A 4.5 h pipeline fails partway often enough that re-running must
     skip what already succeeded.  The manifest is what makes a stage skippable
     without re-deriving its outputs.
  2. The verdict.  Phase 2 reads exactly one value from this file --
     verdict == "ok" -- to decide whether the forecast uses the DA initial
     conditions or its usual restart files.  Nothing else in PFM needs to know
     how the DA works.

verdict is "ok" only when ALL FOUR levels produced a valid t=24 h IC.  A mix of
DA and restart ICs across the nest would be inconsistent, so there is no partial
success as far as Phase 2 is concerned.

There IS a separate "partial" verdict, but it means something different: a
deliberately-requested SUBSET of stages ran and succeeded, as when driving stages
by hand.  Phase 2 gates on == "ok", so "partial" never hands over an IC.
"""
from __future__ import annotations

import datetime as DT
import json
import os
import subprocess
from pathlib import Path

import da_config as C

SCHEMA = 1


def _git_hash(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10
                              ).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


class Manifest:
    def __init__(self, t_fore: str):
        self.t_fore = t_fore
        self.path = C.SCRATCH / "Log" / f"{t_fore}.manifest.json"
        self.data = self._load() or self._new()

    # -- io ----------------------------------------------------------------
    def _new(self) -> dict:
        tf, td = C.parse_t_fore(self.t_fore)
        return {
            "schema": SCHEMA,
            "t_fore": self.t_fore,
            "t_da": td.strftime("%Y%m%d%H"),
            "started": DT.datetime.now().isoformat(timespec="seconds"),
            "host": os.uname().nodename,
            "stages": {},
            "ic_out": {},
            "verdict": "pending",
            "provenance": {
                "pfm_commit": _git_hash(C.PFM_ROOT),
                "blend_space": C.BLEND_SPACE,
                "hfr_res": C.HFR_RES,
            },
        }

    def _load(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            d = json.loads(self.path.read_text())
            return d if d.get("schema") == SCHEMA else None
        except Exception:  # noqa: BLE001
            return None

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str))
        tmp.replace(self.path)          # atomic: a crash mid-write must not
                                        # leave an unparseable manifest

    # -- stages ------------------------------------------------------------
    def status(self, stage: str) -> str:
        return self.data["stages"].get(stage, {}).get("status", "pending")

    def done(self, stage: str) -> bool:
        """Was this stage successful?  Used for the final verdict only.

        NOT used to decide whether to run a stage -- see the note in
        run_PFM_DA.  Each stage re-checks its own outputs on disk, which also
        repairs a missing archive copy.
        """
        return self.status(stage) in ("ok", "skipped")

    def record(self, stage: str, *, status: str, wall_s: float | None = None,
               outputs: dict | None = None, evidence: str = "",
               error: str = "") -> None:
        """outputs maps a label to {'scratch': path, 'archive': path}.

        Both paths are kept so a diagnostic months later can still find a file
        after the scratch copy has aged out of the ~1 week window.
        """
        e = self.data["stages"].setdefault(stage, {})
        e.update(status=status,
                 finished=DT.datetime.now().isoformat(timespec="seconds"))
        if wall_s is not None:
            e["wall_s"] = round(wall_s, 1)
            exp = C.EXPECTED_WALL_S.get(stage)
            if exp:
                e["vs_expected"] = round(wall_s / exp, 2)
        if outputs:
            e["outputs"] = {k: {kk: str(vv) for kk, vv in v.items()}
                            for k, v in outputs.items()}
        if evidence:
            e["evidence"] = evidence
        if error:
            e["error"] = error
        self.save()

    def set_ic(self, level: str, scratch: Path, archive: Path) -> None:
        self.data["ic_out"][level.lower()] = {"scratch": str(scratch),
                                              "archive": str(archive)}
        self.save()

    def finalize(self, verdict: str, note: str = "") -> str:
        """verdict 'ok' requires all four handoff ICs to be recorded."""
        if verdict == "ok":
            missing = [lv for lv in ("lv1", "lv2", "lv3", "lv4")
                       if lv not in self.data["ic_out"]]
            if missing:
                verdict = "failed"
                note = (note + f" (no IC for {','.join(missing)})").strip()
        self.data["verdict"] = verdict
        self.data["note"] = note
        self.data["finished"] = DT.datetime.now().isoformat(timespec="seconds")
        self.save()
        try:
            from da_validate import archive_copy
            archive_copy(self.path, C.ARCHIVE / "manifests")
        except Exception:  # noqa: BLE001
            pass                        # the manifest is already safe on scratch
        return verdict

    # -- the one thing Phase 2 will call -----------------------------------
    @staticmethod
    def verdict_for(t_fore: str) -> str:
        """Read-only: 'ok' | 'partial' | 'failed' | 'timeout' | 'pending' | 'absent'.

        Only 'ok' means the whole pipeline ran and every level produced an IC.
        'partial' means a deliberately-requested subset of stages succeeded --
        a normal result when running stages by hand, and NOT a reason to use the
        DA initial conditions.

        Looks in the archive first so it still answers after scratch ages out.
        """
        for p in (C.ARCHIVE / "manifests" / f"{t_fore}.manifest.json",
                  C.SCRATCH / "Log" / f"{t_fore}.manifest.json"):
            if p.exists():
                try:
                    return json.loads(p.read_text()).get("verdict", "pending")
                except Exception:  # noqa: BLE001
                    return "pending"
        return "absent"

    def summary(self) -> str:
        rows = []
        for k, v in self.data["stages"].items():
            w = v.get("wall_s")
            rows.append(f"    {k:14s} {v.get('status','?'):8s} "
                        f"{(str(round(w/60,1))+' min') if w else '':>10s}  "
                        f"{v.get('evidence', v.get('error',''))[:54]}")
        return (f"  manifest {self.path}\n"
                f"  t_fore {self.data['t_fore']}  t_da {self.data['t_da']}  "
                f"verdict {self.data['verdict']}\n" + "\n".join(rows))
