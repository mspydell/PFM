#!/usr/bin/env python3
"""
da_sandbox.py -- keep a PFM test run out of the operational tree.

    apply_testing_paths(PFM)      redirect every WRITE path into the sandbox
    assert_not_operational(PFM)   refuse to run if any write path escaped

WHY THIS IS CLASSIFIED BY VALUE, NOT BY KEY.  The obvious design is a list of
keys to redirect.  It is also the design that fails quietly: the model-info dict
has 62 absolute-path keys today, somebody adds the 63rd, it is not in the list,
and a test run writes into the operational tree with nothing to say it did.  So
the rule is inverted -- any path under an operational root is a violation unless
it is on the read-only allowlist.  A new key is caught by default, which is the
direction an accident should fall.

READ-ONLY IS NOT REDIRECTED, DELIBERATELY.  Grids, executables, tide data, the
HYCOM/ECMWF/NWM caches and the CDIP archive are inputs.  A test run must read the
SAME inputs as the operational forecast or it is not testing the same thing, and
copying 1 TB of hycom to a sandbox would be absurd.  They are read, never
written, so sharing them is safe.

THE ONE THAT WOULD HURT MOST is lv4_his_web_dir, /projects/www-users/falk/... --
a test run that wrote there would publish test output to the public website.  It
is in OPERATIONAL_ROOTS for that reason and nothing else.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# where the sandbox lives
# ---------------------------------------------------------------------------
SANDBOX_SCRATCH = "/scratch/PFM_Simulations/DA_Sim/PFM_testing"
SANDBOX_ARCHIVE = "/dataSIO/PFM_Simulations/Archive/DA/PFM_testing"

# Trees a test run must never write to.  Prefix match on the resolved value.
OPERATIONAL_ROOTS = (
    "/scratch/PFM_Simulations/LV1_Forecast",
    "/scratch/PFM_Simulations/LV2_Forecast",
    "/scratch/PFM_Simulations/LV3_Forecast",
    "/scratch/PFM_Simulations/LV4_Forecast",
    "/scratch/PFM_Simulations/restart_roms_data_new",
    "/scratch/PFM_Simulations/restart_roms_data",
    "/scratch/PFM_Simulations/restart_swan_data",
    "/scratch/PFM_Simulations/cdip_data",
    "/dataSIO/PFM_Simulations/Archive",
    "/dataSIO/PFM_Simulations/Plots",
    "/projects/www-users",                 # the public website
)

# Inputs.  Read by both the operational forecast and a test run; never written.
# Longest-prefix wins, so Archive/cdip_ncs is allowed even though Archive is not.
READ_ONLY_OK = (
    "/scratch/PFM_Simulations/Grids",
    "/scratch/PFM_Simulations/executables",
    "/scratch/PFM_Simulations/tide_data",
    "/scratch/PFM_Simulations/hycom_data",
    "/scratch/PFM_Simulations/ecmwf_data",
    "/scratch/PFM_Simulations/nwm_ncs",
    "/dataSIO/PFM_Simulations/Archive/cdip_ncs",
    "/dataSIO/PFM_Simulations/Archive/LV1_BC_files",   # read as DA input
)

# How an operational path maps into the sandbox.  Longest prefix first.
_REDIRECT = (
    ("/dataSIO/PFM_Simulations/Archive", SANDBOX_ARCHIVE),
    ("/dataSIO/PFM_Simulations/Plots",   SANDBOX_ARCHIVE + "/Plots"),
    ("/projects/www-users/falk",         SANDBOX_ARCHIVE + "/www"),
    ("/scratch/PFM_Simulations",         SANDBOX_SCRATCH),
)


def _is_path(v) -> bool:
    return isinstance(v, str) and v.startswith("/")


def _allowed(v: str, sandbox: tuple[str, ...] = ()) -> bool:
    """True when this path is fine: a shared input, or already in the sandbox.

    The sandbox has to be checked explicitly, because SANDBOX_ARCHIVE lives
    UNDER /dataSIO/PFM_Simulations/Archive, which is itself an operational root.
    Without this the guard rejects paths it had just redirected correctly -- it
    did exactly that on first run, flagging all eight archive dirs as STILL BAD.
    Longest prefix wins, and the sandbox is always longer than the root it sits
    inside.
    """
    ok = READ_ONLY_OK + tuple(sandbox) + (SANDBOX_SCRATCH, SANDBOX_ARCHIVE)
    hits_ok = max((len(p) for p in ok if v.startswith(p)), default=-1)
    hits_op = max((len(p) for p in OPERATIONAL_ROOTS if v.startswith(p)), default=-1)
    return hits_ok >= hits_op          # ties favour allowed


def offenders(PFM: dict, *, sandbox: tuple[str, ...] = ()) -> dict:
    """Every key whose value still points into the operational tree."""
    bad = {}
    for k, v in PFM.items():
        if not _is_path(v):
            continue
        if any(v.startswith(p) for p in OPERATIONAL_ROOTS) and not _allowed(v, sandbox):
            bad[k] = v
    return bad


def apply_testing_paths(PFM: dict, *, scratch: str = SANDBOX_SCRATCH,
                        archive: str = SANDBOX_ARCHIVE,
                        verbose: bool = True) -> int:
    """Redirect every write path into the sandbox.  Returns how many moved.

    Mirrors the operational layout underneath the sandbox root, so LV4_Forecast/
    His/ stays LV4_Forecast/His/ -- a test tree that looks like the real one is
    far easier to reason about than a flattened one, and any code that builds a
    path by joining known components still works.
    """
    red = tuple((src, dst.replace(SANDBOX_SCRATCH, scratch)
                        .replace(SANDBOX_ARCHIVE, archive)) for src, dst in _REDIRECT)
    n = 0
    for k in sorted(PFM):
        v = PFM[k]
        if not _is_path(v) or _allowed(v, (scratch, archive)):
            continue
        for src, dst in red:            # longest prefix first, see _REDIRECT
            if v.startswith(src):
                PFM[k] = dst + v[len(src):]
                n += 1
                if verbose:
                    print(f"    {k}: -> {PFM[k]}")
                break
    PFM["testing"] = True
    return n


def assert_not_operational(PFM: dict, *, sandbox: tuple[str, ...] = ()) -> None:
    """Refuse to proceed if any write path is still operational.

    Called after apply_testing_paths and again right before anything runs.  It
    is cheap and the failure it prevents is not: a test run that writes into
    LV4_Forecast/His, or republishes web_data_latest.nc from a one-day test, is
    not something you notice from the log.
    """
    bad = offenders(PFM, sandbox=sandbox)
    if bad:
        lines = "\n".join(f"      {k} = {v}" for k, v in sorted(bad.items()))
        raise RuntimeError(
            f"refusing to run: {len(bad)} path(s) still point into the "
            f"OPERATIONAL tree.\n{lines}\n"
            f"    Run apply_testing_paths(PFM) first, or add the path to "
            f"READ_ONLY_OK in da_sandbox.py if it is genuinely an input.")


def ensure_sandbox_dirs(PFM: dict) -> int:
    """mkdir -p every sandbox directory the run will write to."""
    made = 0
    for k, v in PFM.items():
        if not _is_path(v) or _allowed(v):
            continue
        if not v.startswith((SANDBOX_SCRATCH, SANDBOX_ARCHIVE)):
            continue
        d = Path(v if k.endswith("_dir") else str(Path(v).parent))
        if not d.is_dir():
            d.mkdir(parents=True, exist_ok=True)
            made += 1
    return made


if __name__ == "__main__":
    import argparse
    import pickle
    ap = argparse.ArgumentParser(
        description="report which model-info paths point at the operational tree")
    ap.add_argument("pkl")
    ap.add_argument("--redirect", action="store_true",
                    help="also show what apply_testing_paths would do")
    a = ap.parse_args()
    with open(a.pkl, "rb") as fh:
        MI = pickle.load(fh)
    bad = offenders(MI)
    print(f"  {len(bad)} operational write path(s) in {a.pkl}")
    for k, v in sorted(bad.items()):
        print(f"    {k:30s} {v}")
    if a.redirect:
        print("  after apply_testing_paths:")
        apply_testing_paths(MI)
        left = offenders(MI)
        print(f"  remaining operational paths: {len(left)}")
        for k, v in sorted(left.items()):
            print(f"    STILL BAD {k:28s} {v}")
