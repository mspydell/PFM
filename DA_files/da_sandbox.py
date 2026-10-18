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


# Output-name tag for a testing run.
#
# PATH ISOLATION IS NOT ENOUGH, which 2026-10-17 proved.  The sandbox path was
# correct and offenders() reported zero leaks, but link_restarts_into_sandbox had
# symlinked the OPERATIONAL restart for this t_fore into the sandbox under its own
# name -- and RSTNAME is that same name in that same directory.  ROMS opened it
# for writing and wrote STRAIGHT THROUGH the symlink into the operational file,
# replacing a 20-record 15.5 GB forecast restart with a 4-record 3.1 GB one.  The
# next operational cycle then initialised from it: the time was right (record 4 is
# t_fore + 24 h) so nothing failed, and all four levels silently started from
# DA-derived test fields.
#
# A name tag cannot fail that way.  A testing run writes LV1_ocean_rst_<...>_testing.nc,
# which no operational file is ever called, so no symlink can match it and no
# operational reader can pick it up.  Isolation stops depending on every path
# being right and starts being structural.
#
# APPENDED, not inserted.  PFM parses stamps out of these names at FIXED OFFSETS
# -- tail[14:24] in get_restart_file_and_index, tail[13:23] in
# get_swan_restart_file_name -- so '..._testing.nc' keeps every offset valid and
# a testing run can still find its own restarts, while 'LV1_testing_ocean_rst_'
# would shift the stamp and break them.
TESTING_TAG = "testing"


def tag_testing_output_names(PFM: dict, tag: str = TESTING_TAG,
                             verbose: bool = True) -> int:
    """Append `_<tag>` to this run's OUTPUT file names.  Returns how many changed.

    Only the names a run WRITES: histories, ROMS restarts and the SWAN restart.
    Inputs are left alone -- a testing run reads operational and DA files by their
    real names, which is the point.

    Safe to call at pickle-build time for the forecast path: nothing rebuilds
    these keys afterwards (add_file_names_2MI is only reached from the hindcast
    driver and the DA bridge).  If that ever changes, this has to move after it,
    the way the DA's own rename had to.
    """
    suffix = "_" + tag
    n = 0
    for lv in ("1", "2", "3", "4"):
        for key, dirkey in ((f"lv{lv}_his_name", f"lv{lv}_his_dir"),
                            (f"lv{lv}_rst_name", "restart_ncfiles_dir_out")):
            nm = PFM.get(key)
            if not isinstance(nm, str) or not nm or suffix + "." in nm:
                continue
            stem, dot, ext = nm.rpartition(".")
            PFM[key] = (stem + suffix + dot + ext) if dot else nm + suffix
            n += 1
            d = PFM.get(dirkey)
            if isinstance(d, str) and d:
                PFM[key + "_full"] = str(Path(d) / PFM[key])
    nm = PFM.get("lv4_swan_rst_name")
    if isinstance(nm, str) and nm and suffix + "." not in nm:
        stem, dot, ext = nm.rpartition(".")
        PFM["lv4_swan_rst_name"] = (stem + suffix + dot + ext) if dot else nm + suffix
        n += 1
        d = PFM.get("restart_swanfiles_dir_out")
        if isinstance(d, str) and d:
            PFM["lv4_swan_rst_name_full"] = str(Path(d) / PFM["lv4_swan_rst_name"])
    PFM["run_tag"] = tag          # read by may_use_output in init_funs_forecast
    if verbose:
        print(f"    tagged {n} output name(s) with '{tag}': "
              f"{PFM.get('lv1_rst_name')}, {PFM.get('lv4_swan_rst_name')}")
    return n


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


def ensure_sandbox_dirs(PFM: dict, *, sandbox: tuple[str, ...] = (),
                       verbose: bool = False) -> int:
    """mkdir -p every sandbox directory the run will write to.

    Selects paths that ARE in the sandbox.  It must NOT filter on _allowed():
    after apply_testing_paths every write path is in the sandbox, and _allowed
    returns True for exactly those -- so an earlier version that skipped
    "allowed" paths skipped all of them and silently created nothing.

    That was not a harmless no-op.  PFM makes its own Forc directories as a side
    effect of writing forcing, but not Run, His or Plots, so the 2026-10-12 test
    run got: no Run dir -> no .in file -> no sbatch -> no LV1/LV2/LV3 jobs at
    all; no His dir -> LV3_ocean_his_*.nc never appeared -> LV4's BC failed ->
    BC_LV4_tmp_file.pkl was missing -> clm failed -> "aborting simulation".  One
    missing mkdir, and every level failed for a different-looking reason.
    """
    roots = tuple(sandbox) + (SANDBOX_SCRATCH, SANDBOX_ARCHIVE)
    made = 0
    for k, v in sorted(PFM.items()):
        if not _is_path(v) or not v.startswith(roots):
            continue
        d = Path(v if k.endswith("_dir") else str(Path(v).parent))
        if not d.is_dir():
            d.mkdir(parents=True, exist_ok=True)
            made += 1
            if verbose:
                print(f"    mkdir {d}")
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


def link_restarts_into_sandbox(PFM: dict, t_fore, *, verbose: bool = True) -> dict:
    """Symlink the restart files valid at t_fore into the sandbox.

    Why this is needed even when every level starts from a DA initial condition:
    use_restart = 0 settles the OCEAN, but SWAN is initialised independently and
    still wants its hotstart.  With an empty sandbox restart dir the .in
    generator falls back to swan_init_txt_full = 'ZERO', and SWAN cold-starts
    from a flat sea -- which takes longer than a one-day test window to spin up,
    so the waves are meaningless and the test proves nothing about LV4.

    These are the FALLBACK restarts, for the case where the DA produced nothing
    and PFM has to start the way it always has.  When the DA ICs ARE used,
    LV4's wave hotstart comes from the DA's own run instead -- see
    da_ic_handoff.apply_da_swan_hotstart, which sets swan_init_txt_full directly
    and does not go through this directory at all.

    The SWAN files linked here are the ones stamped EARLIER than t_fore, not at
    it.  PFM hotstarts SWAN from the restart whose stamp PLUS hour offset equals
    t_fore -- _024 of the previous day for a daily cycle -- and restart_setup
    calls remove_swan_restarts_eq_foretime FIRST, deleting everything stamped at
    t_fore precisely to force that lookback.  Linking only the t_fore stamp was
    therefore self-defeating: the sandbox linked 1368 files, PFM deleted all
    1368, the glob for '*_???.dat-001' found nothing, and SWAN cold-started from
    ZERO in try6, LV4only and LV4only2 alike.

    Symlinks, not copies: the SWAN set is ~2.5 GB per stamp and is only read.
    Removing a symlink never touches the source, so PFM's own restart
    bookkeeping can delete in the sandbox as freely as it likes.
    """
    import datetime as _dt
    if isinstance(t_fore, str):
        t_fore = _dt.datetime.strptime(t_fore, "%Y%m%d%H")
    stamp = t_fore.strftime("%Y%m%d%H%M")

    swan_pats = []
    for k in range(1, 4):
        back = (t_fore - _dt.timedelta(days=k)).strftime("%Y%m%d%H%M")
        swan_pats.append("LV4_swan_rst_%s*" % back)

    out = {}
    pairs = (("restart_swanfiles_dir_in", "/scratch/PFM_Simulations/restart_swan_data",
              swan_pats),
             ("restart_ncfiles_dir_in", "/scratch/PFM_Simulations/restart_roms_data_new",
              ["*_rst_%s*" % stamp]))
    for key, oper, pats in pairs:
        dest = PFM.get(key)
        if not dest or not dest.startswith((SANDBOX_SCRATCH, SANDBOX_ARCHIVE)):
            continue                      # not sandboxed; nothing to do
        d = Path(dest)
        d.mkdir(parents=True, exist_ok=True)
        new = have = 0
        for pat in pats:
            for src in sorted(Path(oper).glob(pat)):
                link = d / src.name
                if link.is_symlink() or link.exists():
                    have += 1
                    continue
                link.symlink_to(src)
                new += 1
        out[key] = new + have
        if verbose:
            print("    %s: %d restart file(s) available (%d newly linked)"
                  % (key, new + have, new))
            # Warn on TOTAL, not on how many were new.  Counting only new links
            # made a fully-populated directory report "0 ... none matched",
            # which reads as a failure when it is the opposite.
            if new + have == 0:
                what = ("SWAN will cold-start from ZERO and its waves will not "
                        "be meaningful in a short test"
                        if "swan" in key else
                        "any level needing a restart will have none")
                print("      NOTE: nothing matched %s in %s -- %s"
                      % (" or ".join(pats), oper, what))
    return out


def stage_static_run_inputs(PFM: dict, *, verbose: bool = True) -> int:
    """Symlink the static files ROMS expects to find in each level's Run dir.

    VARNAME points at <level>/Run/LV<n>_varinfo.yaml, and LV4's coupled .in also
    names coawst_mss.yaml.  These are fixed inputs dated 2026-07-07 that nothing
    generates -- they simply live in the operational Run directories.  A sandbox
    Run dir is created empty, so ROMS aborts in initialisation:

        YAML_TREE_CREATE - Unable to open input YAML file: ...LV1_varinfo.yaml

    and the job exits in about a minute having written nothing.  It reports
    COMPLETED, which is what makes this one hard to spot: on 2026-10-13 LV1, LV2
    and LV3 all "succeeded" in ~60 s with no history at all.

    Symlinked, not copied: they are read-only inputs, and a test must use the
    SAME metadata as the forecast or it is not testing the same configuration.
    """
    n = 0
    for lv in ("LV1", "LV2", "LV3", "LV4"):
        dst = PFM.get("%s_run_dir" % lv.lower())
        if not dst or not str(dst).startswith((SANDBOX_SCRATCH, SANDBOX_ARCHIVE)):
            continue                      # not sandboxed; operational dirs already have them
        # map the sandbox path back to the operational one it mirrors
        src = Path(str(dst).replace(SANDBOX_SCRATCH, "/scratch/PFM_Simulations"))
        d = Path(dst)
        d.mkdir(parents=True, exist_ok=True)

        # Clear the PREVIOUS run's SWAN Errfiles.  SWAN writes one per rank only
        # when it is dying, so their mere presence reads as "this run failed" --
        # and they are not truncated or removed by the next run, so a successful
        # run inherits the last failure's 72 files and looks broken.  That cost
        # real time twice on 2026-10-14/15: the LV4 forecast of 10/13 completed
        # clean with zero Errfiles of its own, while 72 from the 10/14 failure
        # sat beside it, and the only way to tell was comparing mtimes.
        #
        # Errfiles only.  PRINT01-* is SWAN's actual log and is rewritten every
        # run, so deleting it would lose the previous run's diagnostics for no
        # gain -- and unlike an Errfile its existence means nothing either way.
        stale = sorted(d.glob("Errfile*"))
        for f in stale:
            try:
                f.unlink()
            except OSError:
                pass
        if stale and verbose:
            print("    %s: cleared %d stale SWAN Errfile(s) from a previous run"
                  % (lv, len(stale)))

        if not src.is_dir():
            if verbose:
                print("    %s: no operational Run dir at %s to copy inputs from"
                      % (lv, src))
            continue
        for y in sorted(src.glob("*.yaml")):
            link = d / y.name
            if link.is_symlink() or link.exists():
                continue
            link.symlink_to(y)
            n += 1
            if verbose:
                print("    %s: linked %s" % (lv, y.name))
    if verbose and n == 0:
        print("    static Run inputs: already present")
    return n
