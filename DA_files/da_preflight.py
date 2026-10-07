"""
da_preflight.py — "do we have everything we need?", answered before 3 h of 4D-Var.

PFM_DA normally reads forcing the forecast already fetched, so it should not
download anything.  But "normally" is not "always", and the gap is worth stating
because it is the case that will actually bite:

    PFM Jan-2 00Z fails outright.  Someone reruns it by hand as PFM Jan-2 06Z the
    next morning.  The next scheduled run is PFM Jan-3 00Z, and before it
    PFM_DA runs the window Jan-2 00Z -> Jan-3 00Z.  Nothing was fetched FOR
    Jan-2 00Z, and there is no PFM forecast initialised at Jan-2 00Z to take an
    initial condition from.

So the policy is: **read from disk; verify first; download only the gap.**  This
module is the verify step.  It checks every input a cycle needs, says exactly
what is missing, and says whether the gap is fixable by downloading.  Running it
is cheap -- it opens at most a couple of files -- and it runs before the
expensive stages so a doomed cycle fails in seconds rather than after the atm and
BC are built.

The initial condition is the part with no download fallback, and it is also the
part that turns out to be fine: a PFM LV1 history holds 121 hourly records
spanning 5 DAYS from its init (verified 2026-10-06), so the state at any t_da is
present in up to five different forecasts.  If the run initialised at t_da is
missing, the one from the day before contains the same valid time at lead 24 h.
find_lv1_history searches init-by-init, freshest first, because a shorter lead
means less forecast drift in the IC.
"""
from __future__ import annotations

import datetime as DT
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import da_config as C

try:
    import netCDF4 as nc
except ImportError:  # pragma: no cover
    nc = None

ROMS_EPOCH = DT.datetime(1999, 1, 1)

# A PFM LV1 forecast writes 121 hourly records = 5 days from its init, so the
# state at t_da appears in the runs initialised 0..4 days earlier.  Kept at 4
# rather than 5 so the last candidate still has a margin rather than landing
# exactly on its final record.
LV1_HIS_SPAN_DAYS = 4


@dataclass
class Item:
    name: str
    ok: bool
    detail: str = ""
    fixable: bool = False          # could a download close this gap?
    fix: str = ""                  # how, in words
    resolved: dict = field(default_factory=dict)   # what the stage should use


@dataclass
class Report:
    items: list[Item] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(i.ok for i in self.items)

    @property
    def blocking(self) -> list[Item]:
        """Missing AND not fixable by downloading -- the cycle cannot run."""
        return [i for i in self.items if not i.ok and not i.fixable]

    @property
    def downloadable(self) -> list[Item]:
        return [i for i in self.items if not i.ok and i.fixable]

    def get(self, name: str) -> Item | None:
        for i in self.items:
            if i.name == name:
                return i
        return None

    def text(self) -> str:
        out = []
        for i in self.items:
            mark = "ok  " if i.ok else ("FIX " if i.fixable else "MISS")
            out.append(f"    [{mark}] {i.name:10s} {i.detail}")
            if not i.ok and i.fix:
                out.append(f"                      -> {i.fix}")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# the initial condition
# ---------------------------------------------------------------------------
def find_lv1_history(t_da: DT.datetime,
                     max_lead_days: int = LV1_HIS_SPAN_DAYS
                     ) -> tuple[Path | None, int, str]:
    """(file, lead_hours, note) for a PFM LV1 history containing t_da.

    Searches the runs initialised at t_da, t_da-1d, ... freshest first, in the
    archive and then on scratch.  Freshest first because the IC should carry as
    little forecast drift as possible: the run initialised AT t_da has that state
    at lead 0.

    This replaced an exact-filename lookup that required the run initialised at
    t_da to exist.  It usually does, and when it does nothing changes -- but when
    a cycle has failed and been rerun off-schedule it does not, and that is
    precisely when PFM_DA is being asked to fill the hole.
    """
    for k in range(max_lead_days + 1):
        init = t_da - DT.timedelta(days=k)
        fn = f"LV1_ocean_his_{init:%Y%m%d}0000.nc"
        for base in C.LV1_HIS_DIRS:
            p = base / fn
            if not p.is_file():
                continue
            if not _history_covers(p, t_da):
                continue
            lead = int((t_da - init).total_seconds() // 3600)
            return p, lead, (f"{fn} (in {base}), lead {lead} h"
                             + (" -- the run initialised at t_da"
                                if lead == 0 else
                                f" -- NOTE: the run initialised at t_da is "
                                f"missing, falling back {k} day(s)"))
    return None, -1, ""


def _history_covers(p: Path, t: DT.datetime) -> bool:
    """Does this history actually hold a record at t?  Open it and look.

    The filename says when a run STARTED, not that it finished.  A crashed or
    truncated forecast leaves a short file with a perfectly good name, so the
    time axis is the only honest answer.
    """
    if nc is None:
        return True
    try:
        with nc.Dataset(p) as d:
            if "ocean_time" not in d.variables:
                return False
            raw = np.asarray(d["ocean_time"][:]).ravel().astype(float)
            if raw.size == 0:
                return False
            units = getattr(d["ocean_time"], "units", "")
            scale = 86400.0 if units.strip().lower().startswith("day") else 1.0
            times = [ROMS_EPOCH + DT.timedelta(seconds=float(x) * scale)
                     for x in raw]
    except Exception:  # noqa: BLE001
        return False
    return any(abs((x - t).total_seconds()) <= 60 for x in times)


# ---------------------------------------------------------------------------
# ECMWF
# ---------------------------------------------------------------------------
def ecmwf_valid_times(d: Path, init: DT.datetime) -> set[DT.datetime]:
    """Valid times on disk for one ECMWF init.

    ECMWF dissemination names are  T1D <MMDDHH of init> <YYYYMMDDHH valid> <3 ch>
    e.g.  T1D 100600 2026100601 001  = init 10-06 00Z, valid 2026-10-06 01Z.
    Verified against the files 2026-10-06; an earlier guess that the name ended at
    the valid time matched only the lead-0 file and made a present init look
    absent.
    """
    out = set()
    for f in d.glob(f"T1D{init:%m%d%H}*"):
        if f.name.endswith(".idx"):
            continue
        s = f.name[3 + 6:3 + 6 + 10]        # the YYYYMMDDHH field
        try:
            out.add(DT.datetime.strptime(s, "%Y%m%d%H"))
        except ValueError:
            continue
    return out


def check_ecmwf(t_da: DT.datetime, t_fore: DT.datetime) -> Item:
    """Is there an ECMWF init group on disk whose hourly steps cover the window?

    PFM's own get_atm_data_as_dict walks back through inits (-12 h, up to 4 tries)
    and downloads what it cannot find, so a gap here is fixable and in fact
    self-fixing -- provided the credentials in PFM/.env are loaded, which
    da_pfm_bridge.pfm_env does.  This check exists to say so in advance rather
    than to gate anything.
    """
    d = C.ECMWF_CACHE
    if not d.is_dir():
        return Item("ecmwf", False, f"{d} does not exist", fixable=True,
                    fix="PFM will download; needs WGET_ECMWF_* from PFM/.env")
    want = [t_da + DT.timedelta(hours=h)
            for h in range(int((t_fore - t_da).total_seconds() // 3600) + 1)]
    best = None
    for k in (0, 1, 2):
        init = t_da - DT.timedelta(days=k)
        have = ecmwf_valid_times(d, init)
        if not have:
            continue
        missing = [w for w in want if w not in have]
        if not missing:
            lead = int((t_fore - init).total_seconds() // 3600)
            return Item("ecmwf", True,
                        f"init {init:%Y-%m-%d %H}Z, {len(have)} steps on disk, "
                        f"covers the window at leads 0..{lead} h",
                        resolved={"init": init})
        if best is None or len(missing) < best[1]:
            best = (init, len(missing))
    if best is None:
        return Item("ecmwf", False,
                    f"no ECMWF init group found in {d} for "
                    f"{t_da:%m-%d %H}Z or the 2 days before",
                    fixable=True,
                    fix="get_atm_data_as_dict will fetch it (needs PFM/.env creds)")
    return Item("ecmwf", False,
                f"best init {best[0]:%Y-%m-%d %H}Z is missing {best[1]} of "
                f"{len(want)} hourly steps in the window",
                fixable=True,
                fix="get_atm_data_as_dict will fetch the gap "
                    "(needs PFM/.env creds)")


# ---------------------------------------------------------------------------
# HYCOM
# ---------------------------------------------------------------------------
def check_hycom(t_da: DT.datetime, t_fore: DT.datetime, pkl: Path) -> Item:
    """Which HYCOM init is complete on disk for the window, and what is missing.

    Unlike ecmwf_data/, hycom_data/ is cleaned each cycle and holds only ~2 inits,
    so this is the input most likely to be short -- and the one the Jan-2 edge
    case will trip.
    """
    import da_pfm_bridge as B
    best = None
    for k in (0, 1, 2, 3):
        init = (t_da - DT.timedelta(days=k)).strftime("%Y%m%d")
        names = B.hycom_file_list(init, t_da, t_fore, pkl)
        missing = [f for f in names if not _hycom_file_ok(f)]
        if not missing:
            return Item("hycom", True,
                        f"init {init}T12, all {len(names)} files present",
                        resolved={"init": init})
        if best is None or len(missing) < len(best[1]):
            best = (init, missing, names)
    init, missing, names = best
    return Item("hycom", False,
                f"best init {init}T12 is short: "
                f"{len(names) - len(missing)}/{len(names)} files present",
                fixable=True,
                fix=f"download {len(missing)} missing file(s) for init {init}T12",
                resolved={"init": init, "missing": missing})


def _hycom_file_ok(f: Path) -> bool:
    """Present AND big enough.

    The size floors are PFM's own (check_and_redownload_ncfiles): 0.4 MB for ssh,
    1.5 MB for the 3D variables.  Checking size as well as existence matters
    because a truncated ncks download leaves a small, valid-looking file -- which
    is the failure PFM's own re-download loop was written for.
    """
    f = Path(f)
    if not f.is_file():
        return False
    floor_mb = 0.4 if "_ssh_" in f.name else 1.5
    return os.path.getsize(f) / (1024 * 1024) >= floor_mb


# ---------------------------------------------------------------------------
# HF radar
# ---------------------------------------------------------------------------
def check_hfr(t_da: DT.datetime, t_fore: DT.datetime) -> Item:
    """How many of the window's 25 hourly surveys are already cached.

    Always "fixable": PFM does not fetch HFR, so PFM_DA downloading it IS the
    normal path.  Reported anyway so a cycle's obs coverage is visible up front
    rather than discovered from a thin analysis afterwards.
    """
    n = int((t_fore - t_da).total_seconds() // 3600) + 1
    hours = [t_da + DT.timedelta(hours=h) for h in range(n)]
    have = 0
    for t in hours:
        stamp = t.strftime("%Y%m%d%H%M%S") + "0"
        if any(c.is_dir() and any(c.glob(f"*_s{stamp}_*.nc"))
               for c in C.HFR_CACHES):
            have += 1
    if have == n:
        return Item("hfr", True, f"all {n} hourly surveys cached",
                    resolved={"have": have, "need": n})
    return Item("hfr", False,
                f"{have}/{n} hourly surveys cached",
                fixable=True,
                fix=f"fetch_hfr_nrt.py will get the other {n - have} "
                    f"(NDBC, ERDDAP fallback)",
                resolved={"have": have, "need": n})


# ---------------------------------------------------------------------------
def check_inputs(t_fore: DT.datetime, t_da: DT.datetime, pkl: Path) -> Report:
    """Everything a cycle needs, in one pass."""
    r = Report()

    his, lead, note = find_lv1_history(t_da)
    if his is not None:
        r.items.append(Item("lv1_ic", True, note,
                            resolved={"his": his, "lead_h": lead}))
    else:
        r.items.append(Item(
            "lv1_ic", False,
            f"no PFM LV1 history within {LV1_HIS_SPAN_DAYS} days holds a record "
            f"at {t_da:%Y-%m-%d %H}Z",
            fixable=False,
            fix="nothing to download -- a PFM forecast covering t_da has to "
                "exist.  Rerun a PFM cycle whose window contains t_da, or pick "
                "a t_fore that an existing history covers."))

    r.items.append(check_ecmwf(t_da, t_fore))
    r.items.append(check_hycom(t_da, t_fore, pkl))
    r.items.append(check_hfr(t_da, t_fore))
    return r
