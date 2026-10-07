"""
da_config.py — paths, resources and tunables for the PFM DA pre-step.

Everything the pipeline needs to know about *where* things live is here, so a
path change is a one-file edit rather than a grep.

Layout follows PFM's own conventions:
  * run area on /scratch (BeeGFS, fast; the LV1 4D-Var writes ~13 GB a night)
    laid out per level as Forc/ Run/ His/, mirroring LV{N}_Forecast/
  * archive on /dataSIO, mirroring Archive/LV{N}_His/
  * flat directories with the date in the FILENAME, not per-cycle subdirs
  * keepers are copied to the archive as soon as they validate, and /scratch is
    purged by age -- not archived-then-purged at the end.  A 4.5 h pipeline
    fails partway often enough that the LV1 analysis must be safe on disk before
    LV4 is even attempted.
"""
from __future__ import annotations

import datetime as DT
from pathlib import Path

# ---------------------------------------------------------------------------
# run area and archive
# ---------------------------------------------------------------------------
SCRATCH = Path("/scratch/PFM_Simulations/DA_Sim")
ARCHIVE = Path("/dataSIO/PFM_Simulations/Archive/DA")

LEVELS = ("LV1", "LV2", "LV3", "LV4")

# Forc/ and His/ age out; Run/ does NOT -- it holds the executable, the .in
# templates and the NRM cache, all reused every night.  (PFM's own
# LV4_Forecast/Run spans 90 days for the same reason.)
SCRATCH_SUBDIRS = ("Forc", "Run", "His")
SCRATCH_RETAIN_DAYS = 7
SCRATCH_PURGE_SUBDIRS = ("Forc", "His")

ARCHIVE_SUBDIRS = ("LV1_Prior", "LV1_Analysis", "LV1_Dai",
                   "LV2_His", "LV3_His", "LV4_His",
                   "Inc", "IC_out", "Obs", "Log", "manifests")

# ---------------------------------------------------------------------------
# inputs we read but never write
# ---------------------------------------------------------------------------
PFM_ARCHIVE = Path("/dataSIO/PFM_Simulations/Archive")       # LV{N}_His base ICs
GRIDS = Path("/scratch/PFM_Simulations/Grids")
GRID = {
    "LV1": GRIDS / "GRID_SDTJRE_LV1_rx020_hmask.nc",
    "LV2": GRIDS / "GRID_SDTJRE_LV2_rx020.nc",
    # the RUN uses maskCoast; rx020 is bit-identical except that maskCoast masks
    # 153 shallow coastal cells, some holding salinity up to 49.6 PSU
    "LV3": GRIDS / "GRID_SDTJRE_LV3_rx020_maskCoast.nc",
    "LV4": GRIDS / "GRID_SDTJRE_LV4_mss_oct2024.nc",
}
# Where to look for a PFM LV1 history to take the DA initial condition from,
# in order.  A list, not one path, because the IC may have to come from a run
# other than the one initialised at t_da -- see da_preflight.find_lv1_history.
LV1_HIS_DIRS = [PFM_ARCHIVE_LV1_HIS := Path(
                    "/dataSIO/PFM_Simulations/Archive/LV1_His"),
                Path("/scratch/PFM_Simulations/LV1_Forecast/His")]

# HF radar.  PFM does not use HFR at all, so PFM_DA is the only thing that
# fetches it -- downloading here is the normal path, not the exception it is for
# ECMWF and HYCOM.  Order is PRECEDENCE: the retrospective NCEI cache first, so
# reprocessed data beats the near-real-time feed for an hour both hold.
HFR_CACHE_RETRO = Path("/dataSIO/DA_Simulations/input_files/hfr_cache")
HFR_CACHE_NRT = SCRATCH / "Obs" / "hfr_nrt"
HFR_CACHES = [HFR_CACHE_RETRO, HFR_CACHE_NRT]
# 25 hourly surveys span the window; make_obs_from_hfr tolerates gaps (the NCEI
# catalog has permanent ones) so a cycle is not failed for a few missing hours.
HFR_MIN_HOURS = 20
# An hour can EXIST with no data in it -- NDBC publishes the timestamp before the
# radials are assembled (6 km's newest hour read 0.0% on 2026-10-07).  Such an
# hour must not count as cached: it would pass the count test and contribute no
# observations.  Fraction of the LV1 box that must carry valid u/v for an hour to
# count, set well below the ~21% (6 km) and ~5% (2 km) steady state so a
# genuinely sparse hour still counts.
HFR_MIN_COVERAGE = 0.005

# PFM's own varinfo, one per level, read-only.  Verified byte-identical to the
# research chain's copies, so using PFM's keeps 333 KB x4 out of git and removes
# any chance of the two drifting.
VARINFO = {lv: Path(f"/scratch/PFM_Simulations/{lv}_Forecast/Run/{lv}_varinfo.yaml")
           for lv in ("LV1", "LV2", "LV3", "LV4")}

# PFM operational histories: the COLD base for a child forward when the previous
# night's DA history is not there to warm-start from.
PFM_HIS = {lv: PFM_ARCHIVE / f"{lv}_His" for lv in ("LV1", "LV2", "LV3", "LV4")}

# Which parent each child takes boundaries from.  Strictly sequential: LV3 cannot
# start before LV2 has written its history.
BC_PARENT = {"LV2": "LV1", "LV3": "LV2", "LV4": "LV3"}

# Where the operational forecast deposits SWAN hotstart files, read-only for us.
# 357 GB, named LV4_swan_rst_<init YYYYMMDDHHMM>_<lead HHH>.dat-NNN with one part
# per SWAN rank (72).  PFM_DA never writes here.
SWAN_RST_SHARED = Path("/scratch/PFM_Simulations/restart_swan_data")

# CDIP wave spectra, the source for LV4's SWAN boundary file.  The archive is
# date-stamped and complete (81 stations per cycle); the operational cache holds
# only the current cycle.  PFM_DA reads the archive and stages into its OWN
# directory -- see da_pfm_bridge.stage_cdip_from_archive for why that separation
# is not optional.
CDIP_ARCHIVE = Path("/dataSIO/PFM_Simulations/Archive/cdip_ncs")
CDIP_DA = SCRATCH / "LV4" / "Forc" / "cdip_data"

ECMWF_CACHE = Path("/scratch/PFM_Simulations/ecmwf_data")
HYCOM_CACHE = Path("/scratch/PFM_Simulations/hycom_data")

# ---------------------------------------------------------------------------
# PFM's own code, reused by stages 2-3
# ---------------------------------------------------------------------------
# The DA pre-step runs PFM's builders rather than reimplementing them, so it has
# to know where they live and which config module they are driven by.
PFM_PYDIR = Path(__file__).resolve().parent.parent / "sdpm_py_util"
PFM_DRIVER = Path(__file__).resolve().parent.parent / "driver"
# The input file the OPERATIONAL driver names (driver/run_PFMv2.sh,
# run_PFM_LV1234.sh), not the older pfm_operational_input.py this first pointed
# at.  The difference matters: _new splits restart_files_dir into
# restart_{nc,swan}files_dir_{in,out}, and the SWAN hotstart lookup
# (init_funs_forecast.get_swan_restart_file_name) reads
# restart_swanfiles_dir_in -- a key the old file never sets, so LV4 would have
# died with a KeyError.  It also adds lv{1,2,3}_blank_name and pfm_new.
#
# NOTE the live cron runs ffeddersen's copy of this same filename, which is not
# readable from here, so this is the in-repo version of the right file rather
# than a byte-for-byte match with production.  Stage 2's acceptance test (atm
# bit-identical to the operational output) is what actually confirms the
# settings agree where it counts.
PFM_INPUT_PY = PFM_PYDIR / "pfm_operational_input_new.py"

def minfo_pkl(t_fore: str) -> Path:
    """The DA run's OWN model-info pickle.

    PFM's builders all take the pickle path as an argument (get_model_info is a
    plain pickle load), so giving them ours isolates the DA run completely: the
    operational /scratch/PFM_Simulations/PFM_run_info.pkl is never read and never
    written.  That isolation is what makes Phase 1 bit-safe for the forecast.

    Per t_fore, not one shared file, so two DA runs cannot collide.
    """
    return SCRATCH / "Log" / f"PFM_DA_run_info_{t_fore}.pkl"

# ---------------------------------------------------------------------------
# atmospheric forcing source for the DA window
# ---------------------------------------------------------------------------
# ECMWF, built by PFM's own atm_functions -- the same product and the same code
# the forecast uses.  The DA window [t_fore-24h, t_fore] is closed, so it is
# served by the PREVIOUS day's init at steps 0..24; ecmwf_data/ keeps one init
# group per day (verified 2026-10-06: 99 groups back to 2026-07-07, 101 hourly
# steps each, never pruned), so those files are already on disk.  That is also
# the forcing yesterday's forecast ran its first day with, so the DA analysis is
# consistent with the forcing by construction.
#
# get_atm_data_as_dict already walks back through earlier inits (-12 h, up to 4
# tries) until got_ecmwf_files() says the window is covered, so pointing
# fetch_time at t_da is all this takes.
#
# The rotation on this path is correct: get_atm_data_on_roms_grid rotates with
# angr = RMG['angle'] (atm_functions.py:596-598, applied ~610).  ECMWF is on a
# lat/lon grid, so its u/v are earth-relative and that single stage is the whole
# rotation.
ATM_SOURCE = "ecmwf"

# ECMWF derived pickles must be DA-TAGGED, and this is not cosmetic.
# atm_functions writes the "all" pickle to ecmwf_dir + ecmwf_all_pkl_name -- ONE
# shared file.  PFM's own code raises on a failed rebuild with the comment that
# continuing "would leave the previous cycle's ecmwf_all.pkl in place and
# silently build atm forcing for the wrong day (this is exactly what happened
# 2026-08-31)".  A DA run writing the operational name would do that to the
# forecast deliberately.  The grib files stay shared -- downloads are additive
# and we want the cache -- but every derived pickle gets its own name.
ECMWF_PKL_TAG = "DA"

# ---------------------------------------------------------------------------
# code
# ---------------------------------------------------------------------------
DA_FILES = Path(__file__).resolve().parent
SCRIPTSDIR = DA_FILES / "scripts"
PYDIR = DA_FILES / "python_files"
MASKDIR = DA_FILES / "masks"
RUN_IN = DA_FILES / "run_in"
SLURMDIR = DA_FILES / "slurm"
PFM_ROOT = DA_FILES.parent
PY = "/home/mspydell/anaconda3/envs/PHM-env/bin/python3"

MASKS = {
    "LV2": [],
    "LV3": [MASKDIR / "LV3_incr_mask_sigma1km.nc"],
    "LV4": [MASKDIR / "LV4_incr_mask_sigma1km.nc",
            MASKDIR / "LV4_incr_mask_sigma1km_tjre.nc"],
}

# ---------------------------------------------------------------------------
# resources.  The cluster has 15 nodes (swell-cn03-17), so LV4 at 14 nodes is
# effectively the whole machine -- nothing else can run beside it.
# ---------------------------------------------------------------------------
# These MUST match the #SBATCH --nodes in slurm/, which is what actually
# allocates; this dict is only for reporting and for reasoning about what can run
# beside what.  LV3 said 10 here against the 6 its .sb asks for, so a scheduling
# estimate made from this would have been wrong -- verified against the .sb files
# 2026-10-07.
NODES = {"lv1_da": 9, "increments": 1, "LV2": 6, "LV3": 6, "LV4": 14}


def check_nodes_match_slurm() -> list[str]:
    """Report any stage whose NODES disagrees with its slurm script.

    Cheap, and it catches the class of error above: a number here that drifts from
    the one that actually allocates.
    """
    import re
    files = {"lv1_da": "slurm_4dvar.sb", "increments": "slurm_make_increment.sb",
             "LV2": "slurm_LV2_fwd.sb", "LV3": "slurm_LV3_fwd.sb",
             "LV4": "slurm_LV4_fwd.sb"}
    bad = []
    for key, fn in files.items():
        p = SLURMDIR / fn
        if not p.is_file():
            continue
        m = re.search(r"^#SBATCH\s+--nodes=(\d+)", p.read_text(), re.M)
        if m and int(m.group(1)) != NODES.get(key):
            bad.append(f"{key}: NODES={NODES.get(key)} but {fn} asks for "
                       f"{m.group(1)}")
    return bad

# Measured, except LV3/LV4 which are scaled from PFM's 2.5-day forecast runs.
# Stage 7 records the real numbers so these can be replaced with measurements.
EXPECTED_WALL_S = {
    # preflight opens at most a couple of files; it exists so a doomed cycle
    # fails in seconds instead of after the atm and BC are built.
    "preflight": 10,
    "lv1_ic": 120, "lv1_atm": 300, "lv1_bc": 600, "lv1_obs": 600,
    "lv1_da": 3 * 3600 + 15 * 60,          # measured 3h13
    "increments": 400,                      # ~6.4 min, all three levels, 1 date
    "LV2": 200,                             # measured 2m49
    "LV3": 300,                             # scaled
    "LV4": 3300,                            # scaled from 2h15 / 2.5 days
}
# The whole pre-step must finish before the forecast starts.
#
# The stage estimates above sum to ~4.9 h.  Against a 5 h window (20:00 -> 01:00
# local) that is only ~8 minutes of slack -- NOT the ~35 min first estimated,
# which omitted the 0.45 h of input preparation (IC, atm, BC, obs).  Two ways to
# buy room, in order of preference:
#   * start at 19:30 rather than 20:00: the DA window [t_da, t_fore] closed at
#     00Z = 17:00 local, and HFR at 1-2 km lags ~1.8 h, so the inputs are ready
#     well before 19:30.
#   * prepare lv1_{atm,bc} earlier in the day -- they describe a window that has
#     already closed and do not depend on the observations.
# Until the LV4 1-day wall time is MEASURED rather than scaled, treat the budget
# as not yet proven; that single number is ~19% of the total.
DEFAULT_DEADLINE_H = 5.0
BUDGET_WARN_FRAC = 0.9        # warn when the estimate exceeds this of the window


def budget_check() -> tuple[float, float, str]:
    """(expected_h, deadline_h, message).  Called at startup so a budget that
    cannot close is reported before three hours of 4D-Var are spent."""
    exp = sum(EXPECTED_WALL_S.values()) / 3600.0
    if exp > DEFAULT_DEADLINE_H:
        msg = (f"expected {exp:.2f} h EXCEEDS the {DEFAULT_DEADLINE_H:.2f} h "
               f"deadline -- start earlier or trim a stage")
    elif exp > BUDGET_WARN_FRAC * DEFAULT_DEADLINE_H:
        msg = (f"expected {exp:.2f} h leaves only "
               f"{60*(DEFAULT_DEADLINE_H-exp):.0f} min of slack")
    else:
        msg = f"expected {exp:.2f} h of {DEFAULT_DEADLINE_H:.2f} h"
    return exp, DEFAULT_DEADLINE_H, msg

# ---------------------------------------------------------------------------
# DA settings.  See roms_fourdvar for the provenance of each.
# ---------------------------------------------------------------------------
ROMS_EXE_DA = SCRATCH / "LV1" / "Run" / "romsM_DA_IC"   # IC-only; BC-adjust blew
                                                        # up at the NW corner
ROMS_EXE_FWD = "romsM_forward"
ROMS_EXE_LV4 = "coawstM_noanapassive"   # ANA_PASSIVE undef -> IC dye honoured

# ---------------------------------------------------------------------------
# LV1 4D-Var inputs that are static, large and expensive to rebuild.
# ---------------------------------------------------------------------------
# The normalisation (NRM) and background-error standard deviation (STD) files --
# ~410 MB together -- are referenced from /dataSIO rather than copied into git.
# They are regenerable but not cheaply: the NRM randomisation is hours of compute
# (LV1 used Nrandom=6000), and they do not change from cycle to cycle.  Same
# reasoning as the HFR retrospective cache: a data dependency, documented, not a
# code one.
NRM_DIR = Path("/dataSIO/DA_Simulations/input_files/nrm_files")
STD_DIR = Path("/dataSIO/DA_Simulations/input_files/std_files")

# Decorrelation scales and error amplitudes, matching the LV1_v2 research chain.
# HDECAY_IC is also L_INCR in the downscaler, so the two must agree.
HDECAY_IC_KM = 15
HDECAY_BC_KM = 15
HDECAY_F_KM = 50
URMS_TAG = "urms0p05"
WSRMS_TAG = "wsrms0p02"

# ROMS reads and writes a dozen scratch files per cycle and wants them in one
# place; keeping it separate from Run/ means a purge can clear it without
# touching the executable or the .in templates.
def io_dir() -> Path:
    return SCRATCH / "LV1" / "tmp_input_output"


BLEND_SPACE = "D"        # blending h needs a Dcrit clip that inverts the parent
                         # column at a deep low tide
# 6 km, matching the LV1_v2 research chain (the 3/7-onward cycles) so the
# operational DA sees the same observation set the tuning was done against.
#
# Measured 2026-10-07 02:18Z against NDBC, which corrected an earlier note that
# 6 km "lags ~4.8 h" while 1-2 km lag ~1.8 h.  The newest hour holding ANY data
# is 2.3 h old for both 2 km and 6 km; what differs is how long an hour takes to
# FILL.  Coverage over the LV1 box:
#
#     res    steady coverage   newest hour    hour before
#     2 km       ~5.1%            5.0%           4.8%      (arrives ~full)
#     6 km      ~21.0%            0.0%           8.8%      (still filling)
#
# So 6 km reaches ~21% of its grid against 2 km's ~5%, but its last one or two
# hours are incomplete -- which is what the ~4.8 h figure really described: time
# to FULL coverage, not time to first data.
#
# Consequence for the schedule, stated plainly: a pre-step running ~2-3 h after
# the window closes will find the final hour or two of 6 km thin.  That is
# survivable (HFR_MIN_HOURS below) but those are the hours nearest t_fore, so
# they are the ones the analysis would most like to have.  Switching to 2 km buys
# ~2x the raw obs and loses everything west of xi~84 in LV1 -- measured 191,090
# obs over xi 84..245 at 2 km against 61,022 over xi 11..243 at 6 km.
HFR_RES = "6km"
OBS_ERROR_VAR = 0.015    # smallest that stayed stable from a poor IC; Desroziers
                         # suggests ~2x too large


# ---------------------------------------------------------------------------
def parse_t_fore(t_fore: str) -> tuple[DT.datetime, DT.datetime]:
    """'YYYYMMDDHH' -> (t_fore, t_da).

    t_da = t_fore - 24 h.  The DA window is [t_da, t_fore], so the analysis at
    t = 24 h IS the state at t_fore -- which is exactly the IC the forecast
    needs.  That identity is the whole point, so the entry point takes the
    FORECAST time and derives the DA time, never the other way round.
    """
    if len(t_fore) != 10 or not t_fore.isdigit():
        raise ValueError(f"t_fore must be YYYYMMDDHH, got {t_fore!r}")
    tf = DT.datetime.strptime(t_fore, "%Y%m%d%H")
    return tf, tf - DT.timedelta(days=1)


def ymd(t: DT.datetime) -> str:
    return t.strftime("%Y%m%d")


def scratch_dir(level: str, sub: str) -> Path:
    return SCRATCH / level / sub


def ensure_tree() -> None:
    """Create the run area and archive.  Idempotent."""
    for lv in LEVELS:
        for sub in SCRATCH_SUBDIRS:
            (SCRATCH / lv / sub).mkdir(parents=True, exist_ok=True)
    HFR_CACHE_NRT.mkdir(parents=True, exist_ok=True)
    io_dir().mkdir(parents=True, exist_ok=True)
    for d in ("Inc", "IC_out", "Log", "Obs", "Rst"):
        (SCRATCH / d).mkdir(parents=True, exist_ok=True)
    for d in ARCHIVE_SUBDIRS:
        (ARCHIVE / d).mkdir(parents=True, exist_ok=True)
