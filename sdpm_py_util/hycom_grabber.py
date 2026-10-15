"""Standalone HYCOM cache keeper.

PFM used to fetch HYCOM inside the forecast, which meant a slow or flaky
server became a failed forecast. Since 2026-10-07 tds.hycom.org has been
running ~10x slower (2-6 s per 10-file batch before, 30-45 s after) and,
worse, intermittently serves files with the scale_factor and _FillValue
attributes stripped off. A stripped file has the right dimensions, 100%
finite data and ncks exits 0 -- but the values are raw int16 counts, so a
sea level of -0.681 m arrives as -681. Nothing downstream would catch that.

This module keeps the cache correct out of band so the forecast does not
have to. Step 1 (this file) is the verification half: say precisely which
files are good, which are missing, and which are bad and why. Fetching,
retention and the hourly cron come next.

Modes:
    python hycom_grabber.py verify [--delete]  scan the cache, no network
    python hycom_grabber.py status             one line per cycle
    python hycom_grabber.py fill               verify, delete bad, refetch, prune

Verification is deliberately stricter than the old "is the file bigger than
0.4 MB" test, which was a proxy that happened to catch stripped files and
would stop working the moment HYCOM changed its compression.
"""

import os
import re
import sys
import glob
import json
import time
import fcntl
import shutil
import argparse
import subprocess
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

try:
    import netCDF4 as nc
except ImportError:                                          # pragma: no cover
    print('hycom_grabber needs netCDF4; activate PHM-env first.')
    raise

# ---------------------------------------------------------------------------
# what a good file looks like
# ---------------------------------------------------------------------------
# filename is hy_<var>_<cycle>_<valid>.nc, e.g.
#   hy_ssh_2026-10-13T12:00_2026-10-18T05:00.nc
# the prefix fixes which variable and how many vertical levels to expect.
VAR_OF_PREFIX = {
    'ssh': 'surf_el',
    't3z': 'water_temp',
    's3z': 'salinity',
    'u3z': 'water_u',
    'v3z': 'water_v',
}

# physical bounds, applied AFTER scaling and after dropping fill. these are
# deliberately loose: the point is to catch unscaled int16 counts (a stripped
# ssh reads -681 instead of -0.681), not to referee the ocean state.
PHYS_RANGE = {
    'surf_el':    (-5.0,   5.0),     # m
    'water_temp': (-3.0,  40.0),     # degC
    'salinity':   (0.0,   45.0),     # psu
    'water_u':    (-5.0,   5.0),     # m/s
    'water_v':    (-5.0,   5.0),     # m/s
}

NZ_3D = 40          # vertical levels in the 3z files
MIN_REAL_FRAC = 0.05   # at least this fraction must be non-fill, or the file
                       # carries no usable ocean at all

# cheap pre-filter only. never the verdict -- see the module docstring.
SIZE_FLOOR_MB = {'ssh': 0.02, 'other': 0.10}

OK, MISSING, BAD = 'ok', 'missing', 'bad'

# a cycle runs 192 h: ssh hourly (193 files), the 3z vars every 3 h (65).
LEAD_HOURS_SSH = list(range(0, 193, 1))
LEAD_HOURS_3Z = list(range(0, 193, 3))

KEEP_DAYS = 5          # retain the five most recent cycles, prune the rest
DEFAULT_WORKERS = 10   # hycom's documented ceiling is 10 per IP
DEFAULT_BUDGET_MIN = 45
PROGRESS_EVERY_MIN = 5 # a 45 min run should say something in between
PER_FILE_TRIES = 5     # stripping is random per request: ~40% when measured,
                       # so a handful of immediate retries usually wins

# The THREDDS catalog lists every file the server actually holds -- 7525 of
# them in one 3.3 MB request taking ~8 s. That is cheaper than probing even a
# single cycle (453 files) and it is the only way to tell "hycom has not
# published this yet" apart from "hycom will never publish this".
#
# It matters because the server does have real gaps. Cycle 2026-10-14T12 has
# no t3z and no s3z for leads t0000..t0036 -- temperature and salinity simply
# do not exist for the first 36 h of that forecast, while ssh, u3z and v3z are
# complete. Without the catalog those 26 files look like 26 fetch failures,
# get retried 5x each every hour forever, and the real news (hycom is missing
# a day and a half of T/S) never reaches the log.
CATALOG_URL = ('https://tds.hycom.org/thredds/catalog/datasets/ESPC-D-V02/'
               'data/forecasts/catalog.xml')
CATALOG_CACHE = '.catalog.txt'   # inside the hycom dir
CATALOG_MAX_AGE_MIN = 20         # refetch if older; cycles appear hourly at most

# HYCOM takes the service down twice a day for maintenance, 02:00-03:00 and
# 12:00-13:00 US Eastern. Eastern, not UTC: the offset shifts with daylight
# saving, so an hourly cron written in UTC would drift into the window for
# half the year. Fetching through it produces nothing but failed requests.
MAINT_HOURS_ET = (2, 12)

# for the log banner only -- the maintenance window is Eastern, PFM is Pacific
LOCAL_TZ = 'America/Los_Angeles'

LOCK_NAME = '.grabber.lock'      # inside the hycom dir

# A mirror on /dataSIO, so losing /scratch costs a copy rather than a day of
# re-downloading from a server that has been slow and lossy since 2026-10-07.
# It tracks the cache exactly, retention included: the point is "restore the
# working set", not "keep everything forever".
ARCHIVE_DIR = '/dataSIO/PFM_Simulations/Archive/hycom_raw'

# hycom allows 10 concurrent connections per IP and we use all 10. a PFM
# forecast still fetches hycom itself, with its own 10, so a grabber that
# starts mid-forecast puts 20 connections on the server from this address --
# over the documented limit, and the forum is explicit that exceeding it is
# how an IP gets throttled or blocked. the lock below only keeps grabbers
# apart; this keeps the grabber away from the forecast. it goes away once PFM
# reads the cache instead of filling it.
FETCH_HOST = 'tds.hycom.org'


def cycle_files(cycle):
    """Every filename a complete cycle should have. cycle is '2026-10-13T12'."""
    t0 = datetime.strptime(cycle, '%Y-%m-%dT%H')
    out = []
    for prefix in VAR_OF_PREFIX:
        leads = LEAD_HOURS_SSH if prefix == 'ssh' else LEAD_HOURS_3Z
        for h in leads:
            vt = (t0 + timedelta(hours=h)).strftime('%Y-%m-%dT%H')
            out.append('hy_%s_%s:00_%s:00.nc' % (prefix, cycle, vt))
    return out


def recent_cycles(n=KEEP_DAYS, now=None):
    """The n most recent 12Z cycles, newest first. Today's may not be
    published yet; the fetch simply finds nothing for it and moves on."""
    now = now or datetime.now(timezone.utc)
    return [(now - timedelta(days=d)).strftime('%Y-%m-%dT12') for d in range(n)]


def parse_name(path):
    """hy_<var>_<cycle>_<valid>.nc -> (prefix, cycle, valid). None if it does
    not match; the cache holds the odd stray file and they are not our
    business."""
    b = os.path.basename(path)
    # downloads land as <name>.nc.part and are verified before the rename, so
    # the parser has to see through the suffix -- otherwise every good fetch
    # is rejected as "filename does not parse" and nothing is ever kept.
    if b.endswith('.part'):
        b = b[:-5]
    if not b.startswith('hy_') or not b.endswith('.nc'):
        return None
    try:
        prefix = b[3:6]
        cycle = b[7:20]
        valid = b[24:37]
        if prefix not in VAR_OF_PREFIX:
            return None
        datetime.strptime(cycle, '%Y-%m-%dT%H')
        datetime.strptime(valid, '%Y-%m-%dT%H')
    except (ValueError, IndexError):
        return None
    return prefix, cycle, valid


def server_name(name):
    """Our 'hy_ssh_<cycle>_<valid>.nc' -> hycom's filename for the same field.

    Mirrors ocn_funs_forecast.myname_to_urls so a catalog listing can be
    compared against what we want without building a URL for each one.
    """
    t_f = datetime.strptime(name[7:20], '%Y-%m-%dT%H')
    t_2 = datetime.strptime(name[24:37], '%Y-%m-%dT%H')
    hr = int((t_2 - t_f).total_seconds() // 3600)
    return ('US058GCOM-OPSnce.espc-d-031-hycom_fcst_glby008_%s_t%04d_%s.nc'
            % (t_f.strftime('%Y%m%d%H'), hr, name[3:6]))


def cycle_in_catalog(cycle, catalog):
    """Whether the server offers this cycle at all, as opposed to offering it
    with gaps. 'not published yet' and 'published but incomplete' call for
    very different log lines."""
    tag = 'glby008_%s_' % datetime.strptime(
        cycle, '%Y-%m-%dT%H').strftime('%Y%m%d%H')
    return any(tag in n for n in catalog)


def fetch_catalog(hydir, max_age_min=CATALOG_MAX_AGE_MIN, log=print):
    """Every filename the server currently offers, as a set.

    Cached on disk so several runs in one hour cost one request. Returns an
    empty set on failure, and the caller then falls back to asking for
    everything -- degraded, but never worse than having no catalog at all.
    """
    cache = os.path.join(hydir, CATALOG_CACHE)

    def cached(why):
        """Fall back to the copy on disk at any age.

        Going without a catalog is much worse than using a stale one. The
        server's file list changes once a day, so an hours-old copy is very
        nearly right -- whereas no catalog means queueing cycles that are not
        published, which is the failure the catalog was added to prevent and
        which the hycom forum calls a way to get an IP blocked.
        """
        if not os.path.isfile(cache):
            log('CATALOG %s, and no cached copy -- proceeding without it' % why)
            return set()
        age = (time.time() - os.path.getmtime(cache)) / 60.0
        with open(cache) as f:
            names = {ln.strip() for ln in f if ln.strip()}
        log('CATALOG %s -- using the saved listing (%d names, %.0f min old). '
            'This is hycom\'s index of what it offers, not our data; no '
            'verified file is refetched because of it.'
            % (why, len(names), age))
        return names

    if os.path.isfile(cache):
        age = (time.time() - os.path.getmtime(cache)) / 60.0
        if age < max_age_min:
            with open(cache) as f:
                names = {ln.strip() for ln in f if ln.strip()}
            log('CATALOG %d names (saved listing, %.0f min old)'
                % (len(names), age))
            return names
    # 90 s, not 180: a 2 min budget once went entirely on one catalog attempt
    # that timed out anyway. the fallback below costs nothing.
    try:
        r = subprocess.run(['curl', '-s', '--max-time', '90', CATALOG_URL],
                           capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return cached('fetch timed out')
    if r.returncode != 0 or not r.stdout:
        return cached('fetch failed (rc=%d)' % r.returncode)
    names = set(re.findall(r'name="(US058GCOM[^"]*\.nc)"', r.stdout))
    if not names:
        return cached('parsed 0 names')
    tmp = cache + '.part'
    try:
        with open(tmp, 'w') as f:
            f.write('\n'.join(sorted(names)) + '\n')
        os.replace(tmp, cache)
    except OSError:
        pass
    log('CATALOG %d names listed by the server' % len(names))
    return names


def verify_file(path):
    """Return (state, reason). state is OK / MISSING / BAD.

    The checks run cheapest-first so a scan of a few thousand files stays
    quick: stat, then open, then attributes, then the data itself.
    """
    meta = parse_name(path)
    if meta is None:
        return BAD, 'filename does not parse'
    prefix, _, _ = meta
    varname = VAR_OF_PREFIX[prefix]

    if not os.path.isfile(path):
        return MISSING, 'absent'

    size_mb = os.path.getsize(path) / (1024 * 1024)
    floor = SIZE_FLOOR_MB['ssh'] if prefix == 'ssh' else SIZE_FLOOR_MB['other']
    if size_mb < floor:
        return BAD, 'only %.3f MB' % size_mb

    try:
        ds = nc.Dataset(path)
    except Exception as e:                                   # noqa: BLE001
        return BAD, 'will not open (%s)' % type(e).__name__

    try:
        if varname not in ds.variables:
            return BAD, 'no %s variable' % varname
        v = ds.variables[varname]

        # THE check. a stripped file is the failure mode that motivated this
        # module: everything else about it looks right.
        if not hasattr(v, 'scale_factor'):
            return BAD, 'scale_factor stripped (values would be ~1000x)'
        if not hasattr(v, '_FillValue'):
            return BAD, '_FillValue stripped'

        want_ndim = 3 if prefix == 'ssh' else 4
        if v.ndim != want_ndim:
            return BAD, 'ndim %d, expected %d' % (v.ndim, want_ndim)
        if want_ndim == 4 and v.shape[1] != NZ_3D:
            return BAD, '%d vertical levels, expected %d' % (v.shape[1], NZ_3D)

        # auto-scaling is on by default, so this is already in physical units
        # with fill masked.
        a = np.asarray(v[:])
        if np.ma.isMaskedArray(v[:]):
            a = np.ma.filled(np.ma.masked_invalid(v[:]), np.nan)
        real = a[np.isfinite(a)]
        frac = real.size / a.size if a.size else 0.0
        if frac < MIN_REAL_FRAC:
            return BAD, 'only %.1f%% real values' % (100 * frac)

        lo, hi = PHYS_RANGE[varname]
        rmin, rmax = float(real.min()), float(real.max())
        if rmin < lo or rmax > hi:
            return BAD, 'values %.3f..%.3f outside %s (%.1f..%.1f)' % (
                rmin, rmax, varname, lo, hi)
    finally:
        try:
            ds.close()
        except Exception:                                    # noqa: BLE001
            pass

    return OK, 'ok'


def scan(hycom_dir):
    """Verify every hy_*.nc in the cache. Returns a list of dicts."""
    out = []
    for path in sorted(glob.glob(os.path.join(hycom_dir, 'hy_*.nc'))):
        meta = parse_name(path)
        state, reason = verify_file(path)
        out.append({
            'file': os.path.basename(path),
            'prefix': meta[0] if meta else '?',
            'cycle': meta[1] if meta else '?',
            'valid': meta[2] if meta else '?',
            'state': state,
            'reason': reason,
            'size_mb': round(os.path.getsize(path) / (1024 * 1024), 3)
                       if os.path.isfile(path) else 0.0,
        })
    return out


def cycle_reach_days(rows, cycle):
    """How long a PFM forecast this cycle could drive, in days.

    The limit is the shortest contiguous run of any one variable measured from
    the cycle's own init: a cycle with 8 days of ssh and 2 days of salinity is
    a 2 day cycle. Files after an interior gap do not count, and neither does
    a block that starts late -- that is the 2026-10-14 case, where t3z and s3z
    begin at lead 39 h, so the cycle drives nothing at all despite holding 427
    good files.
    """
    t0 = datetime.strptime(cycle, '%Y-%m-%dT%H')
    worst = None
    for prefix in VAR_OF_PREFIX:
        step = 1 if prefix == 'ssh' else 3
        times = sorted(
            datetime.strptime(r['valid'], '%Y-%m-%dT%H')
            for r in rows
            if r['state'] == OK and r['cycle'] == cycle and r['prefix'] == prefix)
        if not times or times[0] != t0:
            return 0.0                    # nothing usable from the init
        reach = times[0]
        for a, b in zip(times, times[1:]):
            if (b - a).total_seconds() / 3600.0 > step:
                break
            reach = b
        if worst is None or reach < worst:
            worst = reach
    if worst is None:
        return 0.0
    return (worst - t0).total_seconds() / 86400.0


def inventory(rows, catalog, keep, log=print):
    """Close every run with what we actually have. Without this the log says
    what moved but never what is on hand, which is the thing you want at
    07:00 when a forecast is about to start."""
    log('INVENTORY  cycle          files   missing  absent  PFM days')
    for cyc in keep:
        want_n = len(cycle_files(cyc))
        have = [r for r in rows if r['cycle'] == cyc and r['state'] == OK]
        if catalog:
            absent = sum(1 for f in cycle_files(cyc)
                         if server_name(f) not in catalog)
        else:
            absent = 0
        missing = want_n - len(have) - absent
        days = cycle_reach_days(rows, cyc)
        note = ''
        if not have:
            note = '   (not on the server yet)' if absent == want_n else ''
        elif days == 0.0:
            note = '   <-- unusable: a variable is missing from the start'
        log('           %s  %4d/%-4d  %6d  %6d  %7.1f%s'
            % (cyc, len(have), want_n, missing, absent, days, note))


def mirror(hydir, archdir, keep, log=print):
    """Copy the verified cache to /dataSIO and drop what the cache has dropped.

    Copies by name and size, so an interrupted run resumes rather than
    restarting, and writes through a .part so a half-copy is never mistaken
    for a good file. Never the source of truth: if a mirrored file disagrees
    with the cache, the cache wins.
    """
    if not archdir:
        return
    try:
        os.makedirs(archdir, exist_ok=True)
    except OSError as e:
        log('MIRROR  cannot use %s: %s' % (archdir, e))
        return

    t0 = time.time()
    src = {}
    for path in glob.glob(os.path.join(hydir, 'hy_*.nc')):
        meta = parse_name(path)
        if meta and meta[1] in keep:
            src[os.path.basename(path)] = os.path.getsize(path)

    copied = removed = 0
    for name, size in sorted(src.items()):
        dst = os.path.join(archdir, name)
        try:
            if os.path.isfile(dst) and os.path.getsize(dst) == size:
                continue
        except OSError:
            pass
        tmp = dst + '.part'
        try:
            shutil.copyfile(os.path.join(hydir, name), tmp)
            os.replace(tmp, dst)
            try:
                os.chmod(dst, os.stat(dst).st_mode | 0o660)
            except OSError:
                pass
            copied += 1
        except OSError as e:
            log('MIRROR  could not copy %s: %s' % (name, e))
            try:
                os.remove(tmp)
            except OSError:
                pass

    # the mirror follows the cache, retention included
    for path in glob.glob(os.path.join(archdir, 'hy_*.nc')):
        if os.path.basename(path) not in src:
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass

    log('MIRROR  %s: %d copied, %d removed, %d files total, %.1fmin'
        % (archdir, copied, removed, len(src), (time.time() - t0) / 60.0))


def fetch_one(name, hydir, box, tries=PER_FILE_TRIES, scratch=None, log=print):
    """Fetch one file and only keep it if it verifies.

    Writes to <name>.part and renames on success, so a failed or stripped
    download never replaces a good file -- and never leaves a half-file that
    the next scan would have to reason about. Retries in place because the
    stripping is random per request: the same URL served 3 good and 2 stripped
    in five tries when measured on 2026-10-15.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ocn_funs_forecast as O
    url = O.myname_to_urls([name])[0]
    final = os.path.join(hydir, name)
    # When a DAP open fails, ncks decides the URL might be an HPSS path and
    # wgets it, creating ./tds.hycom.org/thredds/dodsC/... under the current
    # directory and leaving junk in it. That is where the tds.hycom.org trees
    # kept appearing in the repo. Run it somewhere disposable instead.
    if scratch is None:
        scratch = os.path.join(hydir, '.ncks_scratch')
    os.makedirs(scratch, exist_ok=True)
    part = final + '.part'
    s_, n_, w_, e_ = box
    last = 'no attempt'
    for k in range(tries):
        try:
            r = subprocess.run(
                ['ncks', '-d', 'lon,' + str(w_) + ',' + str(e_),
                 '-d', 'lat,' + str(s_) + ',' + str(n_),
                 url, '-4', '-O', part],
                capture_output=True, text=True, timeout=300, cwd=scratch)
        except subprocess.TimeoutExpired:
            last = 'timeout'
            continue
        if r.returncode != 0 or not os.path.isfile(part):
            err = (r.stderr or '')
            if 'file not found' in err or 'ERROR 404' in err:
                # the server does not have it; retrying only burns requests
                return False, 'not on server'
            last = 'ncks rc=%d' % r.returncode
            continue
        state, reason = verify_file(part)
        if state == OK:
            os.replace(part, final)
            return True, 'ok on try %d' % (k + 1)
        last = reason
        try:
            os.remove(part)
        except OSError:
            pass
    try:
        if os.path.exists(part):
            os.remove(part)
    except OSError:
        pass
    return False, last


def cycle_published(cycle, log=print):
    """Probe a single cycle. Superseded by the catalog for fill, kept because
    it answers the question for one cycle without pulling 3.3 MB -- useful
    from the shell and as a fallback if the catalog is ever unavailable."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ocn_funs_forecast as O
    name = 'hy_ssh_%s:00_%s:00.nc' % (cycle, cycle)
    url = O.myname_to_urls([name])[0] + '.dds'
    try:
        r = subprocess.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}',
                            '--max-time', '30', url],
                           capture_output=True, text=True, timeout=45)
        return r.stdout.strip() == '200'
    except Exception:                                        # noqa: BLE001
        return False


def prune(hydir, keep, log=print):
    """Drop whole cycles outside the keep list. Retention lives here so the
    grabber and PFM cannot fight over it -- one owner only."""
    seen = {}
    for path in glob.glob(os.path.join(hydir, 'hy_*.nc')):
        meta = parse_name(path)
        if meta:
            seen.setdefault(meta[1], []).append(path)
    removed = 0
    for cyc in sorted(seen):
        if cyc in keep:
            continue
        for path in seen[cyc]:
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
        log('PRUNE   cycle %s removed (%d files)' % (cyc, len(seen[cyc])))
    return removed


def fill(hydir, box, workers=DEFAULT_WORKERS, budget_min=DEFAULT_BUDGET_MIN,
         archdir=ARCHIVE_DIR, log=print):
    """Verify, delete what fails, refetch it along with anything missing.

    Newest cycle first, so a run that runs out of budget still leaves the
    freshest data complete.
    """
    t_start = time.time()
    deadline = t_start + budget_min * 60
    scratch = os.path.join(hydir, '.ncks_scratch')
    keep = recent_cycles()
    # Each run opens with a banner naming the local hour, because these all
    # land in one file per day and "which grab was this" is the first thing
    # you ask when reading it back.
    loc = datetime.now(timezone.utc).astimezone(ZoneInfo(LOCAL_TZ))
    log('=' * 68)
    log('%s GRAB   %s' % (loc.strftime('%H:%M %Z'),
                          loc.strftime('%a %Y-%m-%d')))
    log('=' * 68)
    log('START   keep=%s workers=%d budget=%dmin' % (','.join(keep), workers,
                                                     budget_min))
    prune(hydir, keep, log=log)

    # verify what is on disk, delete anything that fails
    rows = scan(hydir)
    bad = [r for r in rows if r['state'] != OK]
    for r in bad:
        try:
            os.remove(os.path.join(hydir, r['file']))
        except OSError:
            pass
    if bad:
        log('DELETE  %d bad files removed (%s)' %
            (len(bad), bad[0]['reason'].split('(')[0].strip()))

    have = {r['file'] for r in rows if r['state'] == OK}

    # Ask the server what it actually has before asking it for anything. One
    # request, and it removes every doomed fetch from the queue: unpublished
    # cycles, retired cycles, and the per-file gaps hycom leaves behind.
    catalog = fetch_catalog(hydir, log=log)

    want = []
    for cyc in keep:                       # newest first
        need = [f for f in cycle_files(cyc) if f not in have]
        if not need:
            continue
        if not catalog:
            want += need                   # no catalog: fall back to trying
            continue
        offered = [f for f in need if server_name(f) in catalog]
        absent = len(need) - len(offered)
        if absent and not cycle_in_catalog(cyc, catalog):
            log('SKIP    cycle %s not on the server yet (%d files deferred)'
                % (cyc, len(need)))
            continue
        if absent:
            # hycom has the cycle but not all of it. say which fields, because
            # a forecast started from here is missing that variable outright.
            #
            # Skipped for THIS run only. The queue is rebuilt from all five
            # cycles against a fresh listing every run, so anything hycom adds
            # later -- to an old cycle as readily as a new one -- is picked up
            # on the next pass without special handling. We do not know
            # whether hycom backfills: watching 2026-10-14 for 12.5 h on
            # 2026-10-15 showed zero names added to the whole server, but no
            # new cycle published in that window either, so that observes a
            # quiet server rather than a refusal to backfill. Either way the
            # behaviour is the same, which is why this is worth not guessing
            # about.
            miss = {}
            for f in need:
                if server_name(f) not in catalog:
                    miss[f[3:6]] = miss.get(f[3:6], 0) + 1
            log('GAP     cycle %s: server does not have %s right now '
                '(skipped this run, rechecked next run)'
                % (cyc, ', '.join('%d %s' % (n, k)
                                  for k, n in sorted(miss.items()))))
        want += offered
    log('SCAN    %d verified on disk, %d to fetch' % (len(have), len(want)))

    got = failed = skipped = 0
    reasons = {}
    # progress used to be "every 20 waves", i.e. every 200 files. hycom's
    # throughput swings by more than a factor of ten (58 files/min measured
    # one hour, 3 the next), so that was either chatter or forty minutes of
    # silence depending on the day. report on the clock instead.
    next_report = t_start + PROGRESS_EVERY_MIN * 60
    # submit in waves rather than all at once. submitting is instant, so
    # checking the deadline before each submit bounded nothing -- the pool
    # still had to drain every queued task and a 3 min budget ran for as long
    # as the work took. one wave per worker-count keeps the deadline real.
    done_n = 0
    for i in range(0, len(want), workers):
        if time.time() > deadline:
            skipped = len(want) - done_n
            break
        wave = want[i:i + workers]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fetch_one, name, hydir, box,
                              scratch=scratch, log=log): name
                    for name in wave}
            for fu in as_completed(futs):
                ok, why = fu.result()
                done_n += 1
                if ok:
                    got += 1
                else:
                    failed += 1
                    key = why.split('(')[0].strip()
                    reasons[key] = reasons.get(key, 0) + 1
        if time.time() >= next_report:
            done_frac = done_n / float(len(want)) if want else 1.0
            rate = got / max((time.time() - t_start) / 60.0, 1e-9)
            left = (len(want) - done_n) / rate if rate > 0 else float('inf')
            log('....    %d of %d fetched (%.0f%%), %d failed, %.1f/min, '
                '%s' % (got, len(want), 100 * done_frac, failed, rate,
                        ('~%.0f min to finish the queue' % left)
                        if left < 1e4 else 'stalled'))
            next_report = time.time() + PROGRESS_EVERY_MIN * 60

    shutil.rmtree(scratch, ignore_errors=True)

    mins = (time.time() - t_start) / 60.0
    for key, cnt in sorted(reasons.items(), key=lambda kv: -kv[1]):
        log('FAILED  %4d  %s' % (cnt, key))
    if skipped:
        log('BUDGET  %d files left for the next run' % skipped)
    log('DONE    %d fetched, %d failed, %d outstanding, %.1fmin'
        % (got, failed, failed + skipped, mins))

    # Exit status is read by cron, which mails the user on anything non-zero.
    # Leftover work is the normal state of this job, not a fault: the budget
    # is meant to run out, and the odd transient failure gets picked up an
    # hour later. Returning 1 for those would mail on nearly every run and the
    # mail would stop being read -- so reserve it for "this run accomplished
    # nothing and the next one probably will not either".
    # rescan rather than reusing the pre-fetch rows: the whole point is to
    # report what is on disk NOW, after everything this run did.
    rows = scan(hydir)
    inventory(rows, catalog, keep, log=log)
    mirror(hydir, archdir, keep, log=log)

    if want and got == 0 and failed > 0:
        return 1
    return 0


def in_maintenance(now=None):
    """True inside a hycom maintenance hour. Returns (bool, local time str)."""
    et = ZoneInfo('America/New_York')
    now = (now or datetime.now(timezone.utc)).astimezone(et)
    return now.hour in MAINT_HOURS_ET, now.strftime('%H:%M %Z')


def foreign_hycom_fetches():
    """PIDs of hycom ncks fetches that are not ours -- almost always a PFM
    forecast running its own download step.

    Matches on the process NAME and then reads the argv, rather than doing a
    pgrep -f for the whole pattern: -f matches any command line containing the
    text, so a shell running a diagnostic that merely mentions the url counts
    as a fetch. That produced a false BUSY while nothing was downloading.
    """
    try:
        r = subprocess.run(['pgrep', '-x', 'ncks'],
                           capture_output=True, text=True, timeout=30)
    except Exception:                                        # noqa: BLE001
        return []
    out = []
    for ln in r.stdout.split():
        try:
            pid = int(ln)
        except ValueError:
            continue
        try:
            with open('/proc/%d/cmdline' % pid, 'rb') as f:
                argv = f.read().decode('utf-8', 'replace')
        except OSError:
            continue                 # exited between pgrep and here
        if FETCH_HOST in argv:
            out.append(pid)
    return out


class CacheLock:
    """One grabber at a time. The cron fires hourly and a catch-up run can
    exceed an hour, so without this two runs fetch the same files into the
    same .part paths and fight each other."""

    def __init__(self, hydir):
        self.path = os.path.join(hydir, LOCK_NAME)
        self.fh = None

    def __enter__(self):
        self.fh = open(self.path, 'w')
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.fh.close()
            self.fh = None
            return False
        self.fh.write('%d %s\n' % (os.getpid(),
                                   datetime.now(timezone.utc).isoformat()))
        self.fh.flush()
        return True

    def __exit__(self, *exc):
        if self.fh:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()
        return False


def hycom_dir_from_pkl(pkl_fnm):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import init_funs_forecast as initfuns
    return initfuns.get_model_info(pkl_fnm)['hycom_data_dir']


def main():
    ap = argparse.ArgumentParser(description='HYCOM cache verifier')
    ap.add_argument('mode', choices=['verify', 'status', 'fill'])
    ap.add_argument('--pkl', default='/scratch/PFM_Simulations/forecast_info.pkl')
    ap.add_argument('--dir', default=None, help='override the cache directory')
    ap.add_argument('--json', action='store_true', help='machine-readable')
    ap.add_argument('--delete', action='store_true',
                    help='verify: remove files that fail verification')
    ap.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                    help='concurrent fetches (hycom allows 10 per IP)')
    ap.add_argument('--budget', type=float, default=DEFAULT_BUDGET_MIN,
                    help='minutes before fill stops and leaves the rest')
    ap.add_argument('--dry-run', action='store_true',
                    help='fill: say what would happen, touch nothing')
    ap.add_argument('--log', default=None,
                    help='fill: append to this file instead of stdout. '
                         'YYYYMMDD in the name is replaced with today (UTC)')
    ap.add_argument('--force', action='store_true',
                    help='fill: run even during a hycom maintenance hour')
    ap.add_argument('--archive', default=ARCHIVE_DIR,
                    help='fill: mirror the cache here (empty string = off)')
    ap.add_argument('--also-stdout', action='store_true',
                    help='fill: with --log, print as well as append. PFM uses '
                         'this so the grab appears both in the daily grabber '
                         'log and in the forecast log. The cron must not: its '
                         'wrapper already redirects stdout to the same file.')
    args = ap.parse_args()

    hydir = args.dir or hycom_dir_from_pkl(args.pkl)

    if args.workers > 10:
        print('refusing --workers %d: hycom documents a limit of 10 per IP'
              % args.workers)
        return 2

    if args.mode == 'fill':
        box = None
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import init_funs_forecast as initfuns
        b = initfuns.get_model_info(args.pkl)['latlonbox']['L1']
        box = (b[0], b[1], b[2] + 360.0, b[3] + 360.0)
        sink = sys.stdout
        if args.log:
            path = args.log.replace(
                'YYYYMMDD', datetime.now(timezone.utc).strftime('%Y%m%d'))
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            sink = open(path, 'a')

        def stamp(m):
            line = '%s  %s\n' % (
                datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ'), m)
            sink.write(line)
            sink.flush()
            if args.also_stdout and sink is not sys.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()

        down, et_now = in_maintenance()
        if down and not args.force:
            stamp('MAINT   hycom is in its maintenance hour (%s) -- not running'
                  % et_now)
            return 0

        busy = foreign_hycom_fetches()
        if busy and not args.force:
            stamp('BUSY    %d hycom fetches already running (pid %s) -- a '
                  'forecast is downloading; not adding to its connections'
                  % (len(busy), ','.join(str(p) for p in busy[:5])))
            return 0
        if args.dry_run:
            keep = recent_cycles()
            rows = scan(hydir)
            have = {r['file'] for r in rows if r['state'] == OK}
            bad = [r for r in rows if r['state'] != OK]
            catalog = fetch_catalog(hydir, log=stamp)
            want = []
            for c in keep:
                nd = [f for f in cycle_files(c) if f not in have]
                want += ([f for f in nd if server_name(f) in catalog]
                         if catalog else nd)
            stale = sorted({r['cycle'] for r in rows if r['cycle'] not in keep})
            stamp('DRY-RUN keep=%s' % ','.join(keep))
            stamp('DRY-RUN would prune cycles: %s' % (','.join(stale) or 'none'))
            stamp('DRY-RUN would delete %d bad, fetch %d files' % (len(bad), len(want)))
            return 0
        lock = CacheLock(hydir)
        if not lock.__enter__():
            stamp('LOCKED  another grabber is still running -- not starting')
            return 0
        try:
            return fill(hydir, box, workers=args.workers,
                        budget_min=args.budget,
                        archdir=args.archive or None, log=stamp)
        finally:
            lock.__exit__()

    rows = scan(hydir)

    if args.json:
        print(json.dumps({'dir': hydir,
                          'checked_at': datetime.now(timezone.utc).isoformat(),
                          'files': rows}, indent=1))
        return 0 if not [r for r in rows if r['state'] != OK] else 1

    nok = len([r for r in rows if r['state'] == OK])
    bad = [r for r in rows if r['state'] == BAD]

    if args.mode == 'status':
        print('%s  %d files: %d ok, %d bad' % (hydir, len(rows), nok, len(bad)))
        cycles = sorted({r['cycle'] for r in rows})
        for c in cycles:
            cr = [r for r in rows if r['cycle'] == c]
            cb = [r for r in cr if r['state'] != OK]
            line = '  %s  %4d files' % (c, len(cr))
            for p in ('ssh', 't3z', 's3z', 'u3z', 'v3z'):
                pr = [r for r in cr if r['prefix'] == p]
                pb = [r for r in pr if r['state'] != OK]
                line += '  %s %d%s' % (p, len(pr), ('-%d' % len(pb)) if pb else '')
            print(line + ('   <-- %d bad' % len(cb) if cb else ''))
        return 0 if not bad else 1

    # verify
    print('cache: %s' % hydir)
    print('%d files: %d ok, %d bad' % (len(rows), nok, len(bad)))
    if bad:
        print('')
        reasons = {}
        for r in bad:
            key = r['reason'].split('(')[0].split('%')[0].strip()
            reasons.setdefault(key, []).append(r)
        for key in sorted(reasons, key=lambda k: -len(reasons[k])):
            rs = reasons[key]
            print('  %4d  %s' % (len(rs), key))
            for r in rs[:4]:
                print('          %s  (%.3f MB)' % (r['file'], r['size_mb']))
            if len(rs) > 4:
                print('          ... and %d more' % (len(rs) - 4))
        if args.delete:
            gone = 0
            for r in bad:
                try:
                    os.remove(os.path.join(hydir, r['file']))
                    gone += 1
                except OSError as e:
                    print('  could not remove %s: %s' % (r['file'], e))
            print('')
            print('  deleted %d of %d bad files; fill will refetch them' %
                  (gone, len(bad)))
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
