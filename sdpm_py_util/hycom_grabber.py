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
import sys
import glob
import json
import time
import shutil
import argparse
import subprocess
from datetime import datetime, timezone, timedelta
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
PER_FILE_TRIES = 5     # stripping is random per request: ~40% when measured,
                       # so a handful of immediate retries usually wins


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


def fetch_one(name, hydir, box, tries=PER_FILE_TRIES, log=print):
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
    part = final + '.part'
    s_, n_, w_, e_ = box
    last = 'no attempt'
    for k in range(tries):
        try:
            r = subprocess.run(
                ['ncks', '-d', 'lon,' + str(w_) + ',' + str(e_),
                 '-d', 'lat,' + str(s_) + ',' + str(n_),
                 url, '-4', '-O', part],
                capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            last = 'timeout'
            continue
        if r.returncode != 0 or not os.path.isfile(part):
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
    """One cheap HEAD-ish probe per cycle before queueing its 453 files.

    Today's 12Z cycle usually is not up yet. Without this the fill spends its
    whole budget failing against URLs that do not exist -- 1.5 min produced
    150 failures and 0 fetches when measured -- and the hycom forum warns that
    requesting URLs that do not exist is itself a way to get an IP blocked.
    """
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
         log=print):
    """Verify, delete what fails, refetch it along with anything missing.

    Newest cycle first, so a run that runs out of budget still leaves the
    freshest data complete.
    """
    t_start = time.time()
    deadline = t_start + budget_min * 60
    keep = recent_cycles()
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
    want = []
    for cyc in keep:                       # newest first
        need = [f for f in cycle_files(cyc) if f not in have]
        if not need:
            continue
        if not cycle_published(cyc, log=log):
            log('SKIP    cycle %s not published yet (%d files deferred)'
                % (cyc, len(need)))
            continue
        want += need
    log('SCAN    %d verified on disk, %d to fetch' % (len(have), len(want)))

    got = failed = skipped = 0
    reasons = {}
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
            futs = {ex.submit(fetch_one, name, hydir, box, log=log): name
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
        if got and (i // workers) % 20 == 0:
            log('....    %d fetched, %d failed, %.1fmin elapsed'
                % (got, failed, (time.time() - t_start) / 60.0))

    mins = (time.time() - t_start) / 60.0
    for key, cnt in sorted(reasons.items(), key=lambda kv: -kv[1]):
        log('FAILED  %4d  %s' % (cnt, key))
    if skipped:
        log('BUDGET  %d files left for the next run' % skipped)
    log('DONE    %d fetched, %d failed, %d outstanding, %.1fmin'
        % (got, failed, failed + skipped, mins))
    return 0 if (failed + skipped) == 0 else 1


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
        stamp = lambda m: print('%s  %s' % (
            datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ'), m), flush=True)
        if args.dry_run:
            keep = recent_cycles()
            rows = scan(hydir)
            have = {r['file'] for r in rows if r['state'] == OK}
            bad = [r for r in rows if r['state'] != OK]
            want = [f for c in keep for f in cycle_files(c) if f not in have]
            stale = sorted({r['cycle'] for r in rows if r['cycle'] not in keep})
            stamp('DRY-RUN keep=%s' % ','.join(keep))
            stamp('DRY-RUN would prune cycles: %s' % (','.join(stale) or 'none'))
            stamp('DRY-RUN would delete %d bad, fetch %d files' % (len(bad), len(want)))
            return 0
        return fill(hydir, box, workers=args.workers, budget_min=args.budget,
                    log=stamp)

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
