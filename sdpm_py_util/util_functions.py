import os
import glob
import glob as _glob
import re as _re
import numpy as np
import warnings
import pickle
import shutil
import sys
import subprocess
sys.path.append('../sdpm_py_util')
import init_funs_forecast as initfuns


def archive_hind_nc(nc_type,lvl_str,pkl_fnm):
    PFM = initfuns.get_model_info(pkl_fnm)
    if nc_type == 'his':
        input_file = PFM['lv'+lvl_str+'_his_name_full']
        archive_dir = PFM['hind_lv'+lvl_str+'_archive_dir'] 
    elif nc_type == 'river':
        input_file = PFM['lv4_forc_dir'] +'/' + PFM['lv4_river_file']
        archive_dir = PFM['hind_river_archive_dir'] 

    cmd = ['mv',input_file,archive_dir]
    subprocess.Popen(cmd)

def copy_mv_nc_file(nc_type,lvl,pkl_fnm):
    # this copies an atm.nc or river.nc file to the archive location
    # nc_type can be ['atm','river']
    # lvl can be ['lv1','lv2','lv3','lv4']

    lvl_upper = lvl.upper()
    PFM = initfuns.get_model_info(pkl_fnm)
    fcdate = PFM['fetch_time'].strftime("%Y%m%d%H")
    dir_out = PFM['archive_dir']
    if nc_type == 'atm':
        fn_in_full = PFM[lvl+'_forc_dir'] + '/' + PFM[lvl+'_atm_file']
        fn_out = 'atm_' + PFM['atm_model'] + '_' + lvl_upper + '_' + fcdate + '.nc'
    if nc_type == 'river':
        fn_in_full = PFM[lvl+'_forc_dir'] + '/' + PFM[lvl+'_river_file']
        fn_out = 'river_' + lvl_upper + '_' + fcdate + '.nc'

    fn_out_full = dir_out + fn_out
    shutil.move(fn_in_full, fn_out_full)

# ---------------------------------------------------------------------------
# LV4 SWAN restart archive.  PRUNED to 10 days, unlike the grib and LV1_atm
# archives -- one 72-rank set is 2.5 GB, which is a different order entirely
# from 14 MB of gribs.
#
# ONLY THE _024 SET.  That is the one the next cycle hotstarts from.  PFM's
# check_and_move also keeps _000.._120, 21 snapshots per cycle, but nothing
# reads the intermediates and archiving them all would cost ~785 GB for 10-15
# days instead of ~25 GB.
#
#   <ARCHIVE>/DA/    written by the DA's LV4 forward run
#   <ARCHIVE>/PFM/   written by PFM's own LV4 forecast
#
# Kept apart because they are different solutions for the same clock and would
# otherwise collide: PFM's name for a 10/14 start is LV4_swan_rst_202610140000_024
# and the DA's is LV4_DA_swan_rst_202610140000_024, but only because the DA was
# renamed -- before that they were identical strings.
# ---------------------------------------------------------------------------
LV4_SWAN_RST_ARCHIVE = '/dataSIO/PFM_Simulations/Archive/LV4_swan_restarts'

# <anything>_<12-digit stamp>_<3-digit lead>.dat-<3-digit rank>
# Parsed by REGEX, not fixed offsets: the prefix is LV4_swan_rst_ for PFM and
# LV4_DA_swan_rst_ for the DA, so any offset-based parse breaks on one of them.
_LV4_SWAN_RST_RE = _re.compile(r'_(\d{12})_(\d{3})\.dat-(\d{3})$')


def _lv4_swan_rst_dir(PFM, which):
    root = str(PFM.get('lv4_swan_rst_archive_dir', LV4_SWAN_RST_ARCHIVE))
    return os.path.join(root, which)


def _lv4_swan_024_name(PFM):
    """This run's _024 basename, whichever naming its writer used."""
    nm = PFM['lv4_swan_rst_name']
    if nm.endswith('_024.dat'):
        return nm                      # the DA writes the offset in directly
    if nm.endswith('.dat'):
        return nm[:-4] + '_024.dat'    # PFM's check_and_move adds it
    return nm + '_024.dat'


def archive_lv4_swan_restarts(pkl_fnm, which):
    """Copy this run's _024 SWAN restart set into the archive.

    which selects the subdir: 'DA', 'PFM', or a sandboxed run's run_tag.

    Write-once per file.  Every failure is reported and swallowed: archiving a
    restart is never a reason to fail a forecast that has already run.

    Returns the number of parts newly copied.
    """
    import shutil as _sh

    # 'DA' and 'PFM' are the two real sides; a sandboxed run passes its own
    # run_tag ('testing') so its restarts get their own subdir instead of
    # landing in the PFM side's.  Anything else is a caller bug.
    if not (isinstance(which, str) and _re.fullmatch(r'[A-Za-z0-9_]{1,16}', which)):
        print('archive_lv4_swan_restarts: bad "which", got ' + repr(which))
        return 0
    try:
        PFM = initfuns.get_model_info(pkl_fnm)
        src_dir = PFM['restart_swanfiles_dir_out']
        base = _lv4_swan_024_name(PFM)
        src = sorted(_glob.glob(os.path.join(src_dir, base + '-*')))
        if not src:
            print('archive_lv4_swan_restarts: no ' + base + '-* in ' + src_dir +
                  ' to archive')
            return 0
        dir_out = _lv4_swan_rst_dir(PFM, which)
        os.makedirs(dir_out, exist_ok=True)
        new = have = 0
        for f in src:
            dst = os.path.join(dir_out, os.path.basename(f))
            if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(f):
                have += 1
                continue
            tmp = dst + '.part'
            _sh.copyfile(f, tmp)
            os.replace(tmp, dst)
            new += 1
        mb = sum(os.path.getsize(f) for f in src) / 1e6
        print('archive_lv4_swan_restarts[%s]: %s -> %s  (%d new, %d already '
              'there, %.1f GB)' % (which, base, dir_out, new, have, mb / 1000.0))
        return new
    except Exception as e:                                   # noqa: BLE001
        print('archive_lv4_swan_restarts: ' + type(e).__name__ + ': ' + str(e) +
              ' -- not archived (the run is unaffected)')
        return 0


def prune_lv4_swan_restarts(days=10, which=None, pkl_fnm=None):
    """Drop archived LV4 SWAN restarts whose STAMP is older than `days`.

    Pruned on the stamp in the name, not on mtime: the stamp is the simulation
    start the set belongs to, while mtime is merely when it happened to be
    copied -- a backfill would otherwise look new and a re-copy would reset the
    clock on something old.

    Returns the number of files removed.
    """
    import datetime as _dt

    try:
        root = LV4_SWAN_RST_ARCHIVE
        if pkl_fnm is not None:
            root = str(initfuns.get_model_info(pkl_fnm).get(
                'lv4_swan_rst_archive_dir', root))
        sides = (which,) if which else ('DA', 'PFM')
        cut = _dt.datetime.now() - _dt.timedelta(days=float(days))
        n = 0
        for side in sides:
            d = os.path.join(root, side)
            if not os.path.isdir(d):
                continue
            for f in sorted(_glob.glob(os.path.join(d, '*.dat-*'))):
                m = _LV4_SWAN_RST_RE.search(os.path.basename(f))
                if m is None:
                    continue           # not ours to delete; see the regex note
                try:
                    t = _dt.datetime.strptime(m.group(1), '%Y%m%d%H%M')
                except ValueError:
                    continue
                if t < cut:
                    try:
                        os.remove(f)
                        n += 1
                    except OSError:
                        pass
        if n:
            print('prune_lv4_swan_restarts: removed %d file(s) with a stamp '
                  'older than %s days' % (n, days))
        return n
    except Exception as e:                                   # noqa: BLE001
        print('prune_lv4_swan_restarts: ' + type(e).__name__ + ': ' + str(e))
        return 0

_LV1_BC_META_KEYS = ('vinfo', 'ocean_time_ref')


def _share_with_group(path):
    """Make an archived file writable by the rest of feddersen-lab.

    The archive directory is already 2775, so anyone in the group can create
    files there -- but the files themselves land 644 under the usual 022
    umask, so whoever did NOT write one cannot overwrite it. In practice that
    meant a forecast run by one of us could archive a boundary file that the
    other could not then replace. Fixing the umask would work too, but it has
    to be right in cron, in sbatch and in every shell anyone runs this from;
    setting the mode on the file we just wrote is one line and cannot drift.

    Swallows failures: this is a convenience, never a reason to fail a run.
    """
    try:
        os.chmod(path, os.stat(path).st_mode | 0o660)
    except OSError as e:
        print('_share_with_group: could not chmod ' + path + ': ' + str(e))


def copy_into_archive(src, dst, label=''):
    """Copy src onto dst in a shared archive, whoever owns what is already there.

    Two distinct failures were costing us every archive copy on a rerun, and
    neither is fixed by chmod:

      EACCES (13) -- the destination exists, is owned by the other user and is
      mode 644, so opening it for writing is refused.  'cp' and shutil.copy
      both die here.  This is how the corrected 2026-10-18 LV4 web file never
      reached /dataSIO: the web kept serving the previous night's run.

      EPERM (1) -- the destination IS writable, the data copies fine, and then
      shutil.copy/copy2 tries to apply the source's mode or mtime to a file
      owned by someone else and the kernel refuses.  That is the ~90
      "Operation not permitted" lines on the cdip files, every one of which
      had already been written successfully.

    Writing a temp file in the destination directory and renaming it over the
    target avoids both: rename needs write permission on the DIRECTORY, not on
    the file being replaced, and it never touches the old file's metadata.  It
    is also atomic, so a reader never sees a half-copied archive file.

    Then make the result group-writable so the next person does not hit the
    same wall -- the umask here is 022, so a fresh file lands 644 and the
    problem comes straight back.

    dst may be a full path or a destination directory, like shutil.copy.

    Failures are reported and swallowed: archiving must never take a run down.
    Returns True if the file is in place.
    """
    try:
        if os.path.isdir(dst):          # called with a destination DIRECTORY
            dst = os.path.join(dst, os.path.basename(src))
        dst_dir = os.path.dirname(os.path.abspath(dst))
        os.makedirs(dst_dir, exist_ok=True)
        tmp = os.path.join(dst_dir, '.' + os.path.basename(dst) + '.part')
        shutil.copyfile(src, tmp)       # data only: no mode, no mtime, no EPERM
        os.replace(tmp, dst)            # needs the DIRECTORY, not the old file
        _share_with_group(dst)
        return True
    except OSError as e:
        print('copy_into_archive: ' + (label + ': ' if label else '') +
              src + ' -> ' + dst + ' FAILED (' + type(e).__name__ + ': ' +
              str(e) + ') -- continuing, archiving never fails a run')
        try:
            if 'tmp' in dir() and os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False


def _lv1_bc_meta_name(stamp):
    return 'LV1_BCmeta_' + stamp + '.pkl'


def archive_lv1_bc_meta(pkl_fnm, stamp):
    """Archive the two metadata keys LV2 needs out of the LV1 OCN_BC pickle.

    mk_LV2_BC_dict_edges builds LV2's boundary from LV1's HISTORY file, but it
    reads two things out of LV1's OCN_BC pickle: 'vinfo' (variable long_names
    and units) and 'ocean_time_ref' (the epoch).  Nothing else.  When a rerun
    reuses an archived BC .nc it never builds that pickle, so LV2 died with
    FileNotFoundError on ocnBC_tmp_pckl_file.pkl.

    The full pickle is ~44 MB and almost all of it is boundary field data that
    is already in the .nc.  So archive only those two keys -- a few kB.

    Failures are reported and swallowed: this must never take a forecast down.
    Returns the destination path, or None.
    """
    import pickle as _pk

    try:
        PFM = initfuns.get_model_info(pkl_fnm)
        src = os.path.join(PFM['lv1_forc_dir'], PFM['lv1_ocnBC_tmp_pckl_file'])
        if not os.path.isfile(src):
            print('archive_lv1_bc_meta: no ' + src + ' to archive')
            return None
        dir_out = PFM.get('archive_lv1_bc_dir',
                          '/dataSIO/PFM_Simulations/Archive/LV1_BC_files/')
        dst = os.path.join(dir_out, _lv1_bc_meta_name(stamp))
        if os.path.isfile(dst):
            print('archive_lv1_bc_meta: ' + os.path.basename(dst) +
                  ' already archived')
            return dst
        with open(src, 'rb') as fin:
            BC = _pk.load(fin)
        meta = {k: BC[k] for k in _LV1_BC_META_KEYS if k in BC}
        missing = [k for k in _LV1_BC_META_KEYS if k not in meta]
        if missing:
            print('archive_lv1_bc_meta: ' + os.path.basename(src) +
                  ' has no ' + ', '.join(missing) + '; not archiving')
            return None
        os.makedirs(dir_out, exist_ok=True)
        tmp = dst + '.part'
        with open(tmp, 'wb') as fout:
            _pk.dump(meta, fout, protocol=_pk.HIGHEST_PROTOCOL)
        os.replace(tmp, dst)
        _share_with_group(dst)
        print('archive_lv1_bc_meta: archived ' + os.path.basename(dst) +
              ' (%d bytes)' % os.path.getsize(dst))
        return dst
    except Exception as e:                                   # noqa: BLE001
        print('archive_lv1_bc_meta: ' + type(e).__name__ + ': ' + str(e) +
              ' -- not archived (the forecast is unaffected)')
        return None


def stage_lv1_bc_meta(pkl_fnm):
    """Put a minimal LV1 OCN_BC pickle in place so LV2's BC build can proceed.

    Only called when the LV1 BC was REUSED from the archive, which means the
    real pickle was never built.  See archive_lv1_bc_meta for why two keys are
    enough.

    Source preference:
      1. the metadata archived for THIS stamp -- exact,
      2. the newest metadata archived for any stamp,
      3. the live operational LV1 pickle on scratch.
    2 and 3 are not a compromise: 'vinfo' is static variable metadata and
    'ocean_time_ref' is a fixed epoch (1999-01-01), so they do not vary by day.
    The day-specific numbers LV2 actually uses come from LV1's history file.

    Returns True if a pickle is in place, False otherwise (LV2 will then fail
    loudly, which is the right outcome -- better than a silent wrong BC).
    """
    import glob as _gl
    import pickle as _pk

    try:
        PFM = initfuns.get_model_info(pkl_fnm)
        dst = os.path.join(PFM['lv1_forc_dir'], PFM['lv1_ocnBC_tmp_pckl_file'])
        if os.path.isfile(dst):
            return True

        dir_in = PFM.get('archive_lv1_bc_dir',
                         '/dataSIO/PFM_Simulations/Archive/LV1_BC_files/')
        stamp = PFM['sim_time_1'].strftime('%Y%m%d%H')
        cands = [os.path.join(dir_in, _lv1_bc_meta_name(stamp))]
        cands += sorted(_gl.glob(os.path.join(dir_in, _lv1_bc_meta_name('*'))),
                        reverse=True)
        cands.append('/scratch/PFM_Simulations/LV1_Forecast/Forc/'
                     + PFM['lv1_ocnBC_tmp_pckl_file'])

        for src in cands:
            if not os.path.isfile(src):
                continue
            try:
                with open(src, 'rb') as fin:
                    BC = _pk.load(fin)
            except Exception as e:                           # noqa: BLE001
                print('stage_lv1_bc_meta: cannot read ' + src + ': ' +
                      type(e).__name__ + ': ' + str(e))
                continue
            if not all(k in BC for k in _LV1_BC_META_KEYS):
                continue
            meta = {k: BC[k] for k in _LV1_BC_META_KEYS}
            os.makedirs(PFM['lv1_forc_dir'], exist_ok=True)
            tmp = dst + '.part'
            with open(tmp, 'wb') as fout:
                _pk.dump(meta, fout, protocol=_pk.HIGHEST_PROTOCOL)
            os.replace(tmp, dst)
            note = '' if src == cands[0] else \
                   '  (metadata only, taken from another cycle -- vinfo and ' \
                   'ocean_time_ref do not vary by day)'
            print('stage_lv1_bc_meta: wrote ' + os.path.basename(dst) +
                  ' from ' + src + note)
            return True

        print('stage_lv1_bc_meta: found no LV1 OCN_BC metadata to stage; '
              'LV2 BC build will fail')
        return False
    except Exception as e:                                   # noqa: BLE001
        print('stage_lv1_bc_meta: ' + type(e).__name__ + ': ' + str(e))
        return False


def reuse_archived_lv1_bc(pkl_fnm):
    """Use an archived LV1 BC instead of rebuilding it, when one fits exactly.

    Returns True if the archived file was put in place and the HYCOM download
    can be skipped; False to build normally.

    IN NORMAL OPERATION THIS ALWAYS RETURNS FALSE.  The lookup is by the
    forecast's own start time, and tonight's BC has not been archived yet --
    archive_lv1_bc runs AFTER the BC is built, later in this same function.  So
    a new forecast never finds one and nothing changes.  It hits only when a
    PAST forecast start is re-run, which is the testing and hindcast-rerun case.

    That is also why it is worth having: a rerun then uses the SAME boundary the
    original forecast used, rather than whatever HYCOM happens to serve today.
    hycom_data/ keeps about two inits and is cleaned every cycle, so without
    this a rerun of a week-old date silently gets a different ocean -- and on
    2026-10-14 it also meant re-downloading 614 files, roughly 110 minutes,
    for a boundary that was already on disk.

    The file must COVER the window, checked on its time axis rather than its
    name: the 10/12 archive spans exactly one day, which suits --length 1 and
    would be wrong for a five-day run.

    Disable with PFM['lv1_bc_reuse'] = False.
    """
    import datetime as _dt
    import shutil as _sh
    import numpy as _np
    import netCDF4 as _nc

    try:
        PFM = initfuns.get_model_info(pkl_fnm)
        if not PFM.get('lv1_bc_reuse', True):
            return False
        t1, t2 = PFM['sim_time_1'], PFM['sim_time_2']
        dir_in = PFM.get('archive_lv1_bc_dir',
                         '/dataSIO/PFM_Simulations/Archive/LV1_BC_files/')
        src = os.path.join(dir_in, 'LV1_BC_' + t1.strftime('%Y%m%d%H') + '.nc')
        if not os.path.isfile(src):
            return False

        t0 = PFM['modtime0']
        with _nc.Dataset(src) as d:
            spans = []
            for v in [x for x in d.variables if x.endswith('_time')]:
                tv = _np.asarray(d[v][:], dtype=float).ravel()
                if tv.size == 0:
                    return False
                u = str(getattr(d[v], 'units', 'days')).strip().lower()
                sc = 86400.0 if u.startswith('day') else 1.0
                spans.append((t0 + _dt.timedelta(seconds=float(tv.min()) * sc),
                              t0 + _dt.timedelta(seconds=float(tv.max()) * sc)))
        if not spans:
            return False
        lo, hi = max(s[0] for s in spans), min(s[1] for s in spans)
        slack = _dt.timedelta(minutes=1)
        if lo > t1 + slack or hi < t2 - slack:
            print('reuse_archived_lv1_bc: ' + os.path.basename(src) + ' spans ' +
                  lo.strftime('%Y-%m-%d %H:%M') + ' .. ' + hi.strftime('%Y-%m-%d %H:%M') +
                  ', which does not cover ' + t1.strftime('%Y-%m-%d %H:%M') + ' .. ' +
                  t2.strftime('%Y-%m-%d %H:%M') + ' -- building from HYCOM instead')
            return False

        dst = os.path.join(PFM['lv1_forc_dir'], PFM['lv1_bc_file'])
        os.makedirs(PFM['lv1_forc_dir'], exist_ok=True)
        tmp = dst + '.part'
        _sh.copyfile(src, tmp)
        os.replace(tmp, dst)
        print('*** reuse_archived_lv1_bc: using the ARCHIVED boundary ' +
              os.path.basename(src) + ' (covers ' + lo.strftime('%Y-%m-%d %H') +
              'Z .. ' + hi.strftime('%Y-%m-%d %H') + 'Z); skipping the HYCOM '
              'download.  This is a RERUN of a past forecast start. ***')
        # LV2's BC build reads 'vinfo' and 'ocean_time_ref' out of LV1's
        # OCN_BC pickle, which the skipped build never writes.
        stage_lv1_bc_meta(pkl_fnm)
        return True
    except Exception as e:                                   # noqa: BLE001
        print('reuse_archived_lv1_bc: ' + type(e).__name__ + ': ' + str(e) +
              ' -- building from HYCOM instead')
        return False


def archive_lv1_bc(pkl_fnm):
    """Archive LV1_OCEAN_BC.nc as LV1_BC_<YYYYMMDDHH>.nc, stamped from its DATA.

    Why this exists: hycom_data/ keeps only ~2 inits and is cleaned every cycle,
    so nothing on disk records the ocean boundary a past forecast actually ran
    with.  The BC file is ~44 MB, so a year of them is well under 20 GB -- cheap
    insurance against a question ("what BC did the 2026-10-12 forecast use?")
    that is otherwise unanswerable after about two days.

    The stamp is the FIRST ocean_time record, read from the file, not
    PFM['fetch_time'].  Those should agree, and when they do not the file is the
    truth -- a stale derived pickle produces a perfectly valid BC for the wrong
    day, and naming it after the intended day would hide exactly that.  (The
    forecast stack has had that failure: see _assert_covers_window in the DA
    stages.)

    Archiving must never take a forecast down, so every failure here is reported
    and swallowed.  Returns the destination path, or None if nothing was
    archived.
    """
    import datetime as _dt
    import shutil as _sh
    import numpy as _np
    import netCDF4 as _nc

    try:
        PFM = initfuns.get_model_info(pkl_fnm)
        # A tagged run (PFM_testing, DA) must not write the operational BC
        # archive.  It is write-once, so a 1-day test BC filed under tonight's
        # stamp would permanently occupy the slot the real 5-day BC belongs in.
        # reuse_archived_lv1_bc checks coverage, so this is provenance, not a
        # wrong boundary -- but provenance is the whole point of the archive.
        _tag = PFM.get('run_tag')
        if _tag:
            print('archive_lv1_bc: run_tag=' + repr(_tag) + ' -- NOT writing the '
                  'operational LV1_BC archive')
            return None
        src = os.path.join(PFM['lv1_forc_dir'], PFM['lv1_bc_file'])
        if not os.path.isfile(src):
            print('archive_lv1_bc: no ' + src + ' to archive')
            return None

        dir_out = PFM.get('archive_lv1_bc_dir',
                          '/dataSIO/PFM_Simulations/Archive/LV1_BC_files/')

        # the time axis carries bare units ('days'), not a CF string, so the
        # epoch comes from the model-info dict rather than from the attribute
        t0 = PFM['modtime0']
        with _nc.Dataset(src) as d:
            tname = ('ocean_time' if 'ocean_time' in d.variables
                     else next((v for v in d.variables if v.endswith('_time')), None))
            if tname is None:
                print('archive_lv1_bc: ' + os.path.basename(src) +
                      ' has no *_time variable; not archiving')
                return None
            tv = _np.asarray(d[tname][:], dtype=float).ravel()
            units = str(getattr(d[tname], 'units', 'days')).strip().lower()
        if tv.size == 0:
            print('archive_lv1_bc: ' + tname + ' is empty; not archiving')
            return None
        scale = 86400.0 if units.startswith('day') else 1.0
        tfirst = t0 + _dt.timedelta(seconds=float(tv.min()) * scale)

        stamp = tfirst.strftime('%Y%m%d%H')
        fn_out = 'LV1_BC_' + stamp + '.nc'
        dst = os.path.join(dir_out, fn_out)

        # a few-kB companion holding the 'vinfo' / 'ocean_time_ref' keys that
        # LV2's BC build needs.  Archived under the SAME stamp, and before the
        # write-once test below so an already-archived .nc still gains its
        # metadata on the next cycle.
        archive_lv1_bc_meta(pkl_fnm, stamp)

        want = os.path.getsize(src)
        if os.path.isfile(dst):
            # WRITE ONCE.  The operational forecast is the gold standard for a
            # given stamp and must never be replaced by a later writer.
            #
            # The names really can collide: the stamp is the BC's first
            # ocean_time, so a forecast initialised 2026-10-13 00Z and a PFM_DA
            # window running 2026-10-13 00Z -> 10-14 00Z both resolve to
            # LV1_BC_2026101300.nc -- from different HYCOM inits, over different
            # spans.  PFM_DA does not archive today, but nothing stops it later,
            # and an earlier version of this function replaced the file whenever
            # the size differed, which is exactly when the two are NOT the same
            # data.
            #
            # A partial capture is also possible: LV1_BC_2026101200.nc was
            # archived by hand mid-rebuild and holds 25 zeta records instead of
            # 109.  Refusing to overwrite keeps a bad file, but silently
            # replacing a good one is worse, and a size mismatch is reported so
            # it can be dealt with deliberately.
            have = os.path.getsize(dst)
            if have == want:
                print('archive_lv1_bc: ' + fn_out + ' already archived')
            else:
                print('archive_lv1_bc: KEEPING the existing ' + fn_out +
                      ' (%d bytes) and NOT replacing it with %d bytes from %s. '
                      % (have, want, src) +
                      'Same stamp, different content -- check which is right '
                      'before removing one by hand.')
            return dst

        os.makedirs(dir_out, exist_ok=True)
        # copy via a temp name in the SAME directory, then rename: an
        # interrupted copy must not leave a short file that the size test above
        # would later mistake for a complete one
        tmp = dst + '.part'
        _sh.copyfile(src, tmp)
        os.replace(tmp, dst)
        _share_with_group(dst)

        exp = PFM['fetch_time'].strftime('%Y%m%d%H')
        note = '' if stamp == exp else ('   NOTE: fetch_time says ' + exp +
                                        ', the file says ' + stamp)
        print('archive_lv1_bc: ' + src + ' -> ' + dst +
              ' (%.1f MB)' % (want / 1e6) + note)
        return dst
    except Exception as e:
        print('archive_lv1_bc: FAILED (' + type(e).__name__ + ': ' + str(e) +
              ') -- continuing, archiving never fails a forecast')
        return None


def copy_mv_nc_file_v2(nc_type,lvl,pkl_fnm):
    # this copies an atm.nc or river.nc file to the archive location
    # nc_type can be ['atm','river']
    # lvl can be ['lv1','lv2','lv3','lv4']

    lvl_upper = lvl.upper()
    PFM = initfuns.get_model_info(pkl_fnm)
    fcdate = PFM['fetch_time'].strftime("%Y%m%d%H")
    dir_out = PFM['archive_dir']
    if nc_type == 'atm':
        fn_in_full = PFM[lvl+'_forc_dir'] + '/' + PFM[lvl+'_atm_file']
        fn_out = 'atm_' + PFM['atm_model'] + '_' + lvl_upper + '_' + fcdate + '.nc'
    if nc_type == 'river':
        fn_in_full = PFM[lvl+'_forc_dir'] + '/' + PFM[lvl+'_river_file']
        fn_out = 'river_' + lvl_upper + '_' + fcdate + '.nc'

    fn_out_full = dir_out + fn_out
    # was: subprocess.Popen(['cp', ...]) -- which both ignored its own exit
    # code and could not overwrite an archive file owned by the other user.
    copy_into_archive(fn_in_full, fn_out_full, label=nc_type + ' ' + lvl_upper)


def delete_files_by_extension(directory_path, extension):
    """
    Deletes all files with a specific extension within a given directory.

    Args:
        directory_path (str): The path to the directory.
        extension (str): The file extension to target (e.g., ".txt", ".log").
    """
    try:
        for filename in os.listdir(directory_path):
            if filename.endswith(extension):
                file_path = os.path.join(directory_path, filename)
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_files_matching_pattern(directory_path, pattern):
    """
    Returns a list of file paths in a given directory that match a specified pattern.

    Args:
        directory_path (str): The path to the directory to search.
        pattern (str): The pattern to match against filenames (e.g., '*.txt', 'image_*.png').

    Returns:
        list: A list of full file paths matching the pattern.
    """
    full_pattern = os.path.join(directory_path, pattern)
    matching_files = glob.glob(full_pattern)
    return matching_files


def end_of_sim_housekeeping(pkl_fnm):
    
    from init_funs_forecast import get_model_info
    PFM = get_model_info(pkl_fnm)
    yyyymmdd = PFM['yyyymmdd']
    ##  first delete previous netcdf files on website directory
    # this stuff is now done when making the web.nc file
    #web_dir = PFM['lv4_his_web_dir']
    #archive_web_dir = PFM['archive_web_dir']
    #delete_files_by_extension(web_dir, ".nc")
    ## copy web.nc file to website and Archive
    #shutil.copy(PFM['lv4_web_name_full'],web_dir)
    #shutil.copy(PFM['lv4_web_name_full'],archive_web_dir)    
    # Rename the web.nc file on website
    #original_name_in_dest = os.path.join(web_dir, os.path.basename(PFM['lv4_web_name_full']))
    #new_path_in_dest = os.path.join(web_dir, 'web_data_latest.nc')
    #shutil.move(original_name_in_dest, new_path_in_dest)

    # copy all plots to dataSIO
    pattern = "his*.png"
    for dir in [PFM['lv1_plot_dir'],PFM['lv2_plot_dir'],PFM['lv3_plot_dir'],PFM['lv4_plot_dir']]:
        files = get_files_matching_pattern(dir, pattern)
        for file in files:
            copy_into_archive(file, PFM['dataSIO_plot_dir'], label='plot')
    # now for dye plots
    files = get_files_matching_pattern(PFM['lv4_plot_dir'], "dye*png")
    for file in files:
        copy_into_archive(file, PFM['dataSIO_plot_dir'], label='dye plot')
    # now for history files
    pattern = "*.nc"
    list1 = [PFM['lv1_his_dir'],PFM['lv2_his_dir'],PFM['lv3_his_dir']]
    list2 = [PFM['lv1_archive_his_dir'],PFM['lv2_archive_his_dir'],PFM['lv3_archive_his_dir']]
    for dir1, dir2 in zip(list1,list2):
        files = get_files_matching_pattern(dir1, pattern)
        for file in files:
            copy_into_archive(file, dir2, label='history')
    # now for lv4 history file
    pattern = "LV4*.nc"
    files = get_files_matching_pattern(PFM['lv4_his_dir'], pattern)
    for file in files:
        copy_into_archive(file, PFM['lv4_archive_his_dir'], label='LV4 history')
    # copy the log files...
    pattern = "*.log"
    for dir in [PFM['lv1_run_dir'],PFM['lv2_run_dir'],PFM['lv3_run_dir'],PFM['lv4_run_dir']]:
        file0 = get_files_matching_pattern(PFM['lv4_his_dir'], pattern)
        shutil.copy(file0,PFM['log_archive_dir'])
        file2 = PFM['log_archive_dir'] + os.path.basename(file0)
        # modify file name as it doesn't have a date
        file = file2.replace('.log',yyyymmdd+'.log')
        shutil.copy(file2,file)

    ### Note this should work up to dealing with web plots in the .sh file.


def display_timing_info(pkl_fnm):
    from init_funs_forecast import get_model_info
    PFM = get_model_info(pkl_fnm)
    fn_timing = PFM['lv1_run_dir'] + '/LV1_timing_info.pkl'
    with open(fn_timing,'rb') as fout:
        T1 = pickle.load(fout)
    fn_timing = PFM['lv2_run_dir'] + '/LV2_timing_info.pkl'
    with open(fn_timing,'rb') as fout:
        T2 = pickle.load(fout)
    fn_timing = PFM['lv3_run_dir'] + '/LV3_timing_info.pkl'
    with open(fn_timing,'rb') as fout:
        T3 = pickle.load(fout)
    fn_timing = PFM['lv4_run_dir'] + '/LV4_timing_info.pkl'
    with open(fn_timing,'rb') as fout:
        T4 = pickle.load(fout)

    GI = PFM['gridinfo']
    SS = PFM['stretching']

    #print(T1['roms'])

    tp1 = np.sum(T1['process'][:])
    tp1 = np.round( tp1.total_seconds()/60,decimals=2 )
    tatm1 = np.sum(T1['atm'][:])
    tatm1 = np.round( tatm1.total_seconds()/60,decimals=2 )
    tatm2 = np.sum(T2['atm'][:])
    tatm2 = np.round( tatm2.total_seconds()/60,decimals=2 )
    tatm3 = np.sum(T3['atm'][:])
    tatm3 = np.round( tatm3.total_seconds()/60,decimals=2 )
    tatm4 = np.sum(T4['atm'][:])
    tatm4 = np.round( tatm4.total_seconds()/60,decimals=2 )
    
    to1 = np.sum(T1['ic'][:])
    to1 = np.round( to1.total_seconds()/60,decimals=2 )
    to2 = np.sum(T2['ic'][:])
    to2 = np.round( to2.total_seconds()/60,decimals=2 )
    to3 = np.sum(T3['ic'][:])
    to3 = np.round( to3.total_seconds()/60,decimals=2 )
    to4 = np.sum(T4['ic'][:])
    to4 = np.round( to4.total_seconds()/60,decimals=2 )
    
    tb1 = np.sum(T1['bc'][:])
    tb1 = np.round( tb1.total_seconds()/60,decimals=2 )
    tb2 = np.sum(T2['bc'][:])
    tb2 = np.round( tb2.total_seconds()/60,decimals=2 )
    tb3 = np.sum(T3['bc'][:])
    tb3 = np.round( tb3.total_seconds()/60,decimals=2 )
    tb4 = np.sum(T4['bc'][:])
    tb4 = np.round( tb4.total_seconds()/60,decimals=2 )

    tp1 = np.sum(T1['plotting'][:])
    tp1 = np.round( tp1.total_seconds()/60,decimals=2 )
    tp2 = np.sum(T2['plotting'][:])
    tp2 = np.round( tp2.total_seconds()/60,decimals=2 )
    tp3 = np.sum(T3['plotting'][:])
    tp3 = np.round( tp3.total_seconds()/60,decimals=2 )
    tp4 = np.sum(T4['plotting'][:])
    tp4 = np.round( tp4.total_seconds()/60,decimals=2 )

    tr1 = np.sum(T1['roms'][:])
    #print(tr1)
    #print(str(tr1))
    tr1 = np.round( tr1.total_seconds()/60,decimals=2 )
    #print(tr1)
    tr2 = np.sum(T2['roms'][:])
    tr2 = np.round( tr2.total_seconds()/60,decimals=2 )
    tr3 = np.sum(T3['roms'][:])
    tr3 = np.round( tr3.total_seconds()/60,decimals=2 )
    tr4 = np.sum(T4['roms'][:])
    tr4 = np.round( tr4.total_seconds()/60,decimals=2 )


    t_atm = np.round(T1['download_atm'][0].total_seconds()/60,decimals=2)
    t_ocn = np.round(T1['download_ocn'][0].total_seconds()/60,decimals=2)

    tot1 = t_atm + t_ocn + tp1 + tatm1 + to1 + tb1 + tr1 + tp1
    tot2 = tatm2 + to2 + tb2 + tr2 + tp2
    tot3 = tatm3 + to3 + tb3 + tr3 + tp3
    tot4 = tatm4 + to4 + tb4 + tr4 + tp4
    if "swan" in T4:
        #print( T4['swan'][0].total_seconds() )
        tsw = np.round( T4['swan'][0].total_seconds()/60,decimals=2 )
        tot4 = tot4 + tsw

    tot_tot = tot1+tot2+tot3+tot4

    tot1 = np.round(tot1,decimals=2)
    tot2 = np.round(tot2,decimals=2)
    tot3 = np.round(tot3,decimals=2)
    tot4 = np.round(tot4,decimals=2)
    tot_tot = np.round(tot_tot,decimals=2)

    ttt = PFM['tinfo']
    stp1 = int( PFM['forecast_days'] * 24 * 3600 / ttt['L1','dtsec'] )
    stp2 = int( PFM['forecast_days'] * 24 * 3600 / ttt['L2','dtsec'] )
    stp3 = int( PFM['forecast_days'] * 24 * 3600 / ttt['L3','dtsec'] )
    stp4 = int( PFM['forecast_days'] * 24 * 3600 / ttt['L4','dtsec'] )

    #PFM['lv1_use_restart'] = 0
    if PFM['lv1_use_restart'] == 1:
        ocn_ic = 'restart file'
    else:
        ocn_ic = PFM['ocn_model']

    print('\n')
    print(f'----------------------------------------------------------------------')
    print(f'{''}{'PFM general information' : ^70}{''}')
    print(f'----------------------------------------------------------------------')
    print(PFM['start_time'].strftime("%Y-%m-%d %H:%M"), '    :    simulation started [local time]')
    print(PFM['fetch_time'].strftime("%Y-%m-%d %H:%M"), '    :    forecast start time [UTC]')
    print(PFM['fore_end_time'].strftime("%Y-%m-%d %H:%M"), '    :    forecast end time [UTC]')
    print(PFM['forecast_days'], '                 :    forecast length [days]')
    print(f'{PFM['lv4_model'] : <21}{':' : <5}{'LV4 ocean model' : <30}')
    print(f'{PFM['ocn_model'] : <21}{':' : <5}{'ocean boundary conditions' : <30}')
    print(f'{T1['hycom_t0'][0:4] + '-' + T1['hycom_t0'][4:6] + '-' + T1['hycom_t0'][6::] + ' 12:00'   : <21}{':' : <5}{'hycom forecast start time [UTC]' : <30}')
#    print(f'{'20241022' + ' 12:00'   : <21}{':' : <5}{'hycom forecast time' : <30}')
    print(f'{ocn_ic : <21}{':' : <5}{'ocean initial condition' : <30}')
    print(f'{PFM['atm_model'] : <21}{':' : <5}{'atmospheric forcing model' : <30}')
    print(f'----------------------------------------------------------------------')
    print(f'             |             |             |             |             |')
    print(f'             |     LV1     |     LV2     |     LV3     |     LV4     |')
    print(f'             |             |             |             |             |')
    print(f'----------------------------------------------------------------------')
    print(f'{''}{'grid points' : ^70}{''}')
    print(f'{'nx =' : >8}{'|' : >6}{GI['L1','Lm'] : >8}{'|' : >6} {GI['L2','Lm'] : >7}{'|' : >6}{GI['L3','Lm'] : >8}{'|' : >6}{GI['L4','Lm'] : >8}{'|' : >6}')
    print(f'{'ny =' : >8}{'|' : >6}{GI['L1','Mm'] : >8}{'|' : >6} {GI['L2','Mm'] : >7}{'|' : >6}{GI['L3','Mm'] : >8}{'|' : >6}{GI['L4','Mm'] : >8}{'|' : >6}')
    print(f'{'nz =' : >8}{'|' : >6}{SS['L1','Nz'] : >8}{'|' : >6} {SS['L2','Nz'] : >7}{'|' : >6}{SS['L3','Nz'] : >8}{'|' : >6}{SS['L4','Nz'] : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    print(f'{''}{'time steps' : ^70}{''}')
    print(f'{'' : <7}{'|' : >7}{stp1 : >8}{'|' : >6} {stp2 : >7}{'|' : >6}{stp3 : >8}{'|' : >6}{stp4 : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    print(f'{''}{'parallelization' : ^70}{''}')
    print(f'{'nodes' : >11}{'|' : >3}{GI['L1','nnodes'] : >8}{'|' : >6} {GI['L2','nnodes'] : >7}{'|' : >6}{GI['L3','nnodes'] : >8}{'|' : >6}{GI['L4','nnodes'] : >8}{'|' : >6}')
    print(f'{'Ntilei' : >11}{'|' : >3}{GI['L1','ntilei'] : >8}{'|' : >6} {GI['L2','ntilei'] : >7}{'|' : >6}{GI['L3','ntilei'] : >8}{'|' : >6}{GI['L4','ntilei'] : >8}{'|' : >6}')
    print(f'{'Ntilej' : >11}{'|' : >3}{GI['L1','ntilej'] : >8}{'|' : >6} {GI['L2','ntilej'] : >7}{'|' : >6}{GI['L3','ntilej'] : >8}{'|' : >6}{GI['L4','ntilej'] : >8}{'|' : >6}')
    if "swan" in T4:
        print(f'{'  swan CPUs' : >11}{'|' : >3}{'' : >8}{'|' : >6} {'' : >7}{'|' : >6}{'' : >8}{'|' : >6}{GI['L4','np_swan'] : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    print(f'----------------------------------------------------------------------')
    print(f'{''}{'timing information for PFM' : ^70}{''}')
    print(f'{''}{'[in minutes]' : ^70}{''}')
    print(f'----------------------------------------------------------------------')
    print(f'----------------------------------------------------------------------')
    print(f'{'downloading' : >8}{'|' : >3}{'': >8}{'|' : >6} {'' : >7}{'|' : >6}{'' : >8}{'|' : >6}{'' : >8}{'|' : >6}')
    print(f'{'atm' : >7}{'|' : >7}{t_atm : >8}{'|' : >6} {'' : >7}{'|' : >6}{'' : >8}{'|' : >6}{'' : >8}{'|' : >6}')
    print(f'{'ocn' : >7}{'|' : >7}{t_ocn : >8}{'|' : >6} {'' : >7}{'|' : >6}{'' : >8}{'|' : >6}{'' : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    print(f'{'processing' : >7}{'|' : >4}{tp1 : >8}{'|' : >6} {'' : >7}{'|' : >6}{'' : >8}{'|' : >6}{'' : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    print(f'{'atm' : <7}{'|' : >7}{tatm1 : >8}{'|' : >6} {tatm2 : >7}{'|' : >6}{tatm3 : >8}{'|' : >6}{tatm4 : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    print(f'{'ocn ic' : <7}{'|' : >7}{to1 : >8}{'|' : >6} {to2 : >7}{'|' : >6}{to3 : >8}{'|' : >6}{to4 : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    print(f'{'ocn bc' : <7}{'|' : >7}{tb1 : >8}{'|' : >6} {tb2 : >7}{'|' : >6}{tb3 : >8}{'|' : >6}{tb4 : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    if "swan" in T4:
        print(f'{'swan files' : <7}{'|' : >4}{'' : >8}{'|' : >6} {'' : >7}{'|' : >6}{'' : >8}{'|' : >6}{tsw : >8}{'|' : >6}')
        print(f'----------------------------------------------------------------------')
    print(f'{'plotting' : <7}{'|' : >6}{tp1 : >8}{'|' : >6} {tp2 : >7}{'|' : >6}{tp3 : >8}{'|' : >6}{tp4 : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    print(f'{'ROMS' : <7}{'|' : >7}{tr1 : >8}{'|' : >6} {tr2 : >7}{'|' : >6}{tr3 : >8}{'|' : >6}{tr4 : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    print(f'{'totals' : <7}{'|' : >7}{tot1 : >8}{'|' : >6} {tot2 : >7}{'|' : >6}{tot3 : >8}{'|' : >6}{tot4 : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    print(f'{'TOTAL [min]' : <11}{'|' : >3}{tot_tot : ^53}{'|' : >3}')
    print(f'----------------------------------------------------------------------')
    print('\n')


#print(f'max {vnm:6} = {mxx:6.3f} {ulist2[vnm]:5}      at  ( it, ilat, ilon)     =  ({ind_mx[0]:3},{ind_mx[1]:4},{ind_mx[2]:4})')


def make_web_nc_file(pkl_fnm):
    MI = initfuns.get_model_info(pkl_fnm)
    fn_hs = MI['lv4_his_name_full']
    fn_gr = MI['lv4_grid_file']
    print('making the web_essential.nc file...')
    cmd_list = ['python','-W','ignore','web_functions.py','full_his_to_essential',fn_hs,fn_gr,pkl_fnm]
    os.chdir('../web_util')
    ret6 = subprocess.run(cmd_list)   
    if ret6.returncode != 0 : 
        print('something bad happened making web.nc file, aborting move and copy function!')
        sys.exit(1)

    os.chdir('../driver')

    if MI['testing_hook_web']:
        print('we are in testing mode (in util_functions). we will exit this function  now before moving web.nc to the archive.')
        return

    #fn_wb = MI['lv4_web_name_full'] 
    #print('moving and copying', fn_wb, ' file')
    web_dir = MI['lv4_for_web_dir']
    archive_web_dir = MI['archive_web_dir']
    #print('to ', web_dir, ' and ', archive_web_dir)
    #delete_files_by_extension(web_dir, ".nc")
    ## copy web.nc file to website and Archive

    if os.path.exists(MI['lv4_web_name_full']):
        # hook to use shell commands instead...
        use_bash = True
        if use_bash:
            web_full_orig = MI['lv4_web_name_full']
            base_name     = os.path.basename(web_full_orig)
            web_full_v1   = web_dir + 'web_data_latest.nc'
            web_full_v2   = archive_web_dir + base_name 
            print('going to copy ', web_full_orig, ' to')
            print(web_full_v1, ' and ', web_full_v2)
            ok1 = copy_into_archive(web_full_orig, web_full_v1, label='web latest')
            print('done copying to /for_web/.' if ok1 else
                  'FAILED copying to /for_web/ -- the website still has the old file')
            ok2 = copy_into_archive(web_full_orig, web_full_v2, label='web stamped')
            print('done copying to /web/' if ok2 else
                  'FAILED copying to /web/ -- the archive still has the old file')

        else:
            web_dir_2 = web_dir[0:-1]
            print(f"The file '{MI['lv4_web_name_full']}' exists.")
            print('copying web.nc to /dataSIO/... now...')
            #shutil.copy(MI['lv4_web_name_full'],web_dir)  # FF changed 7-july-26
            #shutil.copy(MI['lv4_web_name_full'],archive_web_dir)    # FF changed 7-july-26
            # Rename the web.nc file on website to web_data_latest.nc
            print('copied ', MI['lv4_web_name_full'], ' to /dataSIO/ locations. Renaming in for_web/')
            original_name_in_dest = os.path.join(web_dir, os.path.basename(MI['lv4_web_name_full']))
            new_path_in_dest = os.path.join(web_dir, 'web_data_latest.nc')
            print('renaming ', original_name_in_dest, ' to ')
            print(new_path_in_dest)
            #shutil.move(original_name_in_dest, new_path_in_dest)  #FF changed 7-july-26
            print('...done!')
    else:
        print(f"The file '{MI['lv4_web_name_full']}' does not exist. Aborting function!")
        sys.exit(1)
    
    return ret6

def make_simulation_plots(lvs_to_plt,pkl_fnm):
    procs = dict()
    MI = initfuns.get_model_info(pkl_fnm)
    for lvl in lvs_to_plt:
        if lvl == 'LV4dye': # make dye plots too
            print('doing LV4 dye plots and waiting (Popen)...')
            fn_gr = MI['lv4_grid_file']
            fn_hs = MI['lv4_his_name_full']
            cmd_list = ['python','-W','ignore','plotting_functions.py','make_dye_plots',fn_gr,fn_hs,pkl_fnm]
            os.chdir('../sdpm_py_util')
            procs[lvl] = subprocess.Popen(cmd_list)   
        else:
            print('making ', lvl, ' history file plots and moving on (Popen)...')
            cmd_list = ['python','-W','ignore','plotting_functions.py','make_all_his_figures',lvl,pkl_fnm]
            os.chdir('../sdpm_py_util')
            procs[lvl] = subprocess.Popen(cmd_list)   
    
    print('waiting for plotting to finish...')
    exit_codes = []
    for lvl in lvs_to_plt:
        exit_codes.append(procs[lvl].wait()) # this waits for procesess to finish
    print('...done waiting.')
    
    cnt = 0
    for lvl in lvs_to_plt:
        print(lvl, ' plots made correctly: ',exit_codes[cnt],' (0=yes)')
        cnt = cnt + 1

    os.chdir('../driver')
   
class s_coordinate(object):
    def __init__(self, h, theta_b, theta_s, Tcline, N, hraw=None, zeta=None):
        self.hraw = hraw
        self.h = np.asarray(h)
        self.hmin = self.h.min()
        self.theta_b = theta_b
        self.theta_s = theta_s
        self.Tcline = Tcline
        self.N = int(N)
        self.Np = self.N + 1
        self.hc = min(self.hmin, self.Tcline)
        self.Vtrans = 1

        if self.Tcline > self.hmin:
            warnings.warn(
                'Vertical transformation parameters are not defined correctly: \n Tcline = %d and hmin = %d. \n You need to make sure that Tcline <= hmin when using transformation 1.' % (self.Tcline, self.hmin))

        self.c1 = 1.0
        self.c2 = 2.0
        self.p5 = 0.5

        if zeta is None:
            self.zeta = np.zeros(h.shape)
        else:
            self.zeta = zeta

        self._get_s_rho()
        self._get_s_w()
        self._get_Cs_r()
        self._get_Cs_w()

        self.z_r = z_r(self.h, self.hc, self.N, self.s_rho, self.Cs_r, self.zeta, self.Vtrans)
        self.z_w = z_w(self.h, self.hc, self.Np, self.s_w, self.Cs_w, self.zeta, self.Vtrans)

    def _get_s_rho(self):
        lev = np.arange(1, self.N + 1, 1)
        ds = 1.0 / self.N
        self.s_rho = -self.c1 + (lev - self.p5) * ds

    def _get_s_w(self):
        lev = np.arange(0, self.Np, 1)
        ds = 1.0 / (self.Np - 1)
        self.s_w = -self.c1 + lev * ds

    def _get_Cs_r(self):
        if self.theta_s >= 0:
            Ptheta = np.sinh(self.theta_s * self.s_rho) / np.sinh(self.theta_s)
            Rtheta = np.tanh(self.theta_s * (self.s_rho + self.p5)) / \
                (self.c2 * np.tanh(self.p5 * self.theta_s)) - self.p5
            self.Cs_r = (self.c1 - self.theta_b) * Ptheta + self.theta_b * Rtheta
        else:
            self.Cs_r = self.s_rho

    def _get_Cs_w(self):
        if self.theta_s >= 0:
            Ptheta = np.sinh(self.theta_s * self.s_w) / np.sinh(self.theta_s)
            Rtheta = np.tanh(self.theta_s * (self.s_w + self.p5)) / \
                (self.c2 * np.tanh(self.p5 * self.theta_s)) - self.p5
            self.Cs_w = (self.c1 - self.theta_b) * Ptheta + self.theta_b * Rtheta
        else:
            self.Cs_w = self.s_w

class s_coordinate_4(s_coordinate):
    def __init__(self, h, theta_b, theta_s, Tcline, N, hraw=None, zeta=None):
        self.hraw = hraw
        self.h = np.asarray(h)
        self.hmin = h.min()
        self.theta_b = theta_b
        self.theta_s = theta_s
        self.Tcline = Tcline
        self.N = int(N)
        self.Np = self.N + 1
        self.hc = self.Tcline
        self.Vtrans = 4
        self.c1 = 1.0
        self.c2 = 2.0
        self.p5 = 0.5

        if zeta is None:
            self.zeta = np.zeros(h.shape)
        else:
            self.zeta = zeta

        self._get_s_rho()
        self._get_s_w()
        self._get_Cs_r()
        self._get_Cs_w()

        self.z_r = z_r(self.h, self.hc, self.N, self.s_rho, self.Cs_r, self.zeta, self.Vtrans)
        self.z_w = z_w(self.h, self.hc, self.Np, self.s_w, self.Cs_w, self.zeta, self.Vtrans)

    def _get_Cs_r(self):
        if self.theta_s > 0:
            Csur = (self.c1 - np.cosh(self.theta_s * self.s_rho)) / \
                (np.cosh(self.theta_s) - self.c1)
        else:
            Csur = -self.s_rho**2
        if self.theta_b > 0:
            Cbot = (np.exp(self.theta_b * Csur) - self.c1) / \
                (self.c1 - np.exp(-self.theta_b))
            self.Cs_r = Cbot
        else:
            self.Cs_r = Csur

    def _get_Cs_w(self):
        if self.theta_s > 0:
            Csur = (self.c1 - np.cosh(self.theta_s * self.s_w)) / \
                (np.cosh(self.theta_s) - self.c1)
        else:
            Csur = -self.s_w**2
        if self.theta_b > 0:
            Cbot = (np.exp(self.theta_b * Csur) - self.c1) / \
                (self.c1 - np.exp(-self.theta_b))
            self.Cs_w = Cbot
        else:
            self.Cs_w = Csur

class z_r(object):
    def __init__(self, h, hc, N, s_rho, Cs_r, zeta, Vtrans):
        self.h = h
        self.hc = hc
        self.N = N
        self.s_rho = s_rho
        self.Cs_r = Cs_r
        self.zeta = zeta
        self.Vtrans = Vtrans

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(self.zeta.shape) > len(self.h.shape):
            zeta = self.zeta[key[0]]
            res_index = (slice(None),) + key[1:]
        elif len(self.zeta.shape) > len(self.h.shape):
            zeta = self.zeta[key]
            res_index = slice(None)
        else:
            zeta = self.zeta
            res_index = key

        if self.h.ndim == zeta.ndim:
            zeta = zeta[np.newaxis, :]

        ti = zeta.shape[0]
        z_r = np.empty((ti, self.N) + self.h.shape, 'd')
        if self.Vtrans == 1:
            for n in range(ti):
                for k in range(self.N):
                    z0 = self.hc * self.s_rho[k] + (self.h - self.hc) * self.Cs_r[k]
                    z_r[n, k, :] = z0 + zeta[n, :] * (1.0 + z0 / self.h)
        elif self.Vtrans in [2, 4]:
            for n in range(ti):
                for k in range(self.N):
                    z0 = (self.hc * self.s_rho[k] + self.h * self.Cs_r[k]) / \
                        (self.hc + self.h)
                    z_r[n, k, :] = zeta[n, :] + (zeta[n, :] + self.h) * z0

        return np.squeeze(z_r[res_index])

class z_w(object):
    def __init__(self, h, hc, Np, s_w, Cs_w, zeta, Vtrans):
        self.h = h
        self.hc = hc
        self.Np = Np
        self.s_w = s_w
        self.Cs_w = Cs_w
        self.zeta = zeta
        self.Vtrans = Vtrans

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(self.zeta.shape) > len(self.h.shape):
            zeta = self.zeta[key[0]]
            res_index = (slice(None),) + key[1:]
        elif len(self.zeta.shape) > len(self.h.shape):
            zeta = self.zeta[key]
            res_index = slice(None)
        else:
            zeta = self.zeta
            res_index = key

        if self.h.ndim == zeta.ndim:
            zeta = zeta[np.newaxis, :]

        ti = zeta.shape[0]
        z_w = np.empty((ti, self.Np) + self.h.shape, 'd')
        if self.Vtrans == 1:
            for n in range(ti):
                for k in range(self.Np):
                    z0 = self.hc * self.s_w[k] + (self.h - self.hc) * self.Cs_w[k]
                    z_w[n, k, :] = z0 + zeta[n, :] * (1.0 + z0 / self.h)
        elif self.Vtrans in [2, 4]:
            for n in range(ti):
                for k in range(self.Np):
                    z0 = (self.hc * self.s_w[k] + self.h * self.Cs_w[k]) / \
                        (self.hc + self.h)
                    z_w[n, k, :] = zeta[n, :] + (zeta[n, :] + self.h) * z0

        return np.squeeze(z_w[res_index])
    
if __name__ == "__main__":
    args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
    globals()[args[1]](*args[2:])

