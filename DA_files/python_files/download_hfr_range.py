#!/usr/bin/env python3
"""
download_hfr_range.py

Download USWC 6 km HF-radar hourly totals (RTV = real-time vector) from the
NCEI THREDDS archive.

The file naming embeds a creation timestamp we CAN'T predict:
    rtv-uswc-6km-uwls_v1r0_hfr_s{YYYYMMDDHHMMSS0}_e{YYYYMMDDHHMMSS0}_c{...}.nc

so we discover the exact names by fetching the monthly THREDDS catalog.xml
once per month, then build a lookup keyed by start-time stamp.

Catalog:
    https://www.ncei.noaa.gov/thredds-ocean/catalog/ioos/hfradar/rtv/YYYY/YYYYMM/USWC/catalog.xml

Download (fileServer):
    https://www.ncei.noaa.gov/thredds-ocean/fileServer/ioos/hfradar/rtv/YYYY/YYYYMM/USWC/<filename>

Cache: /dataSIO/DA_Simulations/input_files/hfr_cache/  (~130 KB per hourly file)

CLI:
    python download_hfr_range.py 2026-03-07 2026-03-08 --workers 8
"""
from __future__ import annotations
import argparse
import datetime as DT
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_CACHE_DIR = Path("/dataSIO/DA_Simulations/input_files/hfr_cache")
DEFAULT_RES = "6km"     # options: 500m, 1km, 2km, 6km

# NCEI's THREDDS front end returns 502/503 under load for minutes at a time.
# Be patient rather than reporting the data as absent.
UA = "Mozilla/5.0 (research; mspydell@ucsd.edu)"
CATALOG_TRIES   = 12    # a month's catalog is worth waiting for
CATALOG_BACKOFF = 15    # seconds, multiplied by attempt number
FILE_TRIES      = 4
FILE_BACKOFF    = 5

# Monthly catalogs are ~1.4 MB each and identical across resolutions, so a
# 3-resolution pull would fetch each one three times.  Past months never
# change; cache them on disk and stay off NCEI's back.
CATALOG_CACHE_DIR = DEFAULT_CACHE_DIR / ".catalog_cache"

BASE = "https://www.ncei.noaa.gov/thredds-ocean"
CAT_TMPL  = f"{BASE}/catalog/ioos/hfradar/rtv/{{yyyy}}/{{yyyymm}}/USWC/catalog.xml"
FILE_TMPL = f"{BASE}/fileServer/ioos/hfradar/rtv/{{yyyy}}/{{yyyymm}}/USWC/{{fn}}"


def _dataset_re(res: str) -> re.Pattern:
    return re.compile(r'dataset name="(rtv-uswc-' + re.escape(res) + r'-uwls[^"]+)"')


def _fetch_url(url: str, timeout: int = 60, tries: int = 1,
               backoff: float = 0.0) -> bytes:
    """Fetch url, retrying on any error with linear backoff.
    Raises RuntimeError if every attempt fails -- never returns partial data."""
    last: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            if attempt < tries:
                wait = backoff * attempt
                print(f"    attempt {attempt}/{tries} failed "
                      f"({type(e).__name__}: {e}); retrying in {wait:.0f}s")
                time.sleep(wait)
    raise RuntimeError(f"all {tries} attempt(s) failed for {url}: {last}")


def month_catalog(yyyymm: str, res: str) -> dict[str, str]:
    """Return {start_stamp: filename} for all files of the given resolution
    in YYYYMM's USWC catalog.  Start-stamp is 'YYYYMMDDHHMMSS0'."""
    yyyy = yyyymm[:4]
    url = CAT_TMPL.format(yyyy=yyyy, yyyymm=yyyymm)
    cached = CATALOG_CACHE_DIR / f"{yyyymm}.xml"
    if cached.exists() and cached.stat().st_size > 1024:
        print(f"    using cached catalog {cached}")
        text = cached.read_text(errors="replace")
    else:
        # Deliberately NOT caught: a failed catalog fetch used to return {},
        # which made the run report every hour as "missing from catalog" --
        # i.e. a server outage was indistinguishable from the data not existing.
        text = _fetch_url(url, timeout=300, tries=CATALOG_TRIES,
                          backoff=CATALOG_BACKOFF).decode("utf-8", errors="replace")
        CATALOG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached.write_text(text)
    out: dict[str, str] = {}
    for m in _dataset_re(res).finditer(text):
        fn = m.group(1)
        ms = re.search(r"_s(\d{15})_", fn)
        if ms:
            out[ms.group(1)] = fn
    return out


def stamp_for(t: DT.datetime) -> str:
    """Return the 15-char start-time stamp used by NCEI filenames."""
    return t.strftime("%Y%m%d%H%M%S") + "0"


def _fetch_one(t: DT.datetime, fn: str, out_dir: Path) -> tuple[str, bool, str]:
    yyyy   = t.strftime("%Y")
    yyyymm = t.strftime("%Y%m")
    dst    = out_dir / fn
    if dst.exists() and dst.stat().st_size > 1024:
        return (fn, True, "cached")
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = FILE_TMPL.format(yyyy=yyyy, yyyymm=yyyymm, fn=fn)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    last = ""
    for attempt in range(1, FILE_TRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.rename(dst)
            return (fn, True, "downloaded")
        except (urllib.error.URLError, IOError) as e:
            tmp.unlink(missing_ok=True)
            last = f"{type(e).__name__}: {e}"
            if attempt < FILE_TRIES:
                time.sleep(FILE_BACKOFF * attempt)
    return (fn, False, last)


def iter_hours(start: DT.datetime, end: DT.datetime):
    t = start
    while t <= end:
        yield t
        t += DT.timedelta(hours=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", help="YYYY-MM-DD (UTC)")
    ap.add_argument("end",   help="YYYY-MM-DD (UTC)")
    ap.add_argument("--res", default=DEFAULT_RES,
                    choices=("500m", "1km", "2km", "6km"),
                    help=f"grid resolution (default: {DEFAULT_RES})")
    ap.add_argument("--outdir", default=str(DEFAULT_CACHE_DIR),
                    help=f"output directory (default: {DEFAULT_CACHE_DIR})")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel downloads (NCEI is happy with 8)")
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    start = DT.datetime.strptime(args.start, "%Y-%m-%d")
    end   = DT.datetime.strptime(args.end,   "%Y-%m-%d") + DT.timedelta(hours=24)
    hours = list(iter_hours(start, end))
    print(f"queueing {len(hours)} hourly HFR files (res={args.res}, {args.workers} workers)")
    print(f"outdir: {out_dir}")

    # Fetch one catalog per unique month
    months = sorted({h.strftime("%Y%m") for h in hours})
    catalog: dict[str, str] = {}
    for yyyymm in months:
        print(f"  fetching catalog for {yyyymm} ...")
        catalog.update(month_catalog(yyyymm, args.res))
    print(f"  catalog: {len(catalog)} files known across {len(months)} month(s)")

    # Resolve each target hour to a filename
    tasks = []
    missing = []
    for t in hours:
        s = stamp_for(t)
        fn = catalog.get(s)
        if fn is None:
            missing.append(t)
        else:
            tasks.append((t, fn))
    if missing:
        print(f"  NOTE: {len(missing)} hours have no {args.res} file in the catalog "
              f"(first: {missing[0]}, last: {missing[-1]})")

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_fetch_one, t, fn, out_dir) for t, fn in tasks]
        for f in as_completed(futs):
            name, good, msg = f.result()
            if good:
                ok += 1
                if msg == "downloaded":
                    print(f"  ok  {name}")
            else:
                fail += 1
                print(f"  FAIL {name}: {msg}")
    print(f"done. ok={ok} fail={fail} missing_from_catalog={len(missing)}")


if __name__ == "__main__":
    main()
