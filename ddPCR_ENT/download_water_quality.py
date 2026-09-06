"""
Download San Diego County Water Quality Data (South of Pt. Loma).

This module downloads recent coastal water quality monitoring data (specifically
ddPCR and culture-based Enterococcus / Fecal Indicator Bacteria) for locations
south of Point Loma (Coronado, Silver Strand, Imperial Beach, Tijuana River /
Border Field State Park) and saves the filtered data to a CSV file.

Data Sources:
1. Primary: County of San Diego DEHQ Official Portal (https://cosdapps.sandiegocounty.gov/sdbeachinfo/)
   - Provides live, same-day ddPCR test results as soon as they are processed.
2. Fallback: California State Water Resources Control Board BeachWatch database (https://beachwatch.waterboards.ca.gov/)
3. Fallback: Daily mirrored dataset from sdwaterwatch.com
"""

import sys
import io
import csv
import json
import re
from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict, Any
import urllib.request
import urllib.parse
import urllib.error

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Official County of San Diego Portal URLs & Defaults
COSD_BASE_URL = "https://cosdapps.sandiegocounty.gov/sdbeachinfo"
COSD_SAMPLES_ENDPOINT = f"{COSD_BASE_URL}/screenservices/CoSD_Beach_Water_CW/MainFlow/SampleBlock/ScreenDataSetGetSamples"
COSD_VER_INFO_URL = f"{COSD_BASE_URL}/moduleservices/moduleversioninfo"
COSD_SAMPLE_BLOCK_JS = f"{COSD_BASE_URL}/scripts/CoSD_Beach_Water_CW.MainFlow.SampleBlock.mvc.js"

DEFAULT_MODULE_VERSION = "Y3IareGcvkhpc7CILZGOzA"
DEFAULT_API_VERSION = "L1afk6NDVnwatxTYNUtNLA"

# State BeachWatch URLs
BEACHWATCH_URL = "https://beachwatch.waterboards.ca.gov/public/result.php"
BEACHWATCH_EXPORT_URL = "https://beachwatch.waterboards.ca.gov/public/export.php"
SDWATERWATCH_FALLBACK_URL = "https://sdwaterwatch.com/AllResultData.xls"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Known station IDs south of Point Loma (Coronado, Silver Strand, Imperial Beach, Tijuana River, Border)
SOUTH_STATION_IDS = {
    "EH-010", "EH-030", "EH-040", "EH-050", "EH-060",
    "EH-070", "EH-080", "EH-090", "EH-120",
    "IB-010", "IB-020", "IB-030", "IB-040", "IB-050",
    "IB-060", "IB-068", "IB-070", "IB-079"
}

# Known beach names south of Point Loma
SOUTH_BEACH_NAMES = [
    "Coronado City beaches",
    "Coronado north beach",
    "Silver Strand State Beach",
    "north Imperial Beach",
    "Imperial Beach pier area",
    "Imperial Beach municipal beach, other",
    "Imperial Beach Municipal Beach",
    "Tijuana Slough National Wildlife Refuge",
    "Border Field State Park",
    "Baja California, MEXICO"
]

# Map station IDs to standardized Beach Names
STATION_BEACH_MAP = {
    "EH-060": "Coronado City Beaches",
    "EH-050": "Coronado City Beaches",
    "IB-079": "Coronado City Beaches",
    "EH-040": "Coronado City Beaches",
    "EH-070": "San Diego Bay",
    "EH-080": "San Diego Bay",
    "EH-090": "Silver Strand State Beach",
    "EH-120": "San Diego Bay",
    "IB-070": "Silver Strand State Beach",
    "IB-068": "Silver Strand State Beach",
    "IB-060": "Imperial Beach Municipal Beach",
    "EH-030": "Imperial Beach Municipal Beach",
    "EH-010": "Imperial Beach Municipal Beach",
    "IB-050": "Imperial Beach Municipal Beach",
    "IB-040": "Tijuana Slough National Wildlife Refuge",
    "IB-030": "Tijuana Slough National Wildlife Refuge",
    "IB-020": "Border Field State Park",
    "IB-010": "Border Field State Park",
}

# Bay stations located in South San Diego Bay
SOUTH_BAY_NAMES = [
    "San Diego Bay"
]


def _get_cosd_tokens() -> tuple[str, str]:
    """
    Dynamically discover the current OutSystems versionToken and apiVersion
    from the live application, falling back to known working defaults.
    """
    mod_ver = DEFAULT_MODULE_VERSION
    api_ver = DEFAULT_API_VERSION

    try:
        # 1. Discover module versionToken
        req = urllib.request.Request(COSD_VER_INFO_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("versionToken"):
                mod_ver = data["versionToken"]
    except Exception:
        pass

    try:
        # 2. Discover apiVersion token from SampleBlock script
        req = urllib.request.Request(COSD_SAMPLE_BLOCK_JS, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=5) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
            match = re.search(r'ScreenDataSetGetSamples[\"\'\s,]+screenservices[^\"]+[\"\'\s,]+([A-Za-z0-9_\+\/=]+)', text)
            if match:
                api_ver = match.group(1)
    except Exception:
        pass

    return mod_ver, api_ver


def _fetch_cosd_sdbeachinfo(max_records: int = 2000) -> List[Dict[str, Any]]:
    """
    Fetch the latest live water quality sample records directly from
    the County of San Diego DEHQ portal (sdbeachinfo).
    Returns normalized dictionary rows.
    """
    mod_ver, api_ver = _get_cosd_tokens()

    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "Accept": "application/json",
        "X-CSRFToken": "T6C+9iB49TLra4jEsMeSckDMNhQ=",
        "User-Agent": USER_AGENT
    }

    payload = {
        "versionInfo": {
            "moduleVersion": mod_ver,
            "apiVersion": api_ver
        },
        "viewName": "MainFlow.SamplesReport",
        "screenData": {
            "variables": {
                "SiteId": 121,
                "isPublic": True,
                "TableSort": "Sample.SampleDate DESC",
                "MaxRecords": max_records,
                "StartIndex": 0
            }
        },
        "inputParameters": {
            "StartIndex": 0,
            "MaxRecords": max_records
        }
    }

    req = urllib.request.Request(
        COSD_SAMPLES_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers
    )

    with urllib.request.urlopen(req, timeout=15) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))

    sample_list = res_data.get("data", {}).get("List", {}).get("List", [])
    rows = []

    for item in sample_list:
        site = item.get("Site", {})
        sample = item.get("Sample", {})
        param = item.get("Parameter", {})
        method = item.get("AnalysisMethod", {})
        unit = item.get("Units", {})

        st_id = str(site.get("StationID", "")).strip()
        loc_name = str(site.get("LocationName", "")).strip()
        beach_name = str(site.get("BeachName", "")).strip()
        if not beach_name and st_id in STATION_BEACH_MAP:
            beach_name = STATION_BEACH_MAP[st_id]
        sample_date = str(sample.get("SampleDate", "")).split()[0] if sample.get("SampleDate") else ""
        sample_time = str(sample.get("SampleTime", "")).strip()
        result_raw = sample.get("Result")
        result_str = str(result_raw) if result_raw is not None else ""

        try:
            lat = float(site.get("Latitude", 0) or 0)
        except (ValueError, TypeError):
            lat = 0.0

        try:
            lon = float(site.get("Longitude", 0) or 0)
        except (ValueError, TypeError):
            lon = 0.0

        rows.append({
            "id": str(sample.get("Id", "")),
            "Station_ID": st_id,
            "Station Name": st_id,
            "SampleDate": sample_date,
            "SampleTime": sample_time,
            "parameter": str(param.get("Label", "")).strip(),
            "qualifier": str(sample.get("Qualifier", "=")).strip(),
            "Result": result_str,
            "unit": str(unit.get("Label", "")).strip(),
            "method": str(method.get("Label", "")).strip(),
            "type": "",
            "County": str(site.get("County", "San Diego")).strip(),
            "Description": loc_name,
            "Status": "",
            "Beach Name": beach_name,
            "Latitude": f"{lat:.6f}" if lat else "",
            "Longitude": f"{lon:.6f}" if lon else "",
            "CreateDate": sample_date
        })

    return rows


def _fetch_beachwatch_raw(year: int = None) -> str:
    """
    Fetch tab-delimited water quality data from the California
    State Water Resources Control Board BeachWatch database,
    with fallback to the daily mirror dataset.
    """
    if year is None:
        year = datetime.now().year

    # Try official BeachWatch session-based query first
    try:
        import http.cookiejar
        cookie_jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

        headers = {
            "User-Agent": USER_AGENT,
            "Referer": BEACHWATCH_URL
        }

        # 1. Initialize session
        init_req = urllib.request.Request(BEACHWATCH_URL, headers=headers)
        with opener.open(init_req, timeout=15) as _:
            pass

        # 2. Submit search form for San Diego County (County=10)
        form_data = urllib.parse.urlencode({
            "County": "10",       # San Diego County
            "stationID": "",      # All stations
            "parameter": "",      # All parameters
            "qualifier": "",
            "method": "",
            "created": "",
            "year": str(year),
            "sort": "`SampleDate`",
            "sortOrder": "DESC",
            "submit": "Search"
        }).encode("utf-8")

        search_req = urllib.request.Request(BEACHWATCH_URL, data=form_data, headers=headers)
        with opener.open(search_req, timeout=30) as _:
            pass

        # 3. Download exported TSV file
        export_req = urllib.request.Request(BEACHWATCH_EXPORT_URL, headers=headers)
        with opener.open(export_req, timeout=30) as resp:
            content = resp.read().decode("utf-8", errors="ignore")
            if "Station Name" in content and len(content) > 1000:
                return content
    except Exception as exc:
        print(f"[Warning] Direct query to BeachWatch failed ({exc}). Trying daily mirror fallback...", file=sys.stderr)

    # Fallback to daily mirror
    try:
        req = urllib.request.Request(SDWATERWATCH_FALLBACK_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8", errors="ignore")
            if "Station Name" in content:
                return content
    except Exception as exc:
        raise RuntimeError(f"Failed to download water quality data from all sources: {exc}")

    raise RuntimeError("Downloaded content did not contain valid water quality monitoring data.")


def download_sd_water_quality(
    output_csv: str = "sd_water_quality_south_pt_loma.csv",
    days: int = 14,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    max_latitude: float = 32.695,
    strictly_south_of_tip: bool = False,
    include_bay: bool = True,
    parameter: Optional[str] = None,
    ddpcr_only: bool = False,
    force_beachwatch: bool = False
):
    """
    Downloads water quality data for San Diego County locations south of Pt. Loma,
    filters to the specified time window (default: last 2 weeks), and writes to CSV.

    By default, pulls directly from the County of San Diego DEHQ portal
    (https://cosdapps.sandiegocounty.gov/sdbeachinfo/) for the freshest same-day
    monitoring results (e.g. recent daily ddPCR tests), falling back to the State
    BeachWatch portal if unavailable.

    Parameters:
    -----------
    output_csv : str
        Path to save the output CSV file (default: 'sd_water_quality_south_pt_loma.csv').
    days : int
        Number of past days of data to retain (default: 14 for the last 2 weeks).
        Ignored if `start_date` is explicitly specified.
    start_date : str or datetime, optional
        Start date filter in 'YYYY-MM-DD' format. If None, defaults to `end_date - days`.
    end_date : str or datetime, optional
        End date filter in 'YYYY-MM-DD' format. If None, defaults to today.
    max_latitude : float
        Maximum latitude cutoff (default: 32.695, covers Coronado down to Mexican border).
    strictly_south_of_tip : bool
        If True, restricts strictly south of Point Loma lighthouse tip (< 32.6654 N),
        which excludes Coronado beach and includes Silver Strand down to the border.
        If False (default), includes Coronado, Silver Strand, Imperial Beach, and Border.
    include_bay : bool
        If True (default), includes South San Diego Bay stations (e.g. Glorietta Bay,
        Tidelands Park, Bayside Park J St). If False, ocean-facing beaches only.
    parameter : str, optional
        Filter by water quality parameter (e.g., 'Enterococcus', 'Fecal Coliforms',
        'Total Coliforms'). If None, all parameters are included.
    ddpcr_only : bool
        If True, only retains ddPCR analysis method results ('MCB-ddPCR SOP018-000' or 'ddPCR').
    force_beachwatch : bool
        If True, bypasses the County DEHQ portal and fetches only from State BeachWatch.

    Returns:
    --------
    pandas.DataFrame (if pandas is installed) or List[dict] (if pandas is not installed)
    """
    now = datetime.now()

    # Determine date bounds
    if end_date is None:
        end_dt = now.date()
    elif isinstance(end_date, str):
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    elif isinstance(end_date, datetime):
        end_dt = end_date.date()
    else:
        end_dt = end_date

    if start_date is None:
        start_dt = end_dt - timedelta(days=days)
    elif isinstance(start_date, str):
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    elif isinstance(start_date, datetime):
        start_dt = start_date.date()
    else:
        start_dt = start_date

    all_raw_rows: List[Dict[str, Any]] = []

    # 1. Try fetching from County of San Diego DEHQ (sdbeachinfo) first for latest data
    if not force_beachwatch:
        try:
            print(f"Connecting to County of San Diego DEHQ portal (cosdapps.sandiegocounty.gov/sdbeachinfo)...")
            cosd_rows = _fetch_cosd_sdbeachinfo(max_records=2000)
            if cosd_rows:
                print(f"Successfully retrieved {len(cosd_rows)} records from County DEHQ portal.")
                all_raw_rows.extend(cosd_rows)
        except Exception as exc:
            print(f"[Warning] Failed to fetch from County DEHQ portal: {exc}. Falling back to BeachWatch...", file=sys.stderr)

    # 2. If County DEHQ yielded no records, or if user specifically requested BeachWatch
    if not all_raw_rows:
        print(f"Fetching from California State BeachWatch database ({start_dt} to {end_dt})...")
        raw_tsv = _fetch_beachwatch_raw(year=end_dt.year)
        if start_dt.year < end_dt.year:
            try:
                prev_tsv = _fetch_beachwatch_raw(year=start_dt.year)
                lines1 = raw_tsv.splitlines(True)
                lines2 = [l for l in prev_tsv.splitlines(True) if not l.startswith("id\t")]
                raw_tsv = "".join(lines1 + lines2)
            except Exception as e:
                print(f"[Warning] Could not fetch data for previous year {start_dt.year}: {e}", file=sys.stderr)

        reader = csv.DictReader(io.StringIO(raw_tsv), delimiter="\t")
        all_raw_rows = list(reader)

    # 3. Filter rows for date, geography, parameter, method
    lat_cutoff = 32.6654 if strictly_south_of_tip else max_latitude
    filtered_rows = []
    seen_keys = set()

    for row in all_raw_rows:
        # Date filter
        sample_date_str = row.get("SampleDate", "").strip()
        if not sample_date_str:
            continue
        try:
            row_date = datetime.strptime(sample_date_str, "%Y-%m-%d").date()
        except ValueError:
            continue

        if not (start_dt <= row_date <= end_dt):
            continue

        # Location filter (south of Pt. Loma)
        lat_str = row.get("Latitude", "").strip()
        lon_str = row.get("Longitude", "").strip()
        beach_name = row.get("Beach Name", "").strip()
        st_name = row.get("Station Name", "").strip().upper()
        st_id = row.get("Station_ID", "").strip().upper()

        try:
            lat = float(lat_str)
            lon = float(lon_str)
        except ValueError:
            lat = None
            lon = None

        is_south = False
        # Direct station ID match
        if st_id in SOUTH_STATION_IDS or st_name in SOUTH_STATION_IDS:
            is_south = True
        # Known beach name
        elif any(b.lower() in beach_name.lower() for b in SOUTH_BEACH_NAMES):
            is_south = True
        elif include_bay and any(b.lower() in beach_name.lower() for b in SOUTH_BAY_NAMES) and lat is not None and lat <= max_latitude:
            is_south = True
        elif lat is not None and lon is not None:
            # Geographic bounding: lat <= lat_cutoff and east of Point Loma peninsula (lon > -117.23)
            if lat <= lat_cutoff and lon > -117.23 and not beach_name.lower().startswith("non-accessible"):
                if include_bay or not any(b.lower() in beach_name.lower() for b in SOUTH_BAY_NAMES):
                    is_south = True

        if not is_south:
            continue

        # Optional parameter filter
        if parameter and parameter.strip().lower() not in row.get("parameter", "").strip().lower():
            continue

        # Optional ddPCR filter
        if ddpcr_only:
            method_str = row.get("method", "").lower()
            if "ddpcr" not in method_str:
                continue

        # Deduplicate
        key = (st_id, sample_date_str, row.get("SampleTime", ""), row.get("parameter", ""))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        filtered_rows.append(row)

    # Sort rows chronologically descending by SampleDate, SampleTime, then station
    filtered_rows.sort(
        key=lambda r: (
            r.get("SampleDate", ""),
            r.get("SampleTime", ""),
            r.get("Station Name", "")
        ),
        reverse=True
    )

    print(f"Found {len(filtered_rows)} records matching criteria south of Pt. Loma.")

    # Write to CSV
    if filtered_rows:
        fieldnames = list(filtered_rows[0].keys())
    else:
        fieldnames = [
            "id", "Station_ID", "Station Name", "SampleDate", "SampleTime",
            "parameter", "qualifier", "Result", "unit", "method", "type",
            "County", "Description", "Status", "Beach Name", "Latitude",
            "Longitude", "CreateDate"
        ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"Successfully saved data to '{output_csv}'.")

    if HAS_PANDAS:
        df = pd.DataFrame(filtered_rows)
        if not df.empty and "SampleDate" in df.columns:
            df["SampleDate"] = pd.to_datetime(df["SampleDate"])
            if "Latitude" in df.columns:
                df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
            if "Longitude" in df.columns:
                df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
            if "Result" in df.columns:
                df["Result_Numeric"] = pd.to_numeric(df["Result"], errors="coerce")
        return df

    return filtered_rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download last 2 weeks of SD County water quality data south of Pt. Loma."
    )
    parser.add_argument(
        "--output", "-o",
        default="sd_water_quality_south_pt_loma.csv",
        help="Output CSV filepath (default: sd_water_quality_south_pt_loma.csv)"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=14,
        help="Number of past days of data to download (default: 14)"
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date YYYY-MM-DD (overrides --days)"
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--parameter", "-p",
        default=None,
        help="Filter by parameter, e.g. Enterococcus"
    )
    parser.add_argument(
        "--ddpcr-only",
        action="store_true",
        help="Only include ddPCR test results"
    )
    parser.add_argument(
        "--strictly-south",
        action="store_true",
        help="Strictly south of Point Loma lighthouse tip (< 32.6654 N, excludes Coronado)"
    )
    parser.add_argument(
        "--no-bay",
        action="store_true",
        help="Exclude San Diego Bay stations (ocean beaches only)"
    )
    parser.add_argument(
        "--force-beachwatch",
        action="store_true",
        help="Force download from State BeachWatch instead of County DEHQ portal"
    )

    args = parser.parse_args()

    df_or_rows = download_sd_water_quality(
        output_csv=args.output,
        days=args.days,
        start_date=args.start_date,
        end_date=args.end_date,
        strictly_south_of_tip=args.strictly_south,
        include_bay=not args.no_bay,
        parameter=args.parameter,
        ddpcr_only=args.ddpcr_only,
        force_beachwatch=args.force_beachwatch
    )

    if HAS_PANDAS and isinstance(df_or_rows, pd.DataFrame):
        print("\nSummary of downloaded data:")
        if not df_or_rows.empty:
            print(f"Date range: {df_or_rows['SampleDate'].min().strftime('%Y-%m-%d')} to {df_or_rows['SampleDate'].max().strftime('%Y-%m-%d')}")
            print(f"Stations ({df_or_rows['Station Name'].nunique()}): {', '.join(sorted(df_or_rows['Station Name'].unique()))}")
            print(f"Methods: {', '.join(df_or_rows['method'].unique())}")
            print("\nFirst 10 rows:")
            print(df_or_rows[["Station Name", "Description", "SampleDate", "parameter", "qualifier", "Result", "unit", "method"]].head(10))
        else:
            print("No records matched the criteria.")
