"""
San Diego County Coastal Water Quality Plotting and Web Export Script.

This script:
1. Calls download_sd_water_quality() to fetch the latest monitoring data.
2. Filters for Enterococcus ddPCR and excludes inner bay stations (EH-090, EH-120, EH-080).
3. Groups the 3 Coronado ocean locations (EH-060, EH-050, IB-079) together.
4. Generates and saves 3 publication-ready PNG figures:
   - figure1_timeseries_grid.png: 3x3 grid of individual station time series.
   - figure2_coastal_zones.png: Grouped comparative time series by coastal zone.
   - figure3_alongshore_hovmoller.png: Alongshore Space-Time (Hovmöller) diagram.
5. Exports the summary metrics in web-ready formats:
   - summary_table.html: Styled HTML table fragment for embedding into websites.
   - summary_data.json: JSON dataset for web applications.
   - water_quality_report.html: Complete standalone HTML dashboard with embedded figures.
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless script execution
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from download_water_quality import download_sd_water_quality

# California ddPCR Single-Sample Health Advisory Standard
DDPCR_LIMIT = 1413  # copies / 100 mL

# Stations to exclude from coastal beach analysis
EXCLUDED_STATIONS = {'EH-090', 'EH-120', 'EH-080'}

# Planned station order: Coronado ocean group, Silver Strand/Tidelands, Imperial Beach
ORDERED_STATIONS = [
    'EH-060',  # Coronado North Beach
    'EH-050',  # Coronado Main Lifeguard Tower
    'IB-079',  # Coronado Avenida Lunar
    'IB-068',  # Silver Strand Guard Shack
    'EH-070',  # Tidelands Park
    'IB-060',  # Carnation Ave (IB)
    'EH-030',  # Imperial Beach Pier
    'EH-010',  # Cortez Ave (IB)
    'IB-050',  # End of Seacoast Dr (IB)
]


def setup_plot_style():
    """Configure matplotlib defaults for clean scientific presentation."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#d0d7de'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['figure.autolayout'] = False


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Parse timestamps, filter for Enterococcus, and sort stations."""
    def parse_datetime(row):
        time_str = str(row['SampleTime']).strip() if pd.notna(row['SampleTime']) else '12:00:00'
        if len(time_str) == 5:
            time_str += ':00'
        date_str = str(row['SampleDate']).split()[0]
        try:
            return pd.to_datetime(f"{date_str} {time_str}")
        except Exception:
            return pd.to_datetime(date_str)

    df = df.copy()
    df['Timestamp'] = df.apply(parse_datetime, axis=1)
    df['Result_Numeric'] = pd.to_numeric(df['Result'], errors='coerce')
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    # Filter for Enterococcus ddPCR
    ent_df = df[df['parameter'].str.contains('Enterococcus', case=False, na=False)].copy()

    # Exclude inner bay stations
    ent_df = ent_df[~ent_df['Station Name'].isin(EXCLUDED_STATIONS)].copy()

    # Filter station order to present stations
    station_order = [s for s in ORDERED_STATIONS if s in ent_df['Station Name'].values]

    return ent_df, station_order


def plot_timeseries_grid(
    ent_df: pd.DataFrame,
    station_order: list[str],
    output_path: str,
    gen_time: Optional[str] = None
):
    """
    Generate and save Figure 1: 3x3 multi-panel individual time series.
    The 3 Coronado ocean locations occupy Row 1.
    """
    if gen_time is None:
        gen_time = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    n_stations = len(station_order)
    cols = 3
    rows = int(np.ceil(n_stations / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.4 * rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, st in enumerate(station_order):
        ax = axes[i]
        st_data = ent_df[ent_df['Station Name'] == st].sort_values('Timestamp')
        desc = st_data['Description'].iloc[0] if len(st_data) > 0 else st
        beach = st_data['Beach Name'].iloc[0] if len(st_data) > 0 else ''

        # Line colors by coastal group
        if st in {'EH-060', 'EH-050', 'IB-079'}:
            line_color = '#1f77b4'  # Coronado Blue
        elif st in {'IB-068', 'EH-070'}:
            line_color = '#2ca02c'  # Silver Strand / Bay Green
        else:
            line_color = '#ff7f0e'  # Imperial Beach Orange

        # Time series
        ax.plot(
            st_data['Timestamp'], st_data['Result_Numeric'],
            marker='o', color=line_color, linewidth=1.8, markersize=5.5,
            label='Enterococcus ddPCR'
        )

        # Exceedance points
        exceed = st_data[st_data['Result_Numeric'] >= DDPCR_LIMIT]
        if len(exceed) > 0:
            ax.scatter(
                exceed['Timestamp'], exceed['Result_Numeric'],
                color='#d62728', s=60, zorder=5, label='Exceeds Standard'
            )

        # Regulatory threshold line
        ax.axhline(DDPCR_LIMIT, color='#d62728', linestyle='--', linewidth=1.2, alpha=0.85)

        title = f"{st}: {desc}"
        if beach:
            title += f"\n({beach})"
        ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
        ax.set_yscale('log')
        ax.set_ylim(10, 300000)
        ax.set_ylabel('Copies / 100mL', fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.tick_params(axis='x', rotation=30, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        'San Diego County Coastal Enterococcus ddPCR Time Series (Last 2 Weeks)\n'
        f'Red dashed line = State Health Standard ({DDPCR_LIMIT:,} copies/100mL)  |  Generated: {gen_time}',
        fontsize=13, fontweight='bold', y=0.995
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 1 to: {output_path}")


def plot_coastal_zones(
    ent_df: pd.DataFrame,
    station_order: list[str],
    output_path: str,
    gen_time: Optional[str] = None
):
    """
    Generate and save Figure 2: Grouped comparative time series across the 3 coastal zones.
    Top panel: Coronado Ocean Beaches (EH-060, EH-050, IB-079) grouped together.
    """
    if gen_time is None:
        gen_time = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    def get_coastal_zone(row):
        s = str(row['Station Name']).strip().upper()
        if s in {'EH-060', 'EH-050', 'IB-079'}:
            return 'Coronado Ocean Beaches'
        elif s == 'IB-068':
            return 'Silver Strand'
        elif s in {'IB-060', 'EH-030', 'EH-010', 'IB-050'}:
            return 'Imperial Beach'
        return 'Other'

    df_zoned = ent_df.copy()
    df_zoned['Zone'] = df_zoned.apply(get_coastal_zone, axis=1)

    zones = ['Coronado Ocean Beaches', 'Silver Strand', 'Imperial Beach']
    fig, axes = plt.subplots(len(zones), 1, figsize=(14, 4.2 * len(zones)), sharex=True)

    zone_styles = {
        'Coronado Ocean Beaches': {
            'colors': ['#1b4965', '#2a9d8f', '#e76f51'],
            'markers': ['o', 's', '^']
        },
        'Silver Strand': {
            'colors': ['#3a86ff'],
            'markers': ['o']
        },
        'Imperial Beach': {
            'colors': ['#0077b6', '#0096c7', '#d62828', '#9d0208'],
            'markers': ['o', 's', '^', 'D']
        }
    }

    for ax, zone in zip(axes, zones):
        zone_data = df_zoned[df_zoned['Zone'] == zone].sort_values('Timestamp')
        stations_in_zone = [s for s in station_order if s in zone_data['Station Name'].unique()]

        style = zone_styles.get(zone, {'colors': sns.color_palette('tab10', 10), 'markers': ['o'] * 10})

        for idx, st in enumerate(stations_in_zone):
            sub = zone_data[zone_data['Station Name'] == st]
            desc = sub['Description'].iloc[0] if len(sub) > 0 else st
            col = style['colors'][idx % len(style['colors'])]
            mk = style['markers'][idx % len(style['markers'])]

            ax.plot(
                sub['Timestamp'], sub['Result_Numeric'], marker=mk,
                label=f"{st}: {desc}", linewidth=2.0, markersize=6, color=col
            )

        ax.axhline(
            DDPCR_LIMIT, color='#d62728', linestyle='--', linewidth=1.5,
            label=f'State Standard ({DDPCR_LIMIT:,} copies/100mL)'
        )
        ax.set_yscale('log')
        ax.set_ylim(10, 300000)
        ax.set_ylabel('Copies / 100 mL', fontsize=11)
        ax.set_title(f'Coastal Zone: {zone}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.30, 1.0), fontsize=9.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    axes[-1].set_xlabel('Sample Date', fontsize=11)
    fig.suptitle(
        'Comparative Enterococcus Concentrations by Coastal Zone\n'
        f'State Health Standard: {DDPCR_LIMIT:,} copies/100mL  |  Generated: {gen_time}',
        fontsize=13, fontweight='bold', y=0.995
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 2 to: {output_path}")


def plot_alongshore_hovmoller(
    ent_df: pd.DataFrame,
    output_path: str,
    gen_time: Optional[str] = None
):
    """
    Generate and save Figure 3: Alongshore Space-Time (Hovmöller) diagram.
    Latitude vs. Date showing alongshore bacterial distribution and northward plumes.
    """
    if gen_time is None:
        gen_time = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    daily_st = ent_df.groupby([
        ent_df['Timestamp'].dt.date, 'Station Name', 'Latitude', 'Description'
    ])['Result_Numeric'].max().reset_index()
    daily_st.rename(columns={'Timestamp': 'Date'}, inplace=True)
    daily_st['Date'] = pd.to_datetime(daily_st['Date'])

    fig, ax = plt.subplots(figsize=(13, 6.5))

    scatter = ax.scatter(
        daily_st['Date'],
        daily_st['Latitude'],
        c=np.log10(np.maximum(daily_st['Result_Numeric'], 1)),
        s=np.clip(daily_st['Result_Numeric'] / 180, 45, 650),
        cmap='YlOrRd',
        edgecolors='black',
        linewidth=0.7,
        alpha=0.9
    )

    landmarks = {
        32.6856: 'Coronado North Beach (EH-060)',
        32.6818: 'Coronado Main Tower (EH-050)',
        32.6741: 'Coronado Av. Lunar (IB-079)',
        32.6264: 'Silver Strand Guard Shack (IB-068)',
        32.5857: 'Carnation Ave IB (IB-060)',
        32.5787: 'Imperial Beach Pier (EH-030)',
        32.5727: 'Cortez Ave IB (EH-010)',
        32.5666: 'End of Seacoast IB (IB-050)'
    }

    for lat_val, name in landmarks.items():
        ax.axhline(lat_val, color='#999999', linestyle=':', linewidth=0.8, alpha=0.5)

    ax.set_yticks(list(landmarks.keys()))
    ax.set_yticklabels([f"{name} ({lat:.3f}°N)" for lat, name in landmarks.items()], fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.set_xlabel('Sample Date', fontsize=11)
    ax.set_ylabel('Latitude (Alongshore Position)', fontsize=11)
    ax.set_title(
        'Alongshore Enterococcus Space-Time Distribution (South of Pt. Loma)\n'
        f'Bubble size & color indicate Enterococcus concentration (log10 scale)  |  Generated: {gen_time}',
        fontsize=12, fontweight='bold'
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('log10(Enterococcus Copies / 100 mL)', fontsize=10)

    cbar_limit = np.log10(DDPCR_LIMIT)
    cbar.ax.axhline(cbar_limit, color='red', linestyle='--', linewidth=1.5)
    cbar.ax.text(1.2, cbar_limit, f' Limit ({DDPCR_LIMIT:,})', color='red', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 3 to: {output_path}")


def compute_summary_stats(ent_df: pd.DataFrame, station_order: list[str]) -> list[dict]:
    """Compute key monitoring metrics for each station."""
    stats = []
    for st in station_order:
        sub = ent_df[ent_df['Station Name'] == st].sort_values('Timestamp')
        if len(sub) == 0:
            continue

        desc = sub['Description'].iloc[0]
        beach = sub['Beach Name'].iloc[0]
        lat = sub['Latitude'].iloc[0]
        latest_val = float(sub['Result_Numeric'].iloc[-1])
        latest_date = sub['Timestamp'].iloc[-1].strftime('%Y-%m-%d')
        latest_time = sub['Timestamp'].iloc[-1].strftime('%H:%M')
        max_val = float(sub['Result_Numeric'].max())
        geom_mean = float(np.exp(np.log(np.maximum(sub['Result_Numeric'], 1)).mean()))
        n_samples = len(sub)
        n_exceed = int((sub['Result_Numeric'] >= DDPCR_LIMIT).sum())
        pct_exceed = float((n_exceed / n_samples) * 100) if n_samples > 0 else 0.0

        is_exceeding_now = latest_val >= DDPCR_LIMIT

        stats.append({
            'station': st,
            'description': desc,
            'beach': beach,
            'latitude': round(float(lat), 4) if lat else None,
            'samples': n_samples,
            'latest_value': int(round(latest_val)),
            'latest_date': latest_date,
            'latest_time': latest_time,
            'latest_status': 'EXCEEDS STANDARD' if is_exceeding_now else 'BELOW STANDARD',
            'max_value': int(round(max_val)),
            'geometric_mean': int(round(geom_mean)),
            'exceedance_count': n_exceed,
            'exceedance_pct': round(pct_exceed, 1)
        })

    return stats


def export_html_table(stats: list[dict], output_path: str, gen_time: Optional[str] = None):
    """
    Generate an HTML table snippet with responsive styling and
    color-coded advisory badges, suitable for embedding directly into any website.
    """
    if gen_time is None:
        gen_time = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    rows_html = []
    for s in stats:
        status_class = "badge-danger" if s['latest_value'] >= DDPCR_LIMIT else "badge-success"
        status_text = "Advisory (Exceeds)" if s['latest_value'] >= DDPCR_LIMIT else "Open / Clean"

        rows_html.append(f"""
        <tr>
            <td class="font-mono"><strong>{s['station']}</strong></td>
            <td>{s['description']}</td>
            <td>{s['beach']}</td>
            <td class="text-right">{s['samples']}</td>
            <td class="text-right font-mono"><strong>{s['latest_value']:,}</strong> <span class="text-muted">({s['latest_date']})</span></td>
            <td class="text-center"><span class="badge {status_class}">{status_text}</span></td>
            <td class="text-right font-mono">{s['max_value']:,}</td>
            <td class="text-right font-mono">{s['geometric_mean']:,}</td>
            <td class="text-right">{s['exceedance_count']}/{s['samples']} ({s['exceedance_pct']}%)</td>
        </tr>""")

    html_content = f"""<!-- Water Quality Summary Table (Generated {gen_time}) -->
<style>
.wq-table-container {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 1.5rem 0;
    overflow-x: auto;
}}
.wq-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    color: #24292f;
    border: 1px solid #d0d7de;
}}
.wq-table th {{
    background-color: #f6f8fa;
    color: #24292f;
    font-weight: 600;
    padding: 10px 14px;
    border-bottom: 2px solid #d0d7de;
    text-align: left;
}}
.wq-table th.wq-table-title {{
    background-color: #f6f8fa;
    color: #0969da;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 10px 14px;
    border-bottom: 1px solid #d0d7de;
}}
.wq-table .wq-title-flex {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}}
.wq-table .wq-gen-date {{
    font-weight: 500;
    font-size: 0.85rem;
    color: #57606a;
}}
.wq-table td {{
    padding: 10px 14px;
    border-bottom: 1px solid #e1e4e8;
}}
.wq-table tr:hover {{
    background-color: #f6f8fa;
}}
.wq-table .text-right {{ text-align: right; }}
.wq-table .text-center {{ text-align: center; }}
.wq-table .font-mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
.wq-table .text-muted {{ color: #57606a; font-size: 0.82rem; }}
.badge {{
    display: inline-block;
    padding: 3px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.badge-danger {{
    background-color: #ffebe9;
    color: #cf222e;
    border: 1px solid #ff8182;
}}
.badge-success {{
    background-color: #dafbe1;
    color: #1a7f37;
    border: 1px solid #4ac26b;
}}
</style>
<div class="wq-table-container">
    <table class="wq-table">
        <thead>
            <tr class="wq-table-header-row">
                <th colspan="9" class="wq-table-title">
                    <div class="wq-title-flex">
                        <span>San Diego County Coastal Water Quality Monitoring Summary</span>
                        <span class="wq-gen-date">Generated: {gen_time}</span>
                    </div>
                </th>
            </tr>
            <tr>
                <th>Station</th>
                <th>Location</th>
                <th>Beach / Area</th>
                <th class="text-right">Samples</th>
                <th class="text-right">Latest (copies/100ml)</th>
                <th class="text-center">Status</th>
                <th class="text-right">Max</th>
                <th class="text-right">Geom. Mean</th>
                <th class="text-right">Exceedances (&gt;1,413)</th>
            </tr>
        </thead>
        <tbody>
{''.join(rows_html)}
        </tbody>
    </table>
</div>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved HTML table fragment to: {output_path}")


def export_json_summary(stats: list[dict], output_path: str):
    """Save structured summary data as JSON for web APIs."""
    metadata = {
        'generated_at': datetime.now().astimezone().isoformat(),
        'health_standard_copies_per_100ml': DDPCR_LIMIT,
        'monitoring_agency': 'San Diego County DEHQ',
        'station_count': len(stats),
        'stations': stats
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved summary JSON to: {output_path}")


def export_full_dashboard(
    stats: list[dict],
    fig1_file: str,
    fig2_file: str,
    fig3_file: str,
    output_path: str,
    date_range_str: str,
    html_table_file: Optional[str] = None
):
    """Create a standalone modern HTML report embedding the summary and all 3 figures."""
    table_file = html_table_file if html_table_file and os.path.exists(html_table_file) else "summary_table.html"
    with open(table_file, "r", encoding="utf-8") as f:
        table_snippet = f.read()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>San Diego County Coastal Water Quality Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f8f9fa;
            color: #1f2328;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 8px;
            padding: 32px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        h1 {{ margin-top: 0; color: #0969da; font-size: 1.8rem; }}
        .subtitle {{ color: #57606a; margin-bottom: 24px; font-size: 1rem; }}
        .section-title {{
            font-size: 1.3rem;
            margin: 32px 0 12px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #e1e4e8;
        }}
        .figure-card {{
            background: #ffffff;
            border: 1px solid #d0d7de;
            border-radius: 6px;
            margin-bottom: 24px;
            padding: 16px;
            text-align: center;
        }}
        .figure-card img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .figure-caption {{
            margin-top: 10px;
            font-size: 0.9rem;
            color: #57606a;
            text-align: left;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 16px;
            border-top: 1px solid #d0d7de;
            font-size: 0.85rem;
            color: #57606a;
            text-align: center;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>San Diego County Coastal Water Quality Report</h1>
    <div class="subtitle">Locations South of Point Loma (Coronado, Silver Strand, Imperial Beach) | Window: {date_range_str}</div>

    <div class="section-title">1. Monitoring Summary & Current Status</div>
    <p>Rapid ddPCR <em>Enterococcus</em> monitoring results from the County of San Diego DEHQ. Advisory threshold: <strong>1,413 copies / 100 mL</strong>.</p>
    {table_snippet}

    <div class="section-title">2. Comparative Coastal Zone Time Series</div>
    <div class="figure-card">
        <img src="{os.path.basename(fig2_file)}" alt="Comparative Coastal Zone Time Series">
        <div class="figure-caption"><strong>Figure 1:</strong> Comparative time series across Coronado Ocean Beaches (EH-060, EH-050, IB-079 grouped together), Silver Strand (IB-068), and Imperial Beach.</div>
    </div>

    <div class="section-title">3. Station-by-Station Time Series Grid</div>
    <div class="figure-card">
        <img src="{os.path.basename(fig1_file)}" alt="Station-by-Station Time Series Grid">
        <div class="figure-caption"><strong>Figure 2:</strong> Individual 3x3 multi-panel time series for all 9 active monitoring stations. Red dashed line indicates the 1,413 copies/100mL health advisory limit.</div>
    </div>

    <div class="section-title">4. Alongshore Space-Time (Hovmöller) Diagram</div>
    <div class="figure-card">
        <img src="{os.path.basename(fig3_file)}" alt="Alongshore Space-Time Hovmöller Diagram">
        <div class="figure-caption"><strong>Figure 3:</strong> Latitude vs. Date space-time distribution of Enterococcus concentrations from the Mexican border through Coronado.</div>
    </div>

    <div class="footer">
        Generated on {datetime.now().astimezone().strftime('%B %d, %Y at %H:%M %Z')} | Data Source: County of San Diego DEHQ
    </div>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved complete HTML dashboard to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate water quality plots (PNG) and web export summary (HTML/JSON)."
    )
    parser.add_argument(
        "--days", "-d", type=int, default=14,
        help="Number of days of data to download (default: 14)"
    )
    parser.add_argument(
        "--csv", default="sd_water_quality_south_pt_loma.csv",
        help="Output CSV file name (default: sd_water_quality_south_pt_loma.csv)"
    )
    parser.add_argument(
        "--output-dir", "-o", default=".",
        help="Directory to save figures and web exports (default: current directory)"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip downloading and reuse existing CSV file if available"
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_plot_style()

    csv_path = os.path.join(args.output_dir, args.csv)

    # 1. Download or load CSV
    if args.skip_download and os.path.exists(csv_path):
        print(f"Reusing existing CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"Downloading latest water quality data ({args.days} days)...")
        df = download_sd_water_quality(output_csv=csv_path, days=args.days)

    # 2. Clean & Prepare Data
    ent_df, station_order = prepare_data(df)
    print(f"Active stations to process ({len(station_order)}): {', '.join(station_order)}")

    # 3. Output file paths
    fig1_file = os.path.join(args.output_dir, "figure1_timeseries_grid.png")
    fig2_file = os.path.join(args.output_dir, "figure2_coastal_zones.png")
    fig3_file = os.path.join(args.output_dir, "figure3_alongshore_hovmoller.png")
    html_table_file = os.path.join(args.output_dir, "summary_table.html")
    json_file = os.path.join(args.output_dir, "summary_data.json")
    dashboard_file = os.path.join(args.output_dir, "water_quality_report.html")

    # 4. Generate Figures
    gen_time_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    print(f"\n--- Generating PNG Figures (Generated: {gen_time_str}) ---")
    plot_timeseries_grid(ent_df, station_order, fig1_file, gen_time=gen_time_str)
    plot_coastal_zones(ent_df, station_order, fig2_file, gen_time=gen_time_str)
    plot_alongshore_hovmoller(ent_df, fig3_file, gen_time=gen_time_str)

    # 5. Compute Summary Statistics & Export for Web
    print("\n--- Generating Web Exports ---")
    stats = compute_summary_stats(ent_df, station_order)
    export_html_table(stats, html_table_file, gen_time=gen_time_str)
    export_json_summary(stats, json_file)

    start_str = ent_df['Timestamp'].min().strftime('%b %d, %Y')
    end_str = ent_df['Timestamp'].max().strftime('%b %d, %Y')
    date_range_str = f"{start_str} – {end_str}"
    export_full_dashboard(stats, fig1_file, fig2_file, fig3_file, dashboard_file, date_range_str, html_table_file=html_table_file)

    print("\nProcessing complete! All figures and web exports are ready.")


if __name__ == "__main__":
    main()

