"""Tide-gauge stations compared with PFM sea surface height. Every script in this folder takes
--station with one of these keys (default: all of them).

  noaa      NOAA CO-OPS station ID (water level, air pressure and tide predictions)
  lon, lat  position used to pick the nearest model rho point
  level     PFM grid; LV4 does not reach La Jolla, so La Jolla uses LV3
"""
GRID_DIR = "/dataSIO/PFM_Simulations/Grid"

STATIONS = {
    "SDBay": dict(title="San Diego Bay", noaa="9410170", lon=-117.1767, lat=32.715,
                  level="LV4", grid=f"{GRID_DIR}/GRID_SDTJRE_LV4_mss_oct2024.nc"),  # MATLAB location ID 1
    "LaJolla": dict(title="La Jolla", noaa="9410230", lon=-117.25714, lat=32.86689,
                    level="LV3", grid=f"{GRID_DIR}/GRID_SDTJRE_LV3_rx020.nc"),      # MATLAB location ID 11, Scripps Pier
}
