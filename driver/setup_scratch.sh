#!/usr/bin/env bash
# setup_scratch.sh -- (re)build /scratch/PFM_Simulations tree, group-writable.
#
# For every directory:
#   - mkdir -p
#   - chmod 2775           (owner+group rwx, other r-x, setgid so new files
#                           inherit the parent's group)
#   - setfacl -dm u::rw,g::rw,o::r    (default ACL so new FILES are group rw)
#
# Idempotent: safe to re-run.  Existing directories are left in place;
# only the mode and default ACL are refreshed.

set -euo pipefail

ROOT="/scratch/PFM_Simulations"

DIRS=(
  ""

  "Grids"
  "executables"
  "ecmwf_data"
  "cdip_data"
  "hycom_data"
  "hycom_data/hy_tmp"
  "nwm_ncs"
  "restart_data"
  "restart_roms_data_new"
  "restart_swan_data"
  "restart_swan_data_new"
  "tide_data"
  "test"

  "LV1_Forecast"
  "LV1_Forecast/Forc"
  "LV1_Forecast/Forc/nwm_data"
  "LV1_Forecast/Forc/cdip_data"
  "LV1_Forecast/Run"
  "LV1_Forecast/His"
  "LV1_Forecast/Plots"

  "LV2_Forecast"
  "LV2_Forecast/Forc"
  "LV2_Forecast/Forc/nwm_data"
  "LV2_Forecast/Forc/cdip_data"
  "LV2_Forecast/Run"
  "LV2_Forecast/His"
  "LV2_Forecast/Plots"

  "LV3_Forecast"
  "LV3_Forecast/Forc"
  "LV3_Forecast/Forc/nwm_data"
  "LV3_Forecast/Forc/cdip_data"
  "LV3_Forecast/Run"
  "LV3_Forecast/His"
  "LV3_Forecast/Plots"

  "LV4_Forecast"
  "LV4_Forecast/Forc"
  "LV4_Forecast/Forc/nwm_data"
  "LV4_Forecast/Forc/cdip_data"
  "LV4_Forecast/Run"
  "LV4_Forecast/His"
  "LV4_Forecast/Plots"
)

if ! command -v setfacl >/dev/null 2>&1; then
  echo "ERROR: setfacl not found on PATH" >&2
  exit 1
fi

echo "root: $ROOT"
mkdir -p "$ROOT"

n_made=0
n_updated=0
for rel in "${DIRS[@]}"; do
  d="${ROOT%/}${rel:+/$rel}"
  if [[ -d "$d" ]]; then
    n_updated=$((n_updated + 1))
  else
    n_made=$((n_made + 1))
  fi
  mkdir -p "$d"
  chmod 2775 "$d"
  setfacl -dm u::rw,g::rw,o::r "$d"
done

echo "done: created $n_made new dirs, updated $n_updated existing dirs (${#DIRS[@]} total)"
echo
echo "verify a spot check:"
for d in \
  "$ROOT" \
  "$ROOT/LV3_Forecast/Forc" \
  "$ROOT/LV4_Forecast/Forc/nwm_data"
do
  printf '  %s\n' "$d"
  ls -ld "$d" | sed 's/^/    /'
  getfacl -p "$d" 2>/dev/null | grep -E '^(default|user::|group::|other::)' | sed 's/^/    /'
done
