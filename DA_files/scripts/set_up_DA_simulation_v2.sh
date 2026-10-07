#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# set_up_DA_simulation_v2.sh
#
# LV1_v2 variant of set_up_DA_simulation.sh.  Same behavior for template
# rewrites and NRM caching, but adds a required IO_DIR env var and rewrites
# EVERY output-file path in the LV1 4D-Var .in file to $IO_DIR (aggressive
# staging pattern: ROMS reads staged inputs + writes all outputs into
# /scratch/matt/DA_Simulations/LV1_v2/tmp_input_output/, never /dataSIO).
#
# Env-vars (all required unless noted):
#   START_DATE   YYYY-MM-DD (UTC) — the cycle start
#   IC_FILE      absolute path to initial condition .nc (already in IO_DIR)
#   BRY_FILE     absolute path to boundary .nc          (already in IO_DIR)
#   FRC_FILE     absolute path to atm forcing .nc       (already in IO_DIR)
#   OBS_FILE     absolute path to superobs .nc          (already in IO_DIR)
#   RUN_DIR      dir holding .in templates + executable + NRM files
#                default = /scratch/matt/DA_Simulations/LV1_v2/Run
#   IO_DIR       ROMS runtime scratch — reads staged inputs, writes outputs
#                default = /scratch/matt/DA_Simulations/LV1_v2/tmp_input_output
#   NtileI/NtileJ                        default 12 / 27
#   HDECAY_IC_KM / HDECAY_BC_KM / HDECAY_F_KM       default 15 / 15 / 50
#   URMS_TAG / WSRMS_TAG                            default urms0p05 / wsrms0p02
#
# The output .in file goes to $RUN_DIR/LV1_run_4dvar_${YMD}.in and s4dvar.in
# goes to $RUN_DIR/s4dvar_${YMD}.in, matching the v1 convention.  Only the
# path REFERENCES inside these .in files are redirected to $IO_DIR.
# ---------------------------------------------------------------------------
set -euo pipefail

# ----- required inputs ------------------------------------------------------
: "${START_DATE:?set START_DATE=YYYY-MM-DD (UTC) — the cycle start}"
: "${IC_FILE:?set IC_FILE  — absolute path to initial condition .nc}"
: "${BRY_FILE:?set BRY_FILE — absolute path to boundary .nc}"
: "${FRC_FILE:?set FRC_FILE — absolute path to atm forcing .nc}"
: "${OBS_FILE:?set OBS_FILE — absolute path to superobs .nc}"

DA_EXP="${DA_EXP:-/home/mspydell/research/sdtjreL3/v_intel/roms_fourdvar/DA_experiment}"
RIN_TEMPLATE="${RIN_TEMPLATE:-${DA_EXP}/run_in/LV1_run_4dvar.in}"
S4DVAR_TEMPLATE="${S4DVAR_TEMPLATE:-${DA_EXP}/run_in/s4dvar.in}"
RUN_DIR="${RUN_DIR:-/scratch/matt/DA_Simulations/LV1_v2/Run}"
IO_DIR="${IO_DIR:-/scratch/matt/DA_Simulations/LV1_v2/tmp_input_output}"
YMD="$(date -u -d "$START_DATE" +%Y%m%d)"
RIN_OUT="${RIN_OUT:-${RUN_DIR}/LV1_run_4dvar_${YMD}.in}"
S4DVAR_OUT="${S4DVAR_OUT:-${RUN_DIR}/s4dvar_${YMD}.in}"

NtileI="${NtileI:-12}"
NtileJ="${NtileJ:-27}"

# ----- NRM cache config (unchanged from v1) --------------------------------
NRM_DIR="${NRM_DIR:-/dataSIO/DA_Simulations/input_files/nrm_files}"
GRID_TAG="${GRID_TAG:-LV1}"
HDECAY_IC_KM="${HDECAY_IC_KM:-15}"
HDECAY_BC_KM="${HDECAY_BC_KM:-15}"
HDECAY_F_KM="${HDECAY_F_KM:-50}"

NRM_I_NAME="roms_nrm_i_${GRID_TAG}_HDI${HDECAY_IC_KM}km.nc"
NRM_M_NAME="roms_nrm_m_${GRID_TAG}.nc"
NRM_B_NAME="roms_nrm_b_${GRID_TAG}_HDB${HDECAY_BC_KM}km.nc"
NRM_F_NAME="roms_nrm_f_${GRID_TAG}_HDF${HDECAY_F_KM}km.nc"
NRM_I_CACHE="$NRM_DIR/$NRM_I_NAME"
NRM_M_CACHE="$NRM_DIR/$NRM_M_NAME"
NRM_B_CACHE="$NRM_DIR/$NRM_B_NAME"
NRM_F_CACHE="$NRM_DIR/$NRM_F_NAME"

# ----- STD config (unchanged from v1) --------------------------------------
STD_DIR="${STD_DIR:-/dataSIO/DA_Simulations/input_files/std_files}"
URMS_TAG="${URMS_TAG:-urms0p05}"
WSRMS_TAG="${WSRMS_TAG:-wsrms0p02}"

STD_I_FILE="$STD_DIR/roms_std_i_${GRID_TAG}_${URMS_TAG}.nc"
STD_M_FILE="$STD_DIR/roms_std_m_${GRID_TAG}.nc"
STD_B_FILE="$STD_DIR/roms_std_b_${GRID_TAG}_${URMS_TAG}.nc"
STD_F_FILE="$STD_DIR/roms_std_f_${GRID_TAG}_${WSRMS_TAG}.nc"
for f in "$STD_I_FILE" "$STD_M_FILE" "$STD_B_FILE" "$STD_F_FILE"; do
    [[ -f "$f" ]] || { echo "MISSING STD: $f" >&2; exit 1; }
done

# ----- sanity ---------------------------------------------------------------
for f in "$RIN_TEMPLATE" "$S4DVAR_TEMPLATE" "$IC_FILE" "$BRY_FILE" "$FRC_FILE" "$OBS_FILE"; do
    [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 1; }
done
[[ -d "$RUN_DIR" ]] || { echo "RUN_DIR missing: $RUN_DIR" >&2; exit 1; }
[[ -d "$IO_DIR"  ]] || { echo "IO_DIR missing: $IO_DIR"   >&2; exit 1; }
mkdir -p "$(dirname "$RIN_OUT")"

# ----- DSTART: whole days since 1999-01-01 UTC ------------------------------
DSTART_DAYS=$( awk -v s="$START_DATE" 'BEGIN{
    cmd1 = "date -u -d \""s"\" +%s"; cmd1 | getline t;  close(cmd1)
    cmd2 = "date -u -d 1999-01-01 +%s"; cmd2 | getline t0; close(cmd2)
    printf "%d", (t - t0) / 86400
}' )
echo "  START_DATE=$START_DATE  ->  DSTART=${DSTART_DAYS}.0d0 (days since 1999-01-01 UTC)"
echo "  RUN_DIR = $RUN_DIR"
echo "  IO_DIR  = $IO_DIR"

# ----- stage run.in and s4dvar.in -------------------------------------------
cp -f "$RIN_TEMPLATE"    "$RIN_OUT"
cp -f "$S4DVAR_TEMPLATE" "$S4DVAR_OUT"

esc() { printf '%s' "$1" | sed 's|/|\\/|g'; }
# PFM copy: VARNAME was hardcoded in the template at an absolute roms_fourdvar
# path and was never substituted, so a vendored .in reached back into the research
# tree at run time.  The template now carries a __DA_FILES__ placeholder which is
# expanded here.
DA_FILES="${DA_FILES:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DAF_ESC=$(esc "$DA_FILES")

IC_ESC=$(esc  "$IC_FILE")
BR_ESC=$(esc  "$BRY_FILE")
FR_ESC=$(esc  "$FRC_FILE")
OB_ESC=$(esc  "$OBS_FILE")
S4_ESC=$(esc  "$S4DVAR_OUT")
IO_ESC=$(esc  "$IO_DIR")
RUN_ESC=$(esc "$RUN_DIR")

# DSTART line
# expand the __DA_FILES__ placeholder (VARNAME, and anything else added later)
sed -i "s|__DA_FILES__|${DAF_ESC}|g" "$RIN_OUT"

sed -i "s|^\( *DSTART *=\)[^!]*|\1  ${DSTART_DAYS}.0d0               |" "$RIN_OUT"

# Tile partition
sed -i "s|^\( *NtileI *== *\)[0-9]\+|\1${NtileI}|" "$RIN_OUT"
sed -i "s|^\( *NtileJ *== *\)[0-9]\+|\1${NtileJ}|" "$RIN_OUT"

# ----- run.in path rewrites (INPUTS) ---------------------------------------
# ININAME / BRYNAME / FRCNAME point at staged input files in $IO_DIR.
sed -i "s|^\( *ININAME *== *\)[^ ].*|\1${IC_ESC}|" "$RIN_OUT"
sed -i "s|^\( *BRYNAME *== *\)[^ ].*|\1${BR_ESC}|" "$RIN_OUT"
sed -i "s|^\( *FRCNAME *== *\)[^ ].*|\1${FR_ESC}|" "$RIN_OUT"
# APARNAM points at the staged s4dvar.in (single '=' not '==')
sed -i "s|^\( *APARNAM *= *\)[^ ].*|\1  ${S4_ESC}|" "$RIN_OUT"

# ----- run.in path rewrites (OUTPUTS)  --  the v2 addition -----------------
# The template hard-codes 12 output paths under /scratch/matt/DA_Simulations/Run/.
# Rewrite each to $IO_DIR/ so ROMS writes into tmp_input_output/, not the
# shared v1 Run dir.  Names of the leaf files are preserved.
for var in ITLNAME IRPNAME IADNAME FWDNAME ADSNAME \
           DAINAME RSTNAME HISNAME TLMNAME TLFNAME ADJNAME DIANAME; do
    # replace anything after "== " with $IO_DIR/<original-basename>
    sed -i -E "s|^( *${var} *== *)/[^ ]*/([^/ ]+\.nc[^ ]*)|\1${IO_ESC}/\2|" "$RIN_OUT"
done

# ----- s4dvar.in path rewrites ---------------------------------------------
# OBSname → staged obs file in IO_DIR
sed -i "s|^\( *OBSname *== *\)[^ ].*|\1${OB_ESC}|" "$S4DVAR_OUT"

# STDname{M,I,B,F} — kept on /dataSIO (large + static, read once)
SI_ESC=$(esc "$STD_I_FILE")
SM_ESC=$(esc "$STD_M_FILE")
SB_ESC=$(esc "$STD_B_FILE")
SF_ESC=$(esc "$STD_F_FILE")
sed -i "s|^\( *STDnameM *== *\)[^ ].*|\1${SM_ESC}|" "$S4DVAR_OUT"
sed -i "s|^\( *STDnameI *== *\)[^ ].*|\1${SI_ESC}|" "$S4DVAR_OUT"
sed -i "s|^\( *STDnameB *== *\)[^ ].*|\1${SB_ESC}|" "$S4DVAR_OUT"
sed -i "s|^\( *STDnameF *== *\)[^ ].*|\1${SF_ESC}|" "$S4DVAR_OUT"

# ROMS DA scratch output files → $IO_DIR
for var in HSSname LCZname LZEname MODname ERRname; do
    sed -i "s|^\( *${var} *== *\)[^ ].*|\1${IO_ESC}\\/roms_$(echo ${var} | sed 's/name$//' | tr '[:upper:]' '[:lower:]').nc|" \
        "$S4DVAR_OUT"
done

# ----- NRM cache: decide read vs compute mode (unchanged from v1) ----------
mkdir -p "$NRM_DIR"
if [[ ! -f "$NRM_M_CACHE" ]]; then
    /home/mspydell/anaconda3/envs/PHM-env/bin/python3 -c \
        "import netCDF4 as nc; nc.Dataset('$NRM_M_CACHE','w',format='NETCDF4').close()"
    echo "  created empty M-slot NRM dummy -> $NRM_M_CACHE (unused in strong-constraint)"
fi

if [[ -f "$NRM_I_CACHE" && -f "$NRM_B_CACHE" && -f "$NRM_F_CACHE" ]]; then
    NRM_MODE="read"
    LWRT_LINE="F F F F"
    NRM_I_USE="$NRM_I_CACHE"
    NRM_B_USE="$NRM_B_CACHE"
    NRM_F_USE="$NRM_F_CACHE"
else
    NRM_MODE="compute"
    LWRT_LINE="T T T T"
    NRM_I_USE="${RUN_DIR}/${NRM_I_NAME}"
    NRM_B_USE="${RUN_DIR}/${NRM_B_NAME}"
    NRM_F_USE="${RUN_DIR}/${NRM_F_NAME}"
fi
NRM_M_USE="$NRM_M_CACHE"
NRM_I_ESC=$(esc "$NRM_I_USE")
NRM_M_ESC=$(esc "$NRM_M_USE")
NRM_B_ESC=$(esc "$NRM_B_USE")
NRM_F_ESC=$(esc "$NRM_F_USE")

sed -i "s|^\( *LdefNRM *== *\)[TF ]\+\(.*\)|\1${LWRT_LINE}                \2|" "$S4DVAR_OUT"
sed -i "s|^\( *LwrtNRM *== *\)[TF ]\+\(.*\)|\1${LWRT_LINE}                \2|" "$S4DVAR_OUT"
sed -i "s|^\( *NRMnameM *== *\)[^ ].*|\1${NRM_M_ESC}|" "$S4DVAR_OUT"
sed -i "s|^\( *NRMnameI *== *\)[^ ].*|\1${NRM_I_ESC}|" "$S4DVAR_OUT"
sed -i "s|^\( *NRMnameB *== *\)[^ ].*|\1${NRM_B_ESC}|" "$S4DVAR_OUT"
sed -i "s|^\( *NRMnameF *== *\)[^ ].*|\1${NRM_F_ESC}|" "$S4DVAR_OUT"

echo "  NRM mode = $NRM_MODE (LwrtNRM=$LWRT_LINE)"

# ROMS FIND_FILE runs before LwrtNRM — placeholders required when compute mode
if [[ "$NRM_MODE" == "compute" ]]; then
    /home/mspydell/anaconda3/envs/PHM-env/bin/python3 - <<PYEOF
import netCDF4 as nc
for p in ("$NRM_I_USE", "$NRM_B_USE", "$NRM_F_USE"):
    nc.Dataset(p, "w", format="NETCDF4").close()
PYEOF
    echo "  created empty NRM placeholders in Run/ for FIND_FILE"
fi

echo "  wrote staged run.in    -> $RIN_OUT"
echo "  wrote staged s4dvar.in -> $S4DVAR_OUT"
echo "  post-stage sanity:"
grep -nE "^ *(DSTART|ININAME|BRYNAME|FRCNAME|APARNAM|NtileI|NtileJ|DAINAME|HISNAME|RSTNAME|ADJNAME) " "$RIN_OUT" | grep -v '^[0-9]*:!' | sed 's|^|      run.in    |'
grep -nE "^ *(OBSname|LdefNRM|LwrtNRM|NRMname[MIBF]|STDname[MIBF]|HSSname|LCZname|LZEname|MODname|ERRname) " "$S4DVAR_OUT" | grep -v '^[0-9]*:!' | sed 's|^|      s4dvar.in |'
