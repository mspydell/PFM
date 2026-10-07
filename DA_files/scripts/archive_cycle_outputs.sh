#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# archive_cycle_outputs.sh — after a successful cycle, move the "keepers"
# out of IO_DIR into /dataSIO/DA_Simulations/LV1_v2/, and purge intermediates.
#
# Env-vars:
#   IO_DIR       ROMS runtime scratch (source of files to archive)
#   OUT_ROOT     archive root  (e.g. /dataSIO/DA_Simulations/LV1_v2)
#   YMD          cycle date tag (YYYYMMDD)
#
# Keepers (renamed with the cycle date tag):
#   IO_DIR/ocean_fwd_outer0.nc  -> OUT_ROOT/prior_files/LV1_fwd_outer0_YYYYMMDD.nc
#   IO_DIR/ocean_fwd_outer1.nc  -> OUT_ROOT/analysis_files/LV1_fwd_outer1_YYYYMMDD.nc
#   IO_DIR/roms_mod.nc          -> OUT_ROOT/analysis_files/roms_mod_YYYYMMDD.nc
#   IO_DIR/LV1_TLM_IC.nc        -> OUT_ROOT/increment_files/LV1_TLM_IC_YYYYMMDD.nc
#   IO_DIR/ocean_dai.nc         COPY OUT_ROOT/restart_files/ocean_dai_YYYYMMDD.nc
#                               (original stays in IO_DIR so the next cycle
#                                can chain from it without another download)
#
# Intermediates purged from IO_DIR (regenerated every cycle):
#   ocean_ads.nc  ocean_irp.nc  ocean_iad.nc  ocean_fwd.nc
#   ocean_tlm.nc  ocean_tlf.nc  ocean_adj.nc  ocean_dia.nc
#   LV1_ocean_his_*.nc  LV1_ocean_rst_*.nc
#   roms_hss.nc  roms_lcz.nc  roms_lze.nc  roms_err.nc
# ---------------------------------------------------------------------------
set -euo pipefail

: "${IO_DIR:?set IO_DIR}"
: "${OUT_ROOT:?set OUT_ROOT (archive root, e.g. /dataSIO/DA_Simulations/LV1_v2)}"
: "${YMD:?set YMD (cycle date tag YYYYMMDD)}"

[[ -d "$IO_DIR" ]] || { echo "IO_DIR missing: $IO_DIR" >&2; exit 1; }
mkdir -p "${OUT_ROOT}/prior_files" "${OUT_ROOT}/analysis_files" \
         "${OUT_ROOT}/increment_files" "${OUT_ROOT}/restart_files"

_mv_keeper() {
    # usage: _mv_keeper SRC_BASE OUT_SUB OUT_BASE
    local src="${IO_DIR}/$1" sub="$2" base="$3" dst="${OUT_ROOT}/$2/$3"
    if [[ -f "$src" ]]; then
        mv -f "$src" "$dst"
        echo "    keep: $(basename "$src") -> $sub/$3"
    else
        echo "    WARN: $(basename "$src") not present (skipping)"
    fi
}
_cp_keeper() {
    local src="${IO_DIR}/$1" sub="$2" base="$3" dst="${OUT_ROOT}/$2/$3"
    if [[ -f "$src" ]]; then
        cp -f "$src" "$dst"
        echo "    keep (copy): $(basename "$src") -> $sub/$3"
    else
        echo "    WARN: $(basename "$src") not present (skipping)"
    fi
}

echo "  archiving cycle $YMD outputs from $IO_DIR to $OUT_ROOT/"
_mv_keeper ocean_fwd_outer0.nc  prior_files      "LV1_fwd_outer0_${YMD}.nc"
_mv_keeper ocean_fwd_outer1.nc  analysis_files   "LV1_fwd_outer1_${YMD}.nc"
_mv_keeper roms_mod.nc          analysis_files   "roms_mod_${YMD}.nc"
_mv_keeper LV1_TLM_IC.nc        increment_files  "LV1_TLM_IC_${YMD}.nc"
# ocean_dai stays in IO_DIR for tomorrow's chain; also archived tagged.
_cp_keeper ocean_dai.nc         restart_files    "ocean_dai_${YMD}.nc"

# Purge intermediates (they regenerate every cycle; keeping them just eats space)
echo "  purging IO_DIR intermediates..."
for f in ocean_ads.nc ocean_irp.nc ocean_iad.nc ocean_fwd.nc \
         ocean_tlm.nc ocean_tlf.nc ocean_adj.nc ocean_dia.nc \
         roms_hss.nc roms_lcz.nc roms_lze.nc roms_err.nc; do
    [[ -f "${IO_DIR}/${f}" ]] && rm -f "${IO_DIR}/${f}" && echo "    rm ${f}"
done
# Wildcard-named ones (his + rst carry timestamps)
for f in "${IO_DIR}"/LV1_ocean_his_*.nc "${IO_DIR}"/LV1_ocean_rst_*.nc; do
    [[ -f "$f" ]] && rm -f "$f" && echo "    rm $(basename "$f")"
done

echo "  cycle $YMD archive complete"
