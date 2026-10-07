#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# stage_cycle_inputs.sh — copy the four per-cycle input files into IO_DIR.
#
# Aggressive-staging pattern: ROMS reads inputs from
# /scratch/matt/DA_Simulations/LV1_v2/tmp_input_output/ instead of /dataSIO,
# so per-cycle reads are all on the fast filesystem.
#
# Env-vars:
#   IC_SRC       source path (absolute) of the IC file
#   BRY_SRC      source path (absolute) of the BC file
#   FRC_SRC      source path (absolute) of the atm forcing file
#   OBS_SRC      source path (absolute) of the obs file
#   IO_DIR       destination directory (must exist)
#
# On stdout: prints the four staged destination paths, one per line, prefixed
# with a label — the caller (run_da_cycle_v2.sh) sources these as the
# IC_FILE/BRY_FILE/FRC_FILE/OBS_FILE env vars for set_up_DA_simulation_v2.sh.
#
# The staging preserves the source basename so the .in file's references
# (rewritten by set_up_DA_simulation_v2.sh) are unambiguous.
# ---------------------------------------------------------------------------
set -euo pipefail

: "${IC_SRC:?set IC_SRC to absolute path of the IC .nc}"
: "${BRY_SRC:?set BRY_SRC to absolute path of the BC .nc}"
: "${FRC_SRC:?set FRC_SRC to absolute path of the atm forcing .nc}"
: "${OBS_SRC:?set OBS_SRC to absolute path of the obs .nc}"
: "${IO_DIR:?set IO_DIR to the tmp_input_output directory}"

[[ -d "$IO_DIR" ]] || { echo "IO_DIR missing: $IO_DIR" >&2; exit 1; }

for src in "$IC_SRC" "$BRY_SRC" "$FRC_SRC" "$OBS_SRC"; do
    [[ -f "$src" ]] || { echo "SOURCE MISSING: $src" >&2; exit 1; }
done

_stage() {
    # usage: _stage LABEL SRC
    # IMPORTANT: STDOUT emits EXACTLY ONE line per call — the destination path.
    # Everything human-readable goes to STDERR so the caller can `mapfile` the
    # 4 destinations off stdout without any interleaving.
    local label="$1" src="$2" dst
    dst="${IO_DIR}/$(basename "$src")"
    # ALWAYS copy.  The previous size-equality skip was unsafe: an input
    # regenerated in place with unchanged dimensions (e.g. BC_LV1_*.nc after the
    # in-situ -> potential temperature fix) is byte-for-byte the same SIZE but
    # different CONTENT, so the skip silently re-used the stale staged copy.
    # A full copy of the four inputs is ~440 MB on beegfs -- seconds against a
    # 3.25 h cycle, and not worth the correctness risk.
    cp -f "$src" "$dst"
    echo "  ${label}: staged $(basename "$src") -> $dst" >&2
    # touch to defeat stale Lustre attr cache
    touch "$dst"
    printf 'STAGED_%s=%s\n' "$label" "$dst" >&2
    printf '%s\n' "$dst"
}

# Print destinations to stdout in the order IC, BRY, FRC, OBS.
_stage IC  "$IC_SRC"
_stage BRY "$BRY_SRC"
_stage FRC "$FRC_SRC"
_stage OBS "$OBS_SRC"
