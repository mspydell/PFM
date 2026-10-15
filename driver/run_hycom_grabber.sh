#!/bin/bash
# Keep the HYCOM cache correct and full, out of band from the forecast.
#
# Meant for an hourly cron. Each run verifies every file in the cache, deletes
# any that fail, refetches those plus anything missing, and prunes to the five
# most recent cycles. It is safe to run at any hour: the grabber skips hycom's
# maintenance windows, refuses to start while a forecast is downloading, and
# takes a lock so a slow catch-up run is never doubled up on.
#
# Install with:  crontab -e
#   23 * * * * /home/mspydell/models/PFM_root/PFM/driver/run_hycom_grabber.sh
#
# Deliberately NOT doing what the forecast runners do:
#   - no git branch check or switch. This runs hourly, possibly while a
#     forecast is going; switching the working tree under a running PFM would
#     be a great way to break it.
#   - no .env sourcing. HYCOM over OPeNDAP needs no credentials.

PFM_ROOT=/home/mspydell/models/PFM_root/PFM
CONDA_SH=/home/mspydell/anaconda3/etc/profile.d/conda.sh
ENV_NAME=PHM-env
LOGDIR=${PFM_ROOT}/log

mkdir -p "${LOGDIR}"
LOGFILE=${LOGDIR}/hycom_grabber_$(date -u '+%Y%m%d').log
fail () { echo "$(date -u '+%Y-%m-%dT%H:%MZ')  FATAL $*" >> "${LOGFILE}"; exit 1; }

# cron gets a bare environment, so conda has to be set up by hand -- "conda
# activate" only exists after the shell hook is sourced.
#
# Note the ordering. PATH is NOT touched before this point: if the caller
# already has the env active, "conda activate" is a no-op, so anything
# prepended to PATH beforehand stays in front and wins over the env. That is
# a real failure, not a hypothetical -- it is how the first version of this
# script ran /usr/bin/python and died on "No module named netCDF4" while
# conda activate had reported success.
[ -r "${CONDA_SH}" ] || fail "no conda hook at ${CONDA_SH}"
source "${CONDA_SH}"
conda activate "${ENV_NAME}" || fail "could not activate ${ENV_NAME}"

# ...and verify rather than trust, since activate can succeed while leaving
# the wrong interpreter in front.
PY=$(command -v python)
case "${PY}" in
    */envs/${ENV_NAME}/bin/python) ;;
    *) fail "python is ${PY}, expected the ${ENV_NAME} one" ;;
esac
"${PY}" -c 'import netCDF4' 2>/dev/null || fail "${PY} cannot import netCDF4"

# ncks is the system build at /usr/bin, not a conda package, so make sure the
# bare cron PATH still reaches it.
command -v ncks >/dev/null || fail "ncks not on PATH (${PATH})"

cd "${PFM_ROOT}/sdpm_py_util" || fail "no ${PFM_ROOT}/sdpm_py_util"

# --log takes YYYYMMDD and fills in today, so this leaves one file per day
# rather than one per run. 45 min keeps an hourly cron from overrunning its
# own next firing even in the worst case.
exec "${PY}" -u -W ignore hycom_grabber.py fill \
     --budget 45 \
     --workers 10 \
     --log "${LOGDIR}/hycom_grabber_YYYYMMDD.log"
