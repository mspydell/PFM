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
#   0 * * * * /path/to/your/PFM/driver/run_hycom_grabber.sh
#
# NOTE: one of these per machine is enough. The cache it keeps
# (/scratch/PFM_Simulations/hycom_data) is shared, so a second person running
# this cron on the same host adds nothing and doubles the connections to
# hycom. mspydell runs it; everyone else just benefits from the cache.
#
# Top of the hour on purpose: the 45 min budget then puts the end at :45, so
# a block in the log always starts at :00 and stops by :45.
#
# Deliberately NOT doing what the forecast runners do:
#   - no git branch check or switch. This runs hourly, possibly while a
#     forecast is going; switching the working tree under a running PFM would
#     be a great way to break it.
#   - no .env sourcing. HYCOM over OPeNDAP needs no credentials.

# Derived from this script's own location, not hardcoded: the repo is shared,
# and a pinned /home/<me>/... path means whoever else pulls it silently runs
# my clone with my conda instead of their own.
PFM_ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
CONDA_SH=""
for c in "$HOME/anaconda3/etc/profile.d/conda.sh" \
         "$HOME/miniconda3/etc/profile.d/conda.sh" \
         "$HOME/mambaforge/etc/profile.d/conda.sh" \
         /opt/anaconda3/etc/profile.d/conda.sh; do
    [ -r "$c" ] && { CONDA_SH="$c"; break; }
done
PROXY_SH=/etc/profile.d/proxy.sh
ENV_NAME=PHM-env
LOGDIR=${PFM_ROOT}/log

mkdir -p "${LOGDIR}"
LOGFILE=${LOGDIR}/hycom_grabber_$(date -u '+%Y%m%d').log
fail () { echo "$(date -u '+%Y-%m-%dT%H:%MZ')  FATAL $*" >> "${LOGFILE}"; exit 1; }

# Everything this script says goes to the daily log, including anything conda
# or python prints on their own. Otherwise cron mails it, and a job that mails
# on every run is a job whose mail nobody reads. The grabber's own --log below
# appends to this same file.
exec >> "${LOGFILE}" 2>&1

# cron gets a bare environment, so conda has to be set up by hand -- "conda
# activate" only exists after the shell hook is sourced.
#
# Note the ordering. PATH is NOT touched before this point: if the caller
# already has the env active, "conda activate" is a no-op, so anything
# prepended to PATH beforehand stays in front and wins over the env. That is
# a real failure, not a hypothetical -- it is how the first version of this
# script ran /usr/bin/python and died on "No module named netCDF4" while
# conda activate had reported success.
# This host reaches the outside world only through a proxy, and cron does
# not read /etc/profile.d or .bashrc -- so without this the grabber runs with
# no http_proxy, every request times out, and the symptom is a quiet
# "CATALOG fetch failed (rc=28)" followed by a run that downloads nothing.
# Found exactly that way: the script worked by hand and did nothing under
# env -i. Prefer the system file so an admin changing the proxy fixes this
# too; fall back to the address in .bashrc if it ever disappears.
if [ -r "${PROXY_SH}" ]; then
    source "${PROXY_SH}"
else
    export http_proxy=http://132.239.2.228:3128/
    export https_proxy=http://132.239.2.228:3128/
fi
export ftp_proxy="${ftp_proxy:-$http_proxy}"

[ -n "${CONDA_SH}" ] && [ -r "${CONDA_SH}" ] || fail "no conda hook found (tried \$HOME/anaconda3, miniconda3, mambaforge, /opt/anaconda3)"
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

# prove the proxy actually works before spending 45 minutes finding out it
# does not. the catalog url is the same host everything else will use.
curl -s -o /dev/null --max-time 60 -w '' \
     https://tds.hycom.org/thredds/catalog.xml \
    || fail "cannot reach tds.hycom.org (http_proxy=${http_proxy:-unset})"

cd "${PFM_ROOT}/sdpm_py_util" || fail "no ${PFM_ROOT}/sdpm_py_util"

# --log takes YYYYMMDD and fills in today, so this leaves one file per day
# rather than one per run. 45 min keeps an hourly cron from overrunning its
# own next firing even in the worst case.
exec "${PY}" -u -W ignore hycom_grabber.py fill \
     --budget 45 \
     --workers 10 \
     --log "${LOGDIR}/hycom_grabber_YYYYMMDD.log"
