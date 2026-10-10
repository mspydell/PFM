#!/bin/bash
# Run all four LVs from an EXISTING forecast_info.pkl, without rebuilding it.
#
# Use this instead of run_PFM_LV1234.sh / run_PFMv2.sh when the pickle is
# already set up the way you want it -- in particular when it starts at 00Z.
# Those scripts call driver_run_pfm_phm.py, which runs
# initfuns.initialize_model(in_py, pkl) and regenerates the pickle from
# pfm_operational_input_new.py, resetting fetch_time (typically to 06Z).
# driver_run_forecast_LV1234.py has that call commented out and takes the
# pickle as it finds it.
#
# NOTE: this uses /scratch/PFM_Simulations/forecast_info.pkl, the same pickle
# and the same /scratch tree as the nightly cron in
# /home/ffeddersen/PFM_NEW/run_forecast_LVs_v2.sh. Do not run this while the
# nightly job is running -- they will fight over restart, forcing and history
# files. Check with: squeue -u ffeddersen

cd /home/mspydell/models/PFM_root/PFM
source /home/mspydell/.bashrc

# Credentials for ecmwf, the ucsd pipeline and cdip. Without these the atm step
# dies in get_ecmwf_creds / get_pipeline_creds -- pipeline is unreadable, the
# ecmwf fallback then raises, and there is no third source. Only run_PFMv2.sh
# does this among the driver scripts, which is why it is easy to leave out.
# set -a exports everything assigned until set +a, so the .env values reach the
# python subprocesses rather than just this shell.
set -a
source /home/mspydell/models/PFM_root/PFM/.env
set +a

# check to see what git branch we are on
EXPECTED_BRANCH="main" # Or "master", "develop", etc.

current_branch=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch is: $current_branch"

if [ "$current_branch" != "$EXPECTED_BRANCH" ]; then
  echo "Not on the '$EXPECTED_BRANCH' branch (on '$current_branch'), switching..."
  if ! git switch "$EXPECTED_BRANCH"; then
    echo "FATAL: could not switch to '$EXPECTED_BRANCH'. Not running from an"
    echo "unknown branch -- fix the working tree and rerun."
    exit 1
  fi
  current_branch=$(git rev-parse --abbrev-ref HEAD)
  if [ "$current_branch" != "$EXPECTED_BRANCH" ]; then
    echo "FATAL: switch reported success but we are on '$current_branch'."
    exit 1
  fi
  echo "Current branch is now: $current_branch"
fi
echo "Successfully on the '$EXPECTED_BRANCH' branch. Proceeding with script..."

cd /home/mspydell/models/PFM_root/PFM/driver

#########
#Initialize conda, needed for conda activate to work
eval "$(conda shell.bash hook)"
# Activate the desired environment
conda activate PHM-env

########

info_pkl="/scratch/PFM_Simulations/forecast_info.pkl"

if [ ! -f "$info_pkl" ]; then
  echo "FATAL: $info_pkl does not exist. Nothing to run from."
  exit 1
fi

# dated log, so repeated runs do not overwrite each other
dateZ=$(date '+%Y%m%d%H%M')
fstdout=/home/mspydell/models/PFM_root/PFM/log/LVs_from_pkl_${dateZ}.log

# report what we are about to run from, so the log records the pickle's state
# rather than leaving you to guess which forecast this was
python -u -W "ignore" - "$info_pkl" <<'PYEOF'
import sys
sys.path.append('../sdpm_py_util')
import init_funs_forecast as initfuns
P = initfuns.get_model_info(sys.argv[1])
print('running from the EXISTING pickle (not rebuilt from the .py):')
for k in ['fetch_time','forecast_days','sim_time_1','sim_time_2','levels_to_run']:
    print('   %-14s %s' % (k, P.get(k)))
PYEOF

echo "log: ${fstdout}"

python -u -W "ignore" driver_run_forecast_LV1234.py \
       driver_run_forecast_LV1234 "$info_pkl" > ${fstdout} 2>&1

echo "...done. exit=$?"
cd /home/mspydell/models/PFM_root/PFM
