#!/bin/bash
#script to run PFMv2
cd /home/mspydell/models/PFM_root/PFM
source /home/mspydell/.bashrc

# check to see what git branch we are on
EXPECTED_BRANCH="main" # Or "master", "develop", etc.

current_branch=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch is: $current_branch"

if [ "$current_branch" != "$EXPECTED_BRANCH" ]; then
  echo "Not on the '$EXPECTED_BRANCH' branch (on '$current_branch'), switching..."
  # the switch used to be "git switch $EXCPECTED_BRANCH" -- a typo, so the
  # variable was empty and the switch silently did nothing. nothing checked
  # afterwards either, and the exit was commented out, so a failed switch ran
  # the whole forecast from whatever branch happened to be checked out.
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

dateZ=$(date '+%Y%m%d%H')
fstdout=/home/mspydell/models/PFM_root/PFM/log/LV4_pfm_system_new.log
info_pkl="/scratch/PFM_Simulations/forecast_info_mss_testing.pkl"

python -u -W "ignore" driver_run_LV4_new.py $info_pkl > ${fstdout} 2>&1

cd /home/mspydell/models/PFM_root/PFM