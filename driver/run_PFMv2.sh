#!/bin/bash
#script to run PFMv2
cd /home/mspydell/models/PFM_root/PFM
source /home/mspydell/.bashrc
set -a
source /home/mspydell/models/PFM_root/PFM/.env
set +a


# check to see what git branch we are on
EXPECTED_BRANCH="PHM_development" # Or "master", "develop", etc.

current_branch=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch is: $current_branch"

if [ "$current_branch" != "$EXPECTED_BRANCH" ]; then
  echo "Error: You are not on the '$EXPECTED_BRANCH' branch."
  echo "switching branches..."
  git switch $EXCPECTED_BRANCH
  current_branch2=$(git rev-parse --abbrev-ref HEAD)
  echo "Current branch is now: $current_branch2"
  # exit 1 # Exit with an error code
fi
echo "Successfully on the '$EXPECTED_BRANCH' branch. Proceeding with script..."

cd /home/mspydell/models/PFM_root/PFM/driver

#########
#Initialize conda, needed for conda activate to work
eval "$(conda shell.bash hook)"
# Activate the desired environment
conda activate PHM-env

########

dateZ=$(date '+%Y%m%d')
fstdout=/home/mspydell/models/PFM_root/PFM/log/LVs_forecast_system_${dateZ}.log

#in_py="/home/mspydell/models/PFM_root/PFM/sdpm_py_util/pfm_operational_input_new.py"
in_py="/home/ffeddersen/PFM/sdpm_py_util/pfm_operational_input_new.py"
info_pkl="/scratch/PFM_Simulations/forecast_info_mss_new.pkl"
python -u -W "ignore" driver_run_pfm_phm.py $in_py $info_pkl > ${fstdout} 2>&1

cd /home/mspydell/models/PFM_root/PFM