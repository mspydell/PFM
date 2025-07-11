#!/bin/bash
#script to run PFMv2
cd /home/mspydell/models/PFM_root/PFM
source /home/mspydell/.bashrc

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
fstdout=/home/mspydell/models/PFM_root/PFM/log/LVs_forecast_system_${dateZ}0600Z.log
fsterr=/home/mspydell/models/PFM_root/PFM/log/LVs_forecast_system_${dateZ}0600Z_ERROR.log

#python  -u -W "ignore" driver_run_forecast_all_LVs.py > ${fstdout}  2> >(tee -a ${fstderr} >&2)
in_py="/home/mspydell/models/PFM_root/PFM/sdpm_py_util/pfm_input_v2p0.py"
info_pkl="/scratch/matt/PFM_Simulations_v2/forecast_info.pkl"
python -u -W "ignore" driver_run_pfm_phm.py $in_py $info_pkl > ${fstdout}  2> >(tee -a ${fstderr} >&2)

cd /home/mspydell/models/PFM_root/PFM