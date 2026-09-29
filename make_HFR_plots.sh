
cd /home/ffeddersen/PFM_NEW
source /home/ffeddersen/.bashrc

#set -a
#source /home/ffeddersen/PFM_NEW/.env
#set +a

# check to see what git branch we are on
# this guard is disabled here. if you re-enable it, use this version -- the
# old one had "git switch $EXCPECTED_BRANCH" (a typo, so it did nothing) and a
# commented-out exit, which let a failed switch run from the wrong branch.
#EXPECTED_BRANCH="PHM_development" # Or "master", "develop", etc.

#current_branch=$(git rev-parse --abbrev-ref HEAD)
#echo "Current branch is: $current_branch"

#if [ "$current_branch" != "$EXPECTED_BRANCH" ]; then
#  echo "Not on the '$EXPECTED_BRANCH' branch (on '$current_branch'), switching..."
#  if ! git switch "$EXPECTED_BRANCH"; then
#    echo "FATAL: could not switch to '$EXPECTED_BRANCH'."
#    exit 1
#  fi
#  current_branch=$(git rev-parse --abbrev-ref HEAD)
#  if [ "$current_branch" != "$EXPECTED_BRANCH" ]; then
#    echo "FATAL: switch reported success but we are on '$current_branch'."
#    exit 1
#  fi
#  echo "Current branch is now: $current_branch"
#fi
#echo "Successfully on the '$EXPECTED_BRANCH' branch. Proceeding with script..."

#########
#Initialize conda, needed for conda activate to work
#eval "$(conda shell.bash hook)"
# Activate the desired environment
#conda activate PHM-env

########

#dateZ=$(date '+%Y%m%d')
#fstdout=/home/ffeddersen/PFM_NEW/OBS_QC/HFR_QC_${dateZ}0600Z.log
#fsterr=/home/ffeddersen/PFM_NEW/OBS_QC/HFR_QC_${dateZ}0600Z_ERROR.log

cd /home/ffeddersen/PFM_NEW/qc_obs_py_files

/home/ffeddersen/anaconda3/envs/PHM-env/bin/python3  -u -W "ignore" make_obs_qc_figures.py >>  HFR_err.log   2> >(tee -a HFR_err2.log)




cp /scratch/PFM_Simulations/obs_qc_figures/*latest*.png    /projects/www-users/falk/PFM_Forecast/Plots
cp /scratch/PFM_Simulations/obs_qc_figures/*Z*.png    /dataSIO/HFRadar_files/QC_PNG
rm -f /scratch/PFM_Simulations/obs_qc_figures/*.png
