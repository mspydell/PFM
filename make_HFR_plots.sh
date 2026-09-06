
cd /home/ffeddersen/PFM_NEW
source /home/ffeddersen/.bashrc

#set -a
#source /home/ffeddersen/PFM_NEW/.env
#set +a

# check to see what git branch we are on
#EXPECTED_BRANCH="PHM_development" # Or "master", "develop", etc.

#current_branch=$(git rev-parse --abbrev-ref HEAD)
#echo "Current branch is: $current_branch"

#if [ "$current_branch" != "$EXPECTED_BRANCH" ]; then
#  echo "Error: You are not on the '$EXPECTED_BRANCH' branch."
#  echo "switching branches..."
#  git switch $EXCPECTED_BRANCH
#  current_branch2=$(git rev-parse --abbrev-ref HEAD)
#  echo "Current branch is now: $current_branch2"
  # exit 1 # Exit with an error code
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
