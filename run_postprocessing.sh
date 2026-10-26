#!/bin/bash
#script to run PFMv2
cd /home/ffeddersen/PFM_NEW
source /home/ffeddersen/.bashrc

set -a
source /home/ffeddersen/PFM_NEW/.env
set +a

# check to see what git branch we are on
EXPECTED_BRANCH="main"  #PHM_development" # Or "master", "develop", etc.

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

cd /home/ffeddersen/PFM_NEW/driver

#########
#Initialize conda, needed for conda activate to work
eval "$(conda shell.bash hook)"
# Activate the desired environment
conda activate PHM-env

########

#cd /home/ffeddersen/PFM_NEW/driver

dateZ=$(date '+%Y%m%d')
fstdout=/home/ffeddersen/PFM_NEW/log/LVs_forecast_system_${dateZ}0600Z.log
fsterr=/home/ffeddersen/PFM_NEW/log/LVs_forecast_system_${dateZ}0600Z_ERROR.log


#python  -u -W "ignore" driver_run_forecast_all_LVs.py > ${fstdout}  2> >(tee -a ${fstderr} >&2)
#in_py="/home/ffeddersen/PFM_NEW/sdpm_py_util/pfm_operational_input_new_gfs.py"
#in_py="/home/ffeddersen/PFM_NEW/sdpm_py_util/pfm_operational_input_new.py"
#info_pkl="/scratch/PFM_Simulations/forecast_info.pkl"
#python -u -W "ignore" driver_run_pfm_phm.py $in_py $info_pkl > ${fstdout}  2> >(tee -a ${fstderr} >&2)
#python -u -W "ignore" driver_run_pfm_phm.py $in_py $info_pkl > ${fstdout}  2> >(tee -a ${fstderr} >&2)

cd /home/ffeddersen/PFM_NEW/sdpm_py_util/
python -u -W "ignore" generate_PFM_images_LV123.py >> ${fstdout}  2> >(tee -a ${fstderr} >&2)

cd /home/ffeddersen/PFM_NEW
echo "Now copying to dataSIO and website"
./copy_forecast_to_dataSIO.sh

# next make the HFR plots
cd /home/ffeddersen/PFM_NEW/Hydro_QC/HFRadar
echo "Now making the HF Radar plots"
./make_HFradar.sh

# next make the ddPCR stuff
cd /home/ffeddersen/PFM_NEW
echo "Now making the ddPCR plots"
./make_ddPCR_ENT_Plots.sh

# next make the ddPCR/ENT comparison stuff
cd /home/ffeddersen/PFM_NEW
echo "Now making the ENT-dye  plots"
./make_ENT_Dye_Plots.sh

# now make PTB flowrate and mv qc fig
cd /home/ffeddersen/PFM_NEW/Hydro_QC/FlowRate
./make_Flowrate_Plot.sh

# make Dye Volume flowrate and mv qc fig
cd /home/ffeddersen/PFM_NEW/Hydro_QC/Dye_Volume
./make_dye_volume.sh

# make SD BAy Chart
cd /home/ffeddersen/PFM_NEW/Hydro_QC/Water_Level
./make_water_level.sh

# make Mooring QC
cd /home/ffeddersen/PFM_NEW/Hydro_QC/Moorings
./make_moorings_qc.sh

