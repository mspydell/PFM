
cd /home/ffeddersen/PFM_NEW
source /home/ffeddersen/.bashrc

set -a
source /home/ffeddersen/PFM_NEW/.env
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

#########
#Initialize conda, needed for conda activate to work
#eval "$(conda shell.bash hook)"
# Activate the desired environment
#conda activate PHM-env

########

dateZ=$(date '+%Y%m%d')
fstdout=/home/ffeddersen/PFM_NEW/OBS_QC/ddPCR_stdout.log
fsterr=/home/ffeddersen/PFM_NEW/OBS_QC/ddPCR_stderr.log

cd /home/ffeddersen/PFM_NEW/ddPCR_ENT
python -u -W "ignore" plot_water_quality.py  --days 20 >> ${fstdout}  2> >(tee -a ${fstderr} >&2)

#rm -f /projects/www-users/falk/PFM_Forecast
cp *.png      /projects/www-users/falk/PFM_Forecast/Plots
cp summary_table.html       /projects/www-users/falk/PFM_Forecast/Plots

# then save an archive file
cp figure1_timeseries_grid.png /projects/www-users/falk/PFM_Forecast/OLD_PLOTS/figure1_timeseries_grid_${dateZ}.png
cp figure2_coastal_zones.png /projects/www-users/falk/PFM_Forecast/OLD_PLOTS/figure2_coastal_zones_${dateZ}.png
