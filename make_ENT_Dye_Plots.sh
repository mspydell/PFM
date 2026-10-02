
cd /home/ffeddersen/PFM_NEW
source /home/ffeddersen/.bashrc

set -a
source /home/ffeddersen/PFM_NEW/.env
set +a

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

#########
#Initialize conda, needed for conda activate to work
eval "$(conda shell.bash hook)"
# Activate the desired environment
conda activate PHM-env

########

dateZ=$(date '+%Y%m%d')
fstdout=/home/ffeddersen/PFM_NEW/OBS_QC/ENT_DYE_stdout.log
fsterr=/home/ffeddersen/PFM_NEW/OBS_QC/ENT_DYE_stderr.log

cd /home/ffeddersen/PFM_NEW/ddPCR_ENT
python3 -u -W "ignore" make_forecast_csv.py  >> ${fstdout}  2> >(tee -a ${fstderr} >&2)
python3 -u -W "ignore"  plot_dye_ddpcr.py  >> ${fstdout}  2> >(tee -a ${fstderr} >&2) 
#rm -f /projects/www-users/falk/PFM_Forecast

# transfer image
cp plots/dye_ddpcr_*_2weeks.png      /projects/www-users/falk/PFM_Forecast/Plots

# then save an archive file
cp plots/dye_ddpcr_*_2weeks.png      /projects/www-users/falk/PFM_Forecast/OLD_PLOTS

