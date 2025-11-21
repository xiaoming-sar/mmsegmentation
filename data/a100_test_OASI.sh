#!/bin/bash

#SBATCH -p a100 #accel # a100
#SBATCH --account=nn10004k
#SBATCH --output=test_sam2.1_tiny2.1_3in1Model_Test_fwIOU.txt  #need change
#SBATCH --error=test_error_fwIOU.txt
#SBATCH --job-name=sam2_tiny
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=10G
#SBATCH --time=10:15:00

## Set up job environment:
set -o errexit  # Exit the script on any error

set -o nounset  # Treat any unset variables as an error

# module --quiet purge  # Reset the modules to the system default
module purge --force
source /cluster/projects/nn10004k/packages_install/torch_cu121/bin/activate
# module load  CUDA/12.1.1
# export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH  
# export CUDA_VISIBLE_DEVICES=1 

EXPERIMENT_DIR="sam2.1_tiny_40K_3in1Model_fwIOU" # result foldfer, need change 
CONFIG_FILE="hiera-sam2-small_SegSegformer_SeaObject-1008x1008_MultiA100_3in1.py" # need change based on the testloader 

CHECKPOINT_FILE="/cluster/projects/nn10004k/packages_install/SAM2.1_tiny_OASIs_3in1_40K/iter_40000.pth"
BASE_DATA_DIR="/cluster/home/snf52395/mmsegmentation/data"

WORK_DIR="${BASE_DATA_DIR}/${EXPERIMENT_DIR}"
OUT_DIR="${WORK_DIR}/out_dir"
SHOW_DIR="${WORK_DIR}"

python  /cluster/home/snf52395/mmsegmentation/tools/test.py  "${CONFIG_FILE}" "${CHECKPOINT_FILE}"  \
        --work-dir "${WORK_DIR}"  \
        --out  "${OUT_DIR}"       \
        --show-dir "${SHOW_DIR}"  
        



