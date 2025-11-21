#!/bin/bash

#SBATCH -p accel #accel # a100
#SBATCH --account=nn10004k
#SBATCH --output=test_sam2.1_tiny2.1_segformer.txt  #need change
#SBATCH --error=test_error_fwIOU.txt
#SBATCH --job-name=sam2_tiny_seg
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

# CONFIG_FILES_CHECKPOINT_DICT=['1_DeepLabV3_SeaObject-1008x1008_MultiA100_3in1.py': 'Deeplab_OASIs_3in1_40K' ,
#                                 '1_hiera-sam2-tiny_SegSegformer_SeaObject-1008x1008_MultiA100.py': 'SAM2.1_tiny_OASIs_3in1_40K',
#                                 '1_hiera-sam2-tiny_FPN_OCR_SeaObject-1008x1008_MultiA100.py' 'SAM2.1_Tiny_FPN_OCR_OASIs_3in1_40K' ,
#                                 '1_Hrnet_OCR_SeaObject-1008x1008_MultiA100_3in1.py': 'HrnetOCR_OASIs_3in1_40K',
#                                 '1_hiera-sam2-tiny_OCR_SeaObject-1008x1008_MultiA100.py': 'SAM2.1_Tiny_OCR_OASIs_3in1_40K',     
#                                 '1_Unet_FCN_SeaObject-1008x1008_MultiA100_3in1.py':'Unet_FCN_OASIs_3in1_40K']
          
# DATA_TYPE=["TYPE2", "TYPE2", "TYPE3"]
DATA_TYPE="TYPE2"
# SAVE_NAME = "hiera-sam2-tiny_FPN_OCR" 
EXPERIMENT_DIR="sam2.1_t_segformer_40K_${DATA_TYPE}" 
CONFIG_FILE="1_hiera-sam2-tiny_SegSegformer_SeaObject-1008x1008_MultiA100.py" 

TEST_DATA_ROOT="/cluster/projects/nn10004k/ml_SeaObject_Data/OASIs_dataset_patch1024/${DATA_TYPE}"

# CHECKPOINT_FILE="/cluster/projects/nn10004k/packages_install/${FILL WITH CHECKPOINT DICT BASED ON LOOP OF CONFIG_FILE}/iter_40000.pth"
CHECKPOINT_FILE="/cluster/projects/nn10004k/packages_install/SAM2.1_tiny_OASIs_3in1_40K/iter_40000.pth"
BASE_DATA_DIR="/cluster/projects/nn10004k/packages_install/1models_test"

WORK_DIR="${BASE_DATA_DIR}/${EXPERIMENT_DIR}"
OUT_DIR="${WORK_DIR}/out_dir"
SHOW_DIR="${WORK_DIR}"

python  /cluster/home/snf52395/mmsegmentation/tools/test.py  "${CONFIG_FILE}" "${CHECKPOINT_FILE}"  \
        --work-dir "${WORK_DIR}"  \
        --out  "${OUT_DIR}"       \
        --cfg-options test_dataloader.dataset.data_root="${TEST_DATA_ROOT}"  test_dataloader.batch_size=1           \
        --show-dir "${SHOW_DIR}"  
        



