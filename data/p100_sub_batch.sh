#!/bin/bash

#SBATCH --job-name=sam2
#SBATCH --account=nn10004k
#SBATCH --output=5000_output_sam2.1_small.txt           # Standard output file
#SBATCH --error=error_sam2.txt             # Standard error file
#SBATCH --partition=accel #accel #normal  a100   # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=5 # Number of CPU cores per task
#SBATCH --time=15:15:00               # Maximum runtime (D-HH:MM:SS)
#SBATCH --mem-per-cpu=10G

#SBATCH --gres=gpu:p100:1 
##SBATCH --qos=devel  # for test only 

# grep -A 5 "Class  |  IoU  |  Acc" 

## Set up job environment:
set -o errexit  # Exit the script on any error

set -o nounset  # Treat any unset variables as an error

# module --quiet purge  # Reset the modules to the system default
module purge --force
source /cluster/projects/nn10004k/packages_install/torch_cu121/bin/activate
module load  CUDA/12.1.1
# module load  torchvision/0.13.1-foss-2022a-CUDA-11.7.2
# nvidia-smi
# nvcc --version

# pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
#print the python path with echo
# which python 
#module list
# python -c "import cv2; print(cv2.__version__)"
# python TRAIN.py
# the maximum batch size is 1 for p100
# Define your paths as variables

which python
python --version
python -c "import numpy; print(numpy.__version__, numpy.__file__)"

# export CUDA_LAUNCH_BLOCKING=1
python oasi_data.py
# python /cluster/home/snf52395/mmsegmentation/tools/train.py \
#    /cluster/home/snf52395/mmsegmentation/data/sam_vit-b_test_SeaObject-1024x1024.py \
#     --launcher="slurm" 
  
