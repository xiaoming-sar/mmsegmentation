#!/bin/bash

#SBATCH -p accel
#SBATCH --account=nn10004k
#SBATCH --output=test_adamW_OCRNet_4GPU.txt
#SBATCH --error=error.txt
#SBATCH --job-name=sam
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=5G
#SBATCH --time=10:15:00

## Set up job environment:
set -o errexit  # Exit the script on any error

set -o nounset  # Treat any unset variables as an error

# module --quiet purge  # Reset the modules to the system default
# module purge --force
# source /cluster/projects/nn10004k/packages_install/torch_cu121/bin/activate
# module load  CUDA/12.1.1
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH  
# export CUDA_VISIBLE_DEVICES=1 

python  ../../tools/test.py ../sam_vit-b_OCRNet_SeaObject-1024x1024_MultiGPU.py /cluster/projects/nn10004k/packages_install/seaobject_work1/iter_10000.pth  \
        --work-dir /cluster/home/snf52395/mmsegmentation/data/test_ocrnet10000  \
        --out  /cluster/home/snf52395/mmsegmentation/data/test_ocrnet10000/out_dir  \
        --show-dir /cluster/home/snf52395/mmsegmentation/data/test_ocrnet10000  
        



