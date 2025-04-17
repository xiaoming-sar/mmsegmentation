#!/bin/bash

#SBATCH -p a100
#SBATCH --account=nn10004k
#SBATCH --output=test_sam2.1_small_Type1_a100.txt
#SBATCH --error=error.txt
#SBATCH --job-name=sam
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=20G
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

python  /cluster/home/snf52395/mmsegmentation/tools/test.py  hiera-sam2-small_SegSegformer_SeaObject-1008x1008_MultiA100_TYPE1_Test.py /cluster/projects/nn10004k/packages_install/sam2_test/iter_20000.pth  \
        --work-dir /cluster/home/snf52395/mmsegmentation/data/sam2.1_samll_20000_a100_Type1  \
        --out  /cluster/home/snf52395/mmsegmentation/data/sam2.1_samll_20000_a100_Type1/out_dir  \
        --show-dir /cluster/home/snf52395/mmsegmentation/data/sam2.1_samll_20000_a100_Type1 
        



