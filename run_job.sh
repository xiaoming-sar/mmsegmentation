#!/bin/bash
#SBATCH -p accel
#SBATCH --account=nn10004k
#SBATCH --output=output_adamW_OCRNet_4GPU_80000.txt
#SBATCH --error=error.txt
#SBATCH --job-name=sam
#SBATCH --gres=gpu:p100:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=5G
#SBATCH --time=15:15:00

## Set up job environment:
set -o errexit  # Exit the script on any error

set -o nounset  # Treat any unset variables as an error

# module --quiet purge  # Reset the modules to the system default
module purge --force
source /cluster/projects/nn10004k/packages_install/torch_cu121/bin/activate
module load  CUDA/12.1.1 

# export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
srun python -u tools/train.py data/sam_vit-b_OCRNet_SeaObject-1024x1024_MultiGPU.py  \
    --work-dir /cluster/projects/nn10004k/packages_install/seaobject_ocrnet5000  \
    --launcher=slurm
