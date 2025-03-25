#!/bin/bash
#SBATCH -p a100
#SBATCH --account=nn10004k
#SBATCH --output=output_adamW_OCRNet_4GPU_80000_a100.txt
#SBATCH --error=error.txt
#SBATCH --job-name=sam
#SBATCH --gres=gpu:a100:3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-cpu=10G
#SBATCH --time=2-15:15:00

## Set up job environment:
set -o errexit  # Exit the script on any error

set -o nounset  # Treat any unset variables as an error

# module --quiet purge  # Reset the modules to the system default
module purge --force
source /cluster/projects/nn10004k/packages_install/torch_cu121/bin/activate
module load  CUDA/12.1.1 

# export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
srun python -u tools/train.py data/sam_vit-b_OCRNet_SeaObject-1024x1024_A100.py  \
    --work-dir /cluster/projects/nn10004k/packages_install/seaobject_ocrneta100  \
    --launcher=slurm
