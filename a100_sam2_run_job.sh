#!/bin/bash
#SBATCH -p a100 # a100 accel
#SBATCH --account=nn10004k
#SBATCH --output=5unetFCN_SeaObject-1024x1024_MultiA100_3in1_4p100_40K.txt
#SBATCH --error=5error_4a100_UnetFCN_OASIs_3in1_40K.txt
#SBATCH --job-name=unetfcn
#SBATCH --nodes=1  
#SBATCH --gpus=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4-15:15:00

## Set up job environment:
set -o errexit  # Exit the script on any error

set -o nounset  # Treat any unset variables as an error

# module --quiet purge  # Reset the modules to the system default
module purge --force
source /cluster/projects/nn10004k/packages_install/torch_cu121/bin/activate
module load  CUDA/12.1.1 

# export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
srun python -u tools/train.py data/1_Unet_FCN_SeaObject-1008x1008_MultiA100_3in1.py  \
    --work-dir /cluster/projects/nn10004k/packages_install/Unet_FCN_OASIs_3in1_40K  \
    --launcher=slurm
