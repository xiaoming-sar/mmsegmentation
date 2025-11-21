#!/bin/bash

#!/bin/bash
#SBATCH -p a100
#SBATCH --account=nn10004k
#SBATCH --output=sam2.1small_segformer_4a100_20000_MaSTr1325.txt
#SBATCH --error=error_MaSTr1325.txt
#SBATCH --job-name=sam2.1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=5
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
srun python -u tools/train.py data/hiera-sam2-small_SegSegformer_MaSTr1325-1008x1008_MultiA100.py  \
    --work-dir /cluster/projects/nn10004k/packages_install/sam2_MaSTr1325  \
    --launcher=slurm