#!/bin/bash
#SBATCH -p accel
#SBATCH --account=nn10004k
#SBATCH --output=ouput_adamW_focalloss_4GPU.txt
#SBATCH --error=error.txt
#SBATCH --job-name=sam
#SBATCH --gres=gpu:p100:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=5G
#SBATCH --time=15:15:00

export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
srun python -u tools/train.py data/sam_vit-b_OCRNet_SeaObject-1024x1024_MultiGPU.py --launcher=slurm
