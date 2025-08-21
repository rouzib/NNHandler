#!/bin/bash
#SBATCH --time=0-00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=16G
#SBATCH --account=aip-lplevass
#SBATCH --chdir=/home/r/rouzib/links/scratch
#SBATCH --output=/home/r/rouzib/links/scratch/ddp_%j.out
#SBATCH --mail-user=npayot@gmail.com
#SBATCH --mail-type=BEGIN

module load gcc cuda/12.2 nccl/2.18.3 python/3.11

source /home/r/rouzib/links/scratch/nn_handler/bin/activate

srun torchrun --nnodes=2 --nproc_per_node=4 /home/r/rouzib/links/scratch/NNHandler/tests/ddp_single_node.py