#!/bin/bash
#SBATCH --time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=64G
#SBATCH --account=FILL ACCOUNT

module load gcc cuda/12.2 nccl/2.18.3 python/3.11

source ENV/bin/activate

srun torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_PER_NODE PYTHON_FILE.py

# or
# srun nn_handler_run PYTHON_FILE.py