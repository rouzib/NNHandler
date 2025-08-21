#!/bin/bash
#SBATCH --time=0-00:05:00
#SBATCH --nodes=4             # number of GPU used during training
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1    # number of workers for the data_loaders
#SBATCH --mem-per-cpu=64G
#SBATCH --account=FILL ACCOUNT


module load gcc cuda/12.2 nccl/2.18.3 python/3.11

source ENV/bin/activate

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$((13000 + (${SLURM_JOB_ID} % 20000))) # Pick a TCP port that is very unlikely to collide
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"

srun python PYTHON_FILE.py