#!/bin/bash
#SBATCH --time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gpu-bind=per_task:1
#SBATCH --account=aip-lplevass
#SBATCH --chdir=/home/r/rouzib/links/scratch
#SBATCH --output=/home/r/rouzib/links/scratch/ddp_%j.out
#SBATCH --mail-user=npayot@gmail.com
#SBATCH --mail-type=BEGIN

module load gcc cuda/12.2 nccl/2.18.3 python/3.11

export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1

source /home/r/rouzib/links/scratch/nn_handler/bin/activate

srun python /home/r/rouzib/links/scratch/NNHandler/tests/ddp_single_node.py