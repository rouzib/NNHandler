#!/bin/bash
#SBATCH --time=0-00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # one torchrun per node
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

# --- make prints show up immediately ---
export PYTHONUNBUFFERED=1

# --- pick the first allocated node as rendezvous host ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export RDZV_ENDPOINT="$MASTER_ADDR:$MASTER_PORT"

# --- (optional but helpful) NCCL sanity flags ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1

# Launch one torchrun per node; each will spawn 4 workers (one per GPU)
srun --nodes=${SLURM_JOB_NUM_NODES} --ntasks=${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 \
     --gpus-per-task=${SLURM_GPUS_PER_NODE} \
     torchrun \
       --nnodes=${SLURM_JOB_NUM_NODES} \
       --nproc_per_node=${SLURM_GPUS_PER_NODE} \
       --rdzv_backend=c10d \
       --rdzv_endpoint=${RDZV_ENDPOINT} \
       --rdzv_id=${SLURM_JOB_ID} \
       /home/r/rouzib/links/scratch/NNHandler/tests/ddp_single_node.py
