#!/bin/bash
#SBATCH --time=0-00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=16G
#SBATCH --account=FILL ACCOUNT

module load gcc cuda/12.2 nccl/2.18.3 python/3.11

source ENV/bin/activate

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_FQDN=$(srun -N1 -n1 -w "$MASTER_HOST" bash -lc 'hostname -f')
export MASTER_ADDR="$MASTER_FQDN"
export MASTER_PORT=29500
export RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT}"

# Point NCCL at the default-route interface on the master
export NCCL_SOCKET_IFNAME=$(srun -N1 -n1 -w "$MASTER_HOST" bash -lc "ip -o -4 route show to default | awk '{print \$5}'")

srun --ntasks=${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 --gpus-per-task=${SLURM_GPUS_PER_NODE} \
  torchrun \
    --nnodes=${SLURM_JOB_NUM_NODES} \
    --nproc_per_node=${SLURM_GPUS_PER_NODE} \
    --node_rank=${SLURM_NODEID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${RDZV_ENDPOINT} \
    --rdzv_id=${SLURM_JOB_ID} \
    PYTHON_FILE.py

# or
# srun --ntasks=${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 --gpus-per-task=${SLURM_GPUS_PER_NODE} nn_handler_run PYTHON_FILE.py