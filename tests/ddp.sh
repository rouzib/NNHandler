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
export PYTHONUNBUFFERED=1

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_FQDN=$(srun -N1 -n1 -w "$MASTER_HOST" bash -lc 'hostname -f')
export MASTER_ADDR="$MASTER_FQDN"
export MASTER_PORT=29500
export RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT}"

# Point NCCL at the default-route interface on the master (often ib0/eno1/eth0)
export NCCL_SOCKET_IFNAME=$(srun -N1 -n1 -w "$MASTER_HOST" bash -lc "ip -o -4 route show to default | awk '{print \$5}'")
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1

echo ${RDZV_ENDPOINT}
echo ${NNCL_NCCL_SOCKET_IFNAME}

srun --ntasks=${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 --gpus-per-task=${SLURM_GPUS_PER_NODE} \
  torchrun \
    --nnodes=${SLURM_JOB_NUM_NODES} \
    --nproc_per_node=${SLURM_GPUS_PER_NODE} \
    --node_rank=${SLURM_NODEID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${RDZV_ENDPOINT} \
    --rdzv_id=${SLURM_JOB_ID} \
    /home/r/rouzib/links/scratch/NNHandler/tests/ddp_single_node.py
