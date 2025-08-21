# Distributed Training (DDP) with NNHandler

NNHandler provides seamless integration with PyTorch's Distributed Data Parallel (DDP) for efficient multi-GPU and multi-node training.

## Overview

DDP allows scaling training by running a copy of the script on multiple processes (ranks), typically one per GPU. Each process works on a different shard of the data, and gradients are synchronized across processes during the backward pass. NNHandler automates many aspects of DDP integration.

## Enabling DDP

NNHandler determines whether to use DDP based on the `use_distributed` argument during initialization and environment variables:

1.  **`use_distributed=True`**: Explicitly enables DDP. Requires a valid DDP environment.
2.  **`use_distributed=False`**: Explicitly disables DDP, even if environment variables are present.
3.  **`use_distributed=None` (Default)**: Auto-detects DDP. NNHandler checks for standard PyTorch DDP environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`) or Slurm variables (`SLURM_PROCID`, `SLURM_LOCALID`, `SLURM_NTASKS`). If valid variables indicating `WORLD_SIZE > 1` are found and `torch.distributed` is available, DDP is enabled.

**Running a DDP Script:**
You typically launch DDP training scripts using `torchrun` (recommended) or `torch.distributed.launch` (legacy):

```bash
# Example: Run on localhost with 4 GPUs
torchrun --standalone --nproc_per_node=4 your_training_script.py

# Example: Multi-node setup (requires proper environment variables like MASTER_ADDR, MASTER_PORT)
# Node 0:
# torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=<node0_ip> --master_port=12345 your_training_script.py
# Node 1:
# torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=<node0_ip> --master_port=12345 your_training_script.py
```

Slurm environments often configure the necessary variables automatically when using `srun`. NNHandler also attempts to determine `MASTER_ADDR` from Slurm variables if it's not explicitly set.

For example:
```bash
#!/bin/bash
#SBATCH --time=0-00:02:00
#SBATCH --ntasks=2           # number of GPU used during training
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4    # number of workers for the data_loaders
#SBATCH --mem-per-cpu=8G
#SBATCH --account=ACCOUNT
#SBATCH --chdir=/home/User/projects/ACCOUNT/User/NNHandler
#SBATCH --output=/home/User/projects/ACCOUNT/User/NNHandler/training_%j.out

module load gcc cuda/12.2 nccl/2.18.3 python/3.11  # be sure to import cuda and nccl if using multiple GPUs.

source /home/User/ENV/bin/activate

srun /home/User/ENV/bin/python training.py
```

## SLURM Job Script Templates

NNHandler provides several SLURM job script templates in the `doc` directory to help you run distributed training on HPC clusters:

1. **Single Node, Multiple GPUs** ([single_node_multiple_gpu_slurm_job.sh](../single_node_multiple_gpu_slurm_job.sh)):
   - For training on a single node with multiple GPUs
   - Uses `torchrun` to launch the distributed training
   - Example: 1 node with 4 GPUs

2. **Multiple Nodes, Single GPU per Node** ([multiple_nodes_single_gpu_slurm_job.sh](../multiple_nodes_single_gpu_slurm_job.sh)):
   - For training across multiple nodes, each with a single GPU
   - Uses direct `srun` to launch the Python script
   - Example: 4 nodes with 1 GPU each

3. **Multiple Nodes, Multiple GPUs per Node** ([multiple_nodes_multiple_gpu_slurm_job.sh](../multiple_nodes_multiple_gpu_slurm_job.sh)):
   - For training across multiple nodes, each with multiple GPUs
   - Uses `torchrun` with detailed configuration for distributed training
   - Example: 2 nodes with 4 GPUs each
   - Includes NCCL configuration for optimal network interface

To use these templates:
1. Copy the appropriate template to your project directory
2. Modify the SLURM parameters (time, nodes, GPUs, account, etc.) as needed
3. Update the Python script path and environment settings
4. Submit the job with `sbatch your_job_script.sh`

## How NNHandler Manages DDP

When DDP is enabled, NNHandler handles the following automatically:

1.  **Initialization**: Calls `torch.distributed.init_process_group` with the appropriate backend (`nccl` for GPU, `gloo` for CPU) based on detected environment variables and device availability.
2.  **Device Assignment**: Assigns the correct device (`cuda:LOCAL_RANK` or `cpu`) to each process.
3.  **Model Wrapping**: Wraps the instantiated model with `torch.nn.parallel.DistributedDataParallel`. The `find_unused_parameters` argument for DDP can be controlled via the environment variable `DDP_FIND_UNUSED_PARAMETERS` (defaults to `False`).
4.  **Data Sharding**: When `set_train_loader` or `set_val_loader` is called, the provided `Dataset` is automatically wrapped with `torch.utils.data.DistributedSampler`. This ensures each rank processes a unique, non-overlapping subset of the data. The sampler's epoch is automatically set at the beginning of each training epoch for correct shuffling behavior.
5.  **Gradient Synchronization**: DDP handles gradient averaging across ranks during the `loss.backward()` call within the training loop.
6.  **Metric & Loss Aggregation**: During training and validation:
    *   Each rank calculates loss and metrics on its local batch.
    *   NNHandler aggregates these values across all ranks (typically using an average reduction via `dist.all_reduce`).
    *   Aggregated results are stored and logged only on Rank 0.
7.  **State Saving/Loading**:
    *   `handler.save()`: Only Rank 0 performs the actual file write, ensuring checkpoint integrity. Barriers synchronize ranks. The unwrapped model's `state_dict` is saved.
    *   `handler.load()`: All ranks load the checkpoint file, mapping tensors to their assigned local device. Model weight keys are adjusted automatically if loading a DDP-saved checkpoint into a non-DDP model or vice-versa.
8.  **Logging & Plotting**: Output like logging messages, progress bars, and plots are restricted to Rank 0 to avoid redundant output.
9.  **Sampling/Prediction**: Methods like `handler.sample()` and `handler.log_likelihood()` typically execute only on Rank 0. `handler.predict()` runs inference on all ranks but gathers the results onto Rank 0.
10. **Synchronization**: Barriers (`dist.barrier()`) are used at critical points (e.g., after init, before/after gathering results, before/after saving) to ensure processes stay synchronized.

## Rank 0 Awareness

Many operations are intentionally restricted to Rank 0:

*   Writing log files.
*   Displaying progress bars (`tqdm`).
*   Saving checkpoints (`handler.save`).
*   Saving plots (`handler.plot_losses`, `handler.plot_metrics`).
*   Storing aggregated history (`handler.train_losses`, etc.).
*   Returning aggregated results from methods like `predict`.

Callbacks intended for DDP should often include an internal check (`if self.handler._rank == 0: ...`) if they perform I/O operations or actions that only need to happen once per step/epoch globally.

## Considerations for DDP

*   **Batch Size**: The `batch_size` set in `set_train_loader` or `set_val_loader` is the *per-process* batch size. The total effective batch size across all GPUs is `batch_size * world_size`.
*   **Learning Rate Scaling**: You might need to adjust the learning rate when scaling up the number of GPUs (and thus the effective batch size). Linear scaling (`new_lr = base_lr * world_size`) is a common starting point, but may require tuning.
*   **DistributedSampler**: Ensure validation or prediction `DataLoader`s used in DDP mode also use `DistributedSampler`, typically with `shuffle=False` if you need the results gathered on Rank 0 to be in the original dataset order.
*   **`find_unused_parameters`**: If your model has parameters that do not receive gradients during the forward pass (e.g., in certain conditional execution paths), DDP might hang during the backward pass. Setting the environment variable `DDP_FIND_UNUSED_PARAMETERS=true` before launching your script can resolve this, although it adds some overhead. NNHandler reads this environment variable.
*   **State Loading**: When loading a checkpoint using `handler.load()`, ensure the script is launched with the same DDP configuration (number of processes) as the training run that produced the checkpoint, unless you are intentionally loading only parts of the state (e.g., model weights) into a different setup.
*   **Non-Tensor Data**: Gathering non-tensor data across ranks (e.g., lists of strings or complex objects in prediction) uses `dist.gather_object`, which can be less efficient and consume more memory on Rank 0 compared to tensor operations like `dist.gather` or `dist.all_gather`.

## Standalone DDP Utilities

If you need to use DDP functionality independently of the `NNHandler` class, the framework provides a set of utilities in the `nn_handler.utils` package:

*   **DDP Core Utilities** (`nn_handler.utils.ddp`): Functions for initializing and managing DDP processes, resolving devices, and determining whether to use distributed training.
*   **DDP Decorators** (`nn_handler.utils.ddp_decorators`): Decorators and classes for executing functions in parallel across multiple GPUs, with robust error handling and synchronization.

For detailed documentation on these utilities, see the [DDP Utilities Documentation](utils/ddp_utils.md).
