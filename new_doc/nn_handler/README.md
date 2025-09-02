# NNHandler Module

This directory contains the core components of the `nn_handler` framework for PyTorch model training and management.

## Overview

The `nn_handler` module simplifies the process of working with PyTorch models by providing a high-level interface (`NNHandler`) that manages the complexities of training loops, distributed training, device handling, state saving/loading, and more.

## Key Components

*   **`NNHandler` Class ([nn_handler.md](nn_handler.md))**:
    *   The central class for managing model lifecycle, training, evaluation, and prediction.
    *   Provides methods for setting up the model, optimizer, scheduler, loss function, and data loaders.
    *   Handles the training loop with support for advanced features.
    *   Implemented in `nn_handler_distributed.py`.

*   **Distributed Training ([distributed.md](distributed.md))**:
    *   Built-in support for multi-GPU and multi-node training using PyTorch Distributed Data Parallel (DDP).
    *   Automatic detection of DDP environment and setup.
    *   Handles data distribution, gradient synchronization, and metric aggregation.

*   **AutoSaver ([autosaver.md](autosaver.md))**:
    *   Functionality to automatically save model checkpoints (full handler state) at specified intervals during training.
    *   Helps prevent loss of progress due to interruptions.
    *   Configured via the `NNHandler.auto_save()` method.

*   **Sampler Integration ([sampler.md](sampler.md))**:
    *   Support for using custom sampling algorithms with generative models.
    *   Requires implementing a class inheriting from the `Sampler` abstract base class (defined in `sampler.py`).
    *   Configured via `NNHandler.set_sampler()` and used via `NNHandler.get_samples()`. (Distinct from SDE-based sampling).

*   **Callbacks Subsystem ([callbacks/README.md](callbacks/README.md))**:
    *   A flexible system to inject custom logic at various points in the training lifecycle (e.g., start/end of epoch/batch, saving checkpoints, logging metrics, early stopping).
    *   Callbacks inherit from the base `Callback` class.

*   **Utilities ([utils/README.md](utils/README.md))**:
    *   A collection of utility functions and classes that support the core functionality of the framework.
    *   Includes DDP utilities for working with PyTorch's Distributed Data Parallel functionality.
    *   Can be used both within the framework and independently in your own PyTorch code.

## Further Reading

*   [Getting Started Guide](../getting_started.md)
*   [NNHandler Class API Documentation](nn_handler.md)
*   [Distributed Training (DDP) Guide](distributed.md)
*   [AutoSaver Feature Details](autosaver.md)
*   [Sampler Integration Guide](sampler.md)
*   [Callbacks System Overview](callbacks/README.md)
*   [Utilities Documentation](utils/README.md)