Neural Network Handler (nn_handler)
===================================

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/python-3.10%2B-blue.svg
   :target: https://www.python.org/
   :alt: Python Version

.. image:: https://img.shields.io/badge/pytorch-1.10%2B-orange.svg
   :target: https://pytorch.org/
   :alt: PyTorch Version

The ``nn_handler`` repository provides a comprehensive and flexible Python framework designed to streamline the development, training, evaluation, and management of PyTorch neural network models. It aims to abstract away boilerplate code, allowing researchers and developers to focus on model architecture and experimentation.

NNHandler offers a unified interface supporting:

* Standard training and validation loops.
* Advanced features like Automatic Mixed Precision (AMP), gradient accumulation, and Exponential Moving Average (EMA).
* Seamless integration with **Distributed Data Parallel (DDP)** for multi-GPU and multi-node training.
* A rich, extensible **callback system** for monitoring, checkpointing, visualization, and custom logic.
* Built-in support for **generative models**, including score-based models (SDEs) and custom samplers.
* Comprehensive **model saving and loading**, including full training state resumption.
* Integrated logging and metric tracking with plotting capabilities.
* Support for ``torch.compile`` for potential performance boosts.

Installation
-----------

From GitHub:

.. code-block:: bash

    pip install git+https://github.com/rouzib/NNHandler.git

For development:

.. code-block:: bash

    git clone https://github.com/rouzib/NNHandler.git
    cd NNHandler
    pip install -e .

For full functionality:

.. code-block:: bash

    pip install NNHandler[full]

Documentation
------------

For detailed documentation, please visit the GitHub repository: https://github.com/rouzib/NNHandler

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.