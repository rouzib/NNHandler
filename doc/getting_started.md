# Getting Started with NNHandler

This guide provides a quick introduction to using the NNHandler framework for training and using neural network models.

## Installation

The NNHandler project requires Python 3.10+ and PyTorch 1.10+. Clone the repository and install the dependencies:

```bash
git clone <your-repo-url> # Replace with your repository URL
cd nn_handler # Or your repository's root directory
pip install -r requirements.txt
```
*(It's recommended to use a virtual environment.)*

## Basic Usage: Training a Model

Here's a simple example of training a neural network model using NNHandler. This example runs in a single process. For multi-GPU training, see the note on DDP below.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.nn_handler import NNHandler # Assuming src layout

# 1. Define your Model and Loss
class MyModel(torch.nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.flatten(x, 1) # Flatten image data
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

loss_function = torch.nn.CrossEntropyLoss()

# 2. Prepare your data (Example with random data)
num_samples = 1000
input_features = 784 # 28*28
num_classes = 10
data = np.random.randn(num_samples, input_features)
labels = np.random.randint(0, num_classes, size=num_samples)
data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)
dataset = TensorDataset(data_tensor, labels_tensor)

# 3. Create NNHandler
handler = NNHandler(
    model_class=MyModel,
    # Pass model constructor arguments via model_kwargs
    model_kwargs={"input_size": input_features, "hidden_size": 128, "output_size": num_classes},
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_type=NNHandler.ModelType.CLASSIFICATION, # Specify task type
    logger_mode=NNHandler.LoggingMode.CONSOLE # Log to console
)

# 4. Set up Optimizer
handler.set_optimizer(
    optimizer_class=torch.optim.Adam,
    lr=1e-3
)

# 5. Set up Loss Function
handler.set_loss_fn(loss_function)

# 6. Set up Data Loader
handler.set_train_loader(dataset, batch_size=32, shuffle=True)
# Optional: Validation loader
# handler.set_val_loader(val_dataset, batch_size=32)

# 7. Train the model
print("Starting training...")
handler.train(
    epochs=5, # Train for 5 epochs
    use_amp=True, # Use Automatic Mixed Precision (if CUDA available)
    progress_bar=True # Show epoch progress bar
)
print("Training finished.")

# 8. Save the trained handler state
save_path = "models/my_model_handler_state.pth"
print(f"Saving handler state to {save_path}")
handler.save(save_path) # Saves model, optimizer, history, etc.
```

**Note on Distributed Training (DDP):**
If you have multiple GPUs and want to use DDP:
1.  Ensure your environment is configured for DDP (e.g., using `torchrun`).
2.  NNHandler will automatically detect the DDP environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`) and initialize the distributed process group.
3.  Run your script using `torchrun`:
    ```bash
    torchrun --standalone --nproc_per_node=<num_gpus> your_training_script.py
    ```
4.  NNHandler internally handles model wrapping (DDP), data distribution (DistributedSampler), and gradient/metric aggregation. See the [Distributed Training Guide](nn_handler/distributed.md) for details.

## Using a Trained Model (Loading State)

You can load the entire handler state to resume training or perform inference.

```python
import torch
from src.nn_handler import NNHandler # Assuming src layout
# Make sure MyModel class is defined or imported

# Path where the handler state was saved
load_path = "models/my_model_handler_state.pth"

print(f"Loading handler state from {load_path}")
# Load the entire state (model, optimizer, history, etc.)
# Device is typically set based on the current environment (DDP or single)
loaded_handler = NNHandler.load(
    path=load_path,
    device="cuda" if torch.cuda.is_available() else "cpu"
    # Add skip_optimizer=True, etc. if you only need parts of the state
)
print("Handler loaded.")

# Now you can:
# 1. Resume training:
#    loaded_handler.train(epochs=5) # Continues from the saved state

# 2. Or use the model for prediction:
loaded_handler.eval() # Set model to evaluation mode
model = loaded_handler.model # Access the underlying model (potentially DDP wrapped)

# Prepare some test data (matching the model's input)
test_data = torch.randn(10, 784).to(loaded_handler.device) # Example test data

with torch.no_grad():
    outputs = model(test_data)
    _, predicted_classes = torch.max(outputs, 1)

print(f"Predictions on test data: {predicted_classes.cpu().numpy()}")

# You can also use the handler's predict method (gathers results on Rank 0 in DDP)
# test_dataset = TensorDataset(test_data, torch.zeros(10)) # Dummy labels if needed
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=5)
# predictions_list = loaded_handler.predict(test_loader) # Returns list of batch outputs
# if predictions_list: # None on non-zero ranks in DDP
#    print(f"Predict method output (first batch shape): {predictions_list[0].shape}")
```

## Advanced Usage: Callbacks

Use callbacks to customize the training process (monitoring, saving, etc.).

```python
# (Continuing from the basic training example...)
from src.nn_handler.callbacks import ProgressMonitor, EarlyStopping, ModelCheckpoint
# from src.nn_handler.callbacks.visualisation import ImageReconstruction # Example

# Add callbacks BEFORE training
handler.add_callback(ProgressMonitor()) # Basic progress logging
handler.add_callback(EarlyStopping(monitor="val_loss", patience=3)) # Requires validation data
handler.add_callback(ModelCheckpoint(
    filepath="models/checkpoint_epoch{epoch:02d}_val_loss{val_loss:.2f}.pth", # Dynamic filename
    monitor="val_loss",
    save_best_only=True # Save only the best model based on val_loss
))

# Set up validation data if using validation-based callbacks
val_data = np.random.randn(200, input_features)
val_labels = np.random.randint(0, num_classes, size=200)
val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.long))
handler.set_val_loader(val_dataset, batch_size=32)

# Train with callbacks enabled
print("Starting training with callbacks...")
handler.train(epochs=20, validate_every=1) # Validate every epoch
```

## Advanced Usage: Custom Metrics

Track custom performance metrics during training.

```python
# (Continuing from the basic training example...)
import torch

# Define a custom metric function (e.g., accuracy for classification)
def accuracy(outputs, targets):
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return (correct / total) * 100.0 # Return accuracy percentage

# Add the metric to the handler BEFORE training
handler.add_metric("accuracy", accuracy)

# Train - the metric will be calculated and logged
# Make sure validation data is set if you want validation accuracy
handler.train(epochs=5, validate_every=1)

# Plot the metric history (Rank 0 only)
if handler._rank == 0: # Check if current process is Rank 0
    print("Plotting metrics...")
    handler.plot_metrics(save_path_prefix="plots/my_model_metrics") # Saves plots to files
```

## Next Steps

This guide covers the basics. Explore the detailed documentation for more advanced features and customization options:

*   [NNHandler Module Overview](nn_handler/README.md)
*   [NNHandler Class API](nn_handler/nn_handler.md)
*   [Distributed Training (DDP)](nn_handler/distributed.md)
*   [Callbacks System](nn_handler/callbacks/README.md)