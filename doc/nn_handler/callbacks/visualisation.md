# Visualization Callbacks

Visualization callbacks help in understanding model behavior during training by generating plots, images, or other visual outputs.

*(Note: The exact implementation details depend on your `callbacks/visualisation.py` file. This documentation assumes common examples like image reconstruction or sample generation.)*

## Common Use Cases

*   **Generative Models (VAEs, GANs, Diffusion):** Generate and save sample images periodically to visually assess generation quality.
*   **Autoencoders:** Show input images alongside their reconstructions.
*   **Segmentation/Detection Models:** Overlay predicted masks or bounding boxes on input images.
*   **Attention Mechanisms:** Plot attention maps.

## Key Considerations

*   **Dependencies:** These callbacks often require libraries like `matplotlib`, `PIL` (Pillow), or `torchvision`.
*   **Data:** They might need access to a fixed batch of validation or training data to generate consistent visualizations across epochs.
*   **Frequency:** Generating visualizations can be computationally expensive, so they are typically run less frequently (e.g., every N epochs).
*   **Saving:** Visualizations are usually saved to disk in a specified directory.
*   **DDP:** Visualization generation and saving should almost always be restricted to **Rank 0**.

## Example: `ImageReconstruction` (Hypothetical VAE/Autoencoder)

Generates and saves image reconstructions at regular intervals.

```python
from .base import Callback
import torch
import os
from torchvision.utils import save_image # Requires torchvision

class ImageReconstruction(Callback):
    def __init__(self, dataloader, num_images=8, every_n_epochs=5, save_dir="reconstructions"):
        super().__init__()
        self.dataloader = dataloader # Dataloader providing images for reconstruction
        self.num_images = num_images
        self.every_n_epochs = every_n_epochs
        self.save_dir = save_dir
        self.fixed_batch = None

    def on_train_begin(self, logs=None):
        # Prepare save directory (Rank 0 only)
        if self.handler._rank == 0:
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"Image reconstructions will be saved to: {self.save_dir}")
        
        # Get a fixed batch for consistent visualization (needs care with DDP)
        # Simplest: Rank 0 gets a batch, broadcasts it?
        # Or each rank uses its own first batch from the provided loader?
        # Let's assume Rank 0 gets and holds the batch for saving.
        if self.handler._rank == 0:
            try:
                data_iter = iter(self.dataloader)
                # Handle different batch structures (e.g., (image, label) or just image)
                batch_data = next(data_iter)
                if isinstance(batch_data, (list, tuple)):
                    self.fixed_batch = batch_data[0][:self.num_images].to(self.handler.device)
                elif isinstance(batch_data, torch.Tensor):
                    self.fixed_batch = batch_data[:self.num_images].to(self.handler.device)
                else:
                    print("Warning: ImageReconstruction couldn't understand batch format.")
                    self.fixed_batch = None
            except Exception as e:
                print(f"Warning: Could not get fixed batch for ImageReconstruction: {e}")
                self.fixed_batch = None

    @torch.no_grad()
    def on_epoch_end(self, epoch, logs=None):
        # Only visualize and save on Rank 0
        if self.handler._rank == 0 and (epoch + 1) % self.every_n_epochs == 0:
            if self.fixed_batch is None:
                return
                
            self.handler.eval() # Set model to eval mode
            model_module = self.handler.module # Use unwrapped model
            
            try:
                # Generate reconstruction - assumes model returns reconstruction
                # Adapt based on model's forward signature (e.g., VAE might return multiple things)
                output = model_module(self.fixed_batch)
                reconstruction = None
                if isinstance(output, torch.Tensor):
                    reconstruction = output 
                elif isinstance(output, (list, tuple)): # Common for VAEs (recon, mu, logvar)
                    reconstruction = output[0]
                
                if reconstruction is not None:
                    # Combine original and reconstructed images side-by-side
                    comparison = torch.cat([self.fixed_batch, reconstruction])
                    save_path = os.path.join(self.save_dir, f"reconstruction_epoch_{epoch+1:03d}.png")
                    
                    # Save using torchvision utility (adjust nrow, normalize as needed)
                    save_image(comparison.cpu(), save_path, nrow=self.num_images, normalize=True, scale_each=True)
                else:
                     print(f"Warning: ImageReconstruction couldn't get reconstruction output at epoch {epoch+1}.")

            except Exception as e:
                print(f"Error during image reconstruction visualization at epoch {epoch+1}: {e}")
            finally:
                # Ensure model is set back to train mode if handler was training
                # (NNHandler usually sets train mode at start of next epoch's training phase)
                pass 

    # No state needs saving for this example
```

**Usage:**

```python
# Assuming val_loader provides image data
vis_callback = ImageReconstruction(
    dataloader=handler.val_loader, # Use validation loader for images
    num_images=8, 
    every_n_epochs=10, 
    save_dir="./vae_reconstructions"
)
handler.add_callback(vis_callback)
```

## Example: `SampleGenerator` (Hypothetical GAN/Diffusion)

Generates and saves samples from a generative model.

```python
from .base import Callback
import torch
import os
from torchvision.utils import save_image

class SampleGenerator(Callback):
    def __init__(self, num_samples=16, every_n_epochs=5, save_dir="generated_samples", 
                 latent_dim=100, use_ema=True):
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs
        self.save_dir = save_dir
        self.latent_dim = latent_dim # Example for GANs/VAEs
        self.fixed_noise = None
        self.use_ema = use_ema # Whether to use EMA weights for generation

    def on_train_begin(self, logs=None):
        if self.handler._rank == 0:
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"Generated samples will be saved to: {self.save_dir}")
            # Create fixed noise for consistent generation (Rank 0 only)
            self.fixed_noise = torch.randn(self.num_samples, self.latent_dim, device=self.handler.device)

    @torch.no_grad()
    def on_epoch_end(self, epoch, logs=None):
        # Only generate and save on Rank 0
        if self.handler._rank == 0 and (epoch + 1) % self.every_n_epochs == 0:
            if self.fixed_noise is None:
                 # Generate new noise if fixed noise wasn't created (e.g., resuming)
                 noise = torch.randn(self.num_samples, self.latent_dim, device=self.handler.device)
            else:
                 noise = self.fixed_noise
            
            self.handler.eval() # Set handler to eval mode
            model_module = self.handler.module
            
            # Use EMA weights if configured
            ema_context = self.handler._ema.average_parameters() if (self.handler._ema and self.use_ema) else torch.no_grad()
            
            try:
                with ema_context:
                    # Generate samples - adapt based on model type
                    # GAN generator:
                    # samples = model_module.generator(noise)
                    # Diffusion model (using handler's sample method):
                    if self.handler._model_type == NNHandler.ModelType.SCORE_BASED:
                         samples = self.handler.sample(shape=(self.num_samples, C, H, W), steps=100, apply_ema=self.use_ema, bar=False) # Get C, H, W from config
                    # VAE decoder:
                    # elif hasattr(model_module, 'decode'): 
                    #    samples = model_module.decode(noise)
                    else: 
                        # Assume standard forward generates samples from noise (e.g., basic GAN)
                        samples = model_module(noise)
                
                if samples is not None:
                    save_path = os.path.join(self.save_dir, f"samples_epoch_{epoch+1:03d}.png")
                    save_image(samples.cpu(), save_path, nrow=int(self.num_samples**0.5), normalize=True)
                else:
                    print(f"Warning: SampleGenerator couldn't generate samples at epoch {epoch+1}.")

            except Exception as e:
                 print(f"Error during sample generation visualization at epoch {epoch+1}: {e}")
            finally:
                 pass # Model mode reset handled by handler
```

**Usage:**

```python
gen_callback = SampleGenerator(
    num_samples=16, 
    every_n_epochs=5, 
    save_dir="./gan_samples",
    latent_dim=128 # Adjust as needed for your model
)
handler.add_callback(gen_callback)
```

*(Please adapt the examples above based on the actual code in your `visualisation.py` file.)*