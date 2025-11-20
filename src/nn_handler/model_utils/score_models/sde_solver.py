import contextlib
import logging
from typing import Tuple, Optional, Callable

import torch
from torch import Tensor
from tqdm.auto import tqdm

from .patch_score import patch_score_vectorized
from .schedules import get_t_schedule
from ...utils import ModelType


class SdeSolver:
    def __init__(self, nn_handler: 'NNHandler'):
        self.nn_handler = nn_handler
        self.device = nn_handler.device
        self.sde = nn_handler.sde

    def perform_checks(self, patch_size: Optional[int] = None, stride: Optional[int] = None, ):
        """
        Performs necessary validation checks and determines the patch diffusion mode based
        on provided arguments and internal state. This function ensures that required
        conditions for sampling are met and validates the patch_size and stride arguments.

        Args:
            patch_size: Optional; The size of the patch to be used for diffusion. Must
                be specified together with `stride` or not at all.
            stride: Optional; The stride to be used for patch-based diffusion. Must
                be specified together with `patch_size` or not at all.

        Returns:
            bool: True if patch diffusion mode should be enabled, otherwise False.
        """
        if self.nn_handler._model_type != ModelType.SCORE_BASED:
            raise NotImplementedError("Sampling is only supported for SCORE_BASED models.")
        if self.sde is None: raise RuntimeError("SDE must be set for sampling.")
        if self.nn_handler._model is None: raise RuntimeError("Model must be set for sampling.")

        self.nn_handler.eval(activate=True, log=False)  # Set model to evaluation mode

        if (patch_size is None) != (stride is None):
            raise ValueError("Both patch_size and stride must be specified together (either both set, or both None).")
        elif (patch_size is not None) and (stride is not None):
            patch_diffusion_mode = True
        else:
            patch_diffusion_mode = False

        return patch_diffusion_mode

    def step(self, t, x, dt, condition: Optional[list] = None, patch_size: int = None, stride: int = None,
             patch_chunk: int = None, likelihood_score_fn: Optional[Callable] = None, guidance_factor: float = 1.0):
        """
        Performs a single iteration of the process defined by the `dx` function. This
        method updates the state of the system based on the provided parameters and
        random noise. The process incorporates optional conditional inputs and other
        configurable parameters to guide the iteration.

        Args:
            t (Tensor): The current time step of the process.
            x (Tensor): The current state of the system.
            dt (Tensor): The timestep increment.
            condition (Optional[list]): Optional list of conditional inputs to guide
                the update. Defaults to None.
            patch_size (int): Size of the patches to consider when processing the
                system state. Defaults to None.
            stride (int): Stride length for the patching process. Defaults to None.
            patch_chunk (int): Number of patches processed in one chunk. Defaults to
                None.
            likelihood_score_fn (Optional[Callable]): Optional scoring function for
                computing likelihoods. Defaults to None.
            guidance_factor (float): Scaling factor for conditional guidance. Defaults
                to 1.0.

        Returns:
            Tensor: Updated state of the system after applying the process.
        """
        dw = torch.randn_like(x) * torch.sqrt(dt.abs())
        return self.dx(t, x, dt, dw, condition, patch_size, stride, patch_chunk, likelihood_score_fn, guidance_factor)

    def corrector_step(self, t, x, snr, condition: Optional[list] = None, patch_size: int = None, stride: int = None,
                       patch_chunk: int = None, likelihood_score_fn: Optional[Callable] = None,
                       guidance_factor: float = 1.0):
        """
        Performs a corrector step in the sampling process by applying a score-based update
        to the input tensor. This step adjusts the current state using the score function
        and adds noise to maintain stochasticity.

        Args:
            t: Current timestep in the diffusion process, used to scale the noise and
                determine the score.
            x: Input tensor representing the current state of the data to be updated.
            snr: Signal-to-noise ratio controlling the balance between signal and noise
                during sampling.
            condition: Optional list representing additional conditioning information for
                the score function.
            patch_size: Optional integer representing the size of each patch if patching
                is used in conditioning.
            stride: Optional integer specifying the stride size for overlapping patches.
            patch_chunk: Optional integer defining the chunk size if patches are evaluated
                in splits.
            likelihood_score_fn: Optional callable function to compute the likelihood of
                the score.
            guidance_factor: Floating-point value controlling the influence of the
                conditioning on the score function.
        """
        _, *D = x.shape
        z = torch.randn_like(x)
        epsilon = (snr * self.sde.sigma(t).view(-1, *[1] * len(D))) ** 2
        return x + epsilon * self.get_score(t, x, condition, patch_size, stride, patch_chunk, likelihood_score_fn,
                                            guidance_factor) + z * torch.sqrt(2 * epsilon)

    def get_score(self, t, x, condition: Optional[list] = None, patch_size: int = None, stride: int = None,
                  patch_chunk: int = None, likelihood_score_fn: Optional[Callable] = None,
                  guidance_factor: float = 1.0):
        """
        Calculates a combined score based on model-driven and likelihood-based evaluations for a given input. The function
        supports scoring in a patchwise manner if patch size and stride are provided, or scoring using the entire input.
        Additionally, guidance from a likelihood score can be scaled and incorporated into the overall score.

        Args:
            t: Current timestep or condition indicator for scoring.
            x: Input data to be scored.
            condition: Optional. A list of conditioning values or additional parameters used during scoring. Default is an
                empty list.
            patch_size: Optional. Size of patches to divide the input 'x' for patchwise scoring. Default is None.
            stride: Optional. Stride value for moving the window during patchwise scoring. Default is None.
            patch_chunk: Optional. Number of patches to process at once during patchwise scoring. Default is None.
            likelihood_score_fn: Optional. A callable function that computes likelihood scores for the input 'x'. If not
                provided, default likelihood score is zero. Default is None.
            guidance_factor: Coefficient for scaling the likelihood score contribution to the overall score. Default is 1.0.

        Returns:
            Tensor: Combined score that incorporates model-driven and likelihood-based evaluations.
        """
        if condition is None: condition = []
        is_patch_diffusion = self.perform_checks(patch_size, stride)

        if is_patch_diffusion:
            score = patch_score_vectorized(self.nn_handler, t, x, patch_size, stride, patch_chunk, *condition)
        else:
            score = self.nn_handler.score(t, x, *condition)

        if likelihood_score_fn is not None:
            if is_patch_diffusion:
                likelihood_score = likelihood_score_fn(t, x, score)
            else:
                likelihood_score = likelihood_score_fn(t, x)
        else:
            likelihood_score = torch.zeros_like(x)
        return score + guidance_factor * likelihood_score

    def drift(self, t: Tensor, x: Tensor, condition: Optional[list] = None, patch_size: int = None,
              stride: int = None, patch_chunk: int = None, likelihood_score_fn: Optional[Callable] = None,
              guidance_factor: float = 1.0):
        """
        Computes the drift term for the stochastic differential equation (SDE) while
        incorporating the score-based element. The drift term represents the deterministic
        component of the SDE, to which noise is added.

        Args:
            t: Tensor representing the time variable in the SDE.
            x: Tensor which is the input state at time `t`.
            condition: Optional list providing conditional inputs that may influence the
                score and other calculations.
            patch_size: Specifies the size of the patches, possibly for segmenting `x`
                during score computation.
            stride: Determines the step size or overlap between patches during
                segmentation.
            patch_chunk: Indicates the number of patches to process at a time, useful
                for batch operations to optimize computations.
            likelihood_score_fn: Optional callable function used to compute a likelihood
                score for the data, influencing the score computation.
            guidance_factor: Float value used to scale or adjust the influence of
                conditional guidance in the score computation.

        Returns:
            Tensor representing the computed drift term adjusted by the score and
            diffusion components of the SDE.
        """
        f = self.sde.drift(t, x)
        g = self.sde.diffusion(t, x)
        s = self.get_score(t, x, condition, patch_size, stride, patch_chunk, likelihood_score_fn, guidance_factor)
        return f - g ** 2 * s

    def dx(self, t: Tensor, x: Tensor, dt, dw=None, condition: Optional[list] = None,
           patch_size: int = None, stride: int = None, patch_chunk: int = None,
           likelihood_score_fn: Optional[Callable] = None, guidance_factor: float = 1.0):
        """
        Calculates the stochastic differential equation (SDE) update step for the given variables.
        This function computes the increment for a process using the drift and diffusion terms
        along with the provided inputs. The parameters for patch processing and likelihood-guidance
        can also be customized.

        Args:
            t: Current time step as a tensor.
            x: Current state of the process as a tensor.
            dt: Time increment (step size).
            dw: Optional stochastic increment; if None, generated internally as a tensor.
            condition: Optional list containing input conditions for calculating the drift.
            patch_size: Optional integer specifying the size of patches for processing.
            stride: Optional integer specifying the stride for patch processing.
            patch_chunk: Optional integer specifying the chunk size for patch computation.
            likelihood_score_fn: Optional callable function to compute likelihood-based score.
            guidance_factor: Float value controlling the impact of guidance parameters.

        Returns:
            Tensor representing the updated state of the process after applying the SDE step.
        """
        if dw is None:
            dw = torch.randn_like(x) * torch.sqrt(dt.abs())
        return self.drift(t, x, condition, patch_size, stride, patch_chunk, likelihood_score_fn,
                          guidance_factor) * dt + self.sde.diffusion(t, x) * dw

    @torch.no_grad()
    def solve(self, shape: Tuple[int, ...], steps: int, corrector_steps: int = 0, condition: Optional[list] = None,
              likelihood_score_fn: Optional[Callable] = None, guidance_factor: float = 1.,
              apply_ema: bool = True, bar: bool = True, stop_on_NaN: bool = True, patch_size: int = None,
              stride: int = None, patch_chunk: int = None, corrector_snr: float = 0.1,
              on_step: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None):
        """
        Solves the given stochastic differential equation (SDE) to generate samples using a specified number of
        steps and optional guidance, likelihood scoring, and sampling corrections. Includes options for handling
        NaNs, applying exponential moving averages (EMA), and operating in a patch-based mode.

        Args:
            shape:
                The shape of the samples to generate, where the first dimension is the batch size and the
                remaining dimensions correspond to the data shape.
            steps:
                The number of timesteps to perform during the sampling process.
            corrector_steps:
                The number of corrector steps to apply per timestep for fine adjustments
                (default is 0, which means no corrector steps are applied).
            condition:
                A list of guidance values or conditions to be applied during sampling. Defaults to None,
                which results in an unconditional sampling process.
            likelihood_score_fn:
                An optional function that computes likelihood scores for the data. If None, a zero
                likelihood scoring function is used.
            guidance_factor:
                A float value representing the weighting factor for guidance. Default is 1, with higher
                values increasing the effect of guidance on the results.
            apply_ema:
                If True, applies exponential moving average (EMA) parameters during sampling. Default is True.
            bar:
                If True, displays a progress bar during sampling. Default is True.
            stop_on_NaN:
                If True, the process halts immediately upon the detection of NaN or Inf values in the samples.
                Default is True.
            patch_size:
                The size of individual patches to process in patch-based mode. If None, standard mode
                (non-patch-based) is used.
            stride:
                The stride size for sliding patches in patch-based mode. If None, standard mode
                (non-patch-based) is used.
            patch_chunk:
                Number of patches to process at each time step if patch-based mode is enabled.
            corrector_snr:
                Signal-to-noise ratio used during corrector steps. Default is 0.1.
            on_step:
                Callback function to be executed after each step. Takes i, t, and x as an input.

        Returns:
            torch.Tensor:
                The generated samples with the specified shape.
        """
        patch_diffusion_mode = self.perform_checks(patch_size, stride)

        B, *D = shape  # Batch size and data dimensions
        sampling_from = "prior" if likelihood_score_fn is None else "posterior"
        self.nn_handler.log(f"Starting sampling (Rank 0): Shape={shape}, Steps={steps}, Source='{sampling_from}'",
                            logging.INFO)

        # Define a zero likelihood function if none provided
        if likelihood_score_fn is None:
            def zero_likelihood_score(t, x, patch_score=None): return torch.zeros_like(x)

            likelihood_score_fn = zero_likelihood_score

        if condition is None: condition = []  # Ensure condition is a list

        # Initial sample from prior distribution (on the correct device)
        try:
            x = self.sde.prior(D).sample([B]).to(self.device)
        except Exception as e:
            self.nn_handler.raise_error(RuntimeError, f"Rank 0 failed to sample from SDE prior: {e}", e)

        ema = self.nn_handler.ema
        ema_context = ema.average_parameters() if (ema and apply_ema) else contextlib.nullcontext()

        pbar_iterator = range(steps)
        if bar:
            pbar_iterator = tqdm(pbar_iterator, desc=f"Sampling ({sampling_from})", dynamic_ncols=True)

        t_schedule = get_t_schedule(self.sde, steps, self.device)

        with ema_context:
            for i in pbar_iterator:
                t_current = t_schedule[i]
                t_next = t_schedule[i + 1]
                step_dt = t_next - t_current

                if t_current.item() <= self.sde.epsilon:
                    self.nn_handler.warn(
                        f"Reached time epsilon ({self.sde.epsilon:.4f}) early at step {i}. Stopping sampling.",
                        RuntimeWarning)
                    break  # Stop if time goes below epsilon

                x = x + self.step(t_current, x, step_dt, condition, patch_size, stride, patch_chunk,
                                  likelihood_score_fn, guidance_factor)
                for j in range(corrector_steps):
                    x = self.corrector_step(t_current, x, corrector_snr, condition, patch_size, stride, patch_chunk)

                if on_step is not None:
                    on_step(i, t_current, x)

                # Check for numerical issues
                if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                    error_msg = f"NaN or Inf detected in sampling at step {i + 1} (t={t_current.item():.4f}). Stopping."
                    if stop_on_NaN:
                        self.nn_handler.raise_error(RuntimeError, error_msg)
                    else:
                        self.nn_handler.warn(error_msg, RuntimeWarning)

                # Update progress bar postfix (rank 0)
                if bar:
                    postfix_dict = {"t": f"{t_current.item():.3f}",
                                    "sigma": f"{self.sde.sigma(t_current).item():.2e}",
                                    "|x|": f"{x.abs().mean().item():.2e}"}
                    pbar_iterator.set_postfix(postfix_dict, refresh=False)

        return x
