# -*- coding: utf-8 -*-
"""
 @File    : scheduling_ddpm.py
 @Time    : 2023/3/30 16:32
 @Author  : Ling-An Zeng
 @Email   : linganzeng@gmail.com
 @Software: PyCharm
"""
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers import DDPMScheduler


# from https://github.com/huggingface/diffusers/blob/v0.14.0/src/diffusers/schedulers/scheduling_ddpm.py#L76
class MyDDPMScheduler(DDPMScheduler):
    def __init__(
            self,
            num_train_timesteps: int = 1000,
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            beta_schedule: str = "linear",
            trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
            variance_type: str = "fixed_small",
            clip_sample: bool = True,
            prediction_type: str = "epsilon",
            clip_sample_range: Optional[float] = 1.0,
    ):
        super(MyDDPMScheduler, self).__init__(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            trained_betas,
            variance_type,
            clip_sample,
            prediction_type,
            clip_sample_range=clip_sample_range
        )

    # only for training
    def get_predicted_mean_variance(self, model_output, sample, timestep, original_sample):

        noise_pred, predicted_variance = torch.split(model_output, sample.shape[2], dim=2)
        noise_pred = noise_pred.detach()

        t = timestep
        # num_inference_steps = self.num_inference_steps if self.num_inference_steps else self.config.num_train_timesteps
        # prev_t = timestep - self.config.num_train_timesteps // num_inference_steps
        prev_t = t - 1

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t][..., None, None]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t]
        alpha_prod_t_prev[prev_t < 0] = self.one
        alpha_prod_t_prev = alpha_prod_t_prev[..., None, None]
        # alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = noise_pred
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * noise_pred
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(
                pred_original_sample, -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_mean = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 6. Get variance
        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance_type = self.config.variance_type
        # hacks - were probably added for training stability
        if variance_type == "fixed_small":
            variance = torch.clamp(variance, min=1e-20)
        # for rl-diffuser https://arxiv.org/abs/2205.09991
        elif variance_type == "fixed_small_log":
            variance = torch.log(torch.clamp(variance, min=1e-20))
            variance = torch.exp(0.5 * variance)
        elif variance_type == "fixed_large":
            variance = current_beta_t
        elif variance_type == "fixed_large_log":
            # Glide max_log
            variance = torch.log(current_beta_t)
        elif variance_type == "learned":
            variance = predicted_variance
        elif variance_type == "learned_range":
            min_log = torch.log(variance)
            max_log = torch.log(self.betas.to(t.device)[t])[..., None, None]
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log
        variance[t==0] = 0

        # 7. Get true mean µ_t and variance
        true_mean = pred_original_sample_coeff * original_sample + current_sample_coeff * sample
        true_variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        true_log_variance = torch.log(true_variance)
        true_log_variance[t==0] = 0

        return {
            'true_mean': true_mean,
            'true_log_var': true_log_variance,
            'pred_mean': pred_mean,
            'pred_log_var': variance
        }

    def get_x_start(self, output, x_t, t):

        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        alpha_prod_t = self.alphas_cumprod[t].view([-1, 1, 1])
        # alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else 1
        beta_prod_t = 1 - alpha_prod_t
        # beta_prod_t_prev = 1 - alpha_prod_t_prev
        # print(x_t.shape, output.shape, beta_prod_t.shape, alpha_prod_t.shape)
        # print((x_t - beta_prod_t ** (0.5) * output).shape)
        if self.config.prediction_type == 'epsilon':
            pred_x_start = (x_t - beta_prod_t ** (0.5) * output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == 'sample':
            pred_x_start = output
        else:
            print(f'prediction type {self.config.prediction_type} not found!')
            exit()
        return pred_x_start

    def q_mean_variance(self, original_samples, timesteps):
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        mean = sqrt_alpha_prod * original_samples
        var = sqrt_one_minus_alpha_prod
        return mean, var # 方差有问题，是否需要开方?