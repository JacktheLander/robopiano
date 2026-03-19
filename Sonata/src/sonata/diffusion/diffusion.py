from __future__ import annotations

import torch


class GaussianDiffusion1D:
    def __init__(self, timesteps: int, beta_start: float, beta_end: float, device: torch.device):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)

    def q_sample(self, x0: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[timestep].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timestep].view(-1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def predict_x0(self, xt: torch.Tensor, timestep: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[timestep].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timestep].view(-1, 1, 1)
        return (xt - sqrt_one_minus * predicted_noise) / sqrt_alpha.clamp(min=1e-6)

    @torch.no_grad()
    def sample(self, model, shape: tuple[int, ...], prior: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = torch.randn(shape, device=self.device)
        for timestep in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), timestep, device=self.device, dtype=torch.long)
            predicted_noise = model(x, prior, t, condition)
            alpha = self.alphas[t].view(-1, 1, 1)
            alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1)
            beta = self.betas[t].view(-1, 1, 1)
            mean = (1.0 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1.0 - alpha_bar)) * predicted_noise)
            if timestep > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta) * noise
            else:
                x = mean
        return x
