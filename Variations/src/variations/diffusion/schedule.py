from __future__ import annotations

import math

import torch


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.999)


class GaussianDiffusion:
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        *,
        schedule: str = "linear",
        device: torch.device | str = "cpu",
    ) -> None:
        self.timesteps = int(timesteps)
        self.device = torch.device(device)
        if schedule == "cosine":
            betas = cosine_beta_schedule(self.timesteps)
        elif schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")
        self.betas = betas.to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def to(self, device: torch.device | str) -> "GaussianDiffusion":
        return GaussianDiffusion(
            timesteps=self.timesteps,
            beta_start=float(self.betas[0].item()),
            beta_end=float(self.betas[-1].item()),
            device=device,
        )

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)

    @staticmethod
    def _extract(values: torch.Tensor, timestep: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return values[timestep].view(-1, *([1] * (x.ndim - 1)))

    def q_sample(self, x0: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timestep, x0)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, timestep, x0)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def predict_x0(self, xt: torch.Tensor, timestep: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timestep, xt)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, timestep, xt)
        return (xt - sqrt_one_minus * predicted_noise) / sqrt_alpha.clamp(min=1e-6)

    @torch.no_grad()
    def p_sample(self, model, x: torch.Tensor, timestep: int, target_keys: torch.Tensor) -> torch.Tensor:
        t = torch.full((x.shape[0],), timestep, device=self.device, dtype=torch.long)
        beta_t = self._extract(self.betas, t, x)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x)
        sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, t, x)
        predicted_noise = model(x, t, target_keys)
        model_mean = sqrt_recip_alpha * (x - beta_t * predicted_noise / sqrt_one_minus.clamp(min=1e-6))
        if timestep == 0:
            return model_mean
        posterior_variance = self._extract(self.posterior_variance, t, x)
        return model_mean + torch.sqrt(posterior_variance.clamp(min=1e-20)) * torch.randn_like(x)

    @torch.no_grad()
    def p_sample_loop(
        self,
        model,
        shape: tuple[int, int],
        target_keys: torch.Tensor,
        *,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        if num_inference_steps is not None and num_inference_steps < self.timesteps:
            return self.ddim_sample_loop(model, shape, target_keys, num_inference_steps=num_inference_steps)
        x = torch.randn(shape, device=self.device)
        for timestep in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, timestep, target_keys)
        return x

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model,
        shape: tuple[int, int],
        target_keys: torch.Tensor,
        *,
        num_inference_steps: int,
        eta: float = 0.0,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=self.device)
        steps = torch.linspace(0, self.timesteps - 1, num_inference_steps, device=self.device).long().unique()
        steps = list(reversed([int(step.item()) for step in steps]))
        for idx, timestep in enumerate(steps):
            t = torch.full((shape[0],), timestep, device=self.device, dtype=torch.long)
            eps = model(x, t, target_keys)
            x0 = self.predict_x0(x, t, eps)
            next_timestep = steps[idx + 1] if idx + 1 < len(steps) else -1
            if next_timestep < 0:
                x = x0
                continue
            alpha = self.alphas_cumprod[timestep]
            alpha_next = self.alphas_cumprod[next_timestep]
            sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha) * (1 - alpha / alpha_next)).clamp(min=0.0)
            direction = torch.sqrt((1 - alpha_next - sigma**2).clamp(min=0.0)) * eps
            noise = sigma * torch.randn_like(x) if eta > 0 else 0.0
            x = torch.sqrt(alpha_next) * x0 + direction + noise
        return x

