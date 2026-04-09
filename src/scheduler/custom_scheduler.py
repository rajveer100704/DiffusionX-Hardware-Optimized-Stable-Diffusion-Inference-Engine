import torch

class DiffusionScheduler:
    """
    DDPM-style scheduler for diffusion models.
    Provides explicit computation of alpha schedules and x0 reconstruction metrics.
    """
    def __init__(self, num_steps=50, device="cpu"):
        self.num_steps = num_steps
        self.device = device
        self.init_noise_sigma = 1.0 # Standard DDPM entry sigma

        # Linear beta schedule (matching Stable Diffusion base)
        self.betas = torch.linspace(0.00085, 0.012, num_steps).to(device)

        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

        self.timesteps = torch.arange(num_steps - 1, -1, -1).to(device)

    def set_timesteps(self, steps):
        self.num_steps = steps
        # Scale indices explicitly rather than mathematically for simple implementation
        step_ratio = 1000 // steps
        self.timesteps = (torch.arange(0, steps) * step_ratio).round()[::-1].long().to(self.device)
        self.timesteps = torch.clamp(self.timesteps, 0, len(self.alpha_cumprod)-1)

    def scale_model_input(self, sample, timestep):
        """
        Ensures interchangeability with continuous time schedulers. 
        For standard DDPM, it's just the identity function.
        """
        return sample

    def step(self, noise_pred, t, x):
        """
        Perform one reverse diffusion step.
        """
        # Ensure t is treated as an index for our manual scheduler arrays
        if isinstance(t, torch.Tensor):
            t_idx = (self.timesteps == t).nonzero(as_tuple=True)[0]
            if len(t_idx) > 0:
                t = t_idx[0].item()
            else:
                t = 0 # Fallback
                
        # To align closely with the mathematical request, we use the raw step counter i
        alpha_t = self.alpha_cumprod[t]
        alpha_prev = self.alpha_cumprod[t - 1] if t > 0 else torch.tensor(1.0).to(self.device)

        # Estimate x0 (denoised image)
        x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_prev) * noise_pred

        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_prev) * x0_pred + dir_xt
        
        # Optional Stochasticity (Noise)
        noise = torch.randn_like(x) if t > 0 else 0
        x_prev += torch.sqrt(self.betas[t]) * noise

        # Wrap in an object to act like diffusers ScheduleOutput
        class SchedulerOutput:
            def __init__(self, prev_sample):
                self.prev_sample = prev_sample
        
        return SchedulerOutput(x_prev)
