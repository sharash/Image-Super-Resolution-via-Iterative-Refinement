import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm


class GLoss(nn.Module):
    def __init__(self, scale_factor=4):
        """
        Initialize the G-Loss function
        Args:
            scale_factor: The super-resolution scale factor (2, 3, or 4)
        """
        super(GLoss, self).__init__()
        
        # Define the 8 gradient extraction kernels
        self.kernels = [
            [[-1, 0, 0], [0, 1, 0], [0, 0, 0]],  # top-left
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]],  # top
            [[0, 0, -1], [0, 1, 0], [0, 0, 0]],  # top-right
            [[-1, 0, 0], [0, 1, 0], [0, 0, 0]],  # left
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]],  # right
            [[0, 0, 0], [0, 1, 0], [-1, 0, 0]],  # bottom-left
            [[0, 0, 0], [0, 1, 0], [0, -1, 0]],  # bottom
            [[0, 0, 0], [0, 1, 0], [0, 0, -1]]   # bottom-right
        ]
        
        # Convert the kernels to PyTorch tensors and register them as buffers
        self.grad_kernels = []
        for k in self.kernels:
            kernel = torch.FloatTensor(k).unsqueeze(0).unsqueeze(0)
            self.register_buffer(f'kernel_{len(self.grad_kernels)}', kernel)
            self.grad_kernels.append(kernel)
            
        # Define the sampling rates based on scale factor
        self.scale_factor = scale_factor
        if scale_factor == 2:
            self.sampling_rates = [2]
        elif scale_factor == 3:
            self.sampling_rates = [2, 3]
        else:  # scale_factor == 4
            self.sampling_rates = [2, 4]

    def extract_gradients(self, x):
        """Extract gradient feature maps using 8-directional convolution kernels"""
        batch, channels = x.shape[0], x.shape[1]
        grad_maps = []
        
        # For each channel
        for c in range(channels):
            channel_grads = []
            # For each direction kernel
            for kernel_idx, kernel in enumerate(self.grad_kernels):
                # Apply convolution with padding
                kernel_var = getattr(self, f'kernel_{kernel_idx}')
                grad = F.conv2d(x[:, c:c+1], kernel_var.repeat(1, 1, 1, 1), padding=1)
                channel_grads.append(grad)
                
            # Stack the gradient maps from all directions
            channel_grad_maps = torch.cat(channel_grads, dim=1)
            grad_maps.append(channel_grad_maps)
            
        # Combine gradient maps from all channels
        return torch.cat(grad_maps, dim=1)

    def split_downsample(self, x, rate):
        """Downsample by splitting the image into blocks and stacking as channels"""
        b, c, h, w = x.size()
        # Reshape to create blocks of size rate x rate
        x = x.view(b, c, h // rate, rate, w // rate, rate)
        # Permute and reshape to convert spatial blocks to channels
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(b, c * (rate ** 2), h // rate, w // rate)
        return x

    def forward(self, sr, hr):
        """
        Calculate G-Loss between super-resolution and high-resolution images
        Args:
            sr: Super-resolution image tensor (B, C, H, W)
            hr: High-resolution ground truth tensor (B, C, H, W)
        """
        # 1. Calculate pixel loss (PL)
        pixel_loss = F.l1_loss(sr, hr)
        
        # 2. Calculate gradient loss (G1)
        sr_grads = self.extract_gradients(sr)
        hr_grads = self.extract_gradients(hr)
        g1_loss = F.l1_loss(sr_grads, hr_grads)
        
        total_loss = pixel_loss + g1_loss
        
        # 3. Calculate split gradient losses (SGn)
        for rate in self.sampling_rates:
            # Downsample using splitting
            sr_downsampled = self.split_downsample(sr, rate)
            hr_downsampled = self.split_downsample(hr, rate)
            
            # Extract gradients from downsampled images
            sr_sg_grads = self.extract_gradients(sr_downsampled)
            hr_sg_grads = self.extract_gradients(hr_downsampled)
            
            # Calculate loss with weight adjustment (1/n²)
            weight = 1.0 / (rate ** 2)
            sg_loss = F.l1_loss(sr_sg_grads, hr_sg_grads) * weight
            
            total_loss = total_loss + sg_loss
        
        return total_loss


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None,
        ddim_sampling=False,  # Add this
        ddim_timesteps=200    # Add this
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        self.ddim_sampling = ddim_sampling  # Store DDIM flag
        self.ddim_timesteps = ddim_timesteps  # Store DDIM steps
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        elif self.loss_type == 'gloss':
            self.loss_func = GLoss(scale_factor=4).to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_ddim(self, x, t, t_prev, condition_x=None):
        """
        DDIM sampling - deterministic sampling with skip steps
        """
        # Get model prediction (noise or x0)
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t+1]]).to(x.device)
        if condition_x is not None:
            pred_noise = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
        else:
            pred_noise = self.denoise_fn(x, noise_level)
        
        # Predict x0 from current x_t
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        x_recon.clamp_(-1., 1.)

        # DDIM update rule
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_alpha_cumprod_t_prev = self.sqrt_alphas_cumprod[t_prev] if t_prev >= 0 else torch.ones_like(sqrt_alpha_cumprod_t)
        sigma_t = 0  # DDIM is deterministic (sigma=0)
        
        # Direction pointing to x_t
        pred_dir = torch.sqrt(1 - sqrt_alpha_cumprod_t_prev**2 - sigma_t**2) * pred_noise
        
        # Update x_t-1
        x_prev = sqrt_alpha_cumprod_t_prev * x_recon + pred_dir
        return x_prev

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False, ddim=False, timesteps=None):
        device = self.betas.device
        
        if ddim:
            # DDIM sampling with reduced timesteps
            if timesteps is None:
                timesteps = self.num_timesteps // 10  # Default: 10x fewer steps
            time_range = np.flip(np.arange(0, self.num_timesteps, self.num_timesteps // timesteps))
            time_pairs = list(zip(time_range[:-1], time_range[1:]))
            sample_inter = (1 | (timesteps // 10))
        else:
            # Original DDPM sampling
            time_range = reversed(range(0, self.num_timesteps))
            time_pairs = [(t, t-1) for t in time_range]
            sample_inter = (1 | (self.num_timesteps//10))

        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for t, t_prev in tqdm(time_pairs, desc='sampling loop time step'):
                if ddim:
                    img = self.p_sample_ddim(img, t, t_prev)
                else:
                    img = self.p_sample(img, t)
                if t % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            for t, t_prev in tqdm(time_pairs, desc='sampling loop time step'):
                if ddim:
                    img = self.p_sample_ddim(img, t, t_prev, condition_x=x)
                else:
                    img = self.p_sample(img, t, condition_x=x)
                if t % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
    
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False, ddim=False, timesteps=None):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(
            (batch_size, channels, image_size, image_size),
            continous=continous,
            ddim=ddim,
            timesteps=timesteps
        )

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False, ddim=False, timesteps=None):
        return self.p_sample_loop(
            x_in,
            continous=continous,
            ddim=ddim,
            timesteps=timesteps
        )

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        loss = self.loss_func(noise, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
