import torch
from torch import nn


class NormalMixtureEM(torch.nn.Module):
    def __init__(self, series_length, input_dim=2, n_components=2,
                 n_sem_iters=5, eps=1e-8):
        super().__init__()
        self.n_components = n_components
        self.n_sem_iters = n_sem_iters
        self.series_length = series_length
        weights_shape = [1] * (input_dim - 1)
        weights_shape.append(self.series_length)
        self.eps = eps

        self.init_centers = nn.Parameter(torch.randn(n_components, dtype=torch.float32))
        self.init_scales = nn.Parameter(torch.ones(n_components, dtype=torch.float32), requires_grad=False)
        self.init_weights = nn.Parameter(torch.ones(n_components, dtype=torch.float32) / n_components, requires_grad=False)
        self.prior_p = nn.Parameter(-torch.ones(1, dtype=torch.float32))
        self.weights = nn.Parameter(torch.zeros(weights_shape, dtype=torch.float32))
        self.blend = nn.Parameter(torch.zeros(1, dtype=torch.float32))
    
    def _initialize_parameters(self, windows):
        batch_shape = windows.shape[:-1]
        batch_dims = len(batch_shape)

        data_mean = torch.mean(windows, dim=-1, keepdim=True)
        data_std = torch.std(windows, dim=-1, keepdim=True)
        expand_dims = [1] * batch_dims + [-1]
        mu = self.init_centers.view(*expand_dims)
        mu = mu.expand(*batch_shape, -1) * data_std + data_mean

        sigma = torch.abs(self.init_scales.view(*expand_dims))
        sigma = sigma.expand(*batch_shape, -1) * data_std

        w = self.init_weights.view(*expand_dims).expand(*batch_shape, -1)

        noise_scale = 0.01
        noise = torch.randn_like(mu) * data_std * noise_scale
        mu = mu + noise
        
        return w, mu, sigma

    def _get_parameters(self):
        weights = torch.exp(self.weights).unsqueeze(-1)
        blender = torch.sigmoid(self.blend)
        prior_p = torch.sigmoid(self.prior_p)
        return weights, blender, prior_p

    def _e_step_normal(self, windows, w, mu, sigma):
        log_weights = torch.log((w + self.eps) / (sigma + self.eps)).unsqueeze(-2)
        log_mixture = -0.5 * ((windows.unsqueeze(-1) - mu.unsqueeze(-2)) /
                            (sigma.unsqueeze(-2) + self.eps)) ** 2 + log_weights
        log_normalization = torch.logsumexp(log_mixture, dim=-1, keepdim=True)
        log_responsibilities = log_mixture - log_normalization
        return torch.exp(log_responsibilities)

    def _m_step_normal(self, windows, g, weights, prior_p, blender):
        weighted_g = g * weights
        sum_g = torch.clamp(torch.sum(weighted_g, dim=-2), min=self.eps)

        w = (sum_g + prior_p)
        w = w / w.sum(dim=-1, keepdim=True)

        data_mu = torch.sum(weighted_g * windows.unsqueeze(-1), dim=-2) / sum_g
        prior_mu = torch.mean(windows, dim=-1, keepdim=True).expand(-1, self.n_components)
        mu = data_mu * blender + prior_mu * (1 - blender)

        diff = windows.unsqueeze(-1) - mu.unsqueeze(-2)
        prior_var = torch.var(windows, dim=-1, keepdim=True)
        data_variance = torch.sum(weighted_g * (diff ** 2), dim=-2) / sum_g
        prior_var_expanded = prior_var.repeat(*[1] * (prior_var.dim() - 1), self.n_components)
        blended_variance = data_variance * blender + prior_var_expanded * (1 - blender)
        sigma = torch.sqrt(blended_variance + self.eps)
        return w, mu, sigma

    def forward(self, windows):
        p, a, b = self._initialize_parameters(windows)
        weights, blender, prior_p = self._get_parameters()
        for _ in range(self.n_sem_iters):
            g = self._e_step_normal(windows, p, a, b)
            p, a, b = self._m_step_normal(windows, g, weights, prior_p, blender)
        return g, p, a, b
