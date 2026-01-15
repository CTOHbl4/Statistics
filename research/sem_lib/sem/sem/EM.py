import torch
from torch import nn


class NormalMixtureEM(torch.nn.Module):
    def __init__(self, series_length, input_dim=2, n_components=2,
                 n_sem_iters=5, n_iters=15, eps=1e-8):
        super().__init__()
        self.n_components = n_components
        self.n_sem_iters = n_sem_iters
        self.n_iters = n_iters
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
        a = self.init_centers.view(*expand_dims)
        a = a.expand(*batch_shape, -1) * data_std + data_mean

        b = torch.abs(self.init_scales.view(*expand_dims))
        b = b.expand(*batch_shape, -1) * data_std

        p = self.init_weights.view(*expand_dims).expand(*batch_shape, -1)

        noise_scale = 0.01
        noise = torch.randn_like(a) * data_std * noise_scale
        a = a + noise
        
        return p, a, b

    def _e_step(self, windows, p, a, b):
        log_weights = torch.log((p + self.eps) / (b + self.eps)).unsqueeze(-2)
        log_mixture = -0.5 * ((windows.unsqueeze(-1) - a.unsqueeze(-2)) /
                            (b.unsqueeze(-2) + self.eps)) ** 2 + log_weights
        log_normalization = torch.logsumexp(log_mixture, dim=-1, keepdim=True)
        log_responsibilities = log_mixture - log_normalization
        return torch.exp(log_responsibilities)
    
    def _get_parameters(self):
        weights = torch.exp(self.weights).unsqueeze(-1)
        blender = torch.sigmoid(self.blend)
        prior_p = torch.sigmoid(self.prior_p)
        return weights, blender, prior_p

    def _m_step(self, windows, _g):
        weights, blender, prior_p = self._get_parameters()
        g = _g * weights
        sum_g = torch.clamp(torch.sum(g, dim=-2), min=self.eps)

        p = (sum_g + prior_p)
        p = p / p.sum(dim=-1, keepdim=True)

        data_a = torch.sum(g * windows.unsqueeze(-1), dim=-2) / sum_g
        
        prior_a = torch.mean(windows, dim=-1, keepdim=True)
        prior_a_expanded = prior_a.repeat(*[1] * (prior_a.dim() - 1), self.n_components)
        
        a = data_a * blender + prior_a_expanded * (1 - blender)

        diff = windows.unsqueeze(-1) - a.unsqueeze(-2)
        prior_var = torch.var(windows, dim=-1, keepdim=True)
        
        data_variance = torch.sum(g * (diff ** 2), dim=-2) / sum_g

        prior_var_expanded = prior_var.repeat(*[1] * (prior_var.dim() - 1), self.n_components)
        blended_variance = data_variance * blender + prior_var_expanded * (1 - blender)
        b = torch.sqrt(blended_variance + self.eps)
        return p, a, b

    def forward(self, windows):
        p, a, b = self._initialize_parameters(windows)
        for _ in range(self.n_sem_iters):
            g = self._e_step(windows, p, a, b)
            p, a, b = self._m_step(windows, g)
        return g, p, a, b
