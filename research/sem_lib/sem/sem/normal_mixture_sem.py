import torch
import torch.nn.functional as F
from .windows import Windows


class NormalMixtureSEM:
    def __init__(self, time_series, weights=None, method='EM', n_components=2,
                 tol=1e-3, eps=1e-8, N_init=500,
                 exp_smooth=0.999, alpha=0.6,
                 p_values=torch.tensor([0.01 * i for i in range(1, 100)]),
                 max_iter_percentile=30, median_idx=49, prior_strength=1.0, device="cuda"):
        '''
        weights: None/"exp"/"exp_focus_first",

        method: "EM"/"SEM"
        '''
        self.prior_strength = prior_strength
        self.eps = eps
        self.time_series = time_series
        self.max_iter_percentile = max_iter_percentile
        self.p_values = p_values.to(device)
        self.median_idx = median_idx
        self.n_finder = Windows()
        self.start, self.finish, self.increment, self.series_length = self.n_finder(time_series, N_init=N_init, N_add=N_init, alpha=alpha)
        self.device = device
        self.n_components = n_components
        self.tol = tol
        self.weights_name = weights
        self.method = self._em_steps if method == 'EM' else self._sem_steps
        self.exp_smooth = exp_smooth
        self.p = None  # weights (p_ik) - shape (n_series, n_components)
        self.a = None  # means (a_ik) - shape (n_series, n_components)
        self.b = None  # standard deviations (σ_ik) - shape (n_series, n_components)
        self.g = None  # responsibilities (g_ij) - shape (n_series, series_length, n_components)

    def _em_steps(self):
        self._e_step()
        self._em_m_step()

    def _sem_steps(self):
        self._e_step()
        y, v, p_v = self._s_step()
        self._sem_m_step(y, v, p_v)

    def _prepare_fit(self):
        self.time_series_window = self.n_finder.create_sliding_windows(
            self.time_series[self.start: self.finish], self.series_length) \
                .to(self.device)
        self.n_series = self.time_series_window.shape[0]
        self._initialize_parameters()
        self.weights = self._get_weights()

    def _initialize_parameters(self):
        self.p = torch.ones((self.n_series, self.n_components), device=self.device) / self.n_components
        
        sorted_data, _ = torch.sort(self.time_series_window, 1)
        quantiles = torch.linspace(0, 1, self.n_components + 2)[1:-1]
        indices = (quantiles * (self.series_length - 1)).long()
        self.a = sorted_data[:, indices]

        if self.n_components > 1:
            gaps = self.a[:, 1:] - self.a[:, :-1]
            typical_gap = torch.mean(gaps, 1)
        else:
            typical_gap = torch.std(self.time_series_window, 1)

        self.b = typical_gap.unsqueeze(1) * (0.5 + 0.5 * torch.rand(self.n_components, device=self.device))


    def _get_weights(self):
        if self.weights_name is None:
            return torch.ones((1, self.series_length), dtype=torch.float32).to(self.device)
        exp_weights = torch.tensor([[self.exp_smooth ** (self.series_length - 1 - i) for i in range(self.series_length)]], device=self.device).to(self.device)
        if self.weights_name == 'exp':
            return exp_weights
        if self.weights_name == 'exp_focus_first':
            exp_weights[0, -1] += exp_weights[0, -2]
            return exp_weights
        raise NotImplementedError("weights not implemented")

    def _s_step(self):
        component = torch.multinomial(self.g.reshape((-1, self.n_components)), 1).to(self.device)
        y = F.one_hot(component, num_classes=self.n_components) \
            .reshape(
                (self.n_series, self.series_length, self.n_components)
                ) * self.weights.unsqueeze(-1)
        v = y.sum(dim=1)
        p_v = (y > 0).sum(dim=1)
        return y, v, p_v

    def _e_step(self):
        log_weights = torch.log((self.p.unsqueeze(1) + self.eps) / (self.b.unsqueeze(1) + self.eps))
        log_mixture = -0.5 * ((self.time_series_window.unsqueeze(2) - self.a.unsqueeze(1)) /
                            (self.b.unsqueeze(1) + self.eps)) ** 2 + log_weights
        log_normalization = torch.logsumexp(log_mixture, dim=2, keepdim=True)
        log_responsibilities = log_mixture - log_normalization

        self.g = torch.exp(log_responsibilities)

    def _sem_m_step(self, y, v, p_v):
        self.p = (p_v + self.prior_strength)
        self.p = self.p / self.p.sum(dim=1, keepdim=True)
        data_a = torch.sum(y * self.time_series_window.unsqueeze(2), dim=1) / torch.clamp(v, min=self.eps)

        prior_a = torch.mean(self.time_series_window, dim=1, keepdim=True)
        prior_a_expanded = prior_a.expand(-1, self.n_components)
        blend_weight = v / (v + self.prior_strength)
        self.a = data_a * blend_weight + prior_a_expanded * (1 - blend_weight)

        diff = self.time_series_window.unsqueeze(2) - self.a.unsqueeze(1)
        prior_var = torch.var(self.time_series_window, dim=1, keepdim=True)
        prior_var_expanded = prior_var.expand(-1, self.n_components)
        
        data_variance = torch.sum(y * (diff ** 2), dim=1) / torch.clamp(v, min=self.eps)
        blended_variance = (data_variance * v + prior_var_expanded * self.prior_strength) / (v + self.prior_strength)
        
        blended_std = torch.sqrt(blended_variance + self.eps)
        self.b = blended_std
    
    def _em_m_step(self):
        g = self.g * self.weights.unsqueeze(-1)
        sum_g = torch.sum(g, dim=1)

        self.p = (sum_g + self.prior_strength)
        self.p = self.p / self.p.sum(dim=1, keepdim=True)

        data_a = torch.sum(g * self.time_series_window.unsqueeze(2), dim=1) / torch.clamp(sum_g, min=self.eps)

        prior_a = torch.mean(self.time_series_window, dim=1, keepdim=True)
        blend = sum_g / (sum_g + self.prior_strength)
        
        prior_a_expanded = prior_a.expand(-1, self.n_components)

        self.a = data_a * blend + prior_a_expanded * (1 - blend)

        diff = self.time_series_window.unsqueeze(2) - self.a.unsqueeze(1)
        prior_var = torch.var(self.time_series_window, dim=1, keepdim=True)
        data_variance = torch.sum(g * (diff ** 2), dim=1) / torch.clamp(sum_g, min=self.eps)

        prior_var_expanded = prior_var.expand(-1, self.n_components)
        blended_variance = (data_variance * sum_g + prior_var_expanded * self.prior_strength) / (sum_g + self.prior_strength)

        blended_std = torch.sqrt(blended_variance + self.eps)
        self.b = blended_std

    def _log_likelihood(self):
        log_probs = -0.5 * ((self.time_series_window.unsqueeze(2) - self.a.unsqueeze(1)) / self.b.unsqueeze(1)) ** 2 \
            + torch.log(self.p).unsqueeze(1) - torch.log(self.b).unsqueeze(1) - 0.5 * torch.log(2 * torch.tensor([torch.pi], device=self.device))
        log_likelihood_per_point = torch.logsumexp(log_probs, dim=2)
        total_log_likelihood = (log_likelihood_per_point * self.weights).sum(dim=1)
        return total_log_likelihood.sum().item()

    def _fit_batch(self):
        prev_params_norm = -1.0
        d = 1.0
        while d > self.tol:
            self.method()
            curr_params_norm = (torch.norm(self.p) + torch.norm(self.a) + torch.norm(self.b)).item()
            d = abs(curr_params_norm - prev_params_norm)
            print(curr_params_norm)
            prev_params_norm = curr_params_norm
        return self

    def find_params(self):
        pred_p = torch.tensor([], dtype=torch.float32)
        pred_a = torch.tensor([], dtype=torch.float32)
        pred_b = torch.tensor([], dtype=torch.float32)
        pred_percentiles = torch.tensor([], dtype=torch.float32)
        pred_modes = torch.tensor([], dtype=torch.float32)
        pred_b_t = torch.tensor([], dtype=torch.float32)
        batch_num = 0
        print("Finding parameters batch by batch (to fit into memory).")
        while self.start < len(self.time_series):
            self._prepare_fit()
            print("Batch", batch_num)
            batch_num += 1
            self._fit_batch()
            percentiles = self._batched_mixture_multiple_percentiles()
            modes = self._batched_mixture_mode(percentiles[:, self.median_idx])
            b_t = self._get_B_t()
            pred_percentiles = torch.concat([pred_percentiles, percentiles.cpu()])
            pred_modes = torch.concat([pred_modes, modes.cpu()])
            pred_p = torch.concat([pred_p, self.p.cpu()])
            pred_a = torch.concat([pred_a, self.a.cpu()])
            pred_b = torch.concat([pred_b, self.b.cpu()])
            pred_b_t = torch.concat([pred_b_t, b_t.cpu()])
            self.start += self.increment
            self.finish += self.increment
        return pred_p, pred_a, pred_b, pred_percentiles, pred_b_t, pred_modes

    def _get_B_t(self):
        b_t = torch.sum(self.b * self.b * self.p, dim=1)**0.5
        return b_t.detach().cpu()

    def _batched_mixture_multiple_percentiles(self):
        print("Percentiles iterative optimization. Bisection deltas are printed.")
        batch_size = self.n_series
        n_percentiles = len(self.p_values)

        overall_means = (self.p * self.a).sum(dim=1)
        overall_stds = torch.sqrt((self.p * (self.b**2 + (self.a - overall_means.unsqueeze(1))**2)).sum(dim=1))

        lower = (overall_means - 4.0 * overall_stds).unsqueeze(1).unsqueeze(2)
        upper = (overall_means + 4.0 * overall_stds).unsqueeze(1).unsqueeze(2)

        lower = lower.expand(batch_size, n_percentiles, 1)
        upper = upper.expand(batch_size, n_percentiles, 1)
        p_values = self.p_values.unsqueeze(0).unsqueeze(2).expand(batch_size, n_percentiles, 1)

        for _ in range(self.max_iter_percentile):
            mid = (lower + upper) / 2
            z = (mid - self.a.unsqueeze(1)) / self.b.unsqueeze(1)
            component_cdfs = 0.5 * (1 + torch.erf(z / torch.sqrt(torch.tensor(2.0, device=self.a.device))))
            mixture_cdf = (self.p.unsqueeze(1) * component_cdfs).sum(dim=2).unsqueeze(-1)
            lower = torch.where(mixture_cdf < p_values, mid, lower)
            upper = torch.where(mixture_cdf >= p_values, mid, upper)
            print((upper - lower).sum())
        return (lower + upper).squeeze(2) / 2
    
    def _batched_mixture_mode(self, initial_guesses=None):
        print("Mode iterative optimization. Max change is printed.")
        if initial_guesses is not None:
            x = initial_guesses
        else:
            x = (self.p * self.a).sum(dim=1)

        x = x.unsqueeze(1)
        a = self.a
        b = self.b
        p = self.p
        
        for _ in range(self.max_iter_percentile):
            x_expanded = x.expand(-1, self.n_components)
            exponent = -0.5 * ((x_expanded - a) / b) ** 2
            log_weights = torch.log(p) - 3 * torch.log(b) + exponent
            max_log = torch.max(log_weights, dim=1, keepdim=True)[0]
            weights = torch.exp(log_weights - max_log)

            x_new = torch.sum(weights * a, dim=1, keepdim=True) / torch.sum(weights, dim=1, keepdim=True)

            diff = torch.abs(x_new - x)
            max_diff = torch.max(diff)
            print(max_diff)
            x = x_new
        return x.squeeze(1)
