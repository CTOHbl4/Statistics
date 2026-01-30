import torch
from torch.nn.functional import one_hot
from .windows import find_window_size, get_start_finish_increment, create_sliding_windows


class MixtureSEM:
    def __init__(self, time_series, weights=None, window_size=None, n_components=2,
                 tol=1e-4, eps=1e-8, N_init=5, comp_distr='normal',
                 exp_smooth=0.99, alpha=0.7, prior_strength=1.0, device="cuda"):
        '''
        weights: None/"exp"
        '''
        self.prior_strength = prior_strength
        self.eps = eps
        self.time_series = time_series
        if window_size is None:
            self.start, self.finish, self.increment, self.series_length = find_window_size(time_series, N_init=N_init, N_add=N_init, alpha=alpha)
        else:
            self.series_length = window_size
            self.start, self.finish, self.increment = get_start_finish_increment(window_size)
        self.device = device
        self.n_components = n_components
        self.tol = tol
        self.exp_smooth = exp_smooth
        self.weights = self._get_weights(weights)
        if comp_distr == 'normal':
            self._e_step = self._e_step_normal
            self._m_step = self._m_step_normal
        elif comp_distr == 'student':
            self._e_step = self._e_step_student
            self._m_step = self._m_step_student
        elif comp_distr == 'laplace':
            self._e_step = self._e_step_laplace
            self._m_step = self._m_step_laplace
        else:
            raise ValueError(f"Distribution {comp_distr} not supported")
        # weights (w_ik) - shape (n_series, n_components)
        # means (mu_ik) - shape (n_series, n_components)
        # standard deviations (sigma_ik) - shape (n_series, n_components)
        # responsibilities (g_ij) - shape (n_series, series_length, n_components)

    def _create_windows(self):
        windows = create_sliding_windows(
            self.time_series[self.start: self.finish], self.series_length) \
                .to(self.device)
        return windows

    def _initialize_parameters(self, windows):
        w = torch.ones((windows.shape[0], self.n_components), device=self.device) / self.n_components
        
        sorted_data, _ = torch.sort(windows, 1)
        quantiles = torch.linspace(0, 1, self.n_components + 2)[1:-1]
        indices = (quantiles * (self.series_length - 1)).long()
        mu = sorted_data[:, indices]

        typical_gap = torch.std(windows, 1)

        sigma = typical_gap.unsqueeze(1) * (0.5 + 0.5 * torch.rand(self.n_components, device=self.device))
        args = {'w': w, 'mu': mu, 'sigma': sigma}
        if self._e_step == self._e_step_student:
            nu = torch.ones((windows.shape[0], self.n_components), dtype=torch.float32, device=self.device) * 4
            args['nu'] = nu
        return args
    
    def _get_params_norm(self, **args):
        res = 0
        for name, value in args.items():
            res += torch.norm(value, 1).item()
        return res

    def _get_weights(self, weights_name):
        if weights_name is None:
            return torch.ones((1, self.series_length), dtype=torch.float32).to(self.device)
        if weights_name == 'exp':
            return torch.tensor([[self.exp_smooth ** (self.series_length - 1 - i) for i in range(self.series_length)]], device=self.device).to(self.device)
        raise NotImplementedError("weights not implemented")

    def _s_step(self, g):
        component = torch.multinomial(g.reshape((-1, self.n_components)), 1).to(self.device)
        y = one_hot(component, num_classes=self.n_components) \
            .reshape(
                (-1, self.series_length, self.n_components)
                ) * self.weights.unsqueeze(-1)
        v = y.sum(dim=1)
        p_v = (y > 0).sum(dim=1)
        return y, v, p_v

    def _e_step_normal(self, windows, w, mu, sigma):
        log_weights = torch.log((w.unsqueeze(1) + self.eps) / (sigma.unsqueeze(1) + self.eps))
        log_mixture = -0.5 * ((windows.unsqueeze(2) - mu.unsqueeze(1)) /
                            (sigma.unsqueeze(1) + self.eps)) ** 2 + log_weights
        log_normalization = torch.logsumexp(log_mixture, dim=2, keepdim=True)
        log_responsibilities = log_mixture - log_normalization

        return torch.exp(log_responsibilities)

    def _m_step_normal(self, windows, y, v, p_v, **args):
        w = (p_v + self.prior_strength)
        w = w / w.sum(dim=1, keepdim=True)
        data_mu = torch.sum(y * windows.unsqueeze(2), dim=1) / torch.clamp(v, min=self.eps)

        prior_mu = torch.mean(windows, dim=1, keepdim=True).expand(-1, self.n_components)
        blend_weight = v / (v + self.prior_strength)
        mu = data_mu * blend_weight + prior_mu * (1 - blend_weight)

        diff = windows.unsqueeze(2) - mu.unsqueeze(1)
        prior_var = torch.var(windows, dim=1, keepdim=True).expand(-1, self.n_components)
        
        data_variance = torch.sum(y * (diff ** 2), dim=1) / torch.clamp(v, min=self.eps)
        blended_variance = (data_variance * v + prior_var * self.prior_strength) / (v + self.prior_strength)
        
        blended_std = torch.sqrt(blended_variance + self.eps)
        return {'w': w, 'mu': mu, 'sigma': blended_std}
    
    def _e_step_laplace(self, windows, w, mu, sigma):
        log_weights = torch.log(w.unsqueeze(1) + self.eps)
        abs_diff = torch.abs(windows.unsqueeze(2) - mu.unsqueeze(1))
        log_mixture = -abs_diff / (sigma.unsqueeze(1) + self.eps) + log_weights
        log_normalization = torch.logsumexp(log_mixture, dim=2, keepdim=True)
        return torch.exp(log_mixture - log_normalization)

    def _m_step_laplace(self, windows, y, v, p_v, **args):
        w = (p_v + self.prior_strength)
        w = w / w.sum(dim=1, keepdim=True)
        
        batch_size, series_len, n_comp = y.shape
        y_flat = y.permute(0, 2, 1).reshape(-1, series_len)
        windows_expanded = windows.unsqueeze(1).expand(-1, n_comp, -1).reshape(-1, series_len)
        
        sorted_vals, sort_idx = torch.sort(windows_expanded, dim=1)
        batch_indices = torch.arange(y_flat.shape[0], device=self.device).unsqueeze(1)
        sorted_weights = y_flat[batch_indices, sort_idx]
        cum_weights = torch.cumsum(sorted_weights, dim=1)
        total_weights = cum_weights[:, -1:]
        median_mask = cum_weights >= (total_weights / 2)
        median_idx = torch.argmax(median_mask.float(), dim=1)
        median_vals = sorted_vals[batch_indices.squeeze(), median_idx]
        data_mu = median_vals.view(batch_size, n_comp)
        prior_mu = torch.median(windows, dim=1).values.unsqueeze(1).expand(-1, n_comp)
        blend_weight = v / (v + self.prior_strength)
        mu = data_mu * blend_weight + prior_mu * (1 - blend_weight)
        
        diff = windows.unsqueeze(2) - mu.unsqueeze(1)
        weighted_abs_diff = torch.sum(y * torch.abs(diff), dim=1)
        data_sigma = weighted_abs_diff / torch.clamp(v, min=self.eps)
        prior_sigma = torch.median(torch.abs(windows - torch.median(windows, dim=1).values.unsqueeze(1)), dim=1).values.unsqueeze(1).expand(-1, n_comp)
        blended_std = data_sigma * blend_weight + prior_sigma * (1 - blend_weight)
        return {'w': w, 'mu': mu, 'sigma': blended_std}

    def _e_step_student(self, windows, w, mu, sigma, nu):
        log_weights = torch.log(w.unsqueeze(1) + self.eps)
        diff = windows.unsqueeze(2) - mu.unsqueeze(1)
        scaled_diff = diff / (sigma.unsqueeze(1) + self.eps)
        nu_expanded = nu.unsqueeze(1)

        log_const = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - \
                    0.5 * torch.log(torch.tensor(torch.pi, device=self.device)) - \
                    0.5 * torch.log(nu + self.eps) - \
                    torch.log(sigma + self.eps)
        log_kernel = -((nu_expanded + 1) / 2) * torch.log1p(scaled_diff ** 2 / nu_expanded)

        log_mixture = log_const.unsqueeze(1) + log_kernel + log_weights
        log_normalization = torch.logsumexp(log_mixture, dim=2, keepdim=True)
        return torch.exp(log_mixture - log_normalization)
 
    def _m_step_student(self, windows, y, v, p_v, w, mu, sigma, nu):        
        batch_size, series_len, n_comp = y.shape
        n = series_len
        
        assigned_components = torch.argmax(y, dim=2)
        batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, series_len)
        
        assigned_mu = mu[batch_idx, assigned_components]
        assigned_sigma = sigma[batch_idx, assigned_components]
        assigned_nu = nu[batch_idx, assigned_components]
        
        diff_assigned = windows - assigned_mu
        mahalanobis_sq = (diff_assigned / (assigned_sigma + self.eps)) ** 2
        
        ai = (assigned_nu + 1) / 2
        bi = assigned_nu/2 + 0.5 * mahalanobis_sq
        ci = (bi / (ai + self.eps)).unsqueeze(2)
        
        w_new = (p_v + self.prior_strength)
        w_new = w_new / w_new.sum(dim=1, keepdim=True)
        
        ci_expanded = ci.expand(-1, -1, n_comp)
        y_weighted = y * ci_expanded
        
        numerator_mu = torch.sum(y_weighted * windows.unsqueeze(2), dim=1)
        denominator_mu = torch.sum(y_weighted, dim=1)
        data_mu = numerator_mu / torch.clamp(denominator_mu, min=self.eps)
        
        prior_mu = torch.mean(windows, dim=1, keepdim=True).expand(-1, n_comp)
        blend_weight = v / (v + self.prior_strength)
        mu_new = data_mu * blend_weight + prior_mu * (1 - blend_weight)
        
        diff_new = windows.unsqueeze(2) - mu_new.unsqueeze(1)
        weighted_sq_diff = ci.expand(-1, -1, n_comp) * (diff_new ** 2)
        sum_weighted_sq = torch.sum(weighted_sq_diff, dim=1)
        sigma_sq = sum_weighted_sq / n
        
        data_sigma = torch.sqrt(sigma_sq + self.eps)
        prior_var = torch.var(windows, dim=1, keepdim=True)
        
        prior_sigma_sq = prior_var * (nu - 2) / torch.clamp(nu, min=2.1)
        prior_sigma = torch.sqrt(torch.clamp(prior_sigma_sq, min=self.eps))
        
        sigma_new = data_sigma * blend_weight + prior_sigma * (1 - blend_weight)
        sigma_new = torch.clamp(sigma_new, min=self.eps)
        
        standardized = diff_new / (sigma_new.unsqueeze(1) + self.eps)
        weighted_k4 = torch.sum(y * standardized**4, dim=1)
        weighted_k2 = torch.sum(y * standardized**2, dim=1)
        
        mask = v > 2
        nu_new = nu.clone()
        
        if mask.any():
            kurtosis = weighted_k4[mask] / (weighted_k2[mask]**2 + self.eps) - 3
            kurtosis = torch.clamp(kurtosis, min=0.01, max=100.0)
            nu_update = torch.clamp(4 + 6 / (kurtosis + self.eps), min=2.1, max=30.0)
            nu_new[mask] = 0.8 * nu[mask] + 0.2 * nu_update
        
        return {'w': w_new, 'mu': mu_new, 'sigma': sigma_new, 'nu': nu_new}

    def _fit_batch(self, windows):
        args = self._initialize_parameters(windows)
        prev_params_norm = self._get_params_norm(**args)
        d = 1 + 2 * self.tol
        i = 0
        while (abs(d - 1) > self.tol) and ((i < 5) if windows.shape[0] == 1 else True):
            g = self._e_step(windows, **args)
            y, v, p_v = self._s_step(g)
            args = self._m_step(windows, y, v, p_v, **args)
            curr_params_norm = self._get_params_norm(**args)
            d = curr_params_norm / prev_params_norm
            prev_params_norm = curr_params_norm
            i += 1
        return g, args['w'], args['mu'], args['sigma']

    def find_params(self):
        pred_w = torch.tensor([], dtype=torch.float32, device=self.device)
        pred_mu = torch.tensor([], dtype=torch.float32, device=self.device)
        pred_sigma = torch.tensor([], dtype=torch.float32, device=self.device)
        pred_g = torch.tensor([], dtype=torch.float32, device=self.device)
        batch_num = 0
        while self.start < len(self.time_series):
            batch_num += 1
            g, w, mu, sigma = self._fit_batch(self._create_windows())
            pred_w = torch.concat([pred_w, w])
            pred_mu = torch.concat([pred_mu, mu])
            pred_sigma = torch.concat([pred_sigma, sigma])
            pred_g = torch.concat([pred_g, g])
            self.start += self.increment
            self.finish += self.increment
        return pred_g, pred_w, pred_mu, pred_sigma



    # def _get_B_t(self):
    #     b_t = torch.sum(self.b * self.b * self.p, dim=1)**0.5
    #     return b_t.detach().cpu()

    # def _batched_mixture_multiple_percentiles(self):
    #     print("Percentiles iterative optimization. Bisection deltas are printed.")
    #     batch_size = self.n_series
    #     n_percentiles = len(self.p_values)

    #     overall_means = (self.p * self.a).sum(dim=1)
    #     overall_stds = torch.sqrt((self.p * (self.b**2 + (self.a - overall_means.unsqueeze(1))**2)).sum(dim=1))

    #     lower = (overall_means - 4.0 * overall_stds).unsqueeze(1).unsqueeze(2)
    #     upper = (overall_means + 4.0 * overall_stds).unsqueeze(1).unsqueeze(2)

    #     lower = lower.expand(batch_size, n_percentiles, 1)
    #     upper = upper.expand(batch_size, n_percentiles, 1)
    #     p_values = self.p_values.unsqueeze(0).unsqueeze(2).expand(batch_size, n_percentiles, 1)

    #     for _ in range(self.max_iter_percentile):
    #         mid = (lower + upper) / 2
    #         z = (mid - self.a.unsqueeze(1)) / self.b.unsqueeze(1)
    #         component_cdfs = 0.5 * (1 + torch.erf(z / torch.sqrt(torch.tensor(2.0, device=self.a.device))))
    #         mixture_cdf = (self.p.unsqueeze(1) * component_cdfs).sum(dim=2).unsqueeze(-1)
    #         lower = torch.where(mixture_cdf < p_values, mid, lower)
    #         upper = torch.where(mixture_cdf >= p_values, mid, upper)
    #         print((upper - lower).sum())
    #     return (lower + upper).squeeze(2) / 2
    
    # def _batched_mixture_mode(self, initial_guesses=None):
    #     print("Mode iterative optimization. Max change is printed.")
    #     if initial_guesses is not None:
    #         x = initial_guesses
    #     else:
    #         x = (self.p * self.a).sum(dim=1)

    #     x = x.unsqueeze(1)
    #     a = self.a
    #     b = self.b
    #     p = self.p
        
    #     for _ in range(self.max_iter_percentile):
    #         x_expanded = x.expand(-1, self.n_components)
    #         exponent = -0.5 * ((x_expanded - a) / b) ** 2
    #         log_weights = torch.log(p) - 3 * torch.log(b) + exponent
    #         max_log = torch.max(log_weights, dim=1, keepdim=True)[0]
    #         weights = torch.exp(log_weights - max_log)

    #         x_new = torch.sum(weights * a, dim=1, keepdim=True) / torch.sum(weights, dim=1, keepdim=True)

    #         diff = torch.abs(x_new - x)
    #         max_diff = torch.max(diff)
    #         print(max_diff)
    #         x = x_new
    #     return x.squeeze(1)
