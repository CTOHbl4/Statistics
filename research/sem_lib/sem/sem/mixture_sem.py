import torch
from torch.nn.functional import one_hot
from .windows import find_window_size, get_start_finish_increment, create_sliding_windows
from numpy import sign
from scipy.special import betainc


class MixtureSEM:
    '''
    Works with unsqueezed parameters
    '''
    eps = 1e-8
    @classmethod
    def log_pdf_normal(self, x, mu, sigma, **args):
        return -0.5 * torch.log(2 * torch.pi * sigma**2 + self.eps) - \
                    0.5 * ((x - mu) / (sigma + self.eps))**2

    @classmethod
    def log_pdf_laplace(self, x, mu, sigma, **args):
        return -torch.log(2 * sigma + self.eps) - \
                torch.abs(x - mu) / (sigma + self.eps)

    @classmethod
    def log_pdf_logistic(self, x, mu, sigma, **args):
        abs_z = torch.abs((x - mu) / (sigma + self.eps))
        return -abs_z - torch.log(sigma + self.eps) - \
                    2 * torch.log1p(torch.exp(-abs_z))

    @classmethod
    def log_pdf_student(self, x, mu, sigma, nu, **args):
        z = (x - mu) / (sigma + self.eps)

        log_const = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - \
                    0.5 * torch.log(nu * torch.pi + self.eps)
        log_scale = -torch.log(sigma + self.eps)
        log_kernel = -((nu + 1) / 2) * torch.log1p(z**2 / nu)
        log_density = log_const + log_scale + log_kernel
        return log_density

    @classmethod
    def cdf_normal(self, mid, mu, sigma, **args):
        z = (mid - mu) / (sigma + self.eps)
        return 0.5 * (1 + torch.erf(z / torch.sqrt(torch.tensor(2.0, device=mid.device))))

    @classmethod
    def cdf_laplace(self, mid, mu, sigma, **args):
        z = (mid - mu) / (sigma + self.eps)
        return torch.where(
            z > 0,
            1 - 0.5 * torch.exp(-z),
            0.5 * torch.exp(z)
        )

    @classmethod
    def cdf_logistic(self, mid, mu, sigma, **args):
        z = (mid - mu) / (sigma + self.eps)
        return torch.sigmoid(z)

    @classmethod
    def cdf_student(self, mid, mu, sigma, nu, **args):
        z = ((mid - mu) / (sigma + self.eps)).cpu().numpy()
        nu_cpu = nu.cpu().numpy()
        x = nu_cpu / (nu_cpu + z**2)
        cdf = 0.5 * (1 + sign(z) * (1 - betainc(nu_cpu/2, 0.5, x)))
        return torch.tensor(cdf, dtype=torch.float32, device=mid.device)

    def __init__(self, time_series, window_size=None, n_components=2,
                 tol=1e-4, N_init=5, comp_distr='normal',
                 exp_smooth=1.0, alpha=0.7, prior_strength=1.0, device="cuda"):

        self.prior_strength = prior_strength
        self.time_series = time_series
        if window_size is None:
            self.start, self.finish, self.increment, self.series_length = find_window_size(time_series, N_init=N_init, N_add=N_init, alpha=alpha)
        else:
            self.series_length = window_size
            self.start, self.finish, self.increment = get_start_finish_increment(window_size)
        self.device = device
        self.n_components = n_components
        self.tol = tol
        self.weights = self._get_weights(exp_smooth)
        self.comp_distr = comp_distr
        if comp_distr == 'normal':
            self._m_step = self._m_step_normal
            self._log_pdf = self.log_pdf_normal
            self._cdf = self.cdf_normal
        elif comp_distr == 'student':
            self._m_step = self._m_step_student
            self._log_pdf = self.log_pdf_student
            self._cdf = self.cdf_student
        elif comp_distr == 'laplace':
            self._m_step = self._m_step_laplace
            self._log_pdf = self.log_pdf_laplace
            self._cdf = self.cdf_laplace
        elif comp_distr == 'logistic':
            self._m_step = self._m_step_logistic
            self._log_pdf = self.log_pdf_logistic
            self._cdf = self.cdf_logistic
        else:
            raise ValueError(f"Distribution {comp_distr} not supported")
        # weights (w_ik) - shape (n_series, n_components)
        # means (mu_ik) - shape (n_series, n_components)
        # standard deviations (sigma_ik) - shape (n_series, n_components)
        # responsibilities (g_ij) - shape (n_series, series_length, n_components)

    def _initialize_parameters(self, windows):
        w = torch.ones((windows.shape[0], 1, self.n_components), device=self.device) / self.n_components

        sorted_data, _ = torch.sort(windows, 1)
        quantiles = torch.linspace(0, 1, self.n_components + 2)[1:-1]
        indices = (quantiles * (self.series_length - 1)).long()
        mu = sorted_data[:, indices]

        typical_gap = torch.std(windows, 1, keepdim=True)

        sigma = typical_gap.view(-1, 1, 1).expand((-1, 1, self.n_components))
        args = {'mu': mu.unsqueeze(1), 'sigma': sigma}
        if self._m_step == self._m_step_student:
            nu = torch.ones((windows.shape[0], 1, self.n_components), dtype=torch.float32, device=self.device) * 4
            args['nu'] = nu
        return w, args

    def _get_weights(self, exp_smooth):
        assert 0 < exp_smooth <= 1.0, "Invalid exp_smooth parameter"
        return exp_smooth ** torch.arange(self.series_length - 1, -1, -1, dtype=torch.float32, device=self.device).reshape((1, self.series_length, 1))

    def _s_step(self, g):
        component = torch.multinomial(g.reshape((-1, self.n_components)), 1).to(self.device)
        y = one_hot(component, num_classes=self.n_components) \
            .reshape(
                (-1, self.series_length, self.n_components)
                ) * self.weights
        v = y.sum(dim=1, keepdims=True)
        return y, v
    
    def _e_step(self, windows, w, **args):
        log_density = self._log_pdf(windows, **args)
        log_weights = torch.log(w + self.eps)
        log_mixture = log_density + log_weights
        log_normalization = torch.logsumexp(log_mixture, dim=2, keepdim=True)
        return torch.exp(log_mixture - log_normalization)

    def _m_step_normal(self, windows, y, v, **args):
        w = (v + self.prior_strength)
        w = w / w.sum(dim=2, keepdim=True)
        data_mu = torch.sum(y * windows, dim=1, keepdim=True) / torch.clamp(v, min=self.eps)

        prior_mu = torch.mean(windows, dim=1, keepdim=True).expand(-1, -1, self.n_components)
        blend_weight = v / (v + self.prior_strength)
        mu = data_mu * blend_weight + prior_mu * (1 - blend_weight)

        diff = windows - mu
        prior_var = torch.var(windows, dim=1, keepdim=True).expand(-1, -1, self.n_components)
        
        data_variance = torch.sum(y * (diff ** 2), dim=1, keepdim=True) / torch.clamp(v, min=self.eps)
        blended_variance = (data_variance * v + prior_var * self.prior_strength) / (v + self.prior_strength)
        
        blended_std = torch.sqrt(blended_variance + self.eps)
        return w, {'mu': mu, 'sigma': blended_std}

    def _m_step_logistic(self, windows, y, v, **args):
        w = (v + self.prior_strength)
        w = w / w.sum(dim=2, keepdim=True)

        data_mu = torch.sum(y * windows, dim=1, keepdim=True) / torch.clamp(v, min=self.eps)

        prior_mu = torch.mean(windows, dim=1, keepdim=True).expand(-1, -1, self.n_components)
        blend_weight = v / (v + self.prior_strength)
        mu = data_mu * blend_weight + prior_mu * (1 - blend_weight)

        diff = windows - mu
        data_variance = torch.sum(y * (diff ** 2), dim=1, keepdim=True) / torch.clamp(v, min=self.eps)
        prior_var = torch.var(windows, dim=1, keepdim=True).expand(-1, -1, self.n_components)

        blended_variance = (data_variance * v + prior_var * self.prior_strength) / (v + self.prior_strength)

        blended_s = torch.sqrt(3 * blended_variance + self.eps) / torch.pi
        return w, {'mu': mu, 'sigma': blended_s}

    def _m_step_laplace(self, windows, y, v, **args):
        batch_size, series_len, n_comp = y.shape
        w = (v + self.prior_strength)
        w = w / w.sum(dim=2, keepdim=True)

        y_flat = y.permute(0, 2, 1).reshape(-1, series_len)
        windows_expanded = windows.expand(-1, -1, n_comp)
        windows_flat = windows_expanded.permute(0, 2, 1).reshape(-1, series_len)

        sorted_vals, sort_idx = torch.sort(windows_flat, dim=1)
        batch_indices = torch.arange(y_flat.shape[0], device=self.device).unsqueeze(1)
        sorted_weights = y_flat[batch_indices, sort_idx]

        cum_weights = torch.cumsum(sorted_weights, dim=1)
        total_weights = cum_weights[:, -1:]
        median_mask = cum_weights >= (total_weights / 2)
        median_idx = torch.argmax(median_mask.float(), dim=1)
        median_vals = sorted_vals[batch_indices.squeeze(), median_idx]

        data_mu = median_vals.view(batch_size, n_comp).unsqueeze(1)
        prior_mu = torch.median(windows, dim=1, keepdim=True).values.expand(-1, -1, n_comp)
        blend_weight = v / (v + self.prior_strength)
        mu = data_mu * blend_weight + prior_mu * (1 - blend_weight)

        abs_diff = torch.abs(windows - mu)
        weighted_abs_diff = torch.sum(y * abs_diff, dim=1, keepdim=True)
        data_sigma = weighted_abs_diff / torch.clamp(v, min=self.eps)

        global_median = torch.median(windows, dim=1, keepdim=True).values
        global_abs_dev = torch.abs(windows - global_median)
        prior_mad = torch.median(global_abs_dev, dim=1, keepdim=True).values
        prior_sigma = prior_mad.expand(-1, -1, n_comp)

        sigma = data_sigma * blend_weight + prior_sigma * (1 - blend_weight)
        sigma = torch.clamp(sigma, min=self.eps)
        
        return w, {'mu': mu, 'sigma': sigma}

    def _m_step_student(self, windows, y, v, mu, sigma, nu, **args):        
        batch_size, series_len, n_comp = y.shape
        n = series_len
        
        assigned_components = torch.argmax(y, dim=2)
        batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, series_len)
        
        assigned_mu = mu[batch_idx, :, assigned_components]
        assigned_sigma = sigma[batch_idx, :, assigned_components]
        assigned_nu = nu[batch_idx, :, assigned_components]
        
        diff_assigned = windows - assigned_mu
        mahalanobis_sq = (diff_assigned / (assigned_sigma + self.eps)) ** 2
        
        ai = (assigned_nu + 1) / 2
        bi = assigned_nu/2 + 0.5 * mahalanobis_sq
        ci = (bi / (ai + self.eps))
        
        w = (v + self.prior_strength)
        w = w / w.sum(dim=2, keepdim=True)
        
        ci_expanded = ci.expand(-1, -1, n_comp)
        y_weighted = y * ci_expanded
        
        numerator_mu = torch.sum(y_weighted * windows, dim=1, keepdim=True)
        denominator_mu = torch.sum(y_weighted, dim=1, keepdim=True)
        data_mu = numerator_mu / torch.clamp(denominator_mu, min=self.eps)
        
        prior_mu = torch.mean(windows, dim=1, keepdim=True).expand(-1, -1, n_comp)
        blend_weight = v / (v + self.prior_strength)
        mu = data_mu * blend_weight + prior_mu * (1 - blend_weight)
        
        diff_new = windows - mu
        weighted_sq_diff = ci_expanded * (diff_new ** 2)
        sigma_sq = torch.sum(weighted_sq_diff, dim=1, keepdim=True) / n
        
        data_sigma = torch.sqrt(sigma_sq + self.eps)
        prior_var = torch.var(windows, dim=1, keepdim=True)
        
        prior_sigma_sq = prior_var * (nu - 2) / torch.clamp(nu, min=2.1)
        prior_sigma = torch.sqrt(torch.clamp(prior_sigma_sq, min=self.eps))
        
        sigma = data_sigma * blend_weight + prior_sigma * (1 - blend_weight)
        sigma = torch.clamp(sigma, min=self.eps)
        
        standardized = diff_new / (sigma + self.eps)
        weighted_k4 = torch.sum(y * standardized**4, dim=1, keepdim=True)
        weighted_k2 = torch.sum(y * standardized**2, dim=1, keepdim=True)
        
        mask = v > 2
        
        if mask.any():
            kurtosis = weighted_k4[mask] / (weighted_k2[mask]**2 + self.eps) - 3
            kurtosis = torch.clamp(kurtosis, min=0.01, max=100.0)
            nu_update = torch.clamp(4 + 6 / (kurtosis + self.eps), min=2.1, max=30.0)
            nu[mask] = 0.8 * nu[mask] + 0.2 * nu_update
        
        return w, {'mu': mu, 'sigma': sigma, 'nu': nu}

    def _get_params_norm(self, **args):
        res = 0
        for name, value in args.items():
            res += torch.norm(value, 1).item()
        return res

    def _fit_batch(self, windows, max_iters_one_series=5):
        w, args = self._initialize_parameters(windows)
        prev_params_norm = self._get_params_norm(**args)
        d = 1 + 2 * self.tol
        i = 0
        windows_exp = windows.unsqueeze(-1)
        while (abs(d - 1) > self.tol) and ((i < max_iters_one_series) if windows.shape[0] == 1 else True):
            g = self._e_step(windows_exp, w, **args)
            y, v = self._s_step(g)
            w, args = self._m_step(windows_exp, y, v, **args)
            print(torch.isnan(args['sigma']).any())
            curr_params_norm = self._get_params_norm(**args)
            d = curr_params_norm / prev_params_norm
            prev_params_norm = curr_params_norm
            i += 1
        return g, w, args

    def _create_windows(self):
        windows = create_sliding_windows(
            self.time_series[self.start: self.finish], self.series_length) \
                .to(self.device)
        return windows

    def find_params(self):
        g, w, args = self._fit_batch(self._create_windows())
        args['g'] = g
        args['w'] = w

        sorter = torch.argsort(args['mu'], dim=-1)

        args['mu'] = torch.gather(args['mu'], -1, sorter)
        args['sigma'] = torch.gather(args['sigma'], -1, sorter)
        args['w'] = torch.gather(args['w'], -1, sorter)
        args['g'] = torch.gather(args['g'], -1, sorter.expand(-1, args['g'].shape[1], -1))
        if 'nu' in args:
            args['nu'] = torch.gather(args['nu'], -1, sorter)
        return args

    def batched_mixture_multiple_percentiles(self, w, p_values, **args):
        batch_size = len(w)
        mu = args['mu']
        n_percentiles = len(p_values)

        overall_means = (w * mu).sum(dim=2, keepdims=True)
        overall_stds = torch.sqrt((w * (mu - overall_means)**2).sum(dim=2, keepdim=True))

        lower = (overall_means - 4.0 * overall_stds).reshape(-1, 1, 1)
        upper = (overall_means + 4.0 * overall_stds).reshape(-1, 1, 1)

        lower = lower.expand(batch_size, n_percentiles, 1)
        upper = upper.expand(batch_size, n_percentiles, 1)
        p_values = p_values.unsqueeze(0).unsqueeze(2).expand(batch_size, n_percentiles, 1)

        while (upper - lower).max() > self.tol:
            mid = (lower + upper) / 2
            component_cdfs = self._cdf(mid, **args)
            mixture_cdf = (w * component_cdfs).sum(dim=2, keepdim=True)
            lower = torch.where(mixture_cdf < p_values, mid, lower)
            upper = torch.where(mixture_cdf >= p_values, mid, upper)
        return (lower + upper) / 2

    def batched_mixture_mode(self, w, **args):
        def compute_gradient(x):
            eps = 1e-6
            x_plus = x + eps
            x_minus = x - eps

            log_density_plus = self._log_pdf(x_plus.expand(-1, -1, self.n_components), **args)
            log_mixture_plus = torch.log(w) + log_density_plus
            density_plus = torch.exp(torch.logsumexp(log_mixture_plus, dim=-1, keepdim=True))

            log_density_minus = self._log_pdf(x_minus.expand(-1, -1, self.n_components), **args)
            log_mixture_minus = torch.log(w) + log_density_minus
            density_minus = torch.exp(torch.logsumexp(log_mixture_minus, dim=-1, keepdim=True))

            return (density_plus - density_minus) / (2 * eps)

        def find_peak_from_start(x_start):
            grad = compute_gradient(x_start)
            direction = torch.sign(grad)

            x = x_start.clone()
            mask_not_found = torch.ones_like(direction, dtype=torch.bool)
            alpha = 1e-3

            while torch.any(mask_not_found):
                x = x + alpha * direction * mask_not_found.float()
                grad_new = compute_gradient(x)
                mask_not_found = (grad_new * direction) > 0
            return x
        
        mu = args['mu']

        candidates = []
        for k in range(self.n_components):
            candidates.append(find_peak_from_start(mu[:, :, k:k+1]))

        densities = []
        for candidate in candidates:
            log_density_at_candidate = self._log_pdf(candidate.expand(-1, -1, self.n_components), **args)
            log_mixture_at_candidate = torch.log(w + 1e-8) + log_density_at_candidate
            density_at_candidate = torch.exp(torch.logsumexp(log_mixture_at_candidate, dim=-1))
            densities.append(density_at_candidate)

        densities = torch.cat(densities, dim=-1)
        best_idx = torch.argmax(densities, dim=-1, keepdim=True).unsqueeze(-1)
        candidates_stacked = torch.cat(candidates, dim=-1)
        best_mode = torch.gather(candidates_stacked, -1, best_idx)

        return best_mode
