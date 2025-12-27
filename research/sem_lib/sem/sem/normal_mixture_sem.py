import torch
import torch.nn.functional as F
from .windows import Windows


class NormalMixtureSEM:
    def __init__(self, time_series, weights=None, method='EM', n_components=2,
                 tol=1e-3, eps=1e-6, N_init=500, max_iter_sem=100,
                 exp_smooth=0.999, alpha=0.6,
                 p_values=torch.tensor([0.01 * i for i in range(1, 100)]),
                 max_iter_percentile=20, device="cuda"):
        '''
        weights: None/"exp",

        method: "EM"/"SEM"
        '''
        self.eps = eps
        self.time_series = time_series
        self.max_iter = max_iter_sem
        self.max_iter_percentile = max_iter_percentile
        self.p_values = p_values
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
        self._sem_e_step()
        self._em_m_step()

    def _sem_steps(self):
        self._sem_e_step()
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
        self.p = torch.ones(
            (self.n_series, self.n_components), device=self.device
            ) / self.n_components
        indices = torch.randint(
            0, self.series_length, (self.n_series, self.n_components),
            device=self.device
            )
        self.a = self.time_series_window[torch.arange(self.n_series).unsqueeze(1), indices]
        overall_std = self.time_series_window.std(dim=1)
        self.b = torch.ones(
            (self.n_series, self.n_components), device=self.device
            ) * (overall_std + self.eps).unsqueeze(1)

    def _get_weights(self):
        if self.weights_name is None:
            return torch.ones((1, self.series_length), dtype=torch.float32).to(self.device)
        elif self.weights_name == 'exp':
            return torch.Tensor([[self.exp_smooth ** (self.series_length - 1 - i) for i in range(self.series_length)]], device=self.device).to(self.device)

    def _s_step(self):
        component = torch.multinomial(self.g.reshape((-1, self.n_components)) + self.eps, 1).to(self.device)
        y = F.one_hot(component, num_classes=self.n_components) \
            .reshape(
                (self.n_series, self.series_length, self.n_components)
                ) * self.weights.unsqueeze(-1)
        v = y.sum(dim=1)
        v[v == 0] = self.eps
        p_v = (y > 0).sum(dim=1)
        return y, v, p_v

    def _sem_m_step(self, y, v, p_v):
        self.p = p_v / (self.series_length) + self.eps
        self.a = (y * self.time_series_window.unsqueeze(2)).sum(1) / v
        self.b = torch.sqrt((y * (self.time_series_window.unsqueeze(2) - self.a.unsqueeze(1))**2).sum(1) / v) + self.eps

    def _sem_e_step(self):
        self.g = torch.softmax(-0.5 * ((self.time_series_window.unsqueeze(2) - self.a.unsqueeze(1)) /
                                       self.b.unsqueeze(1)) ** 2 +
                               torch.log(self.p).unsqueeze(1) - torch.log(self.b).unsqueeze(1), dim=2)

    def _em_m_step(self):
        g = self.g * self.weights.unsqueeze(-1)
        sum_g = torch.sum(g, dim=1)
        self.p = sum_g
        self.p = self.p / self.p.sum(1).unsqueeze(1)
        self.a = (g * self.time_series_window.unsqueeze(2)).sum(1) / sum_g
        self.b = torch.sqrt((g * (self.time_series_window.unsqueeze(2) - self.a.unsqueeze(1)) ** 2 + self.eps).sum(1) / sum_g)

    def _log_likelihood(self):
        log_probs = -0.5 * ((self.time_series_window.unsqueeze(2) - self.a.unsqueeze(1)) / self.b.unsqueeze(1)) ** 2 \
            + torch.log(self.p).unsqueeze(1) - torch.log(self.b).unsqueeze(1) - 0.5 * torch.log(2 * torch.Tensor([torch.pi], device=self.device))
        log_likelihood_per_point = torch.logsumexp(log_probs, dim=2)
        total_log_likelihood = (log_likelihood_per_point * self.weights).sum(dim=1)
        return total_log_likelihood.sum().item()

    def _fit_batch(self):
        prev_log_likelihood = self._log_likelihood()
        print(0, prev_log_likelihood)

        for iteration in range(self.max_iter):
            self.method()
            curr_log_likelihood = self._log_likelihood()
            print(1 + iteration, curr_log_likelihood)
            if 0 <= (curr_log_likelihood - prev_log_likelihood) / abs(prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = curr_log_likelihood
        return self

    def find_params(self):
        pred_p = torch.Tensor([])
        pred_a = torch.Tensor([])
        pred_b = torch.Tensor([])
        pred_percentiles = torch.Tensor([])
        pred_b_t = torch.Tensor([])
        batch_num = 0
        while self.start < len(self.time_series):
            print("Batch", batch_num)
            batch_num += 1
            self._prepare_fit()
            self._fit_batch()
            p_percentiles = self._batched_mixture_multiple_percentiles()
            p_b_t = self._get_B_t()
            pred_percentiles = torch.concat([pred_percentiles, p_percentiles])
            pred_p = torch.concat([pred_p, self.p])
            pred_a = torch.concat([pred_a, self.a])
            pred_b = torch.concat([pred_b, self.b])
            pred_b_t = torch.concat([pred_b_t, p_b_t])
            self.start += self.increment
            self.finish += self.increment
        return pred_p, pred_a, pred_b, pred_percentiles, pred_b_t

    def _get_B_t(self):
        b_t = torch.sum(self.b * self.b * self.p, dim=1)**0.5
        return b_t.detach().cpu()

    def _batched_mixture_multiple_percentiles(self):
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
            print('percentile optimization', (upper - lower).sum())
        return (lower + upper).squeeze(2) / 2
