import torch
import numpy as np

MAX_GPU_OCCUPANCE = 2 * 1024 * 1024 * 1024


def calculate_acf(diffs):
    n_series, n_diffs = diffs.shape

    if n_diffs < 2:
        return torch.zeros(n_series, device=diffs.device, dtype=diffs.dtype)

    diff_prev = diffs[:, :-1]
    diff_next = diffs[:, 1:]

    mean_prev = torch.mean(diff_prev, dim=1)
    mean_next = torch.mean(diff_next, dim=1)
    mean_prev_sq = torch.mean(diff_prev ** 2, dim=1)
    mean_next_sq = torch.mean(diff_next ** 2, dim=1)
    mean_product = torch.mean(diff_prev * diff_next, dim=1)

    covariance = mean_product - mean_prev * mean_next
    var_prev = mean_prev_sq - mean_prev ** 2
    var_next = mean_next_sq - mean_next ** 2

    denominator = torch.sqrt(var_prev * var_next)
    denominator = torch.where(denominator < 1e-10, torch.tensor(1e-10, device=denominator.device), denominator)

    acf1 = torch.clamp(covariance / denominator, -1.0, 1.0)
    return acf1


class Windows:
    def create_sliding_windows(self, series, window_size):
        if isinstance(series, np.ndarray):
            series = torch.from_numpy(series).to(torch.float32)
        series_len = len(series)
        n_windows = series_len - window_size + 1
        indices = torch.arange(window_size).unsqueeze(0) + torch.arange(n_windows).unsqueeze(1)
        windows = series[indices]
        return windows

    def _stationarity_check(self, series, window_size, alpha=0.5, max_memory=MAX_GPU_OCCUPANCE):
        batch_size = max_memory // (4 * window_size)
        limit, rem = divmod(len(series) - window_size + 1, batch_size)
        if rem > 0:
            limit += 1
        stationary_mask = True
        max_acf = 0
        for i in range(limit):
            series_array = self.create_sliding_windows(series[i * batch_size: (i+1) * batch_size + window_size - 1], window_size)
            diffs = torch.diff(series_array, dim=1)
            acf = torch.abs(calculate_acf(diffs))
            _max_acf = torch.max(acf)
            if _max_acf > max_acf:
                max_acf = _max_acf

            stationary_mask &= torch.all(acf < alpha)
        print(f"N = {window_size}; Max ACF(1): {max_acf.item()}")
        return stationary_mask

    def _find_N(self, series, alpha=0.5, N_init=100, N_add=10, max_memory=MAX_GPU_OCCUPANCE):
        """
        Find N for which the series is stationary
        """
        while not self._stationarity_check(series, N_init, alpha, max_memory):
            N_init += N_add
            if N_init > len(series):
                print(f"Warning: N_init ({N_init}) exceeds series length ({len(series)})")
                break

        return N_init

    def __call__(self, series, alpha=0.5, N_init=100, N_add=10, max_gpu_memory=MAX_GPU_OCCUPANCE, max_res_memory=1024*1024*1024):
        """
        Find N for which acf(1) < alpha and return slicing parameters:
        start, finish, increment, window_size.

        create_slicing_windows(series[start:finish], window_size)
        start += increment
        finish += increment
        """
        if isinstance(series, np.ndarray):
            series = torch.from_numpy(series).to(torch.float32)
        N = self._find_N(series, alpha, N_init, N_add, max_gpu_memory)
        print("Found window length:", N)
        batch_size = max_res_memory // (4 * N)
        limit, rem = divmod(len(series) - N + 1, batch_size)
        if rem > 0:
            limit += 1

        start = 0
        finish = batch_size + N - 1
        return start, finish, batch_size, N
