# https://ijassa.ipu.ru/index.php/ijassa/article/view/899/538


import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path

from sem.generate_series import create_sde_process
from sem.sem.mixture_em_diff import NormalMixtureEM
from sem.sem.mixture_sem import MixtureSEM
from sem.sem.windows import find_window_size

from statsmodels.tsa.arima.model import ARIMA
from itertools import product
from typing import Union, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ForecastingMixin:
    """
    Mixin class providing multi-step forecasting methods.
    Models should inherit from this mixin and nn.Module.
    """
    
    def stateless_forward_multistep(self, windows: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Autoregressive multi-step forecasting for stateless models.
        
        Args:
            windows: [batch_size, window_size] - input windows
            n_steps: number of steps to forecast
            
        Returns:
            predictions: [batch_size, n_steps]
        """
        forecasts = []
        for _ in range(n_steps):
            forecast = self.forward(windows)
            forecasts.append(forecast)

            windows = torch.cat([
                windows[:, 1:],
                forecast.unsqueeze(-1)
            ], dim=1)
        results = torch.stack(forecasts, dim=1)
        return results
    
    @torch.no_grad()
    def generate_forecast(self, idx: int, 
                         time_series: torch.Tensor,
                         n_test_steps: int,
                         window_size: int) -> torch.Tensor:
        """
        Generate forecast starting at given index.
        
        Args:
            idx: starting index (negative values count from end)
            time_series: full time series data
            n_test_steps: number of steps to forecast
            window_size: size of input window
            train_series: alias for time_series (for backward compatibility)
            
        Returns:
            predictions: forecast values
        """
        if idx < 0:
            idx = time_series.shape[-1] + idx
        idx += 1

        start_idx = idx - window_size
        ts = time_series[..., start_idx: idx]
        assert ts.shape[-1] == window_size, 'not enough data'

        if not isinstance(ts, torch.Tensor):
            ts = torch.tensor(ts, dtype=torch.float32)
        if ts.dim() == 1:
            ts = ts.unsqueeze(0)

        device = next(self.parameters()).device
        window = ts.to(device)

        preds = self.forward_multistep(window, n_test_steps)
        if ts.shape[0] == 1:
            preds = preds.squeeze(0)
            
        return preds
    
    def get_name(self, add: str = "") -> str:
        """
        Get model name with optional suffix.
        
        Args:
            add: string to append to model name
            
        Returns:
            Model class name with suffix
        """
        class_name = self.__class__.__name__
        return class_name + add

    def forward_multistep(self, windows: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Default multi-step forecasting implementation.
        Can be overridden by subclasses for stateful models.
        
        Args:
            windows: [batch_size, window_size]
            n_steps: number of steps to forecast
            
        Returns:
            predictions: [batch_size, n_steps]
        """
        return self.stateless_forward_multistep(windows, n_steps)


class MLP(ForecastingMixin, nn.Module):
    def __init__(self, window_size, hidden_dim=16):
        super().__init__()
        self.window_size = window_size
        self.dwindow_size = window_size - 1
        negative_slope = 0.01
        
        self.mlp = nn.Sequential(
            nn.Linear(self.dwindow_size, hidden_dim, bias=False),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=negative_slope)

    def forward(self, x):
        dx = x[:, 1:] - x[:, :-1]
        delta = self.mlp(dx).squeeze(-1)
        return x[:, -1] + delta


class UltraLightAttention(ForecastingMixin, nn.Module):
    def __init__(self, window_size, embed_dim=8):
        super().__init__()
        self.window_size = window_size
        self.dwindow_size = window_size - 1
        negative_slope = 0.01

        self.embed = nn.Conv1d(1, embed_dim, kernel_size=3, padding=1, bias=False)
        self.position_enc = nn.Parameter(torch.randn(1, self.dwindow_size, embed_dim))

        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads=1, dropout=0.05, 
            bias=False, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.predictor = nn.Sequential(
            nn.Linear(embed_dim + self.dwindow_size, embed_dim, bias=False),
            nn.LeakyReLU(negative_slope),
            nn.Linear(embed_dim, embed_dim // 2, bias=False),
            nn.LeakyReLU(negative_slope),
            nn.Linear(embed_dim // 2, 1, bias=False)
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, negative_slope)
    
    def forward(self, x):
        dx = x[:, 1:] - x[:, :-1]

        dx_embedded = self.embed(dx.unsqueeze(1)).transpose(1, 2)
        dx_embedded = dx_embedded + self.position_enc

        attn_out, _ = self.attn(dx_embedded, dx_embedded, dx_embedded)
        attn_out = self.norm(dx_embedded + attn_out)

        attn_features = attn_out.mean(dim=1)

        delta = self.predictor(torch.cat([dx, attn_features], dim=1)).squeeze(-1)
        
        return x[:, -1] + delta


class EMForecaster(ForecastingMixin, nn.Module):
    def __init__(self, 
                 window_size: int,
                 n_components: int = 3,
                 hidden_dim: int = 16,
                 n_em_iters: int = 5,
                 exp_smooth: float = 1.0):
        super().__init__()
        
        self.dwindow_size = window_size - 1
        self.window_size = window_size
        self.n_components = n_components
        negative_slope = 0.01

        self.em_layer = NormalMixtureEM(
            series_length=self.dwindow_size,
            n_components=n_components,
            n_em_iters=n_em_iters,
            exp_smooth=exp_smooth
        )

        fusion_input_dim = self.dwindow_size + n_components * 5
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim, bias=False),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, negative_slope)

    def forward(self, X_window):
        dX1 = X_window[..., 1:] - X_window[..., :-1]
        _, w, mu, sigma = self.em_layer(dX1)

        fusion_input = torch.cat([
            dX1,
            mu,
            sigma,
            w,
            mu * w,
            sigma * sigma * w
        ], dim=1)
        forecast = X_window[..., -1] + self.fusion(fusion_input).squeeze(-1)
        
        return forecast


class SEMForecaster(ForecastingMixin, nn.Module):
    def __init__(self, 
                 window_size: int,
                 sem_method: MixtureSEM,
                 n_components: int = 3,
                 hidden_dim: int = 16):
        super().__init__()
        
        self.dwindow_size = window_size - 1
        self.window_size = window_size
        self.n_components = n_components
        self.sem_method = sem_method
        negative_slope = 0.01

        fusion_input_dim = self.dwindow_size + n_components * 5
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim, bias=False),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, negative_slope)

    def forward(self, X_window, w, mu, sigma):
        dX1 = X_window[..., 1:] - X_window[..., :-1]

        fusion_input = torch.cat([
            dX1,
            mu,
            sigma,
            w,
            mu * w,
            sigma * sigma * w
        ], dim=1)
        forecast = X_window[..., -1] + self.fusion(fusion_input).squeeze(-1)
        
        return forecast

    def forward_multistep(self, windows: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Autoregressive multi-step forecasting for stateless models.
        
        Args:
            windows: [batch_size, window_size] - input windows
            n_steps: number of steps to forecast
            
        Returns:
            predictions: [batch_size, n_steps]
        """
        forecasts = []
        for _ in range(n_steps):
            dwindows = windows[:, 1:] - windows[:, :-1]
            _, w, args = self.sem_method(dwindows)
            w = w.squeeze(1)
            mu = args['mu'].squeeze(1)
            sigma = args['sigma'].squeeze(1)

            forecast = self.forward(windows, w, mu, sigma)
            forecasts.append(forecast)

            windows = torch.cat([
                windows[:, 1:],
                forecast.unsqueeze(-1)
            ], dim=1)
        results = torch.stack(forecasts, dim=1)
        return results


class ARIMAForecaster:
    def __init__(self, 
                 p_range: range = range(0, 7),
                 d_range: range = range(0, 2),
                 q_range: range = range(0, 5),
                 trend: str = 'c'):

        self.p_range = p_range
        self.d_range = d_range
        self.q_range = q_range
        self.trend = trend
        self.best_model_ = None
        
    def _grid_search(self, time_series: np.ndarray) -> ARIMA:

        best_score = np.inf
        best_model = None
        
        for p, d, q in product(self.p_range, self.d_range, self.q_range):
            try:
                model = ARIMA(time_series, order=(p, d, q), trend=self.trend)
                fitted = model.fit()
                if fitted.aic < best_score:
                    best_score = fitted.aic
                    best_model = fitted
            except:
                continue
        if best_model is None:
            raise RuntimeError("No valid model found during grid search")
        return best_model
    
    def fit_predict(self, 
                    time_series: np.ndarray, 
                    n_pred_steps: int) -> np.ndarray:
        self.best_model_ = self._grid_search(time_series)
        return self.best_model_.forecast(steps=n_pred_steps)
    
    def save_results(self, 
                     gt: Union[List, np.ndarray], 
                     predictions: Union[List, np.ndarray],
                     filename: str) -> np.ndarray:

        gt = np.array(gt)
        predictions = np.array(predictions)
        
        if gt.shape != predictions.shape:
            raise ValueError("Shape mismatch between predictions and ground truth")
            
        result = np.column_stack((predictions, gt))
        np.save(filename, result)



from time import time
import json

TOL = 1e-4
TOL_TEST = 1e-4

class TimeSeriesDataset(Dataset):
    def __init__(self, time_series: np.ndarray, window_size, n_train_steps=1):
        super().__init__()
        self.ts = time_series
        self.n_train_steps = n_train_steps
        self.window_length = window_size

    def __len__(self):
        return len(self.ts) - self.window_length - self.n_train_steps

    def __getitem__(self, idx):
        idx_last = idx + self.window_length
        return self.ts[idx:idx_last], self.ts[idx_last: idx_last + self.n_train_steps]

class StaticMixtureTimeSeriesDataset(Dataset):
    def __init__(self, time_series: np.ndarray, window_size: int, w, mu, sigma, n_train_steps=1):
        super().__init__()
        self.ts = time_series
        self.n_train_steps = n_train_steps
        self.window_length = window_size
        self.w, self.mu, self.sigma = w, mu, sigma

    def __len__(self):
        return len(self.ts) - self.window_length - self.n_train_steps

    def __getitem__(self, idx):
        idx_last = idx + self.window_length
        target = self.ts[idx_last: idx_last + self.n_train_steps]
        window = self.ts[idx:idx_last]

        return window, target, self.w[idx], self.mu[idx], self.sigma[idx]


class HuberLoss(nn.Module):
    def __init__(self, delta=2.5):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        quadratic = torch.clamp(diff, max=self.delta)
        linear = diff - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear
        return loss.mean()


class Trainer:
    def __init__(self, setup, time_series, window_size, len_train, log_path,
                 log_freq_batch=50, n_tests=100, n_test_steps=50, n_train_steps=1, eps=1e-8, device='cuda'):
        self.train_series = time_series[:len_train]
        self.eval_series = time_series[len_train:]
        self.dataloader = setup['dataloader']

        self.window_size = window_size
        self.n_epochs = 50
        self.n_tests = n_tests
        self.n_test_steps = n_test_steps
        self.model = setup['model'].to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), setup['lr'])
        self.criterion = HuberLoss(setup['delta'])
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, 1.0, 0.1, self.n_epochs
        )

        self.loss_threshold = 0.01
        self.eps = eps
        self.device = device
        self.log_freq_batch = log_freq_batch
        self.log_path = log_path
        
        self.n_train_steps = n_train_steps

    @torch.no_grad()
    def validate(self):
        preds = self.model.generate_forecast(
            idx=-1,
            time_series=self.train_series,
            n_test_steps=self.n_test_steps,
            window_size=self.window_size
        ).cpu().numpy()
        target = self.eval_series[:self.n_test_steps]
        return preds, target
    
    @torch.no_grad()
    def evaluate_multistep(self, test_ids):
        preds = np.empty((self.n_tests + 1, self.n_test_steps))
        targets = np.empty((self.n_tests + 1, self.n_test_steps))
        for test in range(self.n_tests):
            idx = test_ids[test]
            target_idx = idx + 1
            pred = self.model.generate_forecast(
                idx=idx,
                time_series=self.train_series,
                n_test_steps=self.n_test_steps,
                window_size=self.window_size
            ).cpu()
            target = self.train_series[target_idx:target_idx + self.n_test_steps]
            preds[test, :] = pred.numpy()
            targets[test, :] = target
        return preds, targets

    def run(self, test_ids):
        epoch_train_time = 0
        epoch_val_time = 0
        start_full = time()
        for epoch in range(1, self.n_epochs + 1):
            self.model.train()
            start_train = time()
            avg_threshold = self.train_epoch(epoch)
            epoch_train_time += time() - start_train
            self.model.eval()

            start_test = time()
            preds, targets = self.evaluate_multistep(test_ids)
            pred, target = self.validate()
            epoch_val_time += time() - start_test
            preds[-1, :] = pred
            targets[-1, :] = target
            print("Test MAPE:", (np.abs(pred - target) / (np.abs(target) + self.eps)).mean() * 100)
            tests_results = np.stack([preds, targets], axis=-1)  # shape: [n_tests+1, n_steps, 2]
            
            np.save(self.log_path / f'epoch_{epoch}', tests_results)
            
            if avg_threshold < self.loss_threshold:
                break
        
        time_full = time() - start_full
        mean_train_time = epoch_train_time / epoch
        mean_test_time = epoch_val_time / epoch / (self.n_test_steps + 1) / (self.n_tests)

        json.dump(
                {
                    'full_time': float(time_full),
                    'epoch_train_time': float(mean_train_time),
                    'epoch_test_time_one_step': float(mean_test_time),
                    'last_loss': float(avg_threshold)
                },
                open(self.log_path / 'times.json', 'w'),
                indent=2
            )
        print()
        print("Training completed!")
        return self.model
    
    def train_epoch(self, epoch_idx=0):
        print("Epoch:", epoch_idx)
        total_loss = 0.0

        for batch_idx, batch in enumerate(self.dataloader):
            batch = list(batch)
            for i in range(len(batch)):
                batch[i] = batch[i].to(self.device)
            target = batch.pop(1).float()
            windows = batch.pop(0).float()
            if self.n_train_steps == 1:
                target = target.squeeze(-1)

            self.optimizer.zero_grad()

            if self.n_train_steps == 1:
                results = self.model.forward(windows, *batch)
                if isinstance(results, tuple):
                    results = results[0]
            else:
                results = self.model.forward_multistep(windows, self.n_train_steps)

            loss = self.criterion(results, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % self.log_freq_batch == 0:
                basic_metrics = self.compute_basic_metrics(results, target)
                print("Batch IDX:", batch_idx)
                print(basic_metrics)
                print()
            if torch.isnan(loss):
                print(f"WARNING: NaN loss at batch {batch_idx}")
                break
        self.scheduler.step()

        avg_loss = total_loss / len(self.dataloader)
        print(f"Epoch {epoch_idx} - Average Loss: {avg_loss:.6f}")
        return avg_loss

    @torch.no_grad()
    def compute_basic_metrics(self, forecast, y_true):
        mae = torch.abs(forecast - y_true).mean().item()
        mse = torch.mean((forecast - y_true) ** 2).item()
        mape = torch.abs((forecast - y_true) / (y_true.abs() + self.eps)).mean().item() * 100
        return {'MAE': mae, 'MSE': mse, 'MAPE': mape}



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_setups(train_series, window_size, batch_size, device='cuda'):
    dwindow_size = window_size - 1
    sem_models = {
        'SEM_normal_2comp': MixtureSEM(train_series, dwindow_size, n_components=2, comp_distr='normal', exp_smooth=1.0, tol=TOL_TEST, device=device),
        'SEM_student_2comp': MixtureSEM(train_series, dwindow_size, n_components=2, comp_distr='student', exp_smooth=1.0, tol=TOL_TEST, device=device),
        'SEM_laplace_2comp': MixtureSEM(train_series, dwindow_size, n_components=2, comp_distr='laplace', exp_smooth=1.0, tol=TOL_TEST, device=device),
        'SEM_logistic_2comp': MixtureSEM(train_series, dwindow_size, n_components=2, comp_distr='logistic', exp_smooth=1.0, tol=TOL_TEST, device=device),
        'SEM_normal_2comp_exp': MixtureSEM(train_series, dwindow_size, n_components=2, comp_distr='normal', exp_smooth=0.7, tol=TOL_TEST, device=device),
        'SEM_student_2comp_exp': MixtureSEM(train_series, dwindow_size, n_components=2, comp_distr='student', exp_smooth=0.7, tol=TOL_TEST, device=device),
        'SEM_laplace_2comp_exp': MixtureSEM(train_series, dwindow_size, n_components=2, comp_distr='laplace', exp_smooth=0.7, tol=TOL_TEST, device=device),
        'SEM_logistic_2comp_exp': MixtureSEM(train_series, dwindow_size, n_components=2, comp_distr='logistic', exp_smooth=0.7, tol=TOL_TEST, device=device),
        'SEM_normal_3comp': MixtureSEM(train_series, dwindow_size, n_components=3, comp_distr='normal', exp_smooth=1.0, tol=TOL_TEST, device=device),
        'SEM_student_3comp': MixtureSEM(train_series, dwindow_size, n_components=3, comp_distr='student', exp_smooth=1.0, tol=TOL_TEST, device=device),
        'SEM_laplace_3comp': MixtureSEM(train_series, dwindow_size, n_components=3, comp_distr='laplace', exp_smooth=1.0, tol=TOL_TEST, device=device),
        'SEM_logistic_3comp': MixtureSEM(train_series, dwindow_size, n_components=3, comp_distr='logistic', exp_smooth=1.0, tol=TOL_TEST, device=device),
        'SEM_normal_3comp_exp': MixtureSEM(train_series, dwindow_size, n_components=3, comp_distr='normal', exp_smooth=0.7, tol=TOL_TEST, device=device),
        'SEM_student_3comp_exp': MixtureSEM(train_series, dwindow_size, n_components=3, comp_distr='student', exp_smooth=0.7, tol=TOL_TEST, device=device),
        'SEM_laplace_3comp_exp': MixtureSEM(train_series, dwindow_size, n_components=3, comp_distr='laplace', exp_smooth=0.7, tol=TOL_TEST, device=device),
        'SEM_logistic_3comp_exp': MixtureSEM(train_series, dwindow_size, n_components=3, comp_distr='logistic', exp_smooth=0.7, tol=TOL_TEST, device=device)
    }

    setups = []

    for name, sem in sem_models.items():
        args = sem.find_params()
        setups.append(
            {
                'name': name,
                'model': SEMForecaster(window_size, sem._fit_batch, sem.n_components),
                'dataloader': DataLoader(
                    StaticMixtureTimeSeriesDataset(train_series, window_size, args['w'].squeeze(1), args['mu'].squeeze(1), args['sigma'].squeeze(1)),
                    batch_size,
                    shuffle=True
                )
            })
    
    setups = setups + \
        [{
            'name': 'EM_2comp',
            'model': EMForecaster(window_size, 2, exp_smooth=1.0),
            'dataloader': DataLoader(TimeSeriesDataset(train_series, window_size), batch_size, shuffle=True)
        },
        {
            'name': 'EM_2comp_exp',
            'model': EMForecaster(window_size, 2, exp_smooth=0.7),
            'dataloader': DataLoader(TimeSeriesDataset(train_series, window_size), batch_size, shuffle=True)
        },
        {
            'name': 'EM_3comp',
            'model': EMForecaster(window_size, 3, exp_smooth=1.0),
            'dataloader': DataLoader(TimeSeriesDataset(train_series, window_size), batch_size, shuffle=True)
        },
        {
            'name': 'EM_3comp_exp',
            'model': EMForecaster(window_size, 3, exp_smooth=0.7),
            'dataloader': DataLoader(TimeSeriesDataset(train_series, window_size), batch_size, shuffle=True)
        }]

    setups = setups + \
        [{
            'name': 'MLP',
            'model': MLP(window_size),
            'dataloader': DataLoader(TimeSeriesDataset(train_series, window_size), batch_size, shuffle=True)
        },
        {
            'name': 'Attention',
            'model': UltraLightAttention(window_size),
            'dataloader': DataLoader(TimeSeriesDataset(train_series, window_size), batch_size, shuffle=True)
        }]
    
    setups = setups + \
        [{
            'name': 'ARIMA',
            'model': ARIMAForecaster()
        }]
    
    lrs = [5e-3, 5e-3, 1.5e-3, 1e-3, 1.5e-3, 1.5e-3, 1.5e-3, 1.5e-3, 3e-3, 3e-3, 1e-3, 5e-4, 1.25e-3, 1.25e-3, 1.25e-3, 1.25e-3,
           4e-3, 3e-3, 2e-3, 1.5e-3, 2e-3, 1.5e-3, -1]
    
    deltas = [2.75, 5.0, 2., 2.5, 2.25, 4.0, 1.5, 2., 2.75, 5.0, 2., 2.5, 2.25, 4.0, 1.5, 2.,
             2.5, 2., 2.5, 2., 1.0, 1.0, -1]
    
    for i in range(len(setups)):
        setups[i]['lr'] = lrs[i]
        setups[i]['delta'] = deltas[i]

    return setups

def create_experiment_folder(base_path="experiments", model_name=None, series_idx=None):
    """
    Create folder structure: experiments/<model_name>/<series_idx>/
    """
    if model_name is None:
        model_name = "default"
    if series_idx is None:
        series_idx = "0"
    
    exp_path = Path(base_path) / model_name / str(series_idx)
    exp_path.mkdir(parents=True, exist_ok=True)
    
    return exp_path

def generate_test_ids(window_size, train_series, n_tests, n_test_steps):
    res = []
    for _ in range(n_tests):
        res.append(random.randint(window_size, len(train_series) - 1 - n_test_steps))
    return res


n_tests = 10
series_length = 1150
train_length = 1000
alpha = 0.7
N_init = 5
N_add = 5
n_epoch_tests = 2
n_test_steps = 150
batch_size = 64
device = 'cuda'

for test in range(n_tests):
    print('_' * 50)
    print(test)
    print('_' * 50)
    set_seed(10 * test)
    series = create_sde_process(series_length)['X']
    train_series = series[:train_length]
    *_, dwindow_size = find_window_size(train_series[1:] - train_series[:-1], alpha, N_init, N_add)
    window_size = dwindow_size + 1
    setups = get_setups(train_series, window_size, batch_size, device)
    test_ids = generate_test_ids(window_size, train_series, n_epoch_tests, n_test_steps)
    for setup in setups:
        set_seed(10 * test)
        print('_' * 50)
        print(setup['name'])
        print('_' * 50)

        log_path = create_experiment_folder(model_name=setup['name'], series_idx=test)
        if len(os.listdir(log_path)):
            print(log_path)
            continue
        if setup['name'] == 'ARIMA':
            model = setup['model']
            forecast = model.fit_predict(train_series, n_test_steps)
            model.save_results(series[train_length:train_length + n_test_steps], forecast, log_path / 'epoch_0')
        else:
            trainer = Trainer(setup, series, window_size, train_length, log_path,
                            log_freq_batch=70, n_tests=n_epoch_tests, n_test_steps=n_test_steps, device=device)
            model = trainer.run(test_ids)