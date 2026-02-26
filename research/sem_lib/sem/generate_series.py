import numpy as np
from typing import Dict


def create_fourier_series(
    length: int = 2000,
    min_value: float = -1.0,
    max_value: float = 1.0,
    num_additives: int = 5,
    min_freq: float = 0.1,
    max_freq: float = 10.0,
) -> Dict:
    """
    Create a Fourier series (single regime).
    
    Each additive is of the form: A * sin(2π * freq * t + phase), 
    where t ∈ [0, 1] with `length` steps.
    
    Args:
        length: Number of time points
        min_value: Minimum value for amplitude scaling
        max_value: Maximum value for amplitude scaling
        num_additives: Number of sine wave additives
        min_freq: Minimum frequency
        max_freq: Maximum frequency
    
    Returns:
        Dictionary containing the series, components, and parameters
    """
    t = np.linspace(0, 1, length)
    amplitude_range = (max_value - min_value) / num_additives
    amplitudes = np.random.uniform(
        amplitude_range * 0.5,
        amplitude_range * 1.5,
        num_additives
    )
    
    frequencies = np.random.uniform(min_freq, max_freq, num_additives)
    phases = np.random.uniform(0, 2 * np.pi, num_additives)

    components = np.zeros((num_additives, length))
    for i in range(num_additives):
        A = amplitudes[i]
        freq = frequencies[i]
        phase = phases[i]
        
        components[i] = A * np.sin(2 * np.pi * freq * t + phase)

    series = components.sum(axis=0)

    current_min = np.min(series)
    current_max = np.max(series)
    scale_factor = (max_value - min_value) / (current_max - current_min)
    center = (current_max + current_min) / 2
    target_center = (max_value + min_value) / 2

    series_final = (series - center) * scale_factor + target_center
    components_final = scale_factor * components + (target_center - scale_factor * center) / len(components)
    
    return {
        'series': series_final,
        'components': components_final,
        'amplitudes': amplitudes,
        'frequencies': frequencies,
        'phases': phases,
        'params': {
            'length': length,
            'min_value': min_value,
            'max_value': max_value,
            'num_additives': num_additives,
            'min_freq': min_freq,
            'max_freq': max_freq,
        }
    }


def create_sde_process(
    length: int = 2000,
    a_min: float = -0.1,
    a_max: float = 0.1,
    a_min_freq: float = 3.0,
    a_max_freq: float = 7.0,
    n_components: int = 5,
    b_long_term: float = 0.2,
    b_alpha: float = 0.1,
    b_beta: float = 0.85,
    dt: float = 1.0,
    leverage: float = -0.2
) -> Dict:
    """
    Create SDE process: dX(t) = a(t)dt + b(t)dW(t)
    
    a(t) is drift coefficient (Fourier series)
    b(t) follows GARCH-like dynamics with optional leverage effect
    
    Args:
        length: Number of time steps
        a_min: Minimum drift value
        a_max: Maximum drift value
        a_min_freq: Minimum frequency for a(t)
        a_max_freq: Maximum frequency for a(t)
        n_components: Number of Fourier terms for a(t)
        b_long_term: Long-term volatility level
        b_alpha: GARCH weight on recent squared returns
        b_beta: GARCH weight on past volatility
        dt: Time step size
        leverage: Leverage effect parameter (negative for typical effect)
    
    Returns:
        Dictionary containing process X(t), coefficients, and parameters
    """
    a_result = create_fourier_series(
        length=length,
        min_value=a_min,
        max_value=a_max,
        num_additives=n_components,
        min_freq=a_min_freq,
        max_freq=a_max_freq
    )
    a_t = a_result['series']

    a_t = a_t - np.mean(a_t)

    b_t = np.zeros(length)
    b_t[0] = b_long_term

    dW = np.random.normal(0, np.sqrt(dt), length)

    X = np.zeros(length)
    increments = np.zeros(length)

    for i in range(1, length):
        returns_sq = increments[i-1] ** 2

        leverage_effect = leverage * min(increments[i-1], 0)

        b_sq = (1 - b_alpha - b_beta) * (b_long_term ** 2) + \
               b_alpha * returns_sq + \
               b_beta * (b_t[i-1] ** 2) + \
               leverage_effect

        b_t[i] = np.sqrt(max(b_sq, 0.001))

        increments[i] = a_t[i] * dt + b_t[i] * dW[i]
        X[i] = X[i-1] + increments[i]
    
    return {
        'X': X,
        'a_t': a_t,
        'b_t': b_t,
        'increments': increments,
        'dW': dW,
        'params': {
            'length': length,
            'a_range': (a_min, a_max),
            'a_freq_range': (a_min_freq, a_max_freq),
            'n_components': n_components,
            'b_long_term': b_long_term,
            'b_alpha': b_alpha,
            'b_beta': b_beta,
            'dt': dt,
            'leverage': leverage
        }
    }
