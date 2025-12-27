import numpy as np


def create_mixture(length, num_components):
    num_pi = 4
    a_k = np.zeros((length, num_components), dtype=np.float32)
    b_k = np.zeros((length, num_components), dtype=np.float32)
    p_k = np.zeros((length, num_components), dtype=np.float32)
    rnd = 4
    for comp in range(num_components):
        rng = np.linspace(comp, comp + 1, length, endpoint=False)
        if comp % 2:
            a_k[:, comp] = np.sin(rnd * rng * num_pi * np.pi + np.pi/rnd)
            b_k[:, comp] = ((0.1 + 1) + a_k[:, comp])
            p_k[:, comp] = a_k[:, comp]
        else:
            a_k[:, comp] = np.cos(comp * rng * num_pi * np.pi + np.pi/rnd)
            b_k[:, comp] = ((0.1 + 1) + a_k[:, comp])
            p_k[:, comp] = a_k[:, comp]
        rnd = np.float32(np.random.uniform(1, num_pi))
    a_k = (a_k - (1 + 1/2**0.5)/2) / 2
    b_k = (b_k - 1/2**0.5 - 1) * 10

    component = np.argmax(p_k, 1)
    a_k_res = a_k[np.arange(length), component]
    b_k_res = b_k[np.arange(length), component]
    deltas = np.random.normal(a_k_res, b_k_res, length).astype(np.float32)
    time_series = np.concatenate([np.zeros(1, dtype=np.float32), deltas.cumsum(dtype=np.float32)])
    return time_series, deltas, a_k_res, b_k_res
