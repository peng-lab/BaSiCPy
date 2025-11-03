import numpy as np

from basicpy.metrics import entropy, fourier_L0_norm
import torch


def test_entropy():
    rand_vals = np.random.rand(1000000) * 0.5
    rand_vals = torch.from_numpy(rand_vals.astype(np.float32))
    _ = entropy(rand_vals, 0, 0.5, bins=100)
    rand_vals = np.random.normal(scale=1.5, size=1000000)
    rand_vals = torch.from_numpy(rand_vals.astype(np.float32))
    _ = entropy(rand_vals, -10, 10, bins=1000)


def test_fourier_L0_norm():
    for shape in [(128, 128), (100, 200)]:
        img = np.random.rand(*shape)
        img = torch.from_numpy(img.astype(np.float32))
        _ = fourier_L0_norm(img)
