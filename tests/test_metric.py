import numpy as np
import torch

from basicpy.metrics import entropy, fourier_L0_norm
from basicpy.metrics_numpy import (
    autotune_cost_numpy,
    entropy as entropy_numpy,
    fourier_L0_norm as fourier_L0_norm_numpy,
)


def test_entropy():
    rand_vals = np.random.rand(1000000) * 0.5
    rand_vals = torch.from_numpy(rand_vals.astype(np.float32))
    _ = entropy(rand_vals, torch.tensor([0]), torch.tensor([0.5]), bins=100)
    rand_vals = np.random.normal(scale=1.5, size=1000000)
    rand_vals = torch.from_numpy(rand_vals.astype(np.float32))
    _ = entropy(rand_vals, torch.tensor([-10]), torch.tensor([10]), bins=1000)


def test_fourier_L0_norm():
    for shape in [(128, 128), (100, 200)]:
        img = np.random.rand(*shape)
        img = torch.from_numpy(img.astype(np.float32))
        _ = fourier_L0_norm(img)


def test_numpy_entropy_regression():
    values = np.array([0.125, 0.125, 0.625, 0.875], dtype=np.float64)
    result = entropy_numpy(values, vmin=0.0, vmax=1.0, bins=4)

    assert np.isclose(result, -0.5 * np.log(2.0), rtol=1e-12, atol=1e-12)


def test_numpy_fourier_l0_norm_constant_image():
    image = np.ones((16, 16), dtype=np.float64)
    result = fourier_L0_norm_numpy(image, fourier_radius=3)

    assert result == 0.0


def test_numpy_autotune_cost_regression():
    transformed_image = np.tile(
        np.array([0.125, 0.125, 0.625, 0.875], dtype=np.float64), (16, 4)
    )
    flatfield = np.ones((16, 16), dtype=np.float64)

    result = autotune_cost_numpy(
        transformed_image,
        flatfield,
        entropy_vmin=0.0,
        entropy_vmax=1.0,
        histogram_bins=4,
    )

    assert np.isclose(result, -0.5 * np.log(2.0), rtol=1e-12, atol=1e-12)
