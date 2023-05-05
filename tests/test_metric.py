import numpy as np

from basicpy.metrics import entropy


def test_entropy():
    rand_vals = np.random.rand(1000000) * 0.5
    entropy_val = entropy(rand_vals, 0, 0.5, bins=100)
    assert np.isclose(entropy_val, np.log(0.5), atol=0.01)
    rand_vals = np.random.normal(scale=1.5, size=1000000)
    entropy_val = entropy(rand_vals, -10, 10, bins=1000)
    assert np.isclose(entropy_val, 0.5 * (np.log(2 * np.pi * 1.5**2) + 1), atol=0.01)
