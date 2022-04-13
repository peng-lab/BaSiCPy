from re import I
from basicpy import data
import numpy as np
import pytest


@pytest.mark.parametrize("data_name", data.EXPERIMENTAL_TEST_DATA_PROPS.keys())
def test_fetch(data_name):
    """Test fetching the data for all datasets."""
    uncorrected, corrected = data.fetch(data_name)
    uncorrected = np.array(list(uncorrected))
    corrected = np.array(list(corrected))

    assert uncorrected.ndim == 3
    assert corrected.ndim == 3
    assert np.array_equal(uncorrected.shape, corrected.shape)
