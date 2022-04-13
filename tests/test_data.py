from re import I
from basicpy import data
import numpy as np


def test_fetch():
    """Test fetching the data for all datasets."""
    for k in data.EXPERIMENTAL_TEST_DATA_PROPS.keys():
        uncorrected, corrected = data.fetch(k)
        uncorrected = np.array(list(uncorrected))
        corrected = np.array(list(corrected))

        assert uncorrected.ndim == 3
        assert corrected.ndim == 3
        assert np.array_equal(uncorrected.shape, corrected.shape)
