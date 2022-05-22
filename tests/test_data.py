from re import I
from basicpy import data
import numpy as np
import pytest


@pytest.mark.parametrize("data_name", data.RESCALED_TEST_DATA_PROPS.keys())
def test_fetch_rescaled(data_name):
    """Test fetching the data for all datasets."""
    images = data.fetch(data_name)
    images = np.array(list(images))

    assert images.ndim == 3
