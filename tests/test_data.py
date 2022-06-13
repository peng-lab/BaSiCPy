from basicpy import data
import numpy as np
import pytest


@pytest.mark.parametrize("data_name", data.RESCALED_TEST_DATA_PROPS.keys())
def test_fetch_rescaled(data_name):
    """Test fetching the data for all datasets."""
    images = data.fetch(data_name)
    images = np.array(list(images))

    assert images.ndim == 3
    assert images.shape[0] > 1
    assert np.any(np.array(images.shape[1:]) == 128)

    data.cell_culture()
    data.timelapse_brightfield()
    data.timelapse_nanog()
    data.timelapse_pu1()
    data.wsi_brain()


# This test takes long time to download the original data and skipped by default.
# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
@pytest.mark.slow
@pytest.mark.parametrize("data_name", data.ORIGINAL_TEST_DATA_PROPS.keys())
def test_fetch_original(data_name):
    images = data.fetch(data_name, original=True)
    images = np.array(list(images))

    assert images.ndim == 3
    assert images.shape[0] > 1
