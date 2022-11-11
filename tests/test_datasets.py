import numpy as np
import pytest

from basicpy import datasets


@pytest.mark.parametrize("data_name", datasets.RESCALED_TEST_DATA_PROPS.keys())
def test_fetch_rescaled(data_name):
    """Test fetching the data for all datasets."""
    images = datasets.fetch(data_name)
    images = np.array(list(images))

    assert images.ndim == 3
    assert images.shape[0] > 1
    assert np.any(np.array(images.shape[1:]) == 128)

    datasets.cell_culture()
    datasets.timelapse_brightfield()
    datasets.timelapse_nanog()
    datasets.timelapse_pu1()
    datasets.wsi_brain()


# This test takes long time to download the original data and skipped by default.
# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
@pytest.mark.slow
@pytest.mark.parametrize("data_name", datasets.ORIGINAL_TEST_DATA_PROPS.keys())
def test_fetch_original(data_name):
    images = datasets.fetch(data_name, original=True)
    images = np.array(list(images))

    assert images.ndim == 3
    assert images.shape[0] > 1
