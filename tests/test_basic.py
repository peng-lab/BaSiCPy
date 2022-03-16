from basicpy import BaSiC
import numpy as np
import pytest
from skimage.transform import resize

# allowed max error for the synthetic test data prediction
SYNTHESIZED_TEST_DATA_MAX_ERROR = 0.2


@pytest.fixture
def synthesized_test_data():

    np.random.seed(42)  # answer to the meaning of life, should work here too

    n_images = 8
    basic = BaSiC(get_darkfield=False)

    """Generate a parabolic gradient to simulate uneven illumination"""
    # Create a gradient
    size = basic.working_size
    grid = np.meshgrid(*(2 * (np.linspace(-size // 2 + 1, size // 2, size),)))

    # Create the parabolic gradient (flatfield) with and offset (darkfield)
    gradient = sum(d ** 2 for d in grid)
    gradient = 0.01 * (np.max(gradient) - gradient) + 10
    gradient_int = gradient.astype(np.uint8)

    # Ground truth, for correctness checking
    truth = gradient / gradient.mean()

    # Create an image stack and add poisson noise
    images = np.random.poisson(lam=gradient_int.flatten(), size=(n_images, size ** 2))
    images = images.transpose().reshape((size, size, n_images))

    return gradient, images, truth


# Ensure BaSiC initialization passes pydantic type checking
def test_basic_verify_init():

    basic = BaSiC()

    assert all([d == 128 for d in basic.darkfield.shape])
    assert all([d == 128 for d in basic.flatfield.shape])

    return


# Test BaSiC fitting function
def test_basic_fit(capsys, synthesized_test_data):

    basic = BaSiC(get_darkfield=False)
    gradient, images, truth = synthesized_test_data

    """Fit with BaSiC"""
    basic.fit(images)

    assert np.max(basic.flatfield / truth) < 1 + SYNTHESIZED_TEST_DATA_MAX_ERROR
    assert np.min(basic.flatfield / truth) > 1 - SYNTHESIZED_TEST_DATA_MAX_ERROR


# Test BaSiC transform function
def test_basic_transform(capsys, synthesized_test_data):

    basic = BaSiC(get_darkfield=False)
    gradient, images, truth = synthesized_test_data

    """Apply the shading model to the images"""
    # flatfield only
    basic.flatfield = gradient
    basic._flatfield = gradient
    corrected = basic.transform(images)
    corrected_error = corrected.mean()
    assert corrected_error < 0.5

    # with darkfield correction
    basic.darkfield = np.full(basic.flatfield.shape, 8)
    basic._darkfield = np.full(basic.flatfield.shape, 8)
    corrected = basic.transform(images)
    assert corrected.mean() < corrected_error

    """Test shortcut"""
    corrected = basic(images)
    assert corrected.mean() < corrected_error


def test_basic_transform_resize(capsys, synthesized_test_data):

    basic = BaSiC(get_darkfield=False)
    gradient, images, truth = synthesized_test_data

    images = resize(images, tuple(d * 2 for d in images.shape[:2]))
    truth = resize(truth, tuple(d * 2 for d in truth.shape[:2]))

    """Apply the shading model to the images"""
    # flatfield only
    basic.flatfield = gradient
    corrected = basic.transform(images)
    corrected_error = corrected.mean()
    assert corrected_error < 0.5

    # with darkfield correction
    basic.darkfield = np.full(basic.flatfield.shape, 8)
    corrected = basic.transform(images)
    assert corrected.mean() == corrected_error
