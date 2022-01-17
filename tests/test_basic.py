from pybasic import BaSiC
import numpy as np


# DEBUG fails because Settings is now a pydantic dataclass
def test_basic_verify_init():

    basic = BaSiC()

    assert all([d == 128 for d in basic.darkfield.shape])
    assert all([d == 128 for d in basic.flatfield.shape])

    return


def test_basic():
    ...


def test_basic_fit(capsys):

    np.random.seed(42)  # answer to the meaning of life, should work here too

    n_images = 8
    basic = BaSiC(get_darkfield=False)

    """Generate a parabolic gradient to simulate uneven illumination"""
    # Create a gradient
    size = basic.working_size
    grid = np.meshgrid(*(2 * (np.linspace(-size // 2 + 1, size // 2, size),)))

    # Create the gradient (flatfield) with and offset (darkfield)
    gradient = sum(d ** 2 for d in grid) ** (1 / 2) + 8
    gradient_int = gradient.astype(np.uint8)

    # Ground truth, for correctness checking
    truth = gradient / gradient.mean()

    # Create an image stack and add poisson noise
    images = np.random.poisson(lam=gradient_int.flatten(), size=(n_images, size ** 2))
    images = images.transpose().reshape((size, size, n_images))

    """Apply the shading model to the images"""
    # flatfield only
    basic.flatfield = gradient
    corrected = basic.predict(images)
    corrected_error = corrected.mean()
    assert corrected_error < 0.5

    # with darkfield correction
    basic.darkfield = np.full(basic.flatfield.shape, 8)
    corrected = basic.predict(images)
    assert corrected.mean() < corrected_error

    """Test shortcut"""
    correct = basic(images)
    assert corrected.mean() < corrected_error
