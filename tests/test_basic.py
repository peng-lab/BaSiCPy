from basicpy import BaSiC
from os import path
import glob
import numpy as np
import pytest
from skimage.io import imread
from skimage.transform import resize
import pooch

# allowed max error for the synthetic test data prediction
SYNTHETIC_TEST_DATA_MAX_ERROR = 0.2

EXPERIMENTAL_TEST_DATA_NAMES = {
    "Cell_culture.zip": "md5:797bbc4c891e5fe59f4200e771b55c3a",
    "Timelapse_brightfield.zip": "md5:460e5f78ac69856705704fedad9f9e59",
    "Timelapse_nanog.zip.zip": "md5:815d53cac35b671269b17bd627d7baa7",
    "Timelapse_Pu1.zip.zip": "md5:bee97561e87c51e90b46da9b439e8b7b",
    "WSI_Brain.zip": "md5:6e163786ddec2a690aa4bb47a64bcded",
}

POOCH = pooch.create(
    path=pooch.os_cache("testdata"),
    # Use the Zenodo DOI
    base_url="doi:10.5281/zenodo.6334810/",
    registry=EXPERIMENTAL_TEST_DATA_NAMES,
)


@pytest.fixture
def synthetic_test_data():

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


@pytest.fixture(params=EXPERIMENTAL_TEST_DATA_NAMES.keys())
def experimental_test_data(request):
    test_file_paths = POOCH.fetch(request.param, processor=pooch.Unzip())
    assert all(path.exists(f) for f in test_file_paths)
    basedir = path.commonpath(test_file_paths)
    uncorrected_paths = sorted(
        glob.glob(path.join(basedir, "Uncorrected*", "**", "*.tif"), recursive=True)
    )
    if len(uncorrected_paths) == 0:
        uncorrected_paths = sorted(
            glob.glob(path.join(basedir, "Uncorrected*", "**", "*.png"), recursive=True)
        )
    corrected_paths = sorted(
        glob.glob(path.join(basedir, "Corrected*", "**", "*.tif"), recursive=True)
    )
    if "WSI_Brain" in request.param:
        uncorrected_paths = list(
            filter(lambda p: "BrainSection" in p, uncorrected_paths)
        )
        corrected_paths = list(filter(lambda p: "BrainSection" in p, corrected_paths))

    assert len(uncorrected_paths) > 0
    assert len(uncorrected_paths) == len(corrected_paths)
    uncorrected = (imread(f) for f in uncorrected_paths)
    corrected = (imread(f) for f in corrected_paths)

    return uncorrected, corrected


# Ensure BaSiC initialization passes pydantic type checking
def test_basic_verify_init():

    basic = BaSiC()

    assert all([d == 128 for d in basic.darkfield.shape])
    assert all([d == 128 for d in basic.flatfield.shape])

    return


# Test BaSiC fitting function (with synthetic data)
def test_basic_fit_synthetic(capsys, synthetic_test_data):

    basic = BaSiC(get_darkfield=False)
    gradient, images, truth = synthetic_test_data

    """Fit with BaSiC"""
    basic.fit(images)

    assert np.max(basic.flatfield / truth) < 1 + SYNTHETIC_TEST_DATA_MAX_ERROR
    assert np.min(basic.flatfield / truth) > 1 - SYNTHETIC_TEST_DATA_MAX_ERROR


def test_basic_fit_experimental(experimental_test_data):
    np.random.seed(42)  # answer to the meaning of life, should work here too
    basic = BaSiC(get_darkfield=False)
    uncorrected, corrected = experimental_test_data
    basic.fit(np.array(list(uncorrected)))


# Test BaSiC transform function
def test_basic_transform(capsys, synthetic_test_data):

    basic = BaSiC(get_darkfield=False)
    gradient, images, truth = synthetic_test_data

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


def test_basic_transform_resize(capsys, synthetic_test_data):

    basic = BaSiC(get_darkfield=False)
    gradient, images, truth = synthetic_test_data

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
