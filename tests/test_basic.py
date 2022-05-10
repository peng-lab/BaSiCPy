from basicpy import BaSiC
import numpy as np
import pytest
from skimage.transform import resize
from pathlib import Path


@pytest.fixture
def test_data():

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

    return gradient, images, truth


# Ensure BaSiC initialization passes pydantic type checking
def test_basic_verify_init():

    basic = BaSiC()

    assert all([d == 128 for d in basic.darkfield.shape])
    assert all([d == 128 for d in basic.flatfield.shape])

    return


# Test BaSiC fitting function
def test_basic_fit(capsys, test_data):

    basic = BaSiC(get_darkfield=False)

    gradient, images, truth = test_data

    """Fit with BaSiC"""
    basic.fit(images)

    # TODO: Implement correctness checks

    # for human error checking
    # with capsys.disabled():
    #     print()
    #     print(truth[60:70, 60:70])
    #     print(basic.flatfield[60:70, 60:70])


# Test BaSiC transform function
def test_basic_transform(capsys, test_data):

    basic = BaSiC(get_darkfield=False)
    gradient, images, truth = test_data

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


def test_basic_transform_resize(capsys, test_data):

    basic = BaSiC(get_darkfield=False)
    gradient, images, truth = test_data

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


def test_basic_save_model(tmp_path: Path):
    model_dir = tmp_path / "test_model"

    basic = BaSiC()

    # set profiles
    basic.flatfield = np.full((128, 128), 1, dtype=np.float64)
    basic.darkfield = np.full((128, 128), 2, dtype=np.float64)

    # save the model
    basic.save_model(model_dir)

    # check that the files exists
    assert (model_dir / "settings.json").exists()
    assert (model_dir / "profiles.npy").exists()

    # load files and check for expected content
    saved_profiles = np.load(model_dir / "profiles.npy")
    profiles = np.dstack((basic.flatfield, basic.darkfield))
    assert np.array_equal(saved_profiles, profiles)

    # TODO check settings contents

    # remove files but not the folder to check for overwriting
    (model_dir / "settings.json").unlink()
    (model_dir / "profiles.npy").unlink()
    # assert not (model_dir / "settings.json").exists()
    # assert not (model_dir / "profiles.npy").exists()

    # an error raises when the model folder exists
    with pytest.raises(FileExistsError):
        basic.save_model(model_dir)

    # overwrites if specified
    basic.save_model(model_dir, overwrite=True)
    assert (model_dir / "settings.json").exists()
    assert (model_dir / "profiles.npy").exists()


@pytest.fixture
def profiles():
    # create and write mock profiles to file
    profiles = np.zeros((128, 128, 2), dtype=np.float64)
    # unique profiles to check that they are in proper place
    profiles[..., 0] = 1
    profiles[..., 1] = 2
    return profiles


@pytest.fixture
def model_path(tmp_path, profiles):
    settings_json = """\
    {"epsilon": 0.2, "estimation_mode": "l0", "get_darkfield": false,
    "lambda_darkfield": 0.0, "lambda_flatfield": 0.0, "max_iterations": 500,
    "max_reweight_iterations": 10, "optimization_tol": 1e-06, "reweighting_tol": 0.001,
    "varying_coeff": true, "working_size": 128}
    """
    with open(tmp_path / "settings.json", "w") as fp:
        fp.write(settings_json)
    np.save(tmp_path / "profiles.npy", profiles)
    return str(tmp_path)


@pytest.mark.parametrize("raises_error", [(True), (False)], ids=["no_model", "model"])
def test_basic_load_model(model_path: str, raises_error: bool, profiles: np.ndarray):
    if raises_error:
        with pytest.raises(FileNotFoundError):
            basic = BaSiC.load_model("/not/a/real/path")
    else:
        # generate an instance from the serialized model
        basic = BaSiC.load_model(model_path)

        # check that the object was created
        assert isinstance(basic, BaSiC)

        # check that the profiles are in the right places
        assert np.array_equal(basic.flatfield, profiles[..., 0])
        assert np.array_equal(basic.darkfield, profiles[..., 1])

        # check that settings are not default
        assert basic.epsilon != BaSiC.__fields__["epsilon"].default
