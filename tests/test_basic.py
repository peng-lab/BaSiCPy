from basicpy import BaSiC
from os import path
import glob
import numpy as np
import pytest
from skimage.io import imread
from skimage.transform import resize
from pathlib import Path

from basicpy.basicpy import FittingMode

# allowed max error for the synthetic test data prediction
SYNTHETIC_TEST_DATA_MAX_ERROR = 0.35
EXPERIMENTAL_TEST_DATA_COUNT = 10
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def synthesized_test_data_3d():

    np.random.seed(42)  # answer to the meaning of life, should work here too

    n_images = 8
    basic = BaSiC(get_darkfield=False)

    """Generate a parabolic gradient to simulate uneven illumination"""
    # Create a gradient
    size = basic.working_size
    grid = np.meshgrid(*(3 * (np.linspace(-size // 2 + 1, size // 2, size),)))

    # Create the parabolic gradient (flatfield) with and offset (darkfield)
    gradient = sum(d**2 for d in grid)
    gradient = 0.01 * (np.max(gradient) - gradient) + 10
    gradient_int = gradient.astype(np.uint8)

    # Ground truth, for correctness checking
    truth = gradient / gradient.mean()

    # Create an image stack and add poisson noise
    images = np.random.poisson(lam=gradient_int, size=(n_images, size, size, size))

    return gradient, images, truth


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
    gradient = sum(d**2 for d in grid)
    gradient = 0.01 * (np.max(gradient) - gradient) + 10
    gradient_int = gradient.astype(np.uint8)

    # Ground truth, for correctness checking
    truth = gradient / gradient.mean()

    # Create an image stack and add poisson noise
    images = np.random.poisson(lam=gradient_int, size=(n_images, size, size))

    return gradient, images, truth


# Ensure BaSiC initialization passes pydantic type checking
def test_basic_verify_init():

    basic = BaSiC()

    assert all([d == 128 for d in basic.darkfield.shape])
    assert all([d == 128 for d in basic.flatfield.shape])

    return


# Test BaSiC fitting function (with synthetic data)
def test_basic_fit_synthetic(synthesized_test_data):

    basic = BaSiC(get_darkfield=False, lambda_flatfield_coef=10)

    gradient, images, truth = synthesized_test_data

    """Fit with BaSiC"""
    basic.fit(images)

    assert np.max(np.abs(basic.flatfield - truth)) < SYNTHETIC_TEST_DATA_MAX_ERROR
    """
    code for debug plotting :
    plt.figure(figsize=(15,5)) ;
    plt.subplot(131) ; plt.imshow(truth) ;plt.title("truth") ;
    plt.colorbar() ;
    plt.subplot(132) ; plt.imshow(basic.flatfield) ;plt.title("estimated") ;
    plt.colorbar() ;
    plt.subplot(133) ; plt.imshow(basic.flatfield / truth) ;plt.title("ratio") ;
    plt.colorbar() ;
    plt.show()
    """


# Test BaSiC fitting function (with synthetic data)
def test_basic_fit_synthetic_3d(synthesized_test_data_3d):

    basic = BaSiC(get_darkfield=False, lambda_flatfield_coef=10)

    gradient, images, truth = synthesized_test_data_3d

    """Fit with BaSiC"""
    basic.fit(images)

    assert np.max(np.abs(basic.flatfield - truth)) < SYNTHETIC_TEST_DATA_MAX_ERROR


# Test BaSiC fitting function (with experimental data)
@pytest.mark.datafiles(
    DATA_DIR / "cell_culture.npz",
    DATA_DIR / "timelapse_brightfield.npz",
    DATA_DIR / "timelapse_nanog.npz",
    DATA_DIR / "timelapse_pu1.npz",
    DATA_DIR / "wsi_brain.npz",
)
def test_basic_fit_experimental(datadir, datafiles):

    # not sure if it is a good practice
    fit_results = list(datadir.glob("*.npz"))
    assert len(fit_results) > 0
    np.random.seed(42)  # answer to the meaning of life, should work here too
    # TODO parametrize?
    fit_results = np.concatenate(
        [np.load(f, allow_pickle=True)["results"] for f in fit_results]
    )
    np.random.shuffle(fit_results)

    for d in fit_results[:EXPERIMENTAL_TEST_DATA_COUNT]:
        image_name = d["image_name"]
        params = d["params"]
        basic = BaSiC(**params)
        images_path = [
            f for f in datafiles.listdir() if f.basename == image_name + ".npz"
        ]
        assert len(images_path) == 1
        images = np.load(str(images_path[0]))["images"]
        basic.fit(images)
        assert np.all(np.isclose(basic.flatfield, d["flatfield"], atol=0.05, rtol=0.05))
        if basic.fitting_mode == FittingMode.approximate:
            tol = 0.2
        else:
            tol = 0.1
        assert np.all(
            np.isclose(
                basic.darkfield,
                d["darkfield"],
                atol=np.max(np.abs(d["darkfield"])) * tol,
                rtol=tol,
            )
        )
        assert np.all(np.isclose(basic.baseline, d["baseline"], atol=tol, rtol=tol))


# Test BaSiC transform function
def test_basic_transform(synthesized_test_data):

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
    assert corrected.mean() <= corrected_error

    """Test shortcut"""
    corrected = basic(images)


def test_basic_transform_resize(synthesized_test_data):

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
    assert corrected.mean() <= corrected_error


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
