from pathlib import Path

import numpy as np
import pytest
from dask import array as da
from pydantic import ValidationError
from skimage.filters import threshold_otsu
from skimage.transform import resize

from basicpy import BaSiC, datasets, metrics

# allowed max error for the synthetic test data prediction
SYNTHETIC_TEST_DATA_MAX_ERROR = 0.35
EXPERIMENTAL_TEST_DATA_COUNT = 10


@pytest.fixture(params=[2, 3])  # param is dimension
def synthesized_test_data(request):

    np.random.seed(42)  # answer to the meaning of life, should work here too
    dim = request.param

    n_images = 8
    basic = BaSiC(get_darkfield=False)

    """Generate a parabolic gradient to simulate uneven illumination"""
    # Create a gradient
    if dim == 2:
        sizes = (basic.working_size, basic.working_size)
    else:
        sizes = tuple(([3] * (dim - 2)) + [basic.working_size, basic.working_size])

    grid = np.array(
        np.meshgrid(
            *[np.linspace(-size // 2 + 1, size // 2, size) for size in sizes],
            indexing="ij"
        )
    )

    # Create the parabolic gradient (flatfield) with and offset (darkfield)
    gradient = np.sum(grid**2, axis=0)
    gradient = 0.01 * (np.max(gradient) - gradient) + 10
    gradient_int = gradient.astype(np.uint8)

    # Ground truth, for correctness checking
    truth = gradient / gradient.mean()

    # Create an image stack and add poisson noise
    images = np.random.poisson(lam=gradient_int, size=[n_images] + list(sizes))

    return gradient, images, truth


# Ensure BaSiC initialization passes pydantic type checking
def test_basic_verify_init():

    basic = BaSiC()

    assert all([d == 128 for d in basic.darkfield.shape])
    assert all([d == 128 for d in basic.flatfield.shape])

    return


@pytest.mark.parametrize("resize_mode", ["jax", "skimage", "skimage_dask"])
def test_basic_resize(synthesized_test_data, resize_mode):
    _, images, _ = synthesized_test_data
    target_size = (*images.shape[:-2], 123, 456)
    resized = np.array(
        [resize(im, target_size[1:], preserve_range=True) for im in images]
    )
    basic = BaSiC(resize_mode=resize_mode)
    if resize_mode != "jax":
        images = da.from_array(images)
    resized2 = basic._resize(images, target_size)
    assert np.allclose(resized, resized2, rtol=0.1, atol=10)


# Test BaSiC fitting function (with synthetic data)
def test_basic_fit_synthetic(synthesized_test_data):

    basic = BaSiC(get_darkfield=False, smoothness_flatfield=1)

    gradient, images, truth = synthesized_test_data

    """Fit with BaSiC"""
    basic.fit(images)

    flatfield = basic.flatfield.copy()
    assert np.max(np.abs(basic.flatfield - truth)) < SYNTHETIC_TEST_DATA_MAX_ERROR
    assert np.array_equal(basic.flatfield.shape, images.shape[1:])

    basic.fit(images * 100)
    assert np.allclose(flatfield, basic.flatfield, rtol=1e-3, atol=1e-3)

    """
    from matplotlib import pyplot as plt
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


# Test BaSiC fitting function (with experimental data)
def test_basic_fit_experimental(datadir):

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
        images = datasets.fetch(image_name, original=False)
        basic.fit(images)
        assert np.all(np.isclose(basic.flatfield, d["flatfield"], atol=0.05, rtol=0.05))
        tol = 0.2
        assert np.allclose(
            basic.darkfield,
            d["darkfield"],
            atol=np.max(np.abs(d["darkfield"])) * tol,
            rtol=tol,
        )
        assert np.allclose(basic.baseline, d["baseline"], atol=tol, rtol=tol)


@pytest.mark.parametrize("early_stop", [False, True])
def test_basic_autotune(early_stop):
    np.random.seed(42)  # answer to the meaning of life, should work here too
    images = datasets.wsi_brain()

    basic = BaSiC(get_darkfield=True)

    vmin, vmax = np.percentile(images, [1, 99])

    transformed = basic.fit_transform(images, timelapse=False)
    cost1 = metrics.autotune_cost(
        transformed, basic.flatfield, entropy_vmin=vmin, entropy_vmax=vmax
    )

    basic.autotune(
        images,
        search_space={
            "smoothness_flatfield": list(np.logspace(-3, 1, 10)),
            "smoothness_darkfield": [0] + list(np.logspace(-3, 1, 10)),
            "sparse_cost_darkfield": [0] + list(np.logspace(-3, 1, 10)),
        },
        init_params={
            "smoothness_flatfield": 0.1,
            "smoothness_darkfield": 1e-3,
            "sparse_cost_darkfield": 1e-3,
        },
        n_iter=10,
        random_state=2023,
        early_stop=early_stop,
    )

    transformed = basic.fit_transform(images, timelapse=False)
    cost2 = metrics.autotune_cost(
        transformed, basic.flatfield, entropy_vmin=vmin, entropy_vmax=vmax
    )

    assert cost2 < cost1


@pytest.mark.parametrize("autosegment", [True, lambda Im: Im < threshold_otsu(Im)])
def test_basic_autosegment(autosegment):
    np.random.seed(42)  # answer to the meaning of life, should work here too
    images = datasets.wsi_brain()
    basic = BaSiC(
        get_darkfield=True,
        autosegment=autosegment,
        working_size=None,
    )
    basic.fit(images)

    if autosegment is True:
        Ws = images > threshold_otsu(images)
    else:
        Ws = autosegment(images)

    basic2 = BaSiC(
        get_darkfield=True,
        autosegment=False,
        working_size=None,
    )
    basic2.fit(images, fitting_weight=Ws)

    assert np.allclose(basic.flatfield, basic2.flatfield, rtol=0.01, atol=0.01)
    assert np.allclose(basic.darkfield, basic2.darkfield, rtol=0.01, atol=0.01)
    assert np.allclose(basic.baseline, basic2.baseline, rtol=0.01, atol=0.01)


# Test BaSiC transform function
@pytest.mark.parametrize("use_dask", [False, True])
def test_basic_transform(synthesized_test_data, use_dask):

    basic = BaSiC(get_darkfield=False)
    gradient, images, truth = synthesized_test_data

    """Apply the shading model to the images"""
    # flatfield only
    basic.flatfield = gradient
    basic.baseline = np.ones((8,))
    if use_dask:
        corrected = basic.transform(da.array(images)).compute()
    else:
        corrected = basic.transform(images)
    corrected_error = np.abs(corrected.mean() - 1.0)
    assert corrected_error < 0.5

    # with darkfield correction
    basic.darkfield = np.full(basic.flatfield.shape, 8)
    if use_dask:
        corrected = basic.transform(da.array(images + 8)).compute()
    else:
        corrected = basic.transform(images + 8)
    corrected_error = np.abs(corrected.mean() - 1.0)
    assert corrected_error < 0.5

    """Test shortcut"""
    corrected = basic(images)


@pytest.fixture(params=[2, 3])  # param is dimension
def basic_object(request):
    dim = request.param
    basic = BaSiC()
    # set profiles
    basic.flatfield = np.full((128,) * dim, 1, dtype=np.float64)
    basic.darkfield = np.full((128,) * dim, 2, dtype=np.float64)
    return basic


def test_basic_save_model(tmp_path: Path, basic_object):
    model_dir = tmp_path / "test_model"

    # save the model
    basic_object.save_model(model_dir)

    # check that the files exists
    assert (model_dir / "settings.json").exists()
    assert (model_dir / "profiles.npy").exists()

    # load files and check for expected content
    saved_profiles = np.load(model_dir / "profiles.npy")
    profiles = np.array((basic_object.flatfield, basic_object.darkfield))
    assert np.array_equal(saved_profiles, profiles)

    # TODO check settings contents

    # remove files but not the folder to check for overwriting
    (model_dir / "settings.json").unlink()
    (model_dir / "profiles.npy").unlink()
    # assert not (model_dir / "settings.json").exists()
    # assert not (model_dir / "profiles.npy").exists()

    # an error raises when the model folder exists
    with pytest.raises(FileExistsError):
        basic_object.save_model(model_dir)

    # overwrites if specified
    basic_object.save_model(model_dir, overwrite=True)
    assert (model_dir / "settings.json").exists()
    assert (model_dir / "profiles.npy").exists()


def test_basic_save_load_model(tmp_path: Path, basic_object):
    model_dir = tmp_path / "test_model"
    flatfield = basic_object.flatfield.copy()
    darkfield = basic_object.darkfield.copy()

    # save the model
    basic_object.save_model(model_dir)
    basic2 = BaSiC.load_model(model_dir)

    assert np.allclose(basic2.flatfield, flatfield)
    assert np.allclose(basic2.darkfield, darkfield)
    assert basic_object.dict() == basic2.dict()


@pytest.fixture
def profiles():
    # create and write mock profiles to file
    profiles = np.zeros((2, 128, 128), dtype=np.float64)
    # unique profiles to check that they are in proper place
    profiles[0] = 1
    profiles[1] = 2
    return profiles


@pytest.fixture
def model_path(tmp_path, profiles):
    settings_json = """\
    {"epsilon": 0.2, "get_darkfield": false,
    "smoothness_darkfield": 0.0, "smoothness_flatfield": 0.0, "max_iterations": 500,
    "max_reweight_iterations": 10, "optimization_tol": 1e-06, "reweighting_tol": 0.001,
    "working_size": 128}
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
        assert np.array_equal(basic.flatfield, profiles[0])
        assert np.array_equal(basic.darkfield, profiles[1])

        # check that settings are not default
        assert basic.epsilon != BaSiC.__fields__["epsilon"].default


def test_no_accepting_wrong_argments() -> None:
    with pytest.raises(ValidationError):
        _ = BaSiC(novalid=True)
