"""Datasets used for testing."""
import glob
from os import path

import numpy as np
import pooch
from skimage.io import imread

ORIGINAL_TEST_DATA_PROPS = {
    "cell_culture": {
        "filename": "Cell_culture.zip",
        "hash": "md5:797bbc4c891e5fe59f4200e771b55c3a",
    },
    "timelapse_brightfield": {
        "filename": "Timelapse_brightfield.zip",
        "hash": "md5:460e5f78ac69856705704fedad9f9e59",
    },
    "timelapse_nanog": {
        "filename": "Timelapse_nanog.zip.zip",
        "hash": "md5:815d53cac35b671269b17bd627d7baa7",
    },
    "timelapse_pu1": {
        "filename": "Timelapse_Pu1.zip.zip",
        "hash": "md5:bee97561e87c51e90b46da9b439e8b7b",
    },
    "wsi_brain": {
        "filename": "WSI_Brain.zip",
        "hash": "md5:6e163786ddec2a690aa4bb47a64bcded",
    },
}

RESCALED_TEST_DATA_PROPS = {
    "cell_culture": {
        "hash": "md5:65b91d0e1e9161826da79f0ed8ea91d1",
    },
    "timelapse_brightfield": {
        "hash": "md5:8bedded105fff92a478bfde56d15f80f",
    },
    "timelapse_nanog": {
        "hash": "md5:252fd6b8a054ce659cc6ba33dc9302ff",
    },
    "timelapse_pu1": {
        "hash": "md5:598ee2343b6562534c09bea05235f657",
    },
    "wsi_brain": {
        "hash": "md5:6e58f605916ddb1b81724f1bfea73560",
    },
}
for k in RESCALED_TEST_DATA_PROPS.keys():
    RESCALED_TEST_DATA_PROPS[k]["filename"] = k + ".npz"

ORIGINAL_POOCH = pooch.create(
    path=pooch.os_cache("basicpy"),
    # Use the Zenodo DOI
    base_url="doi:10.5281/zenodo.6334810/",
    registry={v["filename"]: v["hash"] for v in ORIGINAL_TEST_DATA_PROPS.values()},
)

RESCALED_POOCH = pooch.create(
    path=pooch.os_cache("basicpy"),
    # Use the Zenodo DOI
    base_url="doi:10.5281/zenodo.6974039/",
    # base_url="https://github.com/yfukai/BaSiCpy/raw/data_in_package/data/",
    registry={v["filename"]: v["hash"] for v in RESCALED_TEST_DATA_PROPS.values()},
)


def fetch(data_name: str, original: bool = False):
    """Fetch a sample dataset from Zenodo or GitHub.
    Args:
        data_name: The name of the dataset. Must be one of ["cell_culture",
            "timelapse_brightfield", "timelapse_nanog", "timelapse_pu1",
            "wsi_brain"].
        original: If True, return the original dataset. If False, return the rescaled.

    Returns:
        Iterable[ndarray]: An iterable of uncorrected images.

    Raises:
        ValueError: If the dataset name is not one of the allowed values.
    """
    if original:
        return _fetch_original(data_name)
    else:
        return _fetch_rescaled(data_name)


def _fetch_original(data_name: str):
    if data_name not in ORIGINAL_TEST_DATA_PROPS.keys():
        raise ValueError(f"{data_name} is not a valid test data name")
    file_name = ORIGINAL_TEST_DATA_PROPS[data_name]["filename"]
    test_file_paths = ORIGINAL_POOCH.fetch(file_name, processor=pooch.Unzip())
    assert all(path.exists(f) for f in test_file_paths)
    basedir = path.commonpath(test_file_paths)
    uncorrected_paths = sorted(
        glob.glob(path.join(basedir, "Uncorrected*", "**", "*.tif"), recursive=True)
    )
    if len(uncorrected_paths) == 0:
        uncorrected_paths = sorted(
            glob.glob(path.join(basedir, "Uncorrected*", "**", "*.png"), recursive=True)
        )
    if "WSI_Brain" in file_name:
        uncorrected_paths = list(
            filter(lambda p: "BrainSection" in p, uncorrected_paths)
        )

    assert len(uncorrected_paths) > 0
    uncorrected = (imread(f) for f in uncorrected_paths)
    return uncorrected


def _fetch_rescaled(data_name: str):
    if data_name not in RESCALED_TEST_DATA_PROPS.keys():
        raise ValueError(f"{data_name} is not a valid test data name")
    file_name = RESCALED_TEST_DATA_PROPS[data_name]["filename"]
    test_file_path = RESCALED_POOCH.fetch(file_name)
    return np.load(test_file_path)["images"]


def cell_culture():
    """Returns the rescaled "Cell culture" dataset.

    Returns:
        Iterable[ndarray]: An iterable of sample images.
    """
    return fetch("cell_culture")


def timelapse_brightfield():
    """Returns the rescaled "Timelapse Brightfield" dataset.

    Returns:
        Iterable[ndarray]: An iterable of sample images.
    """
    return fetch("timelapse_brightfield")


def timelapse_nanog():
    """Returns the rescaled "Timelapse Nanog" dataset.

    Returns:
        Iterable[ndarray]: An iterable of sample images.
    """
    return fetch("timelapse_nanog")


def timelapse_pu1():
    """Returns the rescaled "Timelapse Pu.1" dataset.

    Returns:
        Iterable[ndarray]: An iterable of sample images.
    """

    return fetch("timelapse_pu1")


def wsi_brain():
    """Returns the rescaled "WSI Brain" dataset.

    Returns:
        Iterable[ndarray]: An iterable of sample images.
    """

    return fetch("wsi_brain")
