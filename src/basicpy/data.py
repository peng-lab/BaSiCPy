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
        "hash": "md5:b9718febdb2aa7a84f90c7d149cf2caf",
    },
    "timelapse_brightfield": {
        "hash": "md5:87a99a3c3dda7e170e272e9cd6f1ccd6",
    },
    "timelapse_nanog": {
        "hash": "md5:e6c2f7776575e6ad4ff65950f8371c05",
    },
    "timelapse_pu1": {
        "hash": "md5:02c4c74f9375369f2bf3c6f3bbca9109",
    },
    "wsi_brain": {
        "hash": "md5:52bcb25df3187b4947cbf89798933d0c",
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
    # FIXME change the URL when the beta version is about to release
    # see https://www.fatiando.org/pooch/latest/sample-data.html
    base_url="https://github.com/peng-lab/BaSiCPy/raw/dev/data/",
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
