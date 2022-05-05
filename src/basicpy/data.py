import pooch
from os import path
import glob
from skimage.io import imread

EXPERIMENTAL_TEST_DATA_PROPS = {
    "cell_culture": {
        "filename": "Cell_culture.zip",
    },
    "timelapse_brightfield": {
        "filename": "Timelapse_brightfield.zip",
    },
    "timelapse_nanog": {
        "filename": "Timelapse_nanog.zip.zip",
    },
    "timelapse_pu1": {
        "filename": "Timelapse_Pu1.zip.zip",
    },
    "wsi_brain": {
        "filename": "WSI_Brain.zip",
    },
}

POOCH = pooch.create(
    path=pooch.os_cache("basicpy"),
    # Use the Zenodo DOI
    base_url="doi:10.5281/zenodo.6334810/",
    registry={v["filename"]: v["hash"] for v in EXPERIMENTAL_TEST_DATA_PROPS.values()},
)


def fetch(data_name: str):
    """Fetch a sample dataset from Zenodo.
    Args:
        data_name: The name of the dataset. Must be one of ["cell_culture",
            "timelapse_brightfield", "timelapse_nanog", "timelapse_pu1",
            "wsi_brain"].

    Returns:
        Iterable[ndarray]: An iterable of uncorrected images.
        Iterable[ndarray]: An iterable of corrected images by the reference
            implementation (Tingying Peng et al., Nature Communication 8:14836 (2017).)

    Raises:
        ValueError: If the dataset name is not one of the allowed values.
    """
    if data_name not in EXPERIMENTAL_TEST_DATA_PROPS.keys():
        raise ValueError(f"{data_name} is not a valid test data name")
    file_name: str = EXPERIMENTAL_TEST_DATA_PROPS[data_name]["filename"]
    test_file_paths = POOCH.fetch(file_name, processor=pooch.Unzip())
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
    if "WSI_Brain" in file_name:
        uncorrected_paths = list(
            filter(lambda p: "BrainSection" in p, uncorrected_paths)
        )
        corrected_paths = list(filter(lambda p: "BrainSection" in p, corrected_paths))

    assert len(uncorrected_paths) > 0
    assert len(uncorrected_paths) == len(corrected_paths)
    uncorrected = (imread(f) for f in uncorrected_paths)
    corrected = (imread(f) for f in corrected_paths)

    return uncorrected, corrected


def cell_culture():
    """Returns the "Cell culture" dataset.

    Returns:
        Iterable[ndarray]: An iterable of uncorrected images.
        Iterable[ndarray]: An iterable of corrected images by the reference
            implementation (Tingying Peng et al., Nature Communication 8:14836 (2017).)
    """
    return fetch("cell_culture")


def timelapse_brightfield():
    """Returns the "Timelapse Brightfield" dataset.

    Returns:
        Iterable[ndarray]: An iterable of uncorrected images.
        Iterable[ndarray]: An iterable of corrected images by the reference
            implementation (Tingying Peng et al., Nature Communication 8:14836 (2017).)
    """
    return fetch("timelapse_brightfield")


def timelapse_nanog():
    """Returns the "Timelapse Nanog" dataset.

    Returns:
        Iterable[ndarray]: An iterable of uncorrected images.
        Iterable[ndarray]: An iterable of corrected images by the reference
            implementation (Tingying Peng et al., Nature Communication 8:14836 (2017).)
    """
    return fetch("timelapse_nanog")


def timelapse_pu1():
    """Returns the "Timelapse Pu.1" dataset.

    Returns:
        Iterable[ndarray]: An iterable of uncorrected images.
        Iterable[ndarray]: An iterable of corrected images by the reference
            implementation (Tingying Peng et al., Nature Communication 8:14836 (2017).)
    """

    return fetch("timelapse_pu1")


def wsi_brain():
    """Returns the "WSI Brain" dataset.

    Returns:
        Iterable[ndarray]: An iterable of uncorrected images.
        Iterable[ndarray]: An iterable of corrected images by the reference
            implementation (Tingying Peng et al., Nature Communication 8:14836 (2017).)
    """

    return fetch("wsi_brain")[0]
