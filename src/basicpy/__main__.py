"""CLI entry point for BaSiCPy."""
import argparse
import logging
from pathlib import Path
from typing import Optional

from basicpy import BaSiC
from basicpy.basicpy import Device

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)


def main(
    input_dir: Path,
    output_dir: Path,
    pattern: Optional[str],
    timelapse: Optional[bool],
    **settings,
) -> None:

    # Check to make the input and output directories exists
    # TODO: Make output directory if it doesn't exist? Commented now for testing
    # assert input_dir.exists()
    # assert output_dir.exists()

    # For now, just validate command line inputs
    # basic = BaSiC(**settings)

    # TODO: Handle file loading better, use pattern
    # images = image_tools.load_images(input_dir.iterdir())

    # corrected = basic.fit_transform(images, timelapse=settings["timelapse"])

    # TODO: Save the files
    # for i in range(corrected.shape[-1]):

    #     output = corrected[..., i]

    return


def snake_to_camel(string: str):

    split = string.split("_")

    camel_pieces = split[:1] + [s.capitalize() for s in split[1:]]

    return "".join(camel_pieces)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="BaSiCPy", description="BaSiC flatfield shading correction."
    )

    # Logging verbosity
    parser.add_argument(
        "-v",
        "--verbose",
        help="Show verbose outputs when running BaSiC. Useful for debugging.",
        action="store_true",
        required=False,
    )

    # Input arguments for file i/o
    parser.add_argument(
        "--inpDir",
        dest="input_dir",
        help="Path to input images.",
        metavar="FOLDER",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--outDir",
        dest="output_dir",
        help="Path to output images.",
        metavar="FOLDER",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--pattern",
        help="A regular expression pattern for selected images in inpDir.",
        metavar="REGEX",
        type=Path,
        required=False,
    )

    # Input arguments for BaSiC settings
    settings = BaSiC().dict()
    for k, v in settings.items():
        description = BaSiC.__fields__[k].field_info.description
        parser.add_argument(
            f"--{snake_to_camel(k)}",
            dest=k,
            type=type(v),
            help=description,
            metavar=type(v).__name__,
            default=v,
            required=False,
        )
    parser.add_argument(
        "--device",
        help="Device to run on. Must be one of [cpu,gpu,tpu].",
        metavar="DEVICE",
        type=Device,
        default=Device.cpu,
        required=False,
    )
    parser.add_argument(
        "--timelapse",
        help="Apply timelapse or photobleaching correction.",
        action="store_true",
        required=False,
    )

    args = parser.parse_args()
    logger = logging.getLogger("basicpy")
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("BaSiCPy Input Arguments")
    logger.info("-----------------------")
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")

    logger.info("")

    main(**vars(args))
