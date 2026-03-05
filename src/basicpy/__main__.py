import argparse
import logging
import os
from typing import Any, Optional

import tifffile

from basicpy import BaSiC

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)


def snake_to_camel(string: str) -> str:
    split = string.split("_")
    camel_pieces = split[:1] + [s.capitalize() for s in split[1:]]
    return "".join(camel_pieces)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="BaSiCPy",
        description="BaSiC flatfield shading correction.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        help="Show verbose outputs when running BaSiC. Useful for debugging.",
        action="store_true",
    )

    parser.add_argument(
        "--inpDir",
        dest="input_dir",
        help="Path to the input .tif image.",
        metavar="FILE",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--outDir",
        dest="output_dir",
        help="Path to the output .tif image.",
        metavar="FILE",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--pattern",
        help="A regular expression pattern for selected images in inpDir.",
        metavar="REGEX",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--require_autotune",
        help="Enable automatic hyperparameter tuning.",
        action="store_true",
        required=False,
    )

    default_settings = BaSiC().model_dump()

    for k, v in default_settings.items():
        description = BaSiC.model_fields[k].description or ""

        arg_name = f"--{snake_to_camel(k)}"

        if isinstance(v, bool):
            parser.add_argument(
                arg_name,
                dest=k,
                help=description,
                action="store_true" if v is False else "store_false",
                required=False,
            )
        else:
            parser.add_argument(
                arg_name,
                dest=k,
                type=type(v),
                help=description,
                metavar=type(v).__name__,
                default=v,
                required=False,
            )

    parser.add_argument(
        "--timelapse",
        help="Apply timelapse or photobleaching correction.",
        action="store_true",
        required=False,
    )

    return parser


def main(
    input_dir: str,
    output_dir: str,
    pattern: Optional[str] = None,
    require_autotune: bool = False,
    timelapse: bool = False,
    verbose: bool = False,
    **settings: Any,
) -> None:
    logger = logging.getLogger("basicpy")
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    logger.info("BaSiCPy Input Arguments")
    logger.info("-----------------------")
    logger.info(f"input_dir: {input_dir}")
    logger.info(f"output_dir: {output_dir}")
    logger.info(f"pattern: {pattern}")
    logger.info(f"is_timelapse: {timelapse}")
    for k, v in settings.items():
        logger.info(f"{k}: {v}")
    logger.info("")

    settings = {k: v for k, v in settings.items() if v is not None}
    basic = BaSiC(**settings)

    logger.info("BaSiC model initialized successfully.")
    logger.debug(f"Model config: {basic.model_dump()}")

    X = tifffile.imread(input_dir)
    if require_autotune:
        basic.autotune(X, is_timelapse=timelapse)
    corrected = basic.fit_transform(X, is_timelapse=timelapse)
    tifffile.imwrite(output_dir, corrected)
    return


def cli():
    parser = build_parser()
    args = parser.parse_args()
    main(**vars(args))


if __name__ == "__main__":
    cli()
