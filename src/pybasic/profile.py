"""The model to hold illumination correction profiles."""

import functools
from typing import Callable

import numpy as np


class Profile:
    """A class to hold illumination profiles."""

    def __init__(
        self, profile: np.ndarray, type_: str = "flatfield", operation: Callable = None
    ):
        """Init the Model class.

        Args:
            profile: illumination correction profile
            type_: profile type (options are `"flatfield"`, `"darkfield"`)
            operation: operation to apply profile to input image

        Raises:
            ValueError: profile type cannot be identified
        """
        self.profile = profile
        self.type_ = type_

        if operation is None:
            if self.type_ == "flatfield":
                self.operation = functools.partial(np.multiply, self.profile)
            elif self.type_ == "darkfield":
                self.operation = functools.partial(np.add, -self.profile)
            else:
                raise ValueError(
                    "Unidentified profile type. Cannot determine application operation."
                )
        else:
            self.operation = operation

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply illumination profile to input image.

        Args:
            image: input image

        Returns:
            illumination corrected image

        Example:
            >>> profiles = basic.fit(images)
            >>> image_0 = images[0]
            >>> corrected_0 = prof.apply(image_0)
        """
        return self.operation(image)

    def save(self, fname) -> str:
        """Save the profile to a file.

        Args:
            fname: path to file, supported extensions are `".tif"`, `".npy"`, `".mat"`

        Returns:
            saved filename

        Example:
            >>> profiles = basic.fit(images)
            >>> for prof in profiles:
            ...    prof.save(prof.type + ".npy")
            ["flatfield.npy", "darkfield.npy"]
        """
        ...
        return
