"""The model to hold illumination correction profiles."""

import functools
from typing import Callable

import numpy as np


class Profile:
    """A class to hold illumination profiles.

    Attributes:
        profile: illumination correction profile
        type_: profile type (e.g., `"flatfield"`, `"darkfield"`)
        operation: operation for applying profile to input image

    Notes:
        ``Profile.operation`` is initialized with ``functools.partial``, having
        `self.profile` as the first argument to the callable. For example, the
        default `operation` for the `"flatfield"`-type profile is initialized as

        .. code-block:: python

            ...
            if self.type_ == "flatfield":
                self.operation = functools.partial(np.multiply, self.profile)
            ...

    """

    profile: np.ndarray
    type_: str
    operation: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def __init__(
        self,
        profile: np.ndarray,
        type_: str = "flatfield",
        operation: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
    ):
        """Init the Model class.

        Args:
            profile: illumination correction profile
            type_: profile type (e.g., `"flatfield"`, `"darkfield"`)
            operation: operation for applying profile to input image

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
                    "Unrecognized profile type and no `operation` provided."
                )
        else:
            self.operation = functools.partial(operation, self.profile)

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
