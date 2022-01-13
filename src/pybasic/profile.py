"""A class to hold illumination correction profiles."""

import functools
from typing import Callable

import numpy as np

from .types import ArrayLike, PathLike


class Profile:
    """A class to hold illumination profiles."""

    def __init__(
        self,
        array: np.ndarray,
        type_: str = "flatfield",
        operation: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
    ):
        """Initialize the class.

        Args:
            array: illumination correction profile
            type_: profile type (e.g., `"flatfield"`, `"darkfield"`)
            operation: operation for applying profile to input image

        Raises:
            ValueError: profile type cannot be identified

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
        self.array = array
        self.type_ = type_

        if operation is None:
            if self.type_ == "flatfield":
                self.operation = functools.partial(np.multiply, self.profile)  # type: ignore # noqa: E501
            elif self.type_ == "darkfield":
                self.operation = functools.partial(np.add, -self.profile)  # type: ignore # noqa: E501
            else:
                raise ValueError(
                    "Unrecognized profile type and no `operation` provided."
                )
        else:
            self.operation = functools.partial(operation, self.profile)  # type: ignore # noqa: E501

    def apply(self, image: ArrayLike) -> np.ndarray:
        """Apply the profile to an input image.

        Args:
            image: input image

        Returns:
            illumination corrected image

        Example:
            >>> profiles = basic.fit(images)
            >>> image_0 = images[0]
            >>> corrected_0 = prof.apply(image_0)
        """
        return self._operation(image)

    def save(self, fname: PathLike):
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

    @property
    def array(self) -> np.ndarray:
        """The illumination correction profile.

        Can be set by assignment.

        Returns:
            the profile as an array
        """
        return self._array

    @array.setter
    def array(self, value):
        self._array = value

    @property
    def type_(self) -> str:
        """The illumination correction profile type.

        Can be set by assignment.

        Returns:
            the profile type
        """
        return self._type_

    @type_.setter
    def type_(self, value):
        self._type_ = value

    @property
    def operation(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """The operation for applying profile to input image.

        Can be set by assignment.

        Returns:
            the operation
        """
        return self._operation

    @operation.setter
    def operation(self, value):
        self._operation = value
