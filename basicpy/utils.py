import tqdm
import warnings
import numpy as np
import inspect
from functools import wraps
import torch


def _is_uint8_array_like(x) -> bool:
    """Return True if x is a NumPy/Dask/Torch-like object with uint8 dtype."""
    if x is None or not hasattr(x, "dtype"):
        return False

    dtype = x.dtype

    # NumPy / Dask case
    try:
        if np.dtype(dtype) == np.uint8:
            return True
    except TypeError:
        pass

    # Torch case
    if torch is not None and dtype == torch.uint8:
        return True

    return False


def public_api(fn):
    sig = inspect.signature(fn)

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        self._public_call_depth = getattr(self, "_public_call_depth", 0) + 1
        try:
            if self._public_call_depth == 1:
                bound = sig.bind(self, *args, **kwargs)
                bound.apply_defaults()

                images = bound.arguments.get("images", None)
                is_timelapse = bound.arguments.get("is_timelapse", None)

                # ---- uint8 check ----
                if _is_uint8_array_like(images):
                    warnings.warn(
                        "Input images are uint8. This might be a problem. "
                        "Please refer to the documentation for details: "
                        "https://basicpy.readthedocs.io/en/latest/bestpractice.html",
                        UserWarning,
                        stacklevel=2,
                    )

                # ---- sort_intensity logic ----
                if is_timelapse is not None:

                    # non-timelapse but sorting disabled
                    if is_timelapse is False and self.sort_intensity is False:
                        warnings.warn(
                            "sort_intensity=False while is_timelapse=False. "
                            "For non–time-lapse datasets, enabling intensity "
                            "sorting (sort_intensity=True) is usually recommended.",
                            UserWarning,
                            stacklevel=2,
                        )

                    # timelapse but sorting enabled
                    if is_timelapse is True and self.sort_intensity is True:
                        warnings.warn(
                            "sort_intensity=True while is_timelapse=True. "
                            "For time-lapse datasets, intensity sorting is "
                            "usually not recommended. Consider setting "
                            "sort_intensity=False.",
                            UserWarning,
                            stacklevel=2,
                        )

            return fn(self, *args, **kwargs)

        finally:
            self._public_call_depth -= 1

    return wrapper


def make_overlap_chunks(
    n,
    chunk_size,
    overlap=1,
):
    assert 0 <= overlap < chunk_size
    step = chunk_size - overlap
    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(list(range(start, end)))
        if end == n:
            break
        start += step
    return chunks


def maybe_tqdm(iterable, use_tqdm=True, **kwargs):
    if use_tqdm:
        return tqdm.tqdm(iterable, **kwargs)
    else:
        return iterable
