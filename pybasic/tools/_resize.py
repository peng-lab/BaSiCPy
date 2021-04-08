from skimage.transform import resize as skresize
import numpy as np
from typing import List

RESIZE_ORDER = 1
RESIZE_MODE = "symmetric"
_preserve_range = True

def _resize_images_list(images_list: List, side_size: float = None, x_side_size: float = None, y_side_size: float = None):
    if side_size is not None:
        y_side_size = x_side_size = side_size
    resized_images_list = []
    for i, im in enumerate(images_list):
        if im.shape[0] != x_side_size or im.shape[1] != y_side_size:
            resized_images_list.append(skresize(
                im, 
                (x_side_size, y_side_size), 
                order = RESIZE_ORDER, 
                mode = RESIZE_MODE,
                preserve_range = _preserve_range
                )
            )
        else:
            resized_images_list.append(im)
    return resized_images_list

def _resize_image(image: np.ndarray, side_size: float  = None, x_side_size: float = None, y_side_size: float = None):
    if side_size is not None:
        y_side_size = x_side_size = side_size
    if image.shape[0] != x_side_size or image.shape[1] != y_side_size:
        return skresize(
            image,
            (x_side_size, y_side_size), 
            order = RESIZE_ORDER, 
            mode = RESIZE_MODE,
            preserve_range = _preserve_range
        )
    else:
        return image
