from PIL import Image
import numpy as np


def read_f32(file):
    r"""Reads a float32 image packed into RGBA.

    Parameters
    ----------
    file : str
        Path to the image file.

    Returns
    -------
    img : np.ndarray
        of shape [height, width] and dtype float32,
    """
    img = np.asarray(Image.open(file))
    img = unpack_float32(img)
    return img


def unpack_float32(ar8):
    r"""Unpacks an array of little-endian byte quadruplets back to the array of float32 values.

    Parameters
    ----------
    ar8 : np.ndarray
        of shape [**, 4].

    Returns
    -------
    ar : np.ndarray
        of shape [**]
    """
    return ar8.ravel().view('<f4').reshape(ar8.shape[:-1])
