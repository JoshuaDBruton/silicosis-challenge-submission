import sys
import numpy as np
import warnings
import skimage


class XRayResizer(object):
    """Resize an image to a specific size"""

    def __init__(self, size: int, engine="skimage"):
        self.size = size
        self.engine = engine
        if "cv2" in sys.modules:
            print("Setting XRayResizer engine to cv2 could increase performance.")

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.engine == "skimage":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return skimage.transform.resize(
                    img, (1, self.size, self.size), mode="constant", preserve_range=True
                ).astype(np.float32)
        elif self.engine == "cv2":
            import cv2  # pip install opencv-python

            return (
                cv2.resize(
                    img[0, :, :], (self.size, self.size), interpolation=cv2.INTER_AREA
                )
                .reshape(1, self.size, self.size)
                .astype(np.float32)
            )
        else:
            raise Exception("Unknown engine, Must be skimage (default) or cv2.")


def normalize(img, maxval, reshape=False):
    """Scales images to be roughly [-1024 1024]."""

    if img.max() > maxval:
        raise Exception(
            "max image value ({}) higher than expected bound ({}).".format(
                img.max(), maxval
            )
        )

    img = (2 * (img.astype(np.float32) / maxval) - 1.0) * 1024

    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # add color channel
        img = img[None, :, :]

    return img
