import base64
from PIL import PngImagePlugin, Image, ImageOps
from io import BytesIO
import numpy as np
from keras.preprocessing.image import img_to_array
import cv2


def _handle_base64_string_header(base64_img_string: str) -> str:
    """Handle the extra base64 string header (data:image/png;base64,).

    Args:
        base64_img_string (str): The base64 encoded image string.

    Returns:
        str: The base64 encoded image string after removing the header.
    """
    base64_img_string_split = base64_img_string.split(",")
    if len(base64_img_string_split) == 1:
        return base64_img_string_split[0]
    else:
        return base64_img_string_split[1]


def decode_base64_image_string(base64_img_string: str) -> BytesIO:
    """Decode the base64 encoded image string to a BytesIO object.

    Args:
        base64_img_string (str): The base64 encoded image string.

    Returns:
        BytesIO: The BytesIO object of the image.
    """
    return BytesIO(base64.b64decode(_handle_base64_string_header(base64_img_string)))


def convert_to_png(bytes_obj: BytesIO) -> PngImagePlugin.PngImageFile:
    """Convert a BytesIO object to a PNG image format.

    Args:
        bytes_obj (BytesIO): The BytesIO representation of the image.

    Returns:
        PngImagePlugin.PngImageFile: The image in PNG format.
    """
    return Image.open(bytes_obj)


def resize_image(image: PngImagePlugin.PngImageFile) -> Image.Image:
    """Resizes the image to the specified dimensions.

    Args:
        image (PngImagePlugin.PngImageFile): The input PNG image.

    Returns:
        Image.Image: The resized image.
    """
    image.show()
    return image.resize((28, 28))


def invert_image(image: Image.Image) -> Image.Image:
    """Inverts the pixel values of the image.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The inverted image.
    """
    return ImageOps.invert(image.convert("RGB"))


def image_to_numpy_array(image: Image.Image) -> np.ndarray:
    """Convert the image to a numpy array.

    Args:
        image (Image.Image): The input image.

    Returns:
        np.ndarray: The numpy array of the image.
    """
    return img_to_array(image)


def convert_to_grayscale(image_array: np.ndarray) -> np.ndarray:
    """Convert the image to grayscale.

    Args:
        image_array (np.ndarray): The input image.

    Returns:
        np.ndarray: The image after conversion to grayscale.
    """
    return cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)


def add_dims_to_array(image: np.ndarray) -> np.ndarray:
    """Adds 2 extra dimensions to the image array.
    The 1st dimension indicates the number of images per sample.
    The 4th dimension indicates that the image is in grayscale (single channel).

    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: A 4-dimensional numpy array.
    """
    img_grayscale_channeled = np.expand_dims(
        image, axis=2
    )  # Add last channel as 1 (28,28) to (28,28,1)
    return np.expand_dims(
        img_grayscale_channeled, axis=0
    )  # Add frist channel to specify number of input images (1,28,28,1)


def normalize_pixel_values(image: np.ndarray) -> np.ndarray:
    """Normalizes pixel values from 0-255 to 0-1.

    Args:
        image (np.ndarray): The input image tensor.

    Returns:
        np.ndarray: The normalized image tensor.
    """
    return image / 255.0
