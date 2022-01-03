from typing import Callable, List, Tuple
import numpy as np

from scribblenet.preprocessing.utils import (
    add_dims_to_array,
    convert_to_grayscale,
    decode_base64_image_string,
    convert_to_png,
    image_to_numpy_array,
    invert_image,
    normalize_pixel_values,
    resize_image,
    training_normalize_pixel_values,
    training_reshape_image,
    one_hot_encode_labels,
)


class PreProcessor:
    """A class which preprocesses the data.
    """

    def __init__(self) -> None:
        self.train_pipeline: List[Callable] = [
            training_reshape_image,
            training_normalize_pixel_values,
            one_hot_encode_labels,
        ]

        self.predict_pipeline: List[Callable] = [
            decode_base64_image_string,
            convert_to_png,
            resize_image,
            invert_image,
            image_to_numpy_array,
            convert_to_grayscale,
            add_dims_to_array,
            normalize_pixel_values,
        ]

    def train_preprocess(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Runs the training pipeline on the dataset.

        Args:
            X_train (np.ndarray): Train data.
            X_test (np.ndarray): Test data.
            y_train (np.ndarray): Train labels.
            y_test (np.ndarray): Test labels.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The preprocessed dataset.
        """
        for job in self.train_pipeline:
            X_train, X_test, y_train, y_test = job(X_train, X_test, y_train, y_test)
        return X_train, X_test, y_train, y_test

    def predict_preprocess(self, image: str) -> np.ndarray:
        """Runs the prediction pipeline on an image.

        Args:
            image (str): The input image(s).
            job_type (str, optional): Indicates if the train, test or predict pipeline should be run on the image. Defaults to 'predict'.

        Returns:
            np.ndarray: The preprocessed image(s).
        """
        for job in self.predict_pipeline:
            image = job(image)
        return image  # type: ignore
