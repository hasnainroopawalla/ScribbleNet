from typing import Callable, Dict, List
from scribblenet.preprocessing.utils import (
    add_dims_to_array,
    convert_to_grayscale,
    decode_base64_image_string,
    convert_to_png,
    image_to_numpy_array,
    invert_image,
    normalize_pixel_values,
    resize_image,
)
import numpy as np


class PreProcessor:
    def __init__(self) -> None:
        self.train_pipeline: List[Callable] = []

        self.test_pipeline: List[Callable] = []

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

        self.job_to_pipeline_map: Dict[str, List[Callable]] = {
            "train": self.train_pipeline,
            "test": self.test_pipeline,
            "predict": self.predict_pipeline,
        }

    def preprocess(self, image: str, job_type: str = "predict") -> np.ndarray:
        """Runs the desired pipeline on an image.

        Args:
            image (str): The input image(s).
            job_type (str, optional): Indicates if the train, test or predict pipeline should be run on the image. Defaults to 'predict'.

        Returns:
            np.ndarray: The preprocessed image(s).
        """
        for job in self.job_to_pipeline_map[job_type]:
            image = job(image)
        return image  # type: ignore
