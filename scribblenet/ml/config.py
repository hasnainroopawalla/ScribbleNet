from typing import List, Tuple
import os


class MLConfig:
    """A class where all the Machine Learning specific configuration lives.

    Attributes:
        trained_model_path (str): Local path of the trained model requied for predictions.
        image_dims (Tuple[int, int]): Dimensions of each image used for resizing.
        classes_path_100 (str): Local path to the 100_classes.txt class names file.
        classes_path_all (str): Local path to the all_classes.txt class names file.
        dataset_url (str): URL of the Quickdraw dataset required to fetch the dataset for training.
        dataset_path (str): Local path where the downloaded dataset will be stored.
    """

    trained_model_path: str = os.path.join(
        os.path.dirname(__file__), "models/trained_model.h5"
    )
    image_dims: Tuple[int, int] = (28, 28)
    classes_path_100: str = os.path.join(
        os.path.dirname(__file__), "classes/100_classes.txt"
    )
    classes_path_all: str = os.path.join(
        os.path.dirname(__file__), "classes/all_classes.txt"
    )
    dataset_url: str = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
    dataset_path: str = os.path.join(os.path.dirname(__file__), "data/")
