from typing import List, Tuple
import tensorflow as tf
from keras.engine.sequential import Sequential
from urllib.request import urlretrieve
from pathlib import Path
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from scribblenet.ml.config import MLConfig


def load_model() -> Sequential:
    """Loads the trained model from the specified local path.

    Returns:
        Sequential: The loaded model.
    """
    return tf.keras.models.load_model(MLConfig.trained_model_path)


def load_classes(num_classes: str = "100") -> List[str]:
    """Loads the class names from the specified local path.

    Args:
        num_classes (str): Specifies the number of classes to be loaded (can be '100' or 'all').
    Returns:
        List[str]: A list of class names sorted alphabetically.
    """
    with open(
        MLConfig.classes_path_100
        if num_classes == "100"
        else MLConfig.classes_path_all,
        "r",
    ) as f:
        classes = [class_name.rstrip("\n") for class_name in f]
    return sorted(classes)


def _create_dataset_directory():
    """Create a local directory to store the downloaded dataset.
    """
    Path(MLConfig.dataset_path).mkdir(parents=True, exist_ok=True)


def download_dataset(class_names: List[str]):
    """Downloads the Quickdraw dataset by Google.

    Args:
        class_names (List[str]): A List of clases whose data needs to be downloaded.
    """
    _create_dataset_directory()
    for index, class_name in tqdm(enumerate(class_names), total=len(class_names)):
        class_url = class_name.replace("_", "%20")
        url = MLConfig.dataset_url + class_url + ".npy"
        local_path = MLConfig.dataset_path + class_name + ".npy"
        urlretrieve(url, local_path)


def load_dataset(
    num_samples_per_class: int = 16000, test_size: float = 0.33
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads the downloaded data.

    Args:
        num_samples_per_class (int, optional): The number of images to be used from each class for training. Defaults to 16000.
        test_size (float, optional): The percentage of images to be used for testing. Defaults to 0.33.

    Returns:
        np.ndarray: Train data.
        np.ndarray: Test data.
        ynp.ndarray: Train labels.
        np.ndarray: Test labels.
    """
    dataset = sorted(glob.glob(os.path.join(MLConfig.dataset_path, "*.npy")))

    # Initialize variables
    X = np.empty([0, 784])
    y = np.empty([0])

    # Load each class file
    for idx, class_name in tqdm(enumerate(dataset), total=len(dataset)):
        data = np.load(class_name)
        data = data[0:num_samples_per_class, :]
        labels = np.full(data.shape[0], idx)
        X = np.concatenate((X, data), axis=0)
        y = np.append(y, labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, y_train, X_test, y_test
