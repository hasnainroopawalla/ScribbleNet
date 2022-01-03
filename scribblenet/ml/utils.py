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


def load_classes() -> List[str]:
    """Loads the class names from the specified local path.

    Returns:
        List[str]: A list of class names sorted alphabetically.
    """
    with open(MLConfig.classes_path, "r") as f:
        classes = [class_name.rstrip("\n") for class_name in f]
    return sorted(classes)


def _create_dataset_directory():
    """Create a local directory to store the downloaded dataset.
    """
    Path(MLConfig.dataset_path).mkdir(parents=True, exist_ok=True)


def download_dataset(class_names: List[str]):
    _create_dataset_directory()
    for index, class_name in tqdm(enumerate(class_names), total=len(class_names)):
        class_url = class_name.replace("_", "%20")
        # print(index, class_name, class_url)
        url = MLConfig.dataset_url + class_url + ".npy"
        local_path = MLConfig.dataset_path + class_name + ".npy"
        urlretrieve(url, local_path)


def load_dataset(
    num_samples_per_class: int = 16000, test_size: float = 0.33
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = sorted(glob.glob(os.path.join(MLConfig.dataset_path, "*.npy")))

    # Initialize variables
    X = np.empty([0, 784])
    y = np.empty([0])

    # Load each class file
    for idx, class_name in tqdm(enumerate(dataset), total=len(dataset)):
        # print(idx, class_name)
        data = np.load(class_name)
        data = data[0:num_samples_per_class, :]
        labels = np.full(data.shape[0], idx)
        X = np.concatenate((X, data), axis=0)
        y = np.append(y, labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, y_train, X_test, y_test
