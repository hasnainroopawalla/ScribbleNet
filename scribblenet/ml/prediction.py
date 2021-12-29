from typing import Dict, List, Tuple
from scribblenet.preprocessing.preprocessor import PreProcessor
from scribblenet.ml.utils import load_model
import numpy as np
from scribblenet.ml.config import MLConfig


def _get_best_indices_and_accuracies(
    prediction: np.ndarray, num_best_classes: int
) -> Tuple[List[int], List[float]]:
    """Extract the indices and values of the best predictions.

    Args:
        prediction (np.ndarray): The probabilities of the predicted classes.
        num_best_classes (int): The number of classes with the highest probability to be returned.

    Returns:
        List[int]; A List of indices of the best predicted classes.
        List[float]: A List of the corresponding probabilities for the best predicted classes.
    """
    best_class_indices = prediction.argsort()[-num_best_classes:][::-1]
    probabilities = prediction[best_class_indices]
    return list(best_class_indices), list(probabilities)


def _get_best_class_names(best_class_indices: List[int]) -> List[str]:
    """Fetches the readable class names based on the indices.

    Args:
        best_class_indices (List[int]): A List of indices of the best predicted classes.

    Returns:
        List[str]: A List of class names with the highest predicted probabilities.
    """
    return [MLConfig().classes[index] for index in best_class_indices]


def _map_class_names_to_probabilities(
    best_class_names: List[str], probabilities: List[float]
) -> Dict[str, float]:
    """Creates a dictionary mapping readable class names to their corresponding probabilites.

    Args:
        best_class_names (List[str]): A List of class names with the highest predicted probabilities.
        probabilities (List[float]): A List of the probabilities for the best predicted classes.

    Returns:
        Dict[str, float]: A dictionary mapping readable class names to their corresponding probabilites.
    """
    return {
        class_name: int(probability)
        for class_name, probability in zip(best_class_names, probabilities)
    }


def predict(image: str, num_best_classes: int = 5) -> List[str]:
    """Predict the top 'num_best_classes' given an input image.

    Args:
        image (str): The input image.
        num_best_classes (int): The number of classes with the highest probability to be returned. Defaults to 5.

    Returns:
       List[str]: A dictionary mapping predicted readable class names to their corresponding probabilites.
    """
    P = PreProcessor()
    processed_image = P.preprocess(image, job_type="predict")
    model = load_model()
    prediction = model.predict(processed_image)[0]
    best_class_indices, probabilities = _get_best_indices_and_accuracies(
        prediction, num_best_classes
    )
    return _get_best_class_names(best_class_indices)
