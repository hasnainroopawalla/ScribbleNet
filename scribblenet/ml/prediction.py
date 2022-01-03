import warnings
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from scribblenet.ml.utils import load_classes, load_model
from scribblenet.preprocessing.preprocessor import PreProcessor


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
    warnings.warn(
        "This method is outdated due to a conceptual change in the prediction logic.",
        DeprecationWarning,
    )
    best_class_indices = prediction.argsort()[-num_best_classes:][::-1]
    probabilities = prediction[best_class_indices]
    return list(best_class_indices), list(probabilities)


def _map_class_names_to_probabilities(probabilities: List[float]) -> Dict[str, float]:
    """Creates a dictionary mapping readable class names to their corresponding probabilites.

    Args:
        probabilities (List[float]): A List of the probabilities for the best predicted classes.

    Returns:
        Dict[str, float]: A dictionary mapping all readable class names to their corresponding probabilites.
    """
    classes = load_classes()
    return {
        class_name: probability
        for class_name, probability in zip(classes, probabilities)
    }


def _get_best_class_names(
    class_names_to_probabilities: Dict[str, float], num_best_classes: int
) -> List[str]:
    """Returns 'num_best_classes' class names with the best probabilities.

    Args:
        class_names_to_probabilities (Dict[str, float]): A dictionary mapping predicted readable class names to their corresponding probabilites.
        num_best_classes (int): The number of classes with the highest probability to be returned.

    Returns:
        List[str]: A List of size 'num_best_classes' of the best predicted class names.
    """
    best_class_names_to_probabilities = {
        class_name: probability
        for class_name, probability in Counter(
            class_names_to_probabilities
        ).most_common()[:num_best_classes]
    }
    return list(best_class_names_to_probabilities.keys())


def predict(image: str, num_best_classes: int = 5) -> List[str]:
    """Predict the top 'num_best_classes' given an input image.

    Args:
        image (str): The input image.
        num_best_classes (int): The number of classes with the highest probability to be returned. Defaults to 5.

    Returns:
       List[str]: A dictionary mapping predicted readable class names to their corresponding probabilites.
    """
    P = PreProcessor()
    processed_image = P.predict_preprocess(image)
    model = load_model()
    prediction = model.predict(processed_image)[0]

    class_names_to_probabilities = _map_class_names_to_probabilities(prediction)
    return _get_best_class_names(class_names_to_probabilities, num_best_classes)
