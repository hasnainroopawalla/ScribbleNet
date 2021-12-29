import tensorflow as tf
from keras.engine.sequential import Sequential

from scribblenet.ml.config import MLConfig


def load_model() -> Sequential:
    return tf.keras.models.load_model(MLConfig().model_path)
