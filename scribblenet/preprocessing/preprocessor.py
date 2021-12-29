from typing import Callable, List
from scribblenet.preprocessing.utils import add_dims_to_array, convert_to_grayscale, decode_base64_image_string, convert_to_png, image_to_numpy_array, invert_image, normalize_pixel_values, resize_image

class PreProcessor:
    def __init__(self) -> None:
        self.train_pipeline: List[Callable] = [
            
        ]
        
        self.test_pipeline: List[Callable] = [
            
        ]
        
        self.predict_pipeline: List[Callable] = [
            decode_base64_image_string,
            convert_to_png,
            resize_image,
            invert_image,
            image_to_numpy_array,
            convert_to_grayscale,
            add_dims_to_array,
            normalize_pixel_values
        ]
        
