from typing import List, Tuple
import os


class MLConfig:
    trained_model_path: str = os.path.join(
        os.path.dirname(__file__), "models/best_model.h5"
    )
    image_dims: Tuple[int, int] = (28, 28)
    classes_path_100: str = os.path.join(
        os.path.dirname(__file__), "classes/100_classes.txt"
    )
    classes_path_all: str = os.path.join(
        os.path.dirname(__file__), "classes/all_classes.txt"
    )
    dataset_url: str = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
    dataset_path: str = "data/"
    classes: List[str] = [
        "Smiley Face",
        "Alarm Clock",
        "Donut",
        "Butterfly",
        "Hat",
        "Wristwatch",
        "Paper Clip",
        "Rainbow",
        "Microphone",
        "Screwdriver",
        "Chair",
        "Spoon",
        "Snake",
        "Cloud",
        "Scissors",
        "Eye",
        "Bridge",
        "Suitcase",
        "Light Bulb",
        "Mushroom",
        "Book",
        "Dumbbell",
        "Flower",
        "Laptop",
        "Ice Cream",
        "Lightning",
        "Sword",
        "Tree",
        "Headphones",
        "Moon",
        "Baseball Bat",
        "Ceiling Fan",
        "Sun",
        "Bed",
        "Cup",
        "Table",
        "Hammer",
        "Pants",
        "Lollipop",
        "Ladder",
        "Tennis Racquet",
        "Cat",
        "Car",
        "Fan",
        "T-shirt",
        "Umbrella",
        "Bench",
        "Airplane",
        "Envelope",
        "Coffee Cup",
        "Hot Dog",
        "Pizza",
        "Cell Phone",
        "Radio",
        "Baseball",
        "Camera",
        "Bird",
        "Star",
        "Spider",
        "Pencil",
        "Key",
        "Mountain",
        "Circle",
        "Cookie",
        "Candle",
        "Sock",
        "Triangle",
        "Basketball",
        "Knife",
        "Apple",
        "Clock",
    ]
