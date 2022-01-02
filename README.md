<p align="center">
    <!-- <img height=300 src="https://raw.githubusercontent.com/hasnainroopawalla/ShowML/master/static/images/showml.png" alt="ScribbleNet Logo"> -->
</p>

---

<h2 align="center">ScribbleNet</h2>

<div align="center">

[![Linting](https://github.com/hasnainroopawalla/ScribbleNet/actions/workflows/linting.yml/badge.svg)](https://github.com/hasnainroopawalla/ScribbleNet/actions/workflows/linting.yml)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
![Status](https://img.shields.io/badge/status-stable-green.svg)
</div>

---

Check out [the demo](https://www.hasnainr.com/projects/scribblenet.html#demonstration)!

<i>ScribbleNet</i> is a Python library which classifies hand-drawn doodles and is trained on the [QuickDraw dataset (by Google)](https://github.com/googlecreativelab/quickdraw-dataset) consisting of 345 classes and over 100,000 doodles.

This Python package is bundled as a REST API using [Flask](https://flask.palletsprojects.com/en/2.0.x/) and deployed on [Heroku](https://www.heroku.com/).

The front-end is built using vanilla JavaScript, HTML and CSS.

## 📝 Table of Contents
- [Examples](#examples)
- [Machine Learning Stuff](#ml)


## 📦 Examples <a name = "examples"></a>

<!-- Add gifs/images here -->

## 🧮 Machine Learning Stuff <a name = "ml"></a>

### Preprocessing
- 3 pre-processing pipelines (for training, testing and predicting on external data) have been created. Refer to `scribblenet.preprocessor.PreProcessor` ([source](https://github.com/hasnainroopawalla/ScribbleNet/blob/2e81465971e7a387ce9bbf725bf6fea239fddd75/scribblenet/preprocessing/preprocessor.py#L15)).

### Model
<!--


# ScribbleNet - Play Store (Android)

Download the Android App here: https://play.google.com/store/apps/details?id=doodle.classifier


# ScribbleNet

The notebook includes downloading the classes.txt file as well as the entire dataset of 'Quick Draw' images

The 100_classes.txt consists of 100 common classes (a subset of the 345 original classes)

The number of images used per class is limited to 16000 to increase training speed

With my architecture I achieved a Validation Accuracy of 96%


# Instructions to train the model:

Run the Doodle.ipynb on Google Colab (GPU for more disk memory)

# Instructions for Flask:
In the command prompt:
```
python run.py
```
Now, Navigate to 'localhost:5000' on your browser

# Android App:

Cup:

![Cup](https://github.com/hasnainroopawalla/ScribbleNet/blob/master/images/cup.gif)

Envelope:

![Envelope](https://github.com/hasnainroopawalla/Doodle-Classifier/blob/master/images/envelope.gif)

T-shirt:

![Tshirt](https://github.com/hasnainroopawalla/Doodle-Classifier/blob/master/images/tshirt.gif)

# Flask App:

Suitcase:

![Suitcase](https://github.com/hasnainroopawalla/Doodle-Classifier/blob/master/images/suitcase.gif)

Headphones:

![Headphones](https://github.com/hasnainroopawalla/Doodle-Classifier/blob/master/images/headphones.gif)

Lightning:

![Lightning](https://github.com/hasnainroopawalla/Doodle-Classifier/blob/master/images/lightning.gif)


 -->
