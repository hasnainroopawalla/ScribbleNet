<h1 align="center">ScribbleNet</h1>


[![Linting](https://github.com/hasnainroopawalla/ScribbleNet/actions/workflows/linting.yml/badge.svg)](https://github.com/hasnainroopawalla/ScribbleNet/actions/workflows/linting.yml)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
![Status](https://img.shields.io/badge/status-stable-green.svg)


Check out [the demo](https://www.hasnainr.com/projects/scribblenet.html#demonstration)!

<i>ScribbleNet</i> is Python library that classifies hand-drawn doodles into distinct classes (trained on the [QuickDraw dataset by Google](https://github.com/googlecreativelab/quickdraw-dataset)).

This Python package is bundled as a REST API using [Flask](https://flask.palletsprojects.com/en/2.0.x/) and deployed on [Heroku](https://www.heroku.com/).

The front-end is built using vanilla JavaScript, HTML and CSS.

Download the Android App here ([Google Play](https://play.google.com/store/apps/details?id=doodle.classifier)).
>  The repository has recently been refactored and thus, the Android application currently doesn't work.

## 📝 Table of Contents
- [Getting Started](#gettingstarted)
- [Machine Learning Stuff](#ml)


## 🏁 Getting Started <a name = "gettingstarted"></a>
[Working demo](https://www.hasnainr.com/projects/scribblenet.html#demonstration)


To run locally:
1.  ```
    $ git clone https://github.com/hasnainroopawalla/ScribbleNet.git
    $ pip install -r requirements.txt
    $ python3 scribblenet/server.py
    ```
2. Navigate to `127.0.0.1:5000` in your browser.


## 🧮 Machine Learning Stuff <a name = "ml"></a>

### Pre-processing
- Two pre-processing pipelines (for training, predicting on external data) have been created. Refer to `scribblenet.preprocessor.PreProcessor` ([source](https://github.com/hasnainroopawalla/ScribbleNet/blob/b645e1c1299784faebbce4f7efbbdd67758bae0b/scribblenet/preprocessing/preprocessor.py#L19)).

### Training
- A `Jupyter` notebook for training the model on the Quickdraw Dataset can be found [here](https://github.com/hasnainroopawalla/ScribbleNet/blob/master/scribblenet/ml/training.ipynb).
- The notebook uses utility methods from the `scribblenet` package for loading and preprocessing the data.
- The `load_classes()` method accepts `100` or `all` as an argument to indicate if 100 classes or all 345 classes should be loaded.
- The `load_dataset()` method accepts `num_samples_per_class` as an argument to indicate how many samples of each class should be loaded.
