import argparse
import base64
import os
import time
import uuid

import cv2
import numpy as np
import tensorflow as tf  # tensorflow == 1.15.0
from flask import (
    Flask,
    redirect,
    render_template,
    request,
    json,
    jsonify,
    send_from_directory,
    url_for,
)
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image, ImageOps
from tensorflow import keras  # keras == 2.2.5
from io import BytesIO
import io
from typing import List, Dict, Union, ByteString, Any

from scribblenet.ml.prediction import predict

input_img_path = "static/input_img.png"
app = Flask(__name__)


@app.route("/")
def firstpage():
    return render_template("canvas.html")


@app.route("/predict", methods=["POST"])
def doodle_predict():
    image_str = request.form.get("data")
    return jsonify(predict(image_str))


# @app.route("/doodlepredict", methods=["GET", "POST"])
# def predictedclasses():

#     imgstring = request.form.get("data")
#     img = data_uri_to_cv2_img(imgstring)
#     cv2.imwrite(input_img_path, img)
#     objs, pred_accuracy = predict.predict_doodle(input_img_path)
#     return str(objs + pred_accuracy)


def main():
    app.run(host="0.0.0.0", port=9000, debug=False)
