import argparse
import base64
import os
import time
import uuid

from flask import Flask, render_template, request, jsonify

from scribblenet.ml.prediction import predict

app = Flask(__name__)


@app.route("/")
def firstpage():
    return render_template("test.html")


@app.route("/predict", methods=["POST"])
def doodle_predict():
    image_string = request.form["image_string"]
    return jsonify(predict(image_string))


# @app.route("/doodlepredict", methods=["GET", "POST"])
# def predictedclasses():

#     imgstring = request.form.get("data")
#     img = data_uri_to_cv2_img(imgstring)
#     cv2.imwrite(input_img_path, img)
#     objs, pred_accuracy = predict.predict_doodle(input_img_path)
#     return str(objs + pred_accuracy)


def main():
    app.run(host="0.0.0.0", port=9000, debug=True)
